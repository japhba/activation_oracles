# %%

import os
import vllm
import pandas as pd
import torch
from peft import LoraConfig

from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_tokenizer
from nl_probes.utils.eval import run_evaluation

folder = "data_pipelines/factual"

files = os.listdir(folder)

for file in files:
    print(file)

# %%

first_file = files[10]  # .tsv

with open(os.path.join(folder, first_file), "r") as f:
    data = pd.read_csv(f, sep="\t")

# for this demo I'll use the patchscopes dataset
# this one looks for the capital city of countries

print(data.head())

# %%


# %%


dtype = torch.bfloat16


device = torch.device("cuda")


# VLLM_MODEL_NAME = "google/gemma-2-9b-it"
VLLM_MODEL_NAME = "Qwen/Qwen3-8B"
tokenizer = load_tokenizer(VLLM_MODEL_NAME)

# %%
# VLLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

vllm_model = vllm.LLM(
    model=VLLM_MODEL_NAME,
    max_model_len=2000,
    enforce_eager=True,
    enable_lora=True,
    max_lora_rank=32,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
)


# %%
import ast

import nl_probes.dataset_classes.classification as classification
from nl_probes.utils.dataset_utils import TrainingDataPoint

prompt_template = "Respond with the answer only and nothing else. What is {prompt_target}the entity at the end of this phrase: '{context}'?"

raw_datapoints: list[classification.ClassificationDatapoint] = []
for _, row in data.iterrows():
    source_toks = ast.literal_eval(row["source_cropped_toks"])
    assert isinstance(source_toks, list)
    assert len(source_toks) > 0
    source_pos = row["position_source"]
    source = "".join(source_toks[: source_pos + 1])

    prompt_target = row["prompt_target"]
    assert prompt_target[-1] == "x"
    prompt_target = prompt_target[:-1].lower()
    prompt = prompt_template.format(prompt_target=prompt_target, context=source)
    assert isinstance(prompt, str)
    assert len(prompt) > 0

    dp = classification.ClassificationDatapoint(
        activation_prompt=source,
        classification_prompt=prompt,
        target_response=row["object"],
        ds_label=None,
    )
    raw_datapoints.append(dp)


# so, I'm working with the pydantic objects we've been using. They are pretty simple, but you could revert to strings / dicts if you want

# note: This model was only trained on binary Yes / No questions. It appears to work for open ended questions as well, but you may get better results by formatting the prompt as a Yes / No question.
# format example: Answer with 'Yes' or 'No' only. Can we say this is in Russian?
# Also note: You don't need to add the layer number or ??? to the prompt - that will be done for you in the next cell

print(raw_datapoints[0])

print(f"\nThe activation prompt is: {raw_datapoints[0].activation_prompt}")
print("We will collect our activations using this prompt")
print(f"\nThe classification prompt is: {raw_datapoints[0].classification_prompt}")
print("This will be the prompt to our trained investigator model")
print(f"\nThe target response is: {raw_datapoints[0].target_response}")
print("This will be the training target for training, or the correct answer for evaluation")

# %%

prompts = []
for dp in raw_datapoints:
    message = [{"role": "user", "content": dp.classification_prompt}]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompts.append(prompt)


print(prompts[0])


# %%

sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=10)

responses = vllm_model.generate(
    prompts,
    sampling_params,
)


# %%

print(responses[0].outputs[0].text)

# %%

import difflib
import re
import unicodedata

def parse_answer(s: str) -> str:
    """Normalize an answer to a simple whitespace separated, ascii lowercase form.

    - Decodes literal unicode-escape sequences like '\\u00ed' when present.
    - Normalizes unicode and strips diacritics (é -> e).
    - Replaces any non-alphanumeric runs with a single space (commas/periods -> space).
    - Collapses whitespace and lowercases.
    """
    if s is None:
        return ""
    s = s.strip()

    # If the string contains literal backslash-escapes like '\u00ed', decode them.
    if "\\u" in s or "\\x" in s:
        try:
            # only attempt if there are escape sequences; safe-guarded by try/except
            s = s.encode("utf-8").decode("unicode_escape")
        except Exception:
            # fail silently and continue with original string
            pass

    # Unicode normalize then drop combining marks (remove accents)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # Replace any run of non-alphanumeric characters with a single space.
    # This turns "Washington, D.C." -> "Washington D C" rather than joining tokens together.
    s = re.sub(r"[^0-9A-Za-z]+", " ", s)

    # Collapse whitespace and lowercase
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def compare_patchscope_responses(response: str, target_response: str) -> bool:
    response = parse_answer(response)
    target_response = parse_answer(target_response)
    return target_response in response


correct = 0
for response, eval_data_point in zip(responses, raw_datapoints, strict=True):
    target_response = eval_data_point.target_response
    response = response.outputs[0].text
    if compare_patchscope_responses(response, target_response):
        correct += 1
    else:
        print(f"Response: {response}, Target: {target_response}")

print(f"Correct: {correct}")
print(f"Total: {len(raw_datapoints)}")
print(f"Accuracy: {correct / len(raw_datapoints)}")

# %%
