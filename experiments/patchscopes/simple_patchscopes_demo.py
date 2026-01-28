# %%

import os

import pandas as pd
import torch
from peft import LoraConfig

from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.eval import run_evaluation

folder = "datasets/factual"

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
import ast

from pydantic import BaseModel

import nl_probes.dataset_classes.classification as classification
from nl_probes.utils.dataset_utils import TrainingDataPoint

prompt_template = "Respond with only the answer and no other words. What is {prompt_target}this?"

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
    prompt = prompt_template.format(prompt_target=prompt_target)
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
from nl_probes.utils.common import load_model, load_tokenizer

model_name = "Qwen/Qwen3-8B"
tokenizer = load_tokenizer(model_name)
batch_size = 128

# This is what we trained this model with
steering_coefficient = 1.0

# Currently the codebase is somewhat built around the `TrainingDataPoint` class. We'll convert our datapoints to this format

# We currently put the activation prompt in the user role of the chat template
# an offset of -3 is right before the EOS token in the chat template
end_offset = -3

# by setting the min and max window size to 1, we'll only collect activations from the last token
# if we wanted the last 10, we would set min to 10 and max to 10
# if we want random window sizes from 1 to 10, we would set min to 1 and max to 10
min_window_size = 1
max_window_size = 1

# we will construct our dataset using layers 9, 18, and 27
# This means for every datapoint, we will be evaluating it 3 times - once for each layer
act_layers = [9, 18, 27]

# we will save the activations in the TrainingDataPoint object
# if false, we will generate them on the fly
# If using large datasets with many activations, this will reduce memory / disk usage
# for example, with 1M datapoints and window size 10, d_model = 5k, we would use 100GB for 1M * 10 * 5k * 2 (bfloat16).
save_acts = True
batch_size = 128

# if you want to use a trained model organism, pass the path to the lora here
# NOTE: For the current code, if collecting activations from a different LoRA, make sure save_acts is True
# If save_acts is False, we collect the activations on the fly from the base model with no LoRA (although we could support passing in a LoRA for activation collection)
activation_lora_path = None
# %%

training_data = classification.create_vector_dataset(
    raw_datapoints,
    tokenizer,
    model_name,
    batch_size=batch_size,
    act_layers=act_layers,
    min_end_offset=end_offset,
    max_end_offset=end_offset,
    max_window_size=max_window_size,
    min_window_size=min_window_size,
    save_acts=save_acts,
    datapoint_type="patchscope",
    lora_path=activation_lora_path,
)


# %%
print(training_data[0])

# if creating a new dataset, it's highly recommended to manually inspect the dataset
# in this case, context positions is [7] and 7 is the position before the EOS token - perfect!

# Target output: Washington D.C.
# context positions: [7]
# Context input id 0: <|im_start|>
# Context input id 1: user
# Context input id 2:

# Context input id 3:  programs
# Context input id 4:  in
# Context input id 5:  the
# Context input id 6:  United
# Context input id 7:  States
# Context input id 8: <|im_end|>
# Context input id 9:


for i in range(3):
    dp = training_data[i]
    print(f"model prompt: {tokenizer.decode(dp.input_ids)}")
    print(f"steering locations: {dp.positions}")
    for j in range(len(dp.input_ids)):
        print(f"Input id {j}: {tokenizer.decode(dp.input_ids[j])}")
    print(f"Target output: {dp.target_output}")
    print(f"context positions: {dp.context_positions}")
    for j in range(len(dp.context_input_ids)):
        print(f"Context input id {j}: {tokenizer.decode(dp.context_input_ids[j])}")
    print("-" * 100)

# %%
dtype = torch.bfloat16


device = torch.device("cuda")
model = load_model(model_name, dtype, load_in_8bit=False)

injection_layer = 1
submodule = get_hf_submodule(model, injection_layer)

# %%

model.eval()

dummy_config = LoraConfig()
model.add_adapter(dummy_config, adapter_name="default")

# %%
lora_path = "checkpoints_act_pretrain/final"  # for local paths
lora_path = "anon-org/checkpoints_act_pretrain_posttrain"
lora_path = "anon-org/checkpoints_act_cls_latentqa_sae_pretrain_mix_Qwen3-8B"
# lora_path = "checkpoints_act_pretrain_posttrain/final"
responses = run_evaluation(
    eval_data=training_data,
    model=model,
    tokenizer=tokenizer,
    submodule=submodule,
    device=device,
    dtype=dtype,
    global_step=-1,
    lora_path=lora_path,
    eval_batch_size=batch_size,
    steering_coefficient=steering_coefficient,
    generation_kwargs={
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 10,
    },
)

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
for response, eval_data_point in zip(responses, training_data, strict=True):
    target_response = eval_data_point.target_output
    response = response.api_response
    verbalizer_prompt = tokenizer.decode(eval_data_point.input_ids)
    context_prompt = tokenizer.decode(eval_data_point.context_input_ids)
    if compare_patchscope_responses(response, target_response):
        correct += 1
    else:
        print(f"Response: {response}, Target: {target_response}, verbalizer prompt: {verbalizer_prompt}, context prompt: {context_prompt}")

print(f"Correct: {correct}")
print(f"Total: {len(training_data)}")
print(f"Accuracy: {correct / len(training_data)}")
# %%
