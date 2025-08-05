import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import contextlib

# New imports for dataclasses
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Dict, Tuple, Optional, Any
from jaxtyping import Float
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from huggingface_hub import hf_hub_download
import random
import pickle
import asyncio
import re
from rapidfuzz.distance import Levenshtein as lev
from tqdm import tqdm

import interp_tools.saes.jumprelu_sae as jumprelu_sae
import interp_tools.model_utils as model_utils
import interp_tools.api_utils.shared as shared
import interp_tools.api_utils.api_caller as api_caller
import interp_tools.introspect_utils as introspect_utils
from interp_tools.config import SelfInterpTrainingConfig, get_sae_info

# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================


def load_acts(
    model_name: str,
    sae_layer: int,
    sae_width: int,
    layer_percent: int,
    context_length: int | None = None,
) -> dict[str, torch.Tensor]:
    acts_dir = "max_acts"

    ctx_len_str = (
        f"_context_length_{context_length}" if context_length is not None else ""
    )

    acts_filename = f"acts_{model_name}_layer_{sae_layer}_trainer_{sae_width}_layer_percent_{layer_percent}{ctx_len_str}.pt".replace(
        "/", "_"
    )
    print(acts_filename)
    acts_path = os.path.join(acts_dir, acts_filename)
    if not os.path.exists(acts_path):
        from huggingface_hub import hf_hub_download

        path_to_config = hf_hub_download(
            repo_id="adamkarvonen/sae_max_acts",
            filename=acts_filename,
            force_download=False,
            local_dir=acts_dir,
            repo_type="dataset",
        )

    acts_data = torch.load(acts_path)

    return acts_data


def list_decode(x: torch.Tensor, tokenizer: PreTrainedTokenizer):
    assert len(x.shape) == 1 or len(x.shape) == 2
    # Convert to list of lists, even if x is 1D
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Make it 2D for consistent handling

    # Convert tensor to list of list of ints
    token_ids = x.tolist()

    # Convert token ids to token strings
    return [tokenizer.batch_decode(seq, skip_special_tokens=False) for seq in token_ids]


def make_contrastive_prompt(
    feature_acts: torch.Tensor,
    feature_tokens: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
) -> str:
    assert feature_acts.shape == feature_tokens.shape
    assert len(feature_acts.shape) == 2

    prompt = """
I have a list of sentences that relate to a particular concept. In each sentence, there will be a maximally activated location, which will be a word contained between delimiters like <<this>>.

I want you to do 2 steps:
1. First, generate a potential explanation for the pattern that represents the concept.
2. Make a minimal rewrite of each sentence that entirely removes the concept, including both the delimited word and other potentially related words.

Please return a properly formatted answer, using the tags <EXPLANATION>, <PLANNED EDITS>, <SENTENCE 0>, <SENTENCE 1>, <END>, etc, and following the example below.

BEGIN EXAMPLE

1. 's what 12 Monkeys is about: time<< travel>>, although it's not as sci-fi related as it appears. Bruce Willis is a time traveler, which makes 12 Monkeys a sci-fi movie, but the entirety of it is not as complex as one would think.

2. PMtraveler - 03 November 2011 01:54 PMBut the simple act of going back to the past changes the past. I mean, you weren’t around in 1820, right? So it changed.
Ok, now I am confused. Jesus (why pick him?) got to the past by using a time<< machine>>.

RESPONSE:

<EXPLANATION>
I believe the concept relates to time travel.

<PLANNED EDITS>
    
- For sentence one, I will replace 'time travel' with 'ocean travel'. I will also replace the movie '12 Monkeys' with 'Adrift', as 12 Monkeys is related to time-travel.
- For sentence two, I will replace "going back to the past" with "going to a place", and replace the time machine with "traveling".

<SENTENCE 0>
 's what Adrift is about: ocean travel, although it's not as sci-fi related as it appears. Bruce Willis is a ocean traveler, which makes Adrift a sci-fi movie, but the entirety of it is not as complex as one would think.",

<SENTENCE 1>
PMtraveler - 03 November 2011 01:54 PMBut the simple act of going to a place changes that place. I mean, you weren't there before, right? So it changed.
Ok, now I am confused. Jesus (why pick him?) got there by traveling.

<END>

END EXAMPLE

Okay, now here are the real sentences:

{sentences}

RESPONSE:
"""

    formatted_sentences = ""
    for i in range(feature_acts.shape[0]):
        feature_strs = list_decode(feature_tokens[i], tokenizer)[0]
        feature_strs = feature_strs[1:]  # skip bos
        sentence = "".join(feature_strs)
        formatted_sentences += f"\n<SENTENCE {i}>\n{sentence}\n"

    formatted_prompt = prompt.format(sentences=formatted_sentences)

    return formatted_prompt


def build_prompts(str_prompts: list[str]) -> list[shared.ChatHistory]:
    prompts = []

    for prompt in str_prompts:
        prompts.append(shared.ChatHistory().add_user(prompt))

    return prompts


def sentence_distance(
    old_sentence: str,
    new_sentence: str,
) -> float:
    # TODO: Try out word-level distance
    return lev.normalized_distance(old_sentence, new_sentence)


def parse_llm_response(
    response_text: str, num_sentences: int
) -> Tuple[Optional[str], Optional[str], Dict[int, str]]:
    """Parses the LLM's response to extract the explanation and rewritten sentences."""
    explanation_match = re.search(
        r"<EXPLANATION>(.*?)<PLANNED EDITS>", response_text, re.DOTALL
    )
    explanation = explanation_match.group(1).strip() if explanation_match else None

    planned_edits_match = re.search(
        r"<PLANNED EDITS>(.*?)<SENTENCE 0>", response_text, re.DOTALL
    )
    planned_edits = (
        planned_edits_match.group(1).strip() if planned_edits_match else None
    )

    rewritten_sentences = {}
    for i in range(num_sentences):
        # Pattern to find sentence i and capture content until the next sentence tag or the end tag
        pattern = re.compile(
            rf"<SENTENCE {i}>(.*?)(?:<SENTENCE {i + 1}>|<END>)", re.DOTALL
        )
        match = pattern.search(response_text)
        if match:
            rewritten_sentences[i] = match.group(1).strip()

    return explanation, planned_edits, rewritten_sentences


def parallel_eval(
    cfg: SelfInterpTrainingConfig,
    feature_indices: list[int],
    all_prompts: list[str],
    api_responses: list[str],
    acts_data: dict[str, torch.Tensor],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    sae: jumprelu_sae.JumpReluSAE,
    batch_size: int,
) -> list[dict]:
    sentences_to_process = {
        "original": [],
        "rewritten": [],
        "feature_indices": [],
        "reconstruction_info": [],  # To map results back to the correct feature/sentence
    }

    feature_results = []

    num_valid_sentences = 0
    num_invalid_sentences = 0

    for i, feature_idx in enumerate(feature_indices):
        api_prompt = all_prompts[i]
        api_response = api_responses[i]

        feature_tokens = acts_data["max_tokens"][
            feature_idx, : cfg.api_num_sentences_per_feature
        ]
        explanation, planned_edits, rewritten_sentences_dict = parse_llm_response(
            api_response, cfg.api_num_sentences_per_feature
        )

        feature_result = {
            "feature_idx": feature_idx,
            "api_prompt": api_prompt,
            "api_response": api_response,
            "explanation": explanation,
            "planned_edits": planned_edits,
            "sentence_data": [],
            "sentence_metrics": [],
        }
        feature_results.append(feature_result)

        for sent_idx in range(cfg.api_num_sentences_per_feature):
            # [1:] to skip bos token
            original_sentence = "".join(
                list_decode(feature_tokens[sent_idx][1:], tokenizer)[0]
            )
            rewritten_sentence = rewritten_sentences_dict.get(sent_idx)

            if rewritten_sentence is None:
                print(
                    f"  - WARN: Could not parse rewritten sentence {sent_idx} for feature {feature_idx}."
                )
                original_sentence = ""
                rewritten_sentence = ""
                num_invalid_sentences += 1
            else:
                num_valid_sentences += 1

            # Add data to our batch lists
            sentences_to_process["original"].append(original_sentence)
            sentences_to_process["rewritten"].append(rewritten_sentence)
            sentences_to_process["feature_indices"].append(feature_idx)
            # Store the location: (index in all_results_data, sentence_index)
            sentences_to_process["reconstruction_info"].append((i, sent_idx))

    print(
        f"Found {num_valid_sentences} valid sentences and {num_invalid_sentences} invalid sentences."
    )

    for i in tqdm(
        range(0, len(sentences_to_process["original"]), batch_size),
        desc="Evaluating sentence activations",
    ):
        if i % 100 == 0:
            torch.cuda.empty_cache()
        pos_statements = sentences_to_process["original"][i : i + batch_size]
        neg_statements = sentences_to_process["rewritten"][i : i + batch_size]
        feature_indices = sentences_to_process["feature_indices"][i : i + batch_size]
        reconstruction_info = sentences_to_process["reconstruction_info"][
            i : i + batch_size
        ]
        all_sentence_data, all_sentence_metrics = introspect_utils.eval_statements(
            pos_statements,
            neg_statements,
            feature_indices,
            model,
            submodule,
            sae,
            tokenizer,
        )

        for j, (feature_results_idx, sent_idx) in enumerate(reconstruction_info):
            feature_results[feature_results_idx]["sentence_data"].append(
                all_sentence_data[j]
            )
            feature_results[feature_results_idx]["sentence_metrics"].append(
                all_sentence_metrics[j]
            )

    return feature_results


# ==============================================================================
# 3. MAIN EXPERIMENT LOGIC
# ==============================================================================
async def main(cfg: SelfInterpTrainingConfig):
    """Main function to run the contrastive rewriting experiment."""
    print("--- Starting Contrastive Rewriting Experiment ---")
    print(f"Configuration: {asdict(cfg)}")

    # 1. Setup: Seeding, loading models, data, and API
    random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    dtype = torch.bfloat16
    device = torch.device("cuda")

    acts_data = load_acts(
        cfg.model_name,
        cfg.sae_layer,
        cfg.sae_width,
        cfg.layer_percent,
        cfg.context_length,
    )
    caller = api_caller.get_openai_caller("cached_api_calls")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # 2. Feature Selection and Prompt Generation
    num_features_available = acts_data["max_acts"].shape[0]
    feature_indices = random.sample(
        range(num_features_available),
        min(cfg.api_num_features_to_run, num_features_available),
    )
    print(f"\nSelected {len(feature_indices)} random features to analyze.")

    all_prompts = []
    print("Generating prompts for each feature...")
    for feature_idx in feature_indices:
        feature_acts = acts_data["max_acts"][
            feature_idx, : cfg.api_num_sentences_per_feature
        ]
        feature_tokens = acts_data["max_tokens"][
            feature_idx, : cfg.api_num_sentences_per_feature
        ]
        prompt = make_contrastive_prompt(feature_acts, feature_tokens, tokenizer)
        all_prompts.append(prompt)

    # 3. Parallel API Calls
    print(
        f"Sending {len(all_prompts)} prompts to {cfg.api_model_name} for rewriting..."
    )

    total_chars = sum(len(prompt) for prompt in all_prompts)
    print(f"Total characters: {total_chars}")

    chat_prompts = build_prompts(all_prompts)
    try:
        api_responses = await api_caller.run_api_model(
            cfg.api_model_name,
            chat_prompts,
            caller,
            temperature=cfg.api_generation_kwargs["temperature"],
            max_tokens=cfg.api_generation_kwargs["max_tokens"],
            max_par=cfg.api_generation_kwargs["max_par"],
        )
    finally:
        caller.client.close()
    print("Received all API responses.")

    cfg.sae_width, cfg.sae_layer, cfg.sae_layer_percent, cfg.sae_filename = (
        get_sae_info(cfg.sae_repo_id)
    )

    model = introspect_utils.load_model(cfg, device, dtype, use_lora=False)
    sae = introspect_utils.load_sae(cfg, device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    submodule = model_utils.get_submodule(model, cfg.sae_layer)

    all_results_data = parallel_eval(
        cfg,
        feature_indices,
        all_prompts,
        api_responses,
        acts_data,
        model,
        tokenizer,
        submodule,
        sae,
        batch_size=32,
    )

    # 5. Save Results
    print(f"\nSaving all results to {cfg.training_data_filename}...")
    with open(cfg.training_data_filename, "wb") as f:
        # We can also save the config itself for perfect reproducibility
        pickle.dump({"config": asdict(cfg), "results": all_results_data}, f)

    print("--- Experiment Complete ---")


# ==============================================================================
# 4. SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Ensure you are in an environment that supports asyncio, like a Jupyter notebook
    # or a standard Python script.
    cfg = SelfInterpTrainingConfig()
    # cfg.model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # cfg.sae_repo_id = "fnlp/Llama3_1-8B-Base-LXR-32x"

    cfg.model_name = "google/gemma-2-9b-it"
    cfg.sae_repo_id = "google/gemma-scope-9b-it-res"
    cfg.api_num_features_to_run = 2000

    model_name_str = cfg.model_name.replace("/", "_").replace(".", "_")

    cfg.training_data_filename = f"contrastive_rewriting_results_{model_name_str}_num_features_{cfg.api_num_features_to_run}.pkl"
    print(f"Results will be saved to {cfg.training_data_filename}")
    # This is the standard way to run an async function from a sync context.
    asyncio.run(main(cfg))
