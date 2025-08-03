# ==============================================================================
# 0. IMPORTS AND SETUP
# ==============================================================================
import torch
import contextlib

# New imports for dataclasses
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Dict, Tuple, Optional, Any
from jaxtyping import Float
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
import os
from huggingface_hub import hf_hub_download
import random
import pickle
import asyncio
import re
from rapidfuzz.distance import Levenshtein as lev

import interp_tools.saes.jumprelu_sae as jumprelu_sae
import interp_tools.model_utils as model_utils
import interp_tools.api_utils.shared as shared
import interp_tools.api_utils.api_caller as api_caller


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
@dataclass
class Config:
    """Configuration settings for the script."""

    # --- Foundational Settings ---
    model_name: str = "google/gemma-2-9b-it"

    # --- SAE (Sparse Autoencoder) Settings ---
    sae_repo_id: str = "google/gemma-scope-9b-it-res"
    sae_layer: int = 9
    sae_width: int = 131  # For loading the correct max acts file
    layer_percent: int = 25  # For loading the correct max acts file
    context_length: int = 32

    sae_filename: str = field(init=False)

    # --- Experiment Settings ---
    num_features_to_run: int = 200  # How many random features to analyze
    num_sentences_per_feature: int = (
        5  # How many top-activating examples to use per feature
    )
    random_seed: int = 42  # For reproducible feature selection
    results_filename: str = "contrastive_rewriting_results.pkl"

    # --- API and Generation Settings ---
    api_model_name: str = "gpt-4.1-mini"
    # Use a default_factory for mutable types like dicts
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "temperature": 1.0,
            "max_tokens": 2000,
            "max_par": 200,  # Max parallel requests
        }
    )

    def __post_init__(self):
        """Called after the dataclass is initialized."""
        if self.sae_width == 16:
            self.sae_filename = (
                f"layer_{self.sae_layer}/width_16k/average_l0_88/params.npz"
            )
        elif self.sae_width == 131:
            self.sae_filename = f"layer_9/width_131k/average_l0_121/params.npz"
        else:
            raise ValueError(f"Unknown SAE width: {self.sae_width}")


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================


def load_sae_and_model(
    cfg: Config,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, jumprelu_sae.JumpReluSAE]:
    """Loads the model, tokenizer, and SAE from Hugging Face."""
    print(f"Loading model: {cfg.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, device_map="auto", torch_dtype=dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    print(f"Loading SAE for layer {cfg.sae_layer} from {cfg.sae_repo_id}...")
    sae = jumprelu_sae.load_gemma_scope_jumprelu_sae(
        repo_id=cfg.sae_repo_id,
        filename=cfg.sae_filename,
        layer=cfg.sae_layer,
        model_name=cfg.model_name,
        device=device,
        dtype=dtype,
    )

    print("Model, tokenizer, and SAE loaded successfully.")
    return model, tokenizer, sae


def get_bos_eos_pad_mask(
    tokenizer: PreTrainedTokenizer, token_ids: torch.Tensor
) -> torch.Tensor:
    return (
        (token_ids == tokenizer.pad_token_id)
        | (token_ids == tokenizer.eos_token_id)
        | (token_ids == tokenizer.bos_token_id)
    ).to(dtype=torch.bool)


def get_feature_activations_v2(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    sae: jumprelu_sae.JumpReluSAE,
    tokenized_strs: dict[str, torch.Tensor],
    ignore_bos: bool = True,
) -> torch.Tensor:
    with torch.no_grad():
        pos_acts_BLD = model_utils.collect_activations(model, submodule, tokenized_strs)

        encoded_pos_acts_BLF = sae.encode(pos_acts_BLD)

    if ignore_bos:
        bos_mask = tokenized_strs.input_ids == tokenizer.bos_token_id
        assert bos_mask.sum() == encoded_pos_acts_BLF.shape[0], (
            f"Expected {encoded_pos_acts_BLF.shape[0]} BOS tokens, but found {bos_mask.sum()}"
        )

        encoded_pos_acts_BLF[bos_mask] = 0

        mask = get_bos_eos_pad_mask(tokenizer, tokenized_strs.input_ids)
        encoded_pos_acts_BLF[mask] = 0

    return encoded_pos_acts_BLF


def get_feature_activations(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    sae: jumprelu_sae.JumpReluSAE,
    input_str: str,
    feature_idx: int,
    verbose: bool = False,
    add_bos: bool = True,
    ignore_bos: bool = True,
    use_chat_template: bool = False,
) -> torch.Tensor:
    """
    Calculates and prints the SAE feature activations for a pair of sentences.

    This helps verify if a feature responds more strongly to the positive example,
    as predicted by the model's generated explanation.
    """

    if use_chat_template:
        input_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_str}],
            tokenize=False,
            add_generation_prompt=True,
        )

    tokenized_str = tokenizer(
        input_str, return_tensors="pt", add_special_tokens=add_bos
    ).to(model.device)

    with torch.no_grad():
        pos_acts_BLD = model_utils.collect_activations(model, submodule, tokenized_str)

        encoded_pos_acts_BLF = sae.encode(pos_acts_BLD)

    assert encoded_pos_acts_BLF.shape[0] == 1, (
        "Only batch size 1 is currently supported"
    )

    pos_feature_acts = encoded_pos_acts_BLF[0, :, feature_idx]

    if ignore_bos:
        bos_mask = tokenized_str.input_ids[0, :] == tokenizer.bos_token_id
        assert bos_mask.sum() > 0, "BOS token not found in input"
        pos_feature_acts[bos_mask] = 0

    if verbose:
        print(
            f"Feature {feature_idx} max activation: {pos_feature_acts.max():.4f}, mean: {pos_feature_acts.mean():.4f}"
        )

    return pos_feature_acts


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


@torch.no_grad()
def eval_statements(
    pos_statements: list[str],
    neg_statements: list[str],
    feature_indices: list[int],
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    sae: jumprelu_sae.JumpReluSAE,
    tokenizer: PreTrainedTokenizer,
) -> tuple[list[dict], list[dict]]:
    assert len(pos_statements) == len(neg_statements) == len(feature_indices)

    pos_tokenized_strs = tokenizer(
        pos_statements, return_tensors="pt", add_special_tokens=True, padding=True
    ).to(model.device)

    neg_tokenized_strs = tokenizer(
        neg_statements, return_tensors="pt", add_special_tokens=True, padding=True
    ).to(model.device)

    # TODO: shrink SAE to only the features we're interested in for reduced memory usage
    pos_activations_BLF = get_feature_activations_v2(
        model=model,
        tokenizer=tokenizer,
        submodule=submodule,
        sae=sae,
        tokenized_strs=pos_tokenized_strs,
    )

    neg_activations_BLF = get_feature_activations_v2(
        model=model,
        tokenizer=tokenizer,
        submodule=submodule,
        sae=sae,
        tokenized_strs=neg_tokenized_strs,
    )

    all_sentence_data = []
    all_sentence_metrics = []

    for i in range(len(pos_statements)):
        feature_idx = feature_indices[i]

        pos_tokens_L = pos_tokenized_strs.input_ids[i, :]
        neg_tokens_L = neg_tokenized_strs.input_ids[i, :]

        pos_mask_L = get_bos_eos_pad_mask(tokenizer, pos_tokens_L)
        neg_mask_L = get_bos_eos_pad_mask(tokenizer, neg_tokens_L)

        pos_activations_L = pos_activations_BLF[i, :, feature_idx]
        neg_activations_L = neg_activations_BLF[i, :, feature_idx]

        sentence_distance = lev.normalized_distance(
            pos_statements[i], neg_statements[i]
        )

        original_max_activation = pos_activations_L.max().item()
        rewritten_max_activation = neg_activations_L.max().item()

        valid_pos_tokens = len(pos_mask_L) - pos_mask_L.sum()
        valid_neg_tokens = len(neg_mask_L) - neg_mask_L.sum()

        original_mean_activation = pos_activations_L.sum() / valid_pos_tokens
        rewritten_mean_activation = neg_activations_L.sum() / valid_neg_tokens
        original_mean_activation = original_mean_activation.item()
        rewritten_mean_activation = rewritten_mean_activation.item()

        max_activation_ratio = rewritten_max_activation / (
            original_max_activation + 1e-6
        )
        mean_activation_ratio = rewritten_mean_activation / (
            original_mean_activation + 1e-6
        )

        max_activation_ratio = max(max_activation_ratio, 0.0)
        mean_activation_ratio = max(mean_activation_ratio, 0.0)

        max_activation_ratio = min(max_activation_ratio, 1.0)
        mean_activation_ratio = min(mean_activation_ratio, 1.0)

        format_correct = 1.0
        if pos_statements[i] == "" and neg_statements[i] == "":
            format_correct = 0.0

        max_activation_nonzero = 0.0
        if original_max_activation > 0.0 and format_correct == 1.0:
            max_activation_nonzero = 1.0

        success_max_activation_ratio = 1.0
        success_mean_activation_ratio = 1.0

        if format_correct == 1.0 and max_activation_nonzero == 1.0:
            success_max_activation_ratio = max_activation_ratio
            success_mean_activation_ratio = mean_activation_ratio

        sentence_data = {
            "sentence_index": 0,
            "original_sentence": pos_statements[i],
            "rewritten_sentence": neg_statements[i],
        }

        sentence_metrics = {
            "original_max_activation": original_max_activation,
            "rewritten_max_activation": rewritten_max_activation,
            "original_mean_activation": original_mean_activation,
            "rewritten_mean_activation": rewritten_mean_activation,
            "max_activation_ratio": max_activation_ratio,
            "mean_activation_ratio": mean_activation_ratio,
            "sentence_distance": sentence_distance,
            "format_correct": format_correct,
            "max_activation_nonzero": max_activation_nonzero,
            "success_max_activation_ratio": success_max_activation_ratio,
            "success_mean_activation_ratio": success_mean_activation_ratio,
        }

        all_sentence_data.append(sentence_data)
        all_sentence_metrics.append(sentence_metrics)

    return all_sentence_data, all_sentence_metrics


def parallel_eval(
    cfg: Config,
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

    for i, feature_idx in enumerate(feature_indices):
        api_prompt = all_prompts[i]
        api_response = api_responses[i]

        feature_tokens = acts_data["max_tokens"][
            feature_idx, : cfg.num_sentences_per_feature
        ]
        explanation, planned_edits, rewritten_sentences_dict = parse_llm_response(
            api_response, cfg.num_sentences_per_feature
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

        for sent_idx in range(cfg.num_sentences_per_feature):
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

            # Add data to our batch lists
            sentences_to_process["original"].append(original_sentence)
            sentences_to_process["rewritten"].append(rewritten_sentence)
            sentences_to_process["feature_indices"].append(feature_idx)
            # Store the location: (index in all_results_data, sentence_index)
            sentences_to_process["reconstruction_info"].append((i, sent_idx))

    for i in range(0, len(sentences_to_process["original"]), batch_size):
        pos_statements = sentences_to_process["original"][i : i + batch_size]
        neg_statements = sentences_to_process["rewritten"][i : i + batch_size]
        feature_indices = sentences_to_process["feature_indices"][i : i + batch_size]
        reconstruction_info = sentences_to_process["reconstruction_info"][
            i : i + batch_size
        ]
        all_sentence_data, all_sentence_metrics = eval_statements(
            pos_statements,
            neg_statements,
            feature_indices,
            model,
            submodule,
            sae,
            tokenizer,
        )

        for j, (feature_results_idx, sent_idx) in enumerate(reconstruction_info):
            feature_result = feature_results[feature_results_idx]
            feature_result["sentence_data"].append(all_sentence_data[j])
            feature_result["sentence_metrics"].append(all_sentence_metrics[j])

    return feature_results


def serial_eval(
    cfg: Config,
    feature_indices: list[int],
    all_prompts: list[str],
    api_responses: list[str],
    acts_data: dict[str, torch.Tensor],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    sae: jumprelu_sae.JumpReluSAE,
) -> list[dict]:
    all_results_data = []

    for i, feature_idx in enumerate(feature_indices):
        api_prompt = all_prompts[i]
        api_response = api_responses[i]

        feature_tokens = acts_data["max_tokens"][
            feature_idx, : cfg.num_sentences_per_feature
        ]
        explanation, planned_edits, rewritten_sentences_dict = parse_llm_response(
            api_response, cfg.num_sentences_per_feature
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

        # TODO: do in parallel
        for sent_idx in range(cfg.num_sentences_per_feature):
            original_sentence = "".join(
                list_decode(feature_tokens[sent_idx][1:], tokenizer)[0]
            )
            rewritten_sentence = rewritten_sentences_dict.get(sent_idx)

            if rewritten_sentence is None:
                print(
                    f"  - WARN: Could not parse rewritten sentence {sent_idx} for feature {feature_idx}."
                )
                continue

            # Calculate activations
            with torch.no_grad():
                original_acts = get_feature_activations(
                    model,
                    tokenizer,
                    submodule,
                    sae,
                    original_sentence,
                    feature_idx,
                    use_chat_template=False,
                    ignore_bos=True,
                )
                rewritten_acts = get_feature_activations(
                    model,
                    tokenizer,
                    submodule,
                    sae,
                    rewritten_sentence,
                    feature_idx,
                    use_chat_template=False,
                    ignore_bos=True,
                )

            dist = sentence_distance(original_sentence, rewritten_sentence)

            sentence_data = {
                "sentence_index": sent_idx,
                "original_sentence": original_sentence,
                "rewritten_sentence": rewritten_sentence,
            }

            sentence_metrics = {
                "original_max_activation": original_acts.max().item(),
                "rewritten_max_activation": rewritten_acts.max().item(),
                "original_mean_activation": original_acts.mean().item(),
                "rewritten_mean_activation": rewritten_acts.mean().item(),
                "sentence_distance": dist,
            }

            feature_result["sentence_data"].append(sentence_data)
            feature_result["sentence_metrics"].append(sentence_metrics)

        all_results_data.append(feature_result)
        print(f"  - Processed feature {feature_idx} ({i + 1}/{len(feature_indices)})")

    return all_results_data


# ==============================================================================
# 3. MAIN EXPERIMENT LOGIC
# ==============================================================================
async def main(cfg: Config):
    """Main function to run the contrastive rewriting experiment."""
    print("--- Starting Contrastive Rewriting Experiment ---")
    print(f"Configuration: {asdict(cfg)}")

    # 1. Setup: Seeding, loading models, data, and API
    random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    dtype = torch.bfloat16
    dtype = torch.float32
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
        min(cfg.num_features_to_run, num_features_available),
    )
    print(f"\nSelected {len(feature_indices)} random features to analyze.")

    all_prompts = []
    print("Generating prompts for each feature...")
    for feature_idx in feature_indices:
        feature_acts = acts_data["max_acts"][
            feature_idx, : cfg.num_sentences_per_feature
        ]
        feature_tokens = acts_data["max_tokens"][
            feature_idx, : cfg.num_sentences_per_feature
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
            temperature=cfg.generation_kwargs["temperature"],
            max_tokens=cfg.generation_kwargs["max_tokens"],
            max_par=cfg.generation_kwargs["max_par"],
        )
    finally:
        caller.client.close()
    print("Received all API responses.")

    model, tokenizer, sae = load_sae_and_model(cfg, device, dtype)
    submodule = model_utils.get_submodule(model, cfg.sae_layer)

    # 4. Process Results and Calculate Activations
    print("Processing responses and calculating activations...")

    all_results_data_v2 = parallel_eval(
        cfg,
        feature_indices,
        all_prompts,
        api_responses,
        acts_data,
        model,
        tokenizer,
        submodule,
        sae,
        batch_size=10,
        # batch_size=2,
    )

    all_results_data = serial_eval(
        cfg,
        feature_indices,
        all_prompts,
        api_responses,
        acts_data,
        model,
        tokenizer,
        submodule,
        sae,
    )

    temp_filename_1 = "contrastive_rewriting_results_1.pkl"
    temp_filename_2 = "contrastive_rewriting_results_2.pkl"

    with open(temp_filename_1, "wb") as f:
        pickle.dump({"config": asdict(cfg), "results": all_results_data}, f)
    with open(temp_filename_2, "wb") as f:
        pickle.dump({"config": asdict(cfg), "results": all_results_data_v2}, f)

    for i, result1 in enumerate(all_results_data):
        print(result1["feature_idx"])
        result2 = all_results_data_v2[i]

        # print(result1["feature_idx"])
        # print(result2["feature_idx"])
        # print(result1["sentence_metrics"])
        # print(result2["sentence_metrics"])
        # print()

        sentence_metrics1 = result1["sentence_metrics"][0]
        sentence_metrics2 = result2["sentence_metrics"][0]

        sentence_data1 = result1["sentence_data"][0]
        sentence_data2 = result2["sentence_data"][0]

        for key in sentence_data1.keys():
            assert sentence_data1[key] == sentence_data2[key]

        for key in sentence_metrics1.keys():
            # print(key)
            # print(sentence_metrics1[key])
            # print(sentence_metrics2[key])
            # print()
            if sentence_metrics1[key] != sentence_metrics2[key]:
                print(key)
                print(sentence_metrics1[key])
                print(sentence_metrics2[key])
                print()
            # assert sentence_metrics1[key] == sentence_metrics2[key]


# ==============================================================================
# 4. SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Ensure you are in an environment that supports asyncio, like a Jupyter notebook
    # or a standard Python script.
    cfg = Config()
    cfg.num_features_to_run = 20
    # This is the standard way to run an async function from a sync context.
    asyncio.run(main(cfg))
