# %%
# ==============================================================================
# 0. IMPORTS AND SETUP
# ==============================================================================
import torch
import contextlib
from typing import Callable, List, Dict, Tuple, Optional
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

# Make sure you have installed the necessary packages:
# pip install torch transformers jaxtyping einops transformers_stream_generator safetensors sentencepiece accelerate rapidfuzz
# pip install git+https://github.com/google-deepmind/interp_tools.git
import interp_tools.saes.jumprelu_sae as jumprelu_sae
import interp_tools.model_utils as model_utils
import interp_tools.api_utils.shared as shared
import interp_tools.api_utils.api_caller as api_caller


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    """Configuration settings for the script."""

    # --- Foundational Settings ---
    MODEL_NAME = "google/gemma-2-9b-it"
    DTYPE = torch.bfloat16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- SAE (Sparse Autoencoder) Settings ---
    SAE_REPO_ID = "google/gemma-scope-9b-it-res"
    SAE_LAYER = 9
    SAE_FILENAME = f"layer_{SAE_LAYER}/width_16k/average_l0_88/params.npz"
    SAE_WIDTH = 16  # For loading the correct max acts file
    LAYER_PERCENT = 25  # For loading the correct max acts file

    # --- Experiment Settings ---
    NUM_FEATURES_TO_RUN = 200  # How many random features to analyze
    NUM_SENTENCES_PER_FEATURE = 5  # How many top-activating examples to use per feature
    RANDOM_SEED = 42  # For reproducible feature selection
    RESULTS_FILENAME = "contrastive_rewriting_results.pkl"

    # --- API and Generation Settings ---
    API_MODEL_NAME = "gpt-4o"
    GENERATION_KWARGS = {
        "temperature": 1.0,
        "max_tokens": 2000,
        "max_par": 100,  # Max parallel requests
    }


# ==============================================================================
# 2. UTILITY FUNCTIONS (UNMODIFIED)
# ==============================================================================


def load_sae_and_model(
    cfg: Config,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, jumprelu_sae.JumpReluSAE]:
    """Loads the model, tokenizer, and SAE from Hugging Face."""
    print(f"Loading model: {cfg.MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME, device_map="auto", torch_dtype=cfg.DTYPE
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)

    print(f"Loading SAE for layer {cfg.SAE_LAYER} from {cfg.SAE_REPO_ID}...")
    sae = jumprelu_sae.load_gemma_scope_jumprelu_sae(
        repo_id=cfg.SAE_REPO_ID,
        filename=cfg.SAE_FILENAME,
        layer=cfg.SAE_LAYER,
        model_name=cfg.MODEL_NAME,
        device=cfg.DEVICE,
        dtype=cfg.DTYPE,
    )

    print("Model, tokenizer, and SAE loaded successfully.")
    return model, tokenizer, sae


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
    model_name: str, sae_layer: int, sae_width: int, layer_percent: int
) -> dict[str, torch.Tensor]:
    acts_dir = "max_acts"
    acts_filename = f"acts_{model_name}_layer_{sae_layer}_trainer_{sae_width}_layer_percent_{layer_percent}.pt".replace(
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


# ==============================================================================
# 3. MAIN EXPERIMENT LOGIC
# ==============================================================================
async def main(cfg: Config):
    """Main function to run the contrastive rewriting experiment."""
    print("--- Starting Contrastive Rewriting Experiment ---")
    print(f"Configuration: {vars(cfg)}")

    # 1. Setup: Seeding, loading models, data, and API
    random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)

    model, tokenizer, sae = load_sae_and_model(cfg)
    submodule = model_utils.get_submodule(model, cfg.SAE_LAYER)
    acts_data = load_acts(
        cfg.MODEL_NAME, cfg.SAE_LAYER, cfg.SAE_WIDTH, cfg.LAYER_PERCENT
    )
    caller = api_caller.get_openai_caller("cached_api_calls")

    # 2. Feature Selection and Prompt Generation
    num_features_available = acts_data["max_acts"].shape[0]
    feature_indices = random.sample(
        range(num_features_available), cfg.NUM_FEATURES_TO_RUN
    )
    print(f"\nSelected {len(feature_indices)} random features to analyze.")

    all_prompts = []
    print("Generating prompts for each feature...")
    for feature_idx in feature_indices:
        feature_acts = acts_data["max_acts"][
            feature_idx, : cfg.NUM_SENTENCES_PER_FEATURE
        ]
        feature_tokens = acts_data["max_tokens"][
            feature_idx, : cfg.NUM_SENTENCES_PER_FEATURE
        ]
        prompt = make_contrastive_prompt(feature_acts, feature_tokens, tokenizer)
        all_prompts.append(prompt)

    # 3. Parallel API Calls
    print(
        f"Sending {len(all_prompts)} prompts to {cfg.API_MODEL_NAME} for rewriting..."
    )
    chat_prompts = build_prompts(all_prompts)
    try:
        api_responses = await api_caller.run_api_model(
            cfg.API_MODEL_NAME,
            chat_prompts,
            caller,
            temperature=cfg.GENERATION_KWARGS["temperature"],
            max_tokens=cfg.GENERATION_KWARGS["max_tokens"],
            max_par=cfg.GENERATION_KWARGS["max_par"],
        )
    finally:
        caller.client.close()
    print("Received all API responses.")

    # 4. Process Results and Calculate Activations
    print("Processing responses and calculating activations...")
    all_results_data = []

    for i, feature_idx in enumerate(feature_indices):
        api_prompt = all_prompts[i]
        api_response = api_responses[i]

        feature_tokens = acts_data["max_tokens"][
            feature_idx, : cfg.NUM_SENTENCES_PER_FEATURE
        ]
        explanation, planned_edits, rewritten_sentences_dict = parse_llm_response(
            api_response, cfg.NUM_SENTENCES_PER_FEATURE
        )

        feature_result = {
            "feature_idx": feature_idx,
            "api_prompt": api_prompt,
            "api_response": api_response,
            "explanation": explanation,
            "planned_edits": planned_edits,
            "sentence_data": [],
        }

        # TODO: do in parallel
        for sent_idx in range(cfg.NUM_SENTENCES_PER_FEATURE):
            # Reconstruct original sentence (including BOS token string, matching prior logic)
            original_sentence = "".join(
                list_decode(feature_tokens[sent_idx], tokenizer)[0]
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

            feature_result["sentence_data"].append(
                {
                    "sentence_index": sent_idx,
                    "original_sentence": original_sentence,
                    "rewritten_sentence": rewritten_sentence,
                    "original_activations": original_acts.cpu(),
                    "rewritten_activations": rewritten_acts.cpu(),
                    "original_max_activation": original_acts.max().item(),
                    "rewritten_max_activation": rewritten_acts.max().item(),
                    "original_mean_activation": original_acts.mean().item(),
                    "rewritten_mean_activation": rewritten_acts.mean().item(),
                    "sentence_distance": dist,
                }
            )

        all_results_data.append(feature_result)
        print(f"  - Processed feature {feature_idx} ({i + 1}/{len(feature_indices)})")

    # 5. Save Results
    print(f"\nSaving all results to {cfg.RESULTS_FILENAME}...")
    with open(cfg.RESULTS_FILENAME, "wb") as f:
        pickle.dump(all_results_data, f)

    print("--- Experiment Complete ---")


# ==============================================================================
# 4. SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Ensure you are in an environment that supports asyncio, like a Jupyter notebook
    # or a standard Python script.
    cfg = Config()
    cfg.NUM_FEATURES_TO_RUN = 1000
    # This is the standard way to run an async function from a sync context.
    asyncio.run(main(cfg))
