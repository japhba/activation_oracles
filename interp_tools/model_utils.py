from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional
import contextlib


def add_mistral_v3_chat_template(prompts: list[str]) -> list[str]:
    return [f"<s>[INST]{prompt}[/INST]" for prompt in prompts]


def add_gemma_chat_template(prompts: list[str]) -> list[str]:
    return [
        f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        for prompt in prompts
    ]


def add_qwen_chat_template(prompts: list[str]) -> list[str]:
    return [
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        for prompt in prompts
    ]


def add_chat_template(prompts: list[str], model_name: str) -> list[str]:
    if (
        model_name == "mistralai/Ministral-8B-Instruct-2410"
        or model_name == "mistralai/Mistral-Small-24B-Instruct-2501"
    ):
        return add_mistral_v3_chat_template(prompts)
    elif (
        model_name == "google/gemma-2-9b-it"
        or model_name == "google/gemma-2-27b-it"
        or model_name == "google/gemma-2-2b-it"
    ):
        return add_gemma_chat_template(prompts)
    elif model_name == "Qwen/Qwen2.5-3B-Instruct":
        return add_qwen_chat_template(prompts)
    else:
        raise ValueError(f"Please implement a chat template for {model_name}")

    # We don't use this because we want the model's first token to be it's response, not a formatting token
    # formatted_prompts = []
    # for prompt in prompts:
    #     formatted_prompt = tokenizer.apply_chat_template(
    #         [{"role": "user", "content": prompt}],
    #         tokenize=False,
    #     )
    #     formatted_prompts.append(formatted_prompt)
    # return formatted_prompts


def collect_token_ids(tokenizer: AutoTokenizer, candidates: list[str]) -> list[int]:
    """
    Collect the set of token IDs that appear when tokenizing
    any candidate string in `candidates`.
    """
    token_ids = set()
    for c in candidates:
        # add_special_tokens=False so we only get raw subwords
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) > 1:
            ids = ids[:1]
        assert len(ids) == 1, f"Expected 1 token ID, got {len(ids)} for {c}"
        token_ids.update(ids)

    return list(token_ids)


def get_yes_no_ids(
    tokenizer: AutoTokenizer,
    device: torch.device,
    yes_candidates: Optional[list[str]] = None,
    no_candidates: Optional[list[str]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if yes_candidates is None:
        yes_candidates = ["yes", " yes", "Yes", " Yes", "YES", " YES"]
    if no_candidates is None:
        no_candidates = ["no", " no", "No", " No", "NO", " NO"]

    yes_ids = collect_token_ids(tokenizer, yes_candidates)
    no_ids = collect_token_ids(tokenizer, no_candidates)

    yes_ids_t = torch.tensor(yes_ids).to(device)  # for indexing
    no_ids_t = torch.tensor(no_ids).to(device)

    return yes_ids_t, no_ids_t


def get_yes_no_probs(
    tokenizer: AutoTokenizer,
    logits_BV: torch.Tensor,
    yes_candidates: Optional[list[str]] = None,
    no_candidates: Optional[list[str]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = logits_BV.device
    yes_ids_t, no_ids_t = get_yes_no_ids(
        tokenizer,
        yes_candidates=yes_candidates,
        no_candidates=no_candidates,
        device=device,
    )

    probs_BV = torch.nn.functional.softmax(logits_BV, dim=-1)
    yes_probs = probs_BV.index_select(1, yes_ids_t)
    no_probs = probs_BV.index_select(1, no_ids_t)

    yes_probs_B = yes_probs.sum(dim=1)
    no_probs_B = no_probs.sum(dim=1)

    return yes_probs_B, no_probs_B


def get_submodule(model: AutoModelForCausalLM, layer: int):
    """Gets the residual stream submodule"""
    model_name = model.config._name_or_path

    if "pythia" in model_name:
        return model.gpt_neox.layers[layer]
    elif "gemma" in model_name or "mistral" in model_name or "Llama" in model_name:
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")


class EarlyStopException(Exception):
    """Custom exception for stopping model forward pass early."""

    pass


def collect_activations(
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    inputs_BL: dict[str, torch.Tensor],
    use_no_grad: bool = True,
) -> torch.Tensor:
    """
    Registers a forward hook on the submodule to capture the residual (or hidden)
    activations. We then raise an EarlyStopException to skip unneeded computations.

    Args:
        model: The model to run.
        submodule: The submodule to hook into.
        inputs_BL: The inputs to the model.
        use_no_grad: Whether to run the forward pass within a `torch.no_grad()` context. Defaults to True.
    """
    activations_BLD = None

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal activations_BLD
        # For many models, the submodule outputs are a tuple or a single tensor:
        # If "outputs" is a tuple, pick the relevant item:
        #   e.g. if your layer returns (hidden, something_else), you'd do outputs[0]
        # Otherwise just do outputs
        if isinstance(outputs, tuple):
            activations_BLD = outputs[0]
        else:
            activations_BLD = outputs

        raise EarlyStopException("Early stopping after capturing activations")

    handle = submodule.register_forward_hook(gather_target_act_hook)

    # Determine the context manager based on the flag
    context_manager = torch.no_grad() if use_no_grad else contextlib.nullcontext()

    try:
        # Use the selected context manager
        with context_manager:
            _ = model(**inputs_BL)
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        handle.remove()

    return activations_BLD


def get_activations_per_layer(
    model: AutoModelForCausalLM,
    submodules: list[torch.nn.Module],
    tokens_batch: dict[str, torch.Tensor],
    get_final_token_only: bool = False,
) -> tuple[dict[torch.nn.Module, torch.Tensor], torch.Tensor]:
    activations_BLD = {}

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal activations_BLD
        assert isinstance(outputs, tuple)
        if get_final_token_only:
            activations_BLD[module] = outputs[0][:, -1:, :]
        else:
            activations_BLD[module] = outputs[0]

    all_handles = []

    try:
        for submodule in submodules:
            handle = submodule.register_forward_hook(gather_target_act_hook)
            all_handles.append(handle)

        logits_BLV = model(**tokens_batch).logits

        if get_final_token_only:
            logits_BLV = logits_BLV[:, -1:, :]

    except Exception as e:
        print(e)
        raise e
    finally:
        for handle in all_handles:
            handle.remove()

    return activations_BLD, logits_BLV
