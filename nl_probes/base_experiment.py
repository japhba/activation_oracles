# %%

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

from typing import Any
from tqdm import tqdm
from dataclasses import dataclass, field

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# nl_probes imports
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer, layer_percent_to_layer
from nl_probes.utils.dataset_utils import BinaryFeatureResult, TrainingDataPoint, create_training_datapoint
from nl_probes.utils.eval import run_binary_evaluation, run_evaluation
from nl_probes.configs.sft_config import SelfInterpTrainingConfig, read_training_config


# ========================================
# CONFIGURATION - edit here
# ========================================

@dataclass
class VerbalizerEvalConfig:
    """Config for run_verbalizer(). Only controls model/layer/generation settings.

    Callers are responsible for tokenization and position selection — those are
    specified per-input via VerbalizerInputInfo.
    """

    model_name: str

    # Computed in __post_init__ from selected_layer_combination
    selected_act_layers: list[int] = field(default_factory=list)

    injection_layer: int = 1
    layer_combinations: list[list[int]] = field(default_factory=lambda: [[25], [50], [75]])
    selected_layer_combination: list[int] = field(default_factory=lambda: [50])

    activation_input_types: list[str] = field(default_factory=lambda: ["orig", "lora", "diff"])

    verbalizer_generation_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "do_sample": True,
            "temperature": 0.7,
            "max_new_tokens": 40,
            "top_p": 0.9,
        }
    )

    steering_coefficient: float = 1.0
    eval_batch_size: int = 256

    def __post_init__(self):
        if self.selected_layer_combination not in self.layer_combinations:
            raise ValueError(
                f"selected_layer_combination {self.selected_layer_combination} must be in {self.layer_combinations}"
            )

        self.selected_act_layers = [
            layer_percent_to_layer(self.model_name, lp) for lp in self.selected_layer_combination
        ]

        valid_act_types = {"orig", "lora", "diff"}
        invalid = set(self.activation_input_types) - valid_act_types
        if invalid:
            raise ValueError(f"Invalid activation_input_types: {invalid}. Must be in {valid_act_types}")

        if "diff" in self.activation_input_types:
            if "lora" not in self.activation_input_types or "orig" not in self.activation_input_types:
                raise ValueError("Both 'lora' and 'orig' must be in activation_input_types when using 'diff'")


@dataclass
class VerbalizerInputInfo:
    """Info for a single verbalizer query. Caller pre-tokenizes and specifies positions."""

    context_token_ids: list[int]
    positions: list[int]
    verbalizer_prompt: str
    ground_truth: str


# ---------------------------------------------------------------------------
# Tokenization helpers — callers use these to pre-tokenize before building
# VerbalizerInputInfo with the new API.
# ---------------------------------------------------------------------------


def _tokenizer_has_thinking(tokenizer: AutoTokenizer) -> bool:
    """Check if the tokenizer supports thinking tags (e.g. Qwen3, DeepSeek R1)."""
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    return think_id != tokenizer.unk_token_id


def tokenize_chat_messages(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
    continue_final_message: bool = False,
    continue_thinking: bool = False,
) -> list[int]:
    """Apply chat template and tokenize. Returns token IDs (no padding).

    add_generation_prompt: Appends the assistant turn-start tokens (e.g. <|im_start|>assistant\n)
        so the model is ready to generate a new response.
    continue_final_message: Strips the end-of-turn tokens from the last message so the model
        continues generating within that same message. Useful when the last message is a
        partial assistant response you want the model to extend.
    continue_thinking: The last assistant message is a partial thinking trace (mid-thought).
        Prepends <think>\\n to the content so the tokenization reflects an open thinking block.
        For models without thinking tags, this is a no-op. Implies continue_final_message=True
        and enable_thinking=False (we manually add the open tag rather than letting the template
        add a paired <think></think> wrapper).

    add_generation_prompt and continue_final_message are mutually exclusive.
    """
    if continue_thinking:
        assert not add_generation_prompt, (
            "continue_thinking implies continue_final_message, which is mutually exclusive with add_generation_prompt"
        )
        assert messages[-1]["role"] == "assistant", "continue_thinking requires last message to be assistant"
        continue_final_message = True
        enable_thinking = False

    assert not (add_generation_prompt and continue_final_message), (
        "add_generation_prompt and continue_final_message are mutually exclusive"
    )
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        continue_final_message=continue_final_message,
    )

    if continue_thinking and _tokenizer_has_thinking(tokenizer):
        # The template inserts an empty <think>\n\n</think>\n\n block before assistant
        # content. Replace it with an open <think>\n to represent a mid-thought prefix.
        rendered = rendered.replace("<think>\n\n</think>\n\n", "<think>\n", 1)

    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def tokenize_raw_string(
    tokenizer: AutoTokenizer,
    text: str,
) -> list[int]:
    """Tokenize a pre-rendered string. Returns token IDs (no padding)."""
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def compute_segment_positions(
    num_tokens: int,
    start_idx: int = -10,
    end_idx: int = 0,
) -> list[int]:
    """Convert start/end indices (negative = from end) to absolute positions.

    Examples:
        compute_segment_positions(50, -10, 0) → [40, 41, ..., 49]
        compute_segment_positions(50, -10, -2) → [40, 41, ..., 47]
        compute_segment_positions(50, 5, 15) → [5, 6, ..., 14]
    """
    start = num_tokens + start_idx if start_idx < 0 else start_idx
    end = num_tokens + end_idx if end_idx <= 0 else end_idx
    start = max(0, start)
    end = min(num_tokens, end)
    return list(range(start, end))


@dataclass
class VerbalizerResults:
    verbalizer_lora_path: str | None
    target_lora_path: str | None
    context_token_ids: list[int]
    act_key: str
    verbalizer_prompt: str
    ground_truth: str
    num_tokens: int
    responses: list[str]


def _create_single_verbalizer_input(
    acts_BLD_by_layer_dict: dict[int, torch.Tensor],
    context_input_ids: list[int],
    positions: list[int],
    verbalizer_prompt: str,
    prompt_layers: list[int],
    tokenizer: AutoTokenizer,
    batch_idx: int,
    left_pad: int,
    base_meta: dict[str, Any],
) -> TrainingDataPoint:
    """Create a single TrainingDataPoint from the given positions."""
    context_positions_abs = [left_pad + p for p in positions]
    layer_acts: list[torch.Tensor] = []
    for layer in prompt_layers:
        acts_BLD = acts_BLD_by_layer_dict[layer][batch_idx, :]  # [L, D]
        acts_BD = acts_BLD[context_positions_abs]  # [P, D]
        layer_acts.append(acts_BD)
    acts_BD = torch.cat(layer_acts, dim=0)  # [num_layers * P, D]

    return create_training_datapoint(
        datapoint_type="N/A",
        prompt=verbalizer_prompt,
        target_response="N/A",
        layers=prompt_layers,
        num_positions=len(positions),
        tokenizer=tokenizer,
        acts_BD=acts_BD,
        feature_idx=-1,
        context_input_ids=context_input_ids,
        context_positions=positions,
        ds_label="N/A",
        meta_info=base_meta,
    )


def sanitize_lora_name(lora_path: str) -> str:
    return lora_path.replace(".", "_")


def collect_target_activations(
    model: AutoModelForCausalLM,
    inputs_BL: dict[str, torch.Tensor],
    config: VerbalizerEvalConfig,
    target_lora_path: str | None,
) -> dict[str, dict[int, torch.Tensor]]:
    act_types = {}
    is_peft_model = isinstance(model, PeftModel)

    # Collect activations for the whole batch under the requested target model.
    # A None target means the base model, matching training-time on-the-fly
    # materialization in nl_probes.utils.dataset_utils.
    if "lora" in config.activation_input_types:
        if target_lora_path is not None:
            if not is_peft_model:
                model.enable_adapters()
            model.set_adapter(target_lora_path)
            submodules = {layer: get_hf_submodule(model, layer) for layer in config.selected_act_layers}
            lora_acts = collect_activations_multiple_layers(
                model=model,
                submodules=submodules,
                inputs_BL=inputs_BL,
                min_offset=None,
                max_offset=None,
            )
        elif is_peft_model:
            print("\n\n\n\nWarning: target_lora_path is None, collecting lora activations from base model")
            with model.disable_adapter():
                submodules = {layer: get_hf_submodule(model, layer) for layer in config.selected_act_layers}
                lora_acts = collect_activations_multiple_layers(
                    model=model,
                    submodules=submodules,
                    inputs_BL=inputs_BL,
                    min_offset=None,
                    max_offset=None,
                )
        else:
            submodules = {layer: get_hf_submodule(model, layer) for layer in config.selected_act_layers}
            lora_acts = collect_activations_multiple_layers(
                model=model,
                submodules=submodules,
                inputs_BL=inputs_BL,
                min_offset=None,
                max_offset=None,
            )
        act_types["lora"] = lora_acts

    if "orig" in config.activation_input_types:
        if is_peft_model:
            with model.disable_adapter():
                submodules = {layer: get_hf_submodule(model, layer) for layer in config.selected_act_layers}
                orig_acts = collect_activations_multiple_layers(
                    model=model,
                    submodules=submodules,
                    inputs_BL=inputs_BL,
                    min_offset=None,
                    max_offset=None,
                )
                act_types["orig"] = orig_acts
        else:
            model.disable_adapters()
            submodules = {layer: get_hf_submodule(model, layer) for layer in config.selected_act_layers}
            orig_acts = collect_activations_multiple_layers(
                model=model,
                submodules=submodules,
                inputs_BL=inputs_BL,
                min_offset=None,
                max_offset=None,
            )
            act_types["orig"] = orig_acts
            model.enable_adapters()

    if "diff" in config.activation_input_types:
        assert "lora" in act_types and "orig" in act_types, "Both lora and orig activations must be collected for diff"
        diff_acts = {}
        for layer in config.selected_act_layers:
            diff_acts[layer] = act_types["lora"][layer] - act_types["orig"][layer]
            lora_sum = act_types["lora"][layer].sum().item()
            orig_sum = act_types["orig"][layer].sum().item()
            diff_sum = diff_acts[layer].sum().item()

            print(f"Layer {layer}: Lora sum={lora_sum:.2f}, Orig sum={orig_sum:.2f}, Diff sum={diff_sum:.2f}")

        act_types["diff"] = diff_acts
    return act_types


def _build_padded_batch_from_token_ids(
    token_id_lists: list[list[int]],
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build a left-padded batch from pre-tokenized sequences."""
    max_len = max(len(ids) for ids in token_id_lists)
    pad_id = tokenizer.pad_token_id
    padded_ids = []
    attn_masks = []
    for ids in token_id_lists:
        pad_len = max_len - len(ids)
        padded_ids.append([pad_id] * pad_len + ids)
        attn_masks.append([0] * pad_len + [1] * len(ids))
    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attn_masks, dtype=torch.long, device=device),
    }


def _prepare_verbalizer_inputs_for_batch(
    *,
    batch: list[VerbalizerInputInfo],
    start: int,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    target_lora_path: str | None,
    config: VerbalizerEvalConfig,
    device: torch.device,
) -> tuple[list[list[int]], list[TrainingDataPoint]]:
    token_id_lists = [info.context_token_ids for info in batch]
    inputs_BL = _build_padded_batch_from_token_ids(token_id_lists, tokenizer, device)

    target_activations = collect_target_activations(
        model=model,
        inputs_BL=inputs_BL,
        config=config,
        target_lora_path=target_lora_path,
    )

    seq_len = int(inputs_BL["input_ids"].shape[1])
    left_pads: list[int] = []
    for b_idx in range(len(batch)):
        attn = inputs_BL["attention_mask"][b_idx]
        real_len = int(attn.sum().item())
        left_pads.append(seq_len - real_len)

    verbalizer_inputs: list[TrainingDataPoint] = []
    for b_idx, info in enumerate(batch):
        left_pad = left_pads[b_idx]

        for act_key, acts_dict in target_activations.items():
            base_meta = {
                "target_lora_path": target_lora_path,
                "verbalizer_prompt": info.verbalizer_prompt,
                "ground_truth": info.ground_truth,
                "combo_index": start + b_idx,
                "act_key": act_key,
                "num_tokens": len(info.context_token_ids),
                "context_index_within_batch": b_idx,
            }

            verbalizer_inputs.append(
                _create_single_verbalizer_input(
                    acts_BLD_by_layer_dict=acts_dict,
                    context_input_ids=info.context_token_ids,
                    positions=info.positions,
                    verbalizer_prompt=info.verbalizer_prompt,
                    prompt_layers=config.selected_act_layers,
                    tokenizer=tokenizer,
                    batch_idx=b_idx,
                    left_pad=left_pad,
                    base_meta=base_meta,
                )
            )

    return token_id_lists, verbalizer_inputs


def run_verbalizer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    verbalizer_prompt_infos: list[VerbalizerInputInfo],
    verbalizer_lora_path: str | None,
    target_lora_path: str | None,
    config: VerbalizerEvalConfig,
    device: torch.device,
) -> list[VerbalizerResults]:
    """Run verbalizer evaluation.

    Each VerbalizerInputInfo provides pre-tokenized context_token_ids and
    explicit positions. No tokenization happens here.

    Assumptions: Both the verbalizer and lora path are LoRA adapters that have
    already been loaded into the model. Both can be None to use the original model.
    """

    dtype = torch.bfloat16
    injection_submodule = get_hf_submodule(model, config.injection_layer)

    pbar = tqdm(total=len(verbalizer_prompt_infos), desc="Verbalizer Eval Progress", position=1)
    results: list[VerbalizerResults] = []

    for start in range(0, len(verbalizer_prompt_infos), config.eval_batch_size):
        batch = verbalizer_prompt_infos[start : start + config.eval_batch_size]
        token_id_lists, verbalizer_inputs = _prepare_verbalizer_inputs_for_batch(
            batch=batch,
            start=start,
            model=model,
            tokenizer=tokenizer,
            target_lora_path=target_lora_path,
            config=config,
            device=device,
        )

        if verbalizer_lora_path is not None:
            model.set_adapter(verbalizer_lora_path)

        responses = run_evaluation(
            eval_data=verbalizer_inputs,
            model=model,
            tokenizer=tokenizer,
            submodule=injection_submodule,
            device=device,
            dtype=dtype,
            global_step=-1,
            lora_path=verbalizer_lora_path,
            eval_batch_size=config.eval_batch_size,
            steering_coefficient=config.steering_coefficient,
            generation_kwargs=config.verbalizer_generation_kwargs,
        )

        # Aggregate responses per (act_key, combo_index)
        agg: dict[tuple[str, int], dict[str, Any]] = {}
        for r in responses:
            meta = r.meta_info
            key = (meta["act_key"], int(meta["combo_index"]))
            if key not in agg:
                agg[key] = {
                    "verbalizer_prompt": meta["verbalizer_prompt"],
                    "ground_truth": meta["ground_truth"],
                    "num_tokens": int(meta["num_tokens"]),
                    "context_index_within_batch": int(meta["context_index_within_batch"]),
                    "responses": [],
                }
            agg[key]["responses"].append(r.api_response)

        for (act_key, combo_idx), bucket in agg.items():
            b_idx = bucket["context_index_within_batch"]
            record = VerbalizerResults(
                verbalizer_lora_path=verbalizer_lora_path,
                target_lora_path=target_lora_path,
                context_token_ids=token_id_lists[b_idx],
                act_key=act_key,
                verbalizer_prompt=bucket["verbalizer_prompt"],
                ground_truth=bucket["ground_truth"],
                num_tokens=bucket["num_tokens"],
                responses=bucket["responses"],
            )
            results.append(record)

        if verbalizer_lora_path is not None:
            verbalizer_lora_str = verbalizer_lora_path.split("/")[-1][:40]
        else:
            verbalizer_lora_str = "None"

        if target_lora_path is not None:
            target_lora_str = target_lora_path.split("/")[-1][:40]
        else:
            target_lora_str = "None"

        pbar.set_postfix({"inv": verbalizer_lora_str, "target": target_lora_str})
        pbar.update(len(batch))
    pbar.close()

    return results


def run_verbalizer_binary_score(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    verbalizer_prompt_infos: list[VerbalizerInputInfo],
    verbalizer_lora_path: str | None,
    target_lora_path: str | None,
    config: VerbalizerEvalConfig,
    device: torch.device,
    candidate_token_groups: dict[str, list[int]],
) -> list[BinaryFeatureResult]:
    """Run first-token yes/no-style scoring for verbalizer evaluation.

    Returns BinaryFeatureResult objects directly. Ground truth, act_key,
    verbalizer_prompt, etc. are available via result.meta_info.
    """

    dtype = torch.bfloat16
    injection_submodule = get_hf_submodule(model, config.injection_layer)

    pbar = tqdm(total=len(verbalizer_prompt_infos), desc="Verbalizer Binary Eval Progress", position=1)
    results: list[BinaryFeatureResult] = []

    for start in range(0, len(verbalizer_prompt_infos), config.eval_batch_size):
        batch = verbalizer_prompt_infos[start : start + config.eval_batch_size]
        token_id_lists, verbalizer_inputs = _prepare_verbalizer_inputs_for_batch(
            batch=batch,
            start=start,
            model=model,
            tokenizer=tokenizer,
            target_lora_path=target_lora_path,
            config=config,
            device=device,
        )

        if verbalizer_lora_path is not None:
            model.set_adapter(verbalizer_lora_path)

        score_results = run_binary_evaluation(
            eval_data=verbalizer_inputs,
            model=model,
            tokenizer=tokenizer,
            submodule=injection_submodule,
            device=device,
            dtype=dtype,
            lora_path=verbalizer_lora_path,
            eval_batch_size=config.eval_batch_size,
            steering_coefficient=config.steering_coefficient,
            candidate_token_groups=candidate_token_groups,
        )

        results.extend(score_results)

        verbalizer_lora_str = verbalizer_lora_path.split("/")[-1][:40] if verbalizer_lora_path is not None else "None"
        target_lora_str = target_lora_path.split("/")[-1][:40] if target_lora_path is not None else "None"
        pbar.set_postfix({"inv": verbalizer_lora_str, "target": target_lora_str})
        pbar.update(len(batch))
    pbar.close()

    return results


def assert_training_config_matches_verbalizer_eval_config(
    config: VerbalizerEvalConfig,
    training_config: SelfInterpTrainingConfig,
) -> None:
    assert training_config.model_name == config.model_name, (
        f"AO config model {training_config.model_name} != eval model {config.model_name}"
    )

    assert config.selected_layer_combination in training_config.layer_combinations, (
        f"selected_layer_combination {config.selected_layer_combination} must exist in "
        f"AO config layer_combinations {training_config.layer_combinations}"
    )
    selected_combo_idx = training_config.layer_combinations.index(config.selected_layer_combination)

    assert 0 <= selected_combo_idx < len(training_config.layer_combinations), (
        f"selected_combo_idx {selected_combo_idx} out of range for {len(training_config.layer_combinations)} combos"
    )

    layer_combo = training_config.layer_combinations[selected_combo_idx]

    if training_config.act_layer_combinations:
        act_layer_combo = training_config.act_layer_combinations[selected_combo_idx]
    else:
        act_layer_combo = [layer_percent_to_layer(config.model_name, lp) for lp in layer_combo]

    expected_act_layers = [layer_percent_to_layer(config.model_name, lp) for lp in layer_combo]
    assert act_layer_combo == expected_act_layers, (
        f"act layers {act_layer_combo} do not match expected {expected_act_layers} for layer combo {layer_combo}"
    )
    assert config.selected_layer_combination == layer_combo, (
        f"selected_layer_combination {config.selected_layer_combination} != AO combo {layer_combo}"
    )
    assert config.selected_act_layers == act_layer_combo, (
        f"selected_act_layers {config.selected_act_layers} != AO act layers {act_layer_combo}"
    )


def load_plain_adapter(
    model: AutoModelForCausalLM,
    lora_path: str,
) -> str:
    sanitized_lora_name = sanitize_lora_name(lora_path)

    if sanitized_lora_name not in model.peft_config:
        print(f"Loading LoRA: {lora_path}")
        model.load_adapter(
            lora_path,
            adapter_name=sanitized_lora_name,
            is_trainable=False,
            low_cpu_mem_usage=True,
        )

    return sanitized_lora_name


def load_oracle_adapter(
    model: AutoModelForCausalLM,
    lora_path: str,
) -> tuple[str, SelfInterpTrainingConfig]:
    sanitized_lora_name = load_plain_adapter(model, lora_path)
    training_config = read_training_config(lora_path)
    return sanitized_lora_name, training_config
