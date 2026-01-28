import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import einops
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from nl_probes.autointerp_detection_eval.detection_basemodels import (
    SAEV2,
    SAEActivationsV2,
    SAEInfo,
    SentenceInfoV2,
    TokenActivationV2,
)
from nl_probes.sae import get_sae_info, load_max_acts_data, load_sae
from nl_probes.utils.activation_utils import collect_activations, get_hf_submodule
from nl_probes.utils.common import list_decode, load_model, load_tokenizer


@dataclass(kw_only=True)
class SimilarFeature:
    """Represents a feature similar to a target feature."""

    feature_idx: int
    similarity_score: float


def find_most_similar_features(
    sae, target_feature_idx: int, top_k: int = 1, exclude_self: bool = True
) -> list[SimilarFeature]:
    """Find the most similar features to a target feature using cosine similarity of encoder vectors."""
    # Get encoder weights - shape: [d_in, d_sae]
    W_enc = sae.W_enc.data

    # Get the target feature vector - shape: [d_in]
    target_vector = W_enc[:, target_feature_idx]

    # Compute cosine similarity with all other features
    # Normalize the target vector
    target_normalized = F.normalize(target_vector.unsqueeze(0), dim=1)

    # Normalize all encoder vectors
    all_vectors_normalized = F.normalize(W_enc.T, dim=1)  # Shape: [d_sae, d_in]

    # Compute cosine similarities - shape: [d_sae]
    similarities = torch.mm(all_vectors_normalized, target_normalized.T).squeeze()

    if exclude_self:
        # Set similarity to target feature itself to -inf so it's not selected
        similarities[target_feature_idx] = float("-inf")

    # Get top-k most similar features
    top_similarities, top_indices = torch.topk(similarities, k=top_k, largest=True)

    # Create SimilarFeature objects
    similar_features = []
    for sim_score, feature_idx in zip(top_similarities, top_indices, strict=False):
        similar_features.append(
            SimilarFeature(
                feature_idx=int(feature_idx.item()),
                similarity_score=float(sim_score.item()),
            )
        )

    return similar_features


def compute_sae_activations_for_sentences(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae: object,
    submodule: torch.nn.Module,
    tokens_BL: torch.Tensor,
    target_feature_idx: int,
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Compute SAE activations for a list of sentences and return SentenceInfo objects.
    """

    all_acts_BL = []

    # Process sentences in batches
    for i in range(0, tokens_BL.shape[0], batch_size):
        batch_tokens_BL = tokens_BL[i : i + batch_size]
        # attn_mask is all ones because we used sequence packing when constructing the batch
        attn_mask_BL = torch.ones_like(batch_tokens_BL)

        tokenized = {
            "input_ids": batch_tokens_BL,
            "attention_mask": attn_mask_BL,
        }

        with torch.no_grad():
            # Get model activations at the SAE layer for the whole batch
            layer_acts_BLD = collect_activations(model, submodule, tokenized)

            # Encode through SAE
            encoded_acts_BLF = sae.encode(layer_acts_BLD)  # type: ignore

            norms_BL = torch.norm(layer_acts_BLD, dim=-1)
            median_norm = norms_BL.median()
            norm_mask_BL = norms_BL < median_norm * 10

            norm_mask_BL *= attn_mask_BL.bool()

            if tokenizer.bos_token_id is not None:
                bos_mask_BL = batch_tokens_BL != tokenizer.bos_token_id
                norm_mask_BL *= bos_mask_BL

            encoded_acts_BLF *= norm_mask_BL[:, :, None]

            encoded_acts_BL = encoded_acts_BLF[:, :, target_feature_idx]

            all_acts_BL.append(encoded_acts_BL)

    all_acts_BL = torch.cat(all_acts_BL, dim=0)

    return all_acts_BL


def main(
    target_features: Sequence[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    target_sentences: int = 20,
    top_k_similar_features: int = 10,
    negative_sentences: int = 8,  # we don't need so many
    output: str = "hard_negatives_results.jsonl",
    model_name: str = "google/gemma-2-9b-it",
    sae_repo_id: str = "google/gemma-scope-9b-it-res",
    context_length: int = 32,
    hard_negative_threshold: float = 0.05,
    batch_size: int = 20,
    sae_layer_percent: int = 25,
    verbose: bool = False,
):
    # check if output file exists
    if os.path.exists(output):
        print(f"🔍 Output file {output} already exists. Not going to overwrite it.")
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Get SAE info
    sae_info = get_sae_info(sae_repo_id, sae_layer_percent)
    sae_width = sae_info.sae_width
    sae_layer = sae_info.sae_layer
    sae_layer_percent = sae_info.sae_layer_percent
    sae_filename = sae_info.sae_filename

    print("🔧 Configuration:")
    print(f"   Model: {model_name}")
    print(f"   SAE: {sae_repo_id}")
    print(f"   Layer: {sae_layer}")
    print(f"   Width: {sae_width}")
    print(f"   Target Features: {target_features}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Output: {output}")

    # Load max acts data
    print("📊 Loading max acts data...")
    acts_data = load_max_acts_data(
        model_name,
        sae_layer,
        sae_width,
        sae_layer_percent,
        context_length,
    )

    # Validate feature indices
    max_feature_idx = acts_data["max_tokens"].shape[0] - 1
    for feature_idx in target_features:
        if feature_idx > max_feature_idx:
            raise ValueError(f"Feature {feature_idx} not found. Max feature index: {max_feature_idx}")

    # Load model, tokenizer, and SAE
    print("🚀 Loading model and SAE...")
    model = load_model(model_name, dtype)
    tokenizer = load_tokenizer(model_name)
    sae = load_sae(sae_repo_id, sae_filename, sae_layer, model_name, device, dtype)
    submodule = get_hf_submodule(model, sae_layer)  # type: ignore
    # how many features in sae?
    print(f"🔍 Number of features in SAE: {len(sae.W_dec)}")  # type: ignore

    # Process each feature index
    special_tokens = [tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id]
    special_tokens = [tokenizer.decode(token_id, skip_special_tokens=False) for token_id in special_tokens]
    special_tokens = set(special_tokens)

    # open file to append
    with open(output, "a") as f:
        for feature_idx in tqdm(target_features, desc="Processing features"):
            similar_features = find_most_similar_features(sae, feature_idx, top_k=top_k_similar_features)

            pos_tokens_BL = acts_data["max_tokens"][feature_idx, :target_sentences]

            pos_acts_BL = acts_data["max_acts"][feature_idx, :target_sentences]

            max_target_act = pos_acts_BL.max()

            all_similar_tokens_BL = []

            similar_feature_indices = [similar_feature.feature_idx for similar_feature in similar_features]

            all_similar_tokens_KBL = acts_data["max_tokens"][similar_feature_indices, :negative_sentences]

            K, B, L = all_similar_tokens_KBL.shape

            all_similar_tokens_BL = einops.rearrange(all_similar_tokens_KBL, "K B L -> (K B) L")

            all_similar_acts_BL = compute_sae_activations_for_sentences(
                model,
                tokenizer,
                sae,
                submodule,
                all_similar_tokens_BL,
                feature_idx,
                batch_size,
            )

            all_similar_acts_KBL = einops.rearrange(all_similar_acts_BL, "(K B) L -> K B L", K=K, B=B)

            max_similar_acts_KB = all_similar_acts_KBL.max(dim=-1).values
            hard_negatives_mask_KB = max_similar_acts_KB > (hard_negative_threshold * max_target_act)
            all_similar_acts_KBL[hard_negatives_mask_KB] = -1

            all_similar_acts_KBL = all_similar_acts_KBL.cpu()
            all_similar_tokens_KBL = all_similar_tokens_KBL.cpu()
            pos_tokens_BL = pos_tokens_BL.cpu()
            pos_acts_BL = pos_acts_BL.cpu()
            max_target_act = max_target_act.cpu()

            if verbose:
                print(f"\n🎯 Processing feature {feature_idx}...")
                # Find most similar features
                print(f"🔍 Finding {top_k_similar_features} most similar features to feature {feature_idx}...")

                # Get sentences for target feature
                print(f"📝 Getting sentences for target feature {feature_idx}...")

                # Compute actual SAE activations for target feature sentences
                print("🧮 Computing SAE activations for target feature sentences...")

                # Collect all sentences from similar features first
                print(f"📝 Collecting sentences from {len(similar_features)} similar features...")

                # Compute target feature activations on ALL similar feature sentences in batches
                print(
                    f"🧮 Computing SAE activations for {all_similar_tokens_BL.shape[0]} sentences from similar features..."
                )
                print(f"Found {hard_negatives_mask_KB.sum().item()} hard negatives")

            decoded_pos_tokens = list_decode(pos_tokens_BL, tokenizer)

            pos_sentence_infos = []

            for i, pos_tokens in zip(range(len(decoded_pos_tokens)), decoded_pos_tokens):
                token_activations = []
                tokens: list[str] = []
                max_act = 0
                acts_L = pos_acts_BL[i, :].tolist()
                non_special_tokens = [token for token in pos_tokens if token not in special_tokens]
                for j, token in enumerate(non_special_tokens):
                    if token in special_tokens:
                        continue
                    act = acts_L[j]
                    max_act = max(max_act, act)
                    # only save if act > 0 for space reasons
                    if act > 0.0:
                        token_activations.append(TokenActivationV2.model_construct(s=token, act=act, pos=j))
                    # save all tokens
                    tokens.append(token)
                pos_sentence_infos.append(
                    SentenceInfoV2.model_construct(max_act=max_act, tokens=tokens, act_tokens=token_activations)
                )

            pos_sae_activations = SAEActivationsV2(sae_id=feature_idx, sentences=pos_sentence_infos)

            hard_negatives = []

            for k_idx, similar_feature_idx in enumerate(similar_feature_indices):
                decoded_hard_negative_tokens = list_decode(all_similar_tokens_KBL[k_idx], tokenizer)
                all_similar_acts_BL = all_similar_acts_KBL[k_idx]
                hard_negative_sentence_infos = []

                for i, hard_negative_tokens in enumerate(decoded_hard_negative_tokens):
                    # -1 is not possible for SAE acts as the acts are post relu
                    if all_similar_acts_BL[i, 0] == -1:
                        continue
                    token_activations = []
                    tokens: list[str] = []
                    max_act = 0
                    acts_L = all_similar_acts_BL[i, :].tolist()
                    non_special_tokens = [token for token in hard_negative_tokens if token not in special_tokens]
                    for j, token in enumerate(non_special_tokens):
                        if token in special_tokens:
                            continue
                        act = acts_L[j]
                        max_act = max(max_act, act)
                        # only save if act > 0 for space reasons
                        if act > 0.0:
                            token_activations.append(TokenActivationV2.model_construct(s=token, act=act, pos=j))
                        tokens.append(token)
                    hard_negative_sentence_infos.append(
                        SentenceInfoV2.model_construct(
                            max_act=max_act,
                            tokens=tokens,
                            act_tokens=token_activations,
                        )
                    )

                if verbose:
                    print(
                        f"Found {len(hard_negative_sentence_infos)} hard negatives for feature {similar_feature_idx} with similarity {similar_features[k_idx].similarity_score:.4f}"
                    )

                hard_negative_sae_activations = SAEActivationsV2.model_construct(
                    sae_id=similar_feature_idx, sentences=hard_negative_sentence_infos
                )
                hard_negatives.append(hard_negative_sae_activations)

            sae_result = SAEV2(
                sae_id=feature_idx, sae_info=sae_info, activations=pos_sae_activations, hard_negatives=hard_negatives
            )
            f.write(sae_result.model_dump_json(exclude_none=True) + "\n")

    # Write all results to JSONL
    print(f"\n💾 Writing results to {output}...")

    print("✅ Analysis complete!")
    print(f"   Features processed: {target_features}")
    print(f"   Results saved to: {output}")


if __name__ == "__main__":
    # Example usage - customize the feature_idxs and other parameters as needed
    # target_features = list(range(0, 100_000))
    # to_100k = list(range(0, 100_000))
    # 100k to 100_200
    # target_features = list(range(0, 200))
    # min_idx = 20_000
    # max_idx = 20_000
    # max_idx = 30_000
    min_idx = 50_000
    max_idx = 50_600
    target_features = list(range(min_idx, max_idx))

    data_folder = "sae_data"
    os.makedirs(data_folder, exist_ok=True)

    for sae_layer_percent in [25, 50, 75]:
        main(
            # model_name="google/gemma-2-9b-it",
            # sae_repo_id="google/gemma-scope-9b-it-res",
            model_name="Qwen/Qwen3-8B",
            sae_repo_id="anon-org/qwen3-8b-saes",
            target_features=target_features,
            top_k_similar_features=34,
            batch_size=1024,
            target_sentences=32,
            output=f"{data_folder}/qwen_hard_negatives_{min_idx}_{max_idx}_layer_percent_{sae_layer_percent}.jsonl",
            sae_layer_percent=sae_layer_percent,
            verbose=False,
        )
