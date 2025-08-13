#!/usr/bin/env python3
"""
Script to find similar SAE features and verify hard negatives using the Gemma 9B model.

This script:
1. Loads a SAE and finds the most similar features to a target feature
2. Loads the Gemma 9B model and computes actual SAE activations on sentences
3. Identifies sentences from similar features that don't activate for the target feature (hard negatives)
4. Outputs results to JSONL format

Usage as a module:
    from compare_and_verify_hard_negatives import main
    main(feature_idx=0, num_sentences=5, top_k_similar=3)

Or modify the call at the bottom of this file and run directly:
    python compare_and_verify_hard_negatives.py
"""

import os
import torch
import json
import torch.nn.functional as F
from typing import List, Tuple, Optional
from pydantic import BaseModel
from slist import Slist

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM

from huggingface_hub import hf_hub_download
from interp_tools.config import SelfInterpTrainingConfig, get_sae_info
from interp_tools.introspect_utils import load_sae
import interp_tools.model_utils as model_utils


# Pydantic schema classes for JSONL output
class TokenActivation(BaseModel):
    as_str: str
    activation: float
    token_id: int

    def to_prompt_str(self) -> str:
        return f"{self.as_str} ({self.activation:.2f})"


class SentenceInfo(BaseModel):
    max_activation: float
    tokens: List[TokenActivation]
    as_str: str

    def as_activation_vector(self) -> str:
        activation_vector = Slist(self.tokens).map(lambda x: x.to_prompt_str())
        return f"{activation_vector}"


class SAEActivations(BaseModel):
    sae_id: int
    sentences: List[SentenceInfo]


class SAE(BaseModel):
    sae_id: int
    activations: SAEActivations
    # Sentences that do not activate for the given sae_id. But come from a similar SAE
    # Here the sae_id correspond to different similar SAEs.
    # The activations are the activations w.r.t this SAE. And should be low.
    hard_negatives: List[SAEActivations]


def load_max_acts_data(
    model_name: str,
    sae_layer: int,
    sae_width: int,
    layer_percent: int,
    context_length: int = 32,
) -> dict[str, torch.Tensor]:
    """Load the max activating examples data."""
    acts_dir = "max_acts"

    # Construct filename
    acts_filename = f"acts_{model_name}_layer_{sae_layer}_trainer_{sae_width}_layer_percent_{layer_percent}_context_length_{context_length}.pt".replace(
        "/", "_"
    )

    acts_path = os.path.join(acts_dir, acts_filename)

    # Download if not exists
    if not os.path.exists(acts_path):
        print(f"📥 Downloading max acts data: {acts_filename}")
        try:
            path_to_config = hf_hub_download(
                repo_id="adamkarvonen/sae_max_acts",
                filename=acts_filename,
                force_download=False,
                local_dir=acts_dir,
                repo_type="dataset",
            )
            print(f"✅ Downloaded to: {acts_path}")
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            raise

    print(f"📂 Loading max acts data from: {acts_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acts_data = torch.load(acts_path, map_location=device)

    return acts_data


def decode_tokens_to_sentences(
    tokens: torch.Tensor, tokenizer: AutoTokenizer, skip_bos: bool = True
) -> List[str]:
    """Convert token tensors to readable sentences."""
    sentences = []

    for i in range(tokens.shape[0]):
        token_sequence = tokens[i]

        # Skip BOS token if requested
        if skip_bos and len(token_sequence) > 0:
            token_sequence = token_sequence[1:]

        # Decode to string
        sentence = tokenizer.decode(token_sequence.tolist(), skip_special_tokens=True).strip()
        sentences.append(sentence)

    return sentences


def get_feature_max_activating_sentences(
    acts_data: dict[str, torch.Tensor],
    tokenizer: AutoTokenizer,
    feature_idx: int,
    num_sentences: int = 5,
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """
    Get the top maximally activating sentences for a specific feature.

    Returns:
        Tuple of (sentences, activations, tokens)
    """
    if feature_idx >= acts_data["max_tokens"].shape[0]:
        raise ValueError(
            f"Feature {feature_idx} not found. Max feature index: {acts_data['max_tokens'].shape[0] - 1}"
        )

    # Get tokens and activations for this feature
    feature_tokens = acts_data["max_tokens"][
        feature_idx, :num_sentences
    ]  # Shape: [num_sentences, seq_len]
    feature_activations = acts_data["max_acts"][
        feature_idx, :num_sentences
    ]  # Shape: [num_sentences, seq_len]

    # Decode tokens to sentences
    sentences = decode_tokens_to_sentences(feature_tokens, tokenizer)

    return sentences, feature_activations, feature_tokens


def find_most_similar_features(
    sae, target_feature_idx: int, top_k: int = 1, exclude_self: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find the most similar features to a target feature using cosine similarity of encoder vectors."""
    # Get encoder weights - shape: [d_in, d_sae]
    W_enc = sae.W_enc.data  # Remove gradient tracking
    
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
        similarities[target_feature_idx] = float('-inf')
    
    # Get top-k most similar features
    top_similarities, top_indices = torch.topk(similarities, k=top_k, largest=True)
    
    return top_similarities, top_indices


def load_model_and_sae(cfg: SelfInterpTrainingConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer, object, torch.nn.Module]:
    """Load the Gemma 9B model, tokenizer, SAE, and submodule."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load tokenizer
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Load model
    print("🧠 Loading Gemma 9B model...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Load SAE
    print("🔧 Loading SAE...")
    sae = load_sae(cfg, device, dtype)
    
    # Get submodule for activation collection
    submodule = model_utils.get_submodule(model, cfg.sae_layer)
    
    return model, tokenizer, sae, submodule


def compute_sae_activations_for_sentences(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae: object,
    submodule: torch.nn.Module,
    sentences: List[str],
    target_feature_idx: int,
) -> List[SentenceInfo]:
    """
    Compute SAE activations for a list of sentences and return SentenceInfo objects.
    """
    sentence_infos = []
    
    for sentence in sentences:
        # Tokenize sentence
        tokenized = tokenizer(
            sentence,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=512,
        ).to(model.device)
        
        with torch.no_grad():
            # Get model activations at the SAE layer
            layer_acts_BLD = model_utils.collect_activations(model, submodule, tokenized)
            
            # Encode through SAE
            encoded_acts_BLF = sae.encode(layer_acts_BLD)
            
            # Get activations for the target feature - shape: [batch, seq_len]
            feature_acts = encoded_acts_BLF[0, :, target_feature_idx]  # [seq_len]
            
            # Convert to token activations
            token_activations = []
            token_ids = tokenized["input_ids"][0]  # [seq_len]
            
            for i, (token_id, activation) in enumerate(zip(token_ids, feature_acts)):
                token_str = tokenizer.decode([token_id.item()], skip_special_tokens=True)
                token_activations.append(TokenActivation(
                    as_str=token_str,
                    activation=activation.item(),
                    token_id=token_id.item()
                ))
            
            # Create SentenceInfo
            max_activation = feature_acts.max().item()
            sentence_info = SentenceInfo(
                max_activation=max_activation,
                tokens=token_activations,
                as_str=sentence
            )
            
            sentence_infos.append(sentence_info)
    
    return sentence_infos


def identify_hard_negatives(
    similar_sentence_infos: List[SentenceInfo],
    threshold: float = 0.5,
) -> List[SentenceInfo]:
    """
    Identify sentences that have low activation for the target feature.
    These are hard negatives - sentences from similar features that don't activate the target.
    """
    hard_negatives = []
    
    for sentence_info in similar_sentence_infos:
        if sentence_info.max_activation < threshold:
            hard_negatives.append(sentence_info)
    
    return hard_negatives


def main(
    feature_idx: int,
    num_sentences: int = 5,
    top_k_similar: int = 3,
    output: str = "hard_negatives_results.jsonl",
    model_name: str = "google/gemma-2-9b-it",
    sae_repo_id: str = "google/gemma-scope-9b-it-res",
    context_length: int = 32,
    hard_negative_threshold: float = 0.5,
):

    # Setup configuration
    cfg = SelfInterpTrainingConfig()
    cfg.model_name = model_name
    cfg.sae_repo_id = sae_repo_id

    # Get SAE info
    cfg.sae_width, cfg.sae_layer, cfg.sae_layer_percent, cfg.sae_filename = (
        get_sae_info(cfg.sae_repo_id)
    )

    print("🔧 Configuration:")
    print(f"   Model: {cfg.model_name}")
    print(f"   SAE: {cfg.sae_repo_id}")
    print(f"   Layer: {cfg.sae_layer}")
    print(f"   Width: {cfg.sae_width}")
    print(f"   Target Feature: {feature_idx}")
    print(f"   Output: {output}")

    # Load max acts data
    print("📊 Loading max acts data...")
    acts_data = load_max_acts_data(
        cfg.model_name,
        cfg.sae_layer,
        cfg.sae_width,
        cfg.sae_layer_percent,
        context_length,
    )

    # Validate feature index
    max_feature_idx = acts_data["max_tokens"].shape[0] - 1
    if feature_idx > max_feature_idx:
        raise ValueError(f"Feature {feature_idx} not found. Max feature index: {max_feature_idx}")

    # Load model, tokenizer, and SAE
    print("🚀 Loading model and SAE...")
    model, tokenizer, sae, submodule = load_model_and_sae(cfg)

    # Find most similar features
    print(f"🔍 Finding {top_k_similar} most similar features to feature {feature_idx}...")
    similarities, similar_indices = find_most_similar_features(
        sae, feature_idx, top_k=top_k_similar
    )

    # Get sentences for target feature
    print(f"📝 Getting sentences for target feature {feature_idx}...")
    target_sentences, _, _ = get_feature_max_activating_sentences(
        acts_data, tokenizer, feature_idx, num_sentences
    )

    # Compute actual SAE activations for target feature sentences
    print("🧮 Computing SAE activations for target feature sentences...")
    target_sentence_infos = compute_sae_activations_for_sentences(
        model, tokenizer, sae, submodule, target_sentences, feature_idx
    )

    # Analyze similar features and collect hard negatives
    hard_negatives_list = []
    
    for i, (sim_score, sim_idx) in enumerate(zip(similarities, similar_indices)):
        sim_idx_int = sim_idx.item()
        print(f"📝 Analyzing similar feature {sim_idx_int} (similarity: {sim_score:.4f})...")
        
        # Get sentences for this similar feature
        similar_sentences, _, _ = get_feature_max_activating_sentences(
            acts_data, tokenizer, int(sim_idx_int), num_sentences
        )
        
        # Compute target feature activations on similar feature's sentences
        similar_sentence_infos = compute_sae_activations_for_sentences(
            model, tokenizer, sae, submodule, similar_sentences, feature_idx
        )
        
        # Identify hard negatives
        hard_negatives = identify_hard_negatives(
            similar_sentence_infos, hard_negative_threshold
        )
        
        if hard_negatives:
            hard_negatives_sae = SAEActivations(
                sae_id=int(sim_idx_int),
                sentences=hard_negatives
            )
            hard_negatives_list.append(hard_negatives_sae)
            print(f"   Found {len(hard_negatives)} hard negatives from feature {sim_idx_int}")
        else:
            print(f"   No hard negatives found from feature {sim_idx_int}")

    # Create final SAE object
    target_activations = SAEActivations(
        sae_id=feature_idx,
        sentences=target_sentence_infos
    )
    
    sae_result = SAE(
        sae_id=feature_idx,
        activations=target_activations,
        hard_negatives=hard_negatives_list
    )

    # Write to JSONL
    print(f"💾 Writing results to {output}...")
    with open(output, "w") as f:
        f.write(sae_result.model_dump_json() + "\n")
    
    print("✅ Analysis complete!")
    print(f"   Target feature: {feature_idx}")
    print(f"   Target sentences analyzed: {len(target_sentence_infos)}")
    print(f"   Similar features analyzed: {len(similarities)}")
    print(f"   Hard negative groups found: {len(hard_negatives_list)}")
    print(f"   Results saved to: {output}")


if __name__ == "__main__":
    # Example usage - customize the feature_idx and other parameters as needed
    main(feature_idx=0)
