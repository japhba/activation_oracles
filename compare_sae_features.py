#!/usr/bin/env python3
"""
Script to find the most similar SAE feature using cosine similarity and compare their maximally activating sentences.

This script:
1. Loads a SAE and its maximally activating examples data
2. For a target feature, finds the most similar feature using cosine similarity of encoder vectors (W_enc)
3. Displays the maximally activating sentences for both features for comparison

Usage:
    python compare_sae_features.py --feature-idx 12345
    python compare_sae_features.py --feature-idx 12345 --num-sentences 10
    python compare_sae_features.py --feature-idx 12345 --top-k-similar 5
"""

import os
import torch
import argparse
import torch.nn.functional as F
from typing import List, Tuple
try:
    from transformers import AutoTokenizer
except ImportError:
    from transformers.models.auto.tokenization_auto import AutoTokenizer
from huggingface_hub import hf_hub_download
from interp_tools.config import SelfInterpTrainingConfig, get_sae_info
from interp_tools.saes.sae_loading_utils import load_gemma_2_sae
from interp_tools.introspect_utils import load_sae
# python compare_sae_features.py --feature-idx 0 --top-k-similar 5


def load_max_acts_data(
    model_name: str,
    sae_layer: int,
    sae_width: int,
    layer_percent: int,
    context_length: int = 32,
) -> dict[str, torch.Tensor]:
    """
    Load the max activating examples data.
    This is the same function as in view_original_sentences.py
    """
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
    """
    Convert token tensors to readable sentences.

    Args:
        tokens: Shape [num_examples, seq_len]
        tokenizer: Tokenizer to decode with
        skip_bos: Whether to skip the BOS token when decoding

    Returns:
        List of decoded sentences
    """
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
) -> Tuple[List[str], torch.Tensor]:
    """
    Get the top maximally activating sentences for a specific feature.

    Returns:
        Tuple of (sentences, activations)
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

    return sentences, feature_activations


def find_most_similar_features(
    sae, target_feature_idx: int, top_k: int = 1, exclude_self: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the most similar features to a target feature using cosine similarity of encoder vectors.
    
    Args:
        sae: The SAE object with W_enc weights
        target_feature_idx: Index of the target feature
        top_k: Number of most similar features to return
        exclude_self: Whether to exclude the target feature itself from results
    
    Returns:
        Tuple of (similarity_scores, feature_indices) for the top_k most similar features
    """
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


def print_feature_comparison(
    target_feature_idx: int,
    similar_feature_idx: int,
    similarity_score: float,
    target_sentences: List[str],
    target_activations: torch.Tensor,
    similar_sentences: List[str],
    similar_activations: torch.Tensor,
    show_activations: bool = True,
):
    """Print a comparison between target and most similar feature."""
    
    print(f"\n🎯 FEATURE COMPARISON")
    print("=" * 100)
    print(f"Target Feature: {target_feature_idx}")
    print(f"Most Similar Feature: {similar_feature_idx}")
    print(f"Cosine Similarity: {similarity_score:.6f}")
    print("=" * 100)
    
    max_sentences = max(len(target_sentences), len(similar_sentences))
    
    for i in range(max_sentences):
        print(f"\n📝 Example {i + 1}:")
        print("-" * 100)
        
        # Target feature
        if i < len(target_sentences):
            target_sentence = target_sentences[i]
            target_max_act = target_activations[i].max().item() if i < len(target_activations) else 0.0
            target_mean_act = target_activations[i].mean().item() if i < len(target_activations) else 0.0
            
            print(f"🎯 TARGET ({target_feature_idx}):")
            print(f"   Sentence: {target_sentence}")
            if show_activations:
                print(f"   Max Activation: {target_max_act:.4f}")
                print(f"   Mean Activation: {target_mean_act:.4f}")
        else:
            print(f"🎯 TARGET ({target_feature_idx}): No more examples")
        
        print()
        
        # Similar feature
        if i < len(similar_sentences):
            similar_sentence = similar_sentences[i]
            similar_max_act = similar_activations[i].max().item() if i < len(similar_activations) else 0.0
            similar_mean_act = similar_activations[i].mean().item() if i < len(similar_activations) else 0.0
            
            print(f"🔄 SIMILAR ({similar_feature_idx}) [sim: {similarity_score:.3f}]:")
            print(f"   Sentence: {similar_sentence}")
            if show_activations:
                print(f"   Max Activation: {similar_max_act:.4f}")
                print(f"   Mean Activation: {similar_mean_act:.4f}")
        else:
            print(f"🔄 SIMILAR ({similar_feature_idx}): No more examples")


def print_top_similar_features(
    target_feature_idx: int,
    similarities: torch.Tensor,
    indices: torch.Tensor,
    acts_data: dict[str, torch.Tensor],
    tokenizer: AutoTokenizer,
    num_preview_sentences: int = 2,
):
    """Print a summary of top similar features with preview sentences."""
    
    print(f"\n🔍 FEATURE {target_feature_idx} AND ITS MOST SIMILAR FEATURES")
    print("=" * 100)
    
    # First show the target feature itself
    print(f"\n🎯 TARGET FEATURE {target_feature_idx}:")
    try:
        target_sentences, target_acts = get_feature_max_activating_sentences(
            acts_data, tokenizer, target_feature_idx, num_preview_sentences
        )
        
        for j, (sentence, acts) in enumerate(zip(target_sentences, target_acts)):
            max_act = acts.max().item()
            truncated_sentence = sentence[:150] + "..." if len(sentence) > 150 else sentence
            print(f"   {j+1}. [{max_act:.3f}] {truncated_sentence}")
            
    except Exception as e:
        print(f"   Error getting sentences: {e}")
    
    print(f"\n🔄 MOST SIMILAR FEATURES:")
    print("-" * 100)
    
    for i, (sim_score, sim_idx) in enumerate(zip(similarities, indices)):
        sim_idx_int = sim_idx.item()
        sim_score_float = sim_score.item()
        
        print(f"\n#{i+1}. Feature {sim_idx_int} (similarity: {sim_score_float:.6f})")
        
        # Get preview sentences
        try:
            preview_sentences, preview_acts = get_feature_max_activating_sentences(
                acts_data, tokenizer, int(sim_idx_int), num_preview_sentences
            )
            
            for j, (sentence, acts) in enumerate(zip(preview_sentences, preview_acts)):
                max_act = acts.max().item()
                truncated_sentence = sentence[:150] + "..." if len(sentence) > 150 else sentence
                print(f"   {j+1}. [{max_act:.3f}] {truncated_sentence}")
                
        except Exception as e:
            print(f"   Error getting sentences: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare SAE features using cosine similarity of encoder vectors"
    )
    parser.add_argument(
        "--feature-idx", 
        type=int, 
        required=True,
        help="Target feature index to find similar features for"
    )
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=5,
        help="Number of sentences to show per feature",
    )
    parser.add_argument(
        "--top-k-similar",
        type=int,
        default=1,
        help="Number of most similar features to find",
    )
    parser.add_argument(
        "--show-all-similar",
        action="store_true",
        help="Show preview of all top-k similar features instead of detailed comparison",
    )
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="google/gemma-2-9b-it", 
        help="Model name"
    )
    parser.add_argument(
        "--sae-repo-id",
        type=str,
        default="google/gemma-scope-9b-it-res",
        help="SAE repository",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=32,
        help="Context length used for max acts",
    )

    args = parser.parse_args()

    # Setup configuration
    cfg = SelfInterpTrainingConfig()
    cfg.model_name = args.model_name
    cfg.sae_repo_id = args.sae_repo_id

    # Get SAE info
    cfg.sae_width, cfg.sae_layer, cfg.sae_layer_percent, cfg.sae_filename = (
        get_sae_info(cfg.sae_repo_id)
    )

    print("🔧 Configuration:")
    print(f"   Model: {cfg.model_name}")
    print(f"   SAE: {cfg.sae_repo_id}")
    print(f"   Layer: {cfg.sae_layer}")
    print(f"   Width: {cfg.sae_width}")

    # Load tokenizer
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Load SAE
    print("🧠 Loading SAE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    sae = load_sae(cfg, device, dtype)
    
    print(f"   SAE encoder shape: {sae.W_enc.shape}")
    print(f"   SAE decoder shape: {sae.W_dec.shape}")

    # Load max acts data
    print("📊 Loading max acts data...")
    acts_data = load_max_acts_data(
        cfg.model_name,
        cfg.sae_layer,
        cfg.sae_width,
        cfg.sae_layer_percent,
        args.context_length,
    )

    # Validate feature index
    max_feature_idx = acts_data["max_tokens"].shape[0] - 1
    if args.feature_idx > max_feature_idx:
        raise ValueError(f"Feature {args.feature_idx} not found. Max feature index: {max_feature_idx}")

    # Find most similar features
    print(f"🔍 Finding {args.top_k_similar} most similar features to feature {args.feature_idx}...")
    similarities, similar_indices = find_most_similar_features(
        sae, args.feature_idx, top_k=args.top_k_similar
    )

    if args.show_all_similar or args.top_k_similar > 1:
        # Show preview of all similar features with examples
        print_top_similar_features(
            args.feature_idx, similarities, similar_indices, acts_data, tokenizer, args.num_sentences
        )
    else:
        # Detailed comparison with the most similar feature
        most_similar_idx = similar_indices[0].item()
        most_similar_score = similarities[0].item()

        # Get sentences for target feature
        print(f"📝 Getting sentences for target feature {args.feature_idx}...")
        target_sentences, target_activations = get_feature_max_activating_sentences(
            acts_data, tokenizer, args.feature_idx, args.num_sentences
        )

        # Get sentences for most similar feature
        print(f"📝 Getting sentences for similar feature {most_similar_idx}...")
        similar_sentences, similar_activations = get_feature_max_activating_sentences(
            acts_data, tokenizer, int(most_similar_idx), args.num_sentences
        )

        # Print comparison
        print_feature_comparison(
            args.feature_idx,
            int(most_similar_idx),
            most_similar_score,
            target_sentences,
            target_activations,
            similar_sentences,
            similar_activations,
        )


if __name__ == "__main__":
    main()
