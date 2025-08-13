#!/usr/bin/env python3
"""
Script to view the original maximally activating sentences for each SAE feature.

This loads the same max_acts data that api_interp.py uses to generate prompts,
but focuses on just viewing the original sentences that activate each feature most strongly.

Usage:
    python view_original_sentences.py
    python view_original_sentences.py --feature-idx 12345
    python view_original_sentences.py --num-sentences 10
    python view_original_sentences.py --save-to-file sentences_feature_12345.txt
"""

import os
import torch
import argparse
from typing import List, Tuple
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from interp_tools.config import SelfInterpTrainingConfig, get_sae_info


def load_max_acts_data(
    model_name: str,
    sae_layer: int,
    sae_width: int,
    layer_percent: int,
    context_length: int = 32,
) -> dict[str, torch.Tensor]:
    """
    Load the max activating examples data.
    This is the same function as in api_interp.py
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


def print_feature_sentences(
    feature_idx: int,
    sentences: List[str],
    activations: torch.Tensor,
    show_activations: bool = True,
):
    """Print the sentences and their activations for a feature."""

    print(f"\n🎯 Feature {feature_idx} - Top Maximally Activating Sentences")
    print("=" * 80)

    for i, (sentence, acts) in enumerate(zip(sentences, activations)):
        max_activation = acts.max().item()
        mean_activation = acts.mean().item()

        print(f"\n📝 Example {i + 1}:")
        print(f"   Sentence: {sentence}")

        if show_activations:
            print(f"   Max Activation: {max_activation:.4f}")
            print(f"   Mean Activation: {mean_activation:.4f}")


def explore_multiple_features(
    acts_data: dict[str, torch.Tensor],
    tokenizer: AutoTokenizer,
    feature_indices: List[int],
    num_sentences: int = 3,
):
    """Explore multiple features at once."""

    print(f"\n🔍 Exploring {len(feature_indices)} features")
    print("=" * 80)

    for feature_idx in feature_indices:
        try:
            sentences, activations = get_feature_max_activating_sentences(
                acts_data, tokenizer, feature_idx, num_sentences
            )

            print(f"\n🎯 Feature {feature_idx}:")
            for i, (sentence, acts) in enumerate(zip(sentences, activations)):
                max_act = acts.max().item()
                print(
                    f"  {i + 1}. [{max_act:.3f}] {sentence[:100]}{'...' if len(sentence) > 100 else ''}"
                )

        except Exception as e:
            print(f"❌ Error with feature {feature_idx}: {e}")


def save_sentences_to_file(
    feature_idx: int, sentences: List[str], activations: torch.Tensor, filename: str
):
    """Save sentences and activations to a text file."""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Feature {feature_idx} - Maximally Activating Sentences\n")
        f.write("=" * 60 + "\n\n")

        for i, (sentence, acts) in enumerate(zip(sentences, activations)):
            max_activation = acts.max().item()
            mean_activation = acts.mean().item()

            f.write(f"Example {i + 1}:\n")
            f.write(f"Max Activation: {max_activation:.4f}\n")
            f.write(f"Mean Activation: {mean_activation:.4f}\n")
            f.write(f"Sentence: {sentence}\n\n")

    print(f"💾 Saved to: {filename}")


def get_feature_info(acts_data: dict[str, torch.Tensor]):
    """Get basic information about the max acts data."""

    max_tokens = acts_data["max_tokens"]
    max_acts = acts_data["max_acts"]

    print("📊 Max Acts Data Info:")
    print(f"   Number of features: {max_tokens.shape[0]}")
    print(f"   Examples per feature: {max_tokens.shape[1]}")
    print(f"   Sequence length: {max_tokens.shape[2]}")
    print(f"   Data shape: {max_tokens.shape}")

    # Show activation statistics
    all_max_acts = max_acts.max(dim=2)[0]  # Max activation per example
    global_max = all_max_acts.max().item()
    global_mean = all_max_acts.mean().item()

    print(f"   Global max activation: {global_max:.4f}")
    print(f"   Global mean max activation: {global_mean:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="View original maximally activating sentences for SAE features"
    )
    parser.add_argument("--feature-idx", type=int, help="Specific feature to view")
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=5,
        help="Number of sentences to show per feature",
    )
    parser.add_argument("--save-to-file", type=str, help="Save sentences to this file")
    parser.add_argument(
        "--model-name", type=str, default="google/gemma-2-9b-it", help="Model name"
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
    parser.add_argument("--explore-random", type=int, help="Explore N random features")
    parser.add_argument(
        "--feature-range", type=str, help="Explore features in range (e.g., '100-200')"
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

    # Load max acts data
    acts_data = load_max_acts_data(
        cfg.model_name,
        cfg.sae_layer,
        cfg.sae_width,
        cfg.sae_layer_percent,
        args.context_length,
    )

    # Show data info
    get_feature_info(acts_data)

    # Handle different exploration modes
    if args.feature_idx is not None:
        # View specific feature
        sentences, activations = get_feature_max_activating_sentences(
            acts_data, tokenizer, args.feature_idx, args.num_sentences
        )

        print_feature_sentences(args.feature_idx, sentences, activations)

        # Save to file if requested
        if args.save_to_file:
            save_sentences_to_file(
                args.feature_idx, sentences, activations, args.save_to_file
            )

    elif args.explore_random:
        # Explore random features
        import random

        max_feature_idx = acts_data["max_tokens"].shape[0] - 1
        random_features = random.sample(range(max_feature_idx + 1), args.explore_random)

        explore_multiple_features(
            acts_data, tokenizer, random_features, args.num_sentences
        )

    elif args.feature_range:
        # Explore feature range
        try:
            start, end = map(int, args.feature_range.split("-"))
            feature_indices = list(range(start, end + 1))

            explore_multiple_features(
                acts_data, tokenizer, feature_indices, args.num_sentences
            )
        except ValueError:
            print("❌ Invalid range format. Use 'start-end' (e.g., '100-200')")

    else:
        # Default: show some examples
        print("\n💡 Usage examples:")
        print(f"   View feature 0: python {os.path.basename(__file__)} --feature-idx 0")
        print(
            f"   View 10 random features: python {os.path.basename(__file__)} --explore-random 10"
        )
        print(
            f"   View features 100-110: python {os.path.basename(__file__)} --feature-range 100-110"
        )
        print(
            f"   Save feature 0 to file: python {os.path.basename(__file__)} --feature-idx 0 --save-to-file feature_0.txt"
        )

        # Show a few random examples
        import random

        max_feature_idx = acts_data["max_tokens"].shape[0] - 1
        random_features = random.sample(range(max_feature_idx + 1), 3)

        print("\n🎲 Here are 3 random features as examples:")
        explore_multiple_features(acts_data, tokenizer, random_features, 50)


if __name__ == "__main__":
    main()
