import json
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from nl_probes.autointerp_detection_eval.detection_basemodels import SAEInfo
from nl_probes.utils.activation_utils import collect_activations
from nl_probes.utils.common import get_bos_eos_pad_mask


def get_sae_info(sae_repo_id: str, sae_layer_percent: int = 25, sae_width: int | None = None) -> SAEInfo:
    if sae_repo_id == "google/gemma-scope-9b-it-res":
        num_layers = 42
        assert sae_layer_percent == 25
        sae_layer = 9

        # Gemma scope IT saes: https://huggingface.co/google/gemma-scope-9b-it-res/tree/main
        assert sae_layer in [9, 20, 31]

        # Note: For gemma_scope saes you need to specify the L0 if you use different layers / widths

        if sae_width is None:
            sae_width = 131

        if sae_width == 16:
            sae_filename = f"layer_{sae_layer}/width_16k/average_l0_88/params.npz"
        elif sae_width == 131:
            sae_filename = f"layer_{sae_layer}/width_131k/average_l0_121/params.npz"
        else:
            raise ValueError(f"Unknown SAE width: {sae_width}")
    elif sae_repo_id == "fnlp/Llama3_1-8B-Base-LXR-32x":
        num_layers = 32

        assert sae_layer_percent == 25
        sae_layer = int(num_layers * (sae_layer_percent / 100))

        assert sae_layer in [8, 16, 24]

        if sae_width is None:
            sae_width = 32
        sae_filename = ""
    elif sae_repo_id == "anon-org/qwen3-8b-saes":
        num_layers = 36
        sae_layer = int(num_layers * (sae_layer_percent / 100))

        # Only have these SAEs available: https://huggingface.co/anon-org/qwen3-8b-saes/tree/main
        assert sae_layer in [9, 18, 27]

        if sae_width is None:
            sae_width = 2
        sae_filename = f"saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_{sae_layer}/trainer_{sae_width}/ae.pt"
    else:
        raise ValueError(f"Unknown SAE repo ID: {sae_repo_id}")
    return SAEInfo(
        sae_width=sae_width,
        sae_layer=sae_layer,
        sae_layer_percent=sae_layer_percent,
        sae_filename=sae_filename,
        sae_repo_id=sae_repo_id,
    )


# Configuration variables - no longer need a config class


# SAE Classes
class BaseSAE(torch.nn.Module, ABC):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        super().__init__()

        # Required parameters
        self.W_enc = torch.nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = torch.nn.Parameter(torch.zeros(d_sae, d_in))

        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))

        # Required attributes
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.hook_layer = hook_layer
        self.d_sae = d_sae

        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        self.to(dtype=self.dtype, device=self.device)

    @abstractmethod
    def encode(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    @abstractmethod
    def decode(self, feature_acts: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    def to(self, *args, **kwargs):
        """Handle device and dtype updates"""
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)

        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self

    @torch.no_grad()
    def check_decoder_norms(self) -> bool:
        """
        It's important to check that the decoder weights are normalized.
        """
        norms = torch.norm(self.W_dec, dim=1).to(dtype=self.dtype, device=self.device)

        # In bfloat16, it's common to see errors of (1/256) in the norms
        tolerance = 1e-2 if self.W_dec.dtype in [torch.bfloat16, torch.float16] else 1e-5

        if torch.allclose(norms, torch.ones_like(norms), atol=tolerance):
            return True
        else:
            max_diff = torch.max(torch.abs(norms - torch.ones_like(norms)))
            print(f"Decoder weights are not normalized. Max diff: {max_diff.item()}")
            return False


class JumpReluSAE(BaseSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        self.threshold = torch.nn.Parameter(torch.zeros(d_sae, dtype=dtype, device=device))
        self.d_sae = d_sae
        self.d_in = d_in

    def encode(self, x: torch.Tensor):
        pre_acts = x @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, feature_acts: torch.Tensor):
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon


def load_gemma_scope_jumprelu_sae(
    repo_id: str,
    filename: str,
    layer: int,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    local_dir: str = "downloaded_saes",
) -> JumpReluSAE:
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )
    pytorch_path = path_to_params.replace(".npz", ".pt")

    # Doing this because npz files are often insanely slow to load
    if not os.path.exists(pytorch_path):
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v) for k, v in params.items()}

        torch.save(pt_params, pytorch_path)

    pt_params = torch.load(pytorch_path)

    d_in = pt_params["W_enc"].shape[0]
    d_sae = pt_params["W_enc"].shape[1]

    assert d_sae >= d_in

    sae = JumpReluSAE(d_in, d_sae, model_name, layer, device, dtype)
    sae.load_state_dict(pt_params)
    sae.to(dtype=dtype, device=device)

    normalized = sae.check_decoder_norms()
    if not normalized:
        raise ValueError("Decoder norms are not normalized. Implement a normalization method.")

    return sae


class BatchTopKSAE(BaseSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        assert isinstance(k, int) and k > 0
        self.register_buffer("k", torch.tensor(k, dtype=torch.int, device=device))

        # BatchTopK requires a global threshold to use during inference. Must be positive.
        self.use_threshold = True
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=dtype, device=device))

    def encode(self, x: torch.Tensor):
        """Note: x can be either shape (B, F) or (B, L, F)"""
        post_relu_feat_acts_BF = torch.nn.functional.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

        if self.use_threshold:
            if self.threshold < 0:  # type: ignore
                raise ValueError("Threshold is not set. The threshold must be set to use it during inference")
            encoded_acts_BF = post_relu_feat_acts_BF * (post_relu_feat_acts_BF > self.threshold)  # type: ignore
            return encoded_acts_BF

        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)  # type: ignore

        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)
        return encoded_acts_BF

    def decode(self, feature_acts: torch.Tensor):
        return (feature_acts @ self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon


def load_dictionary_learning_batch_topk_sae(
    repo_id: str,
    filename: str,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer: int | None = None,
    local_dir: str = "downloaded_saes",
) -> BatchTopKSAE:
    assert "ae.pt" in filename, f"Filename {filename} does not contain 'ae.pt'"

    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )

    pt_params = torch.load(path_to_params, map_location=torch.device("cpu"))

    config_filename = filename.replace("ae.pt", "config.json")
    path_to_config = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        force_download=False,
        local_dir=local_dir,
    )

    with open(path_to_config) as f:
        config = json.load(f)

    if layer is not None:
        assert layer == config["trainer"]["layer"], (
            f"Layer {layer} not in config {config['trainer']['layer']}, repo id {repo_id}, filename {filename}"
        )
    else:
        layer = config["trainer"]["layer"]

    # Transformer lens often uses a shortened model name
    # assert model_name in config["trainer"]["lm_name"], f"Model name {model_name} not in config {config['trainer']['lm_name']}"

    k = config["trainer"]["k"]

    # Print original keys for debugging
    print("Original keys in state_dict:", pt_params.keys())

    # Map old keys to new keys
    key_mapping = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "encoder.bias": "b_enc",
        "bias": "b_dec",
        "k": "k",
        "threshold": "threshold",
    }

    # Create a new dictionary with renamed keys
    renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}

    # due to the way torch uses nn.Linear, we need to transpose the weight matrices
    renamed_params["W_enc"] = renamed_params["W_enc"].T
    renamed_params["W_dec"] = renamed_params["W_dec"].T

    # Print renamed keys for debugging
    print("Renamed keys in state_dict:", renamed_params.keys())

    sae = BatchTopKSAE(
        d_in=renamed_params["b_dec"].shape[0],
        d_sae=renamed_params["b_enc"].shape[0],
        k=k,
        model_name=model_name,
        hook_layer=layer,  # type: ignore
        device=device,
        dtype=dtype,
    )

    sae.load_state_dict(renamed_params)

    sae.to(device=device, dtype=dtype)

    d_sae, d_in = sae.W_dec.data.shape

    assert d_sae >= d_in

    normalized = sae.check_decoder_norms()
    if not normalized:
        raise ValueError("Decoder vectors are not normalized. Please normalize them")

    return sae


def load_sae(
    sae_repo_id: str,
    sae_filename: str,
    sae_layer: int,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> BaseSAE:
    print(f"Loading SAE for layer {sae_layer} from {sae_repo_id}...")

    if sae_repo_id == "google/gemma-scope-9b-it-res":
        sae = load_gemma_scope_jumprelu_sae(
            repo_id=sae_repo_id,
            filename=sae_filename,
            layer=sae_layer,
            model_name=model_name,
            device=device,
            dtype=dtype,
        )
    elif sae_repo_id == "anon-org/qwen3-8b-saes":
        sae = load_dictionary_learning_batch_topk_sae(
            repo_id=sae_repo_id,
            filename=sae_filename,
            layer=sae_layer,
            model_name=model_name,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown SAE repo ID: {sae_repo_id}")

    return sae


# Pydantic schema classes for JSONL output
def load_max_acts_data(
    model_name: str,
    sae_layer: int,
    sae_width: int,
    layer_percent: int,
    context_length: int = 32,
) -> dict[str, torch.Tensor]:
    """Load the max activating examples data."""
    acts_dir = "max_acts"

    if "gemma" in model_name:
        # Construct filename
        acts_filename = f"acts_{model_name}_layer_{sae_layer}_trainer_{sae_width}_layer_percent_{layer_percent}_context_length_{context_length}.pt".replace(
            "/", "_"
        )

        acts_path = os.path.join(acts_dir, acts_filename)

    elif "Qwen" in model_name:
        acts_filename = f"acts_Qwen_Qwen3-8B_layer_{sae_layer}_trainer_{sae_width}_layer_percent_{layer_percent}_context_length_{context_length}.pt"
        acts_path = os.path.join(acts_dir, acts_filename)

    # Download if not exists
    if not os.path.exists(acts_path):
        print(f"📥 Downloading max acts data: {acts_path}")
        try:
            hf_hub_download(
                repo_id="anon-org/sae_max_acts",
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


def get_feature_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    submodule: torch.nn.Module,
    sae: JumpReluSAE,
    tokenized_strs: dict[str, torch.Tensor],
    ignore_bos: bool = True,
) -> torch.Tensor:
    with torch.no_grad():
        pos_acts_BLD = collect_activations(model, submodule, tokenized_strs)
        encoded_pos_acts_BLF = sae.encode(pos_acts_BLD)

    if ignore_bos:
        bos_mask = tokenized_strs["input_ids"] == tokenizer.bos_token_id
        # Note: I use >=, not ==, because occasionally prompts will contain a BOS token
        assert bos_mask.sum() >= encoded_pos_acts_BLF.shape[0], (
            f"Expected at least {encoded_pos_acts_BLF.shape[0]} BOS tokens, but found {bos_mask.sum()}"
        )

        mask = get_bos_eos_pad_mask(tokenizer, tokenized_strs["input_ids"])
        encoded_pos_acts_BLF[mask] = 0

    return encoded_pos_acts_BLF
