"""Cross-attention activation oracle modules.

Each supervisor transformer layer gets a learned cross-attention adapter that
attends to the corresponding supervisee layer's activations. Noninformative
prefix tokens ("?" slots) are filled via cross-attention, and the supervisor's
self-attention reads from those filled slots.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class CrossAttentionAdapter(nn.Module):
    """Gated cross-attention over supervisee activations (Flamingo-style).

    Q comes from the supervisor hidden states; K/V come from supervisee acts.
    A learned sigmoid gate controls the residual contribution, following the
    Flamingo design. sigmoid(gate_init) sets the initial cross-attention scale.
    """

    def __init__(self, hidden_dim: int = 1024, num_heads: int = 16, gate_init: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Flamingo-style sigmoid gate. sigmoid(0)=0.5 gives projections
        # meaningful gradients from step 1; the gate learns to scale down
        # or up from there.
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(
        self,
        hidden_states: torch.Tensor,
        supervisee_acts: torch.Tensor,
        supervisee_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, D] supervisor hidden states (pre-normed)
            supervisee_acts: [B, L_ctx, D] supervisee activations
            supervisee_mask: [B, L_ctx] bool mask (True = valid token)

        Returns:
            [B, L, D] — hidden_states + gated cross-attention output
        """
        orig_dtype = hidden_states.dtype
        B, L, D = hidden_states.shape
        _, L_ctx, _ = supervisee_acts.shape

        # Cast inputs to match projection weight dtype (handles bf16 activations + fp32 weights)
        proj_dtype = self.q_proj.weight.dtype
        hidden_casted = hidden_states.to(proj_dtype)
        supervisee_casted = supervisee_acts.to(proj_dtype)

        Q = self.q_proj(hidden_casted).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(supervisee_casted).view(B, L_ctx, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(supervisee_casted).view(B, L_ctx, self.num_heads, self.head_dim).transpose(1, 2)

        # Build attention mask for SDPA: [B, 1, 1, L_ctx] broadcast over heads and query positions
        attn_mask = None
        if supervisee_mask is not None:
            # SDPA expects float mask where -inf = masked out
            attn_mask = supervisee_mask[:, None, None, :].to(dtype=Q.dtype)
            attn_mask = (1.0 - attn_mask) * torch.finfo(Q.dtype).min

        attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(attn_out)

        return hidden_states + (torch.sigmoid(self.gate) * out).to(orig_dtype)


class CrossAttentionWrapper(nn.Module):
    """Wraps a transformer layer to inject cross-attention after self-attention + FFN.

    Stores supervisee activations in a class-level dict (per-process, DDP-safe).
    """

    _supervisee_acts_store: dict[int, torch.Tensor] = {}

    def __init__(
        self,
        layer_idx: int,
        original_layer: nn.Module,
        hidden_dim: int,
        num_heads: int = 16,
        gate_init: float = -10.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.original_layer = original_layer
        self.cross_attn = CrossAttentionAdapter(hidden_dim, num_heads, gate_init)
        # RMSNorm before cross-attention (matches Qwen style)
        self.cross_attn_norm = _RMSNorm(hidden_dim)

    def __getattr__(self, name: str):
        """Proxy attribute access to original_layer for model-specific attributes
        (e.g. Qwen3's `attention_type`). Only fires for attributes not found on
        the wrapper itself."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_layer, name)

    def forward(self, *args, **kwargs):
        output = self.original_layer(*args, **kwargs)

        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        if self.layer_idx in CrossAttentionWrapper._supervisee_acts_store:
            supervisee_acts = CrossAttentionWrapper._supervisee_acts_store[self.layer_idx]
            supervisee_mask = CrossAttentionWrapper._supervisee_mask_store.get(self.layer_idx)
            normed = self.cross_attn_norm(hidden_states)
            hidden_states = self.cross_attn(normed, supervisee_acts, supervisee_mask)

        if rest is not None:
            return (hidden_states, *rest)
        return hidden_states

    @classmethod
    def set_supervisee_acts(cls, acts_dict: dict[int, torch.Tensor], mask_dict: dict[int, torch.Tensor] | None = None):
        """Set supervisee activations for all layers."""
        cls._supervisee_acts_store = acts_dict
        cls._supervisee_mask_store = mask_dict if mask_dict is not None else {}

    @classmethod
    def clear_supervisee_acts(cls):
        """Clear stored supervisee activations."""
        cls._supervisee_acts_store = {}
        cls._supervisee_mask_store = {}


# Class-level mask store (initialized alongside acts store)
CrossAttentionWrapper._supervisee_mask_store: dict[int, torch.Tensor] = {}


class _RMSNorm(nn.Module):
    """Simple RMS normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return ((x.float() * norm) * self.weight.float()).to(orig_dtype)


def wrap_model_with_cross_attention(
    model: AutoModelForCausalLM,
    hidden_dim: int,
    num_heads: int = 16,
    gate_init: float = -10.0,
) -> int:
    """Wrap all transformer layers with CrossAttentionWrapper.

    Must be called BEFORE applying LoRA so that LoRA targets land on
    `original_layer.*` paths and cross-attention params are trained from scratch.

    Returns the number of layers wrapped.
    """
    layers = model.model.layers
    num_layers = len(layers)

    for i in range(num_layers):
        original_layer = layers[i]
        # Infer device/dtype from the original layer's first parameter
        param = next(original_layer.parameters())
        wrapped = CrossAttentionWrapper(i, original_layer, hidden_dim, num_heads, gate_init)
        wrapped.cross_attn.to(device=param.device, dtype=param.dtype)
        wrapped.cross_attn_norm.to(device=param.device, dtype=param.dtype)
        # Keep gate in float32 — bfloat16 updates at magnitude ~2.0 are
        # smaller than the representable precision and silently round away.
        wrapped.cross_attn.gate.data = wrapped.cross_attn.gate.data.float()
        layers[i] = wrapped

    return num_layers


def get_cross_attn_state_dicts(model: AutoModelForCausalLM) -> dict[int, dict]:
    """Extract cross-attention state dicts for saving."""
    state_dicts = {}
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        if isinstance(layer, CrossAttentionWrapper):
            state_dicts[i] = {
                "cross_attn": layer.cross_attn.state_dict(),
                "cross_attn_norm": layer.cross_attn_norm.state_dict(),
            }
    return state_dicts


def load_cross_attn_state_dicts(model: AutoModelForCausalLM, state_dicts: dict[int, dict]) -> None:
    """Load cross-attention state dicts into wrapped model."""
    layers = model.model.layers
    for i, sd in state_dicts.items():
        layer = layers[i]
        assert isinstance(layer, CrossAttentionWrapper), f"Layer {i} is not wrapped"
        layer.cross_attn.load_state_dict(sd["cross_attn"])
        layer.cross_attn_norm.load_state_dict(sd["cross_attn_norm"])


def prepare_oracle_input(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context_lengths: torch.Tensor,
    query_input_ids: torch.Tensor,
    query_attention_mask: torch.Tensor,
    query_labels: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build oracle input: [PAD...] [? x N] [query prompt tokens].

    The "?" prefix tokens are noninformative placeholder embeddings whose
    hidden states get filled by cross-attention from supervisee acts.

    Args:
        model: The model (possibly LoRA-wrapped) to get embeddings from
        tokenizer: Tokenizer for getting special token IDs
        context_lengths: [B] number of "?" prefix tokens per example
        query_input_ids: [B, L_query] padded query token IDs
        query_attention_mask: [B, L_query] query attention mask
        query_labels: [B, L_query] query labels (-100 for prompt tokens)
        device: Device to place tensors on

    Returns:
        (inputs_embeds, attention_mask, labels) all [B, N_max + L_query]
    """
    B = query_input_ids.shape[0]
    L_query = query_input_ids.shape[1]
    N_max = int(context_lengths.max().item())

    # Get the embedding layer (handle LoRA wrapping)
    if hasattr(model, "base_model"):
        embed_tokens = model.base_model.model.model.embed_tokens
    else:
        embed_tokens = model.model.embed_tokens

    # Get query embeddings
    query_embeds = embed_tokens(query_input_ids.to(device))  # [B, L_query, D]
    D = query_embeds.shape[-1]

    # Build "?" prefix embeddings
    question_mark_id = tokenizer.encode(" ?", add_special_tokens=False)
    assert len(question_mark_id) == 1, f"Expected single token for ' ?', got {len(question_mark_id)}"
    question_mark_id = question_mark_id[0]
    qmark_embed = embed_tokens(torch.tensor([question_mark_id], device=device))  # [1, D]

    # Build the full sequence: [PAD_embed...] [? x N] [query]
    total_len = N_max + L_query
    inputs_embeds = torch.zeros(B, total_len, D, device=device, dtype=query_embeds.dtype)
    attention_mask = torch.zeros(B, total_len, dtype=torch.bool, device=device)
    labels = torch.full((B, total_len), -100, dtype=torch.long, device=device)

    pad_embed = embed_tokens(torch.tensor([tokenizer.pad_token_id], device=device))  # [1, D]

    for b in range(B):
        n = int(context_lengths[b].item())
        pad_len = N_max - n

        # Fill PAD region
        if pad_len > 0:
            inputs_embeds[b, :pad_len] = pad_embed.expand(pad_len, -1)

        # Fill "?" prefix region
        inputs_embeds[b, pad_len : pad_len + n] = qmark_embed.expand(n, -1)
        attention_mask[b, pad_len : pad_len + n] = True

        # Fill query region
        inputs_embeds[b, N_max:] = query_embeds[b]
        attention_mask[b, N_max:] = query_attention_mask[b].bool()
        labels[b, N_max:] = query_labels[b]

    return inputs_embeds, attention_mask, labels
