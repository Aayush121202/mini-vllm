"""The Qwen2 decoder, plus weight loading.

Everything model-shaped lives here: RMSNorm, RoPE, grouped-query attention, SwiGLU MLP,
the decoder stack, and the safetensors loader. Module names mirror the HuggingFace
checkpoint exactly, so loading is an identity mapping plus tied-embedding handling.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from mini_vllm.cache import AttnMetadata
from mini_vllm.config import ModelConfig

SUPPORTED_ARCHITECTURES = ("Qwen2ForCausalLM",)


# --------------------------------------------------------------------------- layers


class RMSNorm(nn.Module):
    """Matches HF exactly, including the fp32 upcast.

    The upcast is not cosmetic: computing the mean-square in fp16 loses parity against
    the reference and drifts over 24 layers.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF's half-split convention (NOT the interleaved one from the RoFormer paper).

    Getting this wrong produces a model that still emits fluent-looking text while being
    subtly wrong, which is why parity tests exist.
    """
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables.

    Buffers are non-persistent, so state_dict stays clean and one instance can be shared
    across layers. They are built by ambient factory calls so they land on whatever
    device the model is constructed under -- an explicit ``device=`` argument here leaves
    them on CPU while the weights move to the accelerator. They stay fp32 regardless of
    model dtype; fp16 rope tables measurably degrade long-context positions.
    """

    def __init__(self, head_dim: int, max_position_embeddings: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q, k: (N, seq, num_heads, head_dim); positions: (N, seq).

        Positions are per row, not shared: in a decode step every sequence in the batch
        sits at a different absolute position.
        """
        cos = self.cos_cached[positions].to(q.dtype).unsqueeze(2)
        sin = self.sin_cached[positions].to(q.dtype).unsqueeze(2)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(b, s, n_kv, d) -> (b, s, n_kv * n_rep, d) for grouped-query attention."""
    if n_rep == 1:
        return x
    b, s, n_kv, d = x.shape
    return x[:, :, :, None, :].expand(b, s, n_kv, n_rep, d).reshape(b, s, n_kv * n_rep, d)


class Attention(nn.Module):
    """Grouped-query attention.

    Note the signature: the cache is passed *in*. This module never allocates cache
    memory, which is what allows the cache implementation to change underneath it
    without touching the model definition.
    """

    def __init__(self, config: ModelConfig, layer_idx: int, rotary: RotaryEmbedding) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_queries_per_kv = config.num_queries_per_kv
        self.scaling = self.head_dim**-0.5
        self.rotary = rotary

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor, kv_cache, md: AttnMetadata) -> torch.Tensor:
        n, seq, _ = hidden.shape

        q = self.q_proj(hidden).view(n, seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(n, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(n, seq, self.num_kv_heads, self.head_dim)

        q, k = self.rotary(q, k, md.positions)

        # Write first, then read the whole context back. Doing it in that order means
        # prefill and decode are the same code path: a prefill whose prefix is already
        # cached simply finds more context waiting for it.
        flat = (-1, self.num_kv_heads, self.head_dim)
        kv_cache.write(self.layer_idx, k.reshape(flat), v.reshape(flat), md.slot_mapping)
        k_all, v_all = kv_cache.gather(self.layer_idx, md.block_tables)

        # sdpa wants (N, heads, seq, head_dim).
        q = q.transpose(1, 2)
        k_all = repeat_kv(k_all, self.num_queries_per_kv).transpose(1, 2)
        v_all = repeat_kv(v_all, self.num_queries_per_kv).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k_all, v_all, attn_mask=md.attn_mask, scale=self.scaling
        )
        out = out.transpose(1, 2).reshape(n, seq, self.num_heads * self.head_dim)
        return self.o_proj(out)


# ---------------------------------------------------------------------------- model


class Qwen2MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, rotary: RotaryEmbedding) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_idx, rotary)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden: torch.Tensor, kv_cache, md: AttnMetadata) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), kv_cache, md)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Qwen2Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary = RotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, i, self.rotary) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, kv_cache, md: AttnMetadata) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, kv_cache, md)
        return self.norm(hidden)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache,
        md: AttnMetadata,
        *,
        last_token_only: bool = False,
    ) -> torch.Tensor:
        """(N, seq) -> (N, seq, vocab) logits.

        ``last_token_only`` skips the lm_head projection for every position but the last.
        Generation only ever uses the final row, and for Qwen2.5-0.5B that projection is
        896x151936 per token. The parity test needs all rows, so it stays optional.
        """
        hidden = self.model(input_ids, kv_cache, md)
        if last_token_only:
            hidden = hidden[:, -1:, :]
        return self.lm_head(hidden)


# --------------------------------------------------------------------------- loading


def resolve_model_path(model: str | Path) -> Path:
    """Accept a local directory or a HuggingFace repo id, returning a local directory."""
    path = Path(model)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(model),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"],
        )
    )


def load_weights(model_path: Path) -> dict[str, torch.Tensor]:
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {model_path}")
    state: dict[str, torch.Tensor] = {}
    for f in files:
        state.update(load_file(f, device="cpu"))
    return state


def build_model(config: ModelConfig, device: torch.device, dtype: torch.dtype) -> nn.Module:
    """Construct the model directly in the target dtype and device, then load weights.

    Building inside the device context avoids materialising a second full-precision copy
    on CPU. Rotary tables are created explicitly in fp32, so the default-dtype override
    does not degrade them.
    """
    if config.architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"unsupported architecture {config.architecture!r}; "
            f"supported: {list(SUPPORTED_ARCHITECTURES)}"
        )

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = Qwen2ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)

    state = load_weights(Path(config.model_path))
    if config.tie_word_embeddings and "lm_head.weight" not in state:
        # Tied checkpoints omit lm_head entirely; the module shares storage with the
        # embedding, so feeding it the same tensor keeps load_state_dict strict.
        state["lm_head.weight"] = state["model.embed_tokens.weight"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing weights: {sorted(missing)[:8]}")
    if unexpected:
        raise RuntimeError(f"checkpoint has unexpected weights: {sorted(unexpected)[:8]}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
