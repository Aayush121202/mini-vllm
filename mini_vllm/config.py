"""Engine configuration, parsed from a HuggingFace ``config.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

_DTYPE_ALIASES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}


@dataclass(frozen=True)
class ModelConfig:
    model_path: str
    architecture: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool

    @property
    def num_queries_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_json(cls, model_path: str | Path) -> ModelConfig:
        path = Path(model_path)
        with open(path / "config.json") as f:
            hf = json.load(f)

        architecture = hf["architectures"][0]
        hidden_size = hf["hidden_size"]
        num_heads = hf["num_attention_heads"]
        return cls(
            model_path=str(path),
            architecture=architecture,
            hidden_size=hidden_size,
            num_hidden_layers=hf["num_hidden_layers"],
            num_attention_heads=num_heads,
            num_key_value_heads=hf.get("num_key_value_heads", num_heads),
            head_dim=hf.get("head_dim", hidden_size // num_heads),
            intermediate_size=hf["intermediate_size"],
            vocab_size=hf["vocab_size"],
            rms_norm_eps=hf.get("rms_norm_eps", 1e-6),
            rope_theta=hf.get("rope_theta", 10000.0),
            max_position_embeddings=hf.get("max_position_embeddings", 32768),
            tie_word_embeddings=hf.get("tie_word_embeddings", False),
            # Qwen2 always uses bias on q/k/v; Llama-family reads it from config.
            attention_bias=hf.get("attention_bias", architecture.startswith("Qwen2")),
        )


def parse_dtype(spec: str | torch.dtype, device: torch.device) -> torch.dtype:
    """Resolve a dtype spec, defaulting per-device.

    fp32 on CPU/MPS keeps parity tests tight; CUDA prefers bf16 where the hardware
    supports it (Ampere+) and falls back to fp16 on Turing, which has no hardware bf16.
    """
    if isinstance(spec, torch.dtype):
        return spec
    if spec != "auto":
        try:
            return _DTYPE_ALIASES[spec]
        except KeyError:
            raise ValueError(f"unknown dtype {spec!r}; expected one of {sorted(_DTYPE_ALIASES)}")
    if device.type == "cuda":
        # Not torch.cuda.is_bf16_supported(): it counts software emulation, so it reports
        # True on Turing (T4), which has no hardware bf16 and silently ran in a slow,
        # coarse emulated bf16. Hardware bf16 starts at Ampere, compute capability 8.0.
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32
