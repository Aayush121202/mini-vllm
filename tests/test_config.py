from __future__ import annotations

import torch

from mini_vllm.config import parse_dtype


def _cuda(monkeypatch, capability: tuple[int, int]) -> torch.device:
    """A CUDA device spec with a faked compute capability, so this runs without a GPU."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: capability)
    # Real PyTorch reports bf16 as supported on Turing too, because it counts emulation.
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *_a, **_k: True)
    return torch.device("cuda")


def test_auto_dtype_is_fp16_on_turing(monkeypatch):
    assert parse_dtype("auto", _cuda(monkeypatch, (7, 5))) == torch.float16


def test_auto_dtype_is_bf16_on_ampere(monkeypatch):
    assert parse_dtype("auto", _cuda(monkeypatch, (8, 0))) == torch.bfloat16


def test_auto_dtype_is_fp32_off_cuda():
    assert parse_dtype("auto", torch.device("cpu")) == torch.float32
