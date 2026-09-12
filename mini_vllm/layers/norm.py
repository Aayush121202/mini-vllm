from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Matches HF's implementation exactly, including the fp32 upcast.

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
