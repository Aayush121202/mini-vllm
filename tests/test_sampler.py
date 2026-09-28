"""Fast tests: no weights, no network."""

from __future__ import annotations

import pytest
import torch

from mini_vllm.request import Request, RequestOutput, RequestStatus, SamplingParams
from mini_vllm.sampler import Sampler, apply_top_k, apply_top_p


def test_greedy_is_argmax():
    logits = torch.tensor([[1.0, 5.0, 2.0], [9.0, 0.0, 0.0]])
    out = Sampler()(logits, [SamplingParams(temperature=0.0)] * 2)
    assert out.tolist() == [1, 0]


def test_top_k_masks_all_but_k():
    logits = torch.tensor([[1.0, 5.0, 2.0, 4.0]])
    masked = apply_top_k(logits, torch.tensor([2]))
    assert torch.isinf(masked[0, 0]) and torch.isinf(masked[0, 2])
    assert masked[0, 1] == 5.0 and masked[0, 3] == 4.0


def test_top_k_is_per_row():
    logits = torch.tensor([[1.0, 5.0, 2.0, 4.0], [1.0, 5.0, 2.0, 4.0]])
    masked = apply_top_k(logits, torch.tensor([1, 3]))
    assert torch.isfinite(masked[0]).sum() == 1
    assert torch.isfinite(masked[1]).sum() == 3


def test_top_p_keeps_the_crossing_token():
    # softmax over [0, 10] is ~[4.5e-5, 1.0]; top_p=0.5 must still keep one token.
    logits = torch.tensor([[0.0, 10.0]])
    masked = apply_top_p(logits, torch.tensor([0.5]))
    assert torch.isfinite(masked).sum() == 1


def test_batched_sampling_matches_row_by_row():
    """The vectorised path must agree with sampling each row on its own -- otherwise
    a request's output would depend on who else is in the batch."""
    torch.manual_seed(0)
    logits = torch.randn(4, 64)
    params = [
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=0.8, top_k=5, seed=1),
        SamplingParams(temperature=1.0, top_p=0.9, seed=2),
        SamplingParams(temperature=0.0, top_k=3),
    ]

    def gens():
        out = []
        for p in params:
            if p.seed is None:
                out.append(None)
                continue
            g = torch.Generator()
            g.manual_seed(p.seed)
            out.append(g)
        return out

    batched = Sampler()(logits, params, gens()).tolist()
    solo = [Sampler()(logits[i : i + 1], [p], [gens()[i]]).item() for i, p in enumerate(params)]
    assert batched == solo


def test_seeded_sampling_is_reproducible():
    logits = torch.randn(1, 128)
    params = [SamplingParams(temperature=1.0, seed=1234)]

    def run():
        gen = torch.Generator()
        gen.manual_seed(1234)
        return Sampler()(logits, params, [gen]).tolist()

    assert run() == run()


@pytest.mark.parametrize(
    "kwargs",
    [{"temperature": -1.0}, {"top_p": 0.0}, {"top_p": 1.5}, {"top_k": 0}, {"max_tokens": 0}],
)
def test_invalid_sampling_params_rejected(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_appending_a_token_does_not_mark_it_computed():
    """A sampled token has no KV in the cache until the next forward pass writes it."""
    req = Request("r", [1, 2, 3], SamplingParams())
    req.num_computed_tokens = 3
    req.append_token(4)
    assert req.num_computed_tokens == 3
    assert req.num_tokens == 4
    assert req.num_uncomputed_tokens == 1


def test_request_stops_on_length():
    req = Request("r", [1, 2, 3], SamplingParams(max_tokens=2))
    assert req.is_prefill
    req.num_computed_tokens = 3
    req.append_token(4)
    req.check_stop()
    assert not req.status.is_finished
    req.append_token(5)
    req.check_stop()
    assert req.status is RequestStatus.FINISHED_LENGTH
    assert RequestOutput.from_request(req, [5]).finish_reason == "length"


def test_request_stops_on_stop_token():
    req = Request("r", [1], SamplingParams(max_tokens=10, stop_token_ids=[7]))
    req.append_token(7)
    req.check_stop()
    assert req.status is RequestStatus.FINISHED_STOPPED
