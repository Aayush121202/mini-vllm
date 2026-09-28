"""Fast tests: no weights, no network."""

from __future__ import annotations

import pytest
import torch

from mini_vllm.request import RequestOutput, SamplingParams, Request, RequestStatus
from mini_vllm.sampler import Sampler, apply_top_k, apply_top_p


def test_greedy_is_argmax():
    logits = torch.tensor([[1.0, 5.0, 2.0], [9.0, 0.0, 0.0]])
    out = Sampler()(logits, [SamplingParams(temperature=0.0)] * 2)
    assert out.tolist() == [1, 0]


def test_top_k_masks_all_but_k():
    logits = torch.tensor([[1.0, 5.0, 2.0, 4.0]])
    masked = apply_top_k(logits, 2)
    assert torch.isinf(masked[0, 0]) and torch.isinf(masked[0, 2])
    assert masked[0, 1] == 5.0 and masked[0, 3] == 4.0


def test_top_p_keeps_the_crossing_token():
    # softmax over [0, 10] is ~[4.5e-5, 1.0]; top_p=0.5 must still keep one token.
    logits = torch.tensor([[0.0, 10.0]])
    masked = apply_top_p(logits, 0.5)
    assert torch.isfinite(masked).sum() == 1


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


def test_sequence_stops_on_length():
    seq = Request("r", [1, 2, 3], SamplingParams(max_tokens=2))
    assert seq.is_prefill
    seq.num_computed_tokens = 3
    seq.append_token(4)
    seq.check_stop()
    assert not seq.status.is_finished
    seq.append_token(5)
    seq.check_stop()
    assert seq.status is RequestStatus.FINISHED_LENGTH
    assert RequestOutput.from_request(seq, [5]).finish_reason == "length"


def test_sequence_stops_on_stop_token():
    seq = Request("r", [1], SamplingParams(max_tokens=10, stop_token_ids=[7]))
    seq.append_token(7)
    seq.check_stop()
    assert seq.status is RequestStatus.FINISHED_STOPPED
