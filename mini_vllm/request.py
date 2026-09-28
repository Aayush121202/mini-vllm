"""Request state. Deliberately free of any tensor or device concepts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH = auto()
    FINISHED_ABORTED = auto()

    @property
    def is_finished(self) -> bool:
        return self in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH,
            RequestStatus.FINISHED_ABORTED,
        )


@dataclass
class SamplingParams:
    temperature: float = 0.0  # 0.0 means greedy
    top_p: float = 1.0
    top_k: int = -1  # -1 disables
    max_tokens: int = 64
    stop_token_ids: list[int] = field(default_factory=list)
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 (disabled) or a positive integer")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    output_token_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    # Number of tokens whose KV is already in the cache. Prefix cache hits advance this
    # before any forward pass runs.
    num_computed_tokens: int = 0

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def last_token_id(self) -> int:
        return self.output_token_ids[-1] if self.output_token_ids else self.prompt_token_ids[-1]

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens < self.num_prompt_tokens

    @property
    def num_uncomputed_tokens(self) -> int:
        """Tokens that still need a forward pass: the whole prompt for a fresh request,
        one token during decode, everything again after a preemption."""
        return self.num_tokens - self.num_computed_tokens

    def append_token(self, token_id: int) -> None:
        """Note what this does NOT do: bump num_computed_tokens. A sampled token has no
        KV in the cache until the next forward pass writes it."""
        self.output_token_ids.append(token_id)

    def check_stop(self) -> None:
        """Token-level stop conditions. String-level stops live in the detokenizer."""
        if self.output_token_ids and self.last_token_id in self.sampling_params.stop_token_ids:
            self.status = RequestStatus.FINISHED_STOPPED
        elif len(self.output_token_ids) >= self.sampling_params.max_tokens:
            self.status = RequestStatus.FINISHED_LENGTH


@dataclass
class RequestOutput:
    """One iteration's worth of output for one request."""

    request_id: str
    prompt_token_ids: list[int]
    token_ids: list[int]  # everything generated so far
    new_token_ids: list[int]  # delta produced by this step
    finished: bool
    finish_reason: str | None = None

    @classmethod
    def from_request(cls, req: Request, new_token_ids: list[int]) -> "RequestOutput":
        reason = None
        if req.status.is_finished:
            reason = {
                RequestStatus.FINISHED_STOPPED: "stop",
                RequestStatus.FINISHED_LENGTH: "length",
                RequestStatus.FINISHED_ABORTED: "abort",
            }[req.status]
        return cls(
            request_id=req.request_id,
            prompt_token_ids=list(req.prompt_token_ids),
            token_ids=list(req.output_token_ids),
            new_token_ids=new_token_ids,
            finished=req.status.is_finished,
            finish_reason=reason,
        )
