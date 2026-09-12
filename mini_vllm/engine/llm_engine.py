"""The engine contract.

``step()`` runs exactly ONE forward pass and returns incremental output for every live
request. Nothing above this class is allowed to own a decode loop -- that rule is what
makes continuous batching possible in Phase 3 without an API change.

Phase 0 limitation: one sequence runs at a time, others wait in FIFO order. The queue
structure is already here; Phase 1 fills in batching and Phase 3 replaces the inline
policy with a real Scheduler.
"""

from __future__ import annotations

import itertools
from collections import deque
from pathlib import Path

import torch

from mini_vllm.config import CacheConfig, ModelConfig
from mini_vllm.engine.sequence import (
    RequestOutput,
    SamplingParams,
    Sequence,
    SequenceStatus,
)
from mini_vllm.layers.sampler import Sampler
from mini_vllm.models.loader import resolve_model_path
from mini_vllm.runner.model_runner import ModelRunner


class LLMEngine:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "auto",
        max_seq_len: int = 2048,
    ) -> None:
        path = resolve_model_path(model_path)
        self.config = ModelConfig.from_json(path)
        self.runner = ModelRunner(
            self.config,
            CacheConfig(max_seq_len=max_seq_len, max_batch_size=1),
            device=device,
            dtype=dtype,
        )
        self.sampler = Sampler()

        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self._generators: dict[str, torch.Generator] = {}
        self._counter = itertools.count()

        self.tokenizer = None  # lazily loaded; only generate() needs it

    # -- contract -----------------------------------------------------------------

    def add_request(
        self,
        request_id: str | None,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
    ) -> str:
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")
        request_id = request_id or f"req-{next(self._counter)}"
        params = sampling_params or SamplingParams()
        if params.seed is not None:
            gen = torch.Generator(device=self.runner.device)
            gen.manual_seed(params.seed)
            self._generators[request_id] = gen
        self.waiting.append(Sequence(request_id, list(prompt_token_ids), params))
        return request_id

    def step(self) -> list[RequestOutput]:
        """One forward pass. Prefills a waiting request, or decodes the running one."""
        if not self.running and self.waiting:
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            self.runner.reset()
            logits = self.runner.execute(seq.prompt_token_ids, start_pos=0)
            seq.num_computed_tokens = seq.num_prompt_tokens
        elif self.running:
            seq = self.running[0]
            logits = self.runner.execute([seq.last_token_id], start_pos=seq.num_tokens - 1)
        else:
            return []

        token_id = self.sampler(
            logits[-1:], [seq.sampling_params], [self._generators.get(seq.request_id)]
        )
        seq.append_token(int(token_id))
        seq.check_stop()

        output = RequestOutput.from_seq(seq, new_token_ids=[seq.last_token_id])
        if seq.status.is_finished:
            self._retire(seq)
        return [output]

    def abort(self, request_id: str) -> None:
        for seq in list(self.running):
            if seq.request_id == request_id:
                seq.status = SequenceStatus.FINISHED_ABORTED
                self._retire(seq)
        self.waiting = deque(s for s in self.waiting if s.request_id != request_id)
        self._generators.pop(request_id, None)

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running)

    # -- convenience ---------------------------------------------------------------

    def generate(
        self, prompt: str, sampling_params: SamplingParams | None = None, *, chat: bool = True
    ) -> str:
        """Blocking single-prompt helper. Real callers drive step() themselves."""
        tokenizer = self._get_tokenizer()
        if chat and tokenizer.chat_template:
            # Render to text and tokenize separately: apply_chat_template(tokenize=True)
            # returns a plain list in transformers 4.x but a BatchEncoding in 5.x.
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_token_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        else:
            prompt_token_ids = tokenizer(prompt).input_ids

        params = sampling_params or SamplingParams()
        if not params.stop_token_ids and tokenizer.eos_token_id is not None:
            params.stop_token_ids = [tokenizer.eos_token_id]

        request_id = self.add_request(None, prompt_token_ids, params)
        pieces: list[int] = []
        while self.has_unfinished_requests():
            for out in self.step():
                if out.request_id == request_id:
                    pieces.extend(out.new_token_ids)
        # Phase 1 replaces this with the incremental detokenizer: decoding the whole
        # sequence every step is O(n^2) and cannot handle stop strings correctly.
        return tokenizer.decode(pieces, skip_special_tokens=True)

    def _get_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        return self.tokenizer

    def _retire(self, seq: Sequence) -> None:
        if seq in self.running:
            self.running.remove(seq)
        self._generators.pop(seq.request_id, None)
