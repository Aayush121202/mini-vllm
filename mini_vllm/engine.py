"""The engine.

``step()`` runs exactly ONE forward pass and returns incremental output for every live
request. Nothing above this class owns a decode loop -- that rule is what makes
continuous batching possible without an API change.
"""

from __future__ import annotations

import itertools
from collections import deque
from pathlib import Path

import torch

from mini_vllm.cache import AttnMetadata, ContiguousKVCache
from mini_vllm.config import CacheConfig, ModelConfig, parse_dtype
from mini_vllm.model import build_model, resolve_model_path
from mini_vllm.request import Request, RequestOutput, RequestStatus, SamplingParams
from mini_vllm.sampler import Sampler


def get_device(spec: str | torch.device = "auto") -> torch.device:
    if isinstance(spec, torch.device):
        return spec
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        return f"{torch.cuda.get_device_name(device)} (sm_{major}{minor})"
    return device.type


class ModelRunner:
    """Owns the weights and the cache, and runs one forward pass at a time."""

    def __init__(
        self,
        config: ModelConfig,
        cache_config: CacheConfig | None = None,
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "auto",
    ) -> None:
        self.config = config
        self.cache_config = cache_config or CacheConfig()
        self.device = get_device(device)
        self.dtype = parse_dtype(dtype, self.device)
        self.model = build_model(config, self.device, self.dtype)
        self.kv_cache = ContiguousKVCache(
            num_layers=config.num_hidden_layers,
            max_batch_size=self.cache_config.max_batch_size,
            max_seq_len=self.cache_config.max_seq_len,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def __repr__(self) -> str:
        return (
            f"ModelRunner({self.config.architecture} on {describe_device(self.device)}, "
            f"dtype={self.dtype})"
        )

    @torch.inference_mode()
    def execute(
        self, token_ids: list[int], start_pos: int, *, last_token_only: bool = True
    ) -> torch.Tensor:
        """Run one forward pass over ``token_ids`` starting at ``start_pos``.

        Returns logits of shape (seq, vocab), or (1, vocab) when last_token_only.
        """
        is_prefill = len(token_ids) > 1
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        positions = torch.arange(
            start_pos, start_pos + len(token_ids), dtype=torch.long, device=self.device
        )
        md = AttnMetadata(
            is_prefill=is_prefill, start_pos=start_pos, num_new_tokens=len(token_ids)
        )
        logits = self.model(
            input_ids, positions, self.kv_cache, md, last_token_only=last_token_only
        )
        return logits.squeeze(0)

    def reset(self) -> None:
        self.kv_cache.reset()


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

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
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
        self.waiting.append(Request(request_id, list(prompt_token_ids), params))
        return request_id

    def step(self) -> list[RequestOutput]:
        """One forward pass. Prefills a waiting request, or decodes the running one."""
        if not self.running and self.waiting:
            req = self.waiting.popleft()
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            self.runner.reset()
            logits = self.runner.execute(req.prompt_token_ids, start_pos=0)
            req.num_computed_tokens = req.num_prompt_tokens
        elif self.running:
            req = self.running[0]
            logits = self.runner.execute([req.last_token_id], start_pos=req.num_tokens - 1)
        else:
            return []

        token_id = self.sampler(
            logits[-1:], [req.sampling_params], [self._generators.get(req.request_id)]
        )
        req.append_token(int(token_id))
        req.check_stop()

        output = RequestOutput.from_request(req, new_token_ids=[req.last_token_id])
        if req.status.is_finished:
            self._retire(req)
        return [output]

    def abort(self, request_id: str) -> None:
        for req in list(self.running):
            if req.request_id == request_id:
                req.status = RequestStatus.FINISHED_ABORTED
                self._retire(req)
        self.waiting = deque(r for r in self.waiting if r.request_id != request_id)
        self._generators.pop(request_id, None)

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running)

    # -- convenience ---------------------------------------------------------------

    def generate(
        self, prompt: str, sampling_params: SamplingParams | None = None, *, chat: bool = True
    ) -> str:
        """Blocking single-prompt helper. Real callers drive step() themselves."""
        tokenizer = self.get_tokenizer()
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
        # Replaced by the incremental detokenizer: decoding the whole sequence every
        # step is O(n^2) and cannot handle stop strings correctly.
        return tokenizer.decode(pieces, skip_special_tokens=True)

    def get_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        return self.tokenizer

    def _retire(self, req: Request) -> None:
        if req in self.running:
            self.running.remove(req)
        self._generators.pop(req.request_id, None)
