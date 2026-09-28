"""The engine: schedule -> batch -> forward -> sample.

``step()`` runs exactly ONE forward pass and returns incremental output for every
request that advanced. Nothing above this class owns a decode loop, which is what lets
the scheduler remake its decision every iteration.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import torch

from mini_vllm.cache import PagedKVCache, build_attn_metadata
from mini_vllm.config import ModelConfig, parse_dtype
from mini_vllm.model import build_model, resolve_model_path
from mini_vllm.request import Request, RequestOutput, SamplingParams
from mini_vllm.sampler import Sampler
from mini_vllm.scheduler import Scheduler

DEFAULT_NUM_BLOCKS = 1024
DEFAULT_BLOCK_SIZE = 16


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
    """Owns the weights and the KV pool, and runs one forward pass at a time."""

    def __init__(
        self,
        config: ModelConfig,
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "auto",
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        self.config = config
        self.device = get_device(device)
        self.dtype = parse_dtype(dtype, self.device)
        self.model = build_model(config, self.device, self.dtype)
        self.cache = PagedKVCache(
            num_layers=config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def __repr__(self) -> str:
        mib = self.cache.num_blocks * self.cache.bytes_per_block / 1024**2
        return (
            f"ModelRunner({self.config.architecture} on {describe_device(self.device)}, "
            f"dtype={self.dtype}, kv={mib:.0f}MiB / {self.cache.capacity_tokens} tokens)"
        )

    @torch.inference_mode()
    def execute(
        self,
        token_ids: list[list[int]],
        start_positions: list[int],
        block_tables: list[list[int]],
        *,
        last_token_only: bool = True,
    ) -> torch.Tensor:
        """Run one forward pass over a batch.

        Every row must contribute the same number of new tokens: a step is either one
        prefill (N=1, many tokens) or a decode over the running set (N=many, one each).
        Returns logits of shape (N, num_new_tokens, vocab), or (N, 1, vocab) when
        last_token_only.
        """
        num_new = len(token_ids[0])
        if any(len(t) != num_new for t in token_ids):
            raise ValueError("all rows in a batch must contribute the same number of tokens")

        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        md = build_attn_metadata(
            block_tables=block_tables,
            start_positions=start_positions,
            num_new_tokens=num_new,
            block_size=self.cache.block_size,
            device=self.device,
        )
        return self.model(input_ids, self.cache, md, last_token_only=last_token_only)

    def reset(self) -> None:
        self.cache.reset()


class LLMEngine:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "auto",
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        block_size: int = DEFAULT_BLOCK_SIZE,
        max_num_seqs: int = 32,
        static_batching: bool = False,
    ) -> None:
        path = resolve_model_path(model_path)
        self.config = ModelConfig.from_json(path)
        self.runner = ModelRunner(
            self.config, device=device, dtype=dtype, num_blocks=num_blocks, block_size=block_size
        )
        self.scheduler = Scheduler(
            self.runner.cache, max_num_seqs=max_num_seqs, static_batching=static_batching
        )
        self.sampler = Sampler()
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
        self.scheduler.add(Request(request_id, list(prompt_token_ids), params))
        return request_id

    def step(self) -> list[RequestOutput]:
        """One forward pass: a prefill for one request, or a decode for all of them."""
        scheduled = self.scheduler.schedule()
        if not scheduled:
            return []

        requests = scheduled.requests
        if scheduled.is_prefill:
            # Everything not already in the cache: the whole prompt normally, or prompt
            # plus generated tokens when resuming a preempted request.
            batch = [[r.token_ids[r.num_computed_tokens :] for r in requests][0]]
        else:
            batch = [[r.last_token_id] for r in requests]
        starts = [r.num_computed_tokens for r in requests]
        tables = [self.scheduler.block_tables[r.request_id] for r in requests]

        logits = self.runner.execute(batch, starts, tables)[:, -1, :]
        for request, tokens in zip(requests, batch):
            request.num_computed_tokens += len(tokens)

        sampled = self.sampler(
            logits,
            [r.sampling_params for r in requests],
            [self._generators.get(r.request_id) for r in requests],
        )

        outputs = []
        for request, token_id in zip(requests, sampled.tolist()):
            request.append_token(token_id)
            request.check_stop()
            outputs.append(RequestOutput.from_request(request, [token_id]))
            if request.status.is_finished:
                self.scheduler.finish(request)
                self._generators.pop(request.request_id, None)
        return outputs

    def abort(self, request_id: str) -> None:
        self.scheduler.abort(request_id)
        self._generators.pop(request_id, None)

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished()

    def stats(self) -> dict[str, float | int]:
        return {
            "running": len(self.scheduler.running),
            "waiting": len(self.scheduler.waiting),
            "kv_utilization": round(self.runner.cache.utilization(), 4),
            "blocks_used": self.runner.cache.allocator.num_used,
            "blocks_total": self.runner.cache.num_blocks,
            "preemptions": self.scheduler.num_preemptions,
        }

    # -- convenience ---------------------------------------------------------------

    def generate(
        self, prompt: str, sampling_params: SamplingParams | None = None, *, chat: bool = True
    ) -> str:
        """Blocking single-prompt helper. Real callers drive step() themselves."""
        tokenizer = self.get_tokenizer()
        prompt_token_ids = self.encode(prompt, chat=chat)

        params = sampling_params or SamplingParams()
        if not params.stop_token_ids and tokenizer.eos_token_id is not None:
            params.stop_token_ids = [tokenizer.eos_token_id]

        request_id = self.add_request(None, prompt_token_ids, params)
        pieces: list[int] = []
        while self.has_unfinished_requests():
            for out in self.step():
                if out.request_id == request_id:
                    pieces.extend(out.new_token_ids)
        return tokenizer.decode(pieces, skip_special_tokens=True)

    def encode(self, prompt: str, *, chat: bool = True) -> list[int]:
        tokenizer = self.get_tokenizer()
        if chat and tokenizer.chat_template:
            # Render to text and tokenize separately: apply_chat_template(tokenize=True)
            # returns a plain list in transformers 4.x but a BatchEncoding in 5.x.
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            return tokenizer(prompt, add_special_tokens=False).input_ids
        return tokenizer(prompt).input_ids

    def get_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        return self.tokenizer
