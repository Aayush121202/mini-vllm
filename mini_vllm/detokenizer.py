"""Turning tokens back into text, one token at a time.

Two things make this harder than calling ``decode()`` every step. First, it is O(n^2) --
re-decoding the whole sequence for every token. Second, and worse, it is wrong: BPE
tokens do not align to character boundaries, so a single token can be half of a UTF-8
codepoint or can retroactively change how the previous token renders.

So this keeps a two-token sliding window and only emits text that can no longer change,
plus a short tail held back so a stop string that straddles a token boundary is still
caught.
"""

from __future__ import annotations

REPLACEMENT_CHAR = "�"


class IncrementalDetokenizer:
    def __init__(self, tokenizer, stop_strings=(), skip_special_tokens: bool = True) -> None:
        self.tokenizer = tokenizer
        self.stop_strings = [s for s in stop_strings if s]
        self.skip_special_tokens = skip_special_tokens

        self.token_ids: list[int] = []
        self.text = ""  # everything decoded so far
        self.emitted = ""  # everything already handed to the caller
        self.stopped = False

        # Sliding window. Decoding [prefix_offset:] and subtracting [prefix_offset:
        # read_offset] gives the delta, while keeping enough left context that tokens
        # which alter the preceding character still render correctly.
        self.prefix_offset = 0
        self.read_offset = 0

    @property
    def _holdback(self) -> int:
        """Characters kept back so a stop string spanning tokens is not emitted first."""
        return max((len(s) for s in self.stop_strings), default=1) - 1

    def _decode(self, start: int, end: int) -> str:
        return self.tokenizer.decode(
            self.token_ids[start:end], skip_special_tokens=self.skip_special_tokens
        )

    def add_token(self, token_id: int) -> str:
        """Feed one sampled token; return the text that is now safe to stream."""
        self.token_ids.append(token_id)

        previous = self._decode(self.prefix_offset, self.read_offset)
        current = self._decode(self.prefix_offset, len(self.token_ids))

        # A trailing replacement char means we are mid-codepoint: wait for the rest.
        if current.endswith(REPLACEMENT_CHAR) or len(current) <= len(previous):
            return ""

        self.text += current[len(previous) :]
        self.prefix_offset = self.read_offset
        self.read_offset = len(self.token_ids)
        return self._emit()

    def _emit(self) -> str:
        for stop in self.stop_strings:
            # Anything before the held-back tail was already checked, so start the
            # search just far enough back to catch a stop straddling that boundary.
            index = self.text.find(stop, max(0, len(self.emitted) - len(stop)))
            if index != -1:
                out = self.text[:index][len(self.emitted) :]
                self.emitted = self.text[:index]
                self.stopped = True
                return out

        safe = self.text[: len(self.text) - self._holdback] if self._holdback else self.text
        if len(safe) <= len(self.emitted):
            return ""
        out = safe[len(self.emitted) :]
        self.emitted = safe
        return out

    def flush(self) -> str:
        """Release the held-back tail once generation is over."""
        if self.stopped or len(self.text) <= len(self.emitted):
            return ""
        out = self.text[len(self.emitted) :]
        self.emitted = self.text
        return out
