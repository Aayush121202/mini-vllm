"""Incremental detokenization. Offline: uses a byte-level stand-in tokenizer."""

from __future__ import annotations

import pytest

from mini_vllm.detokenizer import IncrementalDetokenizer


class ByteTokenizer:
    """Stands in for byte-level BPE: every token is a byte string, and decoding an
    incomplete UTF-8 sequence yields the replacement character, exactly as HF does."""

    def __init__(self, pieces: list[bytes]) -> None:
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=True) -> str:
        return b"".join(self.pieces[i] for i in ids).decode("utf-8", errors="replace")


def feed(detok: IncrementalDetokenizer, ids) -> str:
    return "".join(detok.add_token(i) for i in ids) + detok.flush()


def test_streams_plain_text():
    tok = ByteTokenizer([b"Hello", b", ", b"world", b"!"])
    detok = IncrementalDetokenizer(tok)
    assert feed(detok, [0, 1, 2, 3]) == "Hello, world!"
    assert detok.text == "Hello, world!"


def test_multibyte_codepoint_split_across_tokens():
    """A smiley is four bytes. Split across two tokens, neither half may be emitted as
    a replacement character."""
    tok = ByteTokenizer([b"hi ", b"\xf0\x9f", b"\x99\x82", b" there"])
    detok = IncrementalDetokenizer(tok)
    out = feed(detok, [0, 1, 2, 3])
    assert out == "hi \U0001f642 there"
    assert "�" not in out


def test_emits_nothing_while_mid_codepoint():
    tok = ByteTokenizer([b"\xf0\x9f", b"\x99\x82"])
    detok = IncrementalDetokenizer(tok)
    assert detok.add_token(0) == ""  # incomplete, must not stream a broken char
    assert detok.add_token(1) + detok.flush() == "\U0001f642"


def test_stop_string_inside_one_token():
    tok = ByteTokenizer([b"answer", b"END", b" more"])
    detok = IncrementalDetokenizer(tok, stop_strings=["END"])
    detok.add_token(0)
    detok.add_token(1)
    assert detok.stopped
    assert detok.emitted == "answer"


def test_stop_string_straddling_a_token_boundary():
    """The reason for the held-back tail: neither token contains '<|end|>' on its own."""
    tok = ByteTokenizer([b"done <|en", b"d|> trailing"])
    detok = IncrementalDetokenizer(tok, stop_strings=["<|end|>"])
    emitted = detok.add_token(0)
    assert "<|en" not in emitted  # tail is held back, not streamed
    detok.add_token(1)
    assert detok.stopped
    assert detok.emitted == "done "


def test_text_after_stop_is_discarded():
    tok = ByteTokenizer([b"keep", b"STOPdrop"])
    detok = IncrementalDetokenizer(tok, stop_strings=["STOP"])
    out = feed(detok, [0, 1])
    assert out == "keep"


def test_streamed_chunks_equal_a_single_decode():
    tok = ByteTokenizer([b"The ", b"qu", b"ick ", b"brown ", b"fox"])
    ids = [0, 1, 2, 3, 4]
    detok = IncrementalDetokenizer(tok)
    assert feed(detok, ids) == tok.decode(ids)


@pytest.mark.slow
def test_matches_the_real_tokenizer(model_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ids = tokenizer("Streaming 🙂 detokenization, incrementally.").input_ids
    detok = IncrementalDetokenizer(tokenizer)
    assert feed(detok, ids) == tokenizer.decode(ids, skip_special_tokens=True)
