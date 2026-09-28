from __future__ import annotations

import os

import pytest

TEST_MODEL = os.environ.get("MINI_VLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="session")
def model_path():
    """Local path to the test checkpoint, downloading it once if needed."""
    from mini_vllm.model import resolve_model_path

    try:
        return resolve_model_path(TEST_MODEL)
    except Exception as exc:  # offline, or no hub access
        pytest.skip(f"could not resolve {TEST_MODEL}: {exc}")


@pytest.fixture(scope="session")
def prompt() -> str:
    return "The capital of France is"
