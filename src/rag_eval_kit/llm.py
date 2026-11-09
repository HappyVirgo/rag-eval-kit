"""Shared Anthropic SDK configuration and error handling."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

import anthropic

MODEL = "claude-opus-5-5"
EFFORT = "medium"
#: Server-side refusal fallback: the API re-routes a declined request by category.
FALLBACK_HEADERS = {"anthropic-beta": "server-side-fallback-2026-07-01"}
FALLBACK_BODY: dict[str, Any] = {"fallbacks": "default"}

T = TypeVar("T")


class LLMError(RuntimeError):
    """Raised when a Claude call fails or returns an unusable response."""


def has_api_key() -> bool:
    """True when ``ANTHROPIC_API_KEY`` is set to a non-empty value."""
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def make_client() -> anthropic.Anthropic:
    """Create an Anthropic client from the environment."""
    return anthropic.Anthropic()


def request_options() -> dict[str, Any]:
    """Keyword arguments every request sends: effort and the refusal fallback."""
    return {
        "output_config": {"effort": EFFORT},
        "extra_headers": FALLBACK_HEADERS,
        "extra_body": FALLBACK_BODY,
    }


def check_stop_reason(stop_reason: str | None) -> None:
    """Raise :class:`LLMError` for stop reasons that mean the content is unusable."""
    if stop_reason == "refusal":
        raise LLMError("Claude declined to answer this request (stop_reason=refusal).")
    if stop_reason == "max_tokens":
        raise LLMError("Claude's response was truncated (stop_reason=max_tokens).")


def call_claude(fn: Callable[[], T]) -> T:
    """Run an SDK call, translating SDK exceptions (most specific first) to :class:`LLMError`."""
    try:
        return fn()
    except anthropic.RateLimitError as exc:
        raise LLMError(f"Rate limited by the Anthropic API; retry later ({exc.status_code}).") from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError(f"Could not reach the Anthropic API: {exc}") from exc
