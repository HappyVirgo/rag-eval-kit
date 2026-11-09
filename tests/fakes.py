"""Offline stand-ins for the Anthropic client used by the Claude-backed components."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use(block_id: str, name: str, tool_input: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def message(stop_reason: str, *content: SimpleNamespace, parsed: Any = None) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=list(content), parsed_output=parsed)


@dataclass
class FakeMessages:
    """Returns scripted responses in order and records every request's kwargs."""

    responses: list[Any]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def _next(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def create(self, **kwargs: Any) -> Any:
        return self._next(**kwargs)

    def parse(self, **kwargs: Any) -> Any:
        return self._next(**kwargs)


class FakeClient:
    def __init__(self, *responses: Any) -> None:
        self.messages = FakeMessages(list(responses))
