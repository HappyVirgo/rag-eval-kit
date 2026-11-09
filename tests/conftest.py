from __future__ import annotations

import pytest

from rag_eval_kit.models import Chunk


def make_chunk(chunk_id: str, text: str, heading: str = "") -> Chunk:
    """Build a chunk whose doc id is the part of ``chunk_id`` before ``#``."""
    return Chunk(chunk_id=chunk_id, doc_id=chunk_id.split("#")[0], text=text, heading=heading, source="test")


@pytest.fixture
def toy_chunks() -> list[Chunk]:
    return [
        make_chunk("pricing#0", "The Team plan costs $12 per user per month.", "Pricing"),
        make_chunk("limits#0", "The Team plan allows 300 requests per minute. Exceeding it gives 429.", "Rate limits"),
        make_chunk("hooks#0", "Webhook deliveries are signed with HMAC-SHA256 in a signature header.", "Webhooks"),
        make_chunk("sso#0", "SAML single sign-on is available on the Business plan.", "SSO"),
    ]


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never call the network: make sure no code path picks up a real key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
