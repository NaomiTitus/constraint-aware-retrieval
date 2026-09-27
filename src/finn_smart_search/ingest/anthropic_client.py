"""Anthropic Batch API client.

Four translations, each a place things can go wrong silently:

  REQUEST   our {custom_id, params} -> the batch envelope
  RESPONSE  their nested message.content[] -> our {custom_id, type, facets, usage}
  STATUS    their processing_status / result.type -> what census.run() branches on
  AUTH      macOS keychain -> ANTHROPIC_API_KEY, never written to disk

The response translation is the fiddly one: the tool input is nested inside
`result.message.content[]` as a `tool_use` block, so positional indexing breaks
the moment the model emits any preamble. Search by type AND tool name.

Truncation is NOT salvaged. `stop_reason == "max_tokens"` means the JSON was cut
mid-structure; a half-parsed facet set is indistinguishable from a real one
downstream, so it maps to `errored` and census.run()'s existing retry picks it up.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any, Iterable, Iterator, Mapping

MAX_BATCH_REQUESTS = 100_000
TOOL_NAME = "record_ad_facets"
KEYCHAIN_SERVICE = "ANTHROPIC_API_KEY"


class NoCredentialsError(RuntimeError):
    """No API key found. Raised before any network call."""


def _keychain_key() -> str | None:
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    key = out.stdout.strip()
    return key or None


def api_key() -> str:
    """Keychain first, environment second. Never persisted, never logged."""
    return _keychain_key() or os.environ.get("ANTHROPIC_API_KEY") or _fail()


def _fail():
    raise NoCredentialsError(
        f"no API key: keychain service {KEYCHAIN_SERVICE!r} is empty and "
        "ANTHROPIC_API_KEY is unset")


def build_batch(requests: Iterable[Mapping[str, Any]]) -> list[dict]:
    return [{"custom_id": r["custom_id"], "params": r["params"]} for r in requests]


def chunk(requests, size: int = MAX_BATCH_REQUESTS) -> Iterator[list]:
    """Order-preserving. Truncating instead would lose ads with no error."""
    reqs = list(requests)
    for i in range(0, len(reqs), size):
        yield reqs[i:i + size]


# A batch is capped in BYTES as well as in requests, and the byte cap is the
# one that binds here. Measured: a census request serialises to ~43,880 bytes
# because it carries the system block and all 15 few-shot examples, and that
# prefix repeats in every request. 9,599 of them is 421 MB against a
# documented 256 MB cap — the census died on 413 Payload Too Large with
# `chunk(size=100_000)` in the path, because 9,599 never approaches a COUNT
# limit. The guard measured the dimension that could not bind.
MAX_BATCH_BYTES = 200_000_000          # under the documented 256 MB, with room


def chunk_bytes(requests, max_bytes: int = MAX_BATCH_BYTES) -> Iterator[list]:
    """Split so each batch serialises under `max_bytes`. Order-preserving.

    A single request larger than the cap is yielded ALONE rather than dropped:
    it will fail loudly at the API, which is strictly better than an ad
    vanishing from the corpus with no error — the failure `chunk` above was
    written to avoid, one dimension over.
    """
    batch: list = []
    size = 2                            # the "[]" wrapper
    for r in requests:
        n = len(json.dumps(r, ensure_ascii=False).encode("utf-8")) + 1
        if batch and size + n > max_bytes:
            yield batch
            batch, size = [], 2
        batch.append(r)
        size += n
    if batch:
        yield batch


def _err(custom_id: str, message: str) -> dict:
    """No `facets` key on failure — census.run() must never see a partial record."""
    return {"custom_id": custom_id, "type": "errored", "error": message}


def parse_result(raw: Mapping[str, Any], tool_name: str = TOOL_NAME) -> dict:
    """`tool_name` is a parameter because the batch client now carries a SECOND
    tool: the LLM judge (`eval/judge_llm.py`). The default preserves the census
    behaviour exactly, so nothing that called this before changes."""
    cid = raw["custom_id"]
    res = raw.get("result") or {}
    kind = res.get("type")

    if kind != "succeeded":
        out = {"custom_id": cid, "type": kind or "errored"}
        if kind not in ("expired", "canceled"):
            out["type"] = "errored"
            out["error"] = str(res.get("error") or kind)
        return out

    msg = res.get("message") or {}

    # Owner ruling: retry, never salvage.
    if msg.get("stop_reason") == "max_tokens":
        return _err(cid, "stop_reason=max_tokens: tool call truncated mid-JSON")

    block = next((b for b in msg.get("content") or []
                  if b.get("type") == "tool_use" and b.get("name") == tool_name), None)
    if block is None:
        names = [b.get("name") or b.get("type") for b in msg.get("content") or []]
        return _err(cid, f"no {tool_name} tool_use block; got {names}")

    return {"custom_id": cid, "type": "succeeded", "facets": block["input"],
            "usage": dict(msg.get("usage") or {})}


class AnthropicBatchClient:
    """Implements the protocol census.run() calls: submit_batch / poll / results."""

    def __init__(self, key: str | None = None, model: str | None = None):
        self._key = key or api_key()
        self.model = model
        self._client = None

    def __repr__(self) -> str:
        """Never render the key: a key in a traceback ends up in logs."""
        return f"<AnthropicBatchClient model={self.model!r}>"

    __str__ = __repr__

    # Plain REST over httpx rather than the SDK: the installed SDK pulls an
    # httpx2 whose decompressor signature mismatches this environment, and the
    # Batch API is three endpoints. Fewer dependencies, explicit protocol.
    BASE = "https://api.anthropic.com/v1/messages/batches"

    def _headers(self) -> dict:
        return {"x-api-key": self._key, "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "accept-encoding": "identity"}   # sidestep decoder issues

    def submit_batch(self, requests) -> str:
        import httpx
        r = httpx.post(self.BASE, headers=self._headers(),
                       json={"requests": build_batch(requests)}, timeout=120)
        r.raise_for_status()
        return r.json()["id"]

    def poll(self, batch_id: str) -> str:
        import httpx
        r = httpx.get(f"{self.BASE}/{batch_id}", headers=self._headers(), timeout=60)
        r.raise_for_status()
        return r.json()["processing_status"]

    def raw_results(self, batch_id: str):
        """JSONL stream, one object per request."""
        import httpx
        with httpx.stream("GET", f"{self.BASE}/{batch_id}/results",
                          headers=self._headers(), timeout=300,
                          follow_redirects=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.strip():
                    yield json.loads(line)

    def results(self, batch_id: str, tool_name: str = TOOL_NAME):
        for raw in self.raw_results(batch_id):
            yield parse_result(raw, tool_name)
