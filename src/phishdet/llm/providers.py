"""Chat-completion providers behind one small interface.

The HTTP providers call the public REST APIs directly with httpx, so there is
no SDK dependency and the request each one sends is easy to read and test.
API keys are read from the environment only when a provider is constructed
without one, i.e. only when you actually run an evaluation against that API.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx


class ProviderError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str


class Provider(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, max_tokens: int = 200) -> Completion: ...


def _api_key(explicit: str | None, env_var: str) -> str:
    if explicit:
        return explicit
    key = os.environ.get(env_var)
    if not key:
        raise ProviderError(f"no API key: set {env_var} or pass api_key=")
    return key


class _HTTPProvider:
    name = "http"
    retry_statuses = {429, 500, 502, 503, 504, 529}

    def __init__(self, model: str, timeout: float = 60.0, max_retries: int = 3, client: httpx.Client | None = None,
                 backoff_s: float = 2.0):
        self.model = model
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.client = client or httpx.Client(timeout=timeout)

    def _post(self, url: str, headers: dict, body: dict) -> tuple[dict, float]:
        last = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self.client.post(url, headers=headers, json=body)
            except httpx.TransportError as e:
                last = f"transport error: {e}"
            else:
                latency = time.perf_counter() - t0
                if resp.status_code == 200:
                    return resp.json(), latency
                last = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in self.retry_statuses:
                    raise ProviderError(f"{self.name} request failed, {last}")
            if attempt < self.max_retries:
                time.sleep(self.backoff_s * (2**attempt))
        raise ProviderError(f"{self.name} request failed after {self.max_retries + 1} attempts, {last}")


class OpenAIProvider(_HTTPProvider):
    """POST /v1/chat/completions with JSON mode."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None,
                 base_url: str = "https://api.openai.com/v1", temperature: float | None = 0.0, **kw):
        super().__init__(model, **kw)
        self._key = _api_key(api_key, "OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def complete(self, system: str, user: str, max_tokens: int = 200) -> Completion:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        data, latency = self._post(f"{self.base_url}/chat/completions", headers, body)
        try:
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
            return Completion(text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)),
                              latency, data.get("model", self.model))
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"unexpected OpenAI response shape: {e}") from e


class AnthropicProvider(_HTTPProvider):
    """POST /v1/messages."""

    name = "anthropic"
    api_version = "2023-06-01"

    def __init__(self, model: str = "claude-haiku-4-5", api_key: str | None = None,
                 base_url: str = "https://api.anthropic.com", temperature: float | None = 0.0, **kw):
        super().__init__(model, **kw)
        self._key = _api_key(api_key, "ANTHROPIC_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def complete(self, system: str, user: str, max_tokens: int = 200) -> Completion:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {
            "x-api-key": self._key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }
        data, latency = self._post(f"{self.base_url}/v1/messages", headers, body)
        try:
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
            usage = data.get("usage", {})
            return Completion(text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)),
                              latency, data.get("model", self.model))
        except (KeyError, TypeError) as e:
            raise ProviderError(f"unexpected Anthropic response shape: {e}") from e


_SUSPICIOUS = re.compile(
    r"verify your|your account|suspend|password|login|log in|sign in|confirm|urgent|click|bank|invoice|mailbox",
    re.IGNORECASE,
)


@dataclass
class FakeProvider:
    """Offline stand-in used by the tests and for dry runs.

    By default it answers with a keyword rule, so a dry run produces sensible
    looking output. ``responses`` overrides that with scripted replies (one per
    call, cycling), which the tests use to feed malformed output. Token counts
    are estimated as characters / 4. Nothing leaves the machine.
    """

    model: str = "fake-rules"
    responses: list[str] | None = None
    latency_s: float = 0.0
    name: str = "fake"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, system: str, user: str, max_tokens: int = 200) -> Completion:
        self.calls.append((system, user))
        if self.responses:
            text = self.responses[(len(self.calls) - 1) % len(self.responses)]
        else:
            email = user.rsplit("<email>", 1)[-1]
            hits = len(_SUSPICIOUS.findall(email))
            label = "phishing" if hits >= 2 else "legitimate"
            conf = min(0.95, 0.55 + 0.1 * hits) if label == "phishing" else 0.7
            text = json.dumps({"label": label, "confidence": round(conf, 2), "reason": f"{hits} suspicious phrases (keyword rule)"})
        if self.latency_s:
            time.sleep(self.latency_s)
        return Completion(text, (len(system) + len(user)) // 4, len(text) // 4, self.latency_s, self.model)


def make_provider(name: str, model: str | None = None, **kw) -> Provider:
    if name == "openai":
        return OpenAIProvider(model or "gpt-4o-mini", **kw)
    if name == "anthropic":
        return AnthropicProvider(model or "claude-haiku-4-5", **kw)
    if name == "fake":
        return FakeProvider(model or "fake-rules")
    raise ValueError(f"unknown provider {name!r}")
