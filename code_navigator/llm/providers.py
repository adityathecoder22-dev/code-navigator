"""Pluggable chat LLM providers, called over plain HTTP (no vendor SDKs needed).

Supported:
  * Ollama (local)                       CODENAV_LLM_PROVIDER=ollama
  * Any OpenAI-compatible endpoint       CODENAV_LLM_PROVIDER=openai
  * Anthropic Claude API                 CODENAV_LLM_PROVIDER=anthropic
  * None (offline, deterministic)        CODENAV_LLM_PROVIDER=none  (default)
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any

import requests

from ..config import Settings
from ..utils import extract_json, log


class LLMError(RuntimeError):
    pass


class LLM(ABC):
    provider = "base"

    def __init__(self, model: str, base_url: str = "", api_key: str | None = None,
                 temperature: float = 0.1, max_tokens: int = 1500, timeout: int = 240):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.calls = 0

    @abstractmethod
    def _complete(self, prompt: str, system: str | None, max_tokens: int,
                  temperature: float, json_mode: bool) -> str: ...

    def complete(self, prompt: str, system: str | None = None, max_tokens: int | None = None,
                 temperature: float | None = None, json_mode: bool = False, retries: int = 2) -> str:
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                self.calls += 1
                return self._complete(
                    prompt, system,
                    max_tokens or self.max_tokens,
                    self.temperature if temperature is None else temperature,
                    json_mode,
                ).strip()
            except (requests.RequestException, LLMError) as exc:
                last_err = exc
                if any(code in str(exc) for code in ("HTTP 400", "HTTP 401", "HTTP 403", "HTTP 404", "not set")):
                    break  # configuration problem - retrying will not help
                if attempt < retries:
                    wait = 2 ** attempt
                    log.warning("LLM call failed (%s), retrying in %ss", exc, wait)
                    time.sleep(wait)
        raise LLMError(f"{self.provider} LLM call failed: {last_err}") from last_err

    def complete_json(self, prompt: str, system: str | None = None, default: Any = None,
                      max_tokens: int = 600) -> Any:
        text = self.complete(prompt, system=system, max_tokens=max_tokens, temperature=0.0, json_mode=True)
        return extract_json(text, default)

    def describe(self) -> str:
        return f"{self.provider}:{self.model}"

    # shared helper
    def _post(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        resp = requests.post(url, json=payload, headers=headers or {}, timeout=self.timeout)
        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code} from {url}: {resp.text[:400]}")
        return resp.json()


class OllamaLLM(LLM):
    provider = "ollama"

    def _complete(self, prompt, system, max_tokens, temperature, json_mode):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens,
                        "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "16384"))},
        }
        if json_mode:
            payload["format"] = "json"
        data = self._post(f"{self.base_url}/api/chat", payload)
        return data.get("message", {}).get("content", "")


class OpenAICompatibleLLM(LLM):
    provider = "openai"

    def _complete(self, prompt, system, max_tokens, temperature, json_mode):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        data = self._post(f"{self.base_url}/chat/completions", payload, headers)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected response: {str(data)[:300]}") from exc


class AnthropicLLM(LLM):
    provider = "anthropic"

    def _complete(self, prompt, system, max_tokens, temperature, json_mode):
        if not self.api_key:
            raise LLMError("LLM_API_KEY (or ANTHROPIC_API_KEY) is not set in .env")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        url = f"{self.base_url}/v1/messages"
        try:
            data = self._post(url, payload, headers)
        except LLMError as exc:
            # some newer models only accept their default sampling settings
            if "HTTP 400" in str(exc) and "temperature" in str(exc):
                payload.pop("temperature", None)
                data = self._post(url, payload, headers)
            else:
                raise
        if data.get("stop_reason") == "refusal":
            raise LLMError("model declined the request")
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


class MockLLM(LLM):
    """Offline stand-in. Returns '{}' for JSON requests so callers use their
    heuristic fallbacks, and a context digest for normal completions."""
    provider = "mock"

    def _complete(self, prompt, system, max_tokens, temperature, json_mode):
        if json_mode:
            return "{}"
        return "(no LLM configured - set CODENAV_LLM_PROVIDER to get real answers)\n\n" + prompt[:1500]


def get_llm(settings: Settings) -> LLM:
    cls = {
        "ollama": OllamaLLM,
        "openai": OpenAICompatibleLLM,
        "anthropic": AnthropicLLM,
        "mock": MockLLM,
        "none": MockLLM,
    }.get(settings.llm_provider)
    if cls is None:
        raise ValueError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'")
    return cls(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
    )
