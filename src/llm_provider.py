"""Pluggable LLM provider abstraction.

Supports Ollama (local) and Google Gemini (cloud).  The active provider is
selected automatically based on environment variables, or explicitly via
config.yaml ``ai.provider``.

Selection logic (when provider is "auto" or omitted):
  1. If GOOGLE_API_KEY is set -> Gemini
  2. Otherwise -> Ollama
"""

from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Iterator

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>")

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds: 2, 4, 8


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by some models (e.g. Qwen3)."""
    return _THINK_RE.sub("", text).strip()


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception is a transient error worth retrying."""
    msg = str(exc).lower()
    for keyword in (
        "429",
        "rate",
        "resource_exhausted",
        "503",
        "unavailable",
        "timeout",
        "timed out",
        "connection",
    ):
        if keyword in msg:
            return True
    return False


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Extract retry delay from error message or use exponential backoff."""
    msg = str(exc)
    import re as _re

    match = _re.search(r"retry\s*(?:in|after)\s*(\d+(?:\.\d+)?)", msg, _re.IGNORECASE)
    if match:
        return max(float(match.group(1)), 1.0)
    return _BACKOFF_BASE * (2**attempt)


# ── Abstract base ────────────────────────────────────────────────────


class LLMProvider(ABC):
    """Abstract base for LLM backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name (e.g. ``'ollama'``, ``'gemini'``)."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier in use."""

    @abstractmethod
    def _do_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Provider-specific chat implementation (no retry logic)."""

    @abstractmethod
    def _do_chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Provider-specific streaming implementation (no retry logic)."""

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request with automatic retry on transient errors."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._do_chat(messages, temperature, max_tokens)
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES and _is_retryable(exc):
                    wait = _retry_delay(exc, attempt)
                    logger.warning(
                        "%s: retryable error (attempt %d/%d), waiting %.1fs: %s",
                        self.name,
                        attempt + 1,
                        _MAX_RETRIES,
                        wait,
                        str(exc)[:120],
                    )
                    time.sleep(wait)
                    continue
                raise
        raise last_exc  # type: ignore[misc]  # unreachable but satisfies type checker

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stream a chat completion with automatic retry on transient errors."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                yield from self._do_chat_stream(messages, temperature, max_tokens)
                return
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES and _is_retryable(exc):
                    wait = _retry_delay(exc, attempt)
                    logger.warning(
                        "%s stream: retryable error (attempt %d/%d), waiting %.1fs",
                        self.name,
                        attempt + 1,
                        _MAX_RETRIES,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                raise
        raise last_exc  # type: ignore[misc]

    def warmup(self) -> None:
        """Optional model warm-up (no-op by default)."""


# ── Ollama ───────────────────────────────────────────────────────────


class OllamaProvider(LLMProvider):
    """Ollama local LLM backend."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str | None = None,
        timeout: float = 120,
    ):
        import ollama

        self._model = model
        self._client = ollama.Client(host=host, timeout=timeout)

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    def _do_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        options: dict = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        response = self._client.chat(
            model=self._model,
            messages=messages,
            options=options,
        )
        return strip_think_tags(response["message"]["content"].strip())

    def _do_chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        options: dict = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        stream = self._client.chat(
            model=self._model,
            messages=messages,
            options=options,
            stream=True,
        )
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token

    def warmup(self) -> None:
        """Load the model into GPU/RAM with a tiny request."""
        try:
            self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": "Reply with OK."},
                    {"role": "user", "content": "OK"},
                ],
                options={"temperature": 0, "num_predict": 5},
            )
        except Exception:
            pass


# ── Google Gemini ────────────────────────────────────────────────────


class GeminiProvider(LLMProvider):
    """Google Gemini (AI Studio) backend with built-in rate limiting."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        requests_per_minute: int = 14,
    ):
        from google import genai  # noqa: F811

        key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY is required for the Gemini provider. "
                "Get one at https://aistudio.google.com/apikey"
            )
        self._model = model
        self._client = genai.Client(api_key=key)
        self._rpm = requests_per_minute
        self._request_times: list[float] = []

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return self._model

    def _throttle(self) -> None:
        """Sleep if necessary to stay within the requests-per-minute limit."""
        if not self._rpm:
            return
        now = time.monotonic()
        window = 60.0
        self._request_times = [t for t in self._request_times if now - t < window]
        if len(self._request_times) >= self._rpm:
            oldest = self._request_times[0]
            sleep_for = window - (now - oldest) + 0.5
            if sleep_for > 0:
                logger.info("Gemini rate limit: sleeping %.1fs", sleep_for)
                time.sleep(sleep_for)
        self._request_times.append(time.monotonic())

    def _do_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        from google.genai import types

        self._throttle()
        system_text, contents = self._split_messages(messages)
        config = types.GenerateContentConfig(temperature=temperature)
        if max_tokens is not None:
            config.max_output_tokens = max_tokens
        if system_text:
            config.system_instruction = system_text

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        return (response.text or "").strip()

    def _do_chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        from google.genai import types

        self._throttle()
        system_text, contents = self._split_messages(messages)
        config = types.GenerateContentConfig(temperature=temperature)
        if max_tokens is not None:
            config.max_output_tokens = max_tokens
        if system_text:
            config.system_instruction = system_text

        for chunk in self._client.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                yield chunk.text

    @staticmethod
    def _split_messages(
        messages: list[dict[str, str]],
    ) -> tuple[str | None, str]:
        """Convert OpenAI-style messages to Gemini's (system_instruction, contents)."""
        system_text: str | None = None
        user_parts: list[str] = []

        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                user_parts.append(msg["content"])

        return system_text, "\n\n".join(user_parts)


# ── Factory ──────────────────────────────────────────────────────────

_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

_OLLAMA_MODEL_PREFIXES = frozenset(
    {
        "qwen",
        "llama",
        "mistral",
        "phi",
        "gemma",
        "deepseek",
        "codellama",
        "vicuna",
        "neural",
        "nous",
        "yi",
        "command",
        "orca",
        "solar",
        "tinyllama",
    }
)


def create_provider(ai_config: dict | None = None) -> LLMProvider:
    """Build the appropriate :class:`LLMProvider` from config + environment.

    Selection order:
    1. Explicit ``ai.provider`` value in config (``ollama`` | ``gemini``).
    2. If ``GOOGLE_API_KEY`` is present -> Gemini.
    3. Fallback -> Ollama.
    """
    ai_cfg = ai_config or {}
    explicit = ai_cfg.get("provider", "auto").lower()
    configured_model = ai_cfg.get("model", "")
    google_key = os.environ.get("GOOGLE_API_KEY", "")
    ollama_host = ai_cfg.get("host") or os.environ.get("OLLAMA_HOST") or None
    timeout = ai_cfg.get("timeout", 120)
    if isinstance(timeout, (int, float)) and timeout > 600:
        timeout = timeout / 1000  # likely milliseconds, convert to seconds

    use_gemini = explicit == "gemini" or (explicit == "auto" and google_key)

    if use_gemini:
        if not google_key:
            raise ValueError(
                "ai.provider is 'gemini' but GOOGLE_API_KEY is not set. "
                "Add it to .env or set ai.provider to 'ollama'."
            )
        gemini_model = ai_cfg.get("gemini_model") or _DEFAULT_GEMINI_MODEL
        if configured_model and not any(
            configured_model.lower().startswith(p) for p in _OLLAMA_MODEL_PREFIXES
        ):
            gemini_model = configured_model

        rpm = ai_cfg.get("requests_per_minute", 14)
        logger.info("Using Gemini provider (model: %s)", gemini_model)
        print(f"  AI provider: Gemini ({gemini_model})")
        return GeminiProvider(
            model=gemini_model,
            api_key=google_key,
            requests_per_minute=rpm,
        )

    model = configured_model or "qwen3:8b"
    logger.info("Using Ollama provider (model: %s, host: %s)", model, ollama_host)
    print(f"  AI provider: Ollama ({model})")
    return OllamaProvider(model=model, host=ollama_host, timeout=timeout)
