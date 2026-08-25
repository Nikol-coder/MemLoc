# -*- coding: utf-8 -*-
"""Shared OpenAI-compatible LLM client with retry logic.

All generation / extraction / judging scripts in MemLoc talk to an
OpenAI-compatible endpoint (Azure OpenAI, OpenAI, or a local vLLM /
sglang server).  Configuration comes entirely from the environment:

* ``OPENAI_API_KEY``  -- API key (``"EMPTY"`` for local servers)
* ``OPENAI_BASE_URL`` -- endpoint URL, e.g. ``http://localhost:8000/v1``

The pipeline scripts only ever call ``LLMClient.chat`` /
``create_client``, so any OpenAI-compatible backend can be plugged in
without touching the pipeline code.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import backoff
from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    NotFoundError,
    InternalServerError,
)

logger = logging.getLogger(__name__)

RETRYABLE_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    NotFoundError,
    InternalServerError,
)


def create_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> OpenAI:
    """Initialize an OpenAI-compatible client.

    Args:
        base_url: endpoint URL; falls back to ``OPENAI_BASE_URL``.
        api_key: API key; falls back to ``OPENAI_API_KEY`` (default
            ``"EMPTY"``, suitable for local vLLM / sglang servers).
    """
    return OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL", None),
    )


def _message_text(completion) -> str:
    """Prefer ``content``; fall back to reasoning fields (Qwen3.x)."""
    msg = completion.choices[0].message
    text = msg.content
    if isinstance(text, list):
        text = "".join(
            p.get("text", "")
            for p in text
            if isinstance(p, dict) and p.get("type") == "text"
        )
    text = (text or "").strip()
    if text:
        return text
    for attr in ("reasoning_content", "reasoning"):
        extra = getattr(msg, attr, None)
        if extra:
            return str(extra).strip()
    return ""


class LLMClient:
    """Thin wrapper around chat completions with exponential backoff."""

    def __init__(
        self,
        model: str,
        client: Optional[OpenAI] = None,
        temperature: float = 0.1,
        max_tokens: int = 4000,
        timeout: int = 180,
        max_tries: int = 5,
    ):
        self.model = model
        self.client = client or create_client()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_tries = max_tries

    @backoff.on_exception(backoff.expo, RETRYABLE_ERRORS, max_tries=5)
    def _chat_with_backoff(self, **kwargs) -> Any:
        return self.client.chat.completions.create(**kwargs)

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Single-turn chat completion.  Returns the response text or ``""``
        when the call permanently failed."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        try:
            completion = self._chat_with_backoff(**kwargs)
            return _message_text(completion)
        except Exception as exc:  # noqa: BLE001 - keep the pipeline running
            logger.error("ChatCompletion failed: %s", exc)
            return ""
