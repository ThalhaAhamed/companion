"""Adapter for the Anthropic Messages API."""
from __future__ import annotations

from typing import List, Optional

import httpx

from app.providers.llm.base import (
    ChatMessage,
    LLMError,
    LLMProvider,
    ProviderStatus,
)

ANTHROPIC_VERSION = "2023-06-01"
# The Messages API rejects requests without max_tokens, so a ceiling is always
# sent even when the user has not configured one.
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    label = "Anthropic"
    # Claude has no response_format switch; JSON is coaxed via prompting and
    # recovered by the base class's extractor.
    supports_json_mode = False
    requires_api_key = True
    default_base_url = "https://api.anthropic.com"

    async def _complete(
        self,
        messages: List[ChatMessage],
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        system_prompt = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        if not turns:
            raise LLMError("Anthropic requires at least one user message.")

        payload = {
            "model": self.config.model,
            "messages": turns,
            "max_tokens": self._max_tokens(max_tokens) or DEFAULT_MAX_TOKENS,
            "temperature": self._temperature(temperature),
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/v1/messages",
                headers=self._headers(),
                json=payload,
            )
            _raise_for_status(response, self.label)
            data = response.json()

        blocks = data.get("content") or []
        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text:
            raise LLMError(f"{self.label} returned no text content.")
        return text

    async def health_check(self) -> ProviderStatus:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(f"{self.base_url}/v1/models", headers=self._headers())
                _raise_for_status(response, self.label)
                data = response.json()
        except LLMError as exc:
            return ProviderStatus(ok=False, detail=str(exc))
        except httpx.HTTPError as exc:
            return ProviderStatus(ok=False, detail=f"Could not reach {self.base_url}: {exc}")

        models = sorted(
            str(item.get("id"))
            for item in data.get("data", [])
            if isinstance(item, dict) and item.get("id")
        )
        return self._connected(models)

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key or "",
            "anthropic-version": ANTHROPIC_VERSION,
        }


def _raise_for_status(response: httpx.Response, label: str) -> None:
    if response.is_success:
        return
    detail = response.text.strip()
    if len(detail) > 500:
        detail = f"{detail[:500]}…"
    raise LLMError(f"{label} request failed ({response.status_code}): {detail}", status_code=response.status_code)
