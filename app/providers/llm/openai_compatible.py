"""
Adapters for the OpenAI chat-completions wire format.

This format is the de-facto standard: OpenAI, Groq, together.ai, vLLM,
LM Studio, OpenRouter and most self-hosted gateways all speak it. Rather than
carrying a vendor SDK per service, one HTTP adapter covers all of them, and
"bring your own OpenAI-compatible endpoint" becomes a first-class option.
"""
from __future__ import annotations

from typing import List, Optional

import httpx

from app.providers.llm.base import (
    ChatMessage,
    LLMError,
    LLMProvider,
    ProviderStatus,
)


class OpenAICompatibleProvider(LLMProvider):
    name = "openai_compatible"
    label = "OpenAI-compatible API"
    supports_json_mode = True
    requires_api_key = False
    default_base_url = "http://localhost:8000/v1"

    async def _complete(
        self,
        messages: List[ChatMessage],
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._temperature(temperature),
        }

        resolved_max_tokens = self._max_tokens(max_tokens)
        if resolved_max_tokens:
            payload["max_tokens"] = resolved_max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            if json_mode and response.status_code in (400, 422):
                # Plenty of servers that speak this format reject
                # response_format (LM Studio wants a json_schema, older vLLM
                # and llama.cpp builds refuse it outright). That used to send
                # every meeting to the rule-based fallback; asking again
                # without it lets complete_json() recover the JSON from text.
                payload.pop("response_format")
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
            _raise_for_status(response, self.label)
            data = response.json()

        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.label} returned an unexpected response shape.") from exc

    async def health_check(self) -> ProviderStatus:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(f"{self.base_url}/models", headers=self._headers())
                _raise_for_status(response, self.label)
                data = response.json()
        except LLMError as exc:
            return ProviderStatus(ok=False, detail=str(exc))
        except httpx.HTTPError as exc:
            return ProviderStatus(ok=False, detail=f"Could not reach {self.base_url}: {exc}")

        listed = [str(item.get("id")) for item in data.get("data", []) if isinstance(item, dict) and item.get("id")]
        models = sorted(model_id for model_id in listed if _is_chat_model(model_id))
        return self._connected(models, listed)

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers


#: Model ids that a /models listing includes but that cannot hold a chat:
#: speech, moderation, embedding and image models.
_NON_CHAT_MARKERS = ("whisper", "tts", "orpheus", "guard", "embed", "moderation", "dall-e", "image", "safeguard", "rerank")


def _is_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(marker in lowered for marker in _NON_CHAT_MARKERS)


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"
    label = "OpenAI"
    requires_api_key = True
    default_base_url = "https://api.openai.com/v1"


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"
    label = "Groq"
    requires_api_key = True
    default_base_url = "https://api.groq.com/openai/v1"


class XAIProvider(OpenAICompatibleProvider):
    """xAI's Grok models. Distinct from Groq (console.groq.com) despite the
    near-identical name: this is x.ai, and it speaks the OpenAI format."""

    name = "xai"
    label = "xAI (Grok)"
    requires_api_key = True
    default_base_url = "https://api.x.ai/v1"


def _raise_for_status(response: httpx.Response, label: str) -> None:
    if response.is_success:
        return
    detail = response.text.strip()
    if len(detail) > 500:
        detail = f"{detail[:500]}…"
    raise LLMError(f"{label} request failed ({response.status_code}): {detail}", status_code=response.status_code)
