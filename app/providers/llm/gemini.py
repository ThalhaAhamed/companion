"""Adapter for the Google Gemini generateContent API."""
from __future__ import annotations

from typing import List, Optional

import httpx

from app.providers.llm.base import (
    ChatMessage,
    LLMError,
    LLMProvider,
    ProviderStatus,
)


class GeminiProvider(LLMProvider):
    name = "gemini"
    label = "Google Gemini"
    supports_json_mode = True
    requires_api_key = True
    default_base_url = "https://generativelanguage.googleapis.com"

    async def _complete(
        self,
        messages: List[ChatMessage],
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        system_prompt = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = [
            {
                # Gemini names the assistant turn "model" rather than "assistant".
                "role": "model" if m.role == "assistant" else "user",
                "parts": [{"text": m.content}],
            }
            for m in messages
            if m.role in ("user", "assistant")
        ]
        if not contents:
            raise LLMError("Gemini requires at least one user message.")

        generation_config = {"temperature": self._temperature(temperature)}
        resolved_max_tokens = self._max_tokens(max_tokens)
        if resolved_max_tokens:
            generation_config["maxOutputTokens"] = resolved_max_tokens
        if json_mode:
            generation_config["responseMimeType"] = "application/json"

        payload = {"contents": contents, "generationConfig": generation_config}
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        url = f"{self.base_url}/v1beta/models/{self.config.model}:generateContent"
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(url, headers=self._headers(), json=payload)
            _raise_for_status(response, self.label)
            data = response.json()

        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError(f"{self.label} returned no candidates.")
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        if not text:
            raise LLMError(f"{self.label} returned no text content.")
        return text

    async def health_check(self) -> ProviderStatus:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{self.base_url}/v1beta/models", headers=self._headers()
                )
                _raise_for_status(response, self.label)
                data = response.json()
        except LLMError as exc:
            return ProviderStatus(ok=False, detail=str(exc))
        except httpx.HTTPError as exc:
            return ProviderStatus(ok=False, detail=f"Could not reach {self.base_url}: {exc}")

        models = sorted(
            str(item.get("name", "")).removeprefix("models/")
            for item in data.get("models", [])
            if isinstance(item, dict) and item.get("name")
        )
        return self._connected(models)

    def _headers(self) -> dict:
        # Sent as a header rather than the ?key= query parameter documented by
        # Google, so the secret never lands in request logs or proxy URLs.
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.config.api_key or "",
        }


def _raise_for_status(response: httpx.Response, label: str) -> None:
    if response.is_success:
        return
    detail = response.text.strip()
    if len(detail) > 500:
        detail = f"{detail[:500]}…"
    raise LLMError(f"{label} request failed ({response.status_code}): {detail}", status_code=response.status_code)
