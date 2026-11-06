"""
Resolves the currently configured LLM provider for the application.

This is the single place that knows how environment settings map onto an
LLMConfig. Everything else asks for a provider and gets one, so swapping
vendors never touches application code.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from app.config import settings
from app.runtime_config import load_config, resolve
from app.providers.llm import (
    DESCRIPTORS,
    LLMConfig,
    LLMConfigError,
    LLMProvider,
    create_llm_provider,
)

# Settings names kept from before provider-agnostic configuration existed, so
# an existing .env keeps working without edits.
_LEGACY_API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "xai": "XAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
_LEGACY_MODELS = {
    "openai": "OPENAI_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
    "groq": "GROQ_MODEL",
    "xai": "XAI_MODEL",
    "gemini": "GEMINI_MODEL",
}


#: Key under Organization.settings holding a workspace's own AI choice.
WORKSPACE_LLM_KEY = "llm"
#: Each member's install decides (the default), or the owner sets one for all.
AI_MODE_MEMBER = "member"
AI_MODE_WORKSPACE = "workspace"


def workspace_llm(org) -> Optional[Dict[str, Any]]:
    """
    The workspace's own provider settings, when its owner chose to unify
    them; None means every member's install uses what it has configured.

    Kept as a plain dict on Organization.settings: the AI choice is a
    property of the workspace, so it has to travel with the database that
    holds the workspace - a member on another machine reads it from there.
    """
    raw = ((getattr(org, "settings", None) or {}).get(WORKSPACE_LLM_KEY)) or {}
    return raw if raw.get("mode") == AI_MODE_WORKSPACE else None


def build_llm_config(workspace: Optional[Dict[str, Any]] = None) -> LLMConfig:
    """
    Assemble an LLMConfig from the effective configuration.

    With a workspace's own settings (see workspace_llm), those are used as
    they are - the owner chose them for everyone, so this install's
    environment and saved values do not apply. Otherwise precedence is
    environment variable, then the value saved during onboarding, then the
    provider's documented default. Legacy per-provider variable names are
    still honoured at the bottom of the chain.
    """
    if workspace is not None:
        return _build_workspace_config(workspace)
    stored = load_config().llm

    provider = str(resolve("LLM_PROVIDER", stored.provider, settings.LLM_PROVIDER) or "")
    provider = provider.strip().lower()
    if not provider:
        raise LLMConfigError("No LLM provider configured.")

    descriptor = DESCRIPTORS.get(provider)
    if descriptor is None:
        supported = ", ".join(sorted(DESCRIPTORS))
        raise LLMConfigError(
            f"Unknown LLM provider '{provider}'. Supported providers: {supported}."
        )

    api_key = resolve(
        "LLM_API_KEY",
        stored.api_key,
        settings.LLM_API_KEY or _legacy(_LEGACY_API_KEYS.get(provider)),
    )
    model = resolve(
        "LLM_MODEL",
        stored.model,
        settings.LLM_MODEL
        or _legacy(_LEGACY_MODELS.get(provider))
        or _default_field(descriptor, "model"),
    )
    base_url = resolve(
        "LLM_BASE_URL",
        stored.base_url,
        settings.LLM_BASE_URL or _default_field(descriptor, "base_url"),
    )

    temperature = stored.temperature
    if temperature is None:
        temperature = settings.LLM_TEMPERATURE

    return LLMConfig(
        provider=provider,
        model=str(model or ""),
        api_key=api_key,
        base_url=base_url,
        temperature=float(temperature),
        max_tokens=stored.max_tokens or settings.LLM_MAX_TOKENS,
    )


def _build_workspace_config(raw: Dict[str, Any]) -> LLMConfig:
    provider = str(raw.get("provider") or "").strip().lower()
    if not provider:
        raise LLMConfigError(
            "This workspace uses one AI provider for everyone, but its owner has not set it up yet "
            "(Members page, AI provider)."
        )
    descriptor = DESCRIPTORS.get(provider)
    if descriptor is None:
        supported = ", ".join(sorted(DESCRIPTORS))
        raise LLMConfigError(f"Unknown LLM provider '{provider}'. Supported providers: {supported}.")
    temperature = raw.get("temperature")
    return LLMConfig(
        provider=provider,
        model=str(raw.get("model") or _default_field(descriptor, "model") or ""),
        api_key=raw.get("api_key") or None,
        base_url=raw.get("base_url") or _default_field(descriptor, "base_url"),
        temperature=float(settings.LLM_TEMPERATURE if temperature is None else temperature),
        max_tokens=raw.get("max_tokens") or settings.LLM_MAX_TOKENS,
        # A workspace owner is anyone who signed up, not the person who runs
        # the server: whether their base URL may reach the server's private
        # network is the operator's call.
        allow_private_network=settings.ALLOW_PRIVATE_LLM_BASE_URL,
    )


def get_llm_provider(workspace: Optional[Dict[str, Any]] = None) -> LLMProvider:
    """The configured provider, ready to use. Raises LLMConfigError if unusable."""
    return create_llm_provider(build_llm_config(workspace))


def try_get_llm_provider(workspace: Optional[Dict[str, Any]] = None) -> Optional[LLMProvider]:
    """
    The configured provider, or None when configuration is absent or invalid.

    Used by callers that have a non-LLM fallback path and should degrade
    rather than fail outright.
    """
    try:
        return get_llm_provider(workspace)
    except LLMConfigError:
        return None


async def provider_for_workspace(org_id, db) -> LLMProvider:
    """
    The provider a request in this workspace should use: the workspace's
    own if the owner unified it, otherwise this install's. Raises
    LLMConfigError if unusable.
    """
    from app import permissions as perms

    org = await perms.load_org(org_id, db)
    return get_llm_provider(workspace_llm(org))


def _legacy(setting_name: Optional[str]) -> Optional[str]:
    if not setting_name:
        return None
    return getattr(settings, setting_name, None)


def _default_field(descriptor, key: str) -> Optional[str]:
    for field_spec in descriptor.fields:
        if field_spec.key == key and field_spec.default:
            return str(field_spec.default)
    return None
