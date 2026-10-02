"""
Tests for the provider-agnostic LLM layer.

These cover the contract every adapter must honour (correct wire format,
correct auth placement, useful errors) rather than any single vendor.
"""
import pytest

from app.providers.llm import (
    DESCRIPTORS,
    LLMConfig,
    LLMConfigError,
    LLMError,
    ChatMessage,
    create_llm_provider,
    describe_providers,
)
from app.providers.llm.anthropic import AnthropicProvider
from app.providers.llm.base import extract_json_object
from app.providers.llm.gemini import GeminiProvider
from app.providers.llm.ollama import OllamaProvider
from app.providers.llm.openai_compatible import GroqProvider, OpenAIProvider, XAIProvider

MESSAGES = [
    ChatMessage(role="system", content="You are a test."),
    ChatMessage(role="user", content="Say hi."),
]


# --------------------------------------------------------------------------
# Registry and configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_name,expected_cls",
    [
        ("openai", OpenAIProvider),
        ("anthropic", AnthropicProvider),
        ("gemini", GeminiProvider),
        ("ollama", OllamaProvider),
        ("groq", GroqProvider),
        ("xai", XAIProvider),
    ],
)
def test_registry_resolves_each_provider(provider_name, expected_cls):
    config = LLMConfig(provider=provider_name, model="m", api_key="k")
    assert isinstance(create_llm_provider(config), expected_cls)


def test_unknown_provider_lists_supported_options():
    config = LLMConfig(provider="not-a-provider", model="m", api_key="k")
    with pytest.raises(LLMConfigError) as exc:
        create_llm_provider(config)
    assert "openai" in str(exc.value)


def test_api_key_required_when_provider_demands_one():
    with pytest.raises(LLMConfigError, match="requires an API key"):
        create_llm_provider(LLMConfig(provider="openai", model="gpt-4.1-mini"))


def test_local_provider_needs_no_api_key():
    provider = create_llm_provider(LLMConfig(provider="ollama", model="llama3.1"))
    assert provider.requires_api_key is False


def test_model_is_required():
    with pytest.raises(LLMConfigError, match="requires a model"):
        create_llm_provider(LLMConfig(provider="openai", model="", api_key="k"))


def test_every_registered_provider_has_a_descriptor():
    described = {d["name"] for d in describe_providers()}
    assert described == set(DESCRIPTORS)


def test_descriptors_only_expose_relevant_fields():
    """Onboarding must not show an API key box for a provider that has none."""
    ollama = next(d for d in describe_providers() if d["name"] == "ollama")
    assert "api_key" not in {f["key"] for f in ollama["fields"]}


# --------------------------------------------------------------------------
# Wire format
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_sends_chat_completions_and_returns_content(httpx_mock):
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json={"choices": [{"message": {"content": "hello"}}]},
    )
    provider = create_llm_provider(
        LLMConfig(provider="openai", model="gpt-4.1-mini", api_key="sk-test")
    )

    assert await provider.complete(MESSAGES, json_mode=True) == "hello"

    request = httpx_mock.get_requests()[0]
    assert request.headers["Authorization"] == "Bearer sk-test"
    import json as _json

    body = _json.loads(request.content)
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_anthropic_splits_system_prompt_and_always_sends_max_tokens(httpx_mock):
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        json={"content": [{"type": "text", "text": "hi"}]},
    )
    provider = create_llm_provider(
        LLMConfig(provider="anthropic", model="claude-sonnet-4-5", api_key="key")
    )

    assert await provider.complete(MESSAGES) == "hi"

    import json as _json

    request = httpx_mock.get_requests()[0]
    body = _json.loads(request.content)
    assert body["system"] == "You are a test."
    assert [m["role"] for m in body["messages"]] == ["user"]
    assert body["max_tokens"] > 0
    assert request.headers["x-api-key"] == "key"


@pytest.mark.asyncio
async def test_gemini_maps_assistant_to_model_and_keeps_key_out_of_url(httpx_mock):
    httpx_mock.add_response(
        json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]},
    )
    provider = create_llm_provider(
        LLMConfig(provider="gemini", model="gemini-2.5-flash", api_key="secret")
    )

    result = await provider.complete(
        [ChatMessage(role="user", content="a"), ChatMessage(role="assistant", content="b")]
    )
    assert result == "ok"

    import json as _json

    request = httpx_mock.get_requests()[0]
    assert "secret" not in str(request.url)
    assert request.headers["x-goog-api-key"] == "secret"
    body = _json.loads(request.content)
    assert [c["role"] for c in body["contents"]] == ["user", "model"]


@pytest.mark.asyncio
async def test_ollama_requests_json_format_when_asked(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:11434/api/chat",
        json={"message": {"content": "{}"}},
    )
    provider = create_llm_provider(LLMConfig(provider="ollama", model="llama3.1"))

    await provider.complete(MESSAGES, json_mode=True)

    import json as _json

    body = _json.loads(httpx_mock.get_requests()[0].content)
    assert body["format"] == "json"
    assert body["stream"] is False


@pytest.mark.asyncio
async def test_http_error_surfaces_provider_label_and_status(httpx_mock):
    httpx_mock.add_response(status_code=401, text="bad key")
    provider = create_llm_provider(
        LLMConfig(provider="openai", model="gpt-4.1-mini", api_key="nope")
    )

    with pytest.raises(LLMError) as exc:
        await provider.complete(MESSAGES)
    assert "401" in str(exc.value)
    assert "OpenAI" in str(exc.value)


# --------------------------------------------------------------------------
# Health checks
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_health_check_flags_unpulled_model(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:11434/api/tags",
        json={"models": [{"name": "mistral"}]},
    )
    provider = create_llm_provider(LLMConfig(provider="ollama", model="llama3.1"))

    status = await provider.health_check()
    assert status.ok is False
    assert "ollama pull llama3.1" in status.detail


@pytest.mark.asyncio
async def test_groq_health_check_flags_a_model_the_provider_does_not_have(httpx_mock):
    """
    Seen in QA: Test connection said "Connected." for a decommissioned Groq
    model, and the first meeting was then processed without AI. The key
    being valid is not the same as the model existing.
    """
    httpx_mock.add_response(
        url="https://api.groq.com/openai/v1/models",
        json={"data": [{"id": "openai/gpt-oss-120b"}, {"id": "llama-3.3-70b-versatile"}, {"id": "whisper-large-v3"}]},
    )
    provider = create_llm_provider(LLMConfig(provider="groq", model="llama-3.1-8b-instant", api_key="gsk_x"))
    status = await provider.health_check()
    assert status.ok is False
    assert "llama-3.1-8b-instant" in status.detail and "Available" in status.detail
    assert "whisper-large-v3" not in status.models  # picker still hides non-chat models

    httpx_mock.add_response(
        url="https://api.groq.com/openai/v1/models",
        json={"data": [{"id": "openai/gpt-oss-120b"}, {"id": "whisper-large-v3"}]},
    )
    provider = create_llm_provider(LLMConfig(provider="groq", model="openai/gpt-oss-120b", api_key="gsk_x"))
    assert (await provider.health_check()).ok is True

    # A model the picker hides (matches a non-chat marker) but the provider
    # does list is still accepted - the check is against everything listed.
    httpx_mock.add_response(
        url="https://api.groq.com/openai/v1/models",
        json={"data": [{"id": "llama-guard-3-8b"}]},
    )
    provider = create_llm_provider(LLMConfig(provider="groq", model="llama-guard-3-8b", api_key="gsk_x"))
    assert (await provider.health_check()).ok is True


@pytest.mark.asyncio
async def test_anthropic_and_gemini_health_checks_verify_the_model(httpx_mock):
    httpx_mock.add_response(url="https://api.anthropic.com/v1/models", json={"data": [{"id": "claude-sonnet-5"}]})
    provider = create_llm_provider(LLMConfig(provider="anthropic", model="claude-2", api_key="sk-ant"))
    status = await provider.health_check()
    assert status.ok is False and "claude-2" in status.detail

    httpx_mock.add_response(
        url="https://generativelanguage.googleapis.com/v1beta/models",
        json={"models": [{"name": "models/gemini-2.5-flash"}]},
    )
    provider = create_llm_provider(LLMConfig(provider="gemini", model="gemini-2.5-flash", api_key="g"))
    assert (await provider.health_check()).ok is True


@pytest.mark.asyncio
async def test_health_check_reports_unreachable_host_without_raising(httpx_mock):
    import httpx as _httpx

    httpx_mock.add_exception(_httpx.ConnectError("refused"))
    provider = create_llm_provider(LLMConfig(provider="ollama", model="llama3.1"))

    status = await provider.health_check()
    assert status.ok is False
    assert "ollama serve" in status.detail


# --------------------------------------------------------------------------
# JSON recovery
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Sure! Here you go:\n{"a": 1}\nHope that helps.',
    ],
)
def test_extract_json_object_recovers_wrapped_output(raw):
    assert extract_json_object(raw) == {"a": 1}


@pytest.mark.parametrize("raw", ["", "no json here", "[1, 2, 3]"])
def test_extract_json_object_returns_none_when_no_object(raw):
    assert extract_json_object(raw) is None


@pytest.mark.asyncio
async def test_complete_json_raises_when_model_returns_prose(httpx_mock):
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json={"choices": [{"message": {"content": "I cannot do that."}}]},
    )
    provider = create_llm_provider(
        LLMConfig(provider="openai", model="gpt-4.1-mini", api_key="sk-test")
    )

    with pytest.raises(LLMError, match="parseable JSON"):
        await provider.complete_json(MESSAGES)


def test_xai_is_distinct_from_groq_with_its_own_base_url():
    """xAI (Grok) and Groq are different vendors; each must keep its own host."""
    xai = create_llm_provider(LLMConfig(provider="xai", model="grok-3-mini", api_key="k"))
    groq = create_llm_provider(LLMConfig(provider="groq", model="llama-3.1-8b", api_key="k"))
    assert isinstance(xai, XAIProvider) and xai.default_base_url == "https://api.x.ai/v1"
    assert isinstance(groq, GroqProvider) and groq.default_base_url == "https://api.groq.com/openai/v1"
    described = {d["name"]: d for d in describe_providers()}
    assert "console.x.ai" in described["xai"]["fields"][0]["help"]


@pytest.mark.asyncio
async def test_a_page_that_is_not_json_is_a_provider_error(httpx_mock):
    """
    A 200 carrying an HTML page - a proxy's error page, or a base URL that
    points at a website - escaped as JSONDecodeError and failed the meeting
    instead of taking the rule-based fallback.
    """
    httpx_mock.add_response(url="http://localhost:11434/api/chat", content=b"<html>proxy error</html>",
                            headers={"content-type": "text/html"})
    provider = create_llm_provider(LLMConfig(provider="ollama", model="m"))
    with pytest.raises(LLMError, match="not JSON"):
        await provider.complete([ChatMessage(role="user", content="hi")])


@pytest.mark.asyncio
async def test_a_server_that_refuses_json_mode_is_asked_again_without_it(httpx_mock):
    """
    LM Studio, older vLLM and llama.cpp builds reject response_format; every
    meeting on them went to the rule-based fallback after a single 400.
    """
    import json

    url = "http://localhost:1234/v1/chat/completions"
    httpx_mock.add_response(url=url, status_code=400, json={"error": "'response_format.type' must be 'json_schema'"})
    httpx_mock.add_response(url=url, json={"choices": [{"message": {"content": 'Here it is: {"summary": "ok"}'}}]})
    provider = create_llm_provider(LLMConfig(provider="openai_compatible", model="m", base_url="http://localhost:1234/v1"))

    assert await provider.complete_json([ChatMessage(role="user", content="extract")]) == {"summary": "ok"}
    first, second = (json.loads(r.content) for r in httpx_mock.get_requests())
    assert "response_format" in first and "response_format" not in second
