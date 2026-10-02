"""
Ask AI context and citations.

Each test pins a failure measured on a ground-truth benchmark of dated,
conflicting meetings (see app/services/ask.py's module docstring): the
2,000-character note cut, relevance-ordered undated context, "sources"
that were simply everything retrieved, no conversation memory, dated
questions that never reached the right meeting, and Ollama silently
cutting the prompt at its default context window.
"""
import json

import pytest

from app.providers.llm import LLMConfig, LLMProvider, ProviderStatus


class _ScriptedProvider(LLMProvider):
    """Answers with a scripted reply and keeps the prompt it was given."""
    name = "scripted"
    label = "Scripted"
    requires_api_key = False
    default_base_url = "http://scripted"

    def __init__(self, config, reply):
        super().__init__(config)
        self.reply = reply
        self.prompts = []

    async def _complete(self, messages, *, json_mode=False, temperature=None, max_tokens=None):
        self.prompts.append([(m.role, m.content) for m in messages])
        return self.reply(messages[-1].content) if callable(self.reply) else self.reply

    async def health_check(self):
        return ProviderStatus(ok=True, detail="ok")


def _use(monkeypatch, reply):
    import app.api.notebook as notebook

    provider = _ScriptedProvider(LLMConfig(provider="scripted", model="m"), reply)

    async def _provider(org_id, db):
        return provider

    monkeypatch.setattr(notebook, "provider_for_workspace", _provider)
    return provider


async def _note(client, title, content):
    r = await client.post("/api/notebook/notes", json={"title": title, "content": content})
    assert r.status_code == 201, r.text
    return r.json()


def _label_of(prompt, title):
    """The [N1]-style label the prompt gave the item with this title."""
    for line in prompt.splitlines():
        if line.startswith("[") and title in line:
            return line[1:line.index("]")]
    raise AssertionError(f"{title!r} not in prompt")


@pytest.mark.asyncio
async def test_a_fact_deep_in_a_long_note_reaches_the_model(authed_client, monkeypatch):
    provider = _use(monkeypatch, "noted")
    body = "# Quarterly ops review\n\n## Summary\n\n" + "\n".join(
        f"- Item {i}: the {['vendor', 'network', 'payroll', 'office'][i % 4]} workstream reported routine progress." for i in range(90)
    ) + "\n\n## Decisions\n\n- The Zurich data centre migration is scheduled for 3 November, owned by Priya.\n"
    assert len(body) > 4000
    await _note(authed_client, "Quarterly ops review", body)

    await authed_client.post("/api/notebook/ask", json={"question": "When is the Zurich data centre migration?"})
    prompt = provider.prompts[-1][-1][1]
    # It used to be cut at 2,000 characters and the model answered "not found".
    assert "Zurich data centre migration is scheduled for 3 November" in prompt
    assert "# Quarterly ops review" in prompt  # the head is always kept


@pytest.mark.asyncio
async def test_sources_are_only_what_the_answer_cited(authed_client, monkeypatch):
    await _note(authed_client, "Hermes planning", "The Hermes launch date is October 20.")
    await _note(authed_client, "Apollo review", "The Apollo launch is now November 12.")
    await _note(authed_client, "Lunch", "Team lunch is on Thursday.")

    def reply(prompt):
        return f"Hermes launches on October 20 [{_label_of(prompt, 'Hermes planning')}]."

    _use(monkeypatch, reply)
    r = (await authed_client.post("/api/notebook/ask", json={"question": "What is the Hermes launch date?"})).json()
    # The marker becomes the source's name, so the sentence still reads.
    assert r["answer"].startswith("Hermes launches on October 20 (Hermes planning, ")
    assert "[N" not in r["answer"]
    assert [s["title"] for s in r["sources"]] == ["Hermes planning"]


@pytest.mark.asyncio
async def test_a_not_found_answer_cites_nothing(authed_client, monkeypatch):
    await _note(authed_client, "Apollo review", "The Apollo launch is now November 12.")
    _use(monkeypatch, "The notes do not contain anything about Project Zeus.")
    r = (await authed_client.post("/api/notebook/ask", json={"question": "When does Project Zeus launch?"})).json()
    # Seven sources used to sit beside answers like this one.
    assert r["sources"] == [] and r["documents"] == []


@pytest.mark.asyncio
async def test_without_citations_sources_fall_back_to_what_the_answer_says(authed_client, monkeypatch):
    await _note(authed_client, "Hermes planning", "Hermes will use PostgreSQL as its database, owned by Sarah.")
    await _note(authed_client, "Lunch", "Team lunch is on Thursday at the corner cafe.")
    _use(monkeypatch, "Hermes will use PostgreSQL as its database, owned by Sarah.")  # a model that ignores citations
    r = (await authed_client.post("/api/notebook/ask", json={"question": "Which database does Hermes use?"})).json()
    assert [s["title"] for s in r["sources"]] == ["Hermes planning"]


@pytest.mark.asyncio
async def test_context_is_dated_and_oldest_first(authed_client, monkeypatch):
    from datetime import datetime

    from app.services.processing import processing_pipeline
    import uuid

    import app.services.memory as memory

    monkeypatch.setattr(memory, "try_get_llm_provider", lambda workspace=None: None)  # rule-based, offline
    provider = _use(monkeypatch, "ok")
    for title, when, line in (
        ("Apollo review", datetime(2026, 9, 25, 10), "Priya: The Apollo launch is now November 12."),
        ("Apollo kickoff", datetime(2026, 9, 10, 10), "Priya: Apollo will launch on October 15."),
        ("Apollo sync", datetime(2026, 9, 20, 10), "Priya: The Apollo launch moved to November 5."),
    ):
        r = await authed_client.post("/api/meetings/upload", json={"title": title, "transcript": line, "started_at": when.isoformat()})
        await processing_pipeline.wait_for(uuid.UUID(r.json()["id"]))

    await authed_client.post("/api/notebook/ask", json={"question": "What is the current Apollo launch date?"})
    system, prompt = provider.prompts[-1][0][1], provider.prompts[-1][-1][1]
    headers = [line for line in prompt.splitlines() if line.startswith("[N")]
    order = [next(i for i, h in enumerate(headers) if t in h) for t in ("Apollo kickoff", "Apollo sync", "Apollo review")]
    assert order == sorted(order), headers
    assert any("Apollo review (2026-09-25)" in h for h in headers), headers
    assert "most recent one is the current state" in system


@pytest.mark.asyncio
async def test_a_follow_up_question_carries_the_conversation(authed_client, monkeypatch):
    provider = _use(monkeypatch, "Daniel.")
    await _note(authed_client, "Apollo review", "Daniel took over Apollo deployment from Sarah.")
    for n in range(12):
        await _note(authed_client, f"Unrelated {n}", f"Office plants need watering, round {n}.")
    history = [
        {"role": "user", "content": "Who is responsible for Apollo deployment?"},
        {"role": "assistant", "content": "Sarah was, until the review [N1]."},
    ]
    r = await authed_client.post("/api/notebook/ask", json={"question": "Who owns it now?", "history": history})
    assert r.status_code == 200, r.text
    prompt = provider.prompts[-1][-1][1]
    assert "Conversation so far" in prompt and "Who is responsible for Apollo deployment?" in prompt
    assert "[N1]" not in prompt.split("Conversation so far")[1]  # old markers don't leak in
    assert "Daniel took over Apollo deployment" in prompt  # retrieved via the earlier question


@pytest.mark.asyncio
async def test_a_question_naming_a_day_reaches_that_meeting(authed_client, monkeypatch):
    provider = _use(monkeypatch, "ok")
    await _note(authed_client, "2026-09-20 · Apollo sync", "Launch moved to November 5. Budget up to 8 lakh.")
    for n in range(30):
        await _note(authed_client, f"2026-09-{(n % 9) + 10:02d} · Planning {n}",
                    f"In this meeting the team discussed launch dates, budgets and deployment owners, round {n}.")
    await authed_client.post("/api/notebook/ask", json={"question": "What happened in the September 20 meeting?"})
    assert "Launch moved to November 5" in provider.prompts[-1][-1][1]


def test_citation_markers_become_the_names_of_what_they_cite():
    from datetime import date

    from app.services.ask import _render_citations, _strip_citations

    items = {"N5": {"title": "2026-09-25 · Apollo review", "date": date(2026, 9, 25)},
             "M2": {"title": "Apollo sync", "date": date(2026, 9, 20)}}
    # A label used as a noun used to leave "In, it is mentioned".
    assert _render_citations("In [N5], it is mentioned.", items)[0] == "In (Apollo review, 2026-09-25), it is mentioned."
    text, cited = _render_citations("Moved to Nov 5 (M2). Budget 10 lakh [N5], [N5].", items)
    assert text == "Moved to Nov 5 (Apollo sync, 2026-09-20). Budget 10 lakh (Apollo review, 2026-09-25)."
    assert cited == ["M2", "N5"]
    assert _strip_citations("Moved to Nov 12 [M3]. Owner is Daniel [N2, M3].") == ("Moved to Nov 12. Owner is Daniel.", ["M3", "N2"])


@pytest.mark.asyncio
async def test_ollama_asks_for_a_context_window_that_fits(httpx_mock, monkeypatch):
    """Ollama's default window cut an 8,948-token prompt to 2,050 tokens, from the start."""
    from app.providers.llm import ChatMessage, create_llm_provider

    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    httpx_mock.add_response(url="http://localhost:11434/api/chat", json={"message": {"content": "ok"}}, is_reusable=True)
    provider = create_llm_provider(LLMConfig(provider="ollama", model="m"))

    await provider.complete([ChatMessage(role="user", content="short")])
    await provider.complete([ChatMessage(role="user", content="x" * 60_000)])
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    await provider.complete([ChatMessage(role="user", content="short")])

    windows = [json.loads(r.content)["options"]["num_ctx"] for r in httpx_mock.get_requests()]
    assert windows == [16384, 32768, 8192]
