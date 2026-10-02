"""
A long note's embedding used to crash: "setting an array element with a
sequence... inhomogeneous shape".

First suspected as a race between two embed_batch calls sharing fastembed's
one tokenizer and ONNX session (embed_batch runs on a worker thread via
asyncio.to_thread, and nothing serialized access to it), and _embed_lock
below closes that door - real, worth keeping, verified against a fake model
built to reproduce exactly that shape of cross-thread corruption.

But it did not fix the actual CI failure: the identical error came back on
the very next push, lock and all - reproduced from tests/test_notebook.py::
test_question_about_the_end_of_a_long_note_still_finds_it, which embeds a
note whose long body is split into two pieces. Ruling out concurrency left
the batch call itself: fastembed batches every piece into one call and pads
every encoding to one uniform length before handing the result to numpy;
that padding step failed for this specific two-piece batch on a CI Linux
runner, with both a warm and a freshly-downloaded model reproducing nothing
on this Windows machine - environment- or platform-dependent inside
fastembed/tokenizers/onnxruntime itself, not something reachable from here.

The real fix is structural: embed_batch now asks the model for one document
at a time, never a multi-item batch, so there is nothing left to pad
unevenly. A model that still returns mismatched lengths (whatever the cause)
is now caught with a clear ValueError instead of an oblique numpy crash from
three modules away.
"""
import threading
import time

import pytest

from app.services.embedding import EmbeddingService


class RacyFakeModel:
    """
    Mimics the shared, unsynchronized state inside fastembed's TextEmbedding:
    one buffer (its "tokenizer + session") that every call writes into,
    holds briefly (standing in for actual ONNX inference time), and reads
    back its own share of - corrupted if another call interleaves.
    """

    def __init__(self, dimension=384):
        self.dimension = dimension
        self.shared_buffer = []
        self.max_concurrent = 0
        self._entered = 0
        self._state_lock = threading.Lock()

    def embed(self, documents):
        with self._state_lock:
            self._entered += 1
            self.max_concurrent = max(self.max_concurrent, self._entered)
        start = len(self.shared_buffer)
        for doc in documents:
            self.shared_buffer.append(doc)
            time.sleep(0.005)  # a window another thread's write can land in
        mine = self.shared_buffer[start:start + len(documents)]
        with self._state_lock:
            self._entered -= 1
        if mine != documents:
            raise ValueError(f"corrupted batch: expected {documents}, got {mine}")
        return [[float(len(d))] * self.dimension for d in documents]


class MismatchedLengthModel:
    """However it happens, a model that hands back the wrong-length vector."""

    def __init__(self, bad_index=1):
        self.bad_index = bad_index
        self.calls = 0

    def embed(self, documents):
        length = 384 if self.calls != self.bad_index else 383
        self.calls += 1
        return [[0.0] * length for _ in documents]


@pytest.fixture
def service():
    svc = EmbeddingService(model_name="fake", dimension=1)
    svc._initialized = True  # skip real model loading
    svc._model = RacyFakeModel()
    return svc


# -- no cross-call race (kept as real, independent hardening) ---------------

def test_concurrent_embed_calls_do_not_interleave(service):
    errors = []
    threads = [
        threading.Thread(target=lambda i=i: (errors.append(e) if (e := _try_embed(service, i)) else None))
        for i in range(12)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert service._model.max_concurrent == 1  # the lock was actually contended, not just unused


def _try_embed(service, i):
    try:
        service.embed_batch([f"doc-{i}-a", f"doc-{i}-b"])
        return None
    except ValueError as exc:
        return exc


def test_without_the_lock_the_same_scenario_does_corrupt(service):
    """Proves the fake model is a faithful stand-in, so the pass above means something."""
    model = service._model
    errors = []

    def hit(i):
        try:
            model.embed([f"doc-{i}-a", f"doc-{i}-b"])
        except ValueError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=hit, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors, "expected the unsynchronized model to corrupt at least one batch"


@pytest.mark.asyncio
async def test_the_async_entry_point_serializes_too(service):
    """embed_batch_async is what production actually calls (via asyncio.to_thread)."""
    import asyncio

    results = await asyncio.gather(*[service.embed_batch_async([f"x-{i}"]) for i in range(8)])
    assert len(results) == 8
    assert service._model.max_concurrent == 1


# -- the actual fix: one document per call, nothing left to pad unevenly ----

class CallSizeRecordingModel:
    """Records how many documents arrived in each call to .embed()."""

    def __init__(self):
        self.call_sizes = []

    def embed(self, documents):
        self.call_sizes.append(len(documents))
        return [[0.0] * 384 for _ in documents]


def test_a_multi_piece_note_is_embedded_one_piece_at_a_time():
    """The model never sees more than one document in a call - nothing to batch-pad."""
    svc = EmbeddingService(model_name="fake", dimension=1)
    svc._initialized = True
    svc._model = CallSizeRecordingModel()

    vectors = svc.embed_batch(["first piece", "a very different second piece, much longer than the first"])

    assert len(vectors) == 2
    assert all(len(v) == 384 for v in vectors)
    assert svc._model.call_sizes == [1, 1]  # two calls, one document each - never batched together


def test_a_model_that_still_returns_mismatched_lengths_is_reported_plainly():
    svc = EmbeddingService(model_name="fake", dimension=1)
    svc._initialized = True
    svc._model = MismatchedLengthModel(bad_index=1)

    with pytest.raises(ValueError, match="mismatched vector lengths"):
        svc.embed_batch(["piece one", "piece two"])


def test_the_fallback_vector_is_the_same_in_every_process():
    """
    The fallback hashed words with hash(), which Python salts per process:
    the same text got a different vector after every restart, so nothing
    stored before a restart matched a search after it.
    """
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "from app.services.embedding import EmbeddingService\n"
        "s = EmbeddingService(); s._initialized = True; s._retry_at = float('inf')\n"
        "v = s.embed_text('apollo launch moved to november')\n"
        "print([i for i, x in enumerate(v) if x])"
    )
    root = Path(__file__).resolve().parents[1]
    runs = {
        subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True,
                       env={**__import__("os").environ, "PYTHONHASHSEED": seed}).stdout
        for seed in ("1", "2")
    }
    assert len(runs) == 1
