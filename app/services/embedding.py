"""
Embedding generation service.

Vectors for transcripts, memories, notes and documents come from a local
all-MiniLM-L6-v2 model run through ONNX (fastembed). The ONNX build produces
the same 384-dimensional vectors as the PyTorch original, so databases
embedded with sentence-transformers keep working, while the dependency is
tens of megabytes instead of the two gigabytes PyTorch needs - which is what
makes a downloadable desktop build possible.

The model weights (~90 MB) live in EMBEDDING_CACHE_DIR. The desktop build
ships them (see scripts/fetch_embedding_model.py) and copies them there on
first start; a server install fetches them on first use. Without them the
service falls back to a deterministic hash embedding so the application
still runs.
"""
import asyncio
import os
import shutil
import threading
import time
import zlib
import numpy as np
from pathlib import Path
from typing import List, Union
from app.config import settings
import logging

logger = logging.getLogger(__name__)

#: How long after a failed model load the next embedding call tries again.
MODEL_RETRY_SECONDS = 600


def _fastembed_name(model_name: str) -> str:
    """Accept the short sentence-transformers name people already have in .env."""
    if "/" in model_name:
        return model_name
    return f"sentence-transformers/{model_name}"


def seed_cache_from_bundle(cache_dir: str | None, bundle_dir: str | None) -> bool:
    """
    Copy bundled model weights into the cache directory if it has none yet.

    The desktop app ships the weights inside its (read-only) install and
    points MEET_COMPANION_BUNDLED_MODELS at them; fastembed wants a writable
    cache, so they are copied once into the data directory rather than read
    in place. Returns True when a copy happened.
    """
    if not cache_dir or not bundle_dir:
        return False
    src, dst = Path(bundle_dir), Path(cache_dir)
    if not src.is_dir() or not any(src.iterdir()):
        return False
    if dst.is_dir() and any(dst.glob("models--*")):
        return False
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.name.startswith("."):
            continue  # hub lock files are not part of the model
        target = dst / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target)
    return True


class EmbeddingService:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL, dimension: int = settings.EMBEDDING_DIMENSION):
        self.model_name = model_name
        self.dimension = dimension
        self._model = None
        self._initialized = False
        # _init_model runs in worker threads (via asyncio.to_thread, from both the
        # startup warmup and any concurrent request that beats it there) - guard
        # against loading the multi-hundred-MB model twice in parallel.
        self._init_lock = threading.Lock()
        # embed_batch runs in worker threads too, and fastembed's TextEmbedding
        # shares a single tokenizer and ONNX session across every call - neither
        # is documented as safe for two threads calling .embed() at once. Left
        # unguarded, two notes saved (or a note save racing background meeting
        # processing) at the same moment could have their token batches
        # interleaved: a batch of encodings that should all be padded to one
        # length comes back with mismatched lengths, and
        # np.array([e.ids for e in encoded]) raises "setting an array element
        # with a sequence" - seen intermittently in CI (test_notebook.py::
        # test_question_about_the_end_of_a_long_note_still_finds_it), never
        # locally, exactly the signature of a timing-dependent race. Since the
        # model is already pinned to threads=1 internally, serializing calls
        # from the outside costs nothing but strict correctness.
        self._embed_lock = threading.Lock()
        # A failed load is tried again after a while (the weights may still
        # have been downloading, or the disk was full) instead of leaving the
        # process on the fallback until it restarts.
        self._retry_at = 0.0

    @property
    def using_fallback(self) -> bool:
        """True while vectors come from the hash fallback, not the model."""
        return self._initialized and self._model is None

    def _init_model(self):
        if self._initialized and (self._model is not None or time.monotonic() < self._retry_at):
            return
        with self._init_lock:
            if self._initialized and (self._model is not None or time.monotonic() < self._retry_at):
                return
            # Set before trying, so calls made while a retry is under way use
            # the fallback instead of queueing behind a slow download.
            self._retry_at = time.monotonic() + MODEL_RETRY_SECONDS
            try:
                from fastembed import TextEmbedding

                if seed_cache_from_bundle(settings.EMBEDDING_CACHE_DIR, os.environ.get("MEET_COMPANION_BUNDLED_MODELS")):
                    logger.info("Copied the bundled embedding model into %s", settings.EMBEDDING_CACHE_DIR)
                kwargs = dict(
                    model_name=_fastembed_name(self.model_name),
                    cache_dir=str(settings.EMBEDDING_CACHE_DIR) if settings.EMBEDDING_CACHE_DIR else None,
                    # One thread: on shared-vCPU hosts more threads contend
                    # rather than speed up, and the desktop app should not
                    # peg every core while a meeting is being indexed.
                    threads=1,
                )
                # fastembed asks the hub before using its cache, and with no
                # network it gives up after minutes of retries - even when
                # every file is already on disk. Local first, download only
                # when there is nothing local.
                try:
                    self._model = TextEmbedding(local_files_only=True, **kwargs)
                    source = "local"
                except Exception:
                    self._model = TextEmbedding(**kwargs)
                    source = "downloaded"
                self._initialized = True
                logger.info(f"Loaded embedding model: {self.model_name} (ONNX, {source})")
            except Exception as e:
                logger.warning(f"Embedding model unavailable ({e}). Using deterministic fallback embedding.")
                self._initialized = True

    async def embed_text_async(self, text: str) -> List[float]:
        """Non-blocking version of embed_text - offloads the CPU-bound model call to a thread
        so it doesn't freeze the event loop (and every other in-flight request) while it runs."""
        return (await self.embed_batch_async([text]))[0]

    async def embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Non-blocking version of embed_batch - see embed_text_async."""
        return await asyncio.to_thread(self.embed_batch, texts)

    async def warmup_async(self):
        """Load the model in a background thread so the first real request isn't the one
        paying the multi-second model load cost (and blocking the event loop while it does)."""
        await asyncio.to_thread(self._init_model)

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of text strings."""
        if not texts:
            return []

        self._init_model()

        # Clean text
        clean_texts = [
            str(t or "").encode("utf-8", "ignore").decode("utf-8").strip()
            for t in texts
        ]

        # One embedding computation at a time process-wide - see the comment
        # on _embed_lock in __init__ for why this has to be unconditional,
        # not just around the fastembed branch: the fallback branch below is
        # cheap enough that contention here is never the bottleneck.
        with self._embed_lock:
            if self._model is not None:
                # One document per call to the model, not the whole list in
                # one batch. Batching relies on fastembed padding every
                # encoding in the call to one uniform length before handing
                # them to numpy; that padding failed intermittently in CI for
                # a batch of two pieces of a long note (different lengths
                # came back for what should have been one padded length),
                # and it reproduced on a CI Linux runner with a freshly
                # downloaded model but not on this Windows machine with
                # either a warm or a from-scratch cache - environment- or
                # platform-dependent in fastembed/tokenizers/onnxruntime
                # itself, not something this service can fix upstream. A
                # single-document call has nothing to pad against, so it
                # cannot produce this failure by construction, at the cost
                # of the batching speedup (embedding is background work
                # here, never a request's hot path).
                vectors = [np.asarray(next(iter(self._model.embed([text]))), dtype=np.float32) for text in clean_texts]
                lengths = {len(v) for v in vectors}
                if len(lengths) > 1:
                    # Would have been the same numpy crash one level up;
                    # caught here with the actual shapes, for a log that
                    # says something a stack trace from np.asarray would not.
                    raise ValueError(f"embedding model returned mismatched vector lengths: {sorted(lengths)}")
                return [v.tolist() for v in vectors]

            # Deterministic lightweight fallback (e.g. if PyTorch cannot load on low disk space)
            # Generates a normalized 384-dimensional vector based on token hashing.
            # crc32, not hash(): Python salts str hashes per process, so the
            # same word landed in a different slot after every restart.
            results = []
            for text in clean_texts:
                vec = np.zeros(self.dimension, dtype=np.float32)
                words = text.lower().split()
                for word in words:
                    h = zlib.crc32(word.encode("utf-8")) % self.dimension
                    vec[h] += 1.0
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                results.append(vec.tolist())
            return results


embedding_service = EmbeddingService()
