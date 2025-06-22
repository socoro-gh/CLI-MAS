"""RAG utilities for example-driven persona retrieval.

Each agent builds (or reloads) a FAISS vector index from its example
utterances.  The index files live in ``history/faiss/<agent>.idx`` next to
``<agent>.json`` which stores the original example strings so we can map
back from FAISS ids to text.

Functions here are intentionally lightweight so they can be imported by
`mas.py` without heavy dependencies at import-time (e.g. the embedding
model is only loaded on first use).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import json
import os
import threading

import numpy as np

# Lazy import of heavy deps so unit tests that patch them out can run fast.
try:
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore
    SentenceTransformer = None  # type: ignore


class _SingletonModelCache:
    """Thread-safe singleton cache for embedding models."""

    _lock = threading.Lock()
    _models: dict[str, "SentenceTransformer"] = {}

    @classmethod
    def get(cls, name: str) -> "SentenceTransformer":
        if SentenceTransformer is None:  # pragma: no cover
            raise RuntimeError("sentence_transformers not available")
        with cls._lock:
            if name not in cls._models:
                cls._models[name] = SentenceTransformer(name)
            return cls._models[name]


def _embed(texts: List[str], model_name: str) -> np.ndarray:
    model = _SingletonModelCache.get(model_name)
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    return emb


class RagIndex:
    """FAISS wrapper that keeps the original strings around."""

    def __init__(self, agent_name: str, root_dir: Path, model_name: str = "all-mpnet-base-v2") -> None:
        self.agent_name = agent_name
        self.root_dir = root_dir
        self.model_name = model_name
        self._index = None  # lazy
        self._samples: List[str] = []
        self._dim: int | None = None

    # ------------------------------- paths ----------------------------------
    def _faiss_path(self) -> Path:
        return self.root_dir / f"{self.agent_name}.idx"

    def _json_path(self) -> Path:
        return self.root_dir / f"{self.agent_name}.json"

    # ------------------------------- build / load ---------------------------
    def build(self, samples: List[str]) -> None:
        """(Re)build index from *samples* and persist."""
        if faiss is None:  # pragma: no cover
            raise RuntimeError("faiss not available – cannot build index")
        self._samples = [s.strip() for s in samples if s.strip()]
        if not self._samples:
            raise ValueError("No non-empty samples to index")
        emb = _embed(self._samples, self.model_name)
        self._dim = emb.shape[1]
        index = faiss.IndexFlatIP(self._dim)
        index.add(emb)
        # persist
        self._faiss_path().parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self._faiss_path()))
        self._json_path().write_text(json.dumps(self._samples, ensure_ascii=False), encoding="utf-8")
        self._index = index

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        if faiss is None:  # pragma: no cover
            return False
        if not self._faiss_path().is_file() or not self._json_path().is_file():
            return False
        try:
            self._index = faiss.read_index(str(self._faiss_path()))
            self._samples = json.loads(self._json_path().read_text(encoding="utf-8"))
            self._dim = self._index.d
            return True
        except Exception:
            return False

    # ------------------------------- query ----------------------------------
    def query(self, text: str, k: int = 3) -> List[str]:
        if self._index is None:
            raise RuntimeError("Index not built or loaded")
        # embed query
        q_emb = _embed([text], self.model_name)
        scores, ids = self._index.search(q_emb, min(k, len(self._samples)))
        return [self._samples[i] for i in ids[0] if i != -1]

    # ------------------------------- helpers --------------------------------
    def ensure(self, samples: List[str]) -> None:
        """Try loading index; if missing build from samples."""
        if not self.load():
            self.build(samples) 