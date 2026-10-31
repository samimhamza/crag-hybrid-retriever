"""
Dense Vector Retriever (`crag_retriever.dense`).
"""

from __future__ import annotations

import hashlib
from typing import Callable, List, Optional

import numpy as np

from crag_retriever.base import BaseRetriever
from crag_retriever.models import Chunk, RetrievalResult, RetrievalSource


def deterministic_semantic_embedder(text: str, dim: int = 128) -> np.ndarray:
    """
    Deterministic pseudo-semantic embedding function for lightweight or offline execution.
    Generates consistent unit-normalized dense vectors without external network or heavy weights.
    """
    cleaned = text.strip().lower()
    vec = np.zeros(dim, dtype=np.float32)
    words = cleaned.split()
    if not words:
        vec[0] = 1.0
        return vec

    for w in words:
        digest = hashlib.sha256(w.encode("utf-8")).digest()
        for i in range(min(dim, len(digest))):
            vec[i] += float(digest[i] - 128) / 128.0

    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec = vec / norm
    return vec


class InMemoryDenseRetriever(BaseRetriever):
    """
    In-memory dense retriever powered by cosine similarity over NumPy matrices.
    Accepts any custom embedding callable `Callable[[str], np.ndarray]` or
    HuggingFace/sentence-transformers model.
    """

    def __init__(
        self,
        embedder: Optional[Callable[[str], np.ndarray]] = None,
        dim: int = 128,
    ) -> None:
        self.dim = dim
        self.embedder = embedder or (lambda text: deterministic_semantic_embedder(text, dim=self.dim))
        self.chunks: List[Chunk] = []
        self._matrix: Optional[np.ndarray] = None

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Embed and index chunks into a normalized NumPy matrix."""
        self.chunks = list(chunks)
        if not self.chunks:
            self._matrix = None
            return

        embeddings: List[np.ndarray] = []
        for c in self.chunks:
            vec = self.embedder(c.content)
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            embeddings.append(vec)

        self._matrix = np.vstack(embeddings).astype(np.float32)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Compute cosine similarity between the query vector and indexed chunk vectors.
        """
        if self._matrix is None or not self.chunks:
            return []

        q_vec = self.embedder(query)
        norm = np.linalg.norm(q_vec)
        if norm > 1e-9:
            q_vec = q_vec / norm

        # Cosine similarity as dot product between unit vectors
        scores = np.dot(self._matrix, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=score,
                    rank=rank,
                    source=RetrievalSource.DENSE,
                )
            )

        return results

