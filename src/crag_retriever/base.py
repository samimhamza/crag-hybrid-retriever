"""
Abstract Base Classes & Interfaces (`crag_retriever.base`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from crag_retriever.models import Chunk, RetrievalResult


class BaseRetriever(ABC):
    """Abstract interface for all candidate retrievers (Dense, Sparse, Hybrid)."""

    @abstractmethod
    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index a collection of chunks for search."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Search indexed chunks and return ranked candidates."""
        pass


class BaseReranker(ABC):
    """Abstract interface for cross-encoder rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank candidates using cross-attention or neural score."""
        pass
