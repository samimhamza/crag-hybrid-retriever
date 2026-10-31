"""
Sparse Lexical Retriever using BM25Okapi (`crag_retriever.lexical`).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, List, Optional

from rank_bm25 import BM25Okapi

from crag_retriever.base import BaseRetriever
from crag_retriever.models import Chunk, RetrievalResult, RetrievalSource


def default_tokenizer(text: str) -> List[str]:
    """
    Standard whitespace & punctuation tokenizer with Unicode NFKD normalization.
    Preserves alphanumeric words, hyphens, and numeric thresholds.
    """
    normalized = unicodedata.normalize("NFKD", text).lower()
    tokens = re.findall(r"\b\w+(?:[-/]\w+)*\b", normalized)
    return tokens


class BM25Retriever(BaseRetriever):
    """
    Lexical candidate retriever powered by Okapi BM25.
    
    Supports custom tokenizers, dynamic corpus indexing, and score normalization.
    """

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.tokenizer = tokenizer or default_tokenizer
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Build the BM25 index over the provided chunks."""
        self.chunks = list(chunks)
        if not self.chunks:
            self._bm25 = None
            return

        tokenized_corpus = [self.tokenizer(c.content) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Query the BM25 index and return the Top-k scoring chunks.
        """
        if self._bm25 is None or not self.chunks:
            return []

        tokens = self.tokenizer(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: List[RetrievalResult] = []
        rank = 1
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=score,
                    rank=rank,
                    source=RetrievalSource.LEXICAL,
                )
            )
            rank += 1

        return results

