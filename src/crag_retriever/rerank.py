"""
Cross-Encoder Neural Reranking using FlashRank (`crag_retriever.rerank`).
"""

from __future__ import annotations

from crag_retriever.base import BaseReranker
from crag_retriever.models import RetrievalResult, RetrievalSource


class FlashRankReranker(BaseReranker):
    """
    High-speed, ONNX-accelerated Cross-Encoder reranker powered by FlashRank.

    Supports multilingual models (e.g., `ms-marco-MultiBERT-L-12`) as well as
    ultra-fast English models (e.g., `ms-marco-MiniLM-L-12-v2`).
    """

    def __init__(
        self,
        model_name: str = "ms-marco-MultiBERT-L-12",
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._ranker = None

    def _get_ranker(self):
        if self._ranker is None:
            try:
                from flashrank import Ranker

                kwargs = {"model_name": self.model_name}
                if self.cache_dir:
                    kwargs["cache_dir"] = self.cache_dir
                self._ranker = Ranker(**kwargs)
            except Exception:
                # Fallback to local heuristic reranker if FlashRank weights cannot be fetched
                self._ranker = None
        return self._ranker

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Rerank a candidate list using cross-encoder attention against the query.
        """
        if not candidates:
            return []

        ranker = self._get_ranker()
        if ranker is not None:
            try:
                from flashrank import RerankRequest

                passages = [{"id": c.chunk.id, "text": c.chunk.content} for c in candidates]
                request = RerankRequest(query=query, passages=passages)
                reranked_raw = ranker.rerank(request)

                chunk_map: dict[str, RetrievalResult] = {c.chunk.id: c for c in candidates}
                reranked_results: list[RetrievalResult] = []

                for rank, item in enumerate(reranked_raw, start=1):
                    cid = item["id"]
                    score = float(item["score"])
                    orig = chunk_map[cid]
                    reranked_results.append(
                        RetrievalResult(
                            chunk=orig.chunk,
                            score=round(score, 6),
                            rank=rank,
                            source=RetrievalSource.CROSS_ENCODER,
                        )
                    )

                if top_k is not None:
                    reranked_results = reranked_results[:top_k]
                return reranked_results
            except Exception:
                pass

        # Robust heuristic fallback (exact token overlap + candidate original score)
        return self._heuristic_fallback_rerank(query, candidates, top_k=top_k)

    def _heuristic_fallback_rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        q_tokens = set(query.lower().split())
        scored: list[RetrievalResult] = []

        for c in candidates:
            c_tokens = set(c.chunk.content.lower().split())
            overlap = len(q_tokens.intersection(c_tokens)) / max(1, len(q_tokens))
            combined_score = 0.5 * c.score + 0.5 * overlap
            scored.append(
                RetrievalResult(
                    chunk=c.chunk,
                    score=round(combined_score, 4),
                    rank=1,
                    source=RetrievalSource.CROSS_ENCODER,
                )
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        if top_k is not None:
            scored = scored[:top_k]

        for rank, r in enumerate(scored, start=1):
            r.rank = rank

        return scored
