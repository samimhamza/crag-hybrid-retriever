"""
Corrective RAG (CRAG) Self-Reflective Retrieval Engine (`crag_retriever.crag`).
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Set, Tuple

from crag_retriever.dense import InMemoryDenseRetriever
from crag_retriever.fusion import ReciprocalRankFusion
from crag_retriever.lexical import BM25Retriever
from crag_retriever.models import (
    Chunk,
    CRAGAction,
    CRAGFeedback,
    RetrievalResult,
    RetrievalSource,
)
from crag_retriever.rerank import FlashRankReranker


class HybridRetriever:
    """
    Coordinates Dense Vector Retrieval + BM25 Lexical Retrieval + RRF + Cross-Encoder Reranking.
    """

    def __init__(
        self,
        dense_retriever: Optional[InMemoryDenseRetriever] = None,
        lexical_retriever: Optional[BM25Retriever] = None,
        reranker: Optional[FlashRankReranker] = None,
        k_rrf: int = 60,
    ) -> None:
        self.dense = dense_retriever or InMemoryDenseRetriever()
        self.lexical = lexical_retriever or BM25Retriever()
        self.fusion = ReciprocalRankFusion(k_rrf=k_rrf)
        self.reranker = reranker or FlashRankReranker()

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Index chunks across both sparse and dense indices."""
        self.dense.index_chunks(chunks)
        self.lexical.index_chunks(chunks)

    def search(
        self,
        query: str,
        dense_k: int = 15,
        lexical_k: int = 15,
        fused_k: int = 10,
        final_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Execute full two-stage hybrid retrieval:
        1. Retrieve top candidates from Dense & BM25 indices.
        2. Fuse via Reciprocal Rank Fusion (RRF).
        3. Rerank top fused candidates via Cross-Encoder.
        """
        dense_results = self.dense.retrieve(query, top_k=dense_k)
        lexical_results = self.lexical.retrieve(query, top_k=lexical_k)

        fused = self.fusion.fuse([dense_results, lexical_results], top_k=fused_k)
        reranked = self.reranker.rerank(query, candidates=fused, top_k=final_k)
        return reranked


class CorrectiveRetriever:
    """
    Corrective RAG (CRAG) Engine.
    
    Evaluates retrieved evidence confidence and self-corrects:
    - High confidence (CORRECT) -> Accepts candidates.
    - Borderline (AMBIGUOUS) -> Rewrites query to focus on missing entities and re-retrieves.
    - Low confidence (INCORRECT) -> Discards noise and expands query for broader coverage.
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        upper_threshold: float = 0.65,
        lower_threshold: float = 0.35,
        max_iterations: int = 3,
        query_rewriter: Optional[Callable[[str, List[str]], str]] = None,
    ) -> None:
        self.retriever = hybrid_retriever or HybridRetriever()
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.max_iterations = max_iterations
        self.query_rewriter = query_rewriter or self._default_query_rewriter

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Index corpus into the underlying hybrid retriever."""
        self.retriever.index_chunks(chunks)

    def grade_evidence(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> CRAGFeedback:
        """
        Self-reflective grading step.
        Evaluates relevance scores, query entity coverage, and potential missing aspects.
        """
        if not results:
            return CRAGFeedback(
                action=CRAGAction.INCORRECT,
                confidence_score=0.0,
                reasons=["No candidate documents retrieved."],
                suggested_reformulations=[query],
                missing_key_aspects=self._extract_key_terms(query),
            )

        top_score = max(r.score for r in results)
        avg_score = sum(r.score for r in results) / len(results)
        confidence = float(0.6 * top_score + 0.4 * avg_score)

        # Check entity/term coverage
        query_terms = set(self._extract_key_terms(query))
        retrieved_text = " ".join(r.chunk.content.lower() for r in results)
        missing_terms = [t for t in query_terms if t.lower() not in retrieved_text]

        coverage_ratio = (len(query_terms) - len(missing_terms)) / max(1, len(query_terms))
        adjusted_confidence = round(0.7 * confidence + 0.3 * coverage_ratio, 4)

        reasons = [
            f"Mean candidate score: {avg_score:.3f}",
            f"Entity coverage ratio: {coverage_ratio:.2f}",
        ]

        if adjusted_confidence >= self.upper_threshold:
            action = CRAGAction.CORRECT
            reasons.append("Retrieved passages satisfy confidence criteria.")
        elif adjusted_confidence >= self.lower_threshold:
            action = CRAGAction.AMBIGUOUS
            reasons.append("Evidence is partially relevant or missing key entity constraints.")
        else:
            action = CRAGAction.INCORRECT
            reasons.append("Retrieved passages lack strong relevance to query intent.")

        suggested = []
        if missing_terms:
            suggested.append(f"{query} {' '.join(missing_terms)}")

        return CRAGFeedback(
            action=action,
            confidence_score=adjusted_confidence,
            reasons=reasons,
            suggested_reformulations=suggested,
            missing_key_aspects=missing_terms,
        )

    def retrieve_with_crag(
        self,
        query: str,
        top_k: int = 5,
    ) -> Tuple[List[RetrievalResult], CRAGFeedback, List[str]]:
        """
        Execute the CRAG loop:
        1. Retrieve hybrid candidates.
        2. Grade evidence (CORRECT / AMBIGUOUS / INCORRECT).
        3. If AMBIGUOUS or INCORRECT and iterations remaining, rewrite query and re-retrieve.
        4. Return final results, feedback diagnosis, and query transformation history.
        """
        current_query = query
        query_history: List[str] = [current_query]
        feedback: Optional[CRAGFeedback] = None
        results: List[RetrievalResult] = []

        for iteration in range(1, self.max_iterations + 1):
            results = self.retriever.search(current_query, final_k=top_k)
            feedback = self.grade_evidence(current_query, results)

            if feedback.action == CRAGAction.CORRECT:
                break

            # Need refinement / query rewrite
            if iteration < self.max_iterations:
                new_query = self.query_rewriter(current_query, feedback.missing_key_aspects)
                if new_query and new_query not in query_history:
                    current_query = new_query
                    query_history.append(current_query)
                else:
                    break

        # Mark source as refined by CRAG
        for r in results:
            r.source = RetrievalSource.CRAG_REFINED

        return results, feedback, query_history

    def _extract_key_terms(self, text: str) -> List[str]:
        words = re.findall(r"\b[A-Za-z0-9_-]{3,}\b", text)
        stopwords = {
            "what", "where", "which", "when", "how", "the", "and", "for", "with",
            "that", "this", "from", "are", "can", "should", "does", "about",
        }
        return [w for w in words if w.lower() not in stopwords]

    def _default_query_rewriter(self, query: str, missing_aspects: List[str]) -> str:
        if not missing_aspects:
            return query
        aspects_str = " ".join(missing_aspects[:3])
        return f"{query} {aspects_str}".strip()
