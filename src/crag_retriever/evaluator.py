"""
Evaluation Metrics & Retrieval Grading Suite (`crag_retriever.evaluator`).
"""

from __future__ import annotations

import math
from typing import List, Set

from crag_retriever.models import EvaluationMetrics, RetrievalResult


class RetrievalEvaluator:
    """
    Standard Information Retrieval (IR) and RAG benchmark evaluation suite.
    Computes Precision@K, Recall@K, MRR, and NDCG@K against ground-truth relevant IDs.
    """

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        """Calculate Precision at rank K."""
        if k <= 0 or not retrieved_ids:
            return 0.0
        k_items = retrieved_ids[:k]
        hits = sum(1 for cid in k_items if cid in ground_truth_ids)
        return hits / float(k)

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        """Calculate Recall at rank K."""
        if not ground_truth_ids:
            return 0.0
        k_items = retrieved_ids[:k]
        hits = sum(1 for cid in k_items if cid in ground_truth_ids)
        return hits / float(len(ground_truth_ids))

    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
        """Calculate Reciprocal Rank (RR) for a single query."""
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in ground_truth_ids:
                return 1.0 / float(rank)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at rank K with binary relevance."""
        if k <= 0 or not retrieved_ids or not ground_truth_ids:
            return 0.0

        # DCG calculation
        dcg = 0.0
        for i, cid in enumerate(retrieved_ids[:k], start=1):
            rel = 1.0 if cid in ground_truth_ids else 0.0
            dcg += (2.0**rel - 1.0) / math.log2(i + 1)

        # Ideal DCG calculation
        ideal_hits = min(len(ground_truth_ids), k)
        idcg = sum((2.0**1.0 - 1.0) / math.log2(i + 1) for i in range(1, ideal_hits + 1))

        if idcg <= 0.0:
            return 0.0
        return dcg / idcg

    @classmethod
    def evaluate(
        cls,
        results: List[RetrievalResult],
        ground_truth_ids: Set[str],
        k: int = 5,
    ) -> EvaluationMetrics:
        """
        Evaluate a list of RetrievalResults against a set of ground-truth IDs.
        """
        retrieved_ids = [r.chunk.id for r in results]
        p_k = cls.precision_at_k(retrieved_ids, ground_truth_ids, k=k)
        r_k = cls.recall_at_k(retrieved_ids, ground_truth_ids, k=k)
        mrr = cls.mean_reciprocal_rank(retrieved_ids, ground_truth_ids)
        ndcg = cls.ndcg_at_k(retrieved_ids, ground_truth_ids, k=k)
        hit = 1.0 if any(cid in ground_truth_ids for cid in retrieved_ids[:k]) else 0.0

        return EvaluationMetrics(
            precision_at_k=round(p_k, 4),
            recall_at_k=round(r_k, 4),
            mrr=round(mrr, 4),
            ndcg_at_k=round(ndcg, 4),
            hit_rate=round(hit, 4),
            k=k,
        )
