"""
Reciprocal Rank Fusion (RRF) & Hybrid Candidate Fusion (`crag_retriever.fusion`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from crag_retriever.models import Chunk, RetrievalResult, RetrievalSource

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: List[List[RetrievalResult]],
    k_rrf: int = DEFAULT_RRF_K,
    top_k: Optional[int] = None,
) -> List[RetrievalResult]:
    """
    Fuse multiple ranked candidate lists using standard Reciprocal Rank Fusion (RRF):
        RRF_Score(d) = sum_{l in ranked_lists} ( 1.0 / (k_rrf + rank_l(d)) )
    
    where rank_l(d) is 1-indexed (1, 2, 3, ...).
    
    Args:
        ranked_lists: List of candidate result lists (e.g., [dense_results, bm25_results]).
        k_rrf: Smoothing constant (default 60, as established in Cormack et al.).
        top_k: Number of fused results to return. If None, returns all distinct candidates.
    
    Returns:
        Sorted list of RetrievalResult objects with fused RRF scores.
    """
    chunk_lookup: Dict[str, Chunk] = {}
    fused_scores: Dict[str, float] = {}

    for candidate_list in ranked_lists:
        for rank_idx, item in enumerate(candidate_list, start=1):
            cid = item.chunk.id
            if cid not in chunk_lookup:
                chunk_lookup[cid] = item.chunk
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (k_rrf + rank_idx))

    sorted_items = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        sorted_items = sorted_items[:top_k]

    results: List[RetrievalResult] = []
    for rank, (cid, score) in enumerate(sorted_items, start=1):
        results.append(
            RetrievalResult(
                chunk=chunk_lookup[cid],
                score=round(score, 6),
                rank=rank,
                source=RetrievalSource.FUSION_RRF,
            )
        )

    return results


class ReciprocalRankFusion:
    """
    Configurable class wrapper for Reciprocal Rank Fusion.
    """

    def __init__(self, k_rrf: int = DEFAULT_RRF_K) -> None:
        self.k_rrf = k_rrf

    def fuse(
        self,
        ranked_lists: List[List[RetrievalResult]],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        return reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            k_rrf=self.k_rrf,
            top_k=top_k,
        )
