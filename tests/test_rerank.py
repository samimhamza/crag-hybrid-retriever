import pytest
from crag_retriever.models import Chunk, RetrievalResult
from crag_retriever.rerank import FlashRankReranker


def test_flashrank_reranker_fallback():
    # Test that when FlashRank runs or falls back, it yields properly ranked candidates
    reranker = FlashRankReranker()
    c1 = Chunk(document_id="doc-1", content="Python fastapi microservices and async io")
    c2 = Chunk(document_id="doc-2", content="Frontend web development with React and Vite")

    candidates = [
        RetrievalResult(chunk=c2, score=0.5, rank=1),
        RetrievalResult(chunk=c1, score=0.4, rank=2),
    ]

    reranked = reranker.rerank(query="async fastapi microservices", candidates=candidates, top_k=2)
    assert len(reranked) == 2
    # c1 should be reranked to top because of exact token matching with query
    assert reranked[0].chunk.id == c1.id
    assert reranked[0].rank == 1
