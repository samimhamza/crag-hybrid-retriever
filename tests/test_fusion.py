import pytest
from crag_retriever.fusion import ReciprocalRankFusion, reciprocal_rank_fusion
from crag_retriever.models import Chunk, RetrievalResult, RetrievalSource


def test_reciprocal_rank_fusion_math():
    c1 = Chunk(document_id="doc-1", content="Chunk 1 content")
    c2 = Chunk(document_id="doc-2", content="Chunk 2 content")
    c3 = Chunk(document_id="doc-3", content="Chunk 3 content")

    # List 1: c1 (rank 1), c2 (rank 2)
    # List 2: c2 (rank 1), c3 (rank 2)
    list1 = [
        RetrievalResult(chunk=c1, score=0.9, rank=1, source=RetrievalSource.DENSE),
        RetrievalResult(chunk=c2, score=0.8, rank=2, source=RetrievalSource.DENSE),
    ]
    list2 = [
        RetrievalResult(chunk=c2, score=10.5, rank=1, source=RetrievalSource.LEXICAL),
        RetrievalResult(chunk=c3, score=8.1, rank=2, source=RetrievalSource.LEXICAL),
    ]

    # With k_rrf = 60:
    # c2 score: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # c1 score: 1/(60+1) = 0.016393
    # c3 score: 1/(60+2) = 0.016129
    # Winner should be c2!
    fused = reciprocal_rank_fusion([list1, list2], k_rrf=60)
    assert len(fused) == 3
    assert fused[0].chunk.id == c2.id
    assert fused[0].rank == 1
    assert fused[1].chunk.id == c1.id
    assert fused[2].chunk.id == c3.id


def test_fusion_class_wrapper():
    rrf = ReciprocalRankFusion(k_rrf=50)
    c1 = Chunk(document_id="doc-1", content="C1")
    res = [RetrievalResult(chunk=c1, score=1.0, rank=1)]
    fused = rrf.fuse([res], top_k=1)
    assert len(fused) == 1
    assert fused[0].source == RetrievalSource.FUSION_RRF
