import pytest
from crag_retriever.evaluator import RetrievalEvaluator
from crag_retriever.models import Chunk, RetrievalResult


def test_retrieval_evaluator_metrics():
    c1 = Chunk(document_id="doc-1", content="Chunk 1")
    c2 = Chunk(document_id="doc-2", content="Chunk 2")
    c3 = Chunk(document_id="doc-3", content="Chunk 3")

    results = [
        RetrievalResult(chunk=c1, score=0.9, rank=1),
        RetrievalResult(chunk=c2, score=0.8, rank=2),
        RetrievalResult(chunk=c3, score=0.7, rank=3),
    ]

    # Ground truth: c2 and c3 are relevant, c1 is not
    ground_truth = {c2.id, c3.id}

    metrics = RetrievalEvaluator.evaluate(results=results, ground_truth_ids=ground_truth, k=3)
    # Precision@3 = 2/3 = 0.6667
    assert metrics.precision_at_k == pytest.approx(0.6667, abs=1e-3)
    # Recall@3 = 2/2 = 1.0
    assert metrics.recall_at_k == 1.0
    # MRR: first hit is at rank 2 -> 1/2 = 0.5
    assert metrics.mrr == 0.5
    # Hit rate = 1.0
    assert metrics.hit_rate == 1.0
    assert metrics.ndcg_at_k > 0.0
