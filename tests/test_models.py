import pytest
from crag_retriever.models import (
    Chunk,
    CRAGAction,
    CRAGFeedback,
    Document,
    EvaluationMetrics,
    RetrievalResult,
    RetrievalSource,
)


def test_document_and_chunk_creation():
    doc = Document(content="Machine Learning Architecture", metadata={"author": "Researcher"})
    assert doc.id is not None
    assert doc.metadata["author"] == "Researcher"

    chunk = Chunk(document_id=doc.id, content="Transformer self-attention mechanism", index_in_doc=0)
    assert chunk.document_id == doc.id
    assert chunk.content.startswith("Transformer")


def test_retrieval_result_properties():
    chunk = Chunk(document_id="doc-1", content="Sample content for testing")
    result = RetrievalResult(chunk=chunk, score=0.85, rank=1, source=RetrievalSource.LEXICAL)
    assert result.content == "Sample content for testing"
    assert result.chunk_id == chunk.id
    assert result.source == RetrievalSource.LEXICAL


def test_crag_feedback():
    feedback = CRAGFeedback(
        action=CRAGAction.CORRECT,
        confidence_score=0.92,
        reasons=["High semantic overlap"],
        suggested_reformulations=[],
    )
    assert feedback.action == CRAGAction.CORRECT
    assert feedback.confidence_score == 0.92
