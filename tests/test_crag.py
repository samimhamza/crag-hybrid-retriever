from crag_retriever.crag import CorrectiveRetriever, HybridRetriever
from crag_retriever.models import Chunk


def test_crag_retrieval_and_reflection():
    retriever = HybridRetriever()
    chunks = [
        Chunk(
            document_id="doc-1",
            content="PostgreSQL supports ACID transactions and multi-version concurrency control MVCC.",
        ),
        Chunk(
            document_id="doc-2",
            content="Redis is an in-memory key-value data structure store used as a database, cache, and message broker.",
        ),
    ]
    retriever.index_chunks(chunks)

    crag = CorrectiveRetriever(hybrid_retriever=retriever, upper_threshold=0.5, lower_threshold=0.2)
    results, feedback, history = crag.retrieve_with_crag("PostgreSQL ACID MVCC", top_k=2)

    assert len(results) > 0
    assert feedback is not None
    assert feedback.confidence_score > 0.0
    assert len(history) >= 1
