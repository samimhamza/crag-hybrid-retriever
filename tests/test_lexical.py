from crag_retriever.lexical import BM25Retriever, default_tokenizer
from crag_retriever.models import Chunk


def test_default_tokenizer():
    tokens = default_tokenizer("Hybrid search with BM25Okapi and 10-20 m3/ha thresholds!")
    assert "hybrid" in tokens
    assert "bm25okapi" in tokens
    assert "10-20" in tokens or "m3/ha" in tokens


def test_bm25_retriever():
    retriever = BM25Retriever()
    chunks = [
        Chunk(
            document_id="doc-1",
            content="Python is an interpreted high-level general-purpose programming language.",
        ),
        Chunk(
            document_id="doc-2",
            content="Rust is a multi-paradigm, general-purpose programming language that emphasizes performance and safety.",
        ),
        Chunk(
            document_id="doc-3",
            content="Kubernetes is an open-source container orchestration system for automating software deployment.",
        ),
    ]
    retriever.index_chunks(chunks)

    results = retriever.retrieve("Rust programming language", top_k=2)
    assert len(results) > 0
    assert results[0].chunk.document_id == "doc-2"
    assert results[0].rank == 1
