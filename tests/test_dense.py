import numpy as np
import pytest
from crag_retriever.dense import InMemoryDenseRetriever, deterministic_semantic_embedder
from crag_retriever.models import Chunk


def test_deterministic_semantic_embedder():
    v1 = deterministic_semantic_embedder("Deep learning neural networks")
    v2 = deterministic_semantic_embedder("Deep learning neural networks")
    v3 = deterministic_semantic_embedder("Totally unrelated culinary recipes")

    # Identical inputs yield identical vectors
    assert np.allclose(v1, v2)
    # Unit normalized
    assert np.isclose(np.linalg.norm(v1), 1.0)
    # Different inputs yield different vectors
    sim_diff = np.dot(v1, v3)
    assert sim_diff < 0.99


def test_in_memory_dense_retriever():
    retriever = InMemoryDenseRetriever(dim=64)
    chunks = [
        Chunk(document_id="doc-a", content="Neural network transformers for natural language processing"),
        Chunk(document_id="doc-b", content="Database indexing with B-Trees and LSM Trees"),
    ]
    retriever.index_chunks(chunks)

    results = retriever.retrieve("transformers for natural language", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.document_id == "doc-a"

