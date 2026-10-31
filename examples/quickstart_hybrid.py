"""
Quickstart: Hybrid Retrieval with BM25, Dense Vector Search, RRF, and Cross-Encoder Reranking.
"""

import sys
from pathlib import Path

# Add src to sys.path for local direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from crag_retriever import (
    BM25Retriever,
    Chunk,
    FlashRankReranker,
    HybridRetriever,
    InMemoryDenseRetriever,
)


def main():
    print("=" * 70)
    print("CRAG Hybrid Retriever: Quickstart Demo")
    print("=" * 70)

    # 1. Define sample knowledge corpus
    documents = [
        "Vector search uses dense embeddings to capture semantic similarity across concepts.",
        "BM25 is a probabilistic lexical ranking function based on Term Frequency (TF) and IDF.",
        "Reciprocal Rank Fusion (RRF) combines diverse ranked search results without score calibration.",
        "Cross-encoders jointly process query and passage via full self-attention for superior reranking.",
        "Corrective RAG (CRAG) evaluates retrieval evidence and self-corrects ambiguous or missing facts.",
    ]
    chunks = [Chunk(document_id=f"doc-{i}", content=text) for i, text in enumerate(documents)]

    # 2. Instantiate and index hybrid engine
    retriever = HybridRetriever(
        dense_retriever=InMemoryDenseRetriever(),
        lexical_retriever=BM25Retriever(),
        reranker=FlashRankReranker(),
        k_rrf=60,
    )
    retriever.index_chunks(chunks)

    # 3. Query the hybrid engine
    query = "How does reciprocal rank fusion combine search rankings?"
    print(f"\nQuery: '{query}'\n")

    results = retriever.search(query=query, final_k=3)

    for res in results:
        print(f"Rank {res.rank} | Score: {res.score:.4f} | Source: {res.source.value}")
        print(f"  Content: {res.content}\n")


if __name__ == "__main__":
    main()
