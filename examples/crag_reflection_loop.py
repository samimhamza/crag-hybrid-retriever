"""
Self-Reflective CRAG Loop: Demonstrating confidence grading, entity detection, and query reformulation.
"""

import sys
from pathlib import Path

# Add src to sys.path for local direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from crag_retriever import (
    Chunk,
    CorrectiveRetriever,
    HybridRetriever,
    RetrievalEvaluator,
)


def main():
    print("=" * 70)
    print("CRAG Self-Reflective Evaluation & Query Rewrite Loop")
    print("=" * 70)

    # 1. Corpus with specific microservice architecture facts
    corpus = [
        "Kubernetes horizontal pod autoscaling (HPA) scales pods based on observed CPU and memory usage.",
        "gRPC uses HTTP/2 multiplexing and Protocol Buffers for high-throughput microservice communication.",
        "Envoy proxy provides resilient service discovery, client-side load balancing, and circuit breaking.",
        "Prometheus scrapes OpenMetrics endpoints to provide real-time operational time-series monitoring.",
    ]
    chunks = [Chunk(document_id=f"doc-{i}", content=text) for i, text in enumerate(corpus)]

    # 2. Build CRAG engine
    hybrid = HybridRetriever()
    crag = CorrectiveRetriever(
        hybrid_retriever=hybrid,
        upper_threshold=0.6,
        lower_threshold=0.3,
        max_iterations=3,
    )
    crag.index_chunks(chunks)

    # 3. Execute self-correcting query
    user_query = "Protocol Buffers HTTP/2 multiplexing in microservices"
    print(f"\nInitial Query: '{user_query}'")

    results, feedback, query_history = crag.retrieve_with_crag(user_query, top_k=2)

    print(f"\nCRAG Action: {feedback.action.value.upper()}")
    print(f"Confidence Score: {feedback.confidence_score}")
    print(f"Query Evolution: {' -> '.join(query_history)}")
    print(f"Diagnostic Notes: {feedback.reasons}")

    print("\nFinal Verified Passages:")
    for r in results:
        print(f"  [Rank {r.rank} - Score {r.score:.4f}] {r.content}")

    # 4. Evaluate against ground truth
    ground_truth_ids = {chunks[1].id}  # doc-1 is gRPC
    metrics = RetrievalEvaluator.evaluate(results, ground_truth_ids, k=2)
    print(f"\nEvaluation Metrics: Precision@2={metrics.precision_at_k}, MRR={metrics.mrr}, NDCG@2={metrics.ndcg_at_k}")


if __name__ == "__main__":
    main()
