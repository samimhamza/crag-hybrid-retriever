"""
CRAG Hybrid Retriever: A clean, modular library for Corrective RAG and Hybrid Retrieval.
"""

from crag_retriever.base import BaseReranker, BaseRetriever
from crag_retriever.crag import CorrectiveRetriever, HybridRetriever
from crag_retriever.dense import InMemoryDenseRetriever
from crag_retriever.evaluator import RetrievalEvaluator
from crag_retriever.fusion import ReciprocalRankFusion, reciprocal_rank_fusion
from crag_retriever.lexical import BM25Retriever
from crag_retriever.models import (
    Chunk,
    CRAGAction,
    CRAGFeedback,
    Document,
    EvaluationMetrics,
    RetrievalResult,
    RetrievalSource,
)
from crag_retriever.rerank import FlashRankReranker

__version__ = "0.1.0"

__all__ = [
    "BaseRetriever",
    "BaseReranker",
    "Document",
    "Chunk",
    "RetrievalResult",
    "RetrievalSource",
    "CRAGAction",
    "CRAGFeedback",
    "EvaluationMetrics",
    "BM25Retriever",
    "InMemoryDenseRetriever",
    "ReciprocalRankFusion",
    "reciprocal_rank_fusion",
    "FlashRankReranker",
    "HybridRetriever",
    "CorrectiveRetriever",
    "RetrievalEvaluator",
]

