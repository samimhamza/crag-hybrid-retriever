"""
Core Data Models for CRAG & Hybrid Retrieval (`crag_retriever.models`).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Container for an ingested document or textual asset."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Granular segment of a document used for vector/lexical indexing."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    index_in_doc: int = 0


class RetrievalSource(str, Enum):
    """Identifier for the retrieval method or pipeline stage."""
    LEXICAL = "lexical"
    DENSE = "dense"
    FUSION_RRF = "fusion_rrf"
    CROSS_ENCODER = "cross_encoder"
    CRAG_REFINED = "crag_refined"


class RetrievalResult(BaseModel):
    """Scored candidate passage returned by a retrieval or reranking step."""
    chunk: Chunk
    score: float
    rank: int = 1
    source: RetrievalSource = RetrievalSource.LEXICAL

    @property
    def content(self) -> str:
        return self.chunk.content

    @property
    def chunk_id(self) -> str:
        return self.chunk.id


class CRAGAction(str, Enum):
    """CRAG reflection decision actions."""
    CORRECT = "correct"        # Evidence is sufficient & high confidence -> accept
    AMBIGUOUS = "ambiguous"    # Evidence is partial or borderline -> rewrite & refine query
    INCORRECT = "incorrect"    # Retrieval failed / irrelevant -> fallback / decompose query


class CRAGFeedback(BaseModel):
    """Diagnostic reflection output from the CRAG grading stage."""
    action: CRAGAction
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)
    suggested_reformulations: List[str] = Field(default_factory=list)
    missing_key_aspects: List[str] = Field(default_factory=list)


class EvaluationMetrics(BaseModel):
    """Standard Information Retrieval and Retrieval Evaluation metrics."""
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    hit_rate: float = 0.0
    k: int = 5

