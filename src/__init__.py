# -*- coding: utf-8 -*-
"""
PHI-Enhanced RLM Core Library -- v4.1.0

Self-evolving recursive language model with phi-geometric mathematics,
multi-agent integration, tool execution, and adversarial self-evaluation.
"""

from .phi_enhanced_rlm import PhiEnhancedRLM, ContextChunk, SubCallResult, LLMResponse
from .phi_separation_novel_mathematics import (
    PHI, PHI_INV, LOG_PHI,
    PhiGramMatrix, SpectralFlow, PhiRenormalizationGroup
)
from .embeddings import get_embedder, CachedEmbedder, EmbeddingConfig
from .cache import SQLiteEmbeddingCache, get_sqlite_cache
from .extractors import (
    extract_pdf_content, extract_docx_content, extract_web_content,
    chunk_python_code, chunk_code_file
)
from .progress import RichProgressManager, get_progress_manager
from .openrouter_backend import OpenRouterBackend

# Evolution engine
from .evolution import PhiEvolutionEngine, EvolutionState
from .phi_attention import PhiAttentionInjector, PhiConfidenceScaler
from .phi_sparse_reasoning import PhiSparseReasoner
from .phi_memory import PhiSpiralMemory
from .session_memory import SessionMemory
from .meta_recursion import MetaRecursiveRLM, RecursionStrategy
from .phi_retrieval import phi_kernel_similarity, phi_retrieval_score
from .phi_bayesian import PhiBayesianOptimizer
from .ensemble_backend import EnsembleBackend
from .streaming import ReasoningEvent, recursive_solve_stream

# Agent integration (v4.0)
from .agent_router import AgentRouter, DEPTH_AGENT_MAP
from .skill_loader import SkillLoader

# Autonomy modules (v4.1)
from .tool_executor import ToolRegistry, ToolResult
from .outcome_tracker import OutcomeTracker
from .phi_critic import PhiCritic, CritiqueResult
from .phi_planner import PhiPlanner, ExecutionPlan

# Vector store (optional - requires chromadb)
try:
    from .vector_store import VectorStore, RLMPipeline, Document, QueryResult  # noqa: F401
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False

__version__ = "4.1.0"
__all__ = [
    # Core RLM
    "PhiEnhancedRLM",
    "ContextChunk",
    "SubCallResult",
    "LLMResponse",
    # Mathematics
    "PHI",
    "PHI_INV",
    "LOG_PHI",
    "PhiGramMatrix",
    "SpectralFlow",
    "PhiRenormalizationGroup",
    # Embeddings & Cache
    "get_embedder",
    "CachedEmbedder",
    "EmbeddingConfig",
    "SQLiteEmbeddingCache",
    "get_sqlite_cache",
    # Extractors
    "extract_pdf_content",
    "extract_docx_content",
    "extract_web_content",
    "chunk_python_code",
    "chunk_code_file",
    # Progress
    "RichProgressManager",
    "get_progress_manager",
    # Backend
    "OpenRouterBackend",
    # Evolution (v3.0)
    "PhiEvolutionEngine",
    "EvolutionState",
    # φ-Attention (v3.0)
    "PhiAttentionInjector",
    "PhiConfidenceScaler",
    # φ-Sparse Reasoning (v3.0)
    "PhiSparseReasoner",
    # φ-Spiral Memory (v3.0)
    "PhiSpiralMemory",
    "SessionMemory",
    # Meta-Recursion (v3.0)
    "MetaRecursiveRLM",
    "RecursionStrategy",
    # φ-Kernel Retrieval (v3.0)
    "phi_kernel_similarity",
    "phi_retrieval_score",
    # φ-Bayesian Optimization (v3.0)
    "PhiBayesianOptimizer",
    # Ensemble Backend (v3.0)
    "EnsembleBackend",
    # Streaming (v3.0)
    "ReasoningEvent",
    "recursive_solve_stream",
    # Agent Integration (v4.0)
    "AgentRouter",
    "DEPTH_AGENT_MAP",
    "SkillLoader",
    # Autonomy (v4.1)
    "ToolRegistry",
    "ToolResult",
    "OutcomeTracker",
    "PhiCritic",
    "CritiqueResult",
    "PhiPlanner",
    "ExecutionPlan",
]
