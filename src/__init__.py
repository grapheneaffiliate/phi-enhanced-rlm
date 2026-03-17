# -*- coding: utf-8 -*-
"""
PHI-Enhanced RLM Core Library -- v4.1.0

Self-evolving recursive language model with phi-geometric mathematics,
multi-agent integration, tool execution, and adversarial self-evaluation.
"""

# Agent integration (v4.0)
from .agent_router import DEPTH_AGENT_MAP, AgentRouter

# PRA controller (v5.2)
from .pra_controller import PRAControlState, PRAController
from .cache import SQLiteEmbeddingCache, get_sqlite_cache
from .code_review import PhiCodeReview, ReviewFinding, ReviewResult
from .embeddings import CachedEmbedder, EmbeddingConfig, get_embedder
from .ensemble_backend import EnsembleBackend

# Evolution engine
from .evolution import EvolutionState, PhiEvolutionEngine
from .extractors import (
    chunk_code_file,
    chunk_python_code,
    extract_docx_content,
    extract_pdf_content,
    extract_web_content,
)
from .meta_recursion import MetaRecursiveRLM, RecursionStrategy
from .openrouter_backend import OpenRouterBackend
from .outcome_tracker import OutcomeTracker
from .phi_attention import PhiAttentionInjector, PhiConfidenceScaler
from .phi_bayesian import PhiBayesianOptimizer
from .phi_critic import CritiqueResult, PhiCritic
from .phi_enhanced_rlm import ContextChunk, LLMResponse, PhiEnhancedRLM, SubCallResult
from .phi_memory import PhiSpiralMemory
from .phi_planner import ExecutionPlan, PhiPlanner
from .phi_retrieval import phi_kernel_similarity, phi_retrieval_score
from .phi_separation_novel_mathematics import (
    LOG_PHI,
    PHI,
    PHI_INV,
    PhiGramMatrix,
    PhiRenormalizationGroup,
    SpectralFlow,
)
from .phi_sparse_reasoning import PhiSparseReasoner
from .progress import RichProgressManager, get_progress_manager
from .session_memory import SessionMemory
from .skill_loader import SkillLoader
from .streaming import ReasoningEvent, recursive_solve_stream

# Autonomy modules (v4.1)
from .tool_executor import ToolRegistry, ToolResult
from .workflow_orchestrator import WORKFLOW_PIPELINE, SuperpowersOrchestrator

# Vector store (optional - requires chromadb)
try:
    from .vector_store import Document, QueryResult, RLMPipeline, VectorStore  # noqa: F401
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False

__version__ = "5.1.0"  # GSM Complete Unification — Zero Weaknesses Release
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
    # Superpowers Integration (v4.2)
    "SuperpowersOrchestrator",
    "WORKFLOW_PIPELINE",
    # Code Review (v4.2)
    "PhiCodeReview",
    "ReviewResult",
    "ReviewFinding",
    # PRA Controller (v5.2)
    "PRAController",
    "PRAControlState",
]
