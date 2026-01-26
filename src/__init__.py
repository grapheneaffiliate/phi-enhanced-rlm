# -*- coding: utf-8 -*-
"""
PHI-Enhanced RLM Core Library
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

# Vector store (optional - requires chromadb)
try:
    from .vector_store import VectorStore, RLMPipeline, Document, QueryResult
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False

__version__ = "2.1.0"
__all__ = [
    "PhiEnhancedRLM",
    "ContextChunk", 
    "SubCallResult",
    "LLMResponse",
    "PHI",
    "PHI_INV",
    "LOG_PHI",
    "PhiGramMatrix",
    "SpectralFlow",
    "PhiRenormalizationGroup",
    "get_embedder",
    "CachedEmbedder",
    "EmbeddingConfig",
    "SQLiteEmbeddingCache",
    "get_sqlite_cache",
    "extract_pdf_content",
    "extract_docx_content",
    "extract_web_content",
    "chunk_python_code",
    "chunk_code_file",
    "RichProgressManager",
    "get_progress_manager",
    "OpenRouterBackend",
]
