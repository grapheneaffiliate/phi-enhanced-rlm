#!/usr/bin/env python3
"""
EMBEDDINGS MODULE FOR PHI-ENHANCED RLM
======================================
Real embeddings with multiple providers and caching.

Providers:
1. OpenAI (via OpenRouter or direct)
2. Sentence-Transformers (local, free)
3. Mock (fallback for testing)

Usage:
    from embeddings import get_embedder, EmbeddingConfig
    
    # Auto-detect best available provider
    embedder = get_embedder()
    
    # Or specify provider
    embedder = get_embedder(EmbeddingConfig(provider="openai"))
    
    # Get embeddings
    vectors = embedder.embed(["text 1", "text 2"])
"""

import os
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""
    provider: str = "auto"  # "auto", "openai", "local", "mock"
    
    # OpenAI/OpenRouter settings
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openai/text-embedding-3-small"
    
    # Local model settings
    local_model: str = "all-MiniLM-L6-v2"
    
    # Caching
    cache_enabled: bool = True
    cache_dir: Optional[str] = None
    
    # Dimensions (for mock/padding)
    dimensions: int = 384
    
    @classmethod
    def from_env(cls) -> 'EmbeddingConfig':
        """Load config from environment variables."""
        return cls(
            provider=os.getenv("EMBEDDING_PROVIDER", "auto"),
            api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("EMBEDDING_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small"),
            local_model=os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            cache_enabled=os.getenv("EMBEDDING_CACHE", "true").lower() == "true",
            cache_dir=os.getenv("EMBEDDING_CACHE_DIR"),
        )


# =============================================================================
# CACHE
# =============================================================================

class EmbeddingCache:
    """Simple file-based embedding cache."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "phi_rlm_embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict = {}
    
    def _hash_text(self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """Get cached embedding if exists."""
        key = self._hash_text(text, model)
        
        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Check file cache
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            try:
                embedding = np.load(cache_file)
                self._memory_cache[key] = embedding
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        
        return None
    
    def set(self, text: str, model: str, embedding: np.ndarray):
        """Cache an embedding."""
        key = self._hash_text(text, model)
        self._memory_cache[key] = embedding
        
        # Save to file
        cache_file = self.cache_dir / f"{key}.npy"
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def get_batch(self, texts: List[str], model: str) -> tuple[List[Optional[np.ndarray]], List[int]]:
        """Get cached embeddings, return (results, missing_indices)."""
        results = []
        missing = []
        
        for i, text in enumerate(texts):
            cached = self.get(text, model)
            results.append(cached)
            if cached is None:
                missing.append(i)
        
        return results, missing


# =============================================================================
# EMBEDDING PROVIDERS
# =============================================================================

class EmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts, return (n_texts, dimensions) array."""
        pass
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model identifier for caching."""
        pass


class OpenAIEmbedder(EmbeddingProvider):
    """OpenAI/OpenRouter embedding provider."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._dimensions = 1536 if "3-large" in config.model else 384
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI/OpenRouter."""
        if not texts:
            return np.array([])
        
        # Handle model name for OpenRouter vs direct OpenAI
        model = self.config.model
        if self.config.base_url == "https://openrouter.ai/api/v1":
            # OpenRouter format
            if not model.startswith("openai/"):
                model = f"openai/{model}"
        else:
            # Direct OpenAI - strip provider prefix
            if "/" in model:
                model = model.split("/")[-1]
        
        response = self.client.embeddings.create(
            model=model,
            input=texts,
        )
        
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        return self.config.model


class LocalEmbedder(EmbeddingProvider):
    """Local sentence-transformers embedding provider."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(config.local_model)
            self._dimensions = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers required: pip install sentence-transformers"
            )
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Get embeddings locally."""
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        return f"local:{self.config.local_model}"


class MockEmbedder(EmbeddingProvider):
    """Mock embedder using deterministic hashing (for testing)."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._dimensions = config.dimensions
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings from text hashes."""
        embeddings = []
        for text in texts:
            h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
            seed = int.from_bytes(h, "little")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self._dimensions)
            v = v / (np.linalg.norm(v) + 1e-12)
            embeddings.append(v)
        return np.array(embeddings)
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        return "mock"


# =============================================================================
# CACHED EMBEDDER WRAPPER
# =============================================================================

class CachedEmbedder:
    """Wrapper that adds caching to any embedding provider."""
    
    def __init__(self, provider: EmbeddingProvider, cache: Optional[EmbeddingCache] = None):
        self.provider = provider
        self.cache = cache
        self._stats = {"hits": 0, "misses": 0, "api_calls": 0}
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Get embeddings with caching."""
        if not texts:
            return np.array([]).reshape(0, self.provider.dimensions)
        
        if self.cache is None:
            self._stats["misses"] += len(texts)
            self._stats["api_calls"] += 1
            return self.provider.embed(texts)
        
        # Check cache
        cached_results, missing_indices = self.cache.get_batch(texts, self.provider.model_name)
        
        self._stats["hits"] += len(texts) - len(missing_indices)
        self._stats["misses"] += len(missing_indices)
        
        # Fetch missing embeddings
        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            self._stats["api_calls"] += 1
            new_embeddings = self.provider.embed(missing_texts)
            
            # Update cache and results
            for idx, (i, embedding) in enumerate(zip(missing_indices, new_embeddings)):
                self.cache.set(texts[i], self.provider.model_name, embedding)
                cached_results[i] = embedding
        
        return np.array(cached_results)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed([text])[0]
    
    @property
    def dimensions(self) -> int:
        return self.provider.dimensions
    
    @property
    def stats(self) -> dict:
        return self._stats.copy()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_embedder(config: Optional[EmbeddingConfig] = None) -> CachedEmbedder:
    """
    Get the best available embedder.
    
    Args:
        config: Optional configuration. If None, loads from environment.
    
    Returns:
        CachedEmbedder instance
    """
    if config is None:
        config = EmbeddingConfig.from_env()
    
    provider = None
    
    if config.provider == "auto":
        # Try providers in order of preference
        
        # 1. Try OpenAI/OpenRouter if API key available
        if config.api_key:
            try:
                provider = OpenAIEmbedder(config)
                logger.info(f"Using OpenAI embeddings: {config.model}")
            except Exception as e:
                logger.warning(f"OpenAI embeddings unavailable: {e}")
        
        # 2. Try local sentence-transformers
        if provider is None:
            try:
                provider = LocalEmbedder(config)
                logger.info(f"Using local embeddings: {config.local_model}")
            except ImportError:
                logger.warning("sentence-transformers not installed")
        
        # 3. Fall back to mock
        if provider is None:
            logger.warning("No embedding provider available, using mock embeddings")
            provider = MockEmbedder(config)
    
    elif config.provider == "openai":
        provider = OpenAIEmbedder(config)
    
    elif config.provider == "local":
        provider = LocalEmbedder(config)
    
    else:  # "mock" or unknown
        provider = MockEmbedder(config)
    
    # Wrap with cache if enabled
    cache = EmbeddingCache(config.cache_dir) if config.cache_enabled else None
    
    return CachedEmbedder(provider, cache)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all embeddings."""
    query_norm = query / (np.linalg.norm(query) + 1e-12)
    emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    return np.dot(emb_norms, query_norm)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate embedding functionality."""
    print("=" * 60)
    print("EMBEDDING MODULE DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Get embedder
    embedder = get_embedder()
    print(f"Provider: {embedder.provider.__class__.__name__}")
    print(f"Dimensions: {embedder.dimensions}")
    print()
    
    # Test texts
    texts = [
        "The golden ratio φ = 1.618 appears throughout mathematics.",
        "E8 is the largest exceptional Lie group with 248 dimensions.",
        "Machine learning uses neural networks for pattern recognition.",
        "The weather today is sunny and warm.",
    ]
    
    print("Embedding texts...")
    embeddings = embedder.embed(texts)
    print(f"Shape: {embeddings.shape}")
    print()
    
    # Show similarities
    print("Pairwise similarities:")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"  [{i}] vs [{j}]: {sim:.4f}")
    print()
    
    # Test caching
    print("Testing cache (re-embedding same texts)...")
    embedder.embed(texts)
    print(f"Cache stats: {embedder.stats}")
    print()
    
    # Query similarity
    query = "What is the golden ratio and its significance?"
    print(f"Query: {query}")
    query_emb = embedder.embed_single(query)
    similarities = batch_cosine_similarity(query_emb, embeddings)
    
    print("Relevance scores:")
    for i, (text, sim) in enumerate(zip(texts, similarities)):
        print(f"  {sim:.4f}: {text[:50]}...")
    
    print()
    print("✓ Demo complete!")


if __name__ == "__main__":
    demo()
