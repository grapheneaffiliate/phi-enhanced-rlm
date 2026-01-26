#!/usr/bin/env python3
"""
PERSISTENT SQLITE EMBEDDING CACHE
=================================
High-performance SQLite-backed cache for embeddings.
Survives restarts and enables rapid repeated analysis.
"""

import sqlite3
import hashlib
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import logging
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    writes: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0


class SQLiteEmbeddingCache:
    """
    SQLite-backed embedding cache with thread safety and automatic cleanup.
    
    Features:
    - Persistent storage across sessions
    - Thread-safe operations
    - Automatic old entry cleanup
    - Batch operations for efficiency
    - Compression for large embeddings
    """
    
    def __init__(self, 
                 db_path: Optional[str] = None,
                 max_entries: int = 100000,
                 max_age_days: int = 30):
        """
        Initialize SQLite cache.
        
        Args:
            db_path: Path to SQLite database (default: ~/.cache/phi_rlm/embeddings.db)
            max_entries: Maximum cache entries before cleanup
            max_age_days: Maximum age of entries before expiration
        """
        if db_path is None:
            cache_dir = Path.home() / ".cache" / "phi_rlm"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = cache_dir / "embeddings.db"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.max_entries = max_entries
        self.max_age_seconds = max_age_days * 24 * 60 * 60
        self._local = threading.local()
        self._stats = CacheStats()
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn
    
    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_accessed ON embeddings(last_accessed)
        """)
        conn.commit()
        logger.info(f"SQLite cache initialized: {self.db_path}")
    
    @staticmethod
    def _hash_text(text: str, model: str) -> str:
        """Generate cache key from text and model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """
        Get cached embedding.
        
        Args:
            text: The text that was embedded
            model: Model identifier
            
        Returns:
            Embedding array or None if not cached
        """
        key = self._hash_text(text, model)
        conn = self._get_conn()
        
        cursor = conn.execute(
            "SELECT embedding, dimensions FROM embeddings WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()
        
        if row is None:
            with self._lock:
                self._stats.misses += 1
            return None
        
        # Update access time and count
        now = time.time()
        conn.execute(
            "UPDATE embeddings SET last_accessed = ?, access_count = access_count + 1 WHERE key = ?",
            (now, key)
        )
        conn.commit()
        
        with self._lock:
            self._stats.hits += 1
        
        # Deserialize embedding
        embedding = np.frombuffer(row[0], dtype=np.float32).reshape(-1)
        return embedding
    
    def set(self, text: str, model: str, embedding: np.ndarray):
        """
        Cache an embedding.
        
        Args:
            text: The text that was embedded
            model: Model identifier
            embedding: Embedding vector
        """
        key = self._hash_text(text, model)
        now = time.time()
        
        # Serialize embedding
        blob = embedding.astype(np.float32).tobytes()
        
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO embeddings 
            (key, model, embedding, dimensions, created_at, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        """, (key, model, blob, len(embedding), now, now))
        conn.commit()
        
        with self._lock:
            self._stats.writes += 1
    
    def get_batch(self, texts: List[str], model: str) -> Tuple[List[Optional[np.ndarray]], List[int]]:
        """
        Get cached embeddings for multiple texts.
        
        Args:
            texts: List of texts
            model: Model identifier
            
        Returns:
            Tuple of (results list, missing indices)
        """
        results = []
        missing = []
        
        for i, text in enumerate(texts):
            emb = self.get(text, model)
            results.append(emb)
            if emb is None:
                missing.append(i)
        
        return results, missing
    
    def set_batch(self, texts: List[str], model: str, embeddings: np.ndarray):
        """
        Cache multiple embeddings.
        
        Args:
            texts: List of texts
            model: Model identifier  
            embeddings: Array of embeddings (n_texts, dimensions)
        """
        now = time.time()
        conn = self._get_conn()
        
        for text, embedding in zip(texts, embeddings):
            key = self._hash_text(text, model)
            blob = embedding.astype(np.float32).tobytes()
            conn.execute("""
                INSERT OR REPLACE INTO embeddings 
                (key, model, embedding, dimensions, created_at, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, 1)
            """, (key, model, blob, len(embedding), now, now))
        
        conn.commit()
        
        with self._lock:
            self._stats.writes += len(texts)
    
    def cleanup(self, force: bool = False):
        """
        Remove old/excess entries.
        
        Args:
            force: Force cleanup even if not needed
        """
        conn = self._get_conn()
        
        # Get current count
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        
        if not force and count < self.max_entries:
            return
        
        logger.info(f"Running cache cleanup ({count} entries)")
        
        # Delete old entries
        cutoff = time.time() - self.max_age_seconds
        conn.execute("DELETE FROM embeddings WHERE last_accessed < ?", (cutoff,))
        
        # If still too many, delete least accessed
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        
        if count > self.max_entries:
            delete_count = count - int(self.max_entries * 0.8)  # Keep 80% capacity
            conn.execute("""
                DELETE FROM embeddings WHERE key IN (
                    SELECT key FROM embeddings 
                    ORDER BY access_count ASC, last_accessed ASC 
                    LIMIT ?
                )
            """, (delete_count,))
        
        conn.execute("VACUUM")
        conn.commit()
        
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        new_count = cursor.fetchone()[0]
        logger.info(f"Cache cleanup complete: {count} -> {new_count} entries")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        conn = self._get_conn()
        
        cursor = conn.execute("SELECT COUNT(*), SUM(LENGTH(embedding)) FROM embeddings")
        row = cursor.fetchone()
        
        stats = CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            writes=self._stats.writes,
            entry_count=row[0] or 0,
            total_size_bytes=row[1] or 0
        )
        return stats
    
    def clear(self):
        """Clear all cached entries."""
        conn = self._get_conn()
        conn.execute("DELETE FROM embeddings")
        conn.execute("VACUUM")
        conn.commit()
        logger.info("Cache cleared")
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# Global cache instance
_global_cache: Optional[SQLiteEmbeddingCache] = None


def get_sqlite_cache() -> SQLiteEmbeddingCache:
    """Get or create global SQLite cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SQLiteEmbeddingCache()
    return _global_cache


if __name__ == "__main__":
    # Demo
    print("SQLite Embedding Cache Demo")
    print("=" * 50)
    
    cache = SQLiteEmbeddingCache()
    
    # Test set/get
    text = "Hello, world!"
    model = "test-model"
    embedding = np.random.randn(384).astype(np.float32)
    
    cache.set(text, model, embedding)
    retrieved = cache.get(text, model)
    
    print(f"Original shape: {embedding.shape}")
    print(f"Retrieved shape: {retrieved.shape}")
    print(f"Match: {np.allclose(embedding, retrieved)}")
    
    # Stats
    stats = cache.get_stats()
    print(f"\nCache stats:")
    print(f"  Hits: {stats.hits}")
    print(f"  Misses: {stats.misses}")
    print(f"  Entries: {stats.entry_count}")
    print(f"  Size: {stats.total_size_bytes / 1024:.1f} KB")
