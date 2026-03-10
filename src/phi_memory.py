#!/usr/bin/env python3
"""
φ-Spiral Memory — golden ratio organized persistent memory.

Memory organized in golden spiral topology:
- Recent/important memories at center (low radius, high impedance)
- Older/peripheral memories at outer spiral (high radius, low impedance)
- Retrieval priority follows impedance funneling: center → edge

This follows the same golden spiral geometry found in:
- Phyllotaxis (leaf arrangement)
- Galaxy arm structure
- The φ-GEH impedance funnel

Key principles:
1. Store: Assign φ-radius based on recency and importance
2. Retrieve: Score = cosine_sim × φ^(-radius/max_radius)
3. Consolidate: Merge similar, prune old, promote frequent
"""

import json
import time
import hashlib
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

from phi_separation_novel_mathematics import PHI, PHI_INV, LOG_PHI

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory in the φ-spiral."""
    key: str  # Unique identifier
    text: str  # The content
    embedding: Optional[List[float]] = None  # Vector representation
    radius: float = 0.0  # Position in spiral (lower = more central)
    importance: float = 0.5  # Importance score [0, 1]
    access_count: int = 0  # How often retrieved
    created_at: float = 0.0  # Unix timestamp
    last_accessed: float = 0.0  # Unix timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)

    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PhiSpiralMemory:
    """
    Memory organized in golden spiral topology.

    Recent/important memories at center (low radius, high impedance).
    Older/peripheral memories at outer spiral (high radius, low impedance).
    Retrieval priority follows impedance funneling: center → edge.
    """

    def __init__(self, db_path: str = "phi_memory.json",
                 max_memories: int = 10000,
                 consolidation_threshold: float = 0.85):
        """
        Args:
            db_path: Path to persistent memory file
            max_memories: Maximum memories before forced pruning
            consolidation_threshold: Cosine similarity threshold for merging
        """
        self.db_path = Path(db_path)
        self.max_memories = max_memories
        self.consolidation_threshold = consolidation_threshold
        self.memories: Dict[str, MemoryEntry] = {}
        self._load()

    def _load(self):
        """Load memories from disk."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                for key, entry_data in data.items():
                    self.memories[key] = MemoryEntry.from_dict(entry_data)
                logger.info(f"Loaded {len(self.memories)} memories from {self.db_path}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load memories: {e}")

    def _save(self):
        """Persist memories to disk."""
        data = {key: entry.to_dict() for key, entry in self.memories.items()}
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2)

    def _compute_key(self, text: str) -> str:
        """Generate unique key for memory entry."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _assign_radius(self, importance: float = 0.5) -> float:
        """
        Assign φ-spiral radius based on current state.

        New memories start at small radius (center of spiral).
        Radius = φ^(rank) where rank is based on insertion order.
        More important memories get smaller radius.
        """
        # Base radius from number of existing memories
        n = len(self.memories)
        base_radius = LOG_PHI * n

        # Adjust by importance (higher importance = smaller radius = more central)
        importance_factor = 1.0 - importance  # Invert: high importance → low radius
        radius = base_radius * (0.5 + importance_factor * 0.5)

        return radius

    def store(self, text: str, embedding: Optional[np.ndarray] = None,
              importance: float = 0.5,
              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a memory with φ-radius assignment.

        Args:
            text: The content to store
            embedding: Vector representation (optional)
            importance: How important this memory is [0, 1]
            metadata: Additional metadata

        Returns:
            Key of the stored memory
        """
        key = self._compute_key(text)
        now = time.time()

        # If memory already exists, update it
        if key in self.memories:
            existing = self.memories[key]
            existing.access_count += 1
            existing.last_accessed = now
            # Promote: decrease radius for frequently accessed memories
            existing.radius *= PHI_INV
            existing.importance = max(existing.importance, importance)
            self._save()
            return key

        # Create new memory
        emb_list = embedding.tolist() if embedding is not None else None
        radius = self._assign_radius(importance)

        entry = MemoryEntry(
            key=key,
            text=text,
            embedding=emb_list,
            radius=radius,
            importance=importance,
            access_count=0,
            created_at=now,
            last_accessed=now,
            metadata=metadata or {},
        )

        self.memories[key] = entry

        # Prune if over capacity
        if len(self.memories) > self.max_memories:
            self._prune()

        self._save()
        return key

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[MemoryEntry]:
        """
        Retrieve using φ-weighted similarity.

        Score = cosine_sim × φ^(-radius/max_radius)
        This naturally prioritizes recent + relevant over old + relevant.
        """
        if not self.memories:
            return []

        # Find max radius for normalization
        max_radius = max(m.radius for m in self.memories.values()) or 1.0

        scored = []
        for key, memory in self.memories.items():
            if memory.embedding is None:
                continue

            emb = np.array(memory.embedding)
            # Cosine similarity
            norm_q = np.linalg.norm(query_embedding)
            norm_e = np.linalg.norm(emb)
            if norm_q == 0 or norm_e == 0:
                cos_sim = 0.0
            else:
                cos_sim = float(np.dot(query_embedding, emb) / (norm_q * norm_e))

            # φ-radius weighting: central memories get higher priority
            radius_weight = PHI ** (-memory.radius / max_radius)

            # Access frequency bonus
            access_bonus = min(memory.access_count * 0.01, 0.1)

            score = cos_sim * radius_weight + access_bonus
            scored.append((score, key, memory))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Update access counts for retrieved memories
        results = []
        for score, key, memory in scored[:k]:
            memory.access_count += 1
            memory.last_accessed = time.time()
            # Promote accessed memories (decrease radius)
            memory.radius *= (1 - 0.01 * PHI_INV)
            results.append(memory)

        self._save()
        return results

    def retrieve_by_text(self, query: str, k: int = 5) -> List[MemoryEntry]:
        """
        Simple text-based retrieval using word overlap.

        Use when embeddings are not available.
        """
        if not self.memories:
            return []

        query_words = set(query.lower().split())

        scored = []
        for key, memory in self.memories.items():
            mem_words = set(memory.text.lower().split())
            if not query_words or not mem_words:
                overlap = 0.0
            else:
                overlap = len(query_words & mem_words) / len(query_words | mem_words)

            max_radius = max(m.radius for m in self.memories.values()) or 1.0
            radius_weight = PHI ** (-memory.radius / max_radius)

            score = overlap * radius_weight
            scored.append((score, memory))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:k]]

    def consolidate(self):
        """
        Periodic memory consolidation.

        1. Merge similar memories (φ-Gram collision detection)
        2. Prune memories with large radius that are never accessed
        3. Promote frequently-accessed memories to smaller radius
        """
        if not self.memories:
            return

        # 1. Merge similar memories
        keys = list(self.memories.keys())
        merged = set()

        for i in range(len(keys)):
            if keys[i] in merged:
                continue
            for j in range(i + 1, len(keys)):
                if keys[j] in merged:
                    continue

                m1 = self.memories[keys[i]]
                m2 = self.memories[keys[j]]

                if m1.embedding and m2.embedding:
                    emb1 = np.array(m1.embedding)
                    emb2 = np.array(m2.embedding)
                    n1, n2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
                    if n1 > 0 and n2 > 0:
                        sim = np.dot(emb1, emb2) / (n1 * n2)
                        if sim > self.consolidation_threshold:
                            # Merge: keep the more important one
                            if m1.importance >= m2.importance:
                                m1.access_count += m2.access_count
                                m1.text = f"{m1.text}\n[Merged]: {m2.text}"
                                merged.add(keys[j])
                            else:
                                m2.access_count += m1.access_count
                                m2.text = f"{m2.text}\n[Merged]: {m1.text}"
                                merged.add(keys[i])
                                break

        for key in merged:
            del self.memories[key]

        # 2. Promote frequently accessed memories
        for memory in self.memories.values():
            if memory.access_count > 5:
                memory.radius *= PHI_INV  # Move toward center

        # 3. Age-based radius increase (memories drift outward over time)
        now = time.time()
        for memory in self.memories.values():
            age_days = (now - memory.created_at) / 86400
            if age_days > 7 and memory.access_count == 0:
                memory.radius *= PHI  # Push outward

        self._save()
        logger.info(f"Consolidated: merged {len(merged)} memories, "
                     f"{len(self.memories)} remaining")

    def _prune(self):
        """Remove memories with the largest radius (least central/important)."""
        if len(self.memories) <= self.max_memories:
            return

        # Sort by radius (largest first — furthest from center)
        sorted_keys = sorted(
            self.memories.keys(),
            key=lambda k: self.memories[k].radius,
            reverse=True,
        )

        # Remove the most peripheral memories
        n_remove = len(self.memories) - self.max_memories
        for key in sorted_keys[:n_remove]:
            del self.memories[key]

        logger.info(f"Pruned {n_remove} memories, {len(self.memories)} remaining")

    def get_stats(self) -> Dict[str, Any]:
        """Return memory system statistics."""
        if not self.memories:
            return {"total": 0}

        radii = [m.radius for m in self.memories.values()]
        accesses = [m.access_count for m in self.memories.values()]

        return {
            "total": len(self.memories),
            "min_radius": min(radii),
            "max_radius": max(radii),
            "mean_radius": np.mean(radii),
            "total_accesses": sum(accesses),
            "avg_accesses": np.mean(accesses),
            "with_embeddings": sum(1 for m in self.memories.values() if m.embedding),
        }
