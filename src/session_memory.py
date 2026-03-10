#!/usr/bin/env python3
"""
Cross-Session Memory — persist what the model learned across conversations.

Stores session traces, results, and evolution states so that:
1. The model can learn from past sessions
2. Frequently asked question types get faster responses
3. Failed strategies are avoided in future sessions
"""

import json
import hashlib
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any

from phi_separation_novel_mathematics import PHI_INV

logger = logging.getLogger(__name__)


@dataclass
class SessionRecord:
    """Record of a single session."""
    session_id: str
    query: str
    answer: str
    confidence: float
    recursion_depth: int
    tokens_used: int
    time_seconds: float
    strategy: str = ""
    success: bool = False
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionRecord':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SessionMemory:
    """Persist and retrieve learning from past sessions."""

    def __init__(self, path: str = "sessions/"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.path / "index.json"
        self.index: List[Dict[str, Any]] = self._load_index()

    def _load_index(self) -> List[Dict[str, Any]]:
        """Load session index."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass
        return []

    def _save_index(self):
        """Save session index."""
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)

    def save_session(self, trace: List[Dict], result: Dict,
                     evolution_state: Optional[Dict] = None) -> str:
        """
        Save session data for future learning.

        Args:
            trace: List of trace entries from the recursive solve
            result: Final result dictionary
            evolution_state: Current evolution state (optional)

        Returns:
            Session ID
        """
        session_id = hashlib.sha256(
            f"{time.time()}{json.dumps(result)[:100]}".encode()
        ).hexdigest()[:12]

        session_data = {
            "session_id": session_id,
            "timestamp": time.time(),
            "trace": trace,
            "result": result,
            "evolution_state": evolution_state,
        }

        session_file = self.path / f"session_{session_id}.json"
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)

        # Update index
        self.index.append({
            "session_id": session_id,
            "timestamp": session_data["timestamp"],
            "query": result.get("query", ""),
            "confidence": result.get("confidence", 0.0),
            "success": result.get("confidence", 0.0) > 0.7,
        })
        self._save_index()

        logger.info(f"Saved session {session_id}")
        return session_id

    def load_session(self, session_id: str) -> Optional[Dict]:
        """Load a specific session by ID."""
        session_file = self.path / f"session_{session_id}.json"
        if session_file.exists():
            with open(session_file, "r") as f:
                return json.load(f)
        return None

    def load_relevant_sessions(self, query: str, k: int = 3) -> List[Dict]:
        """
        Find past sessions relevant to current query.

        Uses word overlap as a simple relevance metric.
        """
        query_words = set(query.lower().split())
        scored = []

        for entry in self.index:
            past_query = entry.get("query", "")
            past_words = set(past_query.lower().split())

            if not query_words or not past_words:
                overlap = 0.0
            else:
                overlap = len(query_words & past_words) / len(query_words | past_words)

            # Recency bonus: recent sessions are more relevant
            age = time.time() - entry.get("timestamp", 0)
            recency = PHI_INV ** (age / 86400)  # Decay by day

            # Success bonus
            success_bonus = 0.2 if entry.get("success", False) else 0.0

            score = overlap * 0.5 + recency * 0.3 + success_bonus * 0.2
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Load full session data for top-k
        results = []
        for _, entry in scored[:k]:
            session_data = self.load_session(entry["session_id"])
            if session_data:
                results.append(session_data)

        return results

    def get_strategy_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregate statistics about which strategies work best.

        Returns strategy_name -> {accuracy, avg_confidence, count}.
        """
        strategy_data: Dict[str, List[Dict]] = {}

        for entry in self.index:
            session = self.load_session(entry["session_id"])
            if not session:
                continue

            evo_state = session.get("evolution_state", {})
            strategy = "default"
            if evo_state:
                # Use the best chunk strategy as the strategy identifier
                scores = evo_state.get("chunk_strategy_scores", {})
                if scores:
                    strategy = max(scores, key=scores.get)

            if strategy not in strategy_data:
                strategy_data[strategy] = []
            strategy_data[strategy].append({
                "confidence": entry.get("confidence", 0.0),
                "success": entry.get("success", False),
            })

        stats = {}
        for strategy, records in strategy_data.items():
            n = len(records)
            stats[strategy] = {
                "count": n,
                "accuracy": sum(1 for r in records if r["success"]) / n if n else 0,
                "avg_confidence": sum(r["confidence"] for r in records) / n if n else 0,
            }

        return stats

    def get_total_sessions(self) -> int:
        return len(self.index)

    def cleanup_old_sessions(self, max_age_days: int = 30):
        """Remove sessions older than max_age_days."""
        cutoff = time.time() - max_age_days * 86400
        to_remove = []

        for entry in self.index:
            if entry.get("timestamp", 0) < cutoff:
                to_remove.append(entry)
                session_file = self.path / f"session_{entry['session_id']}.json"
                if session_file.exists():
                    session_file.unlink()

        for entry in to_remove:
            self.index.remove(entry)

        self._save_index()
        logger.info(f"Cleaned up {len(to_remove)} old sessions")
