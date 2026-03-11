#!/usr/bin/env python3
"""
Outcome Tracker -- record real-world feedback on answer quality.

Supports: thumbs up/down, rating 1-5, task completion confirmation,
and correction text. Feeds into evolution as a weighted fitness signal
using phi-geometric recency weighting.

The key insight: benchmark accuracy with mock backends stays at 0%.
Real feedback from users provides the actual fitness signal that
drives meaningful self-evolution.
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .phi_separation_novel_mathematics import PHI_INV

logger = logging.getLogger(__name__)


@dataclass
class Outcome:
    """A single feedback record for a query-answer pair."""
    query_hash: str
    timestamp: float
    rating: float          # 0.0-1.0 normalized
    task_completed: bool
    correction: str = ""   # What the right answer should have been
    strategy_used: str = ""
    depth_reached: int = 0
    agent_used: str = ""   # Which agent persona handled this
    metadata: Dict[str, Any] = field(default_factory=dict)


class OutcomeTracker:
    """Records and aggregates real-world feedback for evolution fitness.

    Outcomes are persisted to disk and provide a phi-weighted fitness
    signal: more recent outcomes have exponentially more influence,
    with the decay rate governed by phi^(-1).
    """

    def __init__(self, path: str = "outcomes.json"):
        self.path = Path(path)
        self.outcomes: List[Outcome] = []
        self._load()

    def _load(self):
        """Load existing outcomes from disk."""
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self.outcomes = [Outcome(**o) for o in data]
                logger.info(f"Loaded {len(self.outcomes)} outcomes")
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to load outcomes: {e}")

    def _save(self):
        """Persist outcomes to disk."""
        with open(self.path, "w") as f:
            json.dump([asdict(o) for o in self.outcomes], f, indent=2)

    @staticmethod
    def hash_query(query: str) -> str:
        """Create a stable hash for a query."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def record(self, query: str, rating: float,
               task_completed: bool = False,
               correction: str = "",
               strategy: str = "",
               depth: int = 0,
               agent: str = "",
               metadata: Optional[Dict[str, Any]] = None):
        """Record a feedback outcome.

        Args:
            query: The original query text.
            rating: Quality rating, 0.0 (terrible) to 1.0 (perfect).
            task_completed: Whether the task was actually completed.
            correction: What the correct answer should have been.
            strategy: Which meta-recursion strategy was used.
            depth: Maximum recursion depth reached.
            agent: Which agent persona handled this.
            metadata: Additional context.
        """
        outcome = Outcome(
            query_hash=self.hash_query(query),
            timestamp=time.time(),
            rating=max(0.0, min(1.0, rating)),
            task_completed=task_completed,
            correction=correction,
            strategy_used=strategy,
            depth_reached=depth,
            agent_used=agent,
            metadata=metadata or {},
        )
        self.outcomes.append(outcome)
        self._save()
        logger.info(f"Recorded outcome: rating={rating:.2f}, "
                     f"completed={task_completed}")

    def get_fitness_signal(self, window: int = 50) -> float:
        """Compute phi-weighted fitness from recent outcomes.

        Uses exponential recency weighting with phi^(-1) decay:
        newer outcomes matter more than older ones.

        Args:
            window: Maximum number of recent outcomes to consider.

        Returns:
            Weighted average rating in [0, 1]. Returns 0.5 if no data.
        """
        if not self.outcomes:
            return 0.5

        recent = self.outcomes[-window:]
        # phi^(-i) weighting: most recent = highest weight
        weights = [PHI_INV ** i for i in range(len(recent))]
        total_w = sum(weights)
        weighted_sum = sum(
            o.rating * w for o, w in zip(reversed(recent), weights)
        )
        return weighted_sum / total_w

    def get_strategy_fitness(self) -> Dict[str, float]:
        """Get per-strategy fitness scores."""
        strategy_outcomes: Dict[str, List[float]] = {}
        for o in self.outcomes:
            if o.strategy_used:
                if o.strategy_used not in strategy_outcomes:
                    strategy_outcomes[o.strategy_used] = []
                strategy_outcomes[o.strategy_used].append(o.rating)

        return {
            name: sum(ratings) / len(ratings)
            for name, ratings in strategy_outcomes.items()
            if ratings
        }

    def get_agent_fitness(self) -> Dict[str, float]:
        """Get per-agent fitness scores."""
        agent_outcomes: Dict[str, List[float]] = {}
        for o in self.outcomes:
            if o.agent_used:
                if o.agent_used not in agent_outcomes:
                    agent_outcomes[o.agent_used] = []
                agent_outcomes[o.agent_used].append(o.rating)

        return {
            name: sum(ratings) / len(ratings)
            for name, ratings in agent_outcomes.items()
            if ratings
        }

    def get_completion_rate(self, window: int = 50) -> float:
        """Fraction of recent outcomes where the task was completed."""
        if not self.outcomes:
            return 0.0
        recent = self.outcomes[-window:]
        return sum(1 for o in recent if o.task_completed) / len(recent)

    @property
    def total_outcomes(self) -> int:
        return len(self.outcomes)
