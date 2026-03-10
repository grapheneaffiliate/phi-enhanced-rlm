#!/usr/bin/env python3
"""
φ-Governed Self-Evolution Engine

Manages self-modification of RLM parameters based on performance feedback.
Uses golden ratio scaling for update magnitudes — creating diminishing-but-never-zero
adaptation that converges to optimal configurations.

The evolution loop:
1. Run RLM with current EvolutionState parameters
2. Evaluate performance (accuracy, token efficiency, depth efficiency)
3. Apply φ-scaled mutations to budget weights, stopping criteria, chunk strategies
4. Persist evolved state for next generation
"""

import json
import logging
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from phi_separation_novel_mathematics import (
    PHI, PHI_INV, LOG_PHI, EPSILON, CASIMIR_DEGREES, E8_DIM
)

logger = logging.getLogger(__name__)


@dataclass
class EvolutionState:
    """
    Persistent state that evolves across sessions.

    Each field represents a learnable parameter of the RLM system.
    The evolution engine modifies these based on performance feedback.
    """
    generation: int = 0
    # Learned Casimir budget overrides (perturbations from base E8 allocation)
    budget_weights: List[float] = field(default_factory=lambda: [1.0] * 8)
    # Which chunk selection strategies work best (strategy_name -> score)
    chunk_strategy_scores: Dict[str, float] = field(default_factory=lambda: {
        "diversity": 0.5,
        "relevance": 0.5,
        "hybrid": 0.5,
    })
    # Performance by recursion depth
    depth_performance: Dict[str, float] = field(default_factory=dict)
    # Adaptive φ-momentum stopping threshold
    phi_momentum_threshold: float = 0.93
    # Adaptive spectral saturation threshold scale
    spectral_threshold_scale: float = 1.0
    # Max subquestions per recursion level
    branch_factor: int = 3
    # φ-attention injection enabled
    phi_attention_enabled: bool = True
    # φ-sparse pruning enabled
    sparse_pruning_enabled: bool = True
    # Pruning ratio (default φ^-1 ≈ 0.618)
    pruning_ratio: float = 0.618
    # Historical accuracy by generation
    accuracy_history: List[float] = field(default_factory=list)
    # Historical token usage by generation
    token_history: List[float] = field(default_factory=list)
    # Timestamp of last evolution
    last_evolved: str = ""
    # Fitness score of current configuration
    fitness: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvolutionState':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def get_budget_weights_array(self) -> np.ndarray:
        """Return budget weights as numpy array, normalized."""
        weights = np.array(self.budget_weights[:8], dtype=float)
        weights = np.maximum(weights, 0.1)  # Floor at 0.1
        return weights / weights.sum()


@dataclass
class EvaluationResult:
    """Result of evaluating a generation's performance."""
    generation: int
    accuracy: float
    avg_tokens: float
    avg_depth: float
    avg_confidence: float
    improvement: float  # Delta from previous generation
    premature_stops: int  # Stopped too early (low confidence at stop)
    late_stops: int  # Went too deep without convergence
    depth_accuracy: Dict[str, float] = field(default_factory=dict)
    strategy_accuracy: Dict[str, float] = field(default_factory=dict)


class PhiEvolutionEngine:
    """
    Manages self-modification of RLM parameters.

    Key principle: use golden ratio for update magnitude.
    Small updates = φ^(-n) where n = generation.
    This creates diminishing-but-never-zero adaptation.
    """

    def __init__(self, state_path: str = "evolution_state.json"):
        self.state_path = Path(state_path)
        self.state = self.load_or_create()

    def load_or_create(self) -> EvolutionState:
        """Load existing state or create fresh one."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    data = json.load(f)
                logger.info(f"Loaded evolution state: generation {data.get('generation', 0)}")
                return EvolutionState.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load evolution state: {e}, creating fresh")
        return EvolutionState()

    def save(self):
        """Persist current state to disk."""
        self.state.last_evolved = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.state_path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)
        logger.info(f"Saved evolution state: generation {self.state.generation}")

    def learning_rate(self) -> float:
        """
        φ-scaled learning rate.

        Rate = φ^(-generation) — diminishes but never reaches zero.
        Generation 0: 1.0
        Generation 1: 0.618
        Generation 2: 0.382
        Generation 5: 0.090
        Generation 10: 0.008
        """
        return PHI_INV ** self.state.generation

    def evaluate_generation(self, traces: List[Dict],
                            ground_truth: Optional[List[str]] = None) -> EvaluationResult:
        """
        Score how well the current configuration performed.

        Args:
            traces: List of trace dicts from recursive_solve runs
            ground_truth: Optional expected answers for accuracy calculation

        Returns:
            EvaluationResult with comprehensive metrics
        """
        if not traces:
            return EvaluationResult(
                generation=self.state.generation,
                accuracy=0.0, avg_tokens=0.0, avg_depth=0.0,
                avg_confidence=0.0, improvement=0.0,
                premature_stops=0, late_stops=0,
            )

        confidences = [t.get("confidence", 0.5) for t in traces]
        depths = [t.get("depth", 0) for t in traces]
        stop_reasons = [t.get("stop_reason", "none") for t in traces]

        # Count premature and late stops
        premature_stops = sum(
            1 for t in traces
            if t.get("stop_reason") in ("momentum", "spectral")
            and t.get("confidence", 1.0) < 0.5
        )
        late_stops = sum(
            1 for t in traces
            if t.get("stop_reason") == "depth"
            and t.get("confidence", 0.0) < 0.7
        )

        # Accuracy (if ground truth provided)
        accuracy = 0.0
        if ground_truth and len(ground_truth) == len(traces):
            correct = sum(
                1 for t, gt in zip(traces, ground_truth)
                if gt.lower() in str(t.get("answer", "")).lower()
            )
            accuracy = correct / len(traces)

        # Estimate tokens from answer lengths
        avg_tokens = np.mean([len(str(t.get("answer", "")).split()) * 1.3 for t in traces])

        # Improvement from last generation
        prev_accuracy = self.state.accuracy_history[-1] if self.state.accuracy_history else 0.0
        improvement = accuracy - prev_accuracy

        # Depth-specific accuracy
        depth_accuracy = {}
        for t, gt in zip(traces, ground_truth or [""] * len(traces)):
            d = str(t.get("depth", 0))
            if d not in depth_accuracy:
                depth_accuracy[d] = {"correct": 0, "total": 0}
            depth_accuracy[d]["total"] += 1
            if gt and gt.lower() in str(t.get("answer", "")).lower():
                depth_accuracy[d]["correct"] += 1

        depth_acc_scores = {
            d: v["correct"] / v["total"] if v["total"] > 0 else 0.0
            for d, v in depth_accuracy.items()
        }

        return EvaluationResult(
            generation=self.state.generation,
            accuracy=accuracy,
            avg_tokens=avg_tokens,
            avg_depth=np.mean(depths) if depths else 0,
            avg_confidence=np.mean(confidences) if confidences else 0,
            improvement=improvement,
            premature_stops=premature_stops,
            late_stops=late_stops,
            depth_accuracy=depth_acc_scores,
        )

    def evolve(self, evaluation: EvaluationResult) -> EvolutionState:
        """
        Apply φ-scaled parameter updates based on evaluation.

        Key principle: update magnitude = learning_rate × direction
        Learning rate = φ^(-generation), creating convergent adaptation.
        """
        lr = self.learning_rate()
        logger.info(f"Evolving generation {self.state.generation} → {self.state.generation + 1} "
                     f"(lr={lr:.4f}, accuracy={evaluation.accuracy:.3f})")

        # 1. Evolve budget allocation
        self._mutate_budget_allocation(evaluation.depth_accuracy, lr)

        # 2. Evolve stopping criteria
        self._mutate_stopping_criteria(evaluation.premature_stops,
                                        evaluation.late_stops, lr)

        # 3. Evolve chunk strategy scores
        self._mutate_chunk_strategies(evaluation.strategy_accuracy, lr)

        # 4. Evolve branch factor
        self._mutate_branch_factor(evaluation, lr)

        # 5. Evolve pruning ratio
        self._mutate_pruning_ratio(evaluation, lr)

        # Update history and generation
        self.state.accuracy_history.append(evaluation.accuracy)
        self.state.token_history.append(evaluation.avg_tokens)
        self.state.fitness = evaluation.accuracy
        self.state.generation += 1

        # Save evolved state
        self.save()

        return self.state

    def _mutate_budget_allocation(self, depth_accuracy: Dict[str, float],
                                   lr: float):
        """
        Shift budget toward depths that produced best results.

        Re-weight Casimir degrees based on empirical performance
        while maintaining E8 structure (perturbations, not replacement).
        """
        if not depth_accuracy:
            return

        weights = np.array(self.state.budget_weights[:8], dtype=float)

        for depth_str, acc in depth_accuracy.items():
            try:
                idx = int(depth_str)
                if 0 <= idx < 8:
                    # Increase weight for high-accuracy depths, decrease for low
                    delta = (acc - 0.5) * lr  # Centered at 0.5
                    weights[idx] += delta
            except (ValueError, IndexError):
                pass

        # Ensure weights stay positive and normalized
        weights = np.maximum(weights, 0.1)
        weights = weights / weights.sum() * 8  # Maintain total = 8

        self.state.budget_weights = weights.tolist()

    def _mutate_stopping_criteria(self, premature_stops: int,
                                   late_stops: int, lr: float):
        """
        Adapt momentum and spectral thresholds.

        If too many premature stops: relax threshold by φ^(-1) × lr
        If too many late stops: tighten by φ^(-1) × lr
        """
        if premature_stops > late_stops:
            # Too aggressive stopping — relax
            adjustment = PHI_INV * lr * 0.1
            self.state.phi_momentum_threshold = min(
                0.99, self.state.phi_momentum_threshold + adjustment
            )
            self.state.spectral_threshold_scale *= (1 + PHI_INV * lr * 0.05)
        elif late_stops > premature_stops:
            # Not stopping soon enough — tighten
            adjustment = PHI_INV * lr * 0.1
            self.state.phi_momentum_threshold = max(
                0.80, self.state.phi_momentum_threshold - adjustment
            )
            self.state.spectral_threshold_scale *= (1 - PHI_INV * lr * 0.05)

        # Clamp spectral scale
        self.state.spectral_threshold_scale = np.clip(
            self.state.spectral_threshold_scale, 0.5, 2.0
        )

    def _mutate_chunk_strategies(self, strategy_accuracy: Dict[str, float],
                                  lr: float):
        """Update chunk strategy scores based on observed accuracy."""
        for strategy, acc in strategy_accuracy.items():
            if strategy in self.state.chunk_strategy_scores:
                current = self.state.chunk_strategy_scores[strategy]
                # Exponential moving average with φ-scaled learning rate
                self.state.chunk_strategy_scores[strategy] = (
                    (1 - lr) * current + lr * acc
                )

    def _mutate_branch_factor(self, evaluation: EvaluationResult, lr: float):
        """Adapt branch factor based on depth efficiency."""
        if evaluation.avg_depth > 4 and evaluation.accuracy < 0.5:
            # Going deep but not accurate — try wider
            if lr > 0.1:  # Only if still learning significantly
                self.state.branch_factor = min(5, self.state.branch_factor + 1)
        elif evaluation.avg_depth < 2 and evaluation.accuracy > 0.7:
            # Shallow and accurate — can narrow
            self.state.branch_factor = max(1, self.state.branch_factor - 1)

    def _mutate_pruning_ratio(self, evaluation: EvaluationResult, lr: float):
        """Adapt φ-sparse pruning ratio."""
        if evaluation.avg_tokens > 1000 and evaluation.accuracy > 0.5:
            # Using many tokens — prune more aggressively
            self.state.pruning_ratio = max(0.3, self.state.pruning_ratio - lr * 0.05)
        elif evaluation.accuracy < 0.4:
            # Not accurate enough — prune less
            self.state.pruning_ratio = min(0.9, self.state.pruning_ratio + lr * 0.05)

    def get_best_chunk_strategy(self) -> str:
        """Return the highest-scoring chunk selection strategy."""
        scores = self.state.chunk_strategy_scores
        if not scores:
            return "diversity"
        return max(scores, key=scores.get)

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Return a summary of the evolution history."""
        return {
            "generation": self.state.generation,
            "fitness": self.state.fitness,
            "learning_rate": self.learning_rate(),
            "budget_weights": self.state.budget_weights,
            "momentum_threshold": self.state.phi_momentum_threshold,
            "spectral_scale": self.state.spectral_threshold_scale,
            "best_strategy": self.get_best_chunk_strategy(),
            "accuracy_trend": self.state.accuracy_history[-10:],
            "token_trend": self.state.token_history[-10:],
        }
