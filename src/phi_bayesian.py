#!/usr/bin/env python3
"""
φ-Bayesian Optimization — golden ratio acquisition function.

Gradient-free hyperparameter optimization using φ-geometry.
Instead of a Gaussian process surrogate, uses golden ratio spacing
for efficient exploration and φ-weighted exploitation of the best
observed configurations.

Key insight: Golden ratio spacing minimizes aliasing in parameter space,
providing better coverage than uniform or random sampling with the
same number of evaluations.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple

from .phi_separation_novel_mathematics import PHI, PHI_INV

logger = logging.getLogger(__name__)


class PhiBayesianOptimizer:
    """
    Gradient-free hyperparameter optimization using φ-geometry.

    Acquisition function balances exploitation (near best observed)
    and exploration (far from sampled) using golden ratio weights:
    exploit_weight = φ^(-1), explore_weight = φ^(-2).
    """

    def __init__(self, param_bounds: Dict[str, Tuple[float, float]]):
        """
        Args:
            param_bounds: Dict of param_name -> (low, high) bounds
        """
        self.bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.observations: List[Tuple[Dict[str, float], float]] = []

    def suggest(self) -> Dict[str, float]:
        """Suggest next parameter configuration to try."""
        if len(self.observations) < 3:
            return self._phi_grid_sample()

        return self._acquisition_sample()

    def _phi_grid_sample(self) -> Dict[str, float]:
        """
        Sample parameter space using golden ratio spacing.

        Golden ratio sequence: fractional part of n * φ
        This provides quasi-random low-discrepancy coverage.
        """
        n = len(self.observations)
        params = {}
        for i, name in enumerate(self.param_names):
            lo, hi = self.bounds[name]
            # Use golden ratio sequence with dimension offset
            frac = ((n + 1) * PHI + i * PHI_INV) % 1.0
            params[name] = lo + frac * (hi - lo)
        return params

    def _acquisition_sample(self) -> Dict[str, float]:
        """
        φ-acquisition function: balance exploitation and exploration.

        Score = φ^(-1) × predicted_value + φ^(-2) × distance_from_sampled
        """
        # Find best observation
        best_params, best_score = max(self.observations, key=lambda x: x[1])

        # Generate candidates using golden ratio perturbation
        candidates = []
        for trial in range(20):
            params = {}
            for i, name in enumerate(self.param_names):
                lo, hi = self.bounds[name]
                # Perturbation from best + exploration term
                base = best_params[name]
                perturbation = (hi - lo) * 0.1 * np.sin((trial + 1) * PHI + i)
                value = np.clip(base + perturbation, lo, hi)
                params[name] = float(value)
            candidates.append(params)

        # Score candidates using acquisition function
        best_candidate = None
        best_acq_score = -float('inf')

        for candidate in candidates:
            # Exploitation: similarity to best known
            exploit = self._similarity_to_best(candidate, best_params)

            # Exploration: distance from all observations
            explore = self._min_distance_to_observed(candidate)

            acq_score = PHI_INV * exploit * best_score + PHI_INV**2 * explore
            if acq_score > best_acq_score:
                best_acq_score = acq_score
                best_candidate = candidate

        return best_candidate or self._phi_grid_sample()

    def _similarity_to_best(self, candidate: Dict[str, float],
                            best: Dict[str, float]) -> float:
        """Compute normalized similarity to best observation."""
        diffs = []
        for name in self.param_names:
            lo, hi = self.bounds[name]
            rng = hi - lo if hi > lo else 1.0
            diffs.append((candidate[name] - best[name]) / rng)
        dist = np.sqrt(sum(d**2 for d in diffs))
        return PHI ** (-dist)

    def _min_distance_to_observed(self, candidate: Dict[str, float]) -> float:
        """Compute minimum normalized distance to any observed point."""
        if not self.observations:
            return 1.0

        min_dist = float('inf')
        for obs_params, _ in self.observations:
            diffs = []
            for name in self.param_names:
                lo, hi = self.bounds[name]
                rng = hi - lo if hi > lo else 1.0
                diffs.append((candidate[name] - obs_params[name]) / rng)
            dist = np.sqrt(sum(d**2 for d in diffs))
            min_dist = min(min_dist, dist)

        return min_dist

    def observe(self, params: Dict[str, float], score: float):
        """Record an observation."""
        self.observations.append((params, score))
        logger.info(f"Observed score={score:.4f} with params={params}")

    def get_best(self) -> Tuple[Dict[str, float], float]:
        """Return best observed configuration."""
        if not self.observations:
            return {name: (lo + hi) / 2 for name, (lo, hi) in self.bounds.items()}, 0.0
        return max(self.observations, key=lambda x: x[1])

    def get_summary(self) -> Dict:
        """Return optimization summary."""
        if not self.observations:
            return {"n_observations": 0}

        scores = [s for _, s in self.observations]
        best_params, best_score = self.get_best()
        return {
            "n_observations": len(self.observations),
            "best_score": best_score,
            "best_params": best_params,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
        }
