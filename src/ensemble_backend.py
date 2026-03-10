#!/usr/bin/env python3
"""
Multi-model ensemble — route LLM calls by depth, query type, and budget.

Routes different recursion depths to different models:
- Shallow exploration: fast/cheap model
- Deep analysis: powerful model
- Verification: specialized verifier model

This integrates cleanly with PhiEnhancedRLM — just pass EnsembleBackend
as the base_llm_callable parameter.
"""

import logging
from typing import Dict, Callable, Any


logger = logging.getLogger(__name__)


class EnsembleBackend:
    """
    Routes LLM calls to different backends based on:
    1. Recursion depth (shallow -> fast model, deep -> powerful model)
    2. Query type (factual -> fast, analytical -> powerful)
    3. Budget remaining (low budget -> fast model)
    """

    def __init__(self, backends: Dict[str, Callable],
                 depth_threshold: int = 2,
                 budget_threshold: int = 500):
        """
        Args:
            backends: Dict mapping role names to LLM callables.
                      Expected keys: "fast", "powerful", "verifier".
                      At minimum, must include "fast" and "powerful".
                      If only one backend is provided, it's used for all roles.
            depth_threshold: Depths <= this use "fast", deeper use "powerful"
            budget_threshold: Budget below this uses "fast" model
        """
        if len(backends) == 1:
            only = list(backends.values())[0]
            backends = {"fast": only, "powerful": only, "verifier": only}

        self.backends = backends
        self.depth_threshold = depth_threshold
        self.budget_threshold = budget_threshold
        self.routing_stats: Dict[str, int] = {name: 0 for name in backends}
        self._current_depth = 0

    def set_depth(self, depth: int):
        """Set current recursion depth for routing decisions."""
        self._current_depth = depth

    def __call__(self, prompt: str, max_tokens: int = 2048, **kwargs) -> str:
        """Route the call to the appropriate backend."""
        backend_name = self._route(prompt, max_tokens)
        backend = self.backends.get(backend_name, self.backends.get("fast"))

        self.routing_stats[backend_name] = self.routing_stats.get(backend_name, 0) + 1

        logger.debug(f"Routing to {backend_name} (depth={self._current_depth}, "
                     f"budget={max_tokens})")

        return backend(prompt, max_tokens=max_tokens)

    def _route(self, prompt: str, max_tokens: int) -> str:
        """Determine which backend to use."""
        # Low budget -> always fast
        if max_tokens < self.budget_threshold:
            return "fast"

        # Verification prompts -> verifier if available
        prompt_lower = prompt.lower()
        if "verifier" in self.backends:
            verify_keywords = ["verify", "check", "validate", "contradiction",
                              "error", "correct"]
            if any(kw in prompt_lower for kw in verify_keywords):
                return "verifier"

        # Depth-based routing
        if self._current_depth <= self.depth_threshold:
            return "fast"
        else:
            return "powerful"

    def get_routing_stats(self) -> Dict[str, Any]:
        """Return routing statistics."""
        total = sum(self.routing_stats.values())
        return {
            "total_calls": total,
            "per_backend": dict(self.routing_stats),
            "percentages": {
                name: count / total * 100 if total else 0
                for name, count in self.routing_stats.items()
            },
        }
