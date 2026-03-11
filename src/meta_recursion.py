#!/usr/bin/env python3
"""
Meta-Recursion: the model reasons about its own reasoning.

This is structural self-recursion — the model at depth N has different
configuration than at depth N-1. Instead of just recursing on questions,
it recurses on its own architecture:

1. Analyze query type
2. Select recursion strategy (deep-narrow, wide-shallow, balanced, spiral)
3. Configure RLM with strategy
4. Run recursive solve
5. Meta-evaluate: did the strategy work?
6. If poor result, try alternative strategy (meta-recursion)

This closes the "structure → modified structure" loop that makes the
model truly self-evolving.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .phi_enhanced_rlm import PhiEnhancedRLM, SubCallResult
from .phi_separation_novel_mathematics import CASIMIR_DEGREES, PHI, PHI_INV

logger = logging.getLogger(__name__)


@dataclass
class RecursionStrategy:
    """Defines a recursion configuration."""
    name: str
    max_depth: int
    budget_distribution: str  # "casimir", "uniform", "front-loaded", "back-loaded"
    chunk_strategy: str  # "diversity", "relevance", "hybrid"
    branch_factor: int  # max subquestions per level
    phi_attention: bool  # inject φ-structured reasoning
    sparse_pruning: bool  # apply φ-ratio branch pruning
    description: str = ""
    min_depth: int = 0  # minimum depth before early-stop checks engage


@dataclass
class MetaEvaluation:
    """Result of meta-evaluating a strategy's performance."""
    strategy_name: str
    score: float  # Overall quality score [0, 1]
    confidence: float
    depth_reached: int
    tokens_used: int
    time_seconds: float
    stop_reason: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


# Pre-defined strategy templates
STRATEGIES: Dict[str, RecursionStrategy] = {
    "deep_analytical": RecursionStrategy(
        name="deep_analytical",
        max_depth=6,
        budget_distribution="casimir",
        chunk_strategy="diversity",
        branch_factor=2,
        phi_attention=True,
        sparse_pruning=True,
        description="Deep, narrow analysis with φ-structured reasoning. "
                     "Best for complex analytical questions requiring careful logic.",
        min_depth=2,
    ),
    "wide_exploratory": RecursionStrategy(
        name="wide_exploratory",
        max_depth=3,
        budget_distribution="uniform",
        chunk_strategy="relevance",
        branch_factor=4,
        phi_attention=False,
        sparse_pruning=False,
        description="Wide, shallow exploration covering many angles. "
                     "Best for open-ended questions needing broad perspective.",
        min_depth=1,
    ),
    "spiral_convergent": RecursionStrategy(
        name="spiral_convergent",
        max_depth=5,
        budget_distribution="casimir",
        chunk_strategy="hybrid",
        branch_factor=3,
        phi_attention=True,
        sparse_pruning=True,
        description="Balanced spiral convergence using full φ-enhancement. "
                     "Best general-purpose strategy for moderate complexity.",
        min_depth=2,
    ),
    "quick_factual": RecursionStrategy(
        name="quick_factual",
        max_depth=2,
        budget_distribution="front-loaded",
        chunk_strategy="relevance",
        branch_factor=1,
        phi_attention=False,
        sparse_pruning=False,
        description="Quick, focused factual lookup with minimal recursion. "
                     "Best for straightforward factual questions.",
        min_depth=0,
    ),
    "deep_research": RecursionStrategy(
        name="deep_research",
        max_depth=7,
        budget_distribution="casimir",
        chunk_strategy="diversity",
        branch_factor=3,
        phi_attention=True,
        sparse_pruning=True,
        description="Deep research pipeline using the full E8 Casimir hierarchy "
                     "with agent-specialized depths: clarify -> brief -> plan -> "
                     "parallel research -> synthesize -> verify -> report. "
                     "Maps to claude-code-templates deep-research-team agents.",
        min_depth=3,
    ),
    "planned": RecursionStrategy(
        name="planned",
        max_depth=4,
        budget_distribution="casimir",
        chunk_strategy="diversity",
        branch_factor=3,
        phi_attention=True,
        sparse_pruning=True,
        description="Pre-planned execution: decomposes query into a DAG of "
                     "steps before recursion. Steps execute in topological "
                     "order with independent branches parallelized.",
        min_depth=1,
    ),
    "superpowers_workflow": RecursionStrategy(
        name="superpowers_workflow",
        max_depth=7,
        budget_distribution="casimir",
        chunk_strategy="hybrid",
        branch_factor=1,
        phi_attention=True,
        sparse_pruning=False,
        description="Superpowers mandatory pipeline: brainstorm -> plan -> execute -> "
                     "spec-review -> quality-review -> verify -> debug. Serial with hard gates.",
        min_depth=3,
    ),
}


# Query type classifiers (heuristic-based)
QUERY_TYPE_KEYWORDS = {
    "analytical": ["analyze", "explain", "why", "how does", "mechanism", "cause",
                    "compare", "contrast", "evaluate", "assess", "implications"],
    "factual": ["what is", "who is", "when did", "where is", "define",
                "name", "list", "identify"],
    "mathematical": ["calculate", "compute", "solve", "equation", "formula",
                      "how many", "how much", "what is the value"],
    "creative": ["design", "create", "propose", "suggest", "imagine",
                  "what if", "brainstorm"],
    "exploratory": ["explore", "discuss", "overview", "summary",
                     "broad", "perspectives", "opinions"],
    "research": ["research", "investigate", "study", "analyze in depth",
                  "examine", "review", "survey", "assess thoroughly",
                  "evaluate comprehensively", "compare and contrast"],
    "development": ["build", "implement", "create", "develop", "code",
                     "feature", "fix", "refactor", "deploy", "test", "ship"],
}

# Strategy mapping for query types
QUERY_TYPE_STRATEGY = {
    "analytical": "deep_analytical",
    "factual": "quick_factual",
    "mathematical": "deep_analytical",
    "creative": "wide_exploratory",
    "exploratory": "wide_exploratory",
    "research": "deep_research",
    "development": "superpowers_workflow",
    "unknown": "spiral_convergent",
}


class MetaRecursiveRLM:
    """
    RLM that can modify its own recursion strategy mid-run.

    The outer meta-loop reasons about how to reason,
    selecting and adapting strategies based on query type
    and runtime performance.
    """

    def __init__(self, base_rlm: PhiEnhancedRLM,
                 strategies: Optional[Dict[str, RecursionStrategy]] = None):
        """
        Args:
            base_rlm: The base φ-RLM to configure and run
            strategies: Available strategies (uses defaults if None)
        """
        self.rlm = base_rlm
        self.strategies = strategies or STRATEGIES
        self.strategy_history: List[Dict[str, Any]] = []
        self.performance_cache: Dict[str, List[float]] = {}

    def _llm_classify_query(self, query: str) -> str:
        """Use LLM to classify query type when keyword matching is weak."""
        valid_types = list(QUERY_TYPE_STRATEGY.keys())
        type_list = ", ".join(t for t in valid_types if t != "unknown")
        classify_prompt = (
            f"Classify this query into exactly one category: {type_list}\n\n"
            f"Query: {query}\n\n"
            f"Respond with just the category name."
        )
        try:
            resp = self.rlm.llm(classify_prompt, max_tokens=20)
            # Try to parse as JSON first (mock backends return JSON)
            try:
                parsed = json.loads(resp)
                category = parsed.get("answer", "").strip().lower()
            except (json.JSONDecodeError, AttributeError):
                category = resp.strip().lower().replace('"', '').replace("'", "")
            if category in QUERY_TYPE_STRATEGY:
                return category
        except Exception:
            pass
        return "unknown"

    def analyze_query_type(self, query: str) -> str:
        """
        Classify the query type to inform strategy selection.

        Uses keyword matching as a fast heuristic, with LLM-based
        classification as fallback for ambiguous queries.
        """
        query_lower = query.lower()
        scores = {}

        for qtype, keywords in QUERY_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[qtype] = score

        if not scores or max(scores.values()) == 0:
            # Weak keyword signal -- try LLM-based classification fallback
            return self._llm_classify_query(query)

        best_type = max(scores, key=scores.get)
        # If keyword match is weak (<=1), supplement with LLM classification
        if scores[best_type] <= 1:
            llm_type = self._llm_classify_query(query)
            if llm_type != "unknown":
                return llm_type

        return best_type

    def select_strategy(self, query_type: str) -> RecursionStrategy:
        """
        Select recursion strategy based on query type.

        Uses historical performance to improve selection over time.
        """
        strategy_name = QUERY_TYPE_STRATEGY.get(query_type, "spiral_convergent")

        # Check if we have performance data suggesting a different strategy
        if self.performance_cache:
            best_name = max(
                self.performance_cache,
                key=lambda k: np.mean(self.performance_cache[k]) if self.performance_cache[k] else 0
            )
            best_score = np.mean(self.performance_cache[best_name]) if self.performance_cache[best_name] else 0
            default_score = np.mean(self.performance_cache.get(strategy_name, [0.5]))

            # Only override if the learned best is significantly better
            if best_score > default_score + 0.1:
                strategy_name = best_name

        return self.strategies.get(strategy_name, self.strategies["spiral_convergent"])

    def select_alternative_strategy(self, current: RecursionStrategy,
                                     meta_eval: MetaEvaluation) -> RecursionStrategy:
        """
        Select an alternative strategy when the current one performed poorly.

        Logic:
        - If too shallow (stopped early, low confidence): try deeper
        - If too deep (used many tokens, slow): try shallower
        - If low diversity: switch to diversity-focused strategy
        """
        if meta_eval.depth_reached <= 1 and meta_eval.confidence < 0.5:
            # Too shallow — need deeper analysis
            return self.strategies.get("deep_analytical", current)
        elif meta_eval.tokens_used > 2000 and meta_eval.time_seconds > 10:
            # Too expensive — try simpler
            return self.strategies.get("quick_factual", current)
        else:
            # General fallback — try the balanced approach
            return self.strategies.get("spiral_convergent", current)

    def configure_rlm(self, strategy: RecursionStrategy):
        """
        Configure the base RLM parameters according to the strategy.

        This is where structural self-modification happens — the model
        at recursion depth N has a different configuration than at N-1.
        """
        # Reconfigure budget allocation based on strategy
        if strategy.budget_distribution == "front-loaded":
            # More budget for early depths
            weights = [PHI ** (7 - i) for i in range(8)]
        elif strategy.budget_distribution == "back-loaded":
            # More budget for deeper depths
            weights = [PHI ** i for i in range(8)]
        elif strategy.budget_distribution == "uniform":
            weights = [1.0] * 8
        else:
            # Default: Casimir
            weights = CASIMIR_DEGREES.tolist()

        total = sum(weights)
        budget_map = {}
        for i, w in enumerate(weights):
            budget_map[i] = int(self.rlm.total_budget * w / total)
        self.rlm.budget_map = budget_map

    def evaluate_strategy(self, result: SubCallResult,
                           strategy: RecursionStrategy,
                           time_seconds: float = 0.0) -> MetaEvaluation:
        """Evaluate how well the strategy performed."""
        depth = result.metadata.get("depth", 0)
        stop_reason = result.metadata.get("stop_reason", "")
        tokens = len(str(result.value).split()) * 1.3  # Rough estimate

        # Score based on confidence, efficiency, and stop quality
        confidence_score = result.confidence

        # Penalize if stopped for wrong reason
        stop_quality = 1.0
        if stop_reason == "depth" and result.confidence < 0.5:
            stop_quality = 0.5  # Hit max depth without convergence
        elif stop_reason == "momentum" and result.confidence < 0.3:
            stop_quality = 0.3  # Premature stop

        # Efficiency: prefer less tokens for same confidence
        efficiency = result.confidence / max(tokens / 100, 1.0)

        score = (
            PHI * confidence_score +
            PHI_INV * stop_quality +
            PHI_INV ** 2 * min(efficiency, 1.0)
        ) / (PHI + PHI_INV + PHI_INV ** 2)

        strengths = []
        weaknesses = []

        if result.confidence > 0.7:
            strengths.append("High confidence result")
        if tokens < 500:
            strengths.append("Token efficient")
        if stop_reason == "momentum":
            strengths.append("Natural convergence")

        if result.confidence < 0.4:
            weaknesses.append("Low confidence")
        if stop_reason == "depth":
            weaknesses.append("Hit depth limit without convergence")
        if tokens > 2000:
            weaknesses.append("High token usage")

        return MetaEvaluation(
            strategy_name=strategy.name,
            score=score,
            confidence=result.confidence,
            depth_reached=depth,
            tokens_used=int(tokens),
            time_seconds=time_seconds,
            stop_reason=stop_reason,
            strengths=strengths,
            weaknesses=weaknesses,
        )

    def meta_solve(self, query: str, max_meta_depth: int = 3) -> SubCallResult:
        """
        Outer loop: reason about how to reason.

        1. Analyze query type
        2. Select strategy
        3. Configure RLM
        4. Run recursive solve
        5. Meta-evaluate
        6. If poor, try alternative (meta-recursion)
        """
        # Step 1: Analyze query type
        query_type = self.analyze_query_type(query)
        logger.info(f"Query type: {query_type}")

        # Step 2: Select strategy
        strategy = self.select_strategy(query_type)
        logger.info(f"Selected strategy: {strategy.name}")

        # Step 3: Configure RLM
        self.configure_rlm(strategy)

        # Step 4: Run recursive solve
        start = time.time()
        result = self.rlm.recursive_solve(
            query, max_depth=strategy.max_depth, min_depth=strategy.min_depth)
        elapsed = time.time() - start

        # Step 5: Meta-evaluate
        meta_eval = self.evaluate_strategy(result, strategy, elapsed)
        logger.info(f"Strategy {strategy.name}: score={meta_eval.score:.3f}, "
                     f"confidence={meta_eval.confidence:.3f}")

        # Record for learning
        self.strategy_history.append({
            "query_type": query_type,
            "strategy": strategy.name,
            "score": meta_eval.score,
            "confidence": meta_eval.confidence,
        })
        if strategy.name not in self.performance_cache:
            self.performance_cache[strategy.name] = []
        self.performance_cache[strategy.name].append(meta_eval.score)

        # Step 6: If poor result, try different strategy (meta-recursion)
        if meta_eval.score < PHI_INV and max_meta_depth > 0:
            alt_strategy = self.select_alternative_strategy(strategy, meta_eval)

            if alt_strategy.name != strategy.name:
                logger.info(f"Meta-recursion: trying alternative strategy {alt_strategy.name}")

                self.configure_rlm(alt_strategy)
                alt_start = time.time()
                alt_result = self.rlm.recursive_solve(
                    query, max_depth=alt_strategy.max_depth, min_depth=alt_strategy.min_depth)
                alt_elapsed = time.time() - alt_start

                alt_eval = self.evaluate_strategy(alt_result, alt_strategy, alt_elapsed)

                # Record alternative attempt
                if alt_strategy.name not in self.performance_cache:
                    self.performance_cache[alt_strategy.name] = []
                self.performance_cache[alt_strategy.name].append(alt_eval.score)

                # Pick the better result
                if alt_eval.score > meta_eval.score:
                    logger.info(f"Alternative strategy {alt_strategy.name} was better "
                                 f"({alt_eval.score:.3f} vs {meta_eval.score:.3f})")
                    result = alt_result
                    result.metadata["meta_strategy_switch"] = {
                        "from": strategy.name,
                        "to": alt_strategy.name,
                        "improvement": alt_eval.score - meta_eval.score,
                    }

        result.metadata["meta_query_type"] = query_type
        result.metadata["meta_strategy"] = strategy.name

        return result

    def get_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated performance statistics for each strategy."""
        stats = {}
        for name, scores in self.performance_cache.items():
            if scores:
                stats[name] = {
                    "mean_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "n_uses": len(scores),
                    "best_score": float(max(scores)),
                    "worst_score": float(min(scores)),
                }
        return stats
