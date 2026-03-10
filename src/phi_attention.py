#!/usr/bin/env python3
"""
φ-Geometric Attention Modifier

Injects golden ratio mathematics directly into the reasoning process,
not just the orchestration. This is what differentiates φ-RLM from all
other recursive models — the reasoning itself is φ-structured.

Key mechanisms:
1. φ-Structured prompt injection: scaffolds reasoning using E8 Casimir dimensions
2. Golden ratio confidence weighting: primary=φ, secondary=φ^-1, tertiary=φ^-2
3. Torsion-corrected minority view assessment
4. φ-Chain-of-thought: decomposition follows golden ratio priority levels
"""

from typing import List, Optional
from dataclasses import dataclass

from .phi_separation_novel_mathematics import (
    PHI, PHI_INV, EPSILON, CASIMIR_DEGREES
)


# E8 Casimir dimension labels for structured reasoning
CASIMIR_LENSES = {
    2: "binary (true/false, consistent/inconsistent)",
    8: "octuple (the 8 key dimensions of analysis)",
    12: "dodecahedral (12-faceted comprehensive review)",
    14: "14-dimensional cross-validation",
    18: "18-point deep analytical framework",
    20: "20-perspective synthesis",
    24: "24-fold crystallographic analysis",
    30: "30-dimensional complete E8 Coxeter analysis",
}


@dataclass
class PhiReasoningConfig:
    """Configuration for φ-structured reasoning injection."""
    inject_casimir_lens: bool = True
    inject_confidence_weighting: bool = True
    inject_torsion_correction: bool = True
    inject_chain_of_thought: bool = True
    verbosity: str = "standard"  # "minimal", "standard", "detailed"


class PhiAttentionInjector:
    """
    Modifies prompts to induce φ-structured reasoning patterns.

    At each recursion depth, injects a different E8 Casimir perspective,
    golden ratio confidence weighting, and torsion-corrected assessment.
    """

    def __init__(self, config: Optional[PhiReasoningConfig] = None):
        self.config = config or PhiReasoningConfig()

    def inject_phi_structure(self, prompt: str, depth: int,
                              budget: int = 0) -> str:
        """
        Add φ-geometric reasoning scaffolding to prompt.

        Args:
            prompt: The original prompt
            depth: Current recursion depth (0-7 maps to Casimir degrees)
            budget: Remaining token budget

        Returns:
            Modified prompt with φ-structure injection
        """
        casimir_idx = min(depth, len(CASIMIR_DEGREES) - 1)
        casimir_degree = int(CASIMIR_DEGREES[casimir_idx])
        casimir_lens = CASIMIR_LENSES.get(casimir_degree, f"{casimir_degree}-dimensional")

        sections = []

        if self.config.inject_casimir_lens:
            sections.append(self._casimir_lens_section(depth, casimir_degree, casimir_lens))

        if self.config.inject_confidence_weighting:
            sections.append(self._confidence_weighting_section())

        if self.config.inject_torsion_correction:
            sections.append(self._torsion_correction_section())

        injection = "\n".join(sections)

        if self.config.verbosity == "minimal":
            injection = self._minimize(injection)

        return f"{injection}\n\n{prompt}"

    def _casimir_lens_section(self, depth: int, degree: int, lens: str) -> str:
        return f"""[φ-Reasoning Protocol — Depth {depth}, Casimir Lens {degree}]

Structure your analysis using {lens} evaluation dimensions.
Weight your confidence using golden ratio scaling:
  primary evidence = φ ≈ 1.618 weight
  secondary evidence = φ⁻¹ ≈ 0.618 weight
  tertiary evidence = φ⁻² ≈ 0.382 weight"""

    def _confidence_weighting_section(self) -> str:
        return f"""
For each claim, assess:
1. Support strength (scale 0 to φ ≈ {PHI:.3f})
2. Contradiction density (target: below ε = {EPSILON:.4f} ≈ 28/248)
3. Information novelty (new info / total info, target: above {PHI_INV:.3f})"""

    def _torsion_correction_section(self) -> str:
        return f"""
Apply torsion correction: final_confidence = base + {EPSILON:.4f} × minority_view_weight
This ensures minority perspectives are not lost in aggregation."""

    def _minimize(self, text: str) -> str:
        """Reduce injection to essential instructions."""
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        return "\n".join(lines[:5])

    def phi_chain_of_thought(self, query: str, depth: int = 0) -> str:
        """
        Generate φ-structured chain-of-thought decomposition.

        Decomposes reasoning into golden-ratio-weighted priority levels:
          Level 1 (φ weight): Core logical chain — the essential reasoning
          Level 2 (φ⁻¹ weight): Supporting evidence — corroboration
          Level 3 (φ⁻² weight): Edge cases and counterexamples
        """
        return f"""[φ-Chain-of-Thought — Depth {depth}]

Analyze the following query at three priority levels:

LEVEL 1 (weight: {PHI:.3f} — PRIMARY):
Core logical chain. What is the essential reasoning path?
Focus on the most direct, well-supported answer.

LEVEL 2 (weight: {PHI_INV:.3f} — SUPPORTING):
Supporting evidence. What corroborates the primary analysis?
Identify confirming data, analogies, or precedents.

LEVEL 3 (weight: {PHI_INV**2:.3f} — ADVERSARIAL):
Edge cases and counterexamples. What could invalidate the primary analysis?
Consider boundary conditions, exceptions, and alternative interpretations.

Query: {query}

Synthesize all three levels, weighting by the φ-ratios above.
Your final confidence should reflect the balance between support and contradiction."""

    def build_phi_prompt(self, query: str, context_text: str,
                          depth: int, budget: int,
                          use_chain_of_thought: bool = False) -> str:
        """
        Build a complete φ-enhanced prompt.

        Combines φ-structure injection with the original query format.
        This is the drop-in replacement for the prompt builder in recursive_solve.
        """
        if use_chain_of_thought:
            reasoning_scaffold = self.phi_chain_of_thought(query, depth)
        else:
            reasoning_scaffold = ""

        base_prompt = f"""Query: {query}

Context:
{context_text}

Recursion depth: {depth}
Remaining budget: {budget} tokens

Respond in JSON format:
{{"answer": "your answer", "confidence": 0.0-1.0, "subquestions": ["...", "..."]}}
"""
        # Inject φ-structure before the prompt
        enhanced = self.inject_phi_structure(base_prompt, depth, budget)

        if reasoning_scaffold:
            enhanced = f"{reasoning_scaffold}\n\n{enhanced}"

        return enhanced


class PhiConfidenceScaler:
    """
    Scale confidence values using golden ratio geometry.

    Maps raw confidence through φ-based transformations to create
    more informative confidence distributions.
    """

    @staticmethod
    def phi_scale(confidence: float, depth: int) -> float:
        """
        Apply φ-geometric confidence scaling.

        Deeper answers get slight confidence penalty (they're less certain),
        but the penalty follows golden ratio decay.
        """
        depth_penalty = PHI_INV ** (depth + 1)  # 0.618, 0.382, 0.236, ...
        return confidence * (1.0 - (1.0 - depth_penalty) * 0.1)

    @staticmethod
    def aggregate_phi_weighted(confidences: List[float],
                                weights: Optional[List[float]] = None) -> float:
        """
        Aggregate confidences using φ-weighted averaging.

        If no weights provided, uses φ^(-i) for rank i.
        """
        if not confidences:
            return 0.0

        if weights is None:
            weights = [PHI_INV ** i for i in range(len(confidences))]

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(c * w for c, w in zip(confidences, weights))
        return weighted_sum / total_weight

    @staticmethod
    def torsion_correct(base_confidence: float,
                         minority_weight: float) -> float:
        """
        Apply torsion correction to confidence.

        ε = 28/248 ensures minority views are preserved in aggregation.
        """
        return base_confidence + EPSILON * minority_weight
