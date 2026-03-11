#!/usr/bin/env python3
"""
phi-Critic -- adversarial self-evaluation via recursive critique.

A second phi-RLM instance that receives the first instance's answer
and tries to break it. Its confidence in finding flaws becomes a
negative fitness signal for evolution.

This is the 4th QEC verifier, structurally different from the other 3
because it uses full recursive_solve (not a single LLM call) to
generate its critique.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class CritiqueResult:
    """Result of an adversarial critique."""
    quality_score: float      # 0.0 = terrible, 1.0 = no flaws found
    critique_text: str        # The critique itself
    critique_confidence: float  # How confident the critic is in its critique
    flaws_found: int          # Number of distinct flaws identified
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PhiCritic:
    """Adversarial evaluator -- tries to find flaws in answers.

    Uses a separate recursive_solve call with a critique-focused prompt.
    The critique depth is limited (max_depth=2) to keep cost reasonable
    while still leveraging recursive decomposition of the critique task.

    Wire into QEC as a 4th verifier that's structurally different from
    the other 3 (which use single LLM calls).
    """

    def __init__(self, rlm, critique_depth: int = 2):
        """
        Args:
            rlm: A PhiEnhancedRLM instance (can be the same or different).
            critique_depth: Max recursion depth for the critique.
        """
        self.rlm = rlm
        self.critique_depth = critique_depth

    def critique(self, query: str, answer: str,
                 context: str = "") -> CritiqueResult:
        """Generate an adversarial critique of an answer.

        Args:
            query: The original question.
            answer: The answer to critique.
            context: Optional context that was used.

        Returns:
            CritiqueResult with quality score and critique text.
        """
        critique_prompt = (
            f"You are an adversarial critic. Your job is to find errors, "
            f"logical flaws, unsupported claims, or missing steps in this "
            f"answer. Be thorough and skeptical.\n\n"
            f"Original question: {query[:300]}\n\n"
            f"Answer to critique: {answer[:500]}\n\n"
            f"List every flaw you find. If the answer is actually correct "
            f"and complete, say so."
        )

        try:
            result = self.rlm.recursive_solve(
                critique_prompt,
                max_depth=self.critique_depth,
            )

            critique_text = str(result.value)
            critique_confidence = result.confidence

            # Estimate number of flaws from the critique
            flaw_indicators = [
                "error", "flaw", "incorrect", "wrong", "missing",
                "unsupported", "inconsistent", "contradiction",
                "incomplete", "fails to", "does not",
            ]
            flaws_found = sum(
                1 for indicator in flaw_indicators
                if indicator in critique_text.lower()
            )

            # Quality score: high confidence in critique = low quality answer
            # No flaws found = high quality
            if flaws_found == 0:
                quality_score = 0.9 + 0.1 * (1 - critique_confidence)
            else:
                quality_score = max(0.1, 1.0 - (
                    critique_confidence * min(flaws_found, 5) / 5
                ))

            return CritiqueResult(
                quality_score=quality_score,
                critique_text=critique_text,
                critique_confidence=critique_confidence,
                flaws_found=flaws_found,
                metadata={
                    "critique_depth": self.critique_depth,
                    "stop_reason": result.metadata.get("stop_reason", ""),
                },
            )

        except Exception as e:
            logger.error(f"Critique failed: {e}")
            return CritiqueResult(
                quality_score=0.5,
                critique_text=f"Critique failed: {e}",
                critique_confidence=0.0,
                flaws_found=0,
                metadata={"error": str(e)},
            )

    def critique_as_qec_score(self, query: str, answer: str,
                               context: str = "") -> float:
        """Run critique and return a QEC-compatible confidence score.

        Returns:
            Float in [0, 1] suitable for averaging with other QEC scores.
        """
        result = self.critique(query, answer, context)
        return result.quality_score
