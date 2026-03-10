#!/usr/bin/env python3
"""
φ-Sparse Reasoning — prune reasoning paths using golden ratio.

The golden ratio provides a natural pruning threshold: keep the top
φ⁻¹ ≈ 61.8% of branches. This is not arbitrary — it's the proportion
that maximizes information retention per unit of computational cost,
following the same principle as phyllotaxis in nature.

This module handles:
1. Branch selection: keep top φ^(-1) fraction of subquestions
2. Relevance scoring: φ-weighted scoring for subquestions
3. Diversity-aware pruning: don't just keep the top-scored, keep diverse ones
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from .phi_separation_novel_mathematics import PHI, PHI_INV


@dataclass
class ScoredBranch:
    """A subquestion with its relevance score."""
    question: str
    score: float
    diversity_bonus: float = 0.0
    combined_score: float = 0.0


class PhiSparseReasoner:
    """
    Prune the recursion tree using φ-ratio branch selection.

    Keeps only the top φ⁻¹ ≈ 61.8% of branches by default.
    The pruning ratio is adaptive (via evolution engine).
    """

    def __init__(self, pruning_ratio: float = PHI_INV,
                 diversity_weight: float = 0.3):
        """
        Args:
            pruning_ratio: Fraction of branches to keep (default φ⁻¹)
            diversity_weight: Weight for diversity bonus (0 = pure relevance)
        """
        self.pruning_ratio = pruning_ratio
        self.diversity_weight = diversity_weight

    def select_branches(self, subquestions: List[str],
                         scores: List[float]) -> List[str]:
        """
        Keep only the top φ⁻¹ fraction of branches.

        Args:
            subquestions: List of candidate subquestions
            scores: Relevance/quality scores for each subquestion

        Returns:
            Pruned list of subquestions
        """
        if len(subquestions) <= 1:
            return subquestions

        n_keep = max(1, int(len(subquestions) * self.pruning_ratio))
        ranked = sorted(zip(scores, subquestions), reverse=True)
        return [q for _, q in ranked[:n_keep]]

    def select_branches_diverse(self, subquestions: List[str],
                                 scores: List[float],
                                 embeddings: Optional[np.ndarray] = None) -> List[str]:
        """
        Diversity-aware branch selection.

        Combines relevance scores with diversity bonus to avoid
        selecting redundant subquestions.
        """
        if len(subquestions) <= 1:
            return subquestions
        if embeddings is None:
            return self.select_branches(subquestions, scores)

        n_keep = max(1, int(len(subquestions) * self.pruning_ratio))

        # Build scored branches
        branches = []
        for i, (q, s) in enumerate(zip(subquestions, scores)):
            branches.append(ScoredBranch(question=q, score=s))

        # Greedy diverse selection
        selected = []
        remaining = list(range(len(branches)))

        for _ in range(n_keep):
            if not remaining:
                break

            best_idx = None
            best_combined = -float('inf')

            for idx in remaining:
                # Diversity bonus: minimum distance to already selected
                if selected and embeddings is not None:
                    emb = embeddings[idx]
                    min_dist = min(
                        1.0 - np.dot(emb, embeddings[s]) / (
                            np.linalg.norm(emb) * np.linalg.norm(embeddings[s]) + 1e-10
                        )
                        for s in selected
                    )
                    diversity_bonus = min_dist
                else:
                    diversity_bonus = 1.0

                branches[idx].diversity_bonus = diversity_bonus
                combined = (
                    (1 - self.diversity_weight) * branches[idx].score
                    + self.diversity_weight * diversity_bonus
                )
                branches[idx].combined_score = combined

                if combined > best_combined:
                    best_combined = combined
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return [branches[i].question for i in selected]

    def score_subquestions(self, parent_query: str,
                           subquestions: List[str],
                           parent_answer: str = "") -> List[float]:
        """
        Score subquestions based on relevance to parent query.

        Uses simple heuristics (word overlap, question diversity).
        In production, this would use embeddings.
        """
        parent_words = set(parent_query.lower().split())
        answer_words = set(parent_answer.lower().split()) if parent_answer else set()

        scores = []
        for sq in subquestions:
            sq_words = set(sq.lower().split())

            # Relevance: overlap with parent query
            if parent_words:
                relevance = len(sq_words & parent_words) / len(parent_words)
            else:
                relevance = 0.5

            # Novelty: words not in parent answer (new information potential)
            if answer_words:
                novelty = len(sq_words - answer_words) / max(len(sq_words), 1)
            else:
                novelty = 0.5

            # Question quality: longer questions tend to be more specific
            length_score = min(len(sq.split()) / 15.0, 1.0)

            # φ-weighted combination
            score = (
                PHI * relevance +
                PHI_INV * novelty +
                PHI_INV ** 2 * length_score
            ) / (PHI + PHI_INV + PHI_INV ** 2)

            scores.append(float(score))

        return scores

    def adaptive_prune(self, subquestions: List[str],
                        scores: List[float],
                        confidence: float) -> List[str]:
        """
        Adaptive pruning based on current confidence.

        High confidence → prune more aggressively (we're converging)
        Low confidence → keep more branches (need more exploration)
        """
        if confidence > 0.8:
            # High confidence — aggressive pruning
            ratio = self.pruning_ratio * PHI_INV  # ~38.2%
        elif confidence < 0.4:
            # Low confidence — keep more
            ratio = min(1.0, self.pruning_ratio * PHI)  # ~100%
        else:
            ratio = self.pruning_ratio

        n_keep = max(1, int(len(subquestions) * ratio))
        ranked = sorted(zip(scores, subquestions), reverse=True)
        return [q for _, q in ranked[:n_keep]]
