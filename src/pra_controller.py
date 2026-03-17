#!/usr/bin/env python3
"""
Phi-Recursive Architecture (PRA) Controller
=============================================

Implements the self-referential control layer from:
  "The Phi-Recursive Architecture: Self-Referential Dynamics
   for Adaptive Recursive Inference" (McGirl, March 2026)

Core recursion: h_{n+1} = sqrt(h_n + 1)
  - Unique positive fixed point: φ (golden ratio)
  - Induces exploit/explore partition: 1/φ vs 1 - 1/φ

Defect-sensitive controller: h_{n+1}² = h*² - (κ/3)·ρ_n
  - Bounded, well-posed (Theorem 3.1)
  - Converges under vanishing defect (Theorem 3.2)
  - Halts in finite time (Theorem 3.3)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .phi_separation_novel_mathematics import CASIMIR_DEGREES, COXETER_NUMBER, PHI

logger = logging.getLogger(__name__)

# Constants from the paper
PHI_FLOAT = float(PHI)  # 1.618033988749895
PHI_SQUARED = PHI_FLOAT ** 2  # φ² = φ + 1 ≈ 2.618


@dataclass
class PRAControlState:
    """Snapshot of PRA controller state at a recursion depth.

    Attributes:
        h: Current control variable.
        h_target: Target control value (default φ).
        defect: Current information defect ρ_n.
        exploit: Exploit fraction 1/h.
        explore: Explore fraction 1 - 1/h.
        depth: Current recursion depth.
        halted: Whether the controller has signaled halt.
        halt_reason: Why the controller halted (if halted).
    """
    h: float
    h_target: float
    defect: float
    exploit: float
    explore: float
    depth: int
    halted: bool = False
    halt_reason: str = ""


class PRAController:
    """Phi-Recursive Architecture controller.

    Implements Sections 2-5 of the PRA paper:
      - Core recursion h_{n+1} = sqrt(h_n + 1)  [Theorem 2.2]
      - Budget fractions E = 1/h, X = 1 - 1/h  [Corollary 2.4]
      - Defect-sensitive update h² = h*² - (κ/3)·ρ  [Theorem 3.1]
      - Halting: |h - h*| < ε_h AND ρ < ε_ρ  [Theorem 3.3]
      - Equilibrium-centered scheduler  [Section 4.4]

    Args:
        h_0: Initial control state (any h_0 >= 0).
        h_target: Target control value (default φ).
        kappa: Coupling constant for defect sensitivity.
        epsilon_h: Halting threshold for equilibrium error |h - h*|.
        epsilon_rho: Halting threshold for residual defect ρ.
    """

    def __init__(
        self,
        h_0: float = 1.0,
        h_target: float = PHI_FLOAT,
        kappa: float = 1.0,
        epsilon_h: float = 0.01,
        epsilon_rho: float = 0.05,
    ):
        if h_0 < 0:
            raise ValueError(f"h_0 must be nonnegative, got {h_0}")
        if h_target <= 0:
            raise ValueError(f"h_target must be positive, got {h_target}")
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        if epsilon_h <= 0 or epsilon_rho <= 0:
            raise ValueError("Thresholds must be positive")

        self.h_0 = h_0
        self.h_target = h_target
        self.kappa = kappa
        self.epsilon_h = epsilon_h
        self.epsilon_rho = epsilon_rho

        # State tracking
        self._h_current = h_0
        self._depth = 0
        self._history: List[float] = [h_0]
        self._defect_history: List[float] = []

    # =========================================================================
    # Section 2: Core Recursion
    # =========================================================================

    @staticmethod
    def phi_recursion(h: float) -> float:
        """Apply the core self-referential map f(h) = sqrt(h + 1).

        Theorem 2.2: Converges globally to φ from any h >= 0.
        Lemma 2.1: f([0, ∞)) ⊆ [1, ∞).
        """
        return math.sqrt(h + 1.0)

    @staticmethod
    def get_budget_fractions(h: float) -> Tuple[float, float]:
        """Compute exploit/explore fractions from control state.

        Corollary 2.4: E(h) → 1/φ, X(h) → 1 - 1/φ as h → φ.
        Proposition 2.5: 0 < E ≤ 1, 0 ≤ X < 1, E + X = 1 for h ≥ 1.

        Returns:
            (exploit, explore) where exploit = 1/h, explore = 1 - 1/h.
        """
        if h <= 0:
            # Edge case: h=0 shouldn't occur after one step, but handle gracefully
            return (1.0, 0.0)
        exploit = 1.0 / h
        explore = 1.0 - exploit
        return (exploit, explore)

    # =========================================================================
    # Section 3: Defect-Sensitive Controller
    # =========================================================================

    def defect_update(self, rho: float) -> float:
        """Compute defect-sensitive control update.

        h_{n+1} = sqrt(h*² - (κ/3)·ρ)

        Theorem 3.1: Well-defined and bounded in [0, h*] under Assumption A.

        Args:
            rho: Nonnegative information defect.

        Returns:
            Updated control value h_{n+1}.
        """
        # Clamp defect to admissible range (Assumption A)
        rho = self.clamp_defect(rho)
        val = self.h_target ** 2 - (self.kappa / 3.0) * rho
        # Numerical safety: val should be >= 0 by admissibility, but clamp
        return math.sqrt(max(0.0, val))

    def clamp_defect(self, rho: float) -> float:
        """Clamp defect to admissible range [0, 3·h*²/κ].

        Assumption A ensures the square root in defect_update remains real.
        """
        max_rho = self.max_admissible_defect()
        return max(0.0, min(rho, max_rho))

    def max_admissible_defect(self) -> float:
        """Maximum admissible defect: 3·h*²/κ."""
        return 3.0 * self.h_target ** 2 / self.kappa

    def check_admissibility(self, rho: float) -> bool:
        """Check if defect is in admissible range (Assumption A).

        0 ≤ ρ ≤ 3·h*²/κ
        """
        return 0.0 <= rho <= self.max_admissible_defect()

    # =========================================================================
    # Section 3.5: Halting Criterion
    # =========================================================================

    def should_halt(self, h: float, rho: float) -> bool:
        """Check PRA halting criterion (Theorem 3.3).

        Halt when: |h - h*| < ε_h AND ρ < ε_ρ.
        """
        equilibrium_error = abs(h - self.h_target)
        return equilibrium_error < self.epsilon_h and rho < self.epsilon_rho

    # =========================================================================
    # Section 4.2: Context-Geometry Defect
    # =========================================================================

    @staticmethod
    def compute_context_defect(
        logdet_selected: float,
        logdet_full: float,
        regularization: float = 1e-6,
    ) -> float:
        """Compute information defect from context geometry.

        ρ = 1 - log det(M_selected + λI) / log det(M_full + λI)

        Section 4.2: Measures the fraction of context information volume
        NOT captured by the selected subset.

        Args:
            logdet_selected: Log-determinant of selected context Gram matrix.
            logdet_full: Log-determinant of full context Gram matrix.
            regularization: λ for numerical stability (added to logdet values).

        Returns:
            Defect in [0, 1], where 0 = full information, 1 = no information.
        """
        # Handle edge cases
        if logdet_full <= 0 and logdet_selected <= 0:
            return 0.0

        # Stabilize with regularization offset
        ld_sel = logdet_selected + regularization
        ld_full = logdet_full + regularization

        if ld_full <= 0:
            # Full context has near-zero volume (degenerate)
            return 0.0

        ratio = ld_sel / ld_full
        # Clamp to [0, 1] — ratio > 1 can happen with regularization
        defect = 1.0 - max(0.0, min(1.0, ratio))
        return defect

    # =========================================================================
    # Section 4.4: Equilibrium-Centered Scheduler
    # =========================================================================

    def equilibrium_schedule(
        self,
        base_budget: int,
        depth_weight: float,
        alpha: float = 0.618,
    ) -> int:
        """Compute PRA-modulated budget for a recursion depth.

        B_d = B_tot · w̃_d · η(h_d)

        where η(h) = α·(1/h) + (1-α)·(1 - 1/h)

        Section 4.4: The control state modulates how much budget is
        allocated to exploit (refinement) vs explore (diversification).

        Args:
            base_budget: Base budget from Casimir allocation.
            depth_weight: Normalized depth weight w̃_d.
            alpha: Exploit weighting (default 1/φ ≈ 0.618).

        Returns:
            PRA-adjusted budget (integer tokens).
        """
        h = self._h_current
        if h <= 0:
            h = 1.0  # Safety floor

        exploit, explore = self.get_budget_fractions(h)
        eta = alpha * exploit + (1.0 - alpha) * explore
        budget = base_budget * depth_weight * eta

        # Ensure at least 1 token
        return max(1, int(round(budget)))

    # =========================================================================
    # Algorithm 5.1: Full PRA Step
    # =========================================================================

    def step(self, rho: float) -> PRAControlState:
        """Execute one PRA control step (Algorithm 5.1).

        At depth 0: uses core recursion h_{n+1} = sqrt(h_n + 1).
        At depth > 0: uses defect controller h² = h*² - (κ/3)·ρ.

        Args:
            rho: Current information defect.

        Returns:
            PRAControlState with updated h, budget fractions, halt status.
        """
        self._defect_history.append(rho)

        if self._depth == 0:
            # Core recursion establishes intrinsic equilibrium
            h_new = self.phi_recursion(self._h_current)
        else:
            # Defect-sensitive controller tracks progress
            h_new = self.defect_update(rho)

        self._h_current = h_new
        self._history.append(h_new)
        self._depth += 1

        exploit, explore = self.get_budget_fractions(h_new)
        halted = self.should_halt(h_new, rho)

        halt_reason = ""
        if halted:
            halt_reason = "pra_convergence"

        state = PRAControlState(
            h=h_new,
            h_target=self.h_target,
            defect=rho,
            exploit=exploit,
            explore=explore,
            depth=self._depth,
            halted=halted,
            halt_reason=halt_reason,
        )

        logger.debug(
            f"PRA step {self._depth}: h={h_new:.6f}, "
            f"ρ={rho:.6f}, E={exploit:.4f}, X={explore:.4f}, "
            f"halt={halted}"
        )

        return state

    def reset(self, h_0: Optional[float] = None):
        """Reset controller for a new query."""
        self._h_current = h_0 if h_0 is not None else self.h_0
        self._depth = 0
        self._history = [self._h_current]
        self._defect_history = []

    # =========================================================================
    # Section 4.3: Depth-Allocation Prior
    # =========================================================================

    @staticmethod
    def compute_depth_weights() -> List[float]:
        """Compute normalized depth weights from E8 Casimir degrees.

        w_d = φ^(-C_d/30), normalized.

        Section 4.3: Gives more budget to shallower depths (smaller Casimir).
        """
        import numpy as np

        weights = np.power(PHI_FLOAT, -CASIMIR_DEGREES / float(COXETER_NUMBER))
        normalized = weights / np.sum(weights)
        return normalized.tolist()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def h(self) -> float:
        """Current control state."""
        return self._h_current

    @property
    def depth(self) -> int:
        """Current depth counter."""
        return self._depth

    @property
    def history(self) -> List[float]:
        """History of h values."""
        return list(self._history)

    @property
    def defect_history(self) -> List[float]:
        """History of defect values."""
        return list(self._defect_history)

    @property
    def equilibrium_error(self) -> float:
        """Current |h - h*|."""
        return abs(self._h_current - self.h_target)

    def to_dict(self) -> dict:
        """Serialize controller state for tracing."""
        return {
            "h": self._h_current,
            "h_target": self.h_target,
            "kappa": self.kappa,
            "epsilon_h": self.epsilon_h,
            "epsilon_rho": self.epsilon_rho,
            "depth": self._depth,
            "equilibrium_error": self.equilibrium_error,
            "history": self._history[-5:],  # Last 5 values
            "defect_history": self._defect_history[-5:],
        }
