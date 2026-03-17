#!/usr/bin/env python3
"""
Tests for the Phi-Recursive Architecture (PRA) Controller.

Verifies mathematical properties from the PRA paper (McGirl, March 2026):
  - Theorem 2.2: Global convergence to φ
  - Proposition 2.3: Local asymptotic rate 1/(2φ)
  - Corollary 2.4: Budget fraction convergence
  - Proposition 2.5: Budget validity
  - Theorem 3.1: Well-posedness and boundedness
  - Theorem 3.2: Convergence under vanishing defect
  - Theorem 3.3: Finite halting
  - Proposition 3.4: Geometric defect decay inherited
  - Corollary 3.5: Golden equilibrium budget
  - Context-geometry defect computation
  - Algorithm 5.1 integration
"""

import math

import numpy as np
import pytest

from src.pra_controller import PRAControlState, PRAController
from src.phi_separation_novel_mathematics import CASIMIR_DEGREES, PHI

PHI_F = float(PHI)  # 1.618033988749895


# =============================================================================
# Section 2: Core Recursion
# =============================================================================


class TestCoreRecursion:
    """Tests for the self-referential map f(h) = sqrt(h + 1)."""

    def test_fixed_point_is_phi(self):
        """φ satisfies φ = sqrt(φ + 1), equivalently φ² = φ + 1."""
        assert abs(PHI_F - math.sqrt(PHI_F + 1.0)) < 1e-14
        assert abs(PHI_F ** 2 - PHI_F - 1.0) < 1e-14

    @pytest.mark.parametrize("h_0", [0.0, 0.5, 1.0, PHI_F, 2.0, 5.0, 100.0])
    def test_theorem_2_2_global_convergence(self, h_0):
        """Theorem 2.2: h_n → φ from every h_0 ≥ 0."""
        h = h_0
        for _ in range(50):
            h = PRAController.phi_recursion(h)
        assert abs(h - PHI_F) < 1e-10, f"Failed to converge from h_0={h_0}: h={h}"

    def test_lemma_2_1_one_step_entrance(self):
        """Lemma 2.1: f([0, ∞)) ⊆ [1, ∞). After one step, h ≥ 1."""
        for h_0 in [0.0, 0.001, 0.5, 1.0, 10.0, 1000.0]:
            h_1 = PRAController.phi_recursion(h_0)
            assert h_1 >= 1.0, f"h_1={h_1} < 1 from h_0={h_0}"

    def test_contraction_constant(self):
        """On [1, ∞), Lipschitz constant L = 1/(2√2) < 1."""
        L = 1.0 / (2.0 * math.sqrt(2.0))
        assert L < 1.0
        assert abs(L - 0.353553) < 1e-5

    def test_theorem_2_2_contraction_bound(self):
        """After one step, |h_n - φ| ≤ L^(n-1) · |h_1 - φ|."""
        L = 1.0 / (2.0 * math.sqrt(2.0))
        h_0 = 0.0
        h = PRAController.phi_recursion(h_0)  # h_1
        h1_error = abs(h - PHI_F)

        for n in range(2, 20):
            h = PRAController.phi_recursion(h)
            actual_error = abs(h - PHI_F)
            bound = L ** (n - 1) * h1_error
            assert actual_error <= bound + 1e-15, (
                f"n={n}: error={actual_error} > bound={bound}"
            )

    def test_proposition_2_3_local_rate(self):
        """Proposition 2.3: f'(φ) = 1/(2φ) ≈ 0.309016."""
        local_rate = 1.0 / (2.0 * PHI_F)
        assert abs(local_rate - 0.309016994) < 1e-8

        # Verify numerically: near φ, ratio of successive errors → 1/(2φ)
        h = PHI_F + 0.001
        errors = []
        for _ in range(10):
            h = PRAController.phi_recursion(h)
            errors.append(abs(h - PHI_F))

        # Check convergence of error ratio to local rate
        for i in range(5, 9):
            if errors[i - 1] > 1e-15:
                ratio = errors[i] / errors[i - 1]
                assert abs(ratio - local_rate) < 0.01, (
                    f"Error ratio {ratio} not close to 1/(2φ)={local_rate}"
                )


# =============================================================================
# Section 2.5: Budget Fractions
# =============================================================================


class TestBudgetFractions:
    """Tests for exploit/explore fractions."""

    def test_corollary_2_4_convergence(self):
        """Corollary 2.4: E(h_n) → 1/φ, X(h_n) → 1 - 1/φ."""
        h = 0.5
        for _ in range(50):
            h = PRAController.phi_recursion(h)

        exploit, explore = PRAController.get_budget_fractions(h)
        assert abs(exploit - 1.0 / PHI_F) < 1e-10
        assert abs(explore - (1.0 - 1.0 / PHI_F)) < 1e-10

    def test_proposition_2_5_validity(self):
        """Proposition 2.5: 0 < E ≤ 1, 0 ≤ X < 1, E + X = 1 for h ≥ 1."""
        for h in [1.0, 1.5, PHI_F, 2.0, 5.0, 100.0]:
            exploit, explore = PRAController.get_budget_fractions(h)
            assert 0 < exploit <= 1.0, f"E={exploit} out of range for h={h}"
            assert 0 <= explore < 1.0, f"X={explore} out of range for h={h}"
            assert abs(exploit + explore - 1.0) < 1e-14, (
                f"E+X={exploit + explore} ≠ 1 for h={h}"
            )

    def test_phi_equilibrium_fractions(self):
        """At h = φ: exploit = 1/φ = φ-1 ≈ 0.618, explore ≈ 0.382."""
        exploit, explore = PRAController.get_budget_fractions(PHI_F)
        assert abs(exploit - (PHI_F - 1.0)) < 1e-14
        assert abs(explore - (2.0 - PHI_F)) < 1e-14

    def test_h_zero_edge_case(self):
        """h=0 should return (1.0, 0.0) as a safe default."""
        exploit, explore = PRAController.get_budget_fractions(0.0)
        assert exploit == 1.0
        assert explore == 0.0


# =============================================================================
# Section 3: Defect-Sensitive Controller
# =============================================================================


class TestDefectController:
    """Tests for the defect-sensitive control law."""

    def test_theorem_3_1_well_posedness(self):
        """Theorem 3.1: 0 ≤ h_{n+1} ≤ h* under admissible defect."""
        ctrl = PRAController(h_0=1.0, kappa=2.0)
        h_star = ctrl.h_target

        for rho in [0.0, 0.5, 1.0, 2.0, ctrl.max_admissible_defect()]:
            h_new = ctrl.defect_update(rho)
            assert 0.0 <= h_new <= h_star + 1e-12, (
                f"h_new={h_new} out of [0, {h_star}] for ρ={rho}"
            )

    def test_theorem_3_1_zero_defect(self):
        """With ρ=0, h_{n+1} = h* exactly."""
        ctrl = PRAController(h_0=1.0, kappa=1.0)
        h_new = ctrl.defect_update(0.0)
        assert abs(h_new - ctrl.h_target) < 1e-14

    def test_theorem_3_1_max_defect(self):
        """With ρ = 3h*²/κ (max admissible), h_{n+1} = 0."""
        ctrl = PRAController(h_0=1.0, kappa=1.0)
        max_rho = ctrl.max_admissible_defect()
        h_new = ctrl.defect_update(max_rho)
        assert abs(h_new) < 1e-14

    def test_theorem_3_2_convergence_under_vanishing_defect(self):
        """Theorem 3.2: If ρ_n → 0 then h_n → h*."""
        ctrl = PRAController(h_0=1.0, kappa=1.0)

        # Simulate geometrically decaying defect
        rho_0 = 2.0
        q = 0.5
        h_values = []
        for n in range(30):
            rho_n = rho_0 * q ** n
            h_new = ctrl.defect_update(rho_n)
            h_values.append(h_new)

        # Last h should be close to φ
        assert abs(h_values[-1] - PHI_F) < 0.01

    def test_theorem_3_3_finite_halting(self):
        """Theorem 3.3: Halting occurs in finite time under vanishing defect."""
        ctrl = PRAController(h_0=0.5, kappa=1.0, epsilon_h=0.05, epsilon_rho=0.1)

        rho_0 = 2.0
        q = 0.5
        halted = False
        for n in range(100):
            rho_n = rho_0 * q ** n
            state = ctrl.step(rho_n)
            if state.halted:
                halted = True
                break

        assert halted, "PRA did not halt with geometrically decaying defect"

    def test_proposition_3_4_geometric_decay_inherited(self):
        """Proposition 3.4: h*² - h_{n+1}² ≤ (κ/3)·ρ_0·q^n."""
        ctrl = PRAController(h_0=1.0, kappa=2.0)
        h_star = ctrl.h_target
        rho_0 = 1.0
        q = 0.7

        for n in range(20):
            rho_n = rho_0 * q ** n
            h_new = ctrl.defect_update(rho_n)
            squared_error = h_star ** 2 - h_new ** 2
            bound = (ctrl.kappa / 3.0) * rho_0 * q ** n

            assert squared_error >= -1e-14, f"Negative squared error at n={n}"
            assert squared_error <= bound + 1e-14, (
                f"n={n}: squared_error={squared_error} > bound={bound}"
            )

    def test_corollary_3_5_golden_budget(self):
        """Corollary 3.5: With h* = φ and vanishing defect, budget → 1/φ, 1-1/φ."""
        ctrl = PRAController(h_0=1.0, h_target=PHI_F, kappa=1.0)

        # Run with decaying defect
        for n in range(30):
            rho = 2.0 * 0.5 ** n
            state = ctrl.step(rho)

        exploit, explore = state.exploit, state.explore
        assert abs(exploit - 1.0 / PHI_F) < 0.01
        assert abs(explore - (1.0 - 1.0 / PHI_F)) < 0.01

    def test_admissibility_clamping(self):
        """Defect above max admissible is clamped, not rejected."""
        ctrl = PRAController(h_0=1.0, kappa=1.0)
        max_rho = ctrl.max_admissible_defect()

        # Over-limit defect should be clamped
        clamped = ctrl.clamp_defect(max_rho * 2.0)
        assert abs(clamped - max_rho) < 1e-14

        # Negative defect should be clamped to 0
        clamped = ctrl.clamp_defect(-1.0)
        assert clamped == 0.0

    def test_admissibility_check(self):
        """check_admissibility returns correct bool."""
        ctrl = PRAController(kappa=1.0)
        max_rho = ctrl.max_admissible_defect()

        assert ctrl.check_admissibility(0.0)
        assert ctrl.check_admissibility(max_rho)
        assert ctrl.check_admissibility(max_rho / 2.0)
        assert not ctrl.check_admissibility(-0.001)
        assert not ctrl.check_admissibility(max_rho + 0.001)


# =============================================================================
# Section 4.2: Context-Geometry Defect
# =============================================================================


class TestContextDefect:
    """Tests for the context-geometry defect functional."""

    def test_full_selection_zero_defect(self):
        """If selected = full, defect = 0."""
        defect = PRAController.compute_context_defect(5.0, 5.0)
        assert abs(defect) < 1e-10

    def test_empty_selection_max_defect(self):
        """If logdet_selected ≈ 0 and logdet_full >> 0, defect → 1."""
        defect = PRAController.compute_context_defect(0.0, 10.0)
        assert defect > 0.9

    def test_defect_in_range(self):
        """Defect should always be in [0, 1]."""
        for ld_sel, ld_full in [(1.0, 5.0), (3.0, 5.0), (4.9, 5.0), (0.1, 100.0)]:
            defect = PRAController.compute_context_defect(ld_sel, ld_full)
            assert 0.0 <= defect <= 1.0, (
                f"defect={defect} out of [0,1] for ld_sel={ld_sel}, ld_full={ld_full}"
            )

    def test_defect_monotonicity(self):
        """Larger selection → smaller defect (more information captured)."""
        ld_full = 10.0
        defects = []
        for ld_sel in [1.0, 3.0, 5.0, 8.0, 10.0]:
            d = PRAController.compute_context_defect(ld_sel, ld_full)
            defects.append(d)

        for i in range(len(defects) - 1):
            assert defects[i] >= defects[i + 1] - 1e-10, (
                f"Defect not monotone: {defects}"
            )

    def test_degenerate_full_context(self):
        """If logdet_full <= 0, defect = 0 (degenerate context)."""
        assert PRAController.compute_context_defect(0.0, 0.0) == 0.0
        assert PRAController.compute_context_defect(-1.0, -2.0) == 0.0


# =============================================================================
# Section 4.4: Equilibrium-Centered Scheduler
# =============================================================================


class TestEquilibriumScheduler:
    """Tests for the PRA-modulated budget scheduler."""

    def test_positive_budget(self):
        """Scheduler always returns at least 1 token."""
        ctrl = PRAController(h_0=1.0)
        for _ in range(5):
            ctrl.step(0.5)  # Advance state

        budget = ctrl.equilibrium_schedule(100, 0.5)
        assert budget >= 1

    def test_zero_weight_gives_one(self):
        """Depth weight 0 should give minimum budget of 1."""
        ctrl = PRAController(h_0=PHI_F)
        budget = ctrl.equilibrium_schedule(1000, 0.0)
        assert budget == 1

    def test_depth_weights_sum_to_one(self):
        """Casimir depth weights should be normalized."""
        weights = PRAController.compute_depth_weights()
        assert abs(sum(weights) - 1.0) < 1e-10
        assert len(weights) == 8
        assert all(w > 0 for w in weights)

    def test_depth_weights_decrease(self):
        """Weights should roughly decrease with Casimir degree."""
        weights = PRAController.compute_depth_weights()
        # First weight (C=2) should be largest
        assert weights[0] > weights[-1]


# =============================================================================
# Algorithm 5.1: Full Controller Integration
# =============================================================================


class TestAlgorithm51:
    """Tests for the full PRA control loop."""

    def test_step_increments_depth(self):
        """Each step() call increments depth."""
        ctrl = PRAController(h_0=1.0)
        assert ctrl.depth == 0
        ctrl.step(0.5)
        assert ctrl.depth == 1
        ctrl.step(0.3)
        assert ctrl.depth == 2

    def test_first_step_uses_core_recursion(self):
        """At depth 0, step uses h_{n+1} = sqrt(h + 1)."""
        ctrl = PRAController(h_0=1.0)
        state = ctrl.step(0.5)
        # sqrt(1 + 1) = sqrt(2) ≈ 1.414
        assert abs(state.h - math.sqrt(2.0)) < 1e-14

    def test_subsequent_steps_use_defect_controller(self):
        """At depth > 0, step uses defect-sensitive update."""
        ctrl = PRAController(h_0=1.0, kappa=1.0)
        ctrl.step(0.5)  # Depth 0 → core recursion

        # Depth 1 → defect controller: h = sqrt(φ² - (1/3)·ρ)
        state = ctrl.step(0.1)
        expected = math.sqrt(PHI_F ** 2 - (1.0 / 3.0) * 0.1)
        assert abs(state.h - expected) < 1e-14

    def test_reset_clears_state(self):
        """reset() restores controller to initial conditions."""
        ctrl = PRAController(h_0=1.0)
        ctrl.step(0.5)
        ctrl.step(0.3)
        assert ctrl.depth == 2

        ctrl.reset()
        assert ctrl.depth == 0
        assert ctrl.h == 1.0
        assert len(ctrl.history) == 1
        assert len(ctrl.defect_history) == 0

    def test_reset_with_new_h0(self):
        """reset(h_0=X) uses new initial state."""
        ctrl = PRAController(h_0=1.0)
        ctrl.step(0.5)
        ctrl.reset(h_0=2.0)
        assert ctrl.h == 2.0

    def test_full_convergence_loop(self):
        """Full loop: start with high defect, decay to convergence."""
        ctrl = PRAController(
            h_0=0.5, kappa=1.0, epsilon_h=0.05, epsilon_rho=0.1
        )

        rho_0 = 2.0
        q = 0.6
        states = []
        for n in range(50):
            rho = rho_0 * q ** n
            state = ctrl.step(rho)
            states.append(state)
            if state.halted:
                break

        assert states[-1].halted, "Did not halt within 50 steps"
        assert abs(states[-1].h - PHI_F) < 0.1
        assert states[-1].halt_reason == "pra_convergence"

    def test_history_tracking(self):
        """Controller tracks h and defect history."""
        ctrl = PRAController(h_0=1.0)
        ctrl.step(0.5)
        ctrl.step(0.3)

        assert len(ctrl.history) == 3  # h_0, h_1, h_2
        assert len(ctrl.defect_history) == 2
        assert ctrl.history[0] == 1.0
        assert ctrl.defect_history == [0.5, 0.3]

    def test_to_dict_serialization(self):
        """to_dict() returns valid serialization."""
        ctrl = PRAController(h_0=1.0, kappa=2.0)
        ctrl.step(0.5)

        d = ctrl.to_dict()
        assert "h" in d
        assert "kappa" in d
        assert d["kappa"] == 2.0
        assert d["depth"] == 1


# =============================================================================
# Constructor Validation
# =============================================================================


class TestConstructorValidation:
    """Tests for parameter validation."""

    def test_negative_h0_rejected(self):
        with pytest.raises(ValueError, match="nonnegative"):
            PRAController(h_0=-1.0)

    def test_zero_h_target_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            PRAController(h_target=0.0)

    def test_zero_kappa_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            PRAController(kappa=0.0)

    def test_zero_threshold_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            PRAController(epsilon_h=0.0)

    def test_default_construction(self):
        """Default constructor should work with reasonable defaults."""
        ctrl = PRAController()
        assert ctrl.h == 1.0
        assert ctrl.h_target == PHI_F
        assert ctrl.kappa == 1.0
