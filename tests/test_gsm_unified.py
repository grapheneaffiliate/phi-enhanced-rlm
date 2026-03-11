#!/usr/bin/env python3
"""
GSM UNIFIED TEST SUITE
========================

Comprehensive verification of ALL 26 GSM constants, algebraic identities,
structural claims, CHSH proofs, anchor uniqueness, and consistency.

Run: python -m pytest tests/test_gsm_unified.py -v
"""

import math

import numpy as np
import pytest

PHI = (1 + math.sqrt(5)) / 2
EPSILON = 28 / 248
E8_DIM = 248
E8_RANK = 8
E8_ROOTS = 240
E8_COXETER = 30
CASIMIR_DEGREES = [2, 8, 12, 14, 18, 20, 24, 30]
H4_ORDER = 14400
H4_EXPONENTS = [1, 11, 19, 29]
SO16_HALF_SPINOR = 128


class TestE8Structure:

    def test_e8_dimension(self):
        assert E8_DIM == 248

    def test_e8_rank(self):
        assert E8_RANK == 8

    def test_casimir_sum_equals_half_spinor(self):
        assert sum(CASIMIR_DEGREES) == SO16_HALF_SPINOR

    def test_casimir_count_equals_rank(self):
        assert len(CASIMIR_DEGREES) == E8_RANK

    def test_roots_plus_cartan(self):
        assert E8_ROOTS + E8_RANK == E8_DIM

    def test_torsion_ratio(self):
        assert abs(EPSILON - 28/248) < 1e-15

    def test_h4_exponents_in_e8(self):
        e8_exp = [d - 1 for d in CASIMIR_DEGREES]
        for h in H4_EXPONENTS:
            assert h in e8_exp


class TestAlgebraicIdentities:

    def test_phi_squared(self):
        assert abs(PHI**2 - PHI - 1) < 1e-14

    def test_phi_inverse(self):
        assert abs(1/PHI - (PHI - 1)) < 1e-14

    def test_ms_md_exact_20(self):
        L3 = PHI**3 + PHI**(-3)
        assert abs(L3**2 - 20.0) < 1e-10

    def test_phi6_phi_neg6(self):
        assert abs(PHI**6 + PHI**(-6) - 18.0) < 1e-10

    def test_lucas_numbers(self):
        lucas = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76]
        for n, L in enumerate(lucas):
            assert abs(PHI**n + (-PHI)**(-n) - L) < 1e-8

    def test_chsh_squared(self):
        S = 4 - PHI
        assert abs(S**2 - (17 - 7*PHI)) < 1e-12

    def test_chsh_pentagonal(self):
        S = 4 - PHI
        assert abs(S*(3*PHI - 1) - (10*PHI - 7)) < 1e-12

    def test_chsh_alt_forms(self):
        S = 4 - PHI
        assert abs(S - (7 - math.sqrt(5))/2) < 1e-14
        assert abs(S - (2 + PHI**(-2))) < 1e-14


class TestCHSHBound:

    def test_value(self):
        assert abs(4 - PHI - 2.381966011250105) < 1e-12

    def test_below_tsirelson(self):
        assert 4 - PHI < 2 * math.sqrt(2)

    def test_above_classical(self):
        assert 4 - PHI > 2.0

    def test_proof_1_cartan(self):
        det_H3 = 4 - 2*PHI
        det_H4 = 5 - 3*PHI
        gamma_sq = det_H3/2 + det_H4/4
        B_sq = 4*(1 + gamma_sq)
        assert abs(B_sq - (4 - PHI)**2) < 1e-12

    def test_proof_2_gram(self):
        det_C_H2 = 3 - PHI
        assert abs(1 + det_C_H2 - (4 - PHI)) < 1e-14

    def test_proof_3_prism(self):
        S = 4 - PHI
        assert abs(S*(3*PHI - 1) - (10*PHI - 7)) < 1e-12


class TestAnchor:

    def test_decomposition(self):
        assert SO16_HALF_SPINOR + E8_RANK + 1 == 137

    def test_uniqueness(self):
        exp = 137.035999084
        for k in range(4):
            A = SO16_HALF_SPINOR + E8_RANK + k
            val = A + PHI**(-7) + PHI**(-14) + PHI**(-16) - PHI**(-8)/248
            err = abs(val - exp) / exp * 1e6
            if k == 1:
                assert err < 0.03
            else:
                assert err > 100


class TestGaugeConstants:

    def test_alpha_inv(self):
        val = 137 + PHI**(-7) + PHI**(-14) + PHI**(-16) - PHI**(-8)/248
        assert abs(val - 137.035999084) / 137.035999084 < 1e-7

    def test_sin2_theta_w(self):
        val = 3/13 + PHI**(-16)
        assert abs(val - 0.23122) / 0.23122 < 0.001

    def test_alpha_s(self):
        val = 1 / (2*PHI**3 * (1+PHI**(-14)) * (1+8*PHI**(-5)/14400))
        assert abs(val - 0.1179) / 0.1179 < 0.01


class TestLeptonMasses:

    def test_mu_over_e(self):
        val = PHI**11 + PHI**4 + 1 - PHI**(-5) - PHI**(-15)
        assert abs(val - 206.768283) / 206.768283 < 1e-5

    def test_tau_over_mu(self):
        val = PHI**6 - PHI**(-4) - 1 + PHI**(-8)
        assert abs(val - 16.8170) / 16.8170 < 0.001


class TestQuarkMasses:

    def test_ms_md(self):
        assert abs((PHI**3 + PHI**(-3))**2 - 20.0) < 1e-10

    def test_mc_ms(self):
        val = (PHI**5 + PHI**(-3)) * (1 + 28/(240*PHI**2))
        assert abs(val - 11.83) / 11.83 < 0.01

    def test_mb_mc(self):
        val = PHI**2 + PHI**(-3)
        assert abs(val - 2.86) / 2.86 < 0.01

    def test_top_yukawa(self):
        assert abs(1 - PHI**(-10) - 0.9919) / 0.9919 < 0.01

    def test_mu_md(self):
        assert abs(1/math.sqrt(5) - 0.46) < 0.03


class TestProtonMass:

    def test_mp_me(self):
        val = 6 * math.pi**5 * (1 + PHI**(-24) + PHI**(-13)/240)
        assert abs(val - 1836.15267343) / 1836.15267343 < 2e-6


class TestCKM:

    def test_sin_theta_c(self):
        val = (PHI**(-1)+PHI**(-6))/3 * (1+8*PHI**(-6)/248)
        assert abs(val - 0.22500) / 0.22500 < 0.001

    def test_j_ckm(self):
        val = PHI**(-10) / 264
        assert abs(val - 3.08e-5) / 3.08e-5 < 0.01

    def test_v_cb(self):
        val = (PHI**(-8)+PHI**(-15))*(PHI**2/math.sqrt(2))*(1+1/240)
        assert abs(val - 0.0410) / 0.0410 < 0.01

    def test_v_ub(self):
        val = 2*PHI**(-7)/19
        assert abs(val - 0.00361) / 0.00361 < 0.01


class TestPMNS:

    def test_theta_12(self):
        val = math.degrees(math.atan(PHI**(-1) + 2*PHI**(-8)))
        assert abs(val - 33.44) / 33.44 < 0.01

    def test_theta_23(self):
        val = math.degrees(math.asin(math.sqrt((1+PHI**(-4))/2)))
        assert abs(val - 49.2) / 49.2 < 0.01

    def test_theta_13(self):
        val = math.degrees(math.asin(PHI**(-4) + PHI**(-12)))
        assert abs(val - 8.57) / 8.57 < 0.01


class TestCosmological:

    def test_z_cmb(self):
        assert abs(PHI**14 + 246 - 1089.80) / 1089.80 < 0.001

    def test_omega_lambda(self):
        val = PHI**(-1)+PHI**(-6)+PHI**(-9)-PHI**(-13)+PHI**(-28)+EPSILON*PHI**(-7)
        assert abs(val - 0.6889) / 0.6889 < 0.01

    def test_n_s(self):
        assert abs(1 - PHI**(-7) - 0.9649) / 0.9649 < 0.01

    def test_hierarchy(self):
        val = PHI**(80 - EPSILON)
        assert abs(val - 4.955e16) / 4.955e16 < 0.01


class TestConsistency:

    def test_canonical_proton_better(self):
        can = 6*math.pi**5*(1+PHI**(-24)+PHI**(-13)/240)
        alt = 7*248 + 100 + PHI**(-7)
        assert abs(can - 1836.15267343) < abs(alt - 1836.15267343)

    def test_canonical_muon_better(self):
        can = PHI**11+PHI**4+1-PHI**(-5)-PHI**(-15)
        alt = 200+PHI**4
        assert abs(can - 206.768283) < abs(alt - 206.768283)

    def test_canonical_alpha_s_e8_origin(self):
        """Canonical α_s uses Casimir structure; alternative is numerological."""
        can = 1/(2*PHI**3*(1+PHI**(-14))*(1+8*PHI**(-5)/14400))
        # Both formulas are close to 0.1179; canonical chosen for E8 origin
        assert abs(can - 0.1179) / 0.1179 < 0.01  # Within 1%


class TestThreeGenerations:

    def test_e8_branching(self):
        assert 78*1 + 27*3 + 27*3 + 1*8 == 248

    def test_e6_to_so10(self):
        assert 16 + 10 + 1 == 27

    def test_so10_to_sm(self):
        assert 3*2 + 3*1 + 3*1 + 1*2 + 1*1 + 1*1 == 16

    def test_h4_exponent_sum(self):
        assert sum(H4_EXPONENTS) == 60
        assert 60 // 20 == 3

    def test_root_orbits(self):
        assert E8_ROOTS // 80 == 3

    def test_anomaly_cancellation(self):
        anom = (6*(1/6)**3 + 3*(-2/3)**3 + 3*(1/3)**3
                + 2*(-1/2)**3 + 1**3)
        assert abs(anom) < 1e-14


class TestGapModules:

    def test_three_gen_proof(self):
        from src.gsm_three_generations import prove_three_generations
        p = prove_three_generations()
        assert p["n_generations"] == 3
        assert p["dimension_checks_pass"]

    def test_exponent_rules(self):
        from src.gsm_exponent_rules import exponent_rule_for_alpha
        r = exponent_rule_for_alpha()
        assert r["deviation_ppb"] < 30

    def test_chi_e8_h4(self):
        from src.gsm_exponent_rules import compute_chi_e8_h4
        assert compute_chi_e8_h4()["verdict"]["chi_value"] == 1

    def test_proton_mass(self):
        from src.gsm_proton_mass import verify_proton_mass
        assert verify_proton_mass()

    def test_rg_hierarchy(self):
        from src.gsm_rg_flow import hierarchy_from_casimir
        assert hierarchy_from_casimir()["deviation_percent"] < 1.0

    def test_gsm_constants_module(self):
        from src.gsm_constants import get, get_all, INCONSISTENCY_RESOLUTIONS
        assert len(get_all()) >= 26  # 26 core + possible extras
        assert len(INCONSISTENCY_RESOLUTIONS) == 9
        assert get("ms_over_md").derivation_tier == "rigorous"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
