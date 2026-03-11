#!/usr/bin/env python3
"""Tests for SuperpowersOrchestrator and workflow integration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.meta_recursion import (
    QUERY_TYPE_KEYWORDS,
    QUERY_TYPE_STRATEGY,
    STRATEGIES,
    MetaRecursiveRLM,
)
from src.phi_enhanced_rlm import MockLLMBackend, PhiEnhancedRLM
from src.skill_loader import SUPERPOWERS_KEYWORDS
from src.workflow_orchestrator import (
    WORKFLOW_PIPELINE,
    SuperpowersOrchestrator,
)


class TestWorkflowPipeline:
    """Test the WORKFLOW_PIPELINE definition."""

    def test_pipeline_has_8_phases(self):
        assert len(WORKFLOW_PIPELINE) == 8

    def test_phases_map_to_casimir_degrees(self):
        expected_degrees = [2, 8, 12, 14, 18, 20, 24, 30]
        actual = [step.casimir_degree for step in WORKFLOW_PIPELINE]
        assert actual == expected_degrees

    def test_all_phases_have_prompt_injection(self):
        for step in WORKFLOW_PIPELINE:
            assert step.prompt_injection, f"Phase {step.phase} missing prompt_injection"

    def test_all_phases_have_hard_gates(self):
        for step in WORKFLOW_PIPELINE:
            assert step.hard_gate, f"Phase {step.phase} missing hard_gate"

    def test_all_phases_have_red_flags(self):
        for step in WORKFLOW_PIPELINE:
            assert len(step.red_flags) > 0, f"Phase {step.phase} missing red_flags"

    def test_all_phases_have_rationalization_traps(self):
        for step in WORKFLOW_PIPELINE:
            assert len(step.rationalization_traps) > 0, f"Phase {step.phase} missing traps"

    def test_phase_names(self):
        names = [s.phase for s in WORKFLOW_PIPELINE]
        assert "clarify" in names
        assert "plan" in names
        assert "execute" in names
        assert "tdd" in names
        assert "spec_review" in names
        assert "quality_review" in names
        assert "verify" in names
        assert "debug" in names


class TestSuperpowersOrchestrator:
    """Test the orchestrator wrapping phi-RLM."""

    def _make_rlm(self):
        return PhiEnhancedRLM(
            base_llm_callable=MockLLMBackend(seed=42),
            context_chunks=["Build systems.", "Test everything.", "Verify results."],
        )

    def test_orchestrated_solve_returns_dict(self):
        rlm = self._make_rlm()
        sp = SuperpowersOrchestrator(rlm)
        result = sp.orchestrated_solve("Build a calculator with tests")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "confidence" in result
        assert "workflow_trace" in result
        assert "all_gates_passed" in result

    def test_confidence_is_float(self):
        rlm = self._make_rlm()
        sp = SuperpowersOrchestrator(rlm)
        result = sp.orchestrated_solve("What is 2+2?")
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_workflow_trace_populated(self):
        rlm = self._make_rlm()
        sp = SuperpowersOrchestrator(rlm)
        result = sp.orchestrated_solve("Explain recursion")
        assert len(result["workflow_trace"]) >= 1

    def test_get_workflow_summary(self):
        rlm = self._make_rlm()
        sp = SuperpowersOrchestrator(rlm)
        sp.orchestrated_solve("Test query")
        summary = sp.get_workflow_summary()
        assert "Workflow trace:" in summary

    def test_skill_content_missing_dir(self):
        rlm = self._make_rlm()
        sp = SuperpowersOrchestrator(rlm)
        content = sp.get_skill_content("nonexistent-skill")
        assert content == ""

    def test_custom_confidence_threshold(self):
        rlm = self._make_rlm()
        sp = SuperpowersOrchestrator(rlm, confidence_threshold=0.99)
        result = sp.orchestrated_solve("Simple question")
        # With a very high threshold, gate is unlikely to pass
        assert isinstance(result["all_gates_passed"], bool)


class TestSuperpowersStrategy:
    """Test the superpowers_workflow strategy in MetaRecursiveRLM."""

    def test_strategy_exists(self):
        assert "superpowers_workflow" in STRATEGIES

    def test_strategy_max_depth_7(self):
        assert STRATEGIES["superpowers_workflow"].max_depth == 7

    def test_strategy_casimir_budget(self):
        assert STRATEGIES["superpowers_workflow"].budget_distribution == "casimir"

    def test_strategy_phi_attention_enabled(self):
        assert STRATEGIES["superpowers_workflow"].phi_attention is True

    def test_development_query_type_exists(self):
        assert "development" in QUERY_TYPE_KEYWORDS

    def test_development_maps_to_superpowers(self):
        assert QUERY_TYPE_STRATEGY["development"] == "superpowers_workflow"

    def test_development_keywords(self):
        kw = QUERY_TYPE_KEYWORDS["development"]
        assert "build" in kw
        assert "implement" in kw
        assert "fix" in kw
        assert "deploy" in kw

    def test_meta_rlm_selects_superpowers_for_development(self):
        rlm = PhiEnhancedRLM(
            base_llm_callable=MockLLMBackend(seed=42),
            context_chunks=["Build systems."],
        )
        meta = MetaRecursiveRLM(rlm)
        qtype = meta.analyze_query_type("build and implement a new feature")
        assert qtype == "development"


class TestSuperpowersKeywords:
    """Test superpowers keyword mappings in SkillLoader."""

    def test_keywords_dict_exists(self):
        assert isinstance(SUPERPOWERS_KEYWORDS, dict)

    def test_key_skills_present(self):
        assert "brainstorming" in SUPERPOWERS_KEYWORDS
        assert "systematic-debugging" in SUPERPOWERS_KEYWORDS
        assert "test-driven-development" in SUPERPOWERS_KEYWORDS
        assert "verification-before-completion" in SUPERPOWERS_KEYWORDS

    def test_keywords_are_lists(self):
        for skill, keywords in SUPERPOWERS_KEYWORDS.items():
            assert isinstance(keywords, list), f"{skill} keywords not a list"
            assert len(keywords) > 0, f"{skill} has no keywords"


class TestPhiCriticQECIntegration:
    """Test that PhiCritic is wired into QEC verification."""

    def test_qec_returns_4_verifiers(self):
        rlm = PhiEnhancedRLM(
            base_llm_callable=MockLLMBackend(seed=42),
            context_chunks=["Test context."],
        )
        conf, results = rlm.run_qec_verification(
            answer="The answer is 42",
            context="Math question",
            budget=300,
        )
        # Should have 4 results: 3 standard + 1 critic
        assert len(results) == 4
        assert results[3].metadata.get("type") == "critic"

    def test_qec_confidence_valid(self):
        rlm = PhiEnhancedRLM(
            base_llm_callable=MockLLMBackend(seed=42),
            context_chunks=["Test context."],
        )
        conf, results = rlm.run_qec_verification(
            answer="Test answer",
            context="Test context",
            budget=300,
        )
        assert 0.1 <= conf <= 0.99


class TestExports:
    """Test that new modules are properly exported."""

    def test_import_orchestrator(self):
        from src import SuperpowersOrchestrator
        assert SuperpowersOrchestrator is not None

    def test_import_pipeline(self):
        from src import WORKFLOW_PIPELINE
        assert len(WORKFLOW_PIPELINE) == 8
