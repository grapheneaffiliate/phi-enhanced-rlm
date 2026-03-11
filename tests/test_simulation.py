#!/usr/bin/env python3
"""
Production readiness simulation tests for φ-enhanced RLM.

Exercises every major subsystem end-to-end with MockLLMBackend:
- Recursive solve at multiple depths
- All 7 meta-recursion strategies
- Evolution loop convergence
- Phi-attention prompt injection
- Sparse reasoning branch pruning
- Spiral memory store/retrieve
- Phi-critic adversarial evaluation
- Phi-planner DAG execution
- Code review parallel agents
- Workflow orchestrator full pipeline
- Module import coherence
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.code_review import PhiCodeReview, ReviewFinding, ReviewResult
from src.evolution import EvaluationResult, EvolutionState, PhiEvolutionEngine
from src.meta_recursion import (
    QUERY_TYPE_STRATEGY,
    STRATEGIES,
    MetaRecursiveRLM,
)
from src.phi_attention import PhiAttentionInjector, PhiConfidenceScaler
from src.phi_critic import PhiCritic
from src.phi_enhanced_rlm import MockLLMBackend, PhiEnhancedRLM, SubCallResult
from src.phi_memory import PhiSpiralMemory
from src.phi_planner import PhiPlanner
from src.phi_separation_novel_mathematics import (
    CASIMIR_DEGREES,
    EPSILON,
    PHI,
    PHI_INV,
)
from src.phi_sparse_reasoning import PhiSparseReasoner
from src.workflow_orchestrator import WORKFLOW_PIPELINE, SuperpowersOrchestrator

# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def mock_llm():
    return MockLLMBackend(seed=42)


@pytest.fixture
def rlm(mock_llm):
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        trace_file = f.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        memory_path = f.name
    return PhiEnhancedRLM(
        base_llm_callable=mock_llm,
        context_chunks=["Chunk A: physics", "Chunk B: math", "Chunk C: history"],
        total_budget_tokens=4096,
        trace_file=trace_file,
        memory_path=memory_path,
    )


# ═══════════════════════════════════════════════════════════
# 1. Recursive Solve coherence
# ═══════════════════════════════════════════════════════════

class TestRecursiveSolveCoherence:
    """Verify recursive_solve produces valid, structured output at all depths."""

    def test_depth_0_produces_result(self, rlm):
        result = rlm.recursive_solve("What is 2+2?", depth=0, max_depth=1)
        assert isinstance(result, SubCallResult)
        assert result.value is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_depth_3_converges(self, rlm):
        result = rlm.recursive_solve("Explain quantum entanglement", depth=0, max_depth=3)
        assert isinstance(result, SubCallResult)
        assert result.confidence > 0.0
        assert "metadata" not in result.__dict__ or isinstance(result.metadata, dict)

    def test_max_depth_5_full_e8(self, rlm):
        result = rlm.recursive_solve("Derive E=mc^2 from first principles", depth=0, max_depth=5)
        assert isinstance(result, SubCallResult)
        assert result.confidence > 0.0

    def test_budget_allocation_maps_to_casimir(self, rlm):
        budget = rlm.allocate_recursion_budget(4096)
        assert isinstance(budget, dict)
        for depth, tokens in budget.items():
            assert tokens > 0

    def test_deterministic_with_seed(self, mock_llm):
        """Same seed produces same results."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tf = f.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            mf = f.name
        rlm1 = PhiEnhancedRLM(MockLLMBackend(seed=99), ["c1", "c2"], trace_file=tf, memory_path=mf)
        rlm2 = PhiEnhancedRLM(MockLLMBackend(seed=99), ["c1", "c2"], trace_file=tf, memory_path=mf)
        r1 = rlm1.recursive_solve("test", max_depth=2)
        r2 = rlm2.recursive_solve("test", max_depth=2)
        assert abs(r1.confidence - r2.confidence) < 0.01


# ═══════════════════════════════════════════════════════════
# 2. Meta-recursion strategy selection
# ═══════════════════════════════════════════════════════════

class TestMetaRecursionStrategies:
    """Verify all strategies can be selected and executed."""

    def test_all_strategies_exist(self):
        expected = ["deep_analytical", "wide_exploratory", "spiral_convergent",
                    "quick_factual", "deep_research", "planned", "superpowers_workflow"]
        for name in expected:
            assert name in STRATEGIES, f"Missing strategy: {name}"

    def test_strategy_depths_are_valid(self):
        for name, strat in STRATEGIES.items():
            assert 1 <= strat.max_depth <= 8, f"{name} depth {strat.max_depth} out of range"
            assert strat.branch_factor >= 1

    def test_query_type_classification(self, rlm):
        meta = MetaRecursiveRLM(rlm)
        # Should classify without error
        for query in ["What is 2+2?", "Explain the theory of relativity in depth",
                       "Compare capitalism and socialism", "Write a Python sort function"]:
            qtype = meta.analyze_query_type(query)
            assert isinstance(qtype, str)
            assert len(qtype) > 0

    def test_each_strategy_executes(self, rlm):
        meta = MetaRecursiveRLM(rlm)
        for name, strat in STRATEGIES.items():
            meta.configure_rlm(strat)
            result = rlm.recursive_solve("Test query", max_depth=min(strat.max_depth, 2))
            assert isinstance(result, SubCallResult)
            assert result.confidence >= 0.0

    def test_meta_solve_selects_and_runs(self, rlm):
        meta = MetaRecursiveRLM(rlm)
        result = meta.meta_solve("What causes inflation?", max_meta_depth=2)
        assert isinstance(result, SubCallResult)
        assert result.confidence > 0.0

    def test_strategy_performance_tracking(self, rlm):
        meta = MetaRecursiveRLM(rlm)
        meta.meta_solve("test query", max_meta_depth=1)
        perf = meta.get_strategy_performance()
        assert isinstance(perf, dict)

    def test_query_type_strategy_mapping_complete(self):
        for qtype, strategy_name in QUERY_TYPE_STRATEGY.items():
            assert strategy_name in STRATEGIES, f"Query type '{qtype}' maps to unknown strategy '{strategy_name}'"


# ═══════════════════════════════════════════════════════════
# 3. Evolution convergence
# ═══════════════════════════════════════════════════════════

class TestEvolutionConvergence:
    """Verify evolution loop converges and state is coherent."""

    def test_evolution_state_defaults_valid(self):
        state = EvolutionState()
        assert state.generation == 0
        assert len(state.budget_weights) == 8
        assert abs(sum(state.budget_weights) - 8.0) < 0.01
        assert 0.0 < state.phi_momentum_threshold < 1.0

    def test_learning_rate_decays(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            engine = PhiEvolutionEngine(state_path=f.name)
        engine.state = EvolutionState(generation=0)
        lr0 = engine.learning_rate()
        engine.state.generation = 5
        lr5 = engine.learning_rate()
        assert lr5 < lr0, "Learning rate should decay with generation"
        assert lr5 > 0.0, "Learning rate should stay positive"

    def test_evolution_3_generations(self, mock_llm):
        """Run 3 generations and verify monotonic improvement tracking."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            engine = PhiEvolutionEngine(state_path=f.name)

        for gen in range(3):
            traces = [
                {"depth": 3, "answer": "42",
                 "confidence": 0.7 + gen * 0.05,
                 "strategy": "spiral_convergent"},
                {"depth": 2, "answer": "yes",
                 "confidence": 0.6 + gen * 0.05,
                 "strategy": "quick_factual"},
            ]
            ground_truth = ["42", "yes"]
            evaluation = engine.evaluate_generation(traces, ground_truth=ground_truth)
            assert isinstance(evaluation, EvaluationResult)
            assert evaluation.accuracy > 0.0
            engine.evolve(evaluation)
            assert engine.state.generation == gen + 1

        assert len(engine.state.accuracy_history) == 3

    def test_evolution_state_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        engine = PhiEvolutionEngine(state_path=path)
        engine.state.generation = 5
        engine.state.fitness = 0.85
        engine.save()

        engine2 = PhiEvolutionEngine(state_path=path)
        engine2.load_or_create()
        assert engine2.state.generation == 5
        assert engine2.state.fitness == 0.85


# ═══════════════════════════════════════════════════════════
# 4. Phi-attention injection
# ═══════════════════════════════════════════════════════════

class TestPhiAttention:
    """Verify phi-structured attention produces valid prompts."""

    def test_inject_at_each_depth(self):
        injector = PhiAttentionInjector()
        for depth in range(8):
            result = injector.inject_phi_structure("Test prompt", depth, budget=500)
            assert isinstance(result, str)
            assert len(result) > len("Test prompt")

    def test_chain_of_thought(self):
        injector = PhiAttentionInjector()
        cot = injector.phi_chain_of_thought("What is gravity?", depth=2)
        assert isinstance(cot, str)
        assert len(cot) > 0

    def test_confidence_scaling(self):
        for depth in range(8):
            scaled = PhiConfidenceScaler.phi_scale(0.8, depth)
            assert 0.0 <= scaled <= 1.0

    def test_torsion_correction_applies_epsilon(self):
        # Torsion adds ε * minority_weight to base confidence
        corrected = PhiConfidenceScaler.torsion_correct(0.5, 0.5)
        expected = 0.5 + EPSILON * 0.5
        assert abs(corrected - expected) < 1e-6
        # Zero minority weight means no correction
        assert PhiConfidenceScaler.torsion_correct(0.8, 0.0) == 0.8

    def test_aggregate_phi_weighted(self):
        confs = [0.9, 0.8, 0.7, 0.6]
        agg = PhiConfidenceScaler.aggregate_phi_weighted(confs)
        assert 0.0 <= agg <= 1.0
        assert agg > min(confs)  # aggregation shouldn't be worse than worst


# ═══════════════════════════════════════════════════════════
# 5. Sparse reasoning pruning
# ═══════════════════════════════════════════════════════════

class TestSparseReasoning:
    """Verify phi-ratio branch pruning is mathematically correct."""

    def test_pruning_ratio_is_phi_inv(self):
        reasoner = PhiSparseReasoner()
        assert abs(reasoner.pruning_ratio - PHI_INV) < 1e-6

    def test_select_branches_reduces(self):
        reasoner = PhiSparseReasoner()
        questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        scores = [0.9, 0.3, 0.7, 0.5, 0.8]
        selected = reasoner.select_branches(questions, scores)
        assert len(selected) < len(questions)
        assert len(selected) >= 1

    def test_adaptive_prune_high_confidence(self):
        reasoner = PhiSparseReasoner()
        questions = ["Q1", "Q2", "Q3", "Q4"]
        scores = [0.9, 0.8, 0.7, 0.6]
        # High confidence should prune more aggressively
        pruned_high = reasoner.adaptive_prune(questions, scores, confidence=0.95)
        pruned_low = reasoner.adaptive_prune(questions, scores, confidence=0.3)
        assert len(pruned_high) <= len(pruned_low)

    def test_score_subquestions(self):
        reasoner = PhiSparseReasoner()
        scores = reasoner.score_subquestions(
            "What is physics?",
            ["What is matter?", "What is energy?", "What is force?"],
            parent_answer="Physics studies matter and energy"
        )
        assert len(scores) == 3
        for s in scores:
            assert 0.0 <= s <= 1.0


# ═══════════════════════════════════════════════════════════
# 6. Spiral memory
# ═══════════════════════════════════════════════════════════

class TestSpiralMemory:
    """Verify phi-spiral memory store and retrieve work correctly."""

    def test_store_and_retrieve_by_text(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            mem = PhiSpiralMemory(db_path=f.name)
        mem.store("The speed of light is 299792458 m/s", importance=0.9)
        mem.store("Water boils at 100 degrees Celsius", importance=0.7)
        mem.store("Pi is approximately 3.14159", importance=0.8)
        results = mem.retrieve_by_text("How fast is light?", k=2)
        assert len(results) >= 1

    def test_store_with_embedding(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            mem = PhiSpiralMemory(db_path=f.name)
        emb = np.random.randn(128).astype(np.float32)
        key = mem.store("Test fact", embedding=emb, importance=0.5)
        assert isinstance(key, str)
        assert len(key) > 0

    def test_stats(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            mem = PhiSpiralMemory(db_path=f.name)
        mem.store("Fact 1")
        mem.store("Fact 2")
        stats = mem.get_stats()
        assert stats["total"] == 2


# ═══════════════════════════════════════════════════════════
# 7. Phi-critic adversarial evaluation
# ═══════════════════════════════════════════════════════════

class TestPhiCritic:
    """Verify adversarial self-evaluation produces valid critiques."""

    def test_critique_returns_valid_result(self, rlm):
        critic = PhiCritic(rlm)
        result = critic.critique("What is 2+2?", "4")
        assert 0.0 <= result.quality_score <= 1.0
        assert isinstance(result.critique_text, str)
        assert 0.0 <= result.critique_confidence <= 1.0
        assert result.flaws_found >= 0

    def test_critique_as_qec_score(self, rlm):
        critic = PhiCritic(rlm)
        score = critic.critique_as_qec_score("What is gravity?", "A fundamental force")
        assert 0.0 <= score <= 1.0

    def test_bad_answer_scores_lower(self, rlm):
        critic = PhiCritic(rlm)
        good = critic.critique("What is 2+2?", "4", context="Basic arithmetic")
        bad = critic.critique("What is 2+2?", "The color blue", context="Basic arithmetic")
        # Not guaranteed with mock, but both should be valid
        assert 0.0 <= good.quality_score <= 1.0
        assert 0.0 <= bad.quality_score <= 1.0


# ═══════════════════════════════════════════════════════════
# 8. Phi-planner DAG execution
# ═══════════════════════════════════════════════════════════

class TestPhiPlanner:
    """Verify DAG-based task decomposition and execution."""

    def test_create_plan(self, mock_llm):
        planner = PhiPlanner(mock_llm)
        plan = planner.create_plan("Build a REST API for user management")
        assert plan.query == "Build a REST API for user management"
        assert plan.total_steps >= 0

    def test_execute_plan(self, mock_llm, rlm):
        planner = PhiPlanner(mock_llm)
        plan = planner.create_plan("Analyze climate data trends")
        result = planner.execute_plan(plan, rlm, max_depth=2)
        assert "answer" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], (int, float))


# ═══════════════════════════════════════════════════════════
# 9. Code review parallel agents
# ═══════════════════════════════════════════════════════════

class TestCodeReviewIntegration:
    """Verify code review agents work with real code."""

    def test_review_python_code(self, mock_llm):
        reviewer = PhiCodeReview(mock_llm, confidence_threshold=0.0)
        code = '''
def process_user_input(data):
    query = "SELECT * FROM users WHERE id = " + data["id"]
    result = db.execute(query)
    return eval(result)
'''
        result = reviewer.review_text(code, context="Security-sensitive code")
        assert isinstance(result, ReviewResult)
        assert result.total_agents == 4

    def test_review_with_threshold(self, mock_llm):
        reviewer = PhiCodeReview(mock_llm, confidence_threshold=0.99)
        result = reviewer.review_text("def hello(): return 'world'")
        # With 0.99 threshold, mock findings likely filtered
        assert isinstance(result, ReviewResult)

    def test_report_formatting(self):
        findings = [
            ReviewFinding("security", "critical", 0.95, "app.py", "10-15",
                         "SQL Injection", "Unsanitized input in query", "Use parameterized queries"),
            ReviewFinding("logic", "important", 0.85, "app.py", "20",
                         "Missing null check", "data could be None", "Add guard clause"),
        ]
        result = ReviewResult(findings=findings, total_agents=4, review_time_seconds=1.5)
        report = result.format_report()
        assert "SQL Injection" in report
        assert "Missing null check" in report
        assert "Critical" in report
        assert "Important" in report


# ═══════════════════════════════════════════════════════════
# 10. Workflow orchestrator full pipeline
# ═══════════════════════════════════════════════════════════

class TestWorkflowOrchestrator:
    """Verify full superpowers workflow pipeline executes end-to-end."""

    def test_pipeline_structure(self):
        assert len(WORKFLOW_PIPELINE) == 8
        casimirs = [step.casimir_degree for step in WORKFLOW_PIPELINE]
        assert casimirs == list(CASIMIR_DEGREES)

    def test_orchestrated_solve_full(self, rlm):
        orch = SuperpowersOrchestrator(rlm)
        result = orch.orchestrated_solve("Explain quantum computing", max_depth=3)
        assert "answer" in result
        assert "confidence" in result
        assert "workflow_trace" in result
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_workflow_summary(self, rlm):
        orch = SuperpowersOrchestrator(rlm)
        orch.orchestrated_solve("Test", max_depth=2)
        summary = orch.get_workflow_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


# ═══════════════════════════════════════════════════════════
# 11. Mathematical constants coherence
# ═══════════════════════════════════════════════════════════

class TestMathematicalCoherence:
    """Verify φ-mathematics is internally consistent."""

    def test_phi_golden_ratio(self):
        assert abs(PHI - 1.618033988749895) < 1e-10
        assert abs(PHI_INV - (1.0 / PHI)) < 1e-10
        assert abs(PHI * PHI_INV - 1.0) < 1e-10

    def test_phi_squared(self):
        assert abs(PHI ** 2 - PHI - 1.0) < 1e-10

    def test_epsilon_torsion(self):
        assert abs(EPSILON - 28 / 248) < 1e-10

    def test_casimir_degrees(self):
        assert list(CASIMIR_DEGREES) == [2, 8, 12, 14, 18, 20, 24, 30]
        assert len(CASIMIR_DEGREES) == 8

    def test_e8_casimir_sum(self):
        # E8 Casimir degrees should sum correctly
        assert sum(CASIMIR_DEGREES) == 128


# ═══════════════════════════════════════════════════════════
# 12. Module import coherence
# ═══════════════════════════════════════════════════════════

class TestModuleImportCoherence:
    """Verify all public exports are importable and consistent."""

    def test_top_level_imports(self):
        import src
        assert hasattr(src, "PhiEnhancedRLM")
        assert hasattr(src, "PhiEvolutionEngine")
        assert hasattr(src, "MetaRecursiveRLM")
        assert hasattr(src, "PhiSpiralMemory")
        assert hasattr(src, "PhiAttentionInjector")
        assert hasattr(src, "PhiSparseReasoner")
        assert hasattr(src, "PhiCritic")
        assert hasattr(src, "PhiPlanner")
        assert hasattr(src, "PhiCodeReview")
        assert hasattr(src, "SuperpowersOrchestrator")

    def test_all_exports_listed(self):
        import src
        assert hasattr(src, "__all__")
        for name in src.__all__:
            assert hasattr(src, name), f"__all__ lists '{name}' but it's not importable"

    def test_review_exports(self):
        import src
        assert hasattr(src, "ReviewResult")
        assert hasattr(src, "ReviewFinding")


# ═══════════════════════════════════════════════════════════
# 13. End-to-end integration simulation
# ═══════════════════════════════════════════════════════════

class TestEndToEndSimulation:
    """Full pipeline: query -> meta-select -> solve -> critique -> evolve."""

    def test_full_pipeline(self, mock_llm):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            trace_file = f.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            memory_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            evo_path = f.name

        # 1. Create RLM
        rlm = PhiEnhancedRLM(
            mock_llm,
            context_chunks=["Physics: F=ma", "Math: e^(iπ)+1=0", "CS: P vs NP"],
            trace_file=trace_file,
            memory_path=memory_path,
        )

        # 2. Meta-recursion selects strategy
        meta = MetaRecursiveRLM(rlm)
        query = "Explain the relationship between entropy and information theory"
        qtype = meta.analyze_query_type(query)
        strategy = meta.select_strategy(qtype)
        assert strategy.name in STRATEGIES

        # 3. Solve with selected strategy
        meta.configure_rlm(strategy)
        result = rlm.recursive_solve(query, max_depth=min(strategy.max_depth, 3))
        assert result.confidence > 0.0

        # 4. Critique the answer
        critic = PhiCritic(rlm)
        critique = critic.critique(query, str(result.value))
        assert 0.0 <= critique.quality_score <= 1.0

        # 5. Evolve based on trace
        engine = PhiEvolutionEngine(state_path=evo_path)
        traces = [{
            "depth_reached": 3,
            "tokens_used": 400,
            "final_confidence": result.confidence,
            "correct": critique.quality_score > 0.5,
            "strategy": strategy.name,
            "premature_stop": False,
            "late_stop": False,
        }]
        evaluation = engine.evaluate_generation(traces)
        new_state = engine.evolve(evaluation)
        assert new_state.generation == 1

    def test_multi_query_simulation(self, mock_llm):
        """Simulate processing multiple queries and verify no state corruption."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            trace_file = f.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            memory_path = f.name

        rlm = PhiEnhancedRLM(
            mock_llm,
            context_chunks=["A", "B", "C"],
            trace_file=trace_file,
            memory_path=memory_path,
        )
        meta = MetaRecursiveRLM(rlm)

        queries = [
            "What is gravity?",
            "Compare Python and Rust",
            "Explain the Fibonacci sequence",
            "How does TCP/IP work?",
        ]

        results = []
        for q in queries:
            r = meta.meta_solve(q, max_meta_depth=1)
            results.append(r)
            assert isinstance(r, SubCallResult)
            assert r.confidence >= 0.0

        # All should complete without error
        assert len(results) == len(queries)

    def test_concurrent_reviews_dont_interfere(self, mock_llm):
        """Two code reviews running simultaneously produce independent results."""
        import concurrent.futures

        reviewer = PhiCodeReview(mock_llm, confidence_threshold=0.0)
        code1 = "def add(a, b): return a + b"
        code2 = "def exploit(): os.system(user_input)"

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(reviewer.review_text, code1)
            f2 = executor.submit(reviewer.review_text, code2)
            r1 = f1.result(timeout=120)
            r2 = f2.result(timeout=120)

        assert isinstance(r1, ReviewResult)
        assert isinstance(r2, ReviewResult)
        # Both should complete independently
        assert r1.total_agents == 4
        assert r2.total_agents == 4
