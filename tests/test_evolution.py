#!/usr/bin/env python3
"""Tests for the φ-governed self-evolution engine."""

import json
import os
import sys
import tempfile
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evolution import PhiEvolutionEngine, EvolutionState, EvaluationResult
from phi_separation_novel_mathematics import PHI, PHI_INV


class TestEvolutionState:
    """Tests for EvolutionState dataclass."""

    def test_default_state(self):
        state = EvolutionState()
        assert state.generation == 0
        assert len(state.budget_weights) == 8
        assert state.phi_momentum_threshold == 0.93
        assert state.pruning_ratio == pytest.approx(0.618, abs=0.001)

    def test_serialization_roundtrip(self):
        state = EvolutionState(generation=5, fitness=0.75)
        data = state.to_dict()
        restored = EvolutionState.from_dict(data)
        assert restored.generation == 5
        assert restored.fitness == 0.75

    def test_budget_weights_normalization(self):
        state = EvolutionState(budget_weights=[2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        weights = state.get_budget_weights_array()
        assert weights.sum() == pytest.approx(1.0, abs=0.01)
        assert weights[0] > weights[1]  # First should be highest


class TestPhiEvolutionEngine:
    """Tests for the evolution engine."""

    def test_fresh_engine_creation(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            engine = PhiEvolutionEngine(state_path=path)
            assert engine.state.generation == 0
        finally:
            os.unlink(path)

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            engine = PhiEvolutionEngine(state_path=path)
            engine.state.generation = 3
            engine.state.fitness = 0.85
            engine.save()

            engine2 = PhiEvolutionEngine(state_path=path)
            assert engine2.state.generation == 3
            assert engine2.state.fitness == 0.85
        finally:
            os.unlink(path)

    def test_learning_rate_decay(self):
        engine = PhiEvolutionEngine(state_path="/tmp/test_evo_lr.json")
        engine.state.generation = 0
        assert engine.learning_rate() == pytest.approx(1.0)

        engine.state.generation = 1
        assert engine.learning_rate() == pytest.approx(PHI_INV, abs=0.001)

        engine.state.generation = 2
        assert engine.learning_rate() == pytest.approx(PHI_INV ** 2, abs=0.001)

        # Ensure it never reaches zero
        engine.state.generation = 100
        assert engine.learning_rate() > 0

        if os.path.exists("/tmp/test_evo_lr.json"):
            os.unlink("/tmp/test_evo_lr.json")

    def test_evaluate_empty_traces(self):
        engine = PhiEvolutionEngine(state_path="/tmp/test_evo_eval.json")
        result = engine.evaluate_generation([])
        assert result.accuracy == 0.0

        if os.path.exists("/tmp/test_evo_eval.json"):
            os.unlink("/tmp/test_evo_eval.json")

    def test_evolve_updates_generation(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            engine = PhiEvolutionEngine(state_path=path)
            assert engine.state.generation == 0

            eval_result = EvaluationResult(
                generation=0, accuracy=0.6, avg_tokens=500,
                avg_depth=3, avg_confidence=0.7, improvement=0.1,
                premature_stops=1, late_stops=2,
            )
            engine.evolve(eval_result)
            assert engine.state.generation == 1
            assert len(engine.state.accuracy_history) == 1
        finally:
            os.unlink(path)

    def test_stopping_criteria_relaxation(self):
        """Premature stops should relax the momentum threshold."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            engine = PhiEvolutionEngine(state_path=path)
            original_threshold = engine.state.phi_momentum_threshold

            eval_result = EvaluationResult(
                generation=0, accuracy=0.5, avg_tokens=500,
                avg_depth=2, avg_confidence=0.5, improvement=0.0,
                premature_stops=10, late_stops=0,
            )
            engine.evolve(eval_result)
            assert engine.state.phi_momentum_threshold > original_threshold
        finally:
            os.unlink(path)

    def test_stopping_criteria_tightening(self):
        """Late stops should tighten the momentum threshold."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            engine = PhiEvolutionEngine(state_path=path)
            original_threshold = engine.state.phi_momentum_threshold

            eval_result = EvaluationResult(
                generation=0, accuracy=0.5, avg_tokens=500,
                avg_depth=5, avg_confidence=0.5, improvement=0.0,
                premature_stops=0, late_stops=10,
            )
            engine.evolve(eval_result)
            assert engine.state.phi_momentum_threshold < original_threshold
        finally:
            os.unlink(path)


class TestEvolutionLoop:
    """Integration tests for the evolution loop."""

    def test_evolution_loop_runs(self):
        """Verify evolution loop runs without errors."""
        from evolution_loop import run_evolution

        dataset_path = str(Path(__file__).parent.parent / "benchmarks" / "gsm8k_sample.json")
        state_path = "/tmp/test_evo_loop_state.json"

        try:
            evaluations = run_evolution(
                generations=2,
                dataset_path=dataset_path,
                state_path=state_path,
                max_questions=3,
                max_depth=2,
                verbose=False,
            )
            assert len(evaluations) == 2
            for ev in evaluations:
                assert isinstance(ev, EvaluationResult)
        finally:
            if os.path.exists(state_path):
                os.unlink(state_path)
