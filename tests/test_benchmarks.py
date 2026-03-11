#!/usr/bin/env python3
"""Tests for benchmark infrastructure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

from runner import (
    VanillaRLM,
    check_answer,
    estimate_tokens,
    load_dataset,
    run_benchmark,
)

from src.phi_enhanced_rlm import MockLLMBackend, SubCallResult


class TestBenchmarkDatasets:
    """Test that benchmark datasets are valid."""

    def test_gsm8k_loads(self):
        path = str(Path(__file__).parent.parent / "benchmarks" / "gsm8k_sample.json")
        questions = load_dataset(path)
        assert len(questions) == 50
        for q in questions:
            assert q.question
            assert q.expected_answer
            assert q.id

    def test_arc_loads(self):
        path = str(Path(__file__).parent.parent / "benchmarks" / "arc_sample.json")
        questions = load_dataset(path)
        assert len(questions) == 25
        for q in questions:
            assert q.question
            assert q.expected_answer


class TestAnswerChecking:
    """Test the fuzzy answer matching."""

    def test_exact_match(self):
        assert check_answer("The answer is 42", "42")

    def test_numeric_match(self):
        assert check_answer("The result is 3.14", "3.14")

    def test_substring_match(self):
        assert check_answer("Photosynthesis is the process", "photosynthesis")

    def test_no_match(self):
        assert not check_answer("The sky is blue", "red")

    def test_numeric_tolerance(self):
        assert check_answer("Approximately 3.14159", "3.14")


class TestVanillaRLM:
    """Test the vanilla (non-φ) baseline."""

    def test_vanilla_solves(self):
        mock = MockLLMBackend()
        context = ["Chunk 1", "Chunk 2", "Chunk 3"]
        vanilla = VanillaRLM(mock, context)
        result = vanilla.recursive_solve("What is 2+2?", max_depth=2)
        assert isinstance(result, SubCallResult)
        assert result.value is not None
        assert result.confidence > 0

    def test_vanilla_respects_depth(self):
        mock = MockLLMBackend()
        context = ["Chunk 1", "Chunk 2"]
        vanilla = VanillaRLM(mock, context)
        result = vanilla.recursive_solve("Test", max_depth=1)
        assert result.metadata.get("depth", 0) <= 1


class TestBenchmarkRunner:
    """Test the full benchmark pipeline."""

    def test_run_benchmark_small(self):
        path = str(Path(__file__).parent.parent / "benchmarks" / "gsm8k_sample.json")
        summaries = run_benchmark(path, max_questions=3)
        assert "phi" in summaries
        assert "vanilla" in summaries
        assert summaries["phi"].total_questions == 3
        assert summaries["vanilla"].total_questions == 3

    def test_token_estimation(self):
        assert estimate_tokens("hello world") > 0
        assert estimate_tokens("a b c d e") == int(5 * 1.3)
