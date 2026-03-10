#!/usr/bin/env python3
"""
Integration tests for real LLM backend.

Tests that the OpenRouter integration works end-to-end:
- API connection
- Response parsing
- JSON format compliance
- Integration with PhiEnhancedRLM

Set OPENROUTER_API_KEY in .env to run these tests.
Skip with: pytest -m "not integration"
"""

import json
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phi_enhanced_rlm import PhiEnhancedRLM, MockLLMBackend, SubCallResult


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

HAS_API_KEY = bool(os.getenv("OPENROUTER_API_KEY"))


class TestRealLLMIntegration:
    """Tests requiring a real LLM API key."""

    @pytest.mark.skipif(not HAS_API_KEY, reason="OPENROUTER_API_KEY not set")
    def test_openrouter_backend_connection(self):
        """Test that OpenRouter backend can make a real API call."""
        from openrouter_backend import OpenRouterBackend

        backend = OpenRouterBackend()
        response = backend("What is 2 + 2? Respond in JSON: {\"answer\": \"...\", \"confidence\": 0.0-1.0, \"subquestions\": []}", max_tokens=100)

        # Should return valid JSON
        parsed = json.loads(response)
        assert "answer" in parsed
        assert "confidence" in parsed

    @pytest.mark.skipif(not HAS_API_KEY, reason="OPENROUTER_API_KEY not set")
    def test_real_llm_recursive_solve(self):
        """Test full recursive_solve with real LLM."""
        from openrouter_backend import OpenRouterBackend

        backend = OpenRouterBackend()
        context = [
            "Mathematics involves systematic reasoning.",
            "Addition combines quantities together.",
            "Verification confirms correctness.",
        ]
        rlm = PhiEnhancedRLM(
            base_llm_callable=backend,
            context_chunks=context,
            total_budget_tokens=2048,
        )
        result = rlm.recursive_solve("What is 15 + 27?", max_depth=2)

        assert isinstance(result, SubCallResult)
        assert result.value is not None
        assert 0 <= result.confidence <= 1
        assert "depth" in result.metadata


class TestMockLLMValidation:
    """Tests that validate the mock LLM produces correct format."""

    def test_mock_returns_valid_json(self):
        """Mock backend should return parseable JSON."""
        mock = MockLLMBackend()
        response = mock("Test prompt", max_tokens=100)
        parsed = json.loads(response)
        assert "answer" in parsed
        assert "confidence" in parsed
        assert "subquestions" in parsed

    def test_mock_confidence_range(self):
        """Mock confidence should be in [0, 1]."""
        mock = MockLLMBackend()
        for _ in range(20):
            response = mock("Test", max_tokens=100)
            parsed = json.loads(response)
            assert 0 <= parsed["confidence"] <= 1

    def test_mock_subquestions_are_strings(self):
        """Mock subquestions should be list of strings."""
        mock = MockLLMBackend()
        response = mock("Test", max_tokens=100)
        parsed = json.loads(response)
        assert isinstance(parsed["subquestions"], list)
        for sq in parsed["subquestions"]:
            assert isinstance(sq, str)

    def test_rlm_with_mock_end_to_end(self):
        """Full RLM pipeline with mock backend."""
        mock = MockLLMBackend()
        context = ["Context chunk 1", "Context chunk 2", "Context chunk 3"]
        rlm = PhiEnhancedRLM(
            base_llm_callable=mock,
            context_chunks=context,
            total_budget_tokens=2048,
        )
        result = rlm.recursive_solve("What is the meaning of life?", max_depth=3)

        assert isinstance(result, SubCallResult)
        assert result.value is not None
        assert result.confidence > 0
        assert "depth" in result.metadata
        assert "stop_reason" in result.metadata


class TestBenchmarkIntegration:
    """Tests for benchmark runner integration."""

    def test_benchmark_runner_with_mock(self):
        """Benchmark runner should work with mock backend."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
        from runner import run_benchmark, load_dataset, BenchmarkQuestion

        dataset_path = str(Path(__file__).parent.parent / "benchmarks" / "gsm8k_sample.json")
        summaries = run_benchmark(dataset_path, max_questions=5)

        assert "phi" in summaries
        assert "vanilla" in summaries
        assert summaries["phi"].total_questions == 5
        assert summaries["vanilla"].total_questions == 5
        assert 0 <= summaries["phi"].accuracy <= 1

    def test_load_dataset(self):
        """Dataset loader should parse JSON correctly."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
        from runner import load_dataset

        dataset_path = str(Path(__file__).parent.parent / "benchmarks" / "gsm8k_sample.json")
        questions = load_dataset(dataset_path)

        assert len(questions) == 50
        assert questions[0].id == "gsm8k_001"
        assert questions[0].expected_answer == "18"
