#!/usr/bin/env python3
"""
Main evolution loop — run this to let the model self-improve.

Orchestrates the cycle of:
1. Create RLM with current evolved state
2. Run benchmark
3. Evaluate performance
4. Evolve parameters
5. Repeat

Usage:
    python -m src.evolution_loop --generations 10 --dataset benchmarks/gsm8k_sample.json
"""

import json
import logging
import time
from typing import List, Dict


from evolution import PhiEvolutionEngine, EvolutionState, EvaluationResult
from phi_enhanced_rlm import PhiEnhancedRLM, MockLLMBackend
from phi_separation_novel_mathematics import PHI

logger = logging.getLogger(__name__)


def create_rlm_with_evolution(evolution_state: EvolutionState,
                               llm_backend=None,
                               context_chunks=None,
                               total_budget: int = 4096) -> PhiEnhancedRLM:
    """
    Create a PhiEnhancedRLM configured with evolved parameters.

    Args:
        evolution_state: Current evolution state with learned parameters
        llm_backend: LLM callable (uses MockLLMBackend if None)
        context_chunks: Context chunks for the model
        total_budget: Total token budget
    """
    if llm_backend is None:
        llm_backend = MockLLMBackend()
    if context_chunks is None:
        context_chunks = [
            "Mathematical reasoning requires systematic decomposition of problems.",
            "Verify each computational step with independent calculation.",
            "Check boundary conditions, edge cases, and special values.",
            "Use dimensional analysis and unit checking to validate results.",
            "Consider alternative solution approaches for verification.",
        ]

    rlm = PhiEnhancedRLM(
        base_llm_callable=llm_backend,
        context_chunks=context_chunks,
        total_budget_tokens=total_budget,
        evolution_state=evolution_state,
    )
    return rlm


def load_benchmark(dataset_path: str) -> List[Dict]:
    """Load benchmark questions from JSON file."""
    with open(dataset_path, "r") as f:
        return json.load(f)


def run_generation(rlm: PhiEnhancedRLM, questions: List[Dict],
                   max_depth: int = 5) -> List[Dict]:
    """
    Run all benchmark questions through the RLM.

    Returns list of trace dicts with answers and metadata.
    """
    traces = []
    for i, q in enumerate(questions):
        query = q["question"]
        expected = q.get("expected_answer", "")

        try:
            result = rlm.recursive_solve(query, max_depth=max_depth)
            traces.append({
                "question_id": q.get("id", str(i)),
                "answer": str(result.value),
                "expected": expected,
                "confidence": result.confidence,
                "depth": result.metadata.get("depth", 0),
                "stop_reason": result.metadata.get("stop_reason", ""),
            })
        except Exception as e:
            logger.error(f"Error on question {i}: {e}")
            traces.append({
                "question_id": q.get("id", str(i)),
                "answer": f"Error: {e}",
                "expected": expected,
                "confidence": 0.0,
                "depth": 0,
                "stop_reason": "error",
            })

    return traces


def run_evolution(generations: int = 10,
                  dataset_path: str = "benchmarks/gsm8k_sample.json",
                  state_path: str = "evolution_state.json",
                  max_questions: int = 20,
                  max_depth: int = 5,
                  llm_backend=None,
                  context_chunks=None,
                  verbose: bool = True) -> List[EvaluationResult]:
    """
    Main evolution loop — let the model self-improve.

    Args:
        generations: Number of evolution generations to run
        dataset_path: Path to benchmark dataset
        state_path: Path to evolution state file
        max_questions: Maximum questions per generation
        max_depth: Maximum recursion depth
        llm_backend: LLM callable (uses MockLLMBackend if None)
        context_chunks: Context chunks
        verbose: Print progress

    Returns:
        List of EvaluationResult for each generation
    """
    engine = PhiEvolutionEngine(state_path=state_path)
    questions = load_benchmark(dataset_path)[:max_questions]
    ground_truth = [q.get("expected_answer", "") for q in questions]
    evaluations = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"φ-EVOLUTION LOOP: {generations} generations")
        print(f"Dataset: {dataset_path} ({len(questions)} questions)")
        print(f"Starting generation: {engine.state.generation}")
        print(f"{'='*60}\n")

    for gen in range(generations):
        gen_start = time.time()

        # Create RLM with current evolved state
        rlm = create_rlm_with_evolution(
            engine.state,
            llm_backend=llm_backend,
            context_chunks=context_chunks,
        )

        # Run benchmark
        traces = run_generation(rlm, questions, max_depth=max_depth)

        # Evaluate
        evaluation = engine.evaluate_generation(traces, ground_truth)
        evaluations.append(evaluation)

        # Evolve
        engine.evolve(evaluation)

        gen_time = time.time() - gen_start

        if verbose:
            arrow = "↑" if evaluation.improvement > 0 else "↓" if evaluation.improvement < 0 else "→"
            print(
                f"Gen {engine.state.generation - 1:3d}: "
                f"accuracy={evaluation.accuracy:.1%} {arrow} "
                f"tokens={evaluation.avg_tokens:.0f} "
                f"depth={evaluation.avg_depth:.1f} "
                f"Δ={evaluation.improvement:+.3f} "
                f"lr={engine.learning_rate() * PHI:.4f} "  # Next gen's lr
                f"({gen_time:.1f}s)"
            )

    if verbose:
        print(f"\n{'='*60}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*60}")
        summary = engine.get_evolution_summary()
        print(f"Final generation: {summary['generation']}")
        print(f"Final fitness: {summary['fitness']:.3f}")
        print(f"Best strategy: {summary['best_strategy']}")
        print(f"Budget weights: {[f'{w:.2f}' for w in summary['budget_weights']]}")
        if summary['accuracy_trend']:
            print(f"Accuracy trend: {' → '.join(f'{a:.1%}' for a in summary['accuracy_trend'])}")
        print()

    return evaluations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="φ-Evolution Loop")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="benchmarks/gsm8k_sample.json")
    parser.add_argument("--state", type=str, default="evolution_state.json")
    parser.add_argument("--max-questions", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_evolution(
        generations=args.generations,
        dataset_path=args.dataset,
        state_path=args.state,
        max_questions=args.max_questions,
        max_depth=args.max_depth,
    )
