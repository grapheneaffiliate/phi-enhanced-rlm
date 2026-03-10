#!/usr/bin/env python3
"""
Benchmark runner comparing φ-RLM vs vanilla recursive decomposition.

Runs standardized test sets (GSM8K, ARC) through both φ-enhanced and
vanilla baselines, measuring accuracy, token usage, and recursion efficiency.
"""

import json
import time
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phi_enhanced_rlm import PhiEnhancedRLM, SubCallResult, MockLLMBackend
from src.meta_recursion import MetaRecursiveRLM

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkQuestion:
    """A single benchmark question."""
    id: str
    question: str
    expected_answer: str
    category: str = "general"
    difficulty: str = "medium"


@dataclass
class BenchmarkResult:
    """Result from running a single benchmark question."""
    question_id: str
    answer: str
    expected: str
    correct: bool
    confidence: float
    tokens_used: int
    recursion_depth: int
    time_seconds: float
    stop_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSummary:
    """Summary statistics for a benchmark run."""
    model_name: str
    dataset: str
    total_questions: int
    correct: int
    accuracy: float
    avg_tokens: float
    avg_depth: float
    avg_time: float
    avg_confidence: float
    results: List[BenchmarkResult] = field(default_factory=list)


class VanillaRLM:
    """
    Baseline: same recursion without φ-mathematics.

    Uses uniform budget (no Casimir), no diversity-based chunk selection
    (random selection), fixed depth (no momentum/spectral stopping).
    This isolates the contribution of φ-enhancement.
    """

    def __init__(self, base_llm_callable: Callable, context_chunks: List[str],
                 total_budget_tokens: int = 4096):
        self.llm = base_llm_callable
        self.context_chunks = context_chunks
        self.total_budget = total_budget_tokens

    def recursive_solve(self, query: str, depth: int = 0,
                        max_depth: int = 5) -> SubCallResult:
        """Vanilla recursive solve — uniform budget, no φ-enhancements."""
        import random

        # Uniform budget allocation (no Casimir weighting)
        budget = self.total_budget // (max_depth + 1)

        # Random chunk selection (no diversity optimization)
        n_select = min(3, len(self.context_chunks))
        selected = random.sample(self.context_chunks, n_select)
        selected_text = "\n".join(selected)

        prompt = f"""Query: {query}

Context:
{selected_text}

Recursion depth: {depth}
Remaining budget: {budget} tokens

Respond in JSON format:
{{"answer": "your answer", "confidence": 0.0-1.0, "subquestions": ["...", "..."]}}
"""
        try:
            response_str = self.llm(prompt, max_tokens=budget)
            response = json.loads(response_str)
            answer = response.get("answer", "")
            confidence = response.get("confidence", 0.5)
            subquestions = response.get("subquestions", [])
        except Exception as e:
            answer = f"Error: {e}"
            confidence = 0.3
            subquestions = []

        # Fixed depth stopping — no early stop heuristics
        if depth >= max_depth or not subquestions:
            return SubCallResult(
                value=answer,
                confidence=confidence,
                metadata={"depth": depth, "stop_reason": "depth" if depth >= max_depth else "no_subquestions"}
            )

        # Recurse on subquestions (no torsion correction)
        sub_results = []
        for subq in subquestions[:3]:
            sub_result = self.recursive_solve(subq, depth + 1, max_depth)
            sub_results.append(sub_result)

        # Simple average aggregation (no torsion correction)
        if sub_results:
            sub_answers = [r.value for r in sub_results]
            sub_conf = sum(r.confidence for r in sub_results) / len(sub_results)
            final_answer = f"{answer}\n\nSub-analysis: {'; '.join(str(a) for a in sub_answers)}"
            final_conf = (confidence + sub_conf) / 2
        else:
            final_answer = answer
            final_conf = confidence

        return SubCallResult(
            value=final_answer,
            confidence=final_conf,
            metadata={"depth": depth, "stop_reason": "recursion_complete",
                       "n_subquestions": len(subquestions)}
        )


def load_dataset(path: str) -> List[BenchmarkQuestion]:
    """Load benchmark dataset from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    questions = []
    for item in data:
        questions.append(BenchmarkQuestion(
            id=item.get("id", str(len(questions))),
            question=item["question"],
            expected_answer=item["expected_answer"],
            category=item.get("category", "general"),
            difficulty=item.get("difficulty", "medium"),
        ))
    return questions


def check_answer(result_answer: str, expected: str) -> bool:
    """Check if the answer is correct (fuzzy matching)."""
    result_lower = str(result_answer).lower().strip()
    expected_lower = expected.lower().strip()

    # Exact match
    if expected_lower in result_lower:
        return True

    # Numeric match — extract numbers and compare
    import re
    result_nums = re.findall(r'-?\d+\.?\d*', result_lower)
    expected_nums = re.findall(r'-?\d+\.?\d*', expected_lower)

    if expected_nums and result_nums:
        try:
            for en in expected_nums:
                if any(abs(float(rn) - float(en)) < 0.01 for rn in result_nums):
                    return True
        except ValueError:
            pass

    return False


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (words * 1.3)."""
    return int(len(text.split()) * 1.3)


def run_single(model, question: BenchmarkQuestion, max_depth: int = 5) -> BenchmarkResult:
    """Run a single benchmark question through a model."""
    start = time.time()

    if hasattr(model, 'meta_solve'):
        result = model.meta_solve(question.question, max_meta_depth=2)
    else:
        result = model.recursive_solve(question.question, max_depth=max_depth)

    elapsed = time.time() - start
    depth = result.metadata.get("depth", 0)
    stop_reason = result.metadata.get("stop_reason", "")

    correct = check_answer(str(result.value), question.expected_answer)

    return BenchmarkResult(
        question_id=question.id,
        answer=str(result.value),
        expected=question.expected_answer,
        correct=correct,
        confidence=result.confidence,
        tokens_used=estimate_tokens(str(result.value)),
        recursion_depth=depth,
        time_seconds=elapsed,
        stop_reason=stop_reason,
    )


def run_benchmark(dataset_path: str, llm_backend: Optional[Callable] = None,
                  context_chunks: Optional[List[str]] = None,
                  max_questions: int = 50,
                  max_depth: int = 5) -> Dict[str, BenchmarkSummary]:
    """
    Run full benchmark comparing φ-RLM vs vanilla baseline.

    Args:
        dataset_path: Path to benchmark JSON file
        llm_backend: LLM callable (uses MockLLMBackend if None)
        context_chunks: Context chunks for the models
        max_questions: Maximum questions to run
        max_depth: Maximum recursion depth

    Returns:
        Dict with 'phi' and 'vanilla' BenchmarkSummary results
    """
    questions = load_dataset(dataset_path)[:max_questions]

    if llm_backend is None:
        llm_backend = MockLLMBackend()
    if context_chunks is None:
        context_chunks = [
            "Mathematical reasoning requires systematic decomposition.",
            "Verify each step with independent calculation.",
            "Check boundary conditions and edge cases.",
            "Use dimensional analysis to validate results.",
            "Consider alternative solution approaches.",
        ]

    # Create models
    phi_rlm = PhiEnhancedRLM(
        base_llm_callable=llm_backend,
        context_chunks=context_chunks,
        total_budget_tokens=4096,
    )
    vanilla_rlm = VanillaRLM(
        base_llm_callable=llm_backend,
        context_chunks=context_chunks,
        total_budget_tokens=4096,
    )

    # Create meta-recursive wrapper
    meta_rlm = MetaRecursiveRLM(phi_rlm)

    phi_results = []
    vanilla_results = []
    meta_results = []

    for i, question in enumerate(questions):
        logger.info(f"Question {i+1}/{len(questions)}: {question.question[:60]}...")

        phi_result = run_single(phi_rlm, question, max_depth)
        phi_results.append(phi_result)

        vanilla_result = run_single(vanilla_rlm, question, max_depth)
        vanilla_results.append(vanilla_result)

        meta_result = run_single(meta_rlm, question, max_depth)
        meta_results.append(meta_result)

    def summarize(name: str, results: List[BenchmarkResult]) -> BenchmarkSummary:
        correct = sum(1 for r in results if r.correct)
        return BenchmarkSummary(
            model_name=name,
            dataset=dataset_path,
            total_questions=len(results),
            correct=correct,
            accuracy=correct / len(results) if results else 0,
            avg_tokens=sum(r.tokens_used for r in results) / len(results) if results else 0,
            avg_depth=sum(r.recursion_depth for r in results) / len(results) if results else 0,
            avg_time=sum(r.time_seconds for r in results) / len(results) if results else 0,
            avg_confidence=sum(r.confidence for r in results) / len(results) if results else 0,
            results=results,
        )

    return {
        "phi": summarize("φ-Enhanced RLM", phi_results),
        "vanilla": summarize("Vanilla RLM", vanilla_results),
        "meta": summarize("Meta-φ-RLM", meta_results),
    }


def save_results(summaries: Dict[str, BenchmarkSummary], output_dir: str = "benchmarks/results"):
    """Save benchmark results to JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"benchmark_{timestamp}.json"

    output = {}
    for name, summary in summaries.items():
        output[name] = {
            "model_name": summary.model_name,
            "dataset": summary.dataset,
            "total_questions": summary.total_questions,
            "correct": summary.correct,
            "accuracy": summary.accuracy,
            "avg_tokens": summary.avg_tokens,
            "avg_depth": summary.avg_depth,
            "avg_time": summary.avg_time,
            "avg_confidence": summary.avg_confidence,
            "results": [asdict(r) for r in summary.results],
        }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    return output_path


def print_comparison(summaries: Dict[str, BenchmarkSummary]):
    """Print a formatted comparison table."""
    phi = summaries["phi"]
    vanilla = summaries["vanilla"]
    meta = summaries.get("meta")

    print("\n" + "=" * 85)
    print("BENCHMARK COMPARISON: φ-RLM vs Vanilla vs Meta-φ-RLM")
    print("=" * 85)
    if meta:
        print(f"{'Metric':<20} {'φ-RLM':>15} {'Vanilla':>15} {'Meta-φ':>15} {'Best':>15}")
        print("-" * 85)
        models = [("φ-RLM", phi), ("Vanilla", vanilla), ("Meta-φ", meta)]
    else:
        print(f"{'Metric':<20} {'φ-RLM':>15} {'Vanilla':>15} {'Δ':>10}")
        print("-" * 65)
        models = [("φ-RLM", phi), ("Vanilla", vanilla)]

    def best_label(vals, names, higher_better=True):
        idx = vals.index(max(vals)) if higher_better else vals.index(min(vals))
        return names[idx]

    names = [n for n, _ in models]
    accs = [m.accuracy for _, m in models]
    toks = [m.avg_tokens for _, m in models]
    times = [m.avg_time for _, m in models]
    confs = [m.avg_confidence for _, m in models]

    if meta:
        print(f"{'Accuracy':<20} {phi.accuracy:>14.1%} {vanilla.accuracy:>14.1%} {meta.accuracy:>14.1%} {best_label(accs, names):>15}")
        print(f"{'Avg Tokens':<20} {phi.avg_tokens:>15.0f} {vanilla.avg_tokens:>15.0f} {meta.avg_tokens:>15.0f} {best_label(toks, names, False):>15}")
        print(f"{'Avg Depth':<20} {phi.avg_depth:>15.1f} {vanilla.avg_depth:>15.1f} {meta.avg_depth:>15.1f}")
        print(f"{'Avg Time (s)':<20} {phi.avg_time:>15.2f} {vanilla.avg_time:>15.2f} {meta.avg_time:>15.2f} {best_label(times, names, False):>15}")
        print(f"{'Avg Confidence':<20} {phi.avg_confidence:>15.3f} {vanilla.avg_confidence:>15.3f} {meta.avg_confidence:>15.3f} {best_label(confs, names):>15}")
    else:
        print(f"{'Accuracy':<20} {phi.accuracy:>14.1%} {vanilla.accuracy:>14.1%} {phi.accuracy - vanilla.accuracy:>+9.1%}")
        print(f"{'Avg Tokens':<20} {phi.avg_tokens:>15.0f} {vanilla.avg_tokens:>15.0f} {phi.avg_tokens - vanilla.avg_tokens:>+10.0f}")
        print(f"{'Avg Depth':<20} {phi.avg_depth:>15.1f} {vanilla.avg_depth:>15.1f} {phi.avg_depth - vanilla.avg_depth:>+10.1f}")
        print(f"{'Avg Time (s)':<20} {phi.avg_time:>15.2f} {vanilla.avg_time:>15.2f} {phi.avg_time - vanilla.avg_time:>+10.2f}")
        print(f"{'Avg Confidence':<20} {phi.avg_confidence:>15.3f} {vanilla.avg_confidence:>15.3f} {phi.avg_confidence - vanilla.avg_confidence:>+10.3f}")
    print("=" * 85)

    # Token efficiency
    all_models = [(n, m) for n, m in models if m.accuracy > 0]
    if all_models:
        print("\nToken Efficiency (accuracy/token):")
        for name, m in all_models:
            eff = m.accuracy / max(m.avg_tokens, 1)
            print(f"  {name:<10}: {eff:.6f}")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dataset = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/gsm8k_sample.json"
    max_q = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    print(f"Running benchmark on {dataset} ({max_q} questions)...")
    summaries = run_benchmark(dataset, max_questions=max_q)
    print_comparison(summaries)
    save_results(summaries)
