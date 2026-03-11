#!/usr/bin/env python3
"""
Streaming recursive solve — yield intermediate reasoning events.

Provides a generator-based interface that yields ReasoningEvent objects
as recursion progresses, enabling real-time observation of the reasoning
tree being built.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Generator


@dataclass
class ReasoningEvent:
    """An intermediate event during recursive reasoning."""
    type: str  # "chunk_selection", "llm_response", "qec_verification",
               # "subquestion_start", "subquestion_complete", "complete"
    depth: int
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        return f"data: {json.dumps(self.to_dict())}\n\n"


def recursive_solve_stream(rlm, query: str, depth: int = 0,
                           path: tuple = (),
                           max_depth: int = 5) -> Generator[ReasoningEvent, None, Any]:
    """
    Streaming version of recursive_solve.

    Yields ReasoningEvent objects at each step of the recursion,
    then returns the final SubCallResult.

    Args:
        rlm: PhiEnhancedRLM instance
        query: The question to answer
        depth: Current recursion depth
        path: Tuple tracking recursion path
        max_depth: Maximum recursion depth

    Yields:
        ReasoningEvent objects for each step

    Returns:
        SubCallResult (accessible via generator.value after StopIteration)
    """
    from .phi_enhanced_rlm import SubCallResult, simple_tokenize

    # Get budget for this depth
    budget = rlm.get_budget_for_depth(depth)

    # Step 1: Select chunks
    selected = rlm.select_chunks_for_subcall(query=query, max_chunks=3)
    selected_ids = [c.id for c in selected]
    selected_text = "\n".join([c.text for c in selected])

    sub_gram = rlm.phi_gram.submatrix(selected_ids)
    logdet_selected = sub_gram.log_determinant

    yield ReasoningEvent(
        type="chunk_selection",
        depth=depth,
        data={
            "selected_ids": selected_ids,
            "logdet": round(logdet_selected, 4),
            "query": query[:100],
        }
    )

    # Step 2: Build prompt
    phi_attn = rlm._get_phi_attention()
    use_phi = (phi_attn is not None and
               (rlm.evolution_state is None or
                getattr(rlm.evolution_state, 'phi_attention_enabled', True)))

    if use_phi:
        prompt = phi_attn.build_phi_prompt(query, selected_text, depth, budget)
    else:
        prompt = f"""Query: {query}

Context:
{selected_text}

Recursion depth: {depth}
Remaining budget: {budget} tokens

Respond in JSON format:
{{"answer": "your answer", "confidence": 0.0-1.0, "subquestions": ["...", "..."]}}
"""

    # Step 3: Call LLM
    try:
        response_str = rlm.llm(prompt, max_tokens=budget)
        response = json.loads(response_str)
        answer = response.get("answer", "")
        raw_confidence = response.get("confidence", 0.5)
        subquestions = response.get("subquestions", [])
    except Exception as e:
        answer = f"Error: {e}"
        raw_confidence = 0.3
        subquestions = []

    yield ReasoningEvent(
        type="llm_response",
        depth=depth,
        data={
            "answer_preview": answer[:200],
            "raw_confidence": round(raw_confidence, 4),
            "n_subquestions": len(subquestions),
        }
    )

    # Step 4: QEC verification
    revised_conf, verifier_results = rlm.run_qec_verification(
        answer, selected_text, budget // 2
    )
    confidence = (raw_confidence + revised_conf) / 2

    yield ReasoningEvent(
        type="qec_verification",
        depth=depth,
        data={
            "raw_confidence": round(raw_confidence, 4),
            "revised_confidence": round(revised_conf, 4),
            "final_confidence": round(confidence, 4),
            "n_verifiers": len(verifier_results),
        }
    )

    # Update state
    rlm.confidence_history.append(confidence)
    answer_tokens = simple_tokenize(answer)
    new_info = rlm.compute_info_units(answer_tokens)
    rlm.update_information_state(new_info)

    # Determine stop reason
    stop_reason = "none"
    if depth >= max_depth:
        stop_reason = "depth"
    elif rlm.should_verify_early_stop():
        stop_reason = "momentum"
    elif not rlm.should_continue_recursion():
        stop_reason = "spectral"

    rlm.log_trace(depth, query, selected_ids, logdet_selected,
                  confidence, new_info, stop_reason)

    if stop_reason != "none" or not subquestions:
        result = SubCallResult(
            value=answer,
            confidence=confidence,
            metadata={
                "depth": depth,
                "path": path,
                "stop_reason": stop_reason if stop_reason != "none" else "no_subquestions",
                "selected_ids": selected_ids,
            }
        )
        yield ReasoningEvent(
            type="complete",
            depth=depth,
            data={
                "answer_preview": answer[:200],
                "confidence": round(confidence, 4),
                "stop_reason": result.metadata["stop_reason"],
            }
        )
        return result

    # Apply sparse pruning
    phi_sparse = rlm._get_phi_sparse()
    use_sparse = (phi_sparse is not None and
                  (rlm.evolution_state is None or
                   getattr(rlm.evolution_state, 'sparse_pruning_enabled', True)))

    if use_sparse and len(subquestions) > 1:
        scores = phi_sparse.score_subquestions(query, subquestions, answer)
        subquestions = phi_sparse.adaptive_prune(subquestions, scores, confidence)

    branch_factor = 3
    if rlm.evolution_state and hasattr(rlm.evolution_state, 'branch_factor'):
        branch_factor = rlm.evolution_state.branch_factor
    subquestions_limited = subquestions[:branch_factor]

    # Recurse on subquestions
    sub_results = []
    for i, subq in enumerate(subquestions_limited):
        yield ReasoningEvent(
            type="subquestion_start",
            depth=depth,
            data={"index": i, "subquestion": subq[:100]}
        )

        # Recursively stream sub-results
        sub_gen = recursive_solve_stream(rlm, subq, depth + 1, path + (i,), max_depth)
        sub_result = None
        try:
            while True:
                event = next(sub_gen)
                yield event
        except StopIteration as e:
            sub_result = e.value

        if sub_result is not None:
            sub_results.append(sub_result)

        yield ReasoningEvent(
            type="subquestion_complete",
            depth=depth,
            data={
                "index": i,
                "confidence": round(sub_result.confidence, 4) if sub_result else 0,
            }
        )

    # Aggregate
    aggregated = rlm.aggregate_results(sub_results)
    final_answer = f"{answer}\n\nSub-analysis: {aggregated.value}"
    final_conf = (confidence + aggregated.confidence) / 2

    result = SubCallResult(
        value=final_answer,
        confidence=final_conf,
        metadata={
            "depth": depth,
            "path": path,
            "stop_reason": "recursion_complete",
            "n_subquestions": len(subquestions),
            "aggregated_confidence": aggregated.confidence,
        }
    )

    yield ReasoningEvent(
        type="complete",
        depth=depth,
        data={
            "answer_preview": final_answer[:200],
            "confidence": round(final_conf, 4),
            "stop_reason": "recursion_complete",
            "n_subquestions": len(subquestions_limited),
        }
    )

    return result
