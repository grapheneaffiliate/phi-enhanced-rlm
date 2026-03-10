#!/usr/bin/env python3
"""
phi-Enhanced RLM -- Quickstart
===============================
Run this to see the system work in under 30 seconds.
No API key needed -- uses demo mode.

Usage:
    python quickstart.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.phi_enhanced_rlm import PhiEnhancedRLM, MockLLMBackend
from src.meta_recursion import MetaRecursiveRLM
from src.evolution import EvolutionState
from src.agent_router import AgentRouter, DEPTH_AGENT_MAP
from src.phi_separation_novel_mathematics import (
    PHI, PHI_INV, CASIMIR_DEGREES, EPSILON,
)


def main():
    print()
    print("=" * 60)
    print("  phi-Enhanced Recursive Language Model -- Quickstart")
    print("  Version 4.0.0")
    print("=" * 60)
    print()

    # -- 1. Show the math --
    print("[1/6] Mathematical Constants")
    print(f"   Golden ratio (phi):       {PHI:.10f}")
    print(f"   Inverse (phi^-1):         {PHI_INV:.10f}")
    print(f"   E8 Casimir degrees:       {[int(c) for c in CASIMIR_DEGREES]}")
    print(f"   Torsion coefficient:      28/248 = {EPSILON:.6f}")
    print()

    # -- 2. Agent router --
    print("[2/6] Agent Router (8 specialized agents)")
    router = AgentRouter()
    for depth in range(8):
        agent = router.get_agent_name_for_depth(depth)
        casimir = int(CASIMIR_DEGREES[depth])
        print(f"   Depth {depth} (C={casimir:2d}): {agent}")
    print()

    # -- 3. Basic recursive solve --
    print("[3/6] Recursive Solve (demo mode)")
    print("-" * 60)

    context = [
        "The golden ratio phi = 1.618 appears throughout mathematics and nature.",
        "E8 is the largest exceptional Lie group with 248 dimensions.",
        "Recursive Language Models decompose complex queries into sub-tasks.",
        "Information theory quantifies the uncertainty in random variables.",
        "The Casimir effect is a quantum phenomenon from vacuum fluctuations.",
    ]

    mock_llm = MockLLMBackend(seed=42)
    rlm = PhiEnhancedRLM(
        base_llm_callable=mock_llm,
        context_chunks=context,
        total_budget_tokens=2048,
    )

    query = "How does the golden ratio relate to E8 symmetry and recursive reasoning?"
    result = rlm.recursive_solve(query, max_depth=3)

    print(f"   Query:       {query}")
    print(f"   Confidence:  {result.confidence:.1%}")
    print(f"   Stop reason: {result.metadata.get('stop_reason', 'N/A')}")
    print(f"   Depth:       {result.metadata.get('depth', 0)}")
    print()

    # -- 4. Reasoning tree --
    print("[4/6] Reasoning Tree")
    print("-" * 60)
    tree = rlm.get_reasoning_tree()
    for node in tree.get("nodes", []):
        indent = "   " + "  " * node["depth"]
        conf = node["confidence"]
        icon = "+" if conf >= 0.6 else "~" if conf >= 0.4 else "-"
        label = node["query"][:45]
        print(f"{indent}[{icon}] D{node['depth']}: {label}... ({conf:.0%})")
    print()

    # -- 5. Budget allocation --
    print("[5/6] E8 Casimir Budget Allocation")
    print("-" * 60)
    total = sum(rlm.budget_map.values())
    for depth, tokens in rlm.budget_map.items():
        bar_len = int(tokens / total * 40) if total > 0 else 0
        bar = "#" * bar_len + "." * (40 - bar_len)
        casimir = int(CASIMIR_DEGREES[depth])
        print(f"   Depth {depth} (C={casimir:2d}): [{bar}] {tokens} tokens")
    print()

    # -- 6. Meta-recursion --
    print("[6/6] Meta-Recursion (strategy selection)")
    print("-" * 60)

    # Reset the RLM for a fresh solve
    rlm2 = PhiEnhancedRLM(
        base_llm_callable=mock_llm,
        context_chunks=context,
        total_budget_tokens=2048,
    )
    meta = MetaRecursiveRLM(rlm2)
    meta_result = meta.meta_solve(
        "Research the theoretical implications of phi-separation for cryptography."
    )
    print(f"   Query type:  {meta_result.metadata.get('meta_query_type')}")
    print(f"   Strategy:    {meta_result.metadata.get('meta_strategy')}")
    print(f"   Confidence:  {meta_result.confidence:.1%}")
    print()

    # -- Evolution preview --
    print("   Evolution learning rate schedule:")
    for gen in range(6):
        lr = PHI_INV ** gen
        bar_len = int(lr * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        print(f"     Gen {gen}: [{bar}] lr={lr:.4f}")
    print(f"     -> Rate = phi^(-n), diminishes but never reaches zero")
    print()

    # -- Done --
    print("=" * 60)
    print("  Quickstart complete!")
    print("=" * 60)
    print()
    print("  Next steps:")
    print("    python -m src.evolution_loop --generations 10  # Self-evolve")
    print("    python -m pytest tests/ -v                     # Run tests")
    print("    python cli/chat.py                             # Chat mode")
    print("    See README.md for the full guide")
    print()


if __name__ == "__main__":
    main()
