#!/usr/bin/env python3
"""
RUN PHI-ENHANCED RLM WITH OPEN ROUTER
======================================
Simple script to run the RLM with real LLM backend.

Usage:
    python run_rlm.py
    python run_rlm.py "Your custom query here"
"""

import sys
import json
from openrouter_backend import OpenRouterBackend
from phi_enhanced_rlm import PhiEnhancedRLM

def main():
    print("=" * 70)
    print("PHI-ENHANCED RLM WITH OPEN ROUTER (z-ai/glm-4.7)")
    print("=" * 70)
    print()
    
    # Initialize the backend (reads from .env)
    print("Initializing Open Router backend...")
    try:
        backend = OpenRouterBackend()
        print(f"✓ Backend ready with model: {backend.config.model}")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        print("\nMake sure your .env file contains OPENROUTER_API_KEY")
        return
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nInstall with: pip install openai python-dotenv")
        return
    
    print()
    
    # Define context chunks (knowledge base)
    context_chunks = [
        "The golden ratio φ = (1 + √5)/2 ≈ 1.618 appears throughout mathematics, nature, and art. It has unique algebraic properties: φ² = φ + 1 and 1/φ = φ - 1.",
        "E8 is the largest exceptional Lie group with 248 dimensions and 240 roots. Its structure encodes deep connections between geometry, physics, and number theory.",
        "Recursive Language Models (RLMs) decompose complex queries into sub-tasks, enabling LLMs to process inputs far beyond standard context windows through recursive self-calls.",
        "Information theory, developed by Claude Shannon, quantifies uncertainty in random variables. Entropy H(X) = -Σ p(x) log p(x) measures information content.",
        "The Casimir effect is a quantum phenomenon arising from vacuum fluctuations between conducting plates. E8 Casimir degrees are [2, 8, 12, 14, 18, 20, 24, 30].",
        "Machine learning models benefit from hierarchical feature extraction. Deep networks learn increasingly abstract representations at each layer.",
        "Spectral graph theory connects eigenvalues of graph matrices to structural properties like connectivity, clustering, and random walk behavior.",
        "Quantum error correction protects quantum information from decoherence. The φ-threshold for fault-tolerant computation is approximately 0.191.",
    ]
    
    # Initialize RLM with real backend
    print("Initializing PHI-Enhanced RLM...")
    rlm = PhiEnhancedRLM(
        base_llm_callable=backend,
        context_chunks=context_chunks,
        total_budget_tokens=4096,
        trace_file="rlm_trace.jsonl"
    )
    print(f"✓ RLM ready with {len(context_chunks)} context chunks")
    print(f"✓ Budget allocation: {rlm.budget_map}")
    print()
    
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Explain how the golden ratio relates to E8 symmetry and recursive reasoning in AI systems."
    
    print(f"Query: {query}")
    print("-" * 70)
    print()
    
    # Run recursive solve
    print("Running recursive reasoning (this may take a moment)...")
    print()
    
    result = rlm.recursive_solve(query, max_depth=3)
    
    # Display results
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print()
    print(f"Answer:\n{result.value}")
    print()
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Stop Reason: {result.metadata.get('stop_reason', 'N/A')}")
    print(f"Depth Reached: {result.metadata.get('depth', 0)}")
    print()
    
    # Show trace summary
    print("=" * 70)
    print("RECURSION TRACE")
    print("=" * 70)
    try:
        with open("rlm_trace.jsonl", "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    print(f"  Depth {entry['depth']}: conf={entry['confidence']:.3f}, "
                          f"flow={entry['info_flow']:.2f}, stop={entry['stop_reason']}, "
                          f"chunks={entry['selected_ids']}")
    except FileNotFoundError:
        print("  (trace file not found)")
    print()
    
    # Backend stats
    print("=" * 70)
    print("BACKEND STATS")
    print("=" * 70)
    stats = backend.get_stats()
    print(f"  Total API calls: {stats['call_count']}")
    print(f"  Total tokens used: {stats['total_tokens']}")
    print(f"  Model: {stats['model']}")
    print()
    
    print("✓ Complete!")


if __name__ == "__main__":
    main()
