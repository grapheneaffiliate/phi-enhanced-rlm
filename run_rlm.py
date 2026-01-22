#!/usr/bin/env python3
"""
RUN PHI-ENHANCED RLM WITH OPEN ROUTER
======================================
Simple script to run the RLM with real LLM backend.

Usage:
    python run_rlm.py                           # Default query
    python run_rlm.py "Your query here"         # Query from command line
    python run_rlm.py --file query.txt          # Query from file (avoids cmd length limits)
    python run_rlm.py -f query.txt              # Short form
"""

import sys
import json
import argparse
from openrouter_backend import OpenRouterBackend
from phi_enhanced_rlm import PhiEnhancedRLM

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="PHI-Enhanced RLM with Open Router",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_rlm.py "What is E8 symmetry?"
    python run_rlm.py --file long_query.txt
    python run_rlm.py -f query.txt --depth 5
    echo "My query" | python run_rlm.py --stdin
        """
    )
    parser.add_argument("query", nargs="*", help="Query text (optional if using --file)")
    parser.add_argument("-f", "--file", type=str, help="Read query from file")
    parser.add_argument("--stdin", action="store_true", help="Read query from stdin")
    parser.add_argument("--depth", type=int, default=3, help="Maximum recursion depth (default: 3)")
    parser.add_argument("--context", type=str, help="Read context chunks from JSON file")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    # Determine query source
    query = None
    
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8-sig") as f:
                query = f.read().strip()
            # Normalize whitespace (Windows echo can create weird line breaks)
            query = " ".join(query.split())
            if not args.quiet:
                print(f"✓ Loaded query from file: {args.file} ({len(query)} chars)")
        except FileNotFoundError:
            print(f"✗ File not found: {args.file}")
            return
        except Exception as e:
            print(f"✗ Error reading file: {e}")
            return
    elif args.stdin:
        query = sys.stdin.read().strip()
        if not args.quiet:
            print(f"✓ Read query from stdin ({len(query)} chars)")
    elif args.query:
        query = " ".join(args.query)
    else:
        query = "Explain how the golden ratio relates to E8 symmetry and recursive reasoning in AI systems."
    
    if not args.quiet:
        print("=" * 70)
        print("PHI-ENHANCED RLM WITH OPEN ROUTER")
        print("=" * 70)
        print()
    
    # Initialize the backend (reads from .env)
    if not args.quiet:
        print("Initializing Open Router backend...")
    try:
        backend = OpenRouterBackend()
        if not args.quiet:
            print(f"✓ Backend ready with model: {backend.config.model}")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        print("\nMake sure your .env file contains OPENROUTER_API_KEY")
        return
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nInstall with: pip install openai python-dotenv")
        return
    
    if not args.quiet:
        print()
    
    # Load context chunks
    if args.context:
        try:
            with open(args.context, "r", encoding="utf-8") as f:
                context_chunks = json.load(f)
            if not args.quiet:
                print(f"✓ Loaded {len(context_chunks)} context chunks from {args.context}")
        except Exception as e:
            print(f"✗ Error loading context: {e}")
            return
    else:
        # Default context chunks
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
    if not args.quiet:
        print("Initializing PHI-Enhanced RLM...")
    rlm = PhiEnhancedRLM(
        base_llm_callable=backend,
        context_chunks=context_chunks,
        total_budget_tokens=4096,
        trace_file="rlm_trace.jsonl"
    )
    if not args.quiet:
        print(f"✓ RLM ready with {len(context_chunks)} context chunks")
        print(f"✓ Budget allocation: {rlm.budget_map}")
        print()
    
    # Show query (truncated if long)
    if not args.quiet:
        display_query = query[:200] + "..." if len(query) > 200 else query
        print(f"Query: {display_query}")
        print("-" * 70)
        print()
        print("Running recursive reasoning (this may take a moment)...")
        print()
    
    # Run recursive solve
    result = rlm.recursive_solve(query, max_depth=args.depth)
    
    # Display results
    if args.quiet:
        print(result.value)
    else:
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
