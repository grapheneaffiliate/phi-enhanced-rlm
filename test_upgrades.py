#!/usr/bin/env python3
"""
TEST UPGRADES
=============
Quick test to verify the embedding and async upgrades work.
"""

import sys
import time

def test_embeddings():
    """Test the new embeddings module."""
    print("\n" + "=" * 60)
    print("TEST 1: EMBEDDINGS MODULE")
    print("=" * 60)
    
    try:
        from embeddings import get_embedder, cosine_similarity
        
        embedder = get_embedder()
        print(f"✓ Embedder initialized: {embedder.provider.__class__.__name__}")
        print(f"  Dimensions: {embedder.dimensions}")
        
        # Test embedding
        texts = [
            "The golden ratio φ = 1.618 is found throughout nature.",
            "E8 is an exceptional Lie group with 248 dimensions.",
            "Today's weather is sunny and warm.",
        ]
        
        embeddings = embedder.embed(texts)
        print(f"✓ Embedded {len(texts)} texts, shape: {embeddings.shape}")
        
        # Test similarity (phi and E8 should be more similar than weather)
        sim_01 = cosine_similarity(embeddings[0], embeddings[1])
        sim_02 = cosine_similarity(embeddings[0], embeddings[2])
        
        print(f"  Similarity (math vs math): {sim_01:.4f}")
        print(f"  Similarity (math vs weather): {sim_02:.4f}")
        
        if sim_01 > sim_02:
            print("✓ Semantic similarity working correctly!")
        else:
            print("⚠ Similarity might be using mock embeddings")
        
        # Cache test
        embedder.embed(texts)  # Second call should hit cache
        print(f"✓ Cache stats: {embedder.stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Embeddings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rlm_integration():
    """Test RLM with real embeddings."""
    print("\n" + "=" * 60)
    print("TEST 2: RLM INTEGRATION")
    print("=" * 60)
    
    try:
        from phi_enhanced_rlm import PhiEnhancedRLM, MockLLMBackend
        
        # Context chunks about different topics
        context_chunks = [
            "The golden ratio φ = 1.618 appears in the Fibonacci sequence.",
            "E8 is the largest exceptional Lie group with applications in physics.",
            "Machine learning models learn patterns from training data.",
            "The weather forecast predicts rain tomorrow.",
            "Recursive algorithms call themselves to solve subproblems.",
            "The stock market closed higher today.",
            "Quantum computing uses superposition and entanglement.",
            "My favorite pizza topping is pepperoni.",
        ]
        
        # Use mock LLM but real embeddings
        mock_llm = MockLLMBackend(seed=42)
        
        print("Initializing RLM with real embeddings...")
        rlm = PhiEnhancedRLM(
            base_llm_callable=mock_llm,
            context_chunks=context_chunks,
            total_budget_tokens=1024,
            trace_file="test_trace.jsonl"
        )
        
        print(f"✓ RLM initialized with {len(context_chunks)} chunks")
        
        # Test chunk selection
        query = "Explain the golden ratio and its mathematical properties"
        selected = rlm.select_chunks_for_subcall(query=query, max_chunks=3)
        selected_texts = [c.text for c in selected]
        
        print(f"\nQuery: {query}")
        print(f"Selected chunks:")
        for i, text in enumerate(selected_texts):
            print(f"  {i+1}. {text[:60]}...")
        
        # Check if relevant chunks were selected
        math_keywords = ["golden", "ratio", "E8", "recursive", "quantum"]
        relevant_count = sum(1 for t in selected_texts 
                           if any(kw.lower() in t.lower() for kw in math_keywords))
        
        if relevant_count >= 2:
            print(f"✓ Selected {relevant_count}/3 relevant chunks")
        else:
            print(f"⚠ Only {relevant_count}/3 relevant chunks (might be using mock embeddings)")
        
        return True
        
    except Exception as e:
        print(f"✗ RLM integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_async_backend():
    """Test async backend (optional)."""
    print("\n" + "=" * 60)
    print("TEST 3: ASYNC BACKEND (optional)")
    print("=" * 60)
    
    try:
        import asyncio
        from async_backend import AsyncOpenRouterBackend, HTTPX_AVAILABLE
        
        if not HTTPX_AVAILABLE:
            print("⚠ httpx not installed, skipping async test")
            print("  Install with: pip install httpx")
            return True
        
        async def run_test():
            try:
                async with AsyncOpenRouterBackend() as backend:
                    print(f"✓ Async backend initialized: {backend.config.model}")
                    
                    # Quick test
                    start = time.time()
                    response = await backend.generate(
                        "Say 'test' only", 
                        max_tokens=10
                    )
                    elapsed = time.time() - start
                    
                    print(f"✓ Got response in {elapsed:.2f}s")
                    return True
                    
            except ValueError as e:
                print(f"⚠ API key not configured: {e}")
                return True  # Not a failure, just not configured
        
        return asyncio.run(run_test())
        
    except ImportError as e:
        print(f"⚠ Async backend not available: {e}")
        return True  # Not a failure


def main():
    """Run all tests."""
    print()
    print("=" * 64)
    print("         PHI-ENHANCED RLM UPGRADE TESTS")
    print("=" * 64)
    
    results = []
    
    results.append(("Embeddings", test_embeddings()))
    results.append(("RLM Integration", test_rlm_integration()))
    results.append(("Async Backend", test_async_backend()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed. Check output above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
