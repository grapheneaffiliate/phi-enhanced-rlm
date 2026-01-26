#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path

# Fix Windows console encoding via environment
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    # Reconfigure stdout/stderr if possible (Python 3.7+)
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

"""
PHI-ENHANCED RLM v2.0 UPGRADE TESTS
===================================
Verify all new features work correctly.

Run:
    python tests/test_upgrades.py
"""

import os
import json
import tempfile
import time

# Test results tracking
results = {"passed": 0, "failed": 0, "skipped": 0}


def test(name):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print('='*60)
            try:
                func()
                print(f"[PASS] {name}")
                results["passed"] += 1
                return True
            except ImportError as e:
                print(f"[SKIP] {name} (missing dependency: {e})")
                results["skipped"] += 1
                return None
            except Exception as e:
                print(f"[FAIL] {name}")
                print(f"   Error: {e}")
                import traceback
                traceback.print_exc()
                results["failed"] += 1
                return False
        return wrapper
    return decorator


# =============================================================================
# 1. SQLITE CACHE TESTS
# =============================================================================

@test("SQLite Embedding Cache")
def test_sqlite_cache():
    from cache import SQLiteEmbeddingCache
    import numpy as np
    
    # Use temp file (will be cleaned up manually)
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    
    try:
        cache = SQLiteEmbeddingCache(db_path=db_path)
        
        # Test set/get
        text = "Hello, world!"
        model = "test-model"
        embedding = np.random.randn(384).astype(np.float32)
        
        cache.set(text, model, embedding)
        retrieved = cache.get(text, model)
        
        assert retrieved is not None, "Cache returned None"
        assert np.allclose(embedding, retrieved), "Embedding mismatch"
        
        # Test batch
        texts = ["text1", "text2", "text3"]
        embeddings = np.random.randn(3, 384).astype(np.float32)
        cache.set_batch(texts, model, embeddings)
        
        results_list, missing = cache.get_batch(texts, model)
        assert len(missing) == 0, f"Missing indices: {missing}"
        
        # Test stats
        stats = cache.get_stats()
        assert stats.entry_count >= 4, f"Expected 4+ entries, got {stats.entry_count}"
        
        print(f"   Cache stats: {stats.hits} hits, {stats.misses} misses, {stats.entry_count} entries")
        
        # Close cache to release file handles
        cache.close()
    finally:
        # Clean up - ignore errors on Windows
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except:
            pass


# =============================================================================
# 2. EXTRACTOR TESTS
# =============================================================================

@test("Web Content Extraction")
def test_web_extraction():
    from extractors import extract_web_content
    
    html = """
    <html>
    <head><title>Test Article</title></head>
    <body>
    <nav>Navigation</nav>
    <article>
        <h1>Main Title</h1>
        <p>This is the main content of the article. It contains important information.</p>
        <p>Another paragraph with more details about the topic.</p>
    </article>
    <footer>Footer content</footer>
    </body>
    </html>
    """
    
    result = extract_web_content(html, "https://example.com")
    
    assert result.text, "No text extracted"
    assert "main content" in result.text.lower(), "Main content not found"
    assert "navigation" not in result.text.lower(), "Navigation should be removed"
    
    print(f"   Extracted {len(result.text)} chars")
    print(f"   Title: {result.title}")


@test("PDF Extraction")
def test_pdf_extraction():
    from extractors import extract_pdf_content
    
    # Check if we have a test PDF
    test_pdf = Path(__file__).parent / "2512.24601v1.pdf"
    if not test_pdf.exists():
        raise ImportError("No test PDF available")
    
    result = extract_pdf_content(str(test_pdf))
    
    assert result.text, "No text extracted from PDF"
    assert result.metadata.get("page_count", 0) > 0, "No page count"
    
    print(f"   Title: {result.title}")
    print(f"   Pages: {result.metadata.get('page_count')}")
    print(f"   Text length: {len(result.text)} chars")


@test("Code Chunking")
def test_code_chunking():
    from extractors import chunk_python_code, chunk_javascript_code
    
    python_code = '''
def hello():
    """Say hello."""
    print("Hello!")

class Greeter:
    """A greeter class."""
    
    def greet(self, name):
        return f"Hello, {name}!"
    
    def goodbye(self):
        return "Goodbye!"
'''
    
    chunks = chunk_python_code(python_code)
    
    assert len(chunks) >= 2, f"Expected 2+ chunks, got {len(chunks)}"
    
    types = [c["type"] for c in chunks]
    assert "function" in types, "No function chunk found"
    assert "class" in types, "No class chunk found"
    
    print(f"   Found {len(chunks)} chunks:")
    for c in chunks:
        print(f"      {c['type']}: {c['name']}")


# =============================================================================
# 3. PROGRESS UTILITIES TESTS
# =============================================================================

@test("Progress Manager")
def test_progress():
    from progress import get_progress_manager, SimpleProgressManager
    
    pm = get_progress_manager(use_rich=False)  # Use simple for testing
    assert isinstance(pm, SimpleProgressManager)
    
    # Test tracking
    with pm.track_analysis("Test query", total_chunks=5, max_depth=2) as tracker:
        for i in range(5):
            tracker.update(
                processed=i + 1,
                depth=i // 2,
                confidence=0.5 + i * 0.1
            )
    
    print(f"   Progress manager works correctly")


@test("Confidence Visualization")
def test_confidence_viz():
    from progress import visualize_confidence_tree
    
    trace = [
        {"depth": 0, "confidence": 0.75, "info_flow": 150, "selected_ids": [0, 3], "stop_reason": "none"},
        {"depth": 1, "confidence": 0.82, "info_flow": 45, "selected_ids": [1], "stop_reason": "momentum"},
    ]
    
    # Should not raise
    visualize_confidence_tree(trace, use_rich=False)
    print("   Visualization works")


# =============================================================================
# 4. EMBEDDINGS WITH SQLITE CACHE TESTS
# =============================================================================

@test("Embeddings with SQLite Cache")
def test_embeddings_sqlite():
    from embeddings import EmbeddingCache
    import numpy as np
    
    tmpdir = tempfile.mkdtemp()
    try:
        cache = EmbeddingCache(cache_dir=tmpdir, use_sqlite=True)
        
        # Test set/get
        text = "Test embedding text"
        model = "test-model"
        embedding = np.random.randn(384).astype(np.float32)
        
        cache.set(text, model, embedding)
        retrieved = cache.get(text, model)
        
        assert retrieved is not None, "Cache returned None"
        
        # Check stats
        stats = cache.get_stats()
        assert stats.get("backend") in ["sqlite", "memory"], f"Unexpected backend: {stats}"
        
        print(f"   Backend: {stats.get('backend')}")
        print(f"   Entries: {stats.get('entries', stats.get('entry_count', 0))}")
        
        # Close SQLite if available
        if cache._sqlite_cache:
            cache._sqlite_cache.close()
    finally:
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except:
            pass


# =============================================================================
# 5. PHI-ENHANCED RLM TESTS
# =============================================================================

@test("RLM Parallel Processing Setup")
def test_rlm_parallel():
    from phi_enhanced_rlm import PhiEnhancedRLM, MockLLMBackend
    
    mock_llm = MockLLMBackend(seed=42)
    chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
    
    rlm = PhiEnhancedRLM(
        base_llm_callable=mock_llm,
        context_chunks=chunks,
        total_budget_tokens=1024
    )
    
    # Enable parallel
    rlm.enable_parallel(True)
    assert hasattr(rlm, 'parallel_enabled')
    assert rlm.parallel_enabled == True
    
    print("   Parallel processing can be enabled")


@test("RLM Reasoning Tree")
def test_rlm_reasoning_tree():
    from phi_enhanced_rlm import PhiEnhancedRLM, MockLLMBackend
    import tempfile
    
    mock_llm = MockLLMBackend(seed=42)
    chunks = ["The golden ratio φ", "E8 Lie group", "Recursive reasoning"]
    
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        trace_file = f.name
    
    try:
        rlm = PhiEnhancedRLM(
            base_llm_callable=mock_llm,
            context_chunks=chunks,
            total_budget_tokens=1024,
            trace_file=trace_file
        )
        
        # Run a simple query
        result = rlm.recursive_solve("Test query", max_depth=2)
        
        # Get reasoning tree
        tree = rlm.get_reasoning_tree()
        
        assert "error" not in tree, f"Tree error: {tree.get('error')}"
        assert tree["total_nodes"] > 0, "No nodes in tree"
        
        print(f"   Tree has {tree['total_nodes']} nodes")
        print(f"   Max depth: {tree['max_depth']}")
        print(f"   Avg confidence: {tree['avg_confidence']:.1%}")
        
    finally:
        if os.path.exists(trace_file):
            os.unlink(trace_file)


# =============================================================================
# 6. API TESTS (structure only, no server needed)
# =============================================================================

@test("API Models")
def test_api_models():
    from api import (
        AnalyzeRequest, ChatRequest, CompareRequest,
        AnalyzeResponse, ChatResponse, StatusResponse
    )
    
    # Test request models
    req = AnalyzeRequest(query="Test query", max_depth=3)
    assert req.query == "Test query"
    assert req.max_depth == 3
    
    chat_req = ChatRequest(message="Hello", session_id="test-123")
    assert chat_req.message == "Hello"
    
    compare_req = CompareRequest(source1="repo1", source2="repo2")
    assert compare_req.source1 == "repo1"
    
    print("   All API models validate correctly")


@test("API State Management")
def test_api_state():
    from api import AppState
    
    state = AppState()
    
    # Test history
    state.add_to_history("query1", "answer1", 0.8, "session1")
    state.add_to_history("query2", "answer2", 0.9, "session1")
    
    assert len(state.history) == 2
    
    # Test sessions
    session = state.get_session("test-session")
    assert isinstance(session, list)
    
    state.add_to_session("test-session", "user", "Hello")
    state.add_to_session("test-session", "assistant", "Hi there")
    
    session = state.get_session("test-session")
    assert len(session) == 2
    
    print("   State management works correctly")


# =============================================================================
# 7. INTEGRATION TEST
# =============================================================================

@test("End-to-End Integration")
def test_integration():
    from phi_enhanced_rlm import PhiEnhancedRLM, MockLLMBackend
    from progress import get_progress_manager
    import tempfile
    
    # Setup
    mock_llm = MockLLMBackend(seed=42)
    chunks = [
        "The golden ratio φ = 1.618 appears in mathematics.",
        "E8 is a Lie group with 248 dimensions.",
        "Recursive reasoning breaks problems into subproblems.",
    ]
    
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        trace_file = f.name
    
    try:
        # Create RLM
        rlm = PhiEnhancedRLM(
            base_llm_callable=mock_llm,
            context_chunks=chunks,
            total_budget_tokens=1024,
            trace_file=trace_file
        )
        
        # Run analysis
        result = rlm.recursive_solve("What is phi?", max_depth=2)
        
        # Verify result
        assert result.value, "Empty answer"
        assert 0 <= result.confidence <= 1, f"Invalid confidence: {result.confidence}"
        assert "depth" in result.metadata
        
        # Verify trace
        tree = rlm.get_reasoning_tree()
        assert tree["total_nodes"] > 0
        
        print(f"   Answer length: {len(result.value)} chars")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Reasoning nodes: {tree['total_nodes']}")
        
    finally:
        if os.path.exists(trace_file):
            os.unlink(trace_file)


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("PHI-ENHANCED RLM v2.0 UPGRADE TESTS")
    print("=" * 60)
    
    tests = [
        test_sqlite_cache,
        test_web_extraction,
        test_pdf_extraction,
        test_code_chunking,
        test_progress,
        test_confidence_viz,
        test_embeddings_sqlite,
        test_rlm_parallel,
        test_rlm_reasoning_tree,
        test_api_models,
        test_api_state,
        test_integration,
    ]
    
    for test_func in tests:
        test_func()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"[PASS] Passed:  {results['passed']}")
    print(f"[FAIL] Failed:  {results['failed']}")
    print(f"[SKIP] Skipped: {results['skipped']}")
    print("=" * 60)
    
    if results["failed"] > 0:
        print("\nSome tests failed!")
        return 1
    else:
        print("\nAll tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
