#!/usr/bin/env python3
"""
PHI-ENHANCED RECURSIVE LANGUAGE MODEL (RLM) FRAMEWORK
======================================================
Full RLM Orchestrator with φ-Separation Mathematics

Implements complete recursive reasoning with:
1. φ-Gram Chunk Selection (greedy Δlogdet)
2. Casimir Flow Budget Allocation
3. φ-Momentum Early Stopping
4. Spectral Flow Saturation Detection
5. Golden Ratio QEC Verification
6. Torsion-Corrected Aggregation
7. Dependency Cohomology Tracking
8. recursive_solve() Driver Engine
"""

import numpy as np
import hashlib
import json
from itertools import combinations
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import logging

logger = logging.getLogger(__name__)

# Import from the provided mathematics library
from phi_separation_novel_mathematics import (
    PHI, PHI_INV, LOG_PHI, EPSILON, CASIMIR_DEGREES, COXETER_NUMBER,
    PhiGramMatrix, SpectralFlow, PhiRenormalizationGroup, 
    PhiGramCohomology, PhiLattice, TorsionCorrectedOperator
)

# Import real embeddings (with fallback to mock)
try:
    from embeddings import get_embedder, CachedEmbedder, EmbeddingConfig
    REAL_EMBEDDINGS_AVAILABLE = True
except ImportError:
    REAL_EMBEDDINGS_AVAILABLE = False
    logger.warning("embeddings module not found, using mock embeddings")

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ContextChunk:
    """Represents a chunk of text context with its embedding."""
    id: int
    text: str
    embedding: np.ndarray
    
    def __hash__(self):
        return hash(self.id)

@dataclass
class SubCallResult:
    """Represents the result of a recursive sub-call."""
    value: Any
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMResponse:
    """Structured response from LLM backend."""
    answer: str
    confidence: float
    subquestions: List[str]
    raw_tokens: List[str] = field(default_factory=list)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def logdet_psd(M: np.ndarray, jitter: float = 1e-10) -> float:
    """Compute log-determinant of PSD matrix with stabilization."""
    Mj = M + jitter * np.eye(M.shape[0])
    sign, ld = np.linalg.slogdet(Mj)
    if sign <= 0:
        return -np.inf
    return float(ld)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def mock_llm_embedding(text: str, dim: int = 64) -> np.ndarray:
    """Robust mock embedding using cryptographic hash (fallback only)."""
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(h, "little")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return v / (np.linalg.norm(v) + 1e-12)


# Global embedder instance (lazy-initialized)
_global_embedder: Optional['CachedEmbedder'] = None

def get_global_embedder() -> Optional['CachedEmbedder']:
    """Get or create the global embedder instance."""
    global _global_embedder
    if _global_embedder is None and REAL_EMBEDDINGS_AVAILABLE:
        try:
            _global_embedder = get_embedder()
            logger.info(f"Initialized embedder: {_global_embedder.provider.__class__.__name__}")
        except Exception as e:
            logger.warning(f"Failed to initialize embedder: {e}")
    return _global_embedder


def get_embedding(text: str, embedder: Optional['CachedEmbedder'] = None) -> np.ndarray:
    """Get embedding for text using real embeddings if available, else mock."""
    if embedder is None:
        embedder = get_global_embedder()
    
    if embedder is not None:
        try:
            return embedder.embed_single(text)
        except Exception as e:
            logger.warning(f"Embedding failed, falling back to mock: {e}")
    
    return mock_llm_embedding(text)


def get_embeddings_batch(texts: List[str], embedder: Optional['CachedEmbedder'] = None) -> np.ndarray:
    """Get embeddings for multiple texts."""
    if embedder is None:
        embedder = get_global_embedder()
    
    if embedder is not None:
        try:
            return embedder.embed(texts)
        except Exception as e:
            logger.warning(f"Batch embedding failed, falling back to mock: {e}")
    
    return np.array([mock_llm_embedding(text) for text in texts])

def simple_tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer for info tracking."""
    return text.lower().split()

# =============================================================================
# PHI-GRAM MATRIX WITH NUMERICAL STABILITY
# =============================================================================

class PhiGramMatrixForEmbeddings:
    """φ-Gram matrix with log-det stability and effective rank collision detection."""
    
    def __init__(self, embeddings: np.ndarray, delta: Optional[float] = None):
        self.embeddings = embeddings
        self.n = len(embeddings)
        
        if delta is None:
            dists = []
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    dists.append(np.linalg.norm(embeddings[i] - embeddings[j]))
            self.delta = float(np.median(dists)) if dists else 1.0
        else:
            self.delta = delta
            
        self._matrix = None
        self._eigenvalues = None
        
    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = np.zeros((self.n, self.n))
            for i in range(self.n):
                for j in range(self.n):
                    dist = np.linalg.norm(self.embeddings[i] - self.embeddings[j])
                    self._matrix[i, j] = np.power(PHI, -dist / self.delta)
        return self._matrix
    
    @property
    def eigenvalues(self) -> np.ndarray:
        if self._eigenvalues is None:
            self._eigenvalues = np.linalg.eigvalsh(self.matrix)
        return self._eigenvalues
    
    @property
    def log_determinant(self) -> float:
        return logdet_psd(self.matrix)
    
    @property
    def min_eigenvalue(self) -> float:
        return float(np.min(self.eigenvalues))
    
    def effective_rank(self, rel_tol: float = 1e-10) -> int:
        eig = np.clip(self.eigenvalues, 0.0, None)
        max_eig = np.max(eig)
        return int(np.sum(eig > rel_tol * max_eig))
    
    def has_collision(self, rel_tol: float = 1e-12) -> bool:
        return self.effective_rank(rel_tol) < self.n
        
    def submatrix(self, indices: List[int]) -> 'PhiGramMatrixForEmbeddings':
        sub_embeddings = self.embeddings[list(indices)]
        return PhiGramMatrixForEmbeddings(sub_embeddings, self.delta)

# =============================================================================
# MOCK LLM BACKEND (for testing)
# =============================================================================

class MockLLMBackend:
    """Mock LLM backend that produces structured responses."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.call_count = 0
    
    def __call__(self, prompt: str, max_tokens: int = 2048) -> str:
        """Return JSON-formatted LLM response."""
        self.call_count += 1
        
        # Parse query from prompt
        query_hash = hash(prompt) % 10000
        
        # Generate synthetic answer
        answer = f"Answer to query (hash={query_hash}): Based on the analysis..."
        
        # Confidence decays with recursion (simulated by call count)
        base_conf = 0.85 - 0.05 * min(self.call_count, 5)
        confidence = max(0.5, min(0.99, base_conf + self.rng.uniform(-0.05, 0.05)))
        
        # Generate 0-2 subquestions (fewer as confidence increases)
        n_subq = max(0, int(2 - confidence * 2) + self.rng.integers(0, 2))
        subquestions = [f"Subquestion {i+1} from depth {self.call_count}" for i in range(n_subq)]
        
        response = {
            "answer": answer,
            "confidence": round(confidence, 3),
            "subquestions": subquestions
        }
        
        return json.dumps(response)

# =============================================================================
# PHI-ENHANCED RLM CORE CLASS
# =============================================================================

class PhiEnhancedRLM:
    """
    Full Recursive Language Model orchestrator with φ-Separation Mathematics.
    
    Now with real embeddings support for better chunk selection!
    """
    
    def __init__(self, base_llm_callable: Callable, context_chunks: List[str], 
                 embeddings: Optional[np.ndarray] = None,
                 total_budget_tokens: int = 4096,
                 trace_file: str = "rlm_trace.jsonl",
                 embedder: Optional['CachedEmbedder'] = None):
        """
        Args:
            base_llm_callable: Function (prompt, max_tokens) -> JSON string
            context_chunks: List of text chunks
            embeddings: Pre-computed embeddings (optional)
            total_budget_tokens: Total token budget for recursion
            trace_file: Path to trace log file
            embedder: Optional embedder instance (uses global if not provided)
        """
        self.llm = base_llm_callable
        self.context_chunks_text = context_chunks
        self.total_budget = total_budget_tokens
        self.trace_file = Path(trace_file)
        
        # Store embedder for query embedding
        self.embedder = embedder or get_global_embedder()
        
        # Generate embeddings if not provided (using REAL embeddings now!)
        if embeddings is None:
            logger.info(f"Generating embeddings for {len(context_chunks)} chunks...")
            embeddings = get_embeddings_batch(context_chunks, self.embedder)
            logger.info(f"Embeddings shape: {embeddings.shape}")
        
        self.full_embeddings = embeddings
        
        # State tracking
        self.info_history = []
        self.confidence_history = []
        self.prev_answer_tokens = set()
        
        # Initialize chunks
        self.chunks = []
        for i, (text, emb) in enumerate(zip(context_chunks, embeddings)):
            self.chunks.append(ContextChunk(id=i, text=text, embedding=emb))
            
        # Initialize φ-Gram Matrix
        self.phi_gram = PhiGramMatrixForEmbeddings(self.full_embeddings)
        
        # Compute budget allocation
        self.budget_map = self.allocate_recursion_budget(total_budget_tokens)
        
        # Clear trace file
        self.trace_file.write_text("")

    # =========================================================================
    # STEP 2: Query-Conditioned Chunk Selection (Relevance → Diversity)
    # =========================================================================
    
    def select_chunks_for_subcall(self, query: str = "", max_chunks: int = 3, 
                                   sim_threshold: float = 0.98,
                                   relevance_pool_size: int = 5) -> List[ContextChunk]:
        """
        Query-conditioned selection: first filter by relevance, then maximize diversity.
        
        1. Embed the query (using REAL embeddings!)
        2. Score all chunks by relevance (cosine similarity to query)
        3. Take top-K most relevant chunks as candidate pool
        4. Apply greedy Δlogdet selection on this pool for diversity
        
        Args:
            query: The current query/subquestion
            max_chunks: Number of chunks to select
            sim_threshold: Maximum similarity between selected chunks
            relevance_pool_size: Size of relevance-filtered candidate pool
        """
        if len(self.chunks) <= max_chunks:
            return self.chunks
        
        # Step 1: Embed the query using real embeddings
        if query:
            query_embedding = get_embedding(query, self.embedder)
        else:
            query_embedding = np.zeros_like(self.chunks[0].embedding)
        
        # Step 2: Score all chunks by relevance to query
        relevance_scores = []
        for i, chunk in enumerate(self.chunks):
            score = cosine_similarity(query_embedding, chunk.embedding)
            relevance_scores.append((i, score))
        
        # Sort by relevance (highest first)
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Step 3: Take top-K most relevant as candidate pool
        pool_size = min(relevance_pool_size, len(self.chunks))
        candidate_pool = [idx for idx, _ in relevance_scores[:pool_size]]
        
        # Step 4: Apply greedy Δlogdet diversity selection on the candidate pool
        selected_indices = []
        remaining = list(candidate_pool)
        
        # First chunk: highest relevance (already sorted)
        if remaining:
            selected_indices.append(remaining[0])
            remaining = remaining[1:]
        
        # Greedily add chunks maximizing Δlogdet (diversity)
        while len(selected_indices) < max_chunks and remaining:
            current_logdet = self.phi_gram.submatrix(selected_indices).log_determinant
            
            best_gain = -np.inf
            best_next = None
            
            for j in remaining:
                # Check similarity constraint (avoid duplicates)
                valid = True
                for sel_idx in selected_indices:
                    sim = cosine_similarity(self.chunks[j].embedding, 
                                           self.chunks[sel_idx].embedding)
                    if sim >= sim_threshold:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Compute Δlogdet (diversity gain)
                new_indices = selected_indices + [j]
                new_logdet = self.phi_gram.submatrix(new_indices).log_determinant
                gain = new_logdet - current_logdet
                
                if gain > best_gain:
                    best_gain = gain
                    best_next = j
            
            if best_next is not None:
                selected_indices.append(best_next)
                remaining.remove(best_next)
            else:
                break  # No valid chunk found
        
        return [self.chunks[i] for i in selected_indices]

    # =========================================================================
    # Casimir Budget Allocation
    # =========================================================================
    
    def allocate_recursion_budget(self, total_budget_tokens: int) -> Dict[int, int]:
        """Allocate tokens across recursion levels using E8 Casimir structure."""
        levels = len(CASIMIR_DEGREES)
        weights = np.power(PHI, -CASIMIR_DEGREES / COXETER_NUMBER)
        normalized_weights = weights / np.sum(weights)
        
        raw = total_budget_tokens * normalized_weights
        alloc = np.floor(raw).astype(int)
        remainder = total_budget_tokens - int(np.sum(alloc))
        
        fractional = raw - alloc
        order = np.argsort(fractional)[::-1]
        for i in order[:remainder]:
            alloc[i] += 1
            
        return {depth: int(alloc[depth]) for depth in range(levels)}
    
    def get_budget_for_depth(self, depth: int) -> int:
        """Get token budget for a given recursion depth."""
        clamped_depth = min(depth, 7)
        return self.budget_map.get(clamped_depth, self.budget_map[7])

    # =========================================================================
    # STEP 3: Information Flow Tracking
    # =========================================================================
    
    def compute_info_units(self, answer_tokens: List[str]) -> float:
        """Compute new information units (Option A: new unique tokens)."""
        current_set = set(answer_tokens)
        new_tokens = current_set - self.prev_answer_tokens
        # Accumulate tokens rather than replace
        self.prev_answer_tokens = self.prev_answer_tokens | current_set
        return float(len(new_tokens))
    
    def update_information_state(self, new_info_units: float):
        """Update spectral flow history."""
        self.info_history.append(new_info_units)

    # =========================================================================
    # φ-Momentum Early Stopping
    # =========================================================================
    
    def should_verify_early_stop(self, momentum_threshold: float = 0.93,
                                  last_step_thresh: float = 0.003,
                                  var_thresh: float = 5e-5,
                                  window: int = 4) -> bool:
        """Check if confidence has converged via φ-momentum + flatness."""
        if len(self.confidence_history) < window:
            return False
            
        momentum = self.confidence_history[0]
        for signal in self.confidence_history[1:]:
            momentum = PHI_INV * momentum + (1 - PHI_INV) * signal
        
        recent = np.array(self.confidence_history[-window:], dtype=float)
        last_step = float(abs(recent[-1] - recent[-2]))
        variance = float(np.var(recent))
        
        return (momentum >= momentum_threshold) and \
               (last_step <= last_step_thresh) and \
               (variance <= var_thresh)

    # =========================================================================
    # Spectral Flow Saturation
    # =========================================================================
    
    def should_continue_recursion(self, consecutive_steps: int = 2) -> bool:
        """Check if information flow has saturated."""
        if len(self.info_history) < 3 + consecutive_steps:
            return True
        
        flows = np.diff(self.info_history)
        nonzero_flows = flows[np.abs(flows) > 1e-12]
        if len(nonzero_flows) < 2:
            return False
        
        baseline = float(np.median(nonzero_flows[:min(5, len(nonzero_flows))]))
        threshold = EPSILON * baseline
        
        tail = flows[-consecutive_steps:]
        return bool(np.any(tail > threshold))

    # =========================================================================
    # STEP 4: QEC Verification
    # =========================================================================
    
    def run_qec_verification(self, answer: str, context: str, 
                              budget: int) -> Tuple[float, List[SubCallResult]]:
        """Run 3 independent verifier calls for QEC."""
        verifier_prompts = [
            f"Check for contradictions in: {answer[:200]}... Context: {context[:100]}",
            f"Check for missing steps in: {answer[:200]}... Context: {context[:100]}",
            f"Provide counterexample if wrong: {answer[:200]}... Context: {context[:100]}"
        ]
        
        results = []
        for i, prompt in enumerate(verifier_prompts):
            try:
                resp = self.llm(prompt, max_tokens=budget // 3)
                # Parse response (mock returns JSON)
                try:
                    parsed = json.loads(resp)
                    conf = parsed.get("confidence", 0.5)
                except:
                    conf = 0.5
                
                # Score: 1.0 if verification passes, 0.0 if fails
                score = 1.0 if conf > 0.6 else 0.0
                results.append(SubCallResult(value=score, confidence=conf, 
                                            metadata={"verifier": i}))
            except Exception as e:
                results.append(SubCallResult(value=0.5, confidence=0.5, 
                                            metadata={"error": str(e)}))
        
        # Compute revised confidence based on majority
        passes = sum(1 for r in results if r.value > 0.5)
        if passes >= 2:
            revised_conf = 0.85 + 0.05 * passes / 3
        else:
            revised_conf = 0.4 - 0.1 * (3 - passes)
        
        return max(0.1, min(0.99, revised_conf)), results

    # =========================================================================
    # Torsion-Corrected Aggregation
    # =========================================================================
    
    def aggregate_results(self, results: List[SubCallResult]) -> SubCallResult:
        """E8 torsion-corrected aggregation of sub-results."""
        if not results:
            return SubCallResult(value="", confidence=0.0, metadata={})
        
        # For string values, use highest-confidence answer
        # For numeric values, use weighted average with torsion
        
        if all(isinstance(r.value, (int, float)) for r in results):
            values = np.array([r.value for r in results])
            confidences = np.array([r.confidence for r in results])
            
            if np.sum(confidences) == 0:
                confidences[:] = 1.0
            
            base_answer = np.average(values, weights=confidences)
            inv_confidences = 1.0 - confidences
            
            if np.sum(inv_confidences) > 0:
                torsion_term = EPSILON * np.average(values, weights=inv_confidences)
            else:
                torsion_term = 0.0
            
            final_value = float(base_answer + torsion_term)
            final_conf = float(np.average(confidences))
        else:
            # String aggregation: pick highest confidence
            best = max(results, key=lambda r: r.confidence)
            final_value = best.value
            final_conf = best.confidence
        
        return SubCallResult(value=final_value, confidence=final_conf, 
                            metadata={"aggregated_from": len(results)})

    # =========================================================================
    # STEP 5: Dependency Cohomology on Active Graph
    # =========================================================================
    
    def analyze_dependency_structure(self, indices: Optional[List[int]] = None,
                                      similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """Analyze dependency structure on selected indices."""
        if indices is None:
            gram = self.phi_gram
        else:
            gram = self.phi_gram.submatrix(indices)
        
        M = gram.matrix
        A = (M > similarity_threshold).astype(float)
        np.fill_diagonal(A, 0.0)
        
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        eigvals = np.linalg.eigvalsh(L)
        eigvals_sorted = np.sort(eigvals)
        
        harmonic_count = int(np.sum(np.abs(eigvals) < 1e-8))
        spectral_gap = float(eigvals_sorted[1]) if len(eigvals) > 1 else 0.0
        
        return {
            "connected_components": harmonic_count,
            "spectral_gap": spectral_gap,
            "effective_rank": gram.effective_rank(),
            "has_collision": gram.has_collision(),
        }

    # =========================================================================
    # STEP 7: Production Logging
    # =========================================================================
    
    def log_trace(self, depth: int, query: str, selected_ids: List[int],
                  logdet_selected: float, confidence: float, info_flow: float,
                  stop_reason: str):
        """Log a single recursion node to trace file."""
        entry = {
            "depth": depth,
            "query": query[:100],
            "selected_ids": selected_ids,
            "logdet_selected": round(logdet_selected, 4),
            "collision_full": self.phi_gram.has_collision(),
            "collision_selected": self.phi_gram.submatrix(selected_ids).has_collision() if selected_ids else False,
            "confidence": round(confidence, 4),
            "info_flow": round(info_flow, 4),
            "stop_reason": stop_reason
        }
        with open(self.trace_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # =========================================================================
    # STEP 1: MAIN RECURSIVE SOLVE ENGINE
    # =========================================================================
    
    def recursive_solve(self, query: str, depth: int = 0, 
                        path: Tuple[int, ...] = (),
                        max_depth: int = 5) -> SubCallResult:
        """
        Main RLM recursion engine.
        
        Args:
            query: The question to answer
            depth: Current recursion depth
            path: Tuple tracking recursion path (for debugging)
            max_depth: Maximum recursion depth
        
        Returns:
            SubCallResult with final answer, confidence, and metadata
        """
        # Get budget for this depth
        budget = self.get_budget_for_depth(depth)
        
        # Step 1: Select working chunks (query-conditioned)
        selected = self.select_chunks_for_subcall(query=query, max_chunks=3)
        selected_ids = [c.id for c in selected]
        selected_text = "\n".join([c.text for c in selected])
        
        # Compute logdet of selection
        sub_gram = self.phi_gram.submatrix(selected_ids)
        logdet_selected = sub_gram.log_determinant
        
        # Step 2: Build prompt
        prompt = f"""Query: {query}

Context:
{selected_text}

Recursion depth: {depth}
Remaining budget: {budget} tokens

Respond in JSON format:
{{"answer": "your answer", "confidence": 0.0-1.0, "subquestions": ["...", "..."]}}
"""
        
        # Step 3: Call LLM backend
        try:
            response_str = self.llm(prompt, max_tokens=budget)
            response = json.loads(response_str)
            answer = response.get("answer", "")
            raw_confidence = response.get("confidence", 0.5)
            subquestions = response.get("subquestions", [])
        except Exception as e:
            answer = f"Error: {e}"
            raw_confidence = 0.3
            subquestions = []
        
        # Step 4: QEC verification
        revised_conf, verifier_results = self.run_qec_verification(answer, selected_text, budget // 2)
        confidence = (raw_confidence + revised_conf) / 2
        
        # Update confidence history
        self.confidence_history.append(confidence)
        
        # Step 4: Update spectral flow
        answer_tokens = simple_tokenize(answer)
        new_info = self.compute_info_units(answer_tokens)
        self.update_information_state(new_info)
        
        # Determine stop reason
        stop_reason = "none"
        
        # Check depth limit
        if depth >= max_depth:
            stop_reason = "depth"
        
        # Step 5: Early-stop check (φ-momentum)
        elif self.should_verify_early_stop():
            stop_reason = "momentum"
        
        # Step 6: Saturation stop check (spectral flow)
        elif not self.should_continue_recursion():
            stop_reason = "spectral"
        
        # Log this node - info_flow is the current new info units
        self.log_trace(depth, query, selected_ids, logdet_selected, confidence, new_info, stop_reason)
        
        # If stopping, return current answer
        if stop_reason != "none":
            return SubCallResult(
                value=answer,
                confidence=confidence,
                metadata={
                    "depth": depth,
                    "path": path,
                    "stop_reason": stop_reason,
                    "selected_ids": selected_ids,
                    "verifier_results": [r.confidence for r in verifier_results]
                }
            )
        
        # Step 7: Recurse on subquestions
        if not subquestions:
            return SubCallResult(
                value=answer,
                confidence=confidence,
                metadata={"depth": depth, "path": path, "stop_reason": "no_subquestions"}
            )
        
        sub_results = []
        for i, subq in enumerate(subquestions[:3]):  # Limit to 3 subquestions
            sub_result = self.recursive_solve(subq, depth + 1, path + (i,), max_depth)
            sub_results.append(sub_result)
        
        # Aggregate sub-results with torsion correction
        aggregated = self.aggregate_results(sub_results)
        
        # Combine with current answer
        final_answer = f"{answer}\n\nSub-analysis: {aggregated.value}"
        final_conf = (confidence + aggregated.confidence) / 2
        
        return SubCallResult(
            value=final_answer,
            confidence=final_conf,
            metadata={
                "depth": depth,
                "path": path,
                "stop_reason": "recursion_complete",
                "n_subquestions": len(subquestions),
                "aggregated_confidence": aggregated.confidence
            }
        )

# =============================================================================
# DEMONSTRATION
# =============================================================================

def run_full_rlm_demonstration():
    """Run complete RLM demonstration."""
    print("="*70)
    print("PHI-ENHANCED RLM - FULL RECURSIVE ORCHESTRATOR")
    print("="*70)
    print()
    
    # Setup context
    context_chunks = [
        "The golden ratio φ = 1.618 appears throughout mathematics and nature.",
        "E8 is the largest exceptional Lie group with 248 dimensions.",
        "Recursive Language Models decompose complex queries into sub-tasks.",
        "Information theory quantifies the uncertainty in random variables.",
        "The Casimir effect is a quantum phenomenon from vacuum fluctuations.",
        "Machine learning models benefit from hierarchical feature extraction.",
        "Spectral graph theory connects eigenvalues to graph structure.",
    ]
    
    # Create mock LLM backend
    mock_llm = MockLLMBackend(seed=42)
    
    # Initialize RLM
    rlm = PhiEnhancedRLM(
        base_llm_callable=mock_llm,
        context_chunks=context_chunks,
        total_budget_tokens=2048,
        trace_file="rlm_trace.jsonl"
    )
    
    print(f"✓ RLM initialized with {len(context_chunks)} context chunks")
    print(f"✓ Budget allocation: {rlm.budget_map}")
    print()
    
    # Run recursive solve
    query = "Explain how the golden ratio relates to E8 symmetry and recursive reasoning."
    
    print(f"Query: {query}")
    print("-" * 70)
    print()
    
    result = rlm.recursive_solve(query, max_depth=4)
    
    print("="*70)
    print("FINAL RESULT")
    print("="*70)
    print(f"Answer: {result.value[:500]}...")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Stop reason: {result.metadata.get('stop_reason', 'N/A')}")
    print(f"Final depth: {result.metadata.get('depth', 0)}")
    print()
    
    # Show trace
    print("="*70)
    print("RECURSION TRACE")
    print("="*70)
    with open("rlm_trace.jsonl", "r") as f:
        for line in f:
            entry = json.loads(line)
            print(f"  Depth {entry['depth']}: conf={entry['confidence']:.3f}, "
                  f"flow={entry['info_flow']:.2f}, stop={entry['stop_reason']}, "
                  f"chunks={entry['selected_ids']}")
    print()
    
    # Summary
    print("="*70)
    print("✓ RLM demonstration complete!")
    print("="*70)

if __name__ == "__main__":
    run_full_rlm_demonstration()
