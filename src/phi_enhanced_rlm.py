#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys as _sys
import os as _os
# Fix Windows console encoding
if _sys.platform == 'win32':
    _os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    if hasattr(_sys.stdout, 'reconfigure'):
        try:
            _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

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

import numpy as np  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import asyncio  # noqa: E402
import concurrent.futures  # noqa: E402
from typing import List, Dict, Any, Optional, Tuple, Callable  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402

logger = logging.getLogger(__name__)

# Thread pool for parallel processing
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Import from the provided mathematics library
from .phi_separation_novel_mathematics import (  # noqa: E402
    PHI, PHI_INV, EPSILON, CASIMIR_DEGREES, COXETER_NUMBER
)

# Import real embeddings (with fallback to mock)
try:
    from .embeddings import get_embedder, CachedEmbedder, EmbeddingConfig  # noqa: F401
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
                 embedder: Optional['CachedEmbedder'] = None,
                 evolution_state: Optional[Any] = None,
                 memory_path: str = "phi_rlm_memory.json",
                 skills_dir: str = ".claude/skills",
                 enable_tools: Optional[Dict[str, bool]] = None):
        """
        Args:
            base_llm_callable: Function (prompt, max_tokens) -> JSON string
            context_chunks: List of text chunks
            embeddings: Pre-computed embeddings (optional)
            total_budget_tokens: Total token budget for recursion
            trace_file: Path to trace log file
            embedder: Optional embedder instance (uses global if not provided)
            evolution_state: Optional EvolutionState with learned parameters
            memory_path: Path to phi-spiral memory database
            skills_dir: Path to skills directory
            enable_tools: Tool flags, e.g. {"code_exec": True, "web": True, "shell": True}
        """
        self.llm = base_llm_callable
        self.context_chunks_text = context_chunks
        self.total_budget = total_budget_tokens
        self.trace_file = Path(trace_file)
        self.evolution_state = evolution_state
        self._memory_path = memory_path
        self._skills_dir = skills_dir
        self._enable_tools = enable_tools or {}

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

        # Apply evolution state overrides if provided
        if evolution_state is not None:
            self._apply_evolution_state(evolution_state)

        # Initialize φ-attention injector (lazy import to avoid circular deps)
        self._phi_attention = None
        self._phi_sparse = None
        self._spiral_memory = None

        # Initialize agent router and skill loader (claude-code-templates integration)
        self._agent_router = None
        self._skill_loader = None

        # Tool executor for agentic capability (v4.1)
        self._tool_registry = None

        # Semantic info flow tracking (v4.1)
        self._prev_embeddings = []

        # Clear trace file
        self.trace_file.write_text("")

    def _apply_evolution_state(self, state):
        """Apply learned parameters from evolution state."""
        if hasattr(state, 'budget_weights'):
            weights = np.array(state.budget_weights[:8], dtype=float)
            weights = np.maximum(weights, 0.1)
            total = weights.sum()
            for i in range(min(8, len(self.budget_map))):
                self.budget_map[i] = int(self.total_budget * weights[i] / total)

    def _get_phi_attention(self):
        """Lazy-load φ-attention injector."""
        if self._phi_attention is None:
            try:
                from .phi_attention import PhiAttentionInjector
                self._phi_attention = PhiAttentionInjector()
            except ImportError:
                self._phi_attention = False
        return self._phi_attention if self._phi_attention is not False else None

    def _get_phi_sparse(self):
        """Lazy-load φ-sparse reasoner."""
        if self._phi_sparse is None:
            try:
                pruning_ratio = 0.618
                if self.evolution_state and hasattr(self.evolution_state, 'pruning_ratio'):
                    pruning_ratio = self.evolution_state.pruning_ratio
                from .phi_sparse_reasoning import PhiSparseReasoner
                self._phi_sparse = PhiSparseReasoner(pruning_ratio=pruning_ratio)
            except ImportError:
                self._phi_sparse = False
        return self._phi_sparse if self._phi_sparse is not False else None

    def _get_spiral_memory(self):
        """Lazy-load φ-spiral memory."""
        if self._spiral_memory is None:
            try:
                from .phi_memory import PhiSpiralMemory
                self._spiral_memory = PhiSpiralMemory(db_path=self._memory_path)
            except ImportError:
                self._spiral_memory = False
        return self._spiral_memory if self._spiral_memory is not False else None

    def _get_agent_router(self):
        """Lazy-load agent router for depth-specialized personas."""
        if self._agent_router is None:
            try:
                from .agent_router import AgentRouter
                self._agent_router = AgentRouter()
            except ImportError:
                self._agent_router = False
        return self._agent_router if self._agent_router is not False else None

    def _get_skill_loader(self):
        """Lazy-load skill loader for supplementary context."""
        if self._skill_loader is None:
            try:
                from .skill_loader import SkillLoader
                self._skill_loader = SkillLoader(skills_dir=self._skills_dir)
            except ImportError:
                self._skill_loader = False
        return self._skill_loader if self._skill_loader is not False else None

    def _get_tool_registry(self):
        """Lazy-load tool registry for agentic capability."""
        if self._tool_registry is None:
            try:
                from .tool_executor import ToolRegistry
                self._tool_registry = ToolRegistry(
                    enable_code_exec=self._enable_tools.get("code_exec", False),
                    enable_web=self._enable_tools.get("web", False),
                    enable_shell=self._enable_tools.get("shell", False),
                )
            except ImportError:
                self._tool_registry = False
        return self._tool_registry if self._tool_registry is not False else None

    def set_tool_registry(self, registry):
        """Set a custom tool registry (e.g., with code execution enabled)."""
        self._tool_registry = registry

    def compute_semantic_info_units(self, answer_embedding: np.ndarray) -> float:
        """Measure info gain as cosine distance from previous answer embeddings.

        Complements token-based info flow with semantic novelty detection.
        Two answers with different words but same meaning register as low info.
        Two answers with same words but different meaning register as high info.
        """
        if len(self._prev_embeddings) == 0:
            self._prev_embeddings.append(answer_embedding)
            return 1.0

        centroid = np.mean(self._prev_embeddings, axis=0)
        norm_a = np.linalg.norm(answer_embedding)
        norm_c = np.linalg.norm(centroid)
        if norm_a < 1e-10 or norm_c < 1e-10:
            return 0.0
        cos_sim = float(np.dot(answer_embedding, centroid) / (norm_a * norm_c))
        dist = 1.0 - cos_sim
        self._prev_embeddings.append(answer_embedding)
        return max(0.0, dist)

    # =========================================================================
    # STEP 2: Query-Conditioned Chunk Selection (Relevance -> Diversity)
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

        # Step 2: Score all chunks by relevance using φ-kernel retrieval
        try:
            from .phi_retrieval import phi_retrieval_score
            use_phi_retrieval = True
        except ImportError:
            use_phi_retrieval = False

        relevance_scores = []
        for i, chunk in enumerate(self.chunks):
            if use_phi_retrieval:
                score = phi_retrieval_score(
                    query_embedding, chunk.embedding,
                    delta=getattr(self.phi_gram, 'delta', 1.0)
                )
            else:
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
        """Run 3 independent verifier calls for QEC.

        When the agent router is available, uses the fact-checker agent
        persona to enhance verification quality.
        """
        # Use fact-checker agent persona if available
        fc_prefix = ""
        agent_router = self._get_agent_router()
        if agent_router:
            fc_prompt = agent_router.get_verifier_prompt()
            if fc_prompt:
                fc_prefix = f"{fc_prompt}\n\n"

        verifier_prompts = [
            f"{fc_prefix}Check for contradictions in: {answer[:200]}... Context: {context[:100]}",
            f"{fc_prefix}Check for missing steps in: {answer[:200]}... Context: {context[:100]}",
            f"{fc_prefix}Provide counterexample if wrong: {answer[:200]}... Context: {context[:100]}"
        ]

        # Inject phi-attention into QEC verifier prompts (Finding 3 fix)
        phi_attn = self._get_phi_attention()
        if phi_attn and (self.evolution_state is None or
                         getattr(self.evolution_state, 'phi_attention_enabled', True)):
            verifier_prompts = [
                phi_attn.build_phi_prompt(p, "", 5, budget // 3)
                for p in verifier_prompts
            ]

        results = []
        for i, prompt in enumerate(verifier_prompts):
            try:
                resp = self.llm(prompt, max_tokens=budget // 3)
                # Parse response (mock returns JSON)
                try:
                    parsed = json.loads(resp)
                    conf = parsed.get("confidence", 0.5)
                except Exception:
                    conf = 0.5

                # Score: 1.0 if verification passes, 0.0 if fails
                score = 1.0 if conf > 0.6 else 0.0
                results.append(SubCallResult(value=score, confidence=conf,
                                            metadata={"verifier": i}))
            except Exception as e:
                results.append(SubCallResult(value=0.5, confidence=0.5,
                                            metadata={"error": str(e)}))

        # 4th verifier: PhiCritic (structurally different -- uses recursive_solve)
        # Guard against re-entrant calls: critic's recursive_solve would call QEC
        # again, so we only invoke the critic at the top level.
        if not getattr(self, '_critic_active', False):
            try:
                from .phi_critic import PhiCritic
                self._critic_active = True
                critic = PhiCritic(self, critique_depth=1)
                critique = critic.critique(answer, answer, context)
                results.append(SubCallResult(
                    value=critique.quality_score,
                    confidence=critique.quality_score,
                    metadata={"verifier": 3, "type": "critic",
                              "flaws": critique.flaws_found}
                ))
            except Exception:
                pass  # Critic is best-effort enhancement
            finally:
                self._critic_active = False

        # Compute revised confidence based on majority
        passes = sum(1 for r in results if r.value > 0.5)
        total_verifiers = len(results)
        if passes >= (total_verifiers // 2 + 1):
            revised_conf = 0.85 + 0.05 * passes / total_verifiers
        else:
            revised_conf = 0.4 - 0.1 * (total_verifiers - passes)

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
            "stop_reason": stop_reason,
            "timestamp": time.time()
        }
        # Add dependency structure metrics if available
        if len(selected_ids) > 1:
            try:
                dep = self.analyze_dependency_structure(selected_ids)
                entry["dep_components"] = dep["connected_components"]
                entry["dep_spectral_gap"] = round(dep["spectral_gap"], 4)
            except Exception:
                pass
        with open(self.trace_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # =========================================================================
    # PARALLEL SUBQUESTION PROCESSING
    # =========================================================================

    def enable_parallel(self, enabled: bool = True):
        """Enable or disable parallel subquestion processing."""
        self.parallel_enabled = enabled
        logger.info(f"Parallel processing: {'enabled' if enabled else 'disabled'}")

    def _process_subquestions_parallel(self, subquestions: List[str],
                                        depth: int, path: Tuple[int, ...],
                                        max_depth: int) -> List['SubCallResult']:
        """
        Process subquestions in parallel using thread pool.

        Args:
            subquestions: List of subquestions to process
            depth: Current depth
            path: Current path
            max_depth: Maximum depth

        Returns:
            List of SubCallResults
        """
        def process_one(args):
            i, subq = args
            return self.recursive_solve(subq, depth + 1, path + (i,), max_depth)

        # Submit all tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(subquestions), 3)) as executor:
            futures = list(executor.map(process_one, enumerate(subquestions)))

        return list(futures)

    async def recursive_solve_async(self, query: str, depth: int = 0,
                                     path: Tuple[int, ...] = (),
                                     max_depth: int = 5) -> 'SubCallResult':
        """
        Async version of recursive_solve for use with asyncio.

        Uses asyncio.to_thread for LLM calls to avoid blocking.
        """
        # Get budget for this depth
        budget = self.get_budget_for_depth(depth)

        # Select chunks
        selected = self.select_chunks_for_subcall(query=query, max_chunks=3)
        selected_ids = [c.id for c in selected]
        selected_text = "\n".join([c.text for c in selected])

        # Build prompt
        prompt = f"""Query: {query}

Context:
{selected_text}

Recursion depth: {depth}
Remaining budget: {budget} tokens

Respond in JSON format:
{{"answer": "your answer", "confidence": 0.0-1.0, "subquestions": ["...", "..."]}}
"""

        # Call LLM in thread to avoid blocking
        try:
            response_str = await asyncio.to_thread(self.llm, prompt, budget)
            response = json.loads(response_str)
            answer = response.get("answer", "")
            raw_confidence = response.get("confidence", 0.5)
            subquestions = response.get("subquestions", [])
        except Exception as e:
            answer = f"Error: {e}"
            raw_confidence = 0.3
            subquestions = []

        # QEC verification (can also be parallelized)
        revised_conf, verifier_results = await asyncio.to_thread(
            self.run_qec_verification, answer, selected_text, budget // 2
        )
        confidence = (raw_confidence + revised_conf) / 2

        # Update state
        self.confidence_history.append(confidence)
        answer_tokens = simple_tokenize(answer)
        new_info = self.compute_info_units(answer_tokens)
        self.update_information_state(new_info)

        # Determine stop reason
        stop_reason = "none"
        if depth >= max_depth:
            stop_reason = "depth"
        elif self.should_verify_early_stop():
            stop_reason = "momentum"
        elif not self.should_continue_recursion():
            stop_reason = "spectral"

        # Compute logdet
        sub_gram = self.phi_gram.submatrix(selected_ids)
        logdet_selected = sub_gram.log_determinant

        # Log
        self.log_trace(depth, query, selected_ids, logdet_selected, confidence, new_info, stop_reason)

        if stop_reason != "none" or not subquestions:
            return SubCallResult(
                value=answer,
                confidence=confidence,
                metadata={
                    "depth": depth,
                    "path": path,
                    "stop_reason": stop_reason if stop_reason != "none" else "no_subquestions",
                    "selected_ids": selected_ids,
                }
            )

        # Recurse in parallel using asyncio.gather
        tasks = [
            self.recursive_solve_async(subq, depth + 1, path + (i,), max_depth)
            for i, subq in enumerate(subquestions[:3])
        ]
        sub_results = await asyncio.gather(*tasks)

        # Aggregate
        aggregated = self.aggregate_results(list(sub_results))
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
            }
        )

    # =========================================================================
    # REASONING TREE VISUALIZATION
    # =========================================================================

    def get_reasoning_tree(self) -> Dict[str, Any]:
        """
        Get the reasoning tree from the trace file.

        Returns:
            Dict containing tree structure with confidence at each node
        """
        trace = []
        try:
            with open(self.trace_file, "r") as f:
                for line in f:
                    if line.strip():
                        trace.append(json.loads(line))
        except FileNotFoundError:
            return {"error": "No trace file found"}

        if not trace:
            return {"error": "Empty trace"}

        # Build tree structure
        tree = {
            "total_nodes": len(trace),
            "max_depth": max(e.get("depth", 0) for e in trace),
            "avg_confidence": sum(e.get("confidence", 0) for e in trace) / len(trace),
            "nodes": []
        }

        for entry in trace:
            node = {
                "depth": entry.get("depth", 0),
                "query": entry.get("query", "")[:50],
                "confidence": entry.get("confidence", 0),
                "info_flow": entry.get("info_flow", 0),
                "chunks": entry.get("selected_ids", []),
                "stop_reason": entry.get("stop_reason", "none"),
                "logdet": entry.get("logdet_selected", 0),
            }
            tree["nodes"].append(node)

        return tree

    def print_reasoning_tree(self):
        """Print formatted reasoning tree to console."""
        tree = self.get_reasoning_tree()

        if "error" in tree:
            print(f"Error: {tree['error']}")
            return

        print("\n" + "=" * 60)
        print("REASONING TREE")
        print("=" * 60)
        print(f"Total nodes: {tree['total_nodes']}")
        print(f"Max depth: {tree['max_depth']}")
        print(f"Avg confidence: {tree['avg_confidence']:.2%}")
        print("-" * 60)

        for node in tree["nodes"]:
            indent = "  " * node["depth"]
            conf = node["confidence"]

            # ASCII indicator based on confidence
            if conf >= 0.8:
                indicator = "[+]"
            elif conf >= 0.6:
                indicator = "[~]"
            else:
                indicator = "[-]"

            print(f"{indent}{indicator} D{node['depth']}: {node['query'][:40]}...")
            print(f"{indent}    Conf: {conf:.1%} | Info: {node['info_flow']:.1f} | Chunks: {node['chunks']}")
            if node["stop_reason"] != "none":
                print(f"{indent}    `-- Stopped: {node['stop_reason']}")

        print("=" * 60)

    # =========================================================================
    # STREAMING BRIDGE (Finding 8 fix)
    # =========================================================================

    def recursive_solve_stream(self, query: str, depth: int = 0,
                               path: tuple = (),
                               max_depth: int = 5):
        """Generator version of recursive_solve -- yields ReasoningEvent at each step.

        Delegates to the streaming module's implementation, which mirrors
        recursive_solve but yields events at each step for real-time observation.

        Usage:
            for event in rlm.recursive_solve_stream("question"):
                print(f"[{event.type}] depth={event.depth}")
        """
        from .streaming import recursive_solve_stream
        return recursive_solve_stream(self, query, depth, path, max_depth)

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

        # Dependency structure analysis — re-select if chunks too correlated
        if len(selected_ids) > 1:
            dep_analysis = self.analyze_dependency_structure(
                selected_ids, similarity_threshold=0.7
            )
            if dep_analysis["spectral_gap"] < 0.1:
                # Chunks are too correlated, re-select with stricter threshold
                selected = self.select_chunks_for_subcall(
                    query=query, max_chunks=3, sim_threshold=0.85
                )
                selected_ids = [c.id for c in selected]
                selected_text = "\n".join([c.text for c in selected])

        # Retrieve relevant memories from φ-spiral memory
        spiral_mem = self._get_spiral_memory()
        if spiral_mem and spiral_mem.memories:
            query_emb = get_embedding(query, self.embedder)
            memories = spiral_mem.retrieve(query_emb, k=2)
            if memories:
                memory_text = "\n".join([m.text for m in memories])
                selected_text = f"{selected_text}\n\n[Prior Knowledge]:\n{memory_text}"

        # Compute logdet of selection
        sub_gram = self.phi_gram.submatrix(selected_ids)
        logdet_selected = sub_gram.log_determinant

        # Step 2: Build prompt (with φ-attention injection if available)
        phi_attn = self._get_phi_attention()
        use_phi = (phi_attn is not None and
                   (self.evolution_state is None or
                    getattr(self.evolution_state, 'phi_attention_enabled', True)))

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

        # Inject depth-appropriate agent persona (claude-code-templates)
        agent_router = self._get_agent_router()
        if agent_router:
            prompt = agent_router.inject_agent_persona(prompt, depth)

        # Inject relevant skill context if available
        skill_loader = self._get_skill_loader()
        if skill_loader and skill_loader.skill_count > 0:
            relevant_skills = skill_loader.get_relevant_skills(query, max_skills=1)
            if relevant_skills:
                prompt += f"\n\n[Relevant Skill]:\n{relevant_skills[0]}"

        # Inject tool descriptions if tools are available
        tool_registry = self._get_tool_registry()
        if tool_registry and tool_registry.list_tools():
            prompt += f"\n\n{tool_registry.get_tool_descriptions()}"

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

        # Handle tool calls from LLM response
        tool_call = response.get("tool_call") if isinstance(response, dict) else None
        if tool_call and tool_registry:
            tool_name = tool_call.get("name", "")
            tool_params = tool_call.get("params", {})
            tool_cost = tool_registry.get_budget_cost(tool_name)
            if tool_cost <= budget:
                tool_result = tool_registry.execute(tool_name, tool_params)
                if tool_result.success:
                    answer = (f"{answer}\n\n"
                              f"[Tool Output ({tool_result.tool_name})]:\n"
                              f"{tool_result.output}")
                else:
                    logger.warning(f"Tool {tool_name} failed: {tool_result.error}")

        # Step 4: QEC verification
        revised_conf, verifier_results = self.run_qec_verification(answer, selected_text, budget // 2)
        confidence = (raw_confidence + revised_conf) / 2

        # Update confidence history
        self.confidence_history.append(confidence)

        # Step 4: Update spectral flow (token-based + semantic)
        answer_tokens = simple_tokenize(answer)
        new_info = self.compute_info_units(answer_tokens)
        # Supplement with semantic info flow when embeddings available
        try:
            answer_emb = get_embedding(answer[:500], self.embedder)
            semantic_info = self.compute_semantic_info_units(answer_emb)
            new_info = (new_info + semantic_info) / 2  # Hybrid metric
        except Exception:
            pass  # Fall back to token-only info flow
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
            # Store in φ-spiral memory (only at root depth)
            if depth == 0 and spiral_mem:
                try:
                    result_emb = get_embedding(answer[:500], self.embedder)
                    spiral_mem.store(
                        text=f"Q: {query[:200]}\nA: {answer[:300]}",
                        embedding=result_emb,
                        importance=confidence,
                    )
                except Exception:
                    pass
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

        # Apply φ-sparse pruning if available
        phi_sparse = self._get_phi_sparse()
        use_sparse = (phi_sparse is not None and
                      (self.evolution_state is None or
                       getattr(self.evolution_state, 'sparse_pruning_enabled', True)))

        if use_sparse and len(subquestions) > 1:
            scores = phi_sparse.score_subquestions(query, subquestions, answer)
            subquestions = phi_sparse.adaptive_prune(subquestions, scores, confidence)

        # Process subquestions (parallel if enabled)
        branch_factor = 3
        if self.evolution_state and hasattr(self.evolution_state, 'branch_factor'):
            branch_factor = self.evolution_state.branch_factor
        subquestions_limited = subquestions[:branch_factor]

        if getattr(self, 'parallel_enabled', False) and len(subquestions_limited) > 1:
            # Parallel processing
            sub_results = self._process_subquestions_parallel(
                subquestions_limited, depth, path, max_depth
            )
        else:
            # Sequential processing
            sub_results = []
            for i, subq in enumerate(subquestions_limited):
                sub_result = self.recursive_solve(subq, depth + 1, path + (i,), max_depth)
                sub_results.append(sub_result)

        # Aggregate sub-results with torsion correction
        aggregated = self.aggregate_results(sub_results)

        # Combine with current answer
        final_answer = f"{answer}\n\nSub-analysis: {aggregated.value}"
        final_conf = (confidence + aggregated.confidence) / 2

        # Store result in φ-spiral memory (only at root depth)
        if depth == 0 and spiral_mem:
            try:
                result_emb = get_embedding(final_answer[:500], self.embedder)
                spiral_mem.store(
                    text=f"Q: {query[:200]}\nA: {final_answer[:300]}",
                    embedding=result_emb,
                    importance=final_conf,
                )
            except Exception:
                pass  # Memory storage is best-effort

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

    def adversarial_challenge(self, query: str, result: 'SubCallResult') -> 'SubCallResult':
        """Run PhiCritic as an independent adversarial pass on a completed result.

        Unlike QEC (which runs during recursion), this challenges the final answer
        after recursive_solve completes. If the critic finds significant flaws,
        confidence is downgraded.

        Args:
            query: The original query.
            result: The SubCallResult from recursive_solve.

        Returns:
            Updated SubCallResult with critic metadata and adjusted confidence.
        """
        try:
            from .phi_critic import PhiCritic
            critic = PhiCritic(self, critique_depth=2)
            critique = critic.critique(query, str(result.value))
            penalty = critique.flaws_found * 0.05
            adjusted_conf = max(0.1, result.confidence - penalty)
            return SubCallResult(
                value=result.value,
                confidence=adjusted_conf,
                metadata={
                    **result.metadata,
                    "adversarial_critique": critique.critique_text,
                    "adversarial_score": critique.quality_score,
                    "flaws_found": critique.flaws_found,
                    "confidence_before_challenge": result.confidence,
                }
            )
        except Exception as e:
            logger.warning(f"Adversarial challenge failed: {e}")
            return result

    def workflow_solve(self, query: str, max_depth: int = 7,
                       confidence_threshold: float = 0.5) -> dict:
        """Convenience method to run the full superpowers workflow pipeline.

        Wraps SuperpowersOrchestrator so callers don't need to import it separately.

        Args:
            query: The query to solve.
            max_depth: Maximum recursion depth (default 7 for full E8 hierarchy).
            confidence_threshold: Minimum confidence to pass workflow gates.

        Returns:
            Dict with 'result', 'workflow_trace', and 'summary' keys.
        """
        try:
            from .workflow_orchestrator import SuperpowersOrchestrator
            orchestrator = SuperpowersOrchestrator(
                self, confidence_threshold=confidence_threshold
            )
            result = orchestrator.orchestrated_solve(query, max_depth=max_depth)
            return {
                "result": result,
                "workflow_trace": orchestrator.workflow_trace,
                "summary": orchestrator.get_workflow_summary(),
            }
        except ImportError:
            logger.warning("workflow_orchestrator not available, falling back to recursive_solve")
            result = self.recursive_solve(query, max_depth=max_depth)
            return {"result": result, "workflow_trace": [], "summary": ""}

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
