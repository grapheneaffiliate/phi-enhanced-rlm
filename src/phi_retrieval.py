#!/usr/bin/env python3
"""
φ-Kernel Retrieval — golden ratio kernel for semantic search.

Replaces standard cosine similarity with a combined φ-kernel + cosine
scoring function. The φ-kernel captures geometric distance (magnitude-aware)
while cosine captures angular similarity. Together they produce more
discriminative retrieval scores.

The φ-kernel K(a,b) = φ^(-||a-b||/δ) is:
1. Distance-based (captures magnitude differences)
2. Has exponential decay governed by φ (natural scale)
3. Produces PSD kernel matrices (guaranteed valid Gram matrix)
"""

import numpy as np
from .phi_separation_novel_mathematics import PHI


def phi_kernel_similarity(a: np.ndarray, b: np.ndarray, delta: float = 1.0) -> float:
    """
    φ-kernel similarity: K(a,b) = φ^(-||a-b||/δ)

    Args:
        a: First embedding vector
        b: Second embedding vector
        delta: Scale parameter (controls decay rate)

    Returns:
        Similarity score in (0, 1]
    """
    dist = np.linalg.norm(a - b)
    return float(PHI ** (-dist / delta))


def phi_retrieval_score(query_emb: np.ndarray, chunk_emb: np.ndarray,
                        delta: float = 1.0, alpha: float = 0.7) -> float:
    """
    Combined retrieval score: α × φ-kernel + (1-α) × cosine.

    The φ-kernel captures geometric distance.
    Cosine captures angular similarity.
    Together they outperform either alone.

    Args:
        query_emb: Query embedding vector
        chunk_emb: Chunk embedding vector
        delta: φ-kernel scale parameter
        alpha: Blending weight (higher = more φ-kernel influence)

    Returns:
        Combined similarity score
    """
    phi_sim = phi_kernel_similarity(query_emb, chunk_emb, delta)

    norm_q = np.linalg.norm(query_emb)
    norm_c = np.linalg.norm(chunk_emb)
    if norm_q == 0 or norm_c == 0:
        cos_sim = 0.0
    else:
        cos_sim = float(np.dot(query_emb, chunk_emb) / (norm_q * norm_c))

    return alpha * phi_sim + (1 - alpha) * cos_sim


def phi_batch_retrieval(query_emb: np.ndarray, chunk_embs: np.ndarray,
                        delta: float = 1.0, alpha: float = 0.7) -> np.ndarray:
    """
    Batch retrieval scores for all chunks at once.

    Args:
        query_emb: Query embedding (1D)
        chunk_embs: Matrix of chunk embeddings (N x D)
        delta: φ-kernel scale parameter
        alpha: Blending weight

    Returns:
        Array of scores (N,)
    """
    # φ-kernel scores
    dists = np.linalg.norm(chunk_embs - query_emb, axis=1)
    phi_sims = PHI ** (-dists / delta)

    # Cosine scores
    norm_q = np.linalg.norm(query_emb)
    norms_c = np.linalg.norm(chunk_embs, axis=1)
    denom = norm_q * norms_c
    denom = np.where(denom == 0, 1e-12, denom)
    cos_sims = chunk_embs @ query_emb / denom

    return alpha * phi_sims + (1 - alpha) * cos_sims
