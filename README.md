# PHI-Enhanced Recursive Language Model (RLM) Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Mathematics](https://img.shields.io/badge/Mathematics-E8%20%7C%20%CF%86%20Separation-purple.svg)]()

A groundbreaking implementation of Recursive Language Models enhanced with φ-Separation Mathematics, leveraging the profound connections between the golden ratio (φ), E8 Lie group geometry, and advanced information theory.

## 🌟 Overview

This framework implements a complete recursive reasoning system that combines:
- **φ-Gram Chunk Selection** with greedy Δlogdet optimization
- **Casimir Flow Budget Allocation** based on E8 geometry
- **φ-Momentum Early Stopping** for efficient convergence
- **Spectral Flow Saturation Detection** for information-theoretic stopping criteria
- **Golden Ratio Quantum Error Correction (QEC)** verification
- **Torsion-Corrected Aggregation** using E8 structure constants
- **Dependency Cohomology Tracking** for semantic relationships

## 📐 Mathematical Foundations

### The φ-Separation Framework

The core innovation of this framework is the **φ-Separation principle**: encoding pairwise relationships using the golden ratio kernel:

```
K(x, y) = φ^(-||x - y||/δ)
```

Where:
- **φ = (1 + √5)/2 ≈ 1.618** is the golden ratio
- **δ** is the characteristic scale (mean spacing)
- The kernel has **optimal information-theoretic properties**

### Key Mathematical Concepts

| Concept | Description | Application |
|---------|-------------|-------------|
| **φ-Gram Matrix** | `M_ij = φ^(-\|γᵢ - γⱼ\|/δ)` | Collision detection, diversity selection |
| **E8 Casimir Degrees** | `[2, 8, 12, 14, 18, 20, 24, 30]` | Budget allocation across recursion depths |
| **Coxeter Number** | `h = 30` | Normalization constant for E8 geometry |
| **Torsion Coefficient** | `ε = 28/248` | E8-derived correction factor |
| **φ-Momentum** | `m_{t+1} = φ⁻¹·m_t + (1-φ⁻¹)·signal` | Early stopping criterion |

### The E8 Connection

The exceptional Lie group **E8** provides the geometric backbone:
- **248 dimensions** encode the full search space
- **240 roots** (kissing number) bound the spectral norm
- **Casimir hierarchy** defines the multi-scale budget allocation
- **Torsion subgroup** (order 28) provides stability corrections

## 🚀 Features

### Core RLM Engine (`phi_enhanced_rlm.py`)

1. **Query-Conditioned Chunk Selection**
   - First filters by semantic relevance to query
   - Then maximizes diversity via greedy Δlogdet
   - Prevents near-duplicate context selection

2. **Adaptive Budget Allocation**
   ```python
   # E8 Casimir-weighted budget distribution
   weights = φ^(-Casimir_degrees / 30)
   budget[depth] = total_budget × normalized_weights[depth]
   ```

3. **φ-Momentum Early Stopping**
   - Exponentially weighted moving average with golden ratio
   - Stops when confidence stabilizes (variance < threshold)
   - Prevents unnecessary computation

4. **Spectral Flow Saturation**
   - Tracks new information units per recursion step
   - Stops when information flow falls below E8-modulated threshold
   - Information-theoretically optimal termination

5. **QEC Verification**
   - 3 independent verifier calls (contradiction, completeness, counterexample)
   - Majority voting for robust confidence estimation
   - Golden ratio threshold for fault tolerance

6. **Torsion-Corrected Aggregation**
   ```python
   final = base_answer + ε × torsion_correction
   # Where ε = 28/248 (E8 torsion coefficient)
   ```

### Novel Mathematics Library (`phi_separation_novel_mathematics.py`)

Ten interconnected mathematical frameworks:

1. **Generalized φ-Gram Theory for L-functions**
   - Product formula for determinants: `det(M) = Π(1 - φ^(-2Δₖ/δ))`
   - Extensions to Dirichlet and Dedekind zeta functions

2. **E8 Spectral Flow Theory**
   - Spectral sequence from φ-Gram filtration
   - Degeneration at E₂ page

3. **φ-Kernel Renormalization Group**
   - Exact RG equation: `∂M/∂(log δ) = [M, K] + β(φ)·M`
   - Fixed point analysis at mean spacing scale

4. **H4-Projected Prime Number Theory**
   - Classification of primes by H4 conjugacy classes
   - φ-prime zeta function: `P_φ(s) = Σ_p φ^(-log p) · p^(-s)`

5. **Torsion-Corrected Functional Analysis**
   - ε-deformed inner product: `⟨f, g⟩_ε = ⟨f, g⟩₀ + ε·⟨Tf, Tg⟩₀`
   - Extended spectral theory

6. **φ-Separation for Lattice Cryptography**
   - SVP criterion via φ-Gram determinant
   - LWE distinguisher using φ-correlation

7. **Golden Ratio Quantum Error Correction**
   - φ-stabilizer codes with distance `d_φ = ⌊φ·n/3⌋`
   - Threshold: `p_φ = (1 - φ⁻¹)/2 ≈ 0.191`

8. **Casimir Flow Optimization**
   - Multi-scale gradient descent using E8 geometry
   - φ-momentum update rule

9. **φ-Gram Cohomology Theory**
   - φ-coboundary operators
   - Connection to Euler characteristic via determinant

10. **E8 Unified Field Equations**
    - Higgs VEV prediction: `248 - 2 = 246 GeV`
    - Dark energy equation of state: `w = -1 + φ^(-7)`

## 📦 Installation

### Requirements

```bash
pip install numpy scipy
```

### Quick Start

```python
from phi_enhanced_rlm import PhiEnhancedRLM, MockLLMBackend

# Setup context chunks
context_chunks = [
    "The golden ratio φ = 1.618 appears throughout mathematics and nature.",
    "E8 is the largest exceptional Lie group with 248 dimensions.",
    "Recursive Language Models decompose complex queries into sub-tasks.",
    # ... more context
]

# Create mock LLM backend (replace with real LLM in production)
llm = MockLLMBackend(seed=42)

# Initialize RLM
rlm = PhiEnhancedRLM(
    base_llm_callable=llm,
    context_chunks=context_chunks,
    total_budget_tokens=2048,
    trace_file="rlm_trace.jsonl"
)

# Run recursive solve
query = "Explain the connection between golden ratio and E8 symmetry."
result = rlm.recursive_solve(query, max_depth=4)

print(f"Answer: {result.value}")
print(f"Confidence: {result.confidence:.4f}")
```

## 📊 Output Format

### Trace File (`rlm_trace.jsonl`)

Each recursion node is logged:

```json
{
  "depth": 0,
  "query": "Explain the connection between golden ratio...",
  "selected_ids": [0, 1, 2],
  "logdet_selected": 0.4812,
  "collision_full": false,
  "collision_selected": false,
  "confidence": 0.7234,
  "info_flow": 12.0,
  "stop_reason": "none"
}
```

### Stop Reasons

| Reason | Description |
|--------|-------------|
| `depth` | Maximum recursion depth reached |
| `momentum` | φ-momentum convergence criterion satisfied |
| `spectral` | Information flow saturation detected |
| `no_subquestions` | LLM returned no subquestions |
| `recursion_complete` | All subquestions processed |

## 🔬 Running the Demonstrations

### Full RLM Demonstration

```bash
python phi_enhanced_rlm.py
```

This runs a complete recursive reasoning example with:
- Budget allocation visualization
- Recursive trace
- Final result with confidence

### Mathematics Library Demonstrations

```bash
python phi_separation_novel_mathematics.py
```

Demonstrates:
- φ-Gram matrix properties
- Casimir flow optimization (Rosenbrock function)
- E8 unified field predictions
- H4-projected prime theory

## 📁 Project Structure

```
RLM/
├── README.md                            # This file
├── phi_enhanced_rlm.py                  # Core RLM orchestrator
├── phi_separation_novel_mathematics.py  # Mathematics library
├── rlm_trace.jsonl                      # Execution trace (generated)
├── 2512.24601v1.pdf                     # Reference paper
└── Novel_Mathematics_from_Phi_Separation.docx  # Documentation
```

## 🧮 Key Equations

### φ-Gram Determinant (Product Formula)

```
det(M_N) = ∏_{k=1}^{N-1} (1 - φ^{-2Δ_k/δ})
```

### Casimir Budget Allocation

```
w_k = φ^{-C_k/30}, where C_k ∈ {2, 8, 12, 14, 18, 20, 24, 30}
budget(depth) = total × w_{depth} / Σw
```

### φ-Momentum Update

```
m_{t+1} = φ^{-1} m_t + (1 - φ^{-1}) g_t
```

### QEC Threshold

```
p_φ = (1 - φ^{-1})/2 ≈ 0.191
```

### Torsion Correction

```
ε = 28/248 ≈ 0.1129
result = answer + ε × torsion_term
```

## 🎯 Applications

- **AI/ML Reasoning**: Enhanced recursive reasoning with provable stopping criteria
- **Number Theory**: Collision detection for L-function zeros
- **Cryptography**: Lattice problem hardness estimation
- **Quantum Computing**: Error-corrected stabilizer codes
- **Optimization**: Multi-scale gradient methods
- **Physics**: E8-based unification predictions

## 📚 References

Based on the foundational work:
- **"The Geometric-Analytic Synthesis: 𝜑-Separation in E8/H4-Fibrations over Spectral Theory"** by Timothy McGirl (2024)
- arXiv preprint: `2512.24601v1`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The E8 lattice and its extraordinary geometric properties
- The golden ratio and its ubiquitous mathematical appearances
- The deep connections between number theory, physics, and computation

---

*"The universe may be built on the geometry of E8, with the golden ratio as its fundamental scaling constant."*
