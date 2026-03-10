# PHI-Enhanced Recursive Language Model (RLM) v4.0

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18377963.svg)](https://doi.org/10.5281/zenodo.18377963)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/grapheneaffiliate/phi-enhanced-rlm/actions/workflows/test.yml/badge.svg)](https://github.com/grapheneaffiliate/phi-enhanced-rlm/actions)
[![API](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)

A self-evolving AI reasoning system built on golden ratio mathematics. It breaks hard problems into smaller pieces, solves each piece with a specialized agent, checks its own work, and gets smarter over time -- all governed by the golden ratio (phi = 1.618...) and E8 Lie group geometry.

**No AI experience required.** Follow the setup below and you'll have it running in under 2 minutes.

---

## Setup (2 minutes)

### What you need

- **Python 3.9 or newer** -- [download from python.org](https://python.org) if you don't have it
- **Git** -- [download from git-scm.com](https://git-scm.com) if you don't have it

### Step 1: Download and install

```bash
git clone https://github.com/grapheneaffiliate/phi-enhanced-rlm.git
cd phi-enhanced-rlm
bash setup.sh
```

That's it. The setup script installs everything automatically and verifies it works.

**Windows users:** If `bash` isn't available, run these instead:

```bash
pip install -e .
copy .env.template .env
python quickstart.py
```

### Step 2: See it work

```bash
python quickstart.py
```

This runs a full demonstration in about 10 seconds -- no API key needed. You'll see the recursive reasoning engine solve a question, display its reasoning tree, show how it allocates computational budget using E8 geometry, and select a meta-strategy.

### Step 3 (optional): Connect a real AI model

The system works out of the box in demo mode using simulated responses. To connect it to a real AI model for actual reasoning:

1. Get a free API key from [openrouter.ai/keys](https://openrouter.ai/keys)
2. Open the `.env` file in any text editor
3. Replace `your_openrouter_api_key_here` with your actual key
4. Save the file

Now when you run the system, it will use a real AI model instead of simulated responses.

---

## What this system does

Traditional AI gives you one answer in one step. This system works differently:

1. **Decomposes** your question into sub-questions
2. **Routes** each sub-question to a depth-specialized agent (8 agents mapped to E8 Casimir degrees)
3. **Selects** the most relevant context using phi-geometric diversity optimization
4. **Solves** each sub-question recursively (sub-questions can have their own sub-questions)
5. **Verifies** every answer using 3 independent checks (QEC -- Quantum Error Correction inspired)
6. **Combines** results using torsion-corrected aggregation (minority viewpoints are preserved)
7. **Stops** when the reasoning has converged (phi-momentum early stopping)
8. **Learns** from its performance and improves its own parameters over time

### Multi-Agent Architecture (v4.0)

Each recursion depth is served by a different specialized agent:

| Depth | Casimir | Agent | Role |
|-------|---------|-------|------|
| 0 | 2 | research-orchestrator | Route and plan |
| 1 | 8 | query-clarifier | Refine the question |
| 2 | 12 | research-analyst | Primary analysis |
| 3 | 14 | technical-researcher | Deep technical dive |
| 4 | 18 | academic-researcher | Academic rigor |
| 5 | 20 | fact-checker | QEC verification |
| 6 | 24 | research-synthesizer | Torsion-corrected synthesis |
| 7 | 30 | research-orchestrator | Final integration |

---

## How to use it

### Ask a question (Python)

```python
from src import PhiEnhancedRLM
from src.phi_enhanced_rlm import MockLLMBackend

# Your knowledge base (the system searches these for relevant context)
context = [
    "The golden ratio appears throughout nature and mathematics.",
    "E8 is the largest exceptional Lie group with 248 dimensions.",
    "Recursive models decompose complex queries into sub-tasks.",
]

# Create the model
rlm = PhiEnhancedRLM(
    base_llm_callable=MockLLMBackend(seed=42),
    context_chunks=context,
)

# Ask anything
result = rlm.recursive_solve("How does phi relate to E8 symmetry?", max_depth=4)

print(f"Answer: {result.value}")
print(f"Confidence: {result.confidence:.1%}")
```

### Let it choose its own strategy (Meta-Recursion)

```python
from src import MetaRecursiveRLM

meta = MetaRecursiveRLM(rlm)
result = meta.meta_solve("Analyze the implications of phi-separation for cryptography")

# It automatically detected this is an analytical question
# and chose the deep_analytical strategy (6 levels deep, phi-attention ON)
print(f"Strategy: {result.metadata['meta_strategy']}")
```

Five strategies are available:

| Strategy | Max Depth | Best For |
|----------|-----------|----------|
| `deep_analytical` | 6 | Complex analytical questions |
| `wide_exploratory` | 3 | Open-ended, broad questions |
| `spiral_convergent` | 5 | General-purpose (default) |
| `quick_factual` | 2 | Simple factual lookups |
| `deep_research` | 7 | Full E8 hierarchy with agent specialization |

### Watch it evolve

```bash
python -m src.evolution_loop --generations 10
```

This runs the system through 10 generations of self-improvement. Each generation, it adjusts its own parameters (budget allocation, stopping thresholds, pruning ratios, agent assignments) based on how well it performed. The learning rate follows phi^(-n) -- it adapts quickly at first, then makes increasingly fine adjustments.

### Run all tests

```bash
python -m pytest tests/ -v
```

### Interactive chat

```bash
python cli/chat.py
```

Opens an interactive terminal where you can chat with the system. Supports special commands:

| Command | What it does |
|---------|-------------|
| `/depth 5` | Set maximum recursion depth |
| `/file report.txt` | Load and analyze a text file |
| `/pdf document.pdf` | Analyze a PDF document |
| `/url https://...` | Fetch and analyze a web page |
| `/repo owner/repo` | Analyze a GitHub repository |
| `/compare A B` | Compare two sources side by side |
| `/export results.md` | Save the last analysis as markdown |
| `/stream on` | Enable real-time streaming output |
| `/history` | Show past queries |
| `/help` | Show all commands |
| `/quit` | Exit |

### REST API

```bash
pip install fastapi uvicorn
python -m uvicorn api.server:app --reload --port 8000
```

Opens a web API at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze text with recursive reasoning |
| `/chat` | POST | Chat with context and memory |
| `/compare` | POST | Compare two sources |
| `/status` | GET | System health and configuration |
| `/history` | GET | Past query history |
| `/evolution/status` | GET | Evolution engine state |
| `/memory/stats` | GET | Spiral memory statistics |

---

## Project Structure

```
phi-enhanced-rlm/
|
+-- quickstart.py              <- Run this first
+-- setup.sh                   <- One-click installation
+-- CLAUDE.md                  <- Project intelligence for Claude Code
+-- .env.template              <- Configuration template (copy to .env)
|
+-- src/                       <- Core engine (22 modules)
|   +-- phi_enhanced_rlm.py        Main recursive engine
|   |                               - recursive_solve() core loop
|   |                               - phi-Gram chunk selection
|   |                               - QEC verification (3 checks)
|   |                               - Torsion-corrected aggregation
|   |
|   +-- phi_separation_novel_mathematics.py
|   |                               Mathematical foundations
|   |                               - phi-Gram matrix theory
|   |                               - E8 spectral flow
|   |                               - Casimir degrees: [2,8,12,14,18,20,24,30]
|   |                               - 10 interconnected math frameworks
|   |
|   +-- agent_router.py            Multi-agent depth routing (v4.0)
|   |                               - 8 specialized agent personas
|   |                               - Depth-to-agent mapping via E8 Casimir
|   |                               - Evolvable agent assignments
|   |
|   +-- skill_loader.py            Skill file integration (v4.0)
|   |                               - Loads domain skills from .claude/skills/
|   |                               - Keyword-based retrieval
|   |
|   +-- evolution.py                Self-evolution engine
|   |                               - EvolutionState learnable parameters
|   |                               - phi-scaled learning rate (phi^-n)
|   |                               - Budget, stopping, agent mutations
|   |
|   +-- evolution_loop.py           Evolution training loop
|   +-- meta_recursion.py           Meta-reasoning (5 strategies)
|   +-- phi_attention.py            phi-geometric attention injection
|   +-- phi_sparse_reasoning.py     phi-ratio branch pruning
|   +-- phi_memory.py               Golden spiral persistent memory
|   +-- session_memory.py           Cross-session learning
|   +-- phi_retrieval.py            phi-kernel similarity search
|   +-- phi_bayesian.py             phi-Bayesian optimization
|   +-- ensemble_backend.py         Multi-model routing
|   +-- streaming.py                Real-time streaming output
|   +-- embeddings.py               Multi-provider embeddings
|   +-- cache.py                    SQLite embedding cache
|   +-- extractors.py               PDF, DOCX, web, code extraction
|   +-- openrouter_backend.py       OpenRouter API client
|   +-- async_backend.py            Async LLM backend
|   +-- vector_store.py             ChromaDB vector store
|   +-- progress.py                 Terminal progress display
|
+-- tests/                     <- Test suite
|   +-- test_upgrades.py           Core engine tests
|   +-- test_evolution.py          Evolution tests
|   +-- test_benchmarks.py         Benchmark tests
|   +-- test_real_llm.py           Real API integration tests
|
+-- benchmarks/                <- Evaluation datasets
|   +-- runner.py                  phi-RLM vs vanilla comparison
|   +-- gsm8k_sample.json         Math word problems
|   +-- arc_sample.json           Reasoning questions
|
+-- cli/                       <- Command-line tools
|   +-- chat.py                    Interactive chat
|   +-- repo_analyzer.py           GitHub repo analysis
|   +-- run_rlm.py                 Quick query runner
|   +-- validate_rlm.py            System validation
|
+-- api/server.py              <- REST API (FastAPI)
+-- web/                       <- Web interfaces
|   +-- index.html                 Documentation page
|   +-- dashboard.html             Evolution dashboard
|
+-- docs/                      <- Documentation
    +-- UNIFICATION.md             phi-RLM + phi-GEH + PROMETHEUS
    +-- 2512.24601v1.pdf           Reference paper
```

---

## Key Concepts

### The Golden Ratio (phi = 1.618...)

A number found everywhere in nature -- sunflower spirals, galaxy arms, DNA molecules. This project uses it as the governing constant for AI reasoning because it's the "most irrational" number, meaning patterns built on it have the least repetition and maximum information coverage.

### E8 Lie Group

A mathematical structure with 248 dimensions encoding deep symmetries. This project uses its "Casimir degrees" -- [2, 8, 12, 14, 18, 20, 24, 30] -- to decide how much computational budget each reasoning depth gets and which specialized agent handles it.

### Recursive Reasoning

Instead of answering in one shot, the system breaks questions into sub-questions, answers each separately, then combines results. Sub-questions can themselves be broken down further, creating a tree of reasoning.

### Multi-Agent Specialization (v4.0)

Each recursion depth is handled by a different specialized agent persona. The research-orchestrator plans at depth 0, the query-clarifier refines at depth 1, analysts and researchers investigate at depths 2-4, the fact-checker verifies at depth 5, the synthesizer combines at depth 6, and the orchestrator integrates the final result at depth 7. The evolution engine can reassign agents to optimize performance.

### Self-Evolution

After each benchmark run, the system adjusts its own parameters. The adjustment follows phi^(-n) scaling -- big changes early, increasingly subtle refinements over time.

### QEC Verification

Every answer is independently verified by 3 checks (contradiction, completeness, counterexample). When the agent router is active, verification uses the fact-checker agent persona. High confidence only when 2 out of 3 pass.

### Meta-Recursion

Before solving, the system classifies your question (analytical? factual? creative? research?) and picks the best strategy. If the first strategy fails, it tries an alternative.

---

## Mathematical Foundations

```
phi-Gram Kernel:          K(x,y) = phi^(-||x-y||/delta)
Casimir Budget:           weight(d) = phi^(-C_d/30), C in {2,8,12,14,18,20,24,30}
phi-Momentum Stopping:    m(t+1) = phi^(-1) * m(t) + (1-phi^(-1)) * signal(t)
Torsion Correction:       result = base + (28/248) * minority_view
QEC Threshold:            p_phi = (1-phi^(-1))/2 ~ 0.191
Evolution Learning Rate:  lr(n) = phi^(-n)
```

Full details in `src/phi_separation_novel_mathematics.py` -- 10 interconnected mathematical frameworks including phi-Gram determinant theory, E8 spectral flow, renormalization group, H4-projected prime theory, lattice cryptography, quantum error correction codes, Casimir flow optimization, cohomology theory, and unified field equations.

---

## Optional Dependencies

The core runs with just `numpy`, `scipy`, and `openai`. Install extras for more features:

```bash
pip install -e ".[full]"              # Everything
pip install sentence-transformers     # Local embeddings (free)
pip install chromadb                  # Vector database
pip install fastapi uvicorn           # REST API
pip install PyMuPDF                   # PDF support
pip install python-docx               # Word documents
pip install rich                      # Beautiful terminal output
```

---

## Troubleshooting

**"No embedding provider available, using mock embeddings"** -- Normal. Works fine with mock embeddings. For better results, install `sentence-transformers` or add your OpenRouter API key.

**"ChromaDB not installed"** -- Optional. Only needed for large document collections.

**Import errors** -- Always run from the project root directory. Use `from src import ...`.

**Windows encoding issues** -- Set: `set PYTHONIOENCODING=utf-8`

---

## License

MIT License -- see [LICENSE](LICENSE).

## Author

Timothy McGirl -- Geometric Standard Model (GSM) Framework

*"The universe may be built on the geometry of E8, with the golden ratio as its fundamental scaling constant."*
