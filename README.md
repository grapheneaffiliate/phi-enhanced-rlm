# PHI-Enhanced Recursive Language Model (RLM) v3.0

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18377963.svg)](https://doi.org/10.5281/zenodo.18377963)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/grapheneaffiliate/phi-enhanced-rlm/actions/workflows/test.yml/badge.svg)](https://github.com/grapheneaffiliate/phi-enhanced-rlm/actions)
[![API](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)

A self-evolving recursive language model framework built on novel phi-Separation Mathematics. Combines phi-geometric attention, spiral memory, sparse reasoning pruning, and meta-recursion into a unified system that improves itself across sessions.

## Features

- **Streaming Output** -- Real-time response streaming
- **PDF/DOCX Support** -- Analyze documents directly
- **Smart Web Extraction** -- Trafilatura for clean article extraction
- **SQLite Embedding Cache** -- Persistent cache survives restarts
- **Vector Store** -- ChromaDB-backed retrieval for large-scale data analysis
- **REST API** -- FastAPI with OpenAPI docs
- **Conversation Memory** -- Stateful chat with context
- **Parallel Processing** -- Process subquestions concurrently
- **Confidence Visualization** -- See the reasoning tree
- **Comparison Mode** -- Compare repos, URLs, or documents
- **Export Reports** -- Save analyses as markdown
- **Rich Progress** -- Beautiful terminal UI
- **Self-Evolution** -- phi-governed parameter optimization across generations
- **phi-Geometric Attention** -- Golden ratio confidence weighting
- **phi-Sparse Reasoning** -- Intelligent branch pruning at phi^-1 threshold
- **phi-Spiral Memory** -- Golden spiral topology with impedance funneling
- **Meta-Recursion** -- Model reasons about its own reasoning process
- **Session Memory** -- Cross-session learning and strategy optimization
- **Benchmarks** -- GSM8K and ARC evaluation with phi-RLM vs vanilla comparison

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/grapheneaffiliate/phi-enhanced-rlm.git
cd phi-enhanced-rlm
pip install -e ".[full,dev]"
```

### 2. Configure

```bash
cp .env.template .env
# Edit .env: OPENROUTER_API_KEY=sk-or-v1-your-key
```

### 3. Run

```bash
# Interactive chat
phi-rlm
# Or: python cli/chat.py

# REST API server
python api/server.py
# Visit: http://localhost:8000/docs

# Run self-evolution loop
phi-evolve
# Or: python -m src.evolution_loop

# Analyze a repository
phi-analyze
# Or: python cli/repo_analyzer.py
```

---

## Project Structure

```
phi-enhanced-rlm/
├── README.md
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Package config (v3.0.0)
├── .env.template                    # API key template
│
├── src/                             # Core library
│   ├── __init__.py                      # Public API exports
│   ├── phi_enhanced_rlm.py              # Main RLM orchestrator
│   ├── phi_separation_novel_mathematics.py  # 10 novel math frameworks
│   ├── embeddings.py                    # Multi-provider embeddings
│   ├── cache.py                         # SQLite embedding cache
│   ├── vector_store.py                  # ChromaDB vector store
│   ├── extractors.py                    # PDF/DOCX/web/code extractors
│   ├── progress.py                      # Rich progress display
│   ├── openrouter_backend.py            # OpenRouter LLM backend
│   ├── async_backend.py                 # Async LLM operations
│   ├── evolution.py                     # phi-governed self-evolution engine
│   ├── evolution_loop.py                # Evolution main loop
│   ├── phi_attention.py                 # phi-geometric attention injection
│   ├── phi_sparse_reasoning.py          # phi-sparse branch pruning
│   ├── phi_memory.py                    # phi-spiral memory
│   ├── session_memory.py                # Cross-session persistent memory
│   └── meta_recursion.py                # Meta-recursion strategies
│
├── cli/                             # Command-line tools
│   ├── chat.py                          # Interactive chat v2.0
│   ├── run_rlm.py                       # Simple query runner
│   ├── repo_analyzer.py                 # GitHub/URL analyzer
│   └── validate_rlm.py                  # System validation
│
├── api/                             # REST API
│   └── server.py                        # FastAPI server
│
├── web/                             # Web interface
│   └── index.html                       # Standalone web UI
│
├── tests/                           # Test suite
│   ├── test_upgrades.py                 # Core feature tests
│   ├── test_evolution.py                # Evolution engine tests
│   ├── test_benchmarks.py               # Benchmark validation
│   └── test_real_llm.py                 # LLM integration tests
│
├── benchmarks/                      # Benchmark datasets & runner
│   ├── runner.py                        # phi-RLM vs vanilla comparison
│   ├── gsm8k_sample.json               # Grade School Math 8K sample
│   ├── arc_sample.json                  # ARC reasoning sample
│   └── results/                         # Benchmark output
│
├── docs/                            # Documentation
│   ├── 2512.24601v1.pdf                 # Research paper
│   ├── Novel_Mathematics_from_Phi_Separation.docx
│   └── UNIFICATION.md                   # Unification framework
│
└── sessions/                        # Session memory storage
```

---

## Interactive Chat

```bash
phi-rlm
# Or: python cli/chat.py
```

### Commands

| Command | Description |
|---------|-------------|
| `<question>` | Ask any question |
| `/repo owner/repo` | Analyze GitHub repository |
| `/url https://...` | Analyze web page |
| `/local ./path` | Analyze local directory |
| `/pdf path.pdf` | Analyze PDF document |
| `/doc path.docx` | Analyze Word document |
| `/image path.png` | Describe & analyze image |
| `/compare s1 s2` | Compare two sources |
| `/export file.md` | Export last analysis |
| `/history` | Show query history |
| `/stream on\|off` | Toggle streaming |
| `/trace` | Show reasoning tree |
| `/depth N` | Set recursion depth (0-10) |
| `/help` | Show all commands |
| `/quit` | Exit |

---

## REST API

```bash
python api/server.py
# Or: uvicorn api.server:app --reload --port 8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info |
| `GET` | `/status` | System status |
| `GET` | `/docs` | Swagger UI |
| `POST` | `/analyze` | Analyze with RLM |
| `POST` | `/chat` | Chat with memory |
| `POST` | `/compare` | Compare sources |
| `GET` | `/history` | Query history |

### Example: Analyze

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "What is E8?", "max_depth": 3}'
```

```json
{
  "answer": "E8 is the largest exceptional Lie group...",
  "confidence": 0.85,
  "depth_reached": 2,
  "chunks_used": [0, 3, 5]
}
```

---

## Self-Evolution (v3.0)

The phi-evolution engine mutates RLM parameters across generations, using phi-scaled learning rates to optimize stopping criteria, budget allocation, and confidence thresholds.

```bash
# Run the evolution loop
phi-evolve

# Or programmatically
from src.evolution import PhiEvolutionEngine, EvolutionState

engine = PhiEvolutionEngine()
state = engine.evolve(traces, current_state)
print(f"Generation {state.generation}, learning rate: {state.learning_rate:.4f}")
```

The engine:
- Evaluates reasoning traces to compute fitness metrics
- Applies phi-scaled mutations (learning rate decays by phi^-1 per generation)
- Tightens or relaxes stopping criteria based on performance
- Persists state across sessions for continuous improvement

---

## phi-Geometric Attention (v3.0)

Injects phi-structured prompts into the reasoning process with golden ratio confidence weighting.

```python
from src.phi_attention import PhiAttentionInjector, PhiConfidenceScaler

injector = PhiAttentionInjector()
enhanced_prompt = injector.inject(original_prompt, depth=2)

scaler = PhiConfidenceScaler()
weighted = scaler.scale(confidence, depth)  # Weights: phi, phi^-1, phi^-2, ...
```

---

## phi-Sparse Reasoning (v3.0)

Prunes low-value reasoning branches using a phi^-1 threshold (~61.8% retention), with diversity-aware selection to avoid collapsing to a single reasoning path.

```python
from src.phi_sparse_reasoning import PhiSparseReasoner

reasoner = PhiSparseReasoner()
pruned = reasoner.prune(branches)  # Keeps top phi^-1 fraction
```

---

## phi-Spiral Memory (v3.0)

Organizes memory using golden spiral topology with impedance funneling for retrieval.

```python
from src.phi_memory import PhiSpiralMemory
from src.session_memory import SessionMemory

# Within-session spiral memory
memory = PhiSpiralMemory()
memory.store(key, value, confidence)
retrieved = memory.retrieve(query, top_k=5)

# Cross-session persistent memory
session = SessionMemory()
session.record(query, result, strategy)
best_strategy = session.recommend_strategy(new_query)
```

---

## Meta-Recursion (v3.0)

The model reasons about its own reasoning process, selecting from multiple strategies adaptively.

```python
from src.meta_recursion import MetaRecursiveRLM, RecursionStrategy

meta = MetaRecursiveRLM(backend, chunks)
result = meta.solve(query)  # Auto-selects best strategy
```

Available strategies include depth-first, breadth-first, iterative deepening, and confidence-guided search.

---

## Document Analysis

### PDF

```bash
pip install PyMuPDF
```

```python
from src.extractors import extract_pdf_content

result = extract_pdf_content("paper.pdf")
print(f"Title: {result.title}")
print(f"Pages: {result.metadata['page_count']}")
```

### Word Documents

```bash
pip install python-docx
```

```python
from src.extractors import extract_docx_content

result = extract_docx_content("report.docx")
print(result.text)
```

---

## Embedding Cache

Embeddings are cached in SQLite for fast repeated analysis:

```python
from src.cache import SQLiteEmbeddingCache

cache = SQLiteEmbeddingCache()
cache.set("text", "model-v1", embedding_vector)
cached = cache.get("text", "model-v1")

stats = cache.get_stats()
print(f"Hits: {stats.hits}, Entries: {stats.entry_count}")
```

Cache location: `~/.cache/phi_rlm/embeddings.db`

---

## Vector Store Pipeline

For analyzing large datasets (thousands of documents), use the Vector Store + RLM pipeline:

```bash
pip install chromadb sentence-transformers
```

### Ingest Documents

```python
from src.vector_store import VectorStore, RLMPipeline

store = VectorStore("my_research")
store.add_file("paper.pdf")
store.add_directory("./documents", recursive=True)
print(f"Indexed {store.count()} chunks")
```

### Query + Analyze

```python
results = store.query("What are the main findings?", top_k=10)
for r in results:
    print(f"{r.score:.2f}: {r.text[:100]}...")

pipeline = RLMPipeline("my_research")
result = pipeline.analyze(
    "What are the contradictions between these papers?",
    top_k=20,
    max_depth=3
)
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### CLI Usage

```bash
python src/vector_store.py ingest ./documents -c research --recursive
python src/vector_store.py query "main findings" -c research -k 10
python src/vector_store.py analyze "What are the key themes?" -c research
```

Data location: `~/.cache/phi_rlm/vectordb/`

---

## Parallel Processing

```python
from src.phi_enhanced_rlm import PhiEnhancedRLM

rlm = PhiEnhancedRLM(backend, context_chunks)
rlm.enable_parallel(True)  # Subquestions processed concurrently
result = rlm.recursive_solve("Complex query", max_depth=3)
```

---

## Reasoning Tree

```python
rlm = PhiEnhancedRLM(backend, context_chunks)
result = rlm.recursive_solve("Query")
rlm.print_reasoning_tree()
```

```
REASONING TREE
──────────────
D0: What is the golden ratio?
   Conf: 85.0% | Chunks: [0, 3, 5]
  D1: Mathematical definition...
     Conf: 75.0% | Stopped: momentum
  D1: Applications in nature...
     Conf: 80.0% | Stopped: spectral
```

---

## Benchmarks

Compare phi-RLM against vanilla recursive decomposition on standard benchmarks:

```bash
# Run benchmark comparison
python benchmarks/runner.py

# Run benchmark tests
pytest tests/test_benchmarks.py -v
```

Includes GSM8K (math reasoning) and ARC (abstract reasoning) sample datasets. Results are saved to `benchmarks/results/`.

---

## Comparison Mode

```bash
# In chat:
> /compare facebook/react vuejs/vue

# Or via API:
curl -X POST http://localhost:8000/compare \
  -d '{"source1": "react", "source2": "vue"}'
```

---

## Mathematical Foundations

The system is built on 10 novel mathematical frameworks from phi-Separation theory:

### phi-Separation Kernel

```
K(x, y) = phi^(-||x - y||/delta)
```

### E8 Casimir Budget Allocation

Token budgets distributed across recursion depths using E8 Lie group Casimir eigenvalues:

```
Depth 0: 635 tokens (15.5%)
Depth 1: 577 tokens (14.1%)
...
Depth 7: 405 tokens (9.9%)
```

### QEC Threshold

```
p_phi = (1 - phi^{-1})/2 ~ 0.191
```

### Additional Frameworks

- **phi-Gram Theory** -- Gram matrix analysis on phi-scaled inner products
- **E8 Spectral Flow** -- Casimir eigenvalue-guided token allocation
- **phi-Kernel Renormalization** -- Renormalization group flow for confidence calibration
- **H4-Prime Theory** -- H4 Coxeter group structure for multi-scale reasoning
- **Torsion-Corrected FA** -- Factor analysis with geometric torsion corrections
- **phi-Lattice Cryptography** -- Lattice-based verification of reasoning integrity
- **Casimir Cohomology** -- Cohomological analysis for reasoning consistency
- **Unified Field Theory** -- Integration of all frameworks into a single system

See `docs/` for the full research paper and mathematical foundations document.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Skip integration tests (no API key needed)
pytest tests/ -v -m "not integration"

# Run specific test files
pytest tests/test_evolution.py -v
pytest tests/test_benchmarks.py -v
pytest tests/test_real_llm.py -v
```

CI runs tests on Python 3.9, 3.10, 3.11, and 3.12 with ruff linting.

---

## Installation

```bash
# Full install with all optional dependencies
pip install -e ".[full,dev]"

# Core only
pip install -e .

# With benchmark support
pip install -e ".[benchmark]"
```

### CLI Entry Points

After installation, three CLI commands are available:

| Command | Description |
|---------|-------------|
| `phi-rlm` | Interactive chat |
| `phi-analyze` | Repository/URL analyzer |
| `phi-evolve` | Run self-evolution loop |

---

## System Validation

```bash
python cli/validate_rlm.py
```

Checks that all modules load correctly, dependencies are available, and the system is ready.

---

## Contributing

Pull requests welcome! Please run tests and lint before submitting:

```bash
pytest tests/ -v -m "not integration"
ruff check src/ tests/ benchmarks/ --select E,F,W --ignore E501
```

## License

MIT License -- see [LICENSE](LICENSE).

---

*"The universe may be built on E8 geometry, with phi as its fundamental scaling constant."*
