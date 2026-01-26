# PHI-Enhanced Recursive Language Model (RLM) v2.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![API](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)

A recursive language model framework enhanced with φ-Separation Mathematics, featuring streaming responses, document analysis, REST API, and parallel processing.

## ✨ Features

- **📡 Streaming Output** — Real-time response streaming
- **📄 PDF/DOCX Support** — Analyze documents directly
- **🌐 Smart Web Extraction** — Trafilatura for clean article extraction
- **💾 SQLite Embedding Cache** — Persistent cache survives restarts
- **🔌 REST API** — FastAPI with OpenAPI docs
- **💬 Conversation Memory** — Stateful chat with context
- **⚡ Parallel Processing** — Process subquestions concurrently
- **📊 Confidence Visualization** — See the reasoning tree
- **🔄 Comparison Mode** — Compare repos, URLs, or documents
- **📝 Export Reports** — Save analyses as markdown
- **🎨 Rich Progress** — Beautiful terminal UI

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/grapheneaffiliate/phi-enhanced-rlm.git
cd phi-enhanced-rlm
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.template .env
# Edit .env: OPENROUTER_API_KEY=sk-or-v1-your-key
```

### 3. Run

```bash
# Interactive chat
python cli/chat.py

# REST API server
python api/server.py
# Visit: http://localhost:8000/docs
```

---

## 📁 Project Structure

```
phi-enhanced-rlm/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Package configuration
├── .env.template            # API key template
├── .gitignore
│
├── src/                     # Core library
│   ├── __init__.py
│   ├── phi_enhanced_rlm.py      # Main RLM orchestrator
│   ├── phi_separation_novel_mathematics.py  # φ-Math foundations
│   ├── embeddings.py            # Embedding generation
│   ├── cache.py                 # SQLite embedding cache
│   ├── extractors.py            # PDF/DOCX/web extractors
│   ├── progress.py              # Rich progress display
│   ├── openrouter_backend.py    # OpenRouter LLM backend
│   └── async_backend.py         # Async LLM operations
│
├── cli/                     # Command-line tools
│   ├── __init__.py
│   ├── chat.py                  # Interactive chat v2.0
│   ├── run_rlm.py               # Simple query runner
│   ├── repo_analyzer.py         # GitHub/URL analyzer
│   └── validate_rlm.py          # System validation
│
├── api/                     # REST API
│   ├── __init__.py
│   └── server.py                # FastAPI server
│
├── web/                     # Web interface
│   └── index.html
│
├── tests/                   # Test suite
│   ├── __init__.py
│   └── test_upgrades.py
│
├── docs/                    # Documentation
│   ├── 2512.24601v1.pdf         # Research paper
│   └── Novel_Mathematics_from_Phi_Separation.docx
│
└── examples/                # Example outputs
    └── *.json, *.jsonl
```

---

## 💬 Interactive Chat

```bash
python cli/chat.py
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

### Example

```
PHI-ENHANCED RLM INTERACTIVE CHAT v2.0
✓ Backend ready: anthropic/claude-3.5-sonnet

You: What is the golden ratio?

╭─ PHI-RLM Response ─────────────────────────╮
│ The golden ratio (φ ≈ 1.618) is...        │
╰────────────────────────────────────────────╯
Confidence: 85.0%

You: /pdf research_paper.pdf
Extracting PDF...
✓ Analyzed 12 pages

You: /export analysis.md
✓ Exported to analysis.md
```

---

## 🔌 REST API

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

## 📄 Document Analysis

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

## 💾 Embedding Cache

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

## ⚡ Parallel Processing

```python
from src.phi_enhanced_rlm import PhiEnhancedRLM

rlm = PhiEnhancedRLM(backend, context_chunks)
rlm.enable_parallel(True)  # Subquestions processed concurrently

result = rlm.recursive_solve("Complex query", max_depth=3)
```

---

## 📊 Reasoning Tree

```python
rlm = PhiEnhancedRLM(backend, context_chunks)
result = rlm.recursive_solve("Query")
rlm.print_reasoning_tree()
```

```
REASONING TREE
──────────────
🟢 D0: What is the golden ratio?
   Conf: 85.0% | Chunks: [0, 3, 5]
  🟡 D1: Mathematical definition...
     Conf: 75.0% | Stopped: momentum
  🟢 D1: Applications in nature...
     Conf: 80.0% | Stopped: spectral
```

---

## 🔄 Comparison Mode

```bash
# In chat:
> /compare facebook/react vuejs/vue

# Or via API:
curl -X POST http://localhost:8000/compare \
  -d '{"source1": "react", "source2": "vue"}'
```

---

## 🧮 Mathematical Foundations

### φ-Separation Kernel

```
K(x, y) = φ^(-||x - y||/δ)
```

### E8 Casimir Budget Allocation

```
Depth 0: 635 tokens (15.5%)
Depth 1: 577 tokens (14.1%)
...
Depth 7: 405 tokens (9.9%)
```

### QEC Threshold

```
p_φ = (1 - φ^{-1})/2 ≈ 0.191
```

---

## 🧪 Testing

```bash
python tests/test_upgrades.py
```

---

## 📦 Installation (Development)

```bash
# Install with all optional dependencies
pip install -e ".[full,dev]"

# Or just core:
pip install -e .
```

---

## 🤝 Contributing

Pull requests welcome! Please run tests before submitting.

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

*"The universe may be built on E8 geometry, with φ as its fundamental scaling constant."*
