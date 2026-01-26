# PHI-Enhanced Recursive Language Model (RLM) Framework v2.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Mathematics](https://img.shields.io/badge/Mathematics-E8%20%7C%20%CF%86%20Separation-purple.svg)]()
[![API](https://img.shields.io/badge/API-FastAPI-green.svg)]()

A groundbreaking implementation of Recursive Language Models enhanced with φ-Separation Mathematics, featuring streaming responses, document analysis, REST API, and parallel processing.

## 🆕 What's New in v2.0

- **📡 Streaming Output** - Real-time response streaming for long analyses
- **📄 PDF/DOCX Support** - Analyze PDF and Word documents directly
- **🌐 Better Web Extraction** - Trafilatura for clean article extraction
- **💾 SQLite Embedding Cache** - Persistent cache survives restarts
- **🔌 REST API** - FastAPI server with OpenAPI docs
- **💬 Conversation Memory** - Stateful chat with context
- **⚡ Parallel Processing** - Process subquestions concurrently
- **📊 Confidence Visualization** - See the reasoning tree
- **🖼️ Multi-modal Support** - Analyze images with vision models
- **🔄 Comparison Mode** - Compare repos, URLs, or documents
- **📝 Export Reports** - Save analyses as markdown
- **🎨 Rich Progress** - Beautiful terminal UI with progress bars

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd RLM
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.template .env
# Edit .env and add: OPENROUTER_API_KEY=sk-or-v1-your-key
```

### 3. Run

```bash
# Interactive chat (recommended)
python chat.py

# REST API server
python api.py
# Then visit: http://localhost:8000/docs
```

---

## 💬 Interactive Chat v2.0

The feature-rich chat interface:

```bash
python chat.py
```

### All Commands

| Command | Description |
|---------|-------------|
| `<question>` | Ask any question |
| `/repo owner/repo` | Analyze GitHub repository |
| `/url https://...` | Analyze web page (trafilatura extraction) |
| `/local ./path` | Analyze local directory |
| `/pdf path.pdf` | **NEW** Analyze PDF document |
| `/doc path.docx` | **NEW** Analyze Word document |
| `/image path.png` | **NEW** Describe & analyze image |
| `/compare s1 s2` | **NEW** Compare two sources |
| `/export file.md` | **NEW** Export last analysis |
| `/history` | **NEW** Show query history |
| `/stream on\|off` | **NEW** Toggle streaming |
| `/trace` | **NEW** Show reasoning tree |
| `/depth N` | Set recursion depth (0-10) |
| `/context file.json` | Load custom context |
| `/reset` | Reset context and history |
| `/model` | Show current model |
| `/help` | Show all commands |
| `/quit` | Exit |

### Example Session

```
  PHI-ENHANCED RLM INTERACTIVE CHAT v2.0

✓ Backend ready: anthropic/claude-3.5-sonnet
✓ RLM ready (8 chunks, depth=3)

You: What is the golden ratio?
Thinking...

╭─ PHI-RLM Response ────────────────────────────────────────╮
│ The golden ratio (φ) is approximately 1.618...           │
╰───────────────────────────────────────────────────────────╯
Confidence: 85.0%

You: /pdf research_paper.pdf
Extracting PDF: research_paper.pdf...
✓ Extracted: Novel Methods (12 pages)
Analyzing 15 sections...

╭─ PHI-RLM Response ────────────────────────────────────────╮
│ This paper presents a novel approach to...               │
╰───────────────────────────────────────────────────────────╯
Confidence: 82.0%

You: /export analysis.md
✓ Exported to analysis.md
```

---

## 🔌 REST API

Start the FastAPI server:

```bash
python api.py
# Or with auto-reload:
uvicorn api:app --reload --port 8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/status` | System status & stats |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc documentation |
| POST | `/analyze` | Analyze query with RLM |
| POST | `/chat` | Chat with memory |
| POST | `/compare` | Compare two sources |
| GET | `/history` | Get query history |
| DELETE | `/session/{id}` | Delete chat session |

### Example: Analyze

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is E8 symmetry?",
    "max_depth": 3,
    "stream": false
  }'
```

Response:
```json
{
  "answer": "E8 is the largest exceptional Lie group...",
  "confidence": 0.85,
  "depth_reached": 2,
  "stop_reason": "momentum",
  "chunks_used": [0, 3, 5],
  "trace": [...],
  "timestamp": "2024-01-26T12:00:00Z"
}
```

### Example: Chat with Memory

```bash
# First message
curl -X POST http://localhost:8000/chat \
  -d '{"message": "What is phi?", "session_id": "user-123"}'

# Follow-up (uses context)
curl -X POST http://localhost:8000/chat \
  -d '{"message": "How does it relate to E8?", "session_id": "user-123"}'
```

### Example: Compare Sources

```bash
curl -X POST http://localhost:8000/compare \
  -d '{
    "source1": "facebook/react",
    "source2": "vuejs/vue",
    "query": "Compare architectures"
  }'
```

---

## 📄 Document Analysis

### PDF Support

Requires PyMuPDF:
```bash
pip install PyMuPDF
```

```python
from extractors import extract_pdf_content

result = extract_pdf_content("paper.pdf")
print(f"Title: {result.title}")
print(f"Pages: {result.metadata['page_count']}")
print(f"Text: {result.text[:1000]}")
```

CLI:
```bash
python chat.py
> /pdf research_paper.pdf
```

### Word Document Support

Requires python-docx:
```bash
pip install python-docx
```

```python
from extractors import extract_docx_content

result = extract_docx_content("report.docx")
print(result.text)
```

CLI:
```bash
> /doc quarterly_report.docx
```

---

## 🌐 Better Web Extraction

Uses trafilatura for article-quality extraction:

```bash
pip install trafilatura
```

```python
from extractors import extract_web_content

html = "<html>..."
result = extract_web_content(html, url="https://example.com")
print(f"Title: {result.title}")
print(f"Clean text: {result.text}")
```

---

## 💾 SQLite Embedding Cache

Embeddings are now cached in SQLite for persistence:

```python
from cache import SQLiteEmbeddingCache

cache = SQLiteEmbeddingCache()

# Cache survives restarts!
cache.set("hello world", "model-v1", embedding_vector)
cached = cache.get("hello world", "model-v1")  # Fast retrieval

# View stats
stats = cache.get_stats()
print(f"Hits: {stats.hits}, Misses: {stats.misses}")
print(f"Entries: {stats.entry_count}")
```

Cache location: `~/.cache/phi_rlm/embeddings.db`

---

## ⚡ Parallel Processing

Process subquestions concurrently:

```python
from phi_enhanced_rlm import PhiEnhancedRLM

rlm = PhiEnhancedRLM(backend, context_chunks)
rlm.enable_parallel(True)  # Enable parallel processing

result = rlm.recursive_solve("Complex query", max_depth=3)
# Subquestions processed in parallel!
```

Async version:
```python
import asyncio

async def analyze():
    result = await rlm.recursive_solve_async("Query", max_depth=3)
    return result

result = asyncio.run(analyze())
```

---

## 📊 Confidence Visualization

See the reasoning tree:

```python
rlm = PhiEnhancedRLM(backend, context_chunks)
result = rlm.recursive_solve("Query")

# Print tree
rlm.print_reasoning_tree()
```

Output:
```
============================================================
REASONING TREE
============================================================
Total nodes: 5
Max depth: 2
Avg confidence: 78.5%
------------------------------------------------------------
🟢 D0: What is the golden ratio?...
   Conf: 85.0% | Info: 150.0 | Chunks: [0, 3, 5]
  🟡 D1: Define mathematical constant phi...
     Conf: 75.0% | Info: 45.0 | Chunks: [1, 2]
     └─ Stopped: momentum
  🟢 D1: Applications in nature...
     Conf: 80.0% | Info: 38.0 | Chunks: [4, 6]
     └─ Stopped: spectral
============================================================
```

Using rich library:
```python
from progress import visualize_confidence_tree

with open("rlm_trace.jsonl") as f:
    trace = [json.loads(line) for line in f]

visualize_confidence_tree(trace)
```

---

## 🎨 Rich Progress Display

Beautiful progress bars with ETA:

```python
from progress import RichProgressManager

with RichProgressManager().track_analysis("Query", total_chunks=10) as tracker:
    for i in range(10):
        # Process chunk...
        tracker.update(
            processed=i+1,
            depth=i//3,
            confidence=0.5 + i*0.05
        )
```

---

## 🔧 Syntax-Aware Code Chunking

Smart chunking for Python/JavaScript:

```python
from extractors import chunk_python_code, chunk_code_file

# Chunk Python by functions/classes
chunks = chunk_python_code(python_source)
for chunk in chunks:
    print(f"{chunk['type']}: {chunk['name']}")
    # function: process_data
    # class: DataHandler

# Auto-detect language by extension
chunks = chunk_code_file("app.py")
```

---

## 🖼️ Multi-Modal Support

Analyze images with vision models:

```python
from extractors import describe_image, describe_image_url

# Local image
description = describe_image("diagram.png")

# Image URL
description = describe_image_url("https://example.com/image.jpg")
```

CLI:
```bash
> /image architecture_diagram.png
Analyzing image...
✓ Image described

Image Analysis:
The diagram shows a microservices architecture with...

Insights:
This architecture follows best practices for...
```

---

## 📝 Export Reports

Save analyses as formatted markdown:

```bash
> /export report.md
✓ Exported to report.md
```

Report includes:
- Query and response
- Confidence scores
- Reasoning trace table
- Session metadata

---

## 🔄 Comparison Mode

Compare any two sources:

```bash
> /compare facebook/react vuejs/vue
Comparing facebook/react vs vuejs/vue...
Analyzing facebook/react...
Analyzing vuejs/vue...

## Comparison: facebook/react vs vuejs/vue

**Similarities:**
- Both are component-based UI frameworks
- Virtual DOM implementation
- Large ecosystem and community

**Differences:**
- React uses JSX, Vue uses templates
- Vue has built-in state management
- React has more enterprise adoption
```

---

## 📁 Project Structure

```
RLM/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── .env.template               # API key template
│
├── chat.py                     # Interactive chat v2.0
├── api.py                      # FastAPI REST server
├── phi_enhanced_rlm.py         # Core RLM with parallel processing
├── openrouter_backend.py       # LLM backend
├── embeddings.py               # Embeddings with SQLite cache
├── cache.py                    # SQLite embedding cache
├── extractors.py               # PDF/DOCX/web extractors
├── progress.py                 # Rich progress & visualization
├── repo_analyzer.py            # GitHub/local analyzer
├── phi_separation_novel_mathematics.py
│
├── run_rlm.py                  # CLI runner
├── validate_rlm.py             # System validation
└── test_upgrades.py            # Test suite
```

---

## 🔬 Python API

### Basic Usage

```python
from openrouter_backend import OpenRouterBackend
from phi_enhanced_rlm import PhiEnhancedRLM

# Initialize
backend = OpenRouterBackend()
rlm = PhiEnhancedRLM(
    base_llm_callable=backend,
    context_chunks=["chunk1", "chunk2"],
    total_budget_tokens=4096
)

# Enable features
rlm.enable_parallel(True)

# Analyze
result = rlm.recursive_solve("Your question", max_depth=3)

print(f"Answer: {result.value}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Stop reason: {result.metadata['stop_reason']}")

# View reasoning tree
rlm.print_reasoning_tree()
```

### Async Usage

```python
import asyncio

async def main():
    result = await rlm.recursive_solve_async("Query", max_depth=3)
    return result

result = asyncio.run(main())
```

### With Progress Tracking

```python
from progress import get_progress_manager

pm = get_progress_manager()

with pm.track_analysis("Query", total_chunks=len(chunks)) as tracker:
    result = rlm.recursive_solve("Query")
    tracker.update(confidence=result.confidence)
```

---

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PHI-Enhanced RLM v2.0                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐               │
│   │  Chat   │    │   API   │    │   CLI   │               │
│   │  v2.0   │    │ FastAPI │    │ Runner  │               │
│   └────┬────┘    └────┬────┘    └────┬────┘               │
│        │              │              │                     │
│        └──────────────┼──────────────┘                     │
│                       ▼                                     │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              PhiEnhancedRLM Core                    │  │
│   │  • φ-Gram Chunk Selection                          │  │
│   │  • Parallel Subquestion Processing                 │  │
│   │  • Confidence Tree Tracking                        │  │
│   │  • QEC Verification                                │  │
│   └─────────────────────────────────────────────────────┘  │
│                       │                                     │
│        ┌──────────────┼──────────────┐                     │
│        ▼              ▼              ▼                     │
│   ┌─────────┐   ┌──────────┐   ┌──────────┐              │
│   │Embedder │   │ LLM      │   │Extractors│              │
│   │+SQLite  │   │ Backend  │   │PDF/DOCX  │              │
│   │ Cache   │   │OpenRouter│   │Web/Image │              │
│   └─────────┘   └──────────┘   └──────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
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

## 🤝 Contributing

Contributions welcome! Please submit a Pull Request.

## 📄 License

MIT License - see LICENSE file.

---

*"The universe may be built on the geometry of E8, with the golden ratio as its fundamental scaling constant."*
