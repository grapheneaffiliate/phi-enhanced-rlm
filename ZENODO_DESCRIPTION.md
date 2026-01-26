# PHI-Enhanced Recursive Language Model (RLM)

**A recursive reasoning framework that decomposes complex questions into confidence-weighted sub-analyses using φ-Separation Mathematics.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18377963.svg)](https://doi.org/10.5281/zenodo.18377963)

---

## 🎯 What is RLM?

RLM takes a complex question and recursively breaks it down into simpler sub-questions, solving each with tracked confidence scores, then synthesizes a final answer. Think of it as **"thinking step-by-step, but mathematically rigorous."**

```
Question: "How does React compare to Vue?"
    ├── Sub-Q: "What is React's architecture?" (conf: 0.85)
    ├── Sub-Q: "What is Vue's architecture?" (conf: 0.82)  
    ├── Sub-Q: "Performance differences?" (conf: 0.78)
    └── Synthesis → Final answer (conf: 0.81)
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔄 **Recursive Decomposition** | Breaks complex queries into tractable sub-problems |
| 📊 **Confidence Tracking** | Every answer includes a calibrated confidence score |
| 📄 **Document Analysis** | Analyze PDFs, Word docs, web pages, GitHub repos |
| 🗄️ **Vector Store** | ChromaDB-backed RAG for large document collections |
| 💾 **Embedding Cache** | SQLite cache persists across sessions |
| ⚡ **Parallel Processing** | Sub-questions processed concurrently |
| 🔌 **REST API** | FastAPI server with OpenAPI docs |
| 💬 **Interactive Chat** | Rich terminal UI with streaming |
| 🔄 **Comparison Mode** | Compare repos, URLs, or documents side-by-side |

---

## 🚀 Quick Start

```bash
git clone https://github.com/grapheneaffiliate/phi-enhanced-rlm.git
cd phi-enhanced-rlm
pip install -r requirements.txt
cp .env.template .env  # Add OPENROUTER_API_KEY

# Interactive chat
python cli/chat.py

# REST API
python api/server.py  # http://localhost:8000/docs
```

---

## 🧮 Mathematical Foundation

RLM is grounded in **φ-Separation Mathematics**, using:

- **φ-Separation Kernel**: `K(x, y) = φ^(-||x - y||/δ)` for semantic similarity
- **E8 Casimir Budget**: Optimal token allocation across recursion depths
- **QEC Threshold**: `p_φ ≈ 0.191` for confidence calibration

The golden ratio (φ ≈ 1.618) appears throughout as a fundamental scaling constant.

---

## 📦 What's Included

```
phi-enhanced-rlm/
├── src/           # Core library (RLM, embeddings, cache, vector store)
├── cli/           # Interactive chat, repo analyzer, validation
├── api/           # FastAPI REST server
├── tests/         # Test suite
├── docs/          # Research papers & documentation
└── examples/      # Sample outputs
```

---

## 📖 Use Cases

- **Research Analysis**: Feed papers → get synthesized insights with confidence
- **Code Review**: Analyze repositories recursively
- **Document Comparison**: Side-by-side analysis of any two sources
- **Knowledge Base Q&A**: Build a vector store, ask questions

---

## 🔗 Links

- **GitHub**: [grapheneaffiliate/phi-enhanced-rlm](https://github.com/grapheneaffiliate/phi-enhanced-rlm)
- **Documentation**: See README.md in repository
- **License**: MIT

---

## 📄 Citation

```bibtex
@software{rlm2026,
  author       = {McGirl, Tim},
  title        = {PHI-Enhanced Recursive Language Model},
  version      = {v2.1.0},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18377963},
  url          = {https://doi.org/10.5281/zenodo.18377963}
}
```

---

*"Recursive reasoning with mathematical rigor."*
