# PHI-Enhanced Recursive Language Model (RLM)

**A self-evolving recursive reasoning framework using golden ratio (phi) mathematics and E8 Lie group geometry for multi-agent AI problem solving.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18377963.svg)](https://doi.org/10.5281/zenodo.18377963)

---

## What is RLM?

RLM takes a complex question and recursively breaks it down into simpler sub-questions, routing each to a depth-specialized agent, solving with tracked confidence scores, then synthesizing a final answer. All governed by the golden ratio and E8 Casimir degrees.

```
Question: "How does React compare to Vue?"
    |-- [research-orchestrator] Route and plan (Casimir 2)
    |-- [query-clarifier] Refine sub-questions (Casimir 8)
    |-- [research-analyst] Architecture comparison (Casimir 12, conf: 0.85)
    |-- [fact-checker] QEC verification (Casimir 20)
    +-- [research-synthesizer] Final synthesis (conf: 0.81)
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| Recursive Decomposition | Breaks complex queries into tractable sub-problems |
| Multi-Agent Architecture | 8 specialized agents mapped to E8 Casimir degrees |
| Self-Evolution | Phi-scaled learning rate adapts parameters over generations |
| Confidence Tracking | Every answer includes a calibrated confidence score |
| QEC Verification | 3 independent checks (contradiction, completeness, counterexample) |
| Adversarial Critic | Self-evaluation with phi-geometric scoring (v4.1) |
| DAG Task Planner | Dependency-ordered task decomposition (v4.1) |
| Tool Executor | Agentic tool registry for external actions (v4.1) |
| Outcome Tracker | Real-world feedback for evolution fitness (v4.1) |
| MCP Server | Claude Code integration via Model Context Protocol (v4.1) |
| Workflow Orchestrator | 8-phase superpowers pipeline with safety gates (v4.1) |
| Document Analysis | Analyze PDFs, Word docs, web pages, GitHub repos |
| Vector Store | ChromaDB-backed RAG for large document collections |
| Embedding Cache | SQLite cache persists across sessions |
| Parallel Processing | Sub-questions processed concurrently |
| REST API | FastAPI server with OpenAPI docs |
| Interactive Chat | Rich terminal UI with streaming |

---

## Quick Start

```bash
git clone https://github.com/grapheneaffiliate/phi-enhanced-rlm.git
cd phi-enhanced-rlm
bash setup.sh

# Interactive chat
python cli/chat.py

# REST API
python -m uvicorn api.server:app --reload --port 8000

# Self-evolution
python -m src.evolution_loop --generations 10
```

---

## Mathematical Foundation

RLM is grounded in phi-Separation Mathematics, using:

- **phi-Gram Kernel**: `K(x, y) = phi^(-||x - y||/delta)` for semantic similarity
- **E8 Casimir Budget**: Optimal token allocation across 8 recursion depths via `[2, 8, 12, 14, 18, 20, 24, 30]`
- **QEC Threshold**: `p_phi = (1 - phi^(-1))/2 ~ 0.191` for confidence calibration
- **Torsion Correction**: `epsilon = 28/248` preserves minority viewpoints in synthesis
- **phi-Momentum Stopping**: `m(t+1) = phi^(-1) * m(t) + (1 - phi^(-1)) * signal(t)`
- **Evolution Learning Rate**: `lr(n) = phi^(-n)` for diminishing adaptation

Full details in `src/phi_separation_novel_mathematics.py` -- 10 interconnected frameworks.

---

## Architecture (v4.1)

```
phi-enhanced-rlm/
|-- src/                       Core engine (28 modules)
|   |-- phi_enhanced_rlm.py       Recursive engine, QEC, aggregation
|   |-- phi_separation_novel_mathematics.py  E8/phi math foundations
|   |-- agent_router.py           Multi-agent depth routing
|   |-- meta_recursion.py         6 meta-strategies + auto-selection
|   |-- workflow_orchestrator.py   8-phase superpowers pipeline
|   |-- evolution.py               Self-evolution engine
|   |-- phi_critic.py              Adversarial self-evaluation
|   |-- phi_planner.py             DAG-based task decomposition
|   |-- tool_executor.py           Agentic tool registry
|   |-- outcome_tracker.py         Real-world feedback tracking
|   |-- mcp_server.py              MCP server for Claude Code
|   +-- ...                        + 17 supporting modules
|
|-- tests/                     Test suite (70 tests)
|-- benchmarks/                GSM8K + ARC evaluation
|-- cli/                       Chat, repo analyzer, validation
|-- api/                       FastAPI REST server
+-- web/                       Documentation + evolution dashboard
```

---

## Meta-Recursion Strategies

| Strategy | Max Depth | Best For |
|----------|-----------|----------|
| `deep_analytical` | 6 | Complex analytical questions |
| `wide_exploratory` | 3 | Open-ended, broad questions |
| `spiral_convergent` | 5 | General-purpose (default) |
| `quick_factual` | 2 | Simple factual lookups |
| `deep_research` | 7 | Full E8 hierarchy with agent specialization |
| `planned` | 4 | Pre-planned DAG execution with dependency ordering |

---

## Use Cases

- **Research Analysis**: Feed papers, get synthesized insights with confidence
- **Code Review**: Analyze repositories recursively with agent specialization
- **Document Comparison**: Side-by-side analysis of any two sources
- **Knowledge Base Q&A**: Build a vector store, ask questions
- **Autonomous Tasks**: Tool executor enables agentic workflows (v4.1)

---

## Links

- **GitHub**: [grapheneaffiliate/phi-enhanced-rlm](https://github.com/grapheneaffiliate/phi-enhanced-rlm)
- **Documentation**: See README.md in repository
- **License**: MIT

---

## Citation

```bibtex
@software{rlm2026,
  author       = {McGirl, Tim},
  title        = {PHI-Enhanced Recursive Language Model},
  version      = {v4.1.0},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18377963},
  url          = {https://doi.org/10.5281/zenodo.18377963}
}
```

---

*"The universe may be built on the geometry of E8, with the golden ratio as its fundamental scaling constant."*
