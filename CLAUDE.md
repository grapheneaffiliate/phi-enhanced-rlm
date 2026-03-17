# phi-Enhanced Recursive Language Model

## Project Intelligence

This is a self-evolving recursive language model using golden ratio (phi) mathematics.
Core architecture: E8 Lie group geometry governs budget allocation, stopping criteria,
and attention patterns.

## Key Files

- `src/phi_enhanced_rlm.py` -- Core recursive engine (recursive_solve, QEC, aggregation)
- `src/evolution.py` -- Self-evolution with phi-scaled learning rate
- `src/meta_recursion.py` -- Strategy selection and meta-reasoning
- `src/phi_memory.py` -- Golden spiral persistent memory
- `src/phi_separation_novel_mathematics.py` -- E8/phi mathematical foundations
- `src/agent_router.py` -- Depth-to-agent mapping (deep-research-team integration)
- `src/skill_loader.py` -- Loads skills as supplementary context
- `src/phi_attention.py` -- phi-structured attention injection
- `src/phi_sparse_reasoning.py` -- phi-ratio branch pruning
- `src/embeddings.py` -- Multi-provider embedding system
- `src/vector_store.py` -- ChromaDB vector store integration
- `src/tool_executor.py` -- Tool registry for agentic capability (v4.1)
- `src/outcome_tracker.py` -- Real-world feedback for evolution fitness (v4.1)
- `src/phi_critic.py` -- Adversarial self-evaluation (v4.1)
- `src/phi_planner.py` -- DAG-based task decomposition (v4.1)
- `src/mcp_server.py` -- MCP server for Claude Code integration (v4.1)
- `src/pra_controller.py` -- Phi-Recursive Architecture controller (v5.2)

## Architecture

### Recursion Engine

`recursive_solve()` is the main engine with 8 components:
1. phi-Gram chunk selection (greedy delta-logdet diversity)
2. Casimir flow budget allocation (E8 Casimir degrees)
3. phi-momentum early stopping
4. Spectral flow saturation detection
5. Golden ratio QEC verification (3 independent verifiers)
6. Torsion-corrected aggregation (E8 epsilon = 28/248)
7. Dependency cohomology tracking
8. PRA self-referential control (defect-sensitive halting, equilibrium budget)

### Agent Integration (v4.0)

Deep-research-team agents from claude-code-templates are mapped to recursion depths.
Each depth uses a different specialist agent persona:

| Depth | Casimir | Agent               | Role                  |
|-------|---------|---------------------|-----------------------|
| 0     | 2       | research-orchestrator | Route and plan       |
| 1     | 8       | query-clarifier     | Refine the question   |
| 2     | 12      | research-analyst    | Primary analysis      |
| 3     | 14      | technical-researcher | Deep technical dive  |
| 4     | 18      | academic-researcher | Academic rigor        |
| 5     | 20      | fact-checker        | QEC verification      |
| 6     | 24      | research-synthesizer | Torsion synthesis    |
| 7     | 30      | research-orchestrator | Final integration   |

### Meta-Recursion Strategies

- `deep_analytical` -- Deep narrow analysis (max_depth=6)
- `wide_exploratory` -- Wide shallow exploration (max_depth=3)
- `spiral_convergent` -- Balanced phi-enhanced (max_depth=5)
- `quick_factual` -- Minimal recursion lookup (max_depth=2)
- `deep_research` -- Full E8 hierarchy with agent specialization (max_depth=7)
- `planned` -- Pre-planned DAG execution with dependency-ordered steps (max_depth=4)
- `pra` -- Phi-Recursive Architecture with self-referential control law (max_depth=7)

## Conventions

- All recursion depths map to E8 Casimir degrees: [2, 8, 12, 14, 18, 20, 24, 30]
- phi = 1.618033988749895, epsilon = 28/248 (torsion correction)
- Tests: `python -m pytest tests/ -v`
- Evolution: `python -m src.evolution_loop --generations 10`
- Version: 5.2.0

## Development

- Use relative imports within `src/` package (e.g., `from .module import Class`)
- New modules should follow the lazy-loading pattern used by phi_enhanced_rlm.py
- Evolution state persists in `evolution_state.json`
- Trace logs written to `rlm_trace.jsonl`
