#!/usr/bin/env python3
"""
Agent Router -- maps phi-RLM recursion depths to specialized agents.

Instead of using the same LLM prompt at every depth, each depth gets a
different agent persona from claude-code-templates. This is structural
self-recursion: the model at depth N is literally a different agent than
at depth N-1.

The deep-research-team agents map onto E8 Casimir degrees:
  Depth 0 (C=2):  research-orchestrator  -- route and plan
  Depth 1 (C=8):  query-clarifier        -- refine the question
  Depth 2 (C=12): research-analyst       -- primary analysis
  Depth 3 (C=14): technical-researcher   -- deep technical dive
  Depth 4 (C=18): academic-researcher    -- academic rigor
  Depth 5 (C=20): fact-checker           -- QEC verification
  Depth 6 (C=24): research-synthesizer   -- torsion-corrected synthesis
  Depth 7 (C=30): research-orchestrator  -- final integration
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Map E8 Casimir degrees to agent specializations
DEPTH_AGENT_MAP: Dict[int, str] = {
    0: "research-orchestrator",    # Casimir 2:  binary routing
    1: "query-clarifier",          # Casimir 8:  clarify scope
    2: "research-analyst",         # Casimir 12: primary analysis
    3: "technical-researcher",     # Casimir 14: deep technical
    4: "academic-researcher",      # Casimir 18: academic rigor
    5: "fact-checker",             # Casimir 20: verification (QEC)
    6: "research-synthesizer",     # Casimir 24: synthesis
    7: "research-orchestrator",    # Casimir 30: final integration
}

# Default agent persona snippets (used when .claude/agents/ files not installed)
DEFAULT_PERSONAS: Dict[str, str] = {
    "research-orchestrator": (
        "You are a research orchestrator. Plan the research approach, "
        "decompose complex questions into sub-tasks, assign priorities, "
        "and integrate findings from multiple research streams into a "
        "coherent final answer."
    ),
    "query-clarifier": (
        "You are a query clarifier. Analyze the question for ambiguity, "
        "identify implicit assumptions, determine the scope and constraints, "
        "and reformulate the question for maximum precision."
    ),
    "research-analyst": (
        "You are a research analyst. Conduct primary analysis by examining "
        "evidence, identifying patterns, evaluating source quality, and "
        "drawing well-supported conclusions from the available data."
    ),
    "technical-researcher": (
        "You are a technical researcher. Dive deep into technical details, "
        "verify implementation specifics, check algorithmic correctness, "
        "and provide precise technical assessments."
    ),
    "academic-researcher": (
        "You are an academic researcher. Apply rigorous scholarly standards, "
        "cite relevant prior work, identify methodological strengths and "
        "weaknesses, and evaluate claims against the existing literature."
    ),
    "fact-checker": (
        "You are a fact checker. Independently verify claims for accuracy, "
        "check for contradictions, identify unsupported assertions, search "
        "for counterexamples, and rate the reliability of evidence."
    ),
    "research-synthesizer": (
        "You are a research synthesizer. Merge findings from multiple "
        "analysis streams, preserve important contradictions, reconcile "
        "conflicting evidence, rate evidence quality, and produce a "
        "unified synthesis that acknowledges uncertainty."
    ),
}


class AgentRouter:
    """Routes recursive calls to depth-appropriate agent personas.

    Loads agent persona files from .claude/agents/ if available,
    falling back to built-in default personas. The evolution engine
    can optimize which agent goes at which depth by modifying
    the depth-agent mapping.
    """

    def __init__(self, agents_dir: str = ".claude/agents",
                 depth_agent_map: Optional[Dict[int, str]] = None):
        """
        Args:
            agents_dir: Path to directory containing agent persona .md files.
            depth_agent_map: Override mapping from depth to agent name.
                           Uses DEPTH_AGENT_MAP if None.
        """
        self.agents_dir = Path(agents_dir)
        self.depth_map = depth_agent_map or dict(DEPTH_AGENT_MAP)
        self.agent_prompts: Dict[str, str] = {}
        self._load_agents()

    def _load_agents(self):
        """Load agent persona files from .claude/agents/, with defaults."""
        # Start with default personas
        self.agent_prompts = dict(DEFAULT_PERSONAS)

        # Override with file-based personas if available
        if self.agents_dir.exists():
            loaded = 0
            for md_file in self.agents_dir.glob("*.md"):
                name = md_file.stem
                try:
                    content = md_file.read_text(encoding="utf-8")
                    # Extract system prompt (after YAML front matter)
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        self.agent_prompts[name] = parts[2].strip()
                    else:
                        self.agent_prompts[name] = content.strip()
                    loaded += 1
                except Exception as e:
                    logger.warning(f"Failed to load agent {name}: {e}")
            if loaded > 0:
                logger.info(f"Loaded {loaded} agent personas from {self.agents_dir}")

    def get_agent_for_depth(self, depth: int) -> Optional[str]:
        """Get the agent system prompt for a given recursion depth."""
        agent_name = self.depth_map.get(min(depth, 7))
        if agent_name is None:
            return None
        return self.agent_prompts.get(agent_name)

    def get_agent_name_for_depth(self, depth: int) -> str:
        """Get the agent name assigned to a recursion depth."""
        return self.depth_map.get(min(depth, 7), "unknown")

    def inject_agent_persona(self, prompt: str, depth: int) -> str:
        """Prepend the depth-appropriate agent persona to the prompt.

        This is the key integration point: each recursion depth gets a
        different specialist perspective injected into the prompt.
        """
        agent_name = self.get_agent_name_for_depth(depth)
        persona = self.get_agent_for_depth(depth)
        if persona:
            return (
                f"[Agent: {agent_name} | Depth: {depth}]\n\n"
                f"{persona}\n\n"
                f"---\n\n"
                f"{prompt}"
            )
        return prompt

    def get_verifier_prompt(self) -> Optional[str]:
        """Get the fact-checker agent persona for QEC verification."""
        return self.agent_prompts.get("fact-checker")

    def get_synthesizer_prompt(self) -> Optional[str]:
        """Get the research-synthesizer persona for torsion-corrected aggregation."""
        return self.agent_prompts.get("research-synthesizer")

    def get_available_agents(self) -> List[str]:
        """List all loaded agent names."""
        return sorted(self.agent_prompts.keys())

    def update_depth_mapping(self, depth: int, agent_name: str):
        """Update which agent is assigned to a specific depth.

        Called by the evolution engine to optimize agent assignments.
        """
        if agent_name in self.agent_prompts:
            self.depth_map[depth] = agent_name
            logger.info(f"Updated depth {depth} agent: {agent_name}")
        else:
            logger.warning(f"Agent '{agent_name}' not found, keeping current mapping")
