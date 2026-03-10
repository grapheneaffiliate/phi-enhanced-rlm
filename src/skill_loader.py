#!/usr/bin/env python3
"""
Skill Loader -- loads claude-code-templates skills as supplementary context.

Skills are markdown files with structured instructions for specific domains
(e.g., RAG patterns, agent memory systems, evaluation harnesses). The
SkillLoader indexes them and provides relevant skill context to inject into
recursive_solve() prompts, giving the RLM access to domain expertise at
each recursion depth.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Superpowers skill keyword mappings for enhanced retrieval
SUPERPOWERS_KEYWORDS = {
    "brainstorming": ["design", "plan", "idea", "feature", "build", "create", "spec"],
    "writing-plans": ["implement", "plan", "steps", "tasks", "decompose", "break down"],
    "subagent-driven-development": ["execute", "implement", "build", "code", "develop"],
    "test-driven-development": ["test", "tdd", "verify", "assert", "coverage", "red green"],
    "systematic-debugging": ["bug", "error", "fail", "broken", "fix", "debug", "crash"],
    "verification-before-completion": ["done", "complete", "verify", "check", "confirm", "ship"],
    "dispatching-parallel-agents": ["parallel", "concurrent", "multiple", "independent"],
    "requesting-code-review": ["review", "quality", "check", "approve"],
    "receiving-code-review": ["feedback", "review", "comments", "address", "revise", "respond"],
    "executing-plans": ["run", "execute", "step", "progress", "follow", "iterate"],
    "finishing-a-development-branch": ["merge", "branch", "finish", "pr", "pull request", "ship"],
    "using-git-worktrees": ["worktree", "git", "branch", "isolate", "parallel", "checkout"],
    "using-superpowers": ["superpowers", "workflow", "pipeline", "orchestrate", "phases"],
    "writing-skills": ["skill", "template", "write", "author", "markdown", "instructions"],
}


class SkillLoader:
    """Load and index skills from .claude/skills/ directory.

    Skills are markdown files (SKILL.md) organized in subdirectories
    by category. The loader indexes them by name and provides keyword-based
    retrieval to find skills relevant to a given query.
    """

    def __init__(self, skills_dir: str = ".claude/skills"):
        self.skills_dir = Path(skills_dir)
        self.skills: Dict[str, str] = {}
        self._skill_keywords: Dict[str, set] = {}
        self._load_skills()

    def _load_skills(self):
        """Load all SKILL.md files from the skills directory tree."""
        if not self.skills_dir.exists():
            logger.debug(f"Skills directory not found: {self.skills_dir}")
            return

        loaded = 0
        for skill_file in self.skills_dir.rglob("SKILL.md"):
            try:
                rel = skill_file.relative_to(self.skills_dir)
                name = str(rel.parent)
                content = skill_file.read_text(encoding="utf-8")
                self.skills[name] = content

                # Build keyword index from directory name and content header
                keywords = set(
                    name.replace("/", " ").replace("-", " ").replace("_", " ")
                    .lower().split()
                )
                # Also extract keywords from first 200 chars of content
                header = content[:200].lower()
                for word in header.split():
                    clean = word.strip(".,:#*-()[]")
                    if len(clean) > 3:
                        keywords.add(clean)
                # Add superpowers keyword mappings if available
                base_name = name.split("/")[-1] if "/" in name else name
                if base_name in SUPERPOWERS_KEYWORDS:
                    keywords.update(SUPERPOWERS_KEYWORDS[base_name])
                self._skill_keywords[name] = keywords
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load skill {skill_file}: {e}")

        if loaded > 0:
            logger.info(f"Loaded {loaded} skills from {self.skills_dir}")

    def get_relevant_skills(self, query: str, max_skills: int = 2) -> List[str]:
        """Find skills relevant to the query using keyword matching.

        Args:
            query: The search query.
            max_skills: Maximum number of skill excerpts to return.

        Returns:
            List of skill content strings (truncated to 500 chars each).
        """
        if not self.skills:
            return []

        query_words = set(query.lower().split())
        scored: List[Tuple[int, str, str]] = []

        for name, keywords in self._skill_keywords.items():
            overlap = len(query_words & keywords)
            if overlap > 0:
                content = self.skills[name]
                scored.append((overlap, name, content[:500]))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [content for _, _, content in scored[:max_skills]]

    def get_skill(self, name: str) -> Optional[str]:
        """Get a specific skill by its directory name.

        Args:
            name: Skill name (e.g., "ai-research/rag-chroma").

        Returns:
            Full skill content, or None if not found.
        """
        return self.skills.get(name)

    def list_skills(self) -> List[str]:
        """List all available skill names."""
        return sorted(self.skills.keys())

    @property
    def skill_count(self) -> int:
        """Number of loaded skills."""
        return len(self.skills)
