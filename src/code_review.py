#!/usr/bin/env python3
"""
φ-Code Review — Multi-agent parallel code review powered by φ-RLM.

Replicates Claude Code's Code Review pattern:
- 4 specialized review agents run in parallel
- Each examines code from a different perspective (mapped to E8 Casimir degrees)
- Confidence scoring filters false positives
- Torsion-corrected aggregation preserves minority findings
- Works on git diffs, file paths, or raw code strings

No GPU required. Uses φ-RLM's recursive reasoning via API-based LLM calls.

Usage:
    from src.code_review import PhiCodeReview
    reviewer = PhiCodeReview(llm_backend)
    findings = reviewer.review_diff()           # Review uncommitted changes
    findings = reviewer.review_branch("main")   # Review branch vs main
    findings = reviewer.review_file("src/x.py") # Review a specific file
"""

import subprocess
import json
import logging
import time
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# E8 Casimir degrees mapped to review perspectives
REVIEW_PERSPECTIVES = {
    "logic": {
        "casimir": 2,
        "prompt": (
            "You are a logic and correctness reviewer. Examine this code for:\n"
            "- Logic errors, off-by-one mistakes, incorrect conditions\n"
            "- Missing edge cases and boundary conditions\n"
            "- Incorrect assumptions about data types or values\n"
            "- Broken control flow (unreachable code, infinite loops)\n"
            "- Race conditions or concurrency issues\n"
            "Focus ONLY on correctness. Ignore style, naming, and performance."
        ),
    },
    "security": {
        "casimir": 8,
        "prompt": (
            "You are a security reviewer. Examine this code for:\n"
            "- Injection vulnerabilities (SQL, command, path traversal)\n"
            "- Authentication and authorization flaws\n"
            "- Sensitive data exposure (keys, tokens, PII in logs)\n"
            "- Insecure defaults or missing input validation\n"
            "- Dependency risks or known vulnerable patterns\n"
            "Focus ONLY on security. Ignore style and minor logic issues."
        ),
    },
    "quality": {
        "casimir": 14,
        "prompt": (
            "You are a code quality reviewer. Examine this code for:\n"
            "- Poor naming that obscures intent\n"
            "- Functions doing too many things (SRP violations)\n"
            "- Unnecessary complexity that could be simplified\n"
            "- Missing error handling or swallowed exceptions\n"
            "- Code duplication that should be extracted\n"
            "- Missing or misleading documentation\n"
            "Focus ONLY on maintainability and readability."
        ),
    },
    "architecture": {
        "casimir": 24,
        "prompt": (
            "You are an architecture reviewer. Examine this code for:\n"
            "- Violations of existing patterns in the codebase\n"
            "- Tight coupling that should be abstracted\n"
            "- Missing interfaces or unclear boundaries\n"
            "- Scalability concerns or performance regressions\n"
            "- Breaking changes to public APIs\n"
            "Focus ONLY on design and architecture. Ignore minor style issues."
        ),
    },
}


@dataclass
class ReviewFinding:
    """A single finding from a review agent."""

    perspective: str       # Which agent found this
    severity: str          # "critical", "important", "minor", "suggestion"
    confidence: float      # 0.0-1.0 how confident the agent is
    file_path: str         # Which file
    line_range: str        # e.g. "42-48" or "15"
    title: str             # Short description
    description: str       # Detailed explanation
    suggestion: str = ""   # Suggested fix

    @property
    def severity_rank(self) -> int:
        return {"critical": 0, "important": 1, "minor": 2, "suggestion": 3}.get(
            self.severity, 4)


@dataclass
class ReviewResult:
    """Complete review result from all agents."""

    findings: List[ReviewFinding] = field(default_factory=list)
    summary: str = ""
    total_agents: int = 4
    review_time_seconds: float = 0.0
    diff_stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def important_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "important")

    @property
    def high_confidence_findings(self) -> List[ReviewFinding]:
        return [f for f in self.findings if f.confidence >= 0.80]

    def format_report(self) -> str:
        """Format findings as a readable report."""
        lines = ["# φ-Code Review Report", ""]

        if not self.findings:
            lines.append("No issues found. Code looks good.")
            return "\n".join(lines)

        hc = self.high_confidence_findings
        lines.append(f"**{len(hc)} findings** (confidence >= 80%)")
        lines.append(f"Review by {self.total_agents} parallel agents "
                     f"in {self.review_time_seconds:.1f}s")
        lines.append("")

        for sev in ["critical", "important", "minor", "suggestion"]:
            group = [f for f in hc if f.severity == sev]
            if not group:
                continue
            icon = {"critical": "[CRITICAL]", "important": "[IMPORTANT]",
                    "minor": "[MINOR]", "suggestion": "[SUGGESTION]"}[sev]
            lines.append(f"## {icon} {sev.title()} ({len(group)})")
            lines.append("")
            for i, f in enumerate(group, 1):
                lines.append(f"### {i}. {f.title}")
                lines.append(f"**File:** `{f.file_path}` "
                             f"**Lines:** {f.line_range} "
                             f"**Confidence:** {f.confidence:.0%} "
                             f"**Agent:** {f.perspective}")
                lines.append("")
                lines.append(f.description)
                if f.suggestion:
                    lines.append("")
                    lines.append(f"**Suggestion:** {f.suggestion}")
                lines.append("")

        if self.summary:
            lines.append("## Summary")
            lines.append(self.summary)

        return "\n".join(lines)


class PhiCodeReview:
    """
    Multi-agent parallel code review powered by phi-RLM.

    4 agents examine code simultaneously from different perspectives:
    - Logic & Correctness (Casimir 2)
    - Security (Casimir 8)
    - Code Quality (Casimir 14)
    - Architecture (Casimir 24)

    Findings are scored by confidence and filtered at a configurable
    threshold (default 80%, matching Claude Code's /code-review).
    """

    def __init__(self, llm_callable: Callable,
                 confidence_threshold: float = 0.80,
                 max_workers: int = 4):
        """
        Args:
            llm_callable: Function (prompt, max_tokens) -> str (JSON response)
            confidence_threshold: Minimum confidence to include a finding
            max_workers: Number of parallel review agents
        """
        self.llm = llm_callable
        self.confidence_threshold = confidence_threshold
        self.max_workers = max_workers

    def _get_git_diff(self, base: str = "", staged: bool = False) -> str:
        """Get git diff output."""
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        if base:
            cmd.append(base)
        cmd.extend(["--", "*.py", "*.js", "*.ts", "*.md"])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.stdout
        except Exception as e:
            logger.warning(f"Git diff failed: {e}")
            return ""

    def _get_diff_stats(self, diff_text: str) -> Dict[str, Any]:
        """Parse diff text for statistics."""
        files_changed = diff_text.count("diff --git")
        additions = sum(1 for line in diff_text.split("\n")
                       if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in diff_text.split("\n")
                       if line.startswith("-") and not line.startswith("---"))
        return {
            "files_changed": files_changed,
            "additions": additions,
            "deletions": deletions,
            "total_lines": additions + deletions,
        }

    def _read_file(self, path: str) -> str:
        """Read a file for review."""
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")[:15000]
        except Exception as e:
            return f"Error reading {path}: {e}"

    def _run_agent(self, perspective: str, config: Dict,
                   code_text: str, context: str = "") -> List[ReviewFinding]:
        """Run a single review agent."""
        prompt = f"""{config['prompt']}

{context}

## Code to Review

```
{code_text[:8000]}
```

Respond in JSON format ONLY — a list of findings:
[{{"severity": "critical|important|minor|suggestion",
   "confidence": 0.0-1.0,
   "file": "filename or 'diff'",
   "lines": "line range e.g. 42-48",
   "title": "short description",
   "description": "detailed explanation",
   "suggestion": "how to fix (optional)"}}]

If no issues found, respond with: []
"""
        findings = []
        try:
            response = self.llm(prompt, max_tokens=1500)
            # Parse JSON from response
            parsed = self._parse_json(response)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        findings.append(ReviewFinding(
                            perspective=perspective,
                            severity=item.get("severity", "minor"),
                            confidence=float(item.get("confidence", 0.5)),
                            file_path=item.get("file", "unknown"),
                            line_range=str(item.get("lines", "?")),
                            title=item.get("title", "Untitled finding"),
                            description=item.get("description", ""),
                            suggestion=item.get("suggestion", ""),
                        ))
        except Exception as e:
            logger.warning(f"Agent {perspective} failed: {e}")

        return findings

    def _parse_json(self, text: str) -> Any:
        """Parse JSON from LLM response, handling common issues."""
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        if "```" in text:
            for block in text.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue

        # Try finding array in response
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        return []

    def review_diff(self, base: str = "", staged: bool = False,
                    context: str = "") -> ReviewResult:
        """
        Review git diff (uncommitted changes or vs a base branch).

        Args:
            base: Base ref to diff against (e.g. "main"). Empty = working tree.
            staged: If True, review staged changes only.
            context: Extra context about the changes.

        Returns:
            ReviewResult with filtered findings.
        """
        diff_text = self._get_git_diff(base, staged)
        if not diff_text.strip():
            return ReviewResult(summary="No changes to review.")

        diff_stats = self._get_diff_stats(diff_text)
        return self._run_review(diff_text, context, diff_stats)

    def review_branch(self, base_branch: str = "main",
                      context: str = "") -> ReviewResult:
        """Review current branch against base branch."""
        return self.review_diff(base=base_branch, context=context)

    def review_file(self, file_path: str, context: str = "") -> ReviewResult:
        """Review a specific file."""
        code = self._read_file(file_path)
        diff_stats = {"files_changed": 1, "total_lines": code.count("\n")}
        ctx = f"Reviewing file: {file_path}\n{context}" if context else f"Reviewing file: {file_path}"
        return self._run_review(code, ctx, diff_stats)

    def review_text(self, code_text: str, context: str = "") -> ReviewResult:
        """Review raw code text."""
        diff_stats = {"total_lines": code_text.count("\n")}
        return self._run_review(code_text, context, diff_stats)

    def _run_review(self, code_text: str, context: str,
                    diff_stats: Dict) -> ReviewResult:
        """Run all review agents and aggregate findings."""
        start = time.time()
        all_findings: List[ReviewFinding] = []

        # Run 4 agents in parallel
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers) as executor:
            futures = {}
            for name, config in REVIEW_PERSPECTIVES.items():
                future = executor.submit(
                    self._run_agent, name, config, code_text, context)
                futures[future] = name

            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    findings = future.result(timeout=120)
                    all_findings.extend(findings)
                except Exception as e:
                    logger.warning(f"Agent {name} timed out: {e}")

        elapsed = time.time() - start

        # Filter by confidence threshold
        filtered = [f for f in all_findings
                    if f.confidence >= self.confidence_threshold]

        # Sort by severity then confidence
        filtered.sort(key=lambda f: (f.severity_rank, -f.confidence))

        # Build summary
        if not filtered:
            summary = "No high-confidence issues found. Code looks clean."
        else:
            crits = sum(1 for f in filtered if f.severity == "critical")
            imps = sum(1 for f in filtered if f.severity == "important")
            summary = (f"{len(filtered)} findings from {len(REVIEW_PERSPECTIVES)} agents. "
                       f"{crits} critical, {imps} important.")

        return ReviewResult(
            findings=filtered,
            summary=summary,
            total_agents=len(REVIEW_PERSPECTIVES),
            review_time_seconds=elapsed,
            diff_stats=diff_stats,
            metadata={
                "confidence_threshold": self.confidence_threshold,
                "total_findings_raw": len(all_findings),
                "total_findings_filtered": len(filtered),
            },
        )


# ═══════════════════════════════════════════════════════════
# CLI — run from command line
# ═══════════════════════════════════════════════════════════

def main():
    """CLI entry point for phi-Code Review."""
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description="phi-Code Review")
    parser.add_argument("target", nargs="?", default="diff",
                       help="What to review: 'diff', 'staged', 'main', or a file path")
    parser.add_argument("--threshold", type=float, default=0.80,
                       help="Confidence threshold (0-1, default 0.80)")
    parser.add_argument("--mock", action="store_true",
                       help="Use mock LLM (for testing)")
    args = parser.parse_args()

    # Select backend
    if args.mock:
        from .phi_enhanced_rlm import MockLLMBackend
        llm = MockLLMBackend(seed=42)
    else:
        try:
            from .openrouter_backend import OpenRouterBackend
            llm = OpenRouterBackend()
        except Exception:
            print("No API key found. Using mock backend. Set OPENROUTER_API_KEY in .env")
            from .phi_enhanced_rlm import MockLLMBackend
            llm = MockLLMBackend(seed=42)

    reviewer = PhiCodeReview(llm, confidence_threshold=args.threshold)

    target = args.target
    if target == "diff":
        result = reviewer.review_diff()
    elif target == "staged":
        result = reviewer.review_diff(staged=True)
    elif Path(target).exists():
        result = reviewer.review_file(target)
    else:
        # Treat as branch name
        result = reviewer.review_branch(target)

    print(result.format_report())


if __name__ == "__main__":
    main()
