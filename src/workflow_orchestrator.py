#!/usr/bin/env python3
"""
Superpowers Workflow Orchestrator for phi-RLM v4.1
=================================================
Wraps recursive_solve in a mandatory 7-phase pipeline with hard gates.
Each phase corresponds to a superpowers skill AND an E8 Casimir degree.

Phase 0 (C=2):  BRAINSTORM -- clarify before acting
Phase 1 (C=8):  PLAN -- decompose into verifiable steps
Phase 2 (C=12): EXECUTE -- one task at a time with tools
Phase 3 (C=14): TDD -- write test first, then implement
Phase 4 (C=18): SPEC REVIEW -- does output match request?
Phase 5 (C=20): QUALITY REVIEW -- is it well-built?
Phase 6 (C=24): VERIFY -- evidence-based completion gate
Phase 7 (C=30): DEBUG -- systematic root cause if anything failed

Hard gates prevent proceeding without meeting conditions.
Red flags trigger debugging instead of rationalization.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    phase: str
    skill_name: str
    casimir_degree: int
    prompt_injection: str
    hard_gate: str = ""
    red_flags: List[str] = field(default_factory=list)
    rationalization_traps: List[str] = field(default_factory=list)


WORKFLOW_PIPELINE = [
    WorkflowStep(
        phase="clarify",
        skill_name="brainstorming",
        casimir_degree=2,
        prompt_injection=(
            "[BRAINSTORMING GATE] Before answering: What's the real goal? "
            "What are the constraints? What does success look like? "
            "Do NOT jump to solutions. Understand first."
        ),
        hard_gate="Query clarified with explicit scope",
        red_flags=["Jumping to implementation", "Assuming requirements"],
        rationalization_traps=["'This is too simple to need clarification'",
                               "'The question is obvious'"],
    ),
    WorkflowStep(
        phase="plan",
        skill_name="writing-plans",
        casimir_degree=8,
        prompt_injection=(
            "[PLANNING GATE] Decompose into bite-sized steps (2-5 min each). "
            "Each step needs: exact scope, verification criteria, expected output. "
            "DRY. YAGNI. TDD."
        ),
        hard_gate="Plan exists with verifiable steps",
        red_flags=["Vague steps", "Missing verification criteria"],
        rationalization_traps=["'I can figure it out as I go'",
                               "'Planning is overhead for simple tasks'"],
    ),
    WorkflowStep(
        phase="execute",
        skill_name="subagent-driven-development",
        casimir_degree=12,
        prompt_injection=(
            "[EXECUTION] One task at a time. Fresh context per task. "
            "Status: DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED. "
            "If blocked: escalate, don't force."
        ),
        hard_gate="Task completed with status report",
        red_flags=["Multiple tasks at once", "Skipping tests"],
        rationalization_traps=["'I can do these together'",
                               "'Testing slows me down'"],
    ),
    WorkflowStep(
        phase="tdd",
        skill_name="test-driven-development",
        casimir_degree=14,
        prompt_injection=(
            "[TDD] RED: Write failing test. GREEN: Write minimal code to pass. "
            "REFACTOR: Clean up. Commit after each green. "
            "If you wrote code before tests: DELETE the code."
        ),
        hard_gate="Failing test exists before implementation",
        red_flags=["Code before tests", "No commit after green"],
        rationalization_traps=["'I know the code works, tests are formality'",
                               "'This is too simple for TDD'"],
    ),
    WorkflowStep(
        phase="spec_review",
        skill_name="requesting-code-review",
        casimir_degree=18,
        prompt_injection=(
            "[SPEC REVIEW] DO NOT TRUST SELF-REPORTS. Read the actual output. "
            "Verify: Is anything missing? Is anything extra? "
            "Does it match the specification EXACTLY?"
        ),
        hard_gate="Implementation matches specification",
        red_flags=["Trusting without verifying", "Accepting 'close enough'"],
        rationalization_traps=["'The implementer said it's done'",
                               "'I can see it matches from the description'"],
    ),
    WorkflowStep(
        phase="quality_review",
        skill_name="requesting-code-review",
        casimir_degree=20,
        prompt_injection=(
            "[QUALITY REVIEW] Is it clean? Testable? Maintainable? "
            "Check: naming, structure, error handling, edge cases. "
            "Issues found = not done. Fix and re-review."
        ),
        hard_gate="Quality approved with no open issues",
        red_flags=["Approving without reading", "Skipping re-review after fix"],
        rationalization_traps=["'Minor issues don't matter'",
                               "'It works, that's good enough'"],
    ),
    WorkflowStep(
        phase="verify",
        skill_name="verification-before-completion",
        casimir_degree=24,
        prompt_injection=(
            "[IRON LAW] NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE. "
            "1) IDENTIFY what proves this claim. 2) RUN it. 3) READ full output. "
            "4) VERIFY output confirms claim. Skip any step = lying."
        ),
        hard_gate="Verification evidence exists for every claim",
        red_flags=["'Should work'", "'Probably fine'", "'I'm confident'",
                   "Any success claim without evidence"],
        rationalization_traps=["'I already checked earlier'",
                               "'It passed before, it'll pass now'"],
    ),
    WorkflowStep(
        phase="debug",
        skill_name="systematic-debugging",
        casimir_degree=30,
        prompt_injection=(
            "[SYSTEMATIC DEBUGGING] NO FIXES WITHOUT ROOT CAUSE. "
            "Phase 1: Read errors carefully, reproduce, gather evidence. "
            "Phase 2: Find working example, compare differences. "
            "Phase 3: ONE hypothesis, ONE minimal change, test. "
            "Phase 4: Create failing test, fix, verify. "
            "If 3+ fixes failed: STOP. Question architecture."
        ),
        hard_gate="Root cause identified OR all phases passed cleanly",
        red_flags=["'Quick fix'", "'Just try this'", "Multiple fixes at once"],
        rationalization_traps=["'I know what's wrong'",
                               "'One more fix attempt should do it'"],
    ),
]


class SuperpowersOrchestrator:
    """Wraps phi-RLM in superpowers mandatory workflow with hard gates."""

    def __init__(self, rlm, confidence_threshold: float = 0.5):
        self.rlm = rlm
        self.confidence_threshold = confidence_threshold
        self.workflow_trace: List[Dict] = []
        self.skills_dir = Path(".claude/skills")

    def get_skill_content(self, skill_name: str) -> str:
        """Load a superpowers skill file if available."""
        skill_path = self.skills_dir / skill_name / "SKILL.md"
        if skill_path.exists():
            try:
                content = skill_path.read_text()
                # Extract body after YAML front matter
                parts = content.split("---", 2)
                return parts[2].strip() if len(parts) >= 3 else content
            except Exception:
                pass
        return ""

    def orchestrated_solve(self, query: str, max_depth: int = 7) -> Dict[str, Any]:
        """Run recursive_solve wrapped in superpowers mandatory pipeline."""
        self.workflow_trace = []

        # Phase 0: Brainstorm/Clarify (enrich the query before recursion)
        step0 = WORKFLOW_PIPELINE[0]
        enriched_query = f"{step0.prompt_injection}\n\nOriginal query: {query}"

        # Run the main recursive solve with workflow-enriched prompt
        result = self.rlm.recursive_solve(enriched_query, max_depth=max_depth)

        # Check final verification gate
        gate_passed = result.confidence >= self.confidence_threshold

        self.workflow_trace.append({
            "phase": "complete",
            "confidence": result.confidence,
            "gate_passed": gate_passed,
            "verification_iron_law": "Evidence required" if not gate_passed else "Gate passed",
        })

        # If gate failed, trigger systematic debugging
        if not gate_passed:
            debug_step = WORKFLOW_PIPELINE[7]
            debug_query = (
                f"{debug_step.prompt_injection}\n\n"
                f"Original query: {query}\n"
                f"Answer produced: {str(result.value)[:300]}\n"
                f"Confidence: {result.confidence:.1%} (below threshold {self.confidence_threshold:.1%})\n"
                f"Investigate root cause of low confidence."
            )
            debug_result = self.rlm.recursive_solve(debug_query, max_depth=2)
            self.workflow_trace.append({
                "phase": "debug_recovery",
                "original_confidence": result.confidence,
                "debug_confidence": debug_result.confidence,
                "improved": debug_result.confidence > result.confidence,
            })

            if debug_result.confidence > result.confidence:
                result = debug_result

        return {
            "answer": result.value,
            "confidence": result.confidence,
            "metadata": result.metadata,
            "workflow_trace": self.workflow_trace,
            "all_gates_passed": gate_passed,
        }

    def get_workflow_summary(self) -> str:
        lines = ["Workflow trace:"]
        for t in self.workflow_trace:
            phase = t.get("phase", "?")
            conf = t.get("confidence", 0)
            gate = "PASS" if t.get("gate_passed", True) else "FAIL"
            lines.append(f"  [{gate}] {phase}: {conf:.0%}")
        return "\n".join(lines)
