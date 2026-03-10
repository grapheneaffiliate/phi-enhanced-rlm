#!/usr/bin/env python3
"""
phi-Planner -- DAG-based task decomposition before recursion.

Instead of letting each recursion depth generate its own subquestions
ad-hoc, the planner pre-decomposes the query into a directed acyclic
graph (DAG) of steps with explicit dependencies. Steps are executed
in topological order, with independent branches parallelized.

This provides a 6th meta-recursion strategy: "planned".
"""

import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from .phi_separation_novel_mathematics import PHI_INV

logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    """A single step in an execution plan."""
    index: int
    description: str
    depends_on: List[int] = field(default_factory=list)
    tool: str = ""             # Optional tool to use
    result: str = ""           # Filled after execution
    confidence: float = 0.0    # Filled after execution
    completed: bool = False


@dataclass
class ExecutionPlan:
    """A DAG of steps to execute."""
    query: str
    steps: List[PlanStep] = field(default_factory=list)
    total_steps: int = 0

    def get_ready_steps(self) -> List[PlanStep]:
        """Get steps whose dependencies are all completed."""
        return [
            step for step in self.steps
            if not step.completed and all(
                self.steps[dep].completed
                for dep in step.depends_on
                if dep < len(self.steps)
            )
        ]

    def all_completed(self) -> bool:
        return all(step.completed for step in self.steps)


class PhiPlanner:
    """Pre-decompose queries into structured execution plans.

    The planner asks the LLM to generate a DAG of steps before
    any recursive solving begins. This gives the system a roadmap
    rather than relying on depth-first discovery of subquestions.
    """

    def __init__(self, llm_callable, max_steps: int = 8):
        """
        Args:
            llm_callable: LLM backend for plan generation.
            max_steps: Maximum steps in a plan.
        """
        self.llm = llm_callable
        self.max_steps = max_steps

    def create_plan(self, query: str) -> ExecutionPlan:
        """Generate an execution plan for a query.

        Args:
            query: The question to plan for.

        Returns:
            ExecutionPlan with steps and dependencies.
        """
        plan_prompt = (
            f"Decompose this query into {self.max_steps} or fewer ordered "
            f"steps. Each step should be a specific sub-task.\n\n"
            f"Query: {query}\n\n"
            f"Respond in JSON format:\n"
            f'{{"steps": ['
            f'{{"step": "description", "depends_on": [indices]}}, ...'
            f"]}}"
        )

        try:
            response_str = self.llm(plan_prompt, max_tokens=500)
            response = json.loads(response_str)
            raw_steps = response.get("steps", [])

            steps = []
            for i, raw in enumerate(raw_steps[:self.max_steps]):
                if isinstance(raw, dict):
                    deps = raw.get("depends_on", [])
                    # Validate dependency indices
                    valid_deps = [d for d in deps if isinstance(d, int) and 0 <= d < i]
                    steps.append(PlanStep(
                        index=i,
                        description=raw.get("step", raw.get("description", f"Step {i}")),
                        depends_on=valid_deps,
                        tool=raw.get("tool", ""),
                    ))

            if not steps:
                # Fallback: single step
                steps = [PlanStep(index=0, description=query)]

            return ExecutionPlan(
                query=query,
                steps=steps,
                total_steps=len(steps),
            )

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Plan generation failed: {e}, using single-step fallback")
            return ExecutionPlan(
                query=query,
                steps=[PlanStep(index=0, description=query)],
                total_steps=1,
            )

    def execute_plan(self, plan: ExecutionPlan, rlm,
                     max_depth: int = 3) -> Dict[str, Any]:
        """Execute a plan step by step, respecting dependencies.

        Args:
            plan: The execution plan.
            rlm: PhiEnhancedRLM instance for solving steps.
            max_depth: Max recursion depth per step.

        Returns:
            Dict with combined results and metadata.
        """
        iteration = 0
        max_iterations = len(plan.steps) * 2  # Safety limit

        while not plan.all_completed() and iteration < max_iterations:
            ready = plan.get_ready_steps()
            if not ready:
                logger.warning("No ready steps but plan not complete — breaking")
                break

            for step in ready:
                # Build context from completed dependencies
                dep_context = ""
                for dep_idx in step.depends_on:
                    if dep_idx < len(plan.steps):
                        dep = plan.steps[dep_idx]
                        dep_context += f"\n[Step {dep_idx} result]: {dep.result[:200]}"

                step_query = step.description
                if dep_context:
                    step_query = f"{step.description}\n\nPrior results:{dep_context}"

                try:
                    result = rlm.recursive_solve(step_query, max_depth=max_depth)
                    step.result = str(result.value)
                    step.confidence = result.confidence
                except Exception as e:
                    step.result = f"Error: {e}"
                    step.confidence = 0.0

                step.completed = True

            iteration += 1

        # Combine results
        all_results = [s.result for s in plan.steps if s.completed]
        avg_confidence = (
            sum(s.confidence for s in plan.steps) / len(plan.steps)
            if plan.steps else 0.0
        )

        return {
            "answer": "\n\n".join(all_results),
            "confidence": avg_confidence,
            "steps_completed": sum(1 for s in plan.steps if s.completed),
            "total_steps": plan.total_steps,
            "plan": [
                {
                    "step": s.description,
                    "result": s.result[:200],
                    "confidence": s.confidence,
                    "completed": s.completed,
                }
                for s in plan.steps
            ],
        }
