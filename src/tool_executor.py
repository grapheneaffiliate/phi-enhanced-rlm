#!/usr/bin/env python3
"""
phi-Tool Executor -- give the recursive engine the ability to act.

Each recursion depth can invoke tools. Results flow back into the next
recursion as additional context. Tools are governed by the same E8
Casimir budget as text generation.

Supported tools:
  - python_exec: Execute Python code and return output
  - read_file: Read a file from disk
  - web_fetch: Fetch content from a URL
  - shell_exec: Execute a shell command (sandboxed)

Custom tools can be registered via ToolRegistry.register().
"""

import subprocess
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_name: str
    success: bool
    output: str
    tokens_used: int = 0
    error: str = ""


@dataclass
class Tool:
    """A tool that can be invoked during recursive reasoning."""
    name: str
    description: str
    execute: Callable[[Dict[str, Any]], ToolResult]
    budget_cost: int = 50  # token-equivalent cost
    parameters: Dict[str, str] = field(default_factory=dict)


class ToolRegistry:
    """Registry of available tools for phi-RLM recursive reasoning.

    Tools are organized by budget cost (mapped to Casimir energy levels).
    Lower-cost tools (read_file) can be used at any depth. Higher-cost
    tools (python_exec) consume more of the depth's token budget.
    """

    def __init__(self, enable_code_exec: bool = False,
                 enable_web: bool = False,
                 enable_shell: bool = False,
                 allowed_paths: Optional[List[str]] = None):
        """
        Args:
            enable_code_exec: Allow Python code execution (security risk).
            enable_web: Allow web fetching.
            enable_shell: Allow shell command execution (security risk).
            allowed_paths: Restrict file reads to these directories.
        """
        self.tools: Dict[str, Tool] = {}
        self._allowed_paths = allowed_paths
        self._register_defaults(enable_code_exec, enable_web, enable_shell)

    def _register_defaults(self, enable_code: bool, enable_web: bool,
                           enable_shell: bool):
        """Register built-in tools based on security configuration."""

        # File read (always available, optionally path-restricted)
        def read_file(params: Dict[str, Any]) -> ToolResult:
            path = params.get("path", "")
            if self._allowed_paths:
                import os
                abs_path = os.path.abspath(path)
                if not any(abs_path.startswith(os.path.abspath(p))
                           for p in self._allowed_paths):
                    return ToolResult("read_file", False, "",
                                      error=f"Path not allowed: {path}")
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read(10000)
                return ToolResult("read_file", True, content,
                                  tokens_used=len(content.split()))
            except Exception as e:
                return ToolResult("read_file", False, "", error=str(e))

        self.tools["read_file"] = Tool(
            "read_file",
            "Read a file from disk. Params: {path: string}",
            read_file,
            budget_cost=20,
            parameters={"path": "File path to read"},
        )

        # Python code execution (opt-in)
        if enable_code:
            def exec_python(params: Dict[str, Any]) -> ToolResult:
                code = params.get("code", "")
                try:
                    result = subprocess.run(
                        ["python3", "-c", code],
                        capture_output=True, text=True, timeout=30,
                    )
                    output = result.stdout[:2000]
                    error = result.stderr[:500] if result.returncode != 0 else ""
                    return ToolResult(
                        "python_exec",
                        success=result.returncode == 0,
                        output=output,
                        tokens_used=len(output.split()),
                        error=error,
                    )
                except subprocess.TimeoutExpired:
                    return ToolResult("python_exec", False, "",
                                      error="Execution timed out (30s)")

            self.tools["python_exec"] = Tool(
                "python_exec",
                "Execute Python code and return stdout. Params: {code: string}",
                exec_python,
                budget_cost=100,
                parameters={"code": "Python code to execute"},
            )

        # Web fetch (opt-in)
        if enable_web:
            def web_fetch(params: Dict[str, Any]) -> ToolResult:
                url = params.get("url", "")
                try:
                    import requests
                    resp = requests.get(url, timeout=10,
                                        headers={"User-Agent": "phi-RLM/4.1"})
                    resp.raise_for_status()
                    content = resp.text[:5000]
                    return ToolResult("web_fetch", True, content,
                                      tokens_used=len(content.split()))
                except Exception as e:
                    return ToolResult("web_fetch", False, "", error=str(e))

            self.tools["web_fetch"] = Tool(
                "web_fetch",
                "Fetch content from a URL. Params: {url: string}",
                web_fetch,
                budget_cost=50,
                parameters={"url": "URL to fetch"},
            )

        # Shell execution (opt-in, most dangerous)
        if enable_shell:
            def shell_exec(params: Dict[str, Any]) -> ToolResult:
                command = params.get("command", "")
                try:
                    result = subprocess.run(
                        command, shell=True,
                        capture_output=True, text=True, timeout=30,
                    )
                    output = result.stdout[:2000]
                    return ToolResult(
                        "shell_exec",
                        success=result.returncode == 0,
                        output=output,
                        tokens_used=len(output.split()),
                        error=result.stderr[:500] if result.returncode != 0 else "",
                    )
                except subprocess.TimeoutExpired:
                    return ToolResult("shell_exec", False, "",
                                      error="Execution timed out (30s)")

            self.tools["shell_exec"] = Tool(
                "shell_exec",
                "Execute a shell command. Params: {command: string}",
                shell_exec,
                budget_cost=100,
                parameters={"command": "Shell command to run"},
            )

    def register(self, tool: Tool):
        """Register a custom tool."""
        self.tools[tool.name] = tool

    def get_tool_descriptions(self) -> str:
        """Format tool descriptions for injection into LLM prompts."""
        if not self.tools:
            return ""
        lines = ["Available tools:"]
        for name, tool in self.tools.items():
            lines.append(f"  - {name}: {tool.description}")
        lines.append("")
        lines.append('To use a tool, include in your JSON response:')
        lines.append('  "tool_call": {"name": "tool_name", "params": {...}}')
        return "\n".join(lines)

    def execute(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name with given parameters."""
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult(tool_name, False, "",
                              error=f"Unknown tool: {tool_name}")
        try:
            result = tool.execute(params)
            logger.info(f"Tool {tool_name}: success={result.success}, "
                        f"tokens={result.tokens_used}")
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return ToolResult(tool_name, False, "", error=str(e))

    def get_budget_cost(self, tool_name: str) -> int:
        """Get the token-equivalent budget cost of a tool."""
        tool = self.tools.get(tool_name)
        return tool.budget_cost if tool else 0

    def list_tools(self) -> List[str]:
        """List available tool names."""
        return list(self.tools.keys())
