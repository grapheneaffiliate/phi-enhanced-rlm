#!/usr/bin/env python3
"""
MCP Server -- expose phi-RLM as a Model Context Protocol tool.

This lets Claude Code (and other MCP clients) invoke phi-RLM directly
as a tool, enabling recursive reasoning as a service.

Usage:
    python -m src.mcp_server

Provides tools:
  - recursive_solve: Run phi-enhanced recursive reasoning
  - meta_solve: Strategy-selecting recursive reasoning
  - evolution_status: Check current evolution state
"""

import json
import logging
import sys
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PhiRLMServer:
    """Simple MCP-compatible server for phi-RLM.

    Implements the MCP protocol over stdin/stdout for integration
    with Claude Code and other MCP-compatible clients.
    """

    def __init__(self, context_chunks=None, total_budget: int = 4096):
        self._rlm = None
        self._meta = None
        self._evolution = None
        self._context_chunks = context_chunks or [
            "Mathematical reasoning requires systematic decomposition.",
            "Verify each step with independent calculation.",
            "Consider alternative approaches for verification.",
        ]
        self._total_budget = total_budget

    def _get_rlm(self):
        """Lazy-initialize the RLM engine."""
        if self._rlm is None:
            from .phi_enhanced_rlm import PhiEnhancedRLM, MockLLMBackend
            from .evolution import PhiEvolutionEngine

            # Try to load evolution state
            engine = PhiEvolutionEngine()
            self._evolution = engine

            # Try real backend, fall back to mock
            try:
                from .openrouter_backend import OpenRouterBackend
                backend = OpenRouterBackend()
            except Exception:
                backend = MockLLMBackend()

            self._rlm = PhiEnhancedRLM(
                base_llm_callable=backend,
                context_chunks=self._context_chunks,
                total_budget_tokens=self._total_budget,
                evolution_state=engine.state,
            )
        return self._rlm

    def _get_meta(self):
        """Lazy-initialize the meta-recursion engine."""
        if self._meta is None:
            from .meta_recursion import MetaRecursiveRLM
            self._meta = MetaRecursiveRLM(self._get_rlm())
        return self._meta

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming MCP request."""
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        if method == "initialize":
            return self._handle_initialize(req_id)
        elif method == "tools/list":
            return self._handle_list_tools(req_id)
        elif method == "tools/call":
            return self._handle_tool_call(req_id, params)
        else:
            return self._error(req_id, -32601, f"Unknown method: {method}")

    def _handle_initialize(self, req_id) -> Dict:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "phi-rlm",
                    "version": "4.1.0",
                },
            },
        }

    def _handle_list_tools(self, req_id) -> Dict:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "recursive_solve",
                        "description": (
                            "Run phi-enhanced recursive reasoning on a query. "
                            "Uses E8 Casimir budget allocation, QEC verification, "
                            "and torsion-corrected aggregation."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The question to answer",
                                },
                                "max_depth": {
                                    "type": "integer",
                                    "description": "Maximum recursion depth (1-7)",
                                    "default": 4,
                                },
                            },
                            "required": ["query"],
                        },
                    },
                    {
                        "name": "meta_solve",
                        "description": (
                            "Run meta-recursive reasoning with automatic strategy "
                            "selection. Classifies query type and picks optimal "
                            "strategy (deep/wide/spiral/quick/research)."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The question to answer",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                    {
                        "name": "evolution_status",
                        "description": "Get current evolution engine state.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                ],
            },
        }

    def _handle_tool_call(self, req_id, params: Dict) -> Dict:
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        try:
            if tool_name == "recursive_solve":
                return self._tool_recursive_solve(req_id, arguments)
            elif tool_name == "meta_solve":
                return self._tool_meta_solve(req_id, arguments)
            elif tool_name == "evolution_status":
                return self._tool_evolution_status(req_id)
            else:
                return self._error(req_id, -32602, f"Unknown tool: {tool_name}")
        except Exception as e:
            return self._error(req_id, -32000, str(e))

    def _tool_recursive_solve(self, req_id, args: Dict) -> Dict:
        query = args.get("query", "")
        max_depth = min(args.get("max_depth", 4), 7)

        rlm = self._get_rlm()
        result = rlm.recursive_solve(query, max_depth=max_depth)

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "answer": str(result.value),
                        "confidence": result.confidence,
                        "depth": result.metadata.get("depth", 0),
                        "stop_reason": result.metadata.get("stop_reason", ""),
                    }),
                }],
            },
        }

    def _tool_meta_solve(self, req_id, args: Dict) -> Dict:
        query = args.get("query", "")
        meta = self._get_meta()
        result = meta.meta_solve(query)

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "answer": str(result.value),
                        "confidence": result.confidence,
                        "strategy": result.metadata.get("meta_strategy", ""),
                        "query_type": result.metadata.get("meta_query_type", ""),
                    }),
                }],
            },
        }

    def _tool_evolution_status(self, req_id) -> Dict:
        self._get_rlm()  # Ensure engine is loaded
        summary = self._evolution.get_evolution_summary()

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{
                    "type": "text",
                    "text": json.dumps(summary, default=str),
                }],
            },
        }

    def _error(self, req_id, code: int, message: str) -> Dict:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }

    def run_stdio(self):
        """Run the server over stdin/stdout (MCP transport)."""
        logger.info("phi-RLM MCP server starting on stdio")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
            except json.JSONDecodeError:
                error = self._error(None, -32700, "Parse error")
                sys.stdout.write(json.dumps(error) + "\n")
                sys.stdout.flush()


def main():
    """Entry point for MCP server."""
    logging.basicConfig(level=logging.INFO)
    server = PhiRLMServer()
    server.run_stdio()


if __name__ == "__main__":
    main()
