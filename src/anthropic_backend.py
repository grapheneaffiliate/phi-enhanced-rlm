#!/usr/bin/env python3
"""
Anthropic LLM Backend for φ-Enhanced RLM.

Uses the Anthropic SDK with OAuth auth_token (compatible with Claude Code
Max plan OAuth flow). Falls back to ANTHROPIC_API_KEY if available.

Usage:
    from src.anthropic_backend import AnthropicBackend
    backend = AnthropicBackend()  # auto-detects auth method
    response = backend("Your prompt here", max_tokens=256)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None  # type: ignore
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default session ingress token path (Claude Code OAuth)
_SESSION_TOKEN_PATH = Path("/home/claude/.claude/remote/.session_ingress_token")


def _resolve_auth() -> Dict[str, str]:
    """Resolve authentication: OAuth token > API key > error."""
    # 1. Explicit API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        return {"api_key": api_key}

    # 2. Claude Code OAuth session ingress token
    token_path = os.getenv("CLAUDE_SESSION_INGRESS_TOKEN_FILE", str(_SESSION_TOKEN_PATH))
    if Path(token_path).exists():
        token = Path(token_path).read_text().strip()
        if token:
            return {"auth_token": token}

    raise ValueError(
        "No Anthropic auth found. Set ANTHROPIC_API_KEY or run inside Claude Code "
        "with Max plan OAuth."
    )


class AnthropicBackend:
    """
    Anthropic LLM Backend compatible with PhiEnhancedRLM's base_llm_callable.

    Auto-detects auth: Claude Code OAuth (auth_token) or ANTHROPIC_API_KEY.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        timeout: int = 120,
    ):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        auth = _resolve_auth()
        self.client = anthropic.Anthropic(
            timeout=timeout,
            max_retries=max_retries,
            **auth,
        )
        self.model = model
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def __call__(self, prompt: str, max_tokens: int = 2048) -> str:
        """Call the LLM. Returns JSON string compatible with φ-RLM."""
        return self.generate(prompt, max_tokens)

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """Generate a response from Claude."""
        system = (
            "You are a recursive reasoning assistant. "
            "Always respond in valid JSON:\n"
            '{"answer": "...", "confidence": 0.0-1.0, "subquestions": ["..."]}\n'
            "If no subquestions needed, use an empty array."
        )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )

            self.call_count += 1
            self.total_input_tokens += resp.usage.input_tokens
            self.total_output_tokens += resp.usage.output_tokens

            content = resp.content[0].text if resp.content else ""
            return self._ensure_json(content)

        except Exception as e:
            logger.warning(f"Anthropic API error: {e}")
            return json.dumps({
                "answer": f"Error: {e}",
                "confidence": 0.1,
                "subquestions": [],
            })

    def _ensure_json(self, content: str) -> str:
        """Ensure response is valid JSON in expected format."""
        try:
            parsed = json.loads(content)
            if "answer" in parsed and "confidence" in parsed:
                parsed.setdefault("subquestions", [])
                return json.dumps(parsed)
        except json.JSONDecodeError:
            pass

        # Extract from markdown code block
        if "```" in content:
            for block in content.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    parsed = json.loads(block)
                    if "answer" in parsed:
                        parsed.setdefault("subquestions", [])
                        return json.dumps(parsed)
                except json.JSONDecodeError:
                    continue

        # Fallback: wrap raw text
        return json.dumps({
            "answer": content,
            "confidence": 0.7,
            "subquestions": [],
        })

    def get_stats(self) -> Dict[str, Any]:
        """Usage statistics."""
        return {
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "model": self.model,
        }
