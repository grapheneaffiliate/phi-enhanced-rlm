#!/usr/bin/env python3
"""
OPEN ROUTER LLM BACKEND FOR PHI-ENHANCED RLM
=============================================
Production-ready backend using Open Router API with z-ai/glm-4.7 model.

Setup:
1. Copy .env.template to .env
2. Add your OPENROUTER_API_KEY
3. pip install openai python-dotenv

Usage:
    from openrouter_backend import OpenRouterBackend
    
    backend = OpenRouterBackend()
    response = backend("Your prompt here", max_tokens=256)
"""

import os
import json
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. Run: pip install openai")

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    load_dotenv = None  # type: ignore
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not installed. Run: pip install python-dotenv")


@dataclass
class OpenRouterConfig:
    """Configuration for Open Router API."""
    api_key: str
    model: str = "z-ai/glm-4.7"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: int = 120
    max_retries: int = 3
    
    @classmethod
    def from_env(cls) -> 'OpenRouterConfig':
        """Load configuration from environment variables."""
        if DOTENV_AVAILABLE and load_dotenv is not None:
            load_dotenv()
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment. "
                "Please copy .env.template to .env and add your API key."
            )
        
        return cls(
            api_key=api_key,
            model=os.getenv("DEFAULT_MODEL", "z-ai/glm-4.7"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout=int(os.getenv("OPENROUTER_TIMEOUT", "120")),
            max_retries=int(os.getenv("OPENROUTER_MAX_RETRIES", "3")),
        )


class OpenRouterBackend:
    """
    Open Router LLM Backend for PHI-Enhanced RLM.
    
    Compatible with the PhiEnhancedRLM base_llm_callable interface.
    Returns JSON-formatted responses for structured parsing.
    """
    
    def __init__(self, config: Optional[OpenRouterConfig] = None):
        """
        Initialize the Open Router backend.
        
        Args:
            config: OpenRouterConfig instance. If None, loads from environment.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self.config = config or OpenRouterConfig.from_env()
        
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )
        
        self.call_count = 0
        self.total_tokens = 0
    
    def __call__(self, prompt: str, max_tokens: int = 2048) -> str:
        """
        Call the LLM with the given prompt.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            JSON-formatted string with answer, confidence, and subquestions
        """
        return self.generate(prompt, max_tokens)
    
    def generate(self, prompt: str, max_tokens: int = 256, 
                 temperature: float = 0.7) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            JSON-formatted string compatible with PhiEnhancedRLM
        """
        # Build the system prompt for structured output
        system_prompt = """You are a recursive reasoning assistant. 
For each query, provide:
1. A clear, concise answer
2. Your confidence level (0.0-1.0)
3. Any follow-up subquestions that would help deepen the analysis

Always respond in valid JSON format:
{
    "answer": "Your detailed answer here",
    "confidence": 0.85,
    "subquestions": ["Subquestion 1?", "Subquestion 2?"]
}

If no subquestions are needed, use an empty array: "subquestions": []
"""
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/grapheneaffiliate/phi-enhanced-rlm",
                        "X-Title": "PHI-Enhanced RLM",
                    }
                )
                
                self.call_count += 1
                
                # Track token usage if available
                if hasattr(response, 'usage') and response.usage:
                    self.total_tokens += response.usage.total_tokens
                
                message = response.choices[0].message
                content = message.content or ""
                
                # Some models (like z-ai/glm-4.7) put content in reasoning field
                if not content.strip():
                    reasoning = getattr(message, 'reasoning', None)
                    if reasoning:
                        content = str(reasoning)
                
                # Try to parse as JSON, wrap if needed
                if not content.strip():
                    return json.dumps({
                        "answer": "No response from model",
                        "confidence": 0.5,
                        "subquestions": []
                    })
                
                return self._ensure_json_format(content)
                
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Retry {attempt + 1}/{self.config.max_retries} after {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    # Return error as structured response
                    return json.dumps({
                        "answer": f"Error calling LLM: {str(e)}",
                        "confidence": 0.1,
                        "subquestions": []
                    })
    
    def _ensure_json_format(self, content: str) -> str:
        """
        Ensure the response is valid JSON in the expected format.
        
        Args:
            content: Raw LLM response
            
        Returns:
            Valid JSON string
        """
        # Try to parse as-is
        try:
            parsed = json.loads(content)
            # Validate required fields
            if "answer" in parsed and "confidence" in parsed:
                if "subquestions" not in parsed:
                    parsed["subquestions"] = []
                return json.dumps(parsed)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
        if "```json" in content:
            try:
                json_str = content.split("```json")[1].split("```")[0].strip()
                parsed = json.loads(json_str)
                if "answer" in parsed and "confidence" in parsed:
                    if "subquestions" not in parsed:
                        parsed["subquestions"] = []
                    return json.dumps(parsed)
            except (IndexError, json.JSONDecodeError):
                pass
        
        # Fallback: wrap raw content in structured format
        return json.dumps({
            "answer": content,
            "confidence": 0.7,  # Default confidence for unstructured responses
            "subquestions": []
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "model": self.config.model,
        }


def create_backend(api_key: Optional[str] = None, 
                   model: str = "z-ai/glm-4.7") -> OpenRouterBackend:
    """
    Factory function to create an OpenRouter backend.
    
    Args:
        api_key: Optional API key (uses env if not provided)
        model: Model to use (default: z-ai/glm-4.7)
        
    Returns:
        Configured OpenRouterBackend instance
    """
    if api_key:
        config = OpenRouterConfig(api_key=api_key, model=model)
    else:
        config = OpenRouterConfig.from_env()
        config.model = model
    
    return OpenRouterBackend(config)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Demonstrate OpenRouter backend usage."""
    print("=" * 60)
    print("OPEN ROUTER BACKEND DEMONSTRATION")
    print("=" * 60)
    print()
    
    try:
        backend = OpenRouterBackend()
        print(f"✓ Backend initialized with model: {backend.config.model}")
        print()
        
        # Test query
        test_prompt = """Query: What is the significance of the golden ratio in mathematics?

Context:
The golden ratio φ = 1.618 appears throughout mathematics and nature.
E8 is the largest exceptional Lie group with 248 dimensions.

Please provide a structured analysis."""

        print("Sending test query...")
        response = backend(test_prompt, max_tokens=500)
        
        print("\nResponse:")
        print("-" * 40)
        
        try:
            parsed = json.loads(response)
            print(f"Answer: {parsed.get('answer', 'N/A')[:200]}...")
            print(f"Confidence: {parsed.get('confidence', 'N/A')}")
            print(f"Subquestions: {parsed.get('subquestions', [])}")
        except json.JSONDecodeError:
            print(response[:500])
        
        print()
        print(f"Stats: {backend.get_stats()}")
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nTo use this backend:")
        print("1. Copy .env.template to .env")
        print("2. Add your OPENROUTER_API_KEY")
        print("3. Run: pip install openai python-dotenv")
    except ImportError as e:
        print(f"Import error: {e}")


if __name__ == "__main__":
    demo()
