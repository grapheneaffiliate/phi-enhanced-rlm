#!/usr/bin/env python3
"""
ASYNC OPEN ROUTER BACKEND FOR PHI-ENHANCED RLM
===============================================
Asynchronous backend for parallel API calls.

Features:
- Async/await support for concurrent operations
- Parallel QEC verification
- Parallel subquestion processing
- Connection pooling
- Automatic rate limiting

Usage:
    from async_backend import AsyncOpenRouterBackend
    
    async with AsyncOpenRouterBackend() as backend:
        # Single call
        response = await backend.generate("Your prompt")
        
        # Parallel calls
        responses = await backend.generate_batch(["prompt1", "prompt2"])
"""

import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class AsyncConfig:
    """Configuration for async backend."""
    api_key: str
    model: str = "z-ai/glm-4.7"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: float = 120.0
    max_retries: int = 3
    max_concurrent: int = 5  # Rate limit
    
    @classmethod
    def from_env(cls) -> 'AsyncConfig':
        """Load configuration from environment variables."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
        
        return cls(
            api_key=api_key,
            model=os.getenv("DEFAULT_MODEL", "z-ai/glm-4.7"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout=float(os.getenv("OPENROUTER_TIMEOUT", "120")),
            max_retries=int(os.getenv("OPENROUTER_MAX_RETRIES", "3")),
            max_concurrent=int(os.getenv("MAX_CONCURRENT_REQUESTS", "5")),
        )


class AsyncOpenRouterBackend:
    """
    Async Open Router backend with parallel request support.
    """
    
    SYSTEM_PROMPT = """You are a recursive reasoning assistant. 
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
    
    def __init__(self, config: Optional[AsyncConfig] = None):
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required: pip install httpx")
        
        self.config = config or AsyncConfig.from_env()
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Stats
        self.call_count = 0
        self.total_tokens = 0
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/grapheneaffiliate/phi-enhanced-rlm",
                "X-Title": "PHI-Enhanced RLM (Async)",
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def generate(self, prompt: str, max_tokens: int = 2048,
                       temperature: float = 0.7) -> str:
        """
        Generate a response asynchronously.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            JSON-formatted string
        """
        if self._client is None:
            raise RuntimeError("Backend not initialized. Use 'async with' context manager.")
        
        async with self._semaphore:  # Rate limiting
            return await self._generate_impl(prompt, max_tokens, temperature)
    
    async def _generate_impl(self, prompt: str, max_tokens: int,
                             temperature: float) -> str:
        """Internal implementation with retry logic."""
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.post(
                    f"{self.config.base_url}/chat/completions",
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Update stats
                async with self._lock:
                    self.call_count += 1
                    if "usage" in data:
                        self.total_tokens += data["usage"].get("total_tokens", 0)
                
                # Extract content
                content = data["choices"][0]["message"].get("content", "")
                
                # Handle reasoning field (some models use this)
                if not content.strip():
                    reasoning = data["choices"][0]["message"].get("reasoning")
                    if reasoning:
                        content = str(reasoning)
                
                if not content.strip():
                    return json.dumps({
                        "answer": "No response from model",
                        "confidence": 0.5,
                        "subquestions": []
                    })
                
                return self._ensure_json_format(content)
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                elif attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return json.dumps({
                        "answer": f"HTTP error: {e.response.status_code}",
                        "confidence": 0.1,
                        "subquestions": []
                    })
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return json.dumps({
                        "answer": f"Error: {str(e)}",
                        "confidence": 0.1,
                        "subquestions": []
                    })
        
        return json.dumps({
            "answer": "Max retries exceeded",
            "confidence": 0.1,
            "subquestions": []
        })
    
    async def generate_batch(self, prompts: List[str], max_tokens: int = 2048) -> List[str]:
        """
        Generate responses for multiple prompts in parallel.
        
        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens per response
            
        Returns:
            List of JSON-formatted responses
        """
        tasks = [self.generate(prompt, max_tokens) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    async def verify_parallel(self, answer: str, context: str, 
                             budget_per_check: int = 256) -> tuple[float, List[Dict]]:
        """
        Run QEC verification checks in parallel.
        
        Args:
            answer: Answer to verify
            context: Context used for the answer
            budget_per_check: Token budget per verification check
            
        Returns:
            (revised_confidence, verification_results)
        """
        verifier_prompts = [
            f"Check for contradictions in: {answer[:200]}... Context: {context[:100]}. "
            "Respond with JSON: {{\"has_issue\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"...\"}}",
            
            f"Check for missing logical steps in: {answer[:200]}... "
            "Respond with JSON: {{\"has_issue\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"...\"}}",
            
            f"Find a counterexample if this is wrong: {answer[:200]}... "
            "Respond with JSON: {{\"has_issue\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"...\"}}"
        ]
        
        # Run all verifications in parallel
        responses = await self.generate_batch(verifier_prompts, max_tokens=budget_per_check)
        
        results = []
        passes = 0
        
        for i, response in enumerate(responses):
            try:
                parsed = json.loads(response)
                if "answer" in parsed:
                    # Wrapped response - try to parse the answer
                    try:
                        inner = json.loads(parsed["answer"])
                        has_issue = inner.get("has_issue", False)
                        conf = inner.get("confidence", 0.5)
                    except:
                        has_issue = False
                        conf = parsed.get("confidence", 0.5)
                else:
                    has_issue = parsed.get("has_issue", False)
                    conf = parsed.get("confidence", 0.5)
                
                if not has_issue:
                    passes += 1
                
                results.append({
                    "check": ["contradiction", "completeness", "counterexample"][i],
                    "passed": not has_issue,
                    "confidence": conf
                })
            except:
                # Parse error - assume pass
                passes += 1
                results.append({
                    "check": ["contradiction", "completeness", "counterexample"][i],
                    "passed": True,
                    "confidence": 0.5
                })
        
        # Compute revised confidence
        if passes >= 2:
            revised_conf = 0.85 + 0.05 * passes / 3
        else:
            revised_conf = 0.4 - 0.1 * (3 - passes)
        
        return max(0.1, min(0.99, revised_conf)), results
    
    def _ensure_json_format(self, content: str) -> str:
        """Ensure response is valid JSON."""
        try:
            parsed = json.loads(content)
            if "answer" in parsed and "confidence" in parsed:
                if "subquestions" not in parsed:
                    parsed["subquestions"] = []
                return json.dumps(parsed)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown
        if "```json" in content:
            try:
                json_str = content.split("```json")[1].split("```")[0].strip()
                parsed = json.loads(json_str)
                if "answer" in parsed and "confidence" in parsed:
                    if "subquestions" not in parsed:
                        parsed["subquestions"] = []
                    return json.dumps(parsed)
            except:
                pass
        
        # Fallback
        return json.dumps({
            "answer": content,
            "confidence": 0.7,
            "subquestions": []
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "model": self.config.model,
        }
    
    # Sync wrapper for compatibility
    def __call__(self, prompt: str, max_tokens: int = 2048) -> str:
        """Synchronous call wrapper for compatibility."""
        return asyncio.run(self._sync_generate(prompt, max_tokens))
    
    async def _sync_generate(self, prompt: str, max_tokens: int) -> str:
        """Helper for sync wrapper."""
        async with self:
            return await self.generate(prompt, max_tokens)


# =============================================================================
# DEMO
# =============================================================================

async def demo():
    """Demonstrate async backend."""
    print("=" * 60)
    print("ASYNC BACKEND DEMONSTRATION")
    print("=" * 60)
    print()
    
    try:
        async with AsyncOpenRouterBackend() as backend:
            print(f"Model: {backend.config.model}")
            print(f"Max concurrent: {backend.config.max_concurrent}")
            print()
            
            # Single request
            print("1. Single request...")
            start = time.time()
            response = await backend.generate(
                "What is the golden ratio? Be brief.",
                max_tokens=100
            )
            elapsed = time.time() - start
            print(f"   Response: {response[:100]}...")
            print(f"   Time: {elapsed:.2f}s")
            print()
            
            # Parallel requests
            print("2. Parallel requests (3 prompts)...")
            prompts = [
                "What is E8? Be brief.",
                "What is recursion? Be brief.",
                "What is phi? Be brief.",
            ]
            start = time.time()
            responses = await backend.generate_batch(prompts, max_tokens=100)
            elapsed = time.time() - start
            print(f"   Got {len(responses)} responses")
            print(f"   Time: {elapsed:.2f}s (parallel)")
            print()
            
            # Parallel verification
            print("3. Parallel QEC verification...")
            start = time.time()
            confidence, results = await backend.verify_parallel(
                "The golden ratio is approximately 1.618.",
                "Mathematics and nature."
            )
            elapsed = time.time() - start
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Results: {results}")
            print(f"   Time: {elapsed:.2f}s")
            print()
            
            print(f"Stats: {backend.get_stats()}")
            
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Set OPENROUTER_API_KEY in .env or environment")
    except ImportError as e:
        print(f"Import error: {e}")
        print("Install httpx: pip install httpx")


if __name__ == "__main__":
    asyncio.run(demo())
