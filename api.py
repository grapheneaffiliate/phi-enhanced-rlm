#!/usr/bin/env python3
"""
PHI-ENHANCED RLM REST API
=========================
FastAPI server with OpenAPI documentation.

Endpoints:
    POST /analyze - Analyze text/repo/URL with RLM
    POST /chat - Chat with context and memory
    GET /status - System status
    POST /compare - Compare two sources
    GET /history - Query history
    
Run:
    uvicorn api:app --reload --port 8000
    
Docs:
    http://localhost:8000/docs (Swagger)
    http://localhost:8000/redoc (ReDoc)
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import RLM components
from openrouter_backend import OpenRouterBackend
from phi_enhanced_rlm import PhiEnhancedRLM

# =============================================================================
# DATA MODELS
# =============================================================================

class AnalyzeRequest(BaseModel):
    """Request model for /analyze endpoint."""
    query: str = Field(..., description="The question to analyze")
    context: Optional[List[str]] = Field(None, description="Custom context chunks")
    source: Optional[str] = Field(None, description="GitHub repo, URL, or local path")
    max_depth: int = Field(3, ge=0, le=10, description="Maximum recursion depth")
    stream: bool = Field(False, description="Enable streaming response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the significance of the golden ratio?",
                "max_depth": 3,
                "stream": False
            }
        }


class ChatRequest(BaseModel):
    """Request model for /chat endpoint."""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")
    context: Optional[List[str]] = Field(None, description="Additional context")
    max_depth: int = Field(2, ge=0, le=10, description="Recursion depth")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Explain E8 symmetry",
                "session_id": "user-123"
            }
        }


class CompareRequest(BaseModel):
    """Request model for /compare endpoint."""
    source1: str = Field(..., description="First source (repo, URL, or path)")
    source2: str = Field(..., description="Second source")
    query: Optional[str] = Field(None, description="Specific comparison query")
    max_depth: int = Field(2, ge=0, le=10)


class AnalyzeResponse(BaseModel):
    """Response model for /analyze."""
    answer: str
    confidence: float
    depth_reached: int
    stop_reason: str
    chunks_used: List[int]
    trace: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: str


class ChatResponse(BaseModel):
    """Response model for /chat."""
    response: str
    confidence: float
    session_id: str
    turn_number: int
    timestamp: str


class CompareResponse(BaseModel):
    """Response model for /compare."""
    source1_summary: str
    source2_summary: str
    comparison: str
    similarities: List[str]
    differences: List[str]
    confidence: float


class StatusResponse(BaseModel):
    """Response model for /status."""
    status: str
    model: str
    version: str
    uptime_seconds: float
    total_requests: int
    active_sessions: int
    cache_stats: Dict[str, Any]


class HistoryEntry(BaseModel):
    """Single history entry."""
    query: str
    answer: str
    confidence: float
    timestamp: str
    session_id: Optional[str]


# =============================================================================
# APPLICATION STATE
# =============================================================================

@dataclass
class AppState:
    """Application state management."""
    backend: Optional[OpenRouterBackend] = None
    start_time: float = 0.0
    request_count: int = 0
    sessions: Dict[str, List[Dict]] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    max_history: int = 100
    
    def add_to_history(self, query: str, answer: str, confidence: float, 
                       session_id: Optional[str] = None):
        """Add entry to history."""
        entry = {
            "query": query,
            "answer": answer[:500],  # Truncate for storage
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id
        }
        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_session(self, session_id: str) -> List[Dict]:
        """Get or create session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]
    
    def add_to_session(self, session_id: str, role: str, content: str):
        """Add message to session."""
        session = self.get_session(session_id)
        session.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        # Keep last 20 turns
        if len(session) > 40:
            self.sessions[session_id] = session[-40:]


state = AppState()


# =============================================================================
# LIFESPAN HANDLER
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""
    # Startup
    print("Initializing PHI-Enhanced RLM API...")
    
    try:
        state.backend = OpenRouterBackend()
        state.start_time = time.time()
        print(f"✓ Backend ready: {state.backend.config.model}")
    except Exception as e:
        print(f"✗ Backend initialization failed: {e}")
        print("API will start but analysis endpoints will fail.")
    
    yield
    
    # Shutdown
    print("Shutting down...")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="PHI-Enhanced RLM API",
    description="""
## PHI-Enhanced Recursive Language Model API

Advanced recursive reasoning with φ-Separation Mathematics.

### Features
- **Recursive Analysis**: Deep multi-level reasoning
- **Confidence Scores**: Calibrated reliability metrics
- **Conversation Memory**: Stateful chat sessions
- **Streaming**: Real-time response streaming
- **Comparison**: Side-by-side source analysis

### Authentication
Set `OPENROUTER_API_KEY` in server environment.
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_default_context() -> List[str]:
    """Get default context chunks."""
    return [
        "The golden ratio φ = (1 + √5)/2 ≈ 1.618 appears throughout mathematics, nature, and art.",
        "E8 is the largest exceptional Lie group with 248 dimensions and 240 roots.",
        "Recursive Language Models decompose complex queries into sub-tasks recursively.",
        "Information theory quantifies uncertainty. Entropy H(X) = -Σ p(x) log p(x).",
        "E8 Casimir degrees are [2, 8, 12, 14, 18, 20, 24, 30].",
        "Deep networks learn hierarchical feature representations at each layer.",
        "Spectral graph theory connects eigenvalues to graph structure properties.",
        "Quantum error correction φ-threshold for fault tolerance is approximately 0.191.",
    ]


async def stream_analysis(rlm: PhiEnhancedRLM, query: str, max_depth: int):
    """Generator for streaming analysis results."""
    import json
    
    # Yield status updates
    yield f"data: {json.dumps({'status': 'starting', 'query': query})}\n\n"
    await asyncio.sleep(0.1)
    
    yield f"data: {json.dumps({'status': 'analyzing', 'depth': 0})}\n\n"
    
    # Run analysis
    result = rlm.recursive_solve(query, max_depth=max_depth)
    
    # Yield result
    response = {
        "status": "complete",
        "answer": result.value,
        "confidence": result.confidence,
        "metadata": result.metadata
    }
    yield f"data: {json.dumps(response)}\n\n"


def fetch_source_content(source: str) -> List[str]:
    """Fetch content from a source and return as chunks."""
    from repo_analyzer import RepoAnalyzer, GitHubFetcher, URLFetcher, LocalFetcher
    
    # Determine fetcher type
    if source.startswith("https://github.com/") or \
       ("/" in source and not source.startswith(("http://", "https://", "/", "."))):
        fetcher = GitHubFetcher()
    elif source.startswith(("http://", "https://")):
        fetcher = URLFetcher()
    else:
        fetcher = LocalFetcher()
    
    try:
        files = fetcher.fetch(source)
    finally:
        if hasattr(fetcher, 'cleanup'):
            fetcher.cleanup()
    
    # Convert to chunks
    chunks = []
    for filepath, content in files.items():
        chunk = f"=== {filepath} ===\n{content[:2000]}"
        chunks.append(chunk)
        if len(chunks) >= 20:
            break
    
    return chunks


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API root - basic info."""
    return {
        "name": "PHI-Enhanced RLM API",
        "version": "2.0.0",
        "docs": "/docs",
        "status": "/status"
    }


@app.get("/status", response_model=StatusResponse, tags=["Info"])
async def get_status():
    """Get system status and statistics."""
    cache_stats = {}
    try:
        from cache import get_sqlite_cache
        cache = get_sqlite_cache()
        stats = cache.get_stats()
        cache_stats = {
            "hits": stats.hits,
            "misses": stats.misses,
            "entries": stats.entry_count,
            "size_kb": stats.total_size_bytes / 1024
        }
    except:
        pass
    
    return StatusResponse(
        status="healthy" if state.backend else "degraded",
        model=state.backend.config.model if state.backend else "not initialized",
        version="2.0.0",
        uptime_seconds=time.time() - state.start_time if state.start_time else 0,
        total_requests=state.request_count,
        active_sessions=len(state.sessions),
        cache_stats=cache_stats
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze(request: AnalyzeRequest):
    """
    Analyze a query using PHI-Enhanced RLM.
    
    Supports:
    - Direct text queries with custom context
    - GitHub repository analysis
    - URL content analysis
    - Local path analysis
    - Streaming responses
    """
    if state.backend is None:
        raise HTTPException(status_code=503, detail="Backend not initialized")
    
    state.request_count += 1
    
    # Get context
    if request.source:
        try:
            context = fetch_source_content(request.source)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch source: {e}")
    elif request.context:
        context = request.context
    else:
        context = get_default_context()
    
    # Handle streaming
    if request.stream:
        rlm = PhiEnhancedRLM(
            base_llm_callable=state.backend,
            context_chunks=context,
            total_budget_tokens=4096,
            trace_file="api_trace.jsonl"
        )
        return StreamingResponse(
            stream_analysis(rlm, request.query, request.max_depth),
            media_type="text/event-stream"
        )
    
    # Regular analysis
    rlm = PhiEnhancedRLM(
        base_llm_callable=state.backend,
        context_chunks=context,
        total_budget_tokens=4096,
        trace_file="api_trace.jsonl"
    )
    
    result = rlm.recursive_solve(request.query, max_depth=request.max_depth)
    
    # Read trace
    trace = []
    try:
        with open("api_trace.jsonl", "r") as f:
            for line in f:
                trace.append(json.loads(line))
    except:
        pass
    
    # Add to history
    state.add_to_history(request.query, result.value, result.confidence)
    
    return AnalyzeResponse(
        answer=result.value,
        confidence=result.confidence,
        depth_reached=result.metadata.get("depth", 0),
        stop_reason=result.metadata.get("stop_reason", "unknown"),
        chunks_used=result.metadata.get("selected_ids", []),
        trace=trace,
        metadata=result.metadata,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat with conversation memory.
    
    Uses session_id to maintain conversation context across requests.
    """
    if state.backend is None:
        raise HTTPException(status_code=503, detail="Backend not initialized")
    
    state.request_count += 1
    
    # Get or create session
    session_id = request.session_id or f"session-{int(time.time())}"
    session = state.get_session(session_id)
    
    # Build context with conversation history
    context = get_default_context()
    
    # Add conversation history to context
    if session:
        history_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}"
            for m in session[-6:]  # Last 3 turns
        ])
        context.insert(0, f"Previous conversation:\n{history_text}")
    
    # Add custom context
    if request.context:
        context.extend(request.context)
    
    # Initialize RLM
    rlm = PhiEnhancedRLM(
        base_llm_callable=state.backend,
        context_chunks=context,
        total_budget_tokens=4096,
        trace_file="chat_trace.jsonl"
    )
    
    # Run analysis
    result = rlm.recursive_solve(request.message, max_depth=request.max_depth)
    
    # Update session
    state.add_to_session(session_id, "user", request.message)
    state.add_to_session(session_id, "assistant", result.value)
    
    # Add to history
    state.add_to_history(request.message, result.value, result.confidence, session_id)
    
    return ChatResponse(
        response=result.value,
        confidence=result.confidence,
        session_id=session_id,
        turn_number=len(session) // 2,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/compare", response_model=CompareResponse, tags=["Analysis"])
async def compare(request: CompareRequest):
    """
    Compare two sources (repos, URLs, or paths).
    
    Analyzes both sources and produces a structured comparison.
    """
    if state.backend is None:
        raise HTTPException(status_code=503, detail="Backend not initialized")
    
    state.request_count += 1
    
    # Fetch both sources
    try:
        chunks1 = fetch_source_content(request.source1)
        chunks2 = fetch_source_content(request.source2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch sources: {e}")
    
    # Analyze first source
    rlm1 = PhiEnhancedRLM(
        base_llm_callable=state.backend,
        context_chunks=chunks1,
        total_budget_tokens=2048
    )
    result1 = rlm1.recursive_solve(
        request.query or "Summarize the main purpose and features.",
        max_depth=request.max_depth
    )
    
    # Analyze second source
    rlm2 = PhiEnhancedRLM(
        base_llm_callable=state.backend,
        context_chunks=chunks2,
        total_budget_tokens=2048
    )
    result2 = rlm2.recursive_solve(
        request.query or "Summarize the main purpose and features.",
        max_depth=request.max_depth
    )
    
    # Generate comparison
    comparison_context = [
        f"Source 1 ({request.source1}):\n{result1.value[:1000]}",
        f"Source 2 ({request.source2}):\n{result2.value[:1000]}",
    ]
    
    rlm_compare = PhiEnhancedRLM(
        base_llm_callable=state.backend,
        context_chunks=comparison_context,
        total_budget_tokens=2048
    )
    
    comparison_query = """Compare these two sources:
1. What are the main similarities?
2. What are the key differences?
3. Which would you recommend and why?

Format response as:
SIMILARITIES:
- point 1
- point 2

DIFFERENCES:
- point 1
- point 2

RECOMMENDATION:
Your recommendation here."""
    
    comparison_result = rlm_compare.recursive_solve(comparison_query, max_depth=1)
    
    # Parse comparison (simple extraction)
    comparison_text = comparison_result.value
    similarities = []
    differences = []
    
    if "SIMILARITIES:" in comparison_text:
        sim_section = comparison_text.split("SIMILARITIES:")[1].split("DIFFERENCES:")[0]
        similarities = [line.strip().lstrip("- ") for line in sim_section.strip().split("\n") if line.strip().startswith("-")]
    
    if "DIFFERENCES:" in comparison_text:
        diff_section = comparison_text.split("DIFFERENCES:")[1].split("RECOMMENDATION:")[0] if "RECOMMENDATION:" in comparison_text else comparison_text.split("DIFFERENCES:")[1]
        differences = [line.strip().lstrip("- ") for line in diff_section.strip().split("\n") if line.strip().startswith("-")]
    
    return CompareResponse(
        source1_summary=result1.value[:500],
        source2_summary=result2.value[:500],
        comparison=comparison_text,
        similarities=similarities or ["No explicit similarities identified"],
        differences=differences or ["No explicit differences identified"],
        confidence=(result1.confidence + result2.confidence + comparison_result.confidence) / 3
    )


@app.get("/history", response_model=List[HistoryEntry], tags=["Info"])
async def get_history(limit: int = 20, session_id: Optional[str] = None):
    """
    Get query history.
    
    Optionally filter by session_id.
    """
    history = state.history
    
    if session_id:
        history = [h for h in history if h.get("session_id") == session_id]
    
    return [
        HistoryEntry(
            query=h["query"],
            answer=h["answer"],
            confidence=h["confidence"],
            timestamp=h["timestamp"],
            session_id=h.get("session_id")
        )
        for h in history[-limit:]
    ]


@app.delete("/session/{session_id}", tags=["Chat"])
async def delete_session(session_id: str):
    """Delete a chat session."""
    if session_id in state.sessions:
        del state.sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the API server."""
    print("=" * 60)
    print("PHI-Enhanced RLM REST API")
    print("=" * 60)
    print()
    print("Starting server...")
    print("Docs: http://localhost:8000/docs")
    print("ReDoc: http://localhost:8000/redoc")
    print()
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
