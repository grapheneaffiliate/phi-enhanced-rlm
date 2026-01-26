#!/usr/bin/env python3
"""
PROGRESS AND STREAMING UTILITIES
================================
Rich progress bars and streaming support for RLM.
"""

import sys
import time
from typing import Optional, Iterator, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

# Try to import rich for beautiful output
try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
    )
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    from rich.text import Text
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class ProgressState:
    """Track progress state for RLM operations."""
    total_chunks: int = 0
    processed_chunks: int = 0
    current_depth: int = 0
    max_depth: int = 0
    current_query: str = ""
    confidence: float = 0.0
    info_flow: float = 0.0
    start_time: float = 0.0
    phase: str = "initializing"
    
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time else 0
    
    @property
    def eta(self) -> Optional[float]:
        if self.processed_chunks == 0 or self.total_chunks == 0:
            return None
        rate = self.processed_chunks / max(self.elapsed, 0.1)
        remaining = self.total_chunks - self.processed_chunks
        return remaining / rate if rate > 0 else None


class RichProgressManager:
    """Rich-based progress display for RLM operations."""
    
    def __init__(self, console: Optional['Console'] = None):
        if not RICH_AVAILABLE:
            raise ImportError("rich library required: pip install rich")
        
        self.console = console or Console()
        self.state = ProgressState()
        self._progress: Optional[Progress] = None
        self._task_id: Optional[int] = None
    
    @contextmanager
    def track_analysis(self, query: str, total_chunks: int, max_depth: int = 5):
        """Context manager for tracking analysis progress."""
        self.state = ProgressState(
            total_chunks=total_chunks,
            max_depth=max_depth,
            current_query=query[:50] + "..." if len(query) > 50 else query,
            start_time=time.time(),
            phase="analyzing"
        )
        
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=False
        )
        
        with self._progress:
            self._task_id = self._progress.add_task(
                f"[cyan]Depth 0/{max_depth}",
                total=total_chunks
            )
            try:
                yield self
            finally:
                self._progress.update(self._task_id, completed=total_chunks)
    
    def update(self, 
               processed: Optional[int] = None,
               depth: Optional[int] = None,
               confidence: Optional[float] = None,
               info_flow: Optional[float] = None,
               phase: Optional[str] = None):
        """Update progress state."""
        if processed is not None:
            self.state.processed_chunks = processed
        if depth is not None:
            self.state.current_depth = depth
        if confidence is not None:
            self.state.confidence = confidence
        if info_flow is not None:
            self.state.info_flow = info_flow
        if phase is not None:
            self.state.phase = phase
        
        if self._progress and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=self.state.processed_chunks,
                description=f"[cyan]Depth {self.state.current_depth}/{self.state.max_depth} | "
                           f"Conf: {self.state.confidence:.1%}"
            )
    
    def show_reasoning_tree(self, trace: list):
        """Display reasoning tree from trace."""
        if not trace:
            return
        
        tree = Tree(f"[bold green]Query: {self.state.current_query}")
        
        depth_nodes = {-1: tree}
        
        for entry in trace:
            depth = entry.get("depth", 0)
            parent_depth = depth - 1
            parent = depth_nodes.get(parent_depth, tree)
            
            conf = entry.get("confidence", 0)
            info = entry.get("info_flow", 0)
            stop = entry.get("stop_reason", "none")
            chunks = entry.get("selected_ids", [])
            
            # Color based on confidence
            if conf >= 0.8:
                color = "green"
            elif conf >= 0.6:
                color = "yellow"
            else:
                color = "red"
            
            label = f"[{color}]D{depth}[/] conf={conf:.2f} info={info:.1f} chunks={chunks}"
            if stop != "none":
                label += f" [dim](stopped: {stop})[/]"
            
            node = parent.add(label)
            depth_nodes[depth] = node
        
        self.console.print(tree)


class SimpleProgressManager:
    """Simple text-based progress for when rich is not available."""
    
    def __init__(self):
        self.state = ProgressState()
    
    @contextmanager
    def track_analysis(self, query: str, total_chunks: int, max_depth: int = 5):
        """Simple progress tracking."""
        self.state = ProgressState(
            total_chunks=total_chunks,
            max_depth=max_depth,
            current_query=query[:50],
            start_time=time.time()
        )
        print(f"Analyzing: {query[:50]}...")
        try:
            yield self
        finally:
            print(f"\nComplete! ({self.state.elapsed:.1f}s)")
    
    def update(self, **kwargs):
        """Update and print progress."""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        
        # Simple progress bar (ASCII-safe)
        pct = self.state.processed_chunks / max(self.state.total_chunks, 1)
        bar_len = 30
        filled = int(pct * bar_len)
        bar = "#" * filled + "-" * (bar_len - filled)
        
        sys.stdout.write(f"\r[{bar}] {pct:.0%} D{self.state.current_depth} conf={self.state.confidence:.1%}")
        sys.stdout.flush()
    
    def show_reasoning_tree(self, trace: list):
        """Simple text tree."""
        if not trace:
            return
        
        print("\nReasoning Tree:")
        for entry in trace:
            depth = entry.get("depth", 0)
            indent = "  " * depth
            conf = entry.get("confidence", 0)
            stop = entry.get("stop_reason", "none")
            print(f"{indent}├─ D{depth}: conf={conf:.2f}, stop={stop}")


def get_progress_manager(use_rich: bool = True) -> Any:
    """Get appropriate progress manager."""
    if use_rich and RICH_AVAILABLE:
        return RichProgressManager()
    return SimpleProgressManager()


# =============================================================================
# STREAMING RESPONSE WRAPPER
# =============================================================================

class StreamingResponse:
    """Wrapper for streaming RLM responses."""
    
    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        """
        Args:
            callback: Function to call with each chunk
        """
        self.callback = callback or (lambda x: print(x, end="", flush=True))
        self.chunks: list = []
        self.complete = False
    
    def write(self, text: str):
        """Write a chunk to the stream."""
        self.chunks.append(text)
        self.callback(text)
    
    def finish(self):
        """Mark stream as complete."""
        self.complete = True
        self.callback("\n")
    
    @property
    def full_text(self) -> str:
        """Get complete response text."""
        return "".join(self.chunks)


def stream_chunks(text: str, chunk_size: int = 10, delay: float = 0.02) -> Iterator[str]:
    """
    Yield text in chunks for simulated streaming.
    
    Args:
        text: Full text to stream
        chunk_size: Characters per chunk
        delay: Delay between chunks
        
    Yields:
        Text chunks
    """
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
        time.sleep(delay)


# =============================================================================
# CONFIDENCE VISUALIZATION
# =============================================================================

def visualize_confidence_tree(trace: list, use_rich: bool = True):
    """
    Visualize confidence at each recursion depth.
    
    Args:
        trace: List of trace entries from RLM
        use_rich: Use rich library for output
    """
    if use_rich and RICH_AVAILABLE:
        console = Console()
        
        table = Table(title="Confidence Visualization")
        table.add_column("Depth", style="cyan", justify="center")
        table.add_column("Confidence", justify="center")
        table.add_column("Info Flow", justify="right")
        table.add_column("Chunks", justify="center")
        table.add_column("Stop Reason", style="dim")
        
        for entry in trace:
            depth = entry.get("depth", 0)
            conf = entry.get("confidence", 0)
            info = entry.get("info_flow", 0)
            chunks = str(entry.get("selected_ids", []))
            stop = entry.get("stop_reason", "none")
            
            # Confidence bar
            bar_len = 20
            filled = int(conf * bar_len)
            if conf >= 0.8:
                color = "green"
            elif conf >= 0.6:
                color = "yellow"
            else:
                color = "red"
            conf_bar = f"[{color}]{'█' * filled}{'░' * (bar_len - filled)}[/] {conf:.1%}"
            
            table.add_row(str(depth), conf_bar, f"{info:.1f}", chunks, stop)
        
        console.print(table)
    else:
        print("\nConfidence by Depth:")
        print("-" * 60)
        for entry in trace:
            depth = entry.get("depth", 0)
            conf = entry.get("confidence", 0)
            bar = "#" * int(conf * 20) + "-" * (20 - int(conf * 20))
            print(f"D{depth}: [{bar}] {conf:.1%}")


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("Progress Utilities Demo")
    print("=" * 50)
    
    # Test progress manager
    pm = get_progress_manager()
    
    # Simulate analysis
    with pm.track_analysis("What is the golden ratio?", total_chunks=10, max_depth=3) as tracker:
        for i in range(10):
            time.sleep(0.2)
            tracker.update(
                processed=i + 1,
                depth=min(i // 3, 3),
                confidence=0.5 + i * 0.05
            )
    
    print("\n")
    
    # Test trace visualization
    sample_trace = [
        {"depth": 0, "confidence": 0.75, "info_flow": 150, "selected_ids": [0, 3, 5], "stop_reason": "none"},
        {"depth": 1, "confidence": 0.82, "info_flow": 45, "selected_ids": [1, 2], "stop_reason": "none"},
        {"depth": 2, "confidence": 0.88, "info_flow": 12, "selected_ids": [4], "stop_reason": "momentum"},
    ]
    
    visualize_confidence_tree(sample_trace)
    
    if RICH_AVAILABLE:
        pm.show_reasoning_tree(sample_trace)
