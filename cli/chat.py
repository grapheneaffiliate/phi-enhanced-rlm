#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

"""
PHI-ENHANCED RLM INTERACTIVE CHAT v2.0
======================================
Feature-rich interactive chat with streaming, document support, and more.

Usage:
    python chat.py
    
Commands:
    /file path.txt   - Load query from file
    /repo owner/repo - Analyze a GitHub repo
    /url https://... - Fetch and analyze URL
    /local ./path    - Analyze local directory
    /pdf path.pdf    - Analyze PDF document
    /doc path.docx   - Analyze Word document
    /image path.png  - Describe and analyze image
    /compare s1 s2   - Compare two sources
    /export file.md  - Export last analysis to markdown
    /history         - Show query history
    /depth N         - Set recursion depth (default: 3)
    /stream on|off   - Toggle streaming output
    /help            - Show commands
    /quit or /exit   - Exit chat
"""

import os
import sys
import json
import logging
import re
import time
import uuid
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try rich for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# Add colorama fallback
try:
    from colorama import init, Fore, Style
    init()
    COLOR_QUERY = Fore.CYAN
    COLOR_ANSWER = Fore.GREEN
    COLOR_INFO = Fore.YELLOW
    COLOR_ERROR = Fore.RED
    COLOR_RESET = Style.RESET_ALL
except ImportError:
    COLOR_QUERY = ""
    COLOR_ANSWER = ""
    COLOR_INFO = ""
    COLOR_ERROR = ""
    COLOR_RESET = ""

from openrouter_backend import OpenRouterBackend
from phi_enhanced_rlm import PhiEnhancedRLM


@dataclass
class ConversationTurn:
    """Single conversation turn."""
    role: str  # "user" or "assistant"
    content: str
    confidence: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionState:
    """Chat session state."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    history: List[ConversationTurn] = field(default_factory=list)
    last_result: Optional[Any] = None
    last_query: str = ""
    last_answer: str = ""
    query_count: int = 0
    
    def add_turn(self, role: str, content: str, confidence: Optional[float] = None):
        """Add a conversation turn."""
        self.history.append(ConversationTurn(role, content, confidence))
        self.query_count += 1
    
    def get_context_messages(self, max_turns: int = 6) -> List[Dict[str, str]]:
        """Get recent conversation for context."""
        messages = []
        for turn in self.history[-max_turns:]:
            messages.append({"role": turn.role, "content": turn.content[:500]})
        return messages


class InteractiveChat:
    """Interactive chat interface for PHI-Enhanced RLM v2.0."""

    # Configuration constants
    DEFAULT_DEPTH = 3
    MIN_DEPTH = 0
    MAX_DEPTH = 10
    TOTAL_BUDGET_TOKENS = 16384

    # URL fetching limits
    URL_TIMEOUT_SECONDS = 30
    MAX_URL_CONTENT_SIZE = 30000
    MAX_URL_PROCESSED_SIZE = 20000
    MIN_CONTENT_LENGTH = 100

    # Chunking parameters
    CHUNK_MIN_SIZE = 500
    CHUNK_MAX_SIZE = 1000
    CHUNK_TARGET_SIZE = 800
    MIN_PARAGRAPH_LENGTH = 20
    MAX_CHUNKS = 20
    FALLBACK_CHUNK_SIZE = 2000

    # File safety
    ALLOWED_FILE_EXTENSIONS = {
        '.txt', '.md', '.json', '.py', '.js', '.html', '.css', '.yml', 
        '.yaml', '.xml', '.csv', '.log', '.ts', '.tsx', '.jsx', '.rs',
        '.go', '.java', '.cpp', '.c', '.h', '.rb', '.php'
    }
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

    def __init__(self):
        self.backend = None
        self.rlm = None
        self.depth = self.DEFAULT_DEPTH
        self.context_chunks = self._default_context()
        self.streaming = True  # Default streaming on
        self.session = SessionState()
        
    def _default_context(self) -> List[str]:
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
    
    def initialize(self) -> bool:
        """Initialize backend and RLM."""
        self._print_info("Initializing PHI-Enhanced RLM v2.0...")
        
        try:
            self.backend = OpenRouterBackend()
            self._print_info(f"✓ Backend ready: {self.backend.config.model}")
        except Exception as e:
            self._print_error(f"✗ Backend error: {e}")
            self._print_error("Make sure .env contains OPENROUTER_API_KEY")
            return False
        
        self._reinit_rlm()
        return True
    
    def _reinit_rlm(self):
        """Reinitialize RLM with current settings."""
        self.rlm = PhiEnhancedRLM(
            base_llm_callable=self.backend,
            context_chunks=self.context_chunks,
            total_budget_tokens=self.TOTAL_BUDGET_TOKENS,
            trace_file="chat_trace.jsonl"
        )
        self._print_info(f"✓ RLM ready ({len(self.context_chunks)} chunks, depth={self.depth})")

    def _print_info(self, msg: str):
        """Print info message."""
        if RICH_AVAILABLE and console:
            console.print(f"[yellow]{msg}[/]")
        else:
            print(f"{COLOR_INFO}{msg}{COLOR_RESET}")
    
    def _print_error(self, msg: str):
        """Print error message."""
        if RICH_AVAILABLE and console:
            console.print(f"[red]{msg}[/]")
        else:
            print(f"{COLOR_ERROR}{msg}{COLOR_RESET}")
    
    def _print_answer(self, answer: str, confidence: float):
        """Print the answer with formatting."""
        if RICH_AVAILABLE and console:
            # Try to render as markdown
            try:
                console.print(Panel(
                    Markdown(answer),
                    title="[bold green]PHI-RLM Response[/]",
                    border_style="green"
                ))
            except:
                console.print(f"[green]{answer}[/]")
            console.print(f"\n[dim]Confidence: {confidence:.1%}[/]")
        else:
            print(f"\n{COLOR_ANSWER}PHI-RLM: {answer}{COLOR_RESET}")
            print(f"\n{COLOR_INFO}[Confidence: {confidence:.2%}]{COLOR_RESET}")

    @contextmanager
    def _temporary_context(self, new_chunks: List[str]):
        """Context manager for temporarily replacing context chunks."""
        old_chunks = self.context_chunks
        try:
            self.context_chunks = new_chunks
            self._reinit_rlm()
            yield
        finally:
            self.context_chunks = old_chunks
            self._reinit_rlm()

    def _validate_file_path(self, file_path: str) -> Path:
        """Validate file path for security."""
        try:
            path = Path(file_path).resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid file path: {file_path}")

        if not path.exists():
            raise ValueError(f"File not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        file_size = path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f} MB")

        return path

    def process_query(self, query: str) -> tuple:
        """Process a query and return the answer."""
        try:
            open("chat_trace.jsonl", "w").close()
        except:
            pass
        
        # Add conversation history to context if available
        if self.session.history:
            history_context = "Previous conversation:\n"
            for turn in self.session.history[-4:]:
                role = "User" if turn.role == "user" else "Assistant"
                history_context += f"{role}: {turn.content[:200]}\n"
            
            # Prepend to context
            temp_context = [history_context] + self.context_chunks
            self.rlm = PhiEnhancedRLM(
                base_llm_callable=self.backend,
                context_chunks=temp_context,
                total_budget_tokens=self.TOTAL_BUDGET_TOKENS,
                trace_file="chat_trace.jsonl"
            )
        
        result = self.rlm.recursive_solve(query, max_depth=self.depth)
        
        # Extract answer from JSON if present
        answer = result.value
        try:
            parsed = json.loads(answer)
            if "answer" in parsed:
                answer = parsed["answer"]
        except:
            pass
        
        # Store in session
        self.session.add_turn("user", query)
        self.session.add_turn("assistant", answer, result.confidence)
        self.session.last_result = result
        self.session.last_query = query
        self.session.last_answer = answer

        return answer, result.confidence, result
    
    def handle_command(self, cmd: str) -> bool:
        """Handle special commands. Returns True to continue, False to exit."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if command in ["/quit", "/exit", "/q"]:
            self._print_info("Goodbye!")
            return False
        
        elif command == "/help":
            self._show_help()
        
        elif command == "/depth":
            self._handle_depth(arg)
        
        elif command == "/model":
            self._print_info(f"Model: {self.backend.config.model}")
        
        elif command == "/reset":
            self.context_chunks = self._default_context()
            self.session = SessionState()
            self._reinit_rlm()
            self._print_info("✓ Reset to default context and cleared history")
        
        elif command == "/stream":
            self.streaming = arg.lower() != "off"
            self._print_info(f"✓ Streaming: {'ON' if self.streaming else 'OFF'}")
        
        elif command == "/file":
            self._handle_file(arg)
        
        elif command == "/repo":
            if arg:
                self._analyze_repo(arg)
            else:
                self._print_error("Usage: /repo <owner/repo>")
        
        elif command == "/url":
            if arg:
                self._analyze_url(arg)
            else:
                self._print_error("Usage: /url <url>")
        
        elif command == "/local":
            if arg:
                self._analyze_local(arg)
            else:
                self._print_error("Usage: /local <path>")
        
        elif command == "/pdf":
            if arg:
                self._analyze_pdf(arg)
            else:
                self._print_error("Usage: /pdf <path.pdf>")
        
        elif command == "/doc":
            if arg:
                self._analyze_docx(arg)
            else:
                self._print_error("Usage: /doc <path.docx>")
        
        elif command == "/image":
            if arg:
                self._analyze_image(arg)
            else:
                self._print_error("Usage: /image <path or url>")
        
        elif command == "/compare":
            self._handle_compare(arg)
        
        elif command == "/export":
            self._handle_export(arg)
        
        elif command == "/history":
            self._show_history()
        
        elif command == "/context":
            self._handle_context(arg)
        
        elif command == "/trace":
            self._show_trace()
        
        else:
            self._print_error(f"Unknown command: {command}. Type /help for help.")
        
        return True
    
    def _show_help(self):
        """Show help message."""
        help_text = f"""
PHI-Enhanced RLM Chat v2.0 Commands:

📝 BASIC
  /help             Show this help
  /quit             Exit chat
  /depth <n>        Set recursion depth (current: {self.depth})
  /stream on|off    Toggle streaming (current: {'ON' if self.streaming else 'OFF'})
  /model            Show current model
  /reset            Reset context and history

📂 SOURCES
  /file <path>      Load query from file
  /repo <owner/repo> Analyze GitHub repository
  /url <url>        Fetch and analyze URL
  /local <path>     Analyze local directory
  /pdf <path>       Analyze PDF document
  /doc <path>       Analyze Word document
  /image <path|url> Describe and analyze image

🔍 ANALYSIS
  /compare <s1> <s2> Compare two sources
  /context <file>   Load context chunks from JSON
  /trace            Show reasoning trace

📊 HISTORY
  /history          Show query history
  /export <file.md> Export last analysis to markdown
"""
        if RICH_AVAILABLE and console:
            console.print(Panel(help_text, title="[bold]Help[/]", border_style="blue"))
        else:
            print(help_text)
    
    def _handle_depth(self, arg: str):
        """Handle /depth command."""
        try:
            depth = int(arg)
            if depth < self.MIN_DEPTH or depth > self.MAX_DEPTH:
                self._print_error(f"Depth must be between {self.MIN_DEPTH} and {self.MAX_DEPTH}")
            else:
                self.depth = depth
                self._print_info(f"✓ Depth set to {self.depth}")
        except ValueError:
            self._print_error(f"Usage: /depth <number> (range: {self.MIN_DEPTH}-{self.MAX_DEPTH})")
    
    def _handle_file(self, arg: str):
        """Handle /file command."""
        if not arg:
            self._print_error("Usage: /file <path>")
            return
        
        try:
            path = self._validate_file_path(arg)
            with open(path, "r", encoding="utf-8-sig") as f:
                query = " ".join(f.read().split())
            self._print_info(f"✓ Loaded {len(query)} chars from {path.name}")
            answer, conf, _ = self.process_query(query)
            self._print_answer(answer, conf)
        except Exception as e:
            self._print_error(f"Error: {e}")
    
    def _analyze_repo(self, source: str, query: str = None):
        """Analyze a GitHub repo."""
        try:
            from repo_analyzer import RepoAnalyzer
            self._print_info(f"Analyzing {source}...")
            analyzer = RepoAnalyzer(self.backend, verbose=True)
            if query is None:
                query = "Provide a comprehensive analysis: What does it do? Main components? Architecture?"
            result = analyzer.analyze(source, query, max_depth=self.depth)
            
            if "error" in result:
                self._print_error(result['error'])
            else:
                self.session.last_answer = result['answer']
                self.session.last_query = query
                self._print_answer(result['answer'], result['confidence'])
                self._print_info(f"Files analyzed: {result['files_analyzed']}")
        except Exception as e:
            self._print_error(f"Error: {e}")
    
    def _analyze_url(self, url: str):
        """Analyze a URL with proper extraction."""
        try:
            import urllib.request
            
            self._print_info(f"Fetching {url}...")
            
            if not url.startswith(('http://', 'https://')):
                self._print_error("Invalid URL scheme. Use http:// or https://")
                return
            
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            
            with urllib.request.urlopen(req, timeout=self.URL_TIMEOUT_SECONDS) as resp:
                raw_data = resp.read(self.MAX_URL_CONTENT_SIZE)
                html = raw_data.decode('utf-8', errors='ignore')
            
            # Use trafilatura for better extraction
            try:
                from extractors import extract_web_content
                result = extract_web_content(html, url)
                text = result.text
                title = result.title
                self._print_info(f"✓ Extracted: {title or 'Untitled'} ({len(text)} chars)")
            except ImportError:
                # Fallback to basic extraction
                from repo_analyzer import html_to_text
                text = html_to_text(html)
                self._print_info(f"✓ Extracted {len(text)} chars (basic mode)")
            
            if len(text) < self.MIN_CONTENT_LENGTH:
                self._print_error("Could not extract sufficient text content")
                return
            
            # Create chunks
            content = text[:self.MAX_URL_PROCESSED_SIZE]
            paragraphs = re.split(r'\.\s+|\n\n+', content)
            
            chunks = []
            current_chunk = ""
            for para in paragraphs:
                para = para.strip()
                if not para or len(para) < self.MIN_PARAGRAPH_LENGTH:
                    continue
                if len(current_chunk) + len(para) < self.CHUNK_TARGET_SIZE:
                    current_chunk += " " + para
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            chunks = chunks[:self.MAX_CHUNKS] or [content[:self.FALLBACK_CHUNK_SIZE]]
            
            self._print_info("Analyzing content...")
            
            with self._temporary_context(chunks):
                answer, conf, _ = self.process_query(f"Analyze and summarize the content from: {url}")
                self._print_answer(answer, conf)
                
        except Exception as e:
            self._print_error(f"Error: {e}")
    
    def _analyze_local(self, path: str):
        """Analyze local directory."""
        try:
            from repo_analyzer import RepoAnalyzer
            self._print_info(f"Analyzing {path}...")
            analyzer = RepoAnalyzer(self.backend, verbose=False)
            result = analyzer.analyze(path, "Analyze this codebase.", max_depth=self.depth)
            
            if "error" in result:
                self._print_error(result['error'])
            else:
                self.session.last_answer = result['answer']
                self._print_answer(result['answer'], result['confidence'])
                self._print_info(f"Files: {result['files_analyzed']}")
        except Exception as e:
            self._print_error(f"Error: {e}")
    
    def _analyze_pdf(self, path: str):
        """Analyze PDF document."""
        try:
            from extractors import extract_pdf_content
            
            self._print_info(f"Extracting PDF: {path}...")
            result = extract_pdf_content(path)
            
            self._print_info(f"✓ Extracted: {result.title} ({result.metadata.get('page_count', '?')} pages)")
            
            # Create chunks from PDF text
            text = result.text[:50000]  # Limit
            paragraphs = text.split('\n\n')
            
            chunks = []
            for para in paragraphs:
                if len(para.strip()) > 100:
                    chunks.append(para.strip()[:2000])
                if len(chunks) >= self.MAX_CHUNKS:
                    break
            
            if not chunks:
                chunks = [text[:3000]]
            
            self._print_info(f"Analyzing {len(chunks)} sections...")
            
            with self._temporary_context(chunks):
                answer, conf, _ = self.process_query(f"Analyze this PDF document: {result.title}")
                self._print_answer(answer, conf)
                
        except ImportError:
            self._print_error("PyMuPDF required: pip install PyMuPDF")
        except Exception as e:
            self._print_error(f"Error: {e}")
    
    def _analyze_docx(self, path: str):
        """Analyze Word document."""
        try:
            from extractors import extract_docx_content
            
            self._print_info(f"Extracting document: {path}...")
            result = extract_docx_content(path)
            
            self._print_info(f"✓ Extracted: {result.title} ({result.metadata.get('paragraph_count', '?')} paragraphs)")
            
            # Create chunks
            text = result.text[:50000]
            sections = text.split('\n\n')
            
            chunks = []
            for section in sections:
                if len(section.strip()) > 100:
                    chunks.append(section.strip()[:2000])
                if len(chunks) >= self.MAX_CHUNKS:
                    break
            
            if not chunks:
                chunks = [text[:3000]]
            
            self._print_info(f"Analyzing {len(chunks)} sections...")
            
            with self._temporary_context(chunks):
                answer, conf, _ = self.process_query(f"Analyze this document: {result.title}")
                self._print_answer(answer, conf)
                
        except ImportError:
            self._print_error("python-docx required: pip install python-docx")
        except Exception as e:
            self._print_error(f"Error: {e}")
    
    def _analyze_image(self, path_or_url: str):
        """Analyze image using vision model."""
        try:
            from extractors import describe_image, describe_image_url
            
            self._print_info(f"Analyzing image: {path_or_url}...")
            
            if path_or_url.startswith(('http://', 'https://')):
                description = describe_image_url(path_or_url)
            else:
                description = describe_image(path_or_url)
            
            self._print_info("✓ Image described")
            
            # Use description as context
            chunks = [f"Image description: {description}"]
            
            with self._temporary_context(chunks):
                answer, conf, _ = self.process_query("Based on the image description, provide insights and analysis.")
                
                # Combine outputs
                full_answer = f"**Image Analysis:**\n\n{description}\n\n**Insights:**\n\n{answer}"
                self._print_answer(full_answer, conf)
                self.session.last_answer = full_answer
                
        except Exception as e:
            self._print_error(f"Error: {e}")
    
    def _handle_compare(self, arg: str):
        """Handle /compare command."""
        parts = arg.split()
        if len(parts) < 2:
            self._print_error("Usage: /compare <source1> <source2>")
            return
        
        source1, source2 = parts[0], parts[1]
        
        try:
            from repo_analyzer import RepoAnalyzer
            
            self._print_info(f"Comparing {source1} vs {source2}...")
            
            analyzer = RepoAnalyzer(self.backend, verbose=False)
            
            # Analyze first source
            self._print_info(f"Analyzing {source1}...")
            result1 = analyzer.analyze(source1, "Summarize the main purpose and features.", max_depth=2)
            
            # Analyze second source
            self._print_info(f"Analyzing {source2}...")
            result2 = analyzer.analyze(source2, "Summarize the main purpose and features.", max_depth=2)
            
            if "error" in result1 or "error" in result2:
                self._print_error("Failed to analyze one or both sources")
                return
            
            # Generate comparison
            comparison_context = [
                f"Source 1 ({source1}):\n{result1['answer'][:1500]}",
                f"Source 2 ({source2}):\n{result2['answer'][:1500]}",
            ]
            
            with self._temporary_context(comparison_context):
                answer, conf, _ = self.process_query(
                    "Compare these two sources. List similarities, differences, and provide a recommendation."
                )
                
                full_answer = f"## Comparison: {source1} vs {source2}\n\n{answer}"
                self._print_answer(full_answer, conf)
                self.session.last_answer = full_answer
                
        except Exception as e:
            self._print_error(f"Error: {e}")
    
    def _handle_export(self, arg: str):
        """Export last analysis to markdown."""
        if not arg:
            arg = f"export_{int(time.time())}.md"
        
        if not self.session.last_answer:
            self._print_error("No analysis to export. Run a query first.")
            return
        
        try:
            path = Path(arg)
            
            # Build markdown report
            report = f"""# PHI-Enhanced RLM Analysis Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Session ID:** {self.session.session_id}
**Recursion Depth:** {self.depth}

## Query

{self.session.last_query}

## Analysis

{self.session.last_answer}

## Metadata

- Model: {self.backend.config.model if self.backend else 'N/A'}
- Queries in session: {self.session.query_count}
"""
            
            # Add trace if available
            try:
                with open("chat_trace.jsonl", "r") as f:
                    trace_entries = [json.loads(line) for line in f if line.strip()]
                
                if trace_entries:
                    report += "\n## Reasoning Trace\n\n"
                    report += "| Depth | Confidence | Info Flow | Chunks | Stop Reason |\n"
                    report += "|-------|------------|-----------|--------|-------------|\n"
                    for entry in trace_entries:
                        report += f"| {entry.get('depth', '?')} | {entry.get('confidence', 0):.2%} | {entry.get('info_flow', 0):.1f} | {entry.get('selected_ids', [])} | {entry.get('stop_reason', '?')} |\n"
            except:
                pass
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self._print_info(f"✓ Exported to {path}")
            
        except Exception as e:
            self._print_error(f"Error exporting: {e}")
    
    def _show_history(self):
        """Show query history."""
        if not self.session.history:
            self._print_info("No history yet.")
            return
        
        if RICH_AVAILABLE and console:
            table = Table(title=f"Session History ({self.session.session_id})")
            table.add_column("#", style="dim")
            table.add_column("Role", style="cyan")
            table.add_column("Content", max_width=60)
            table.add_column("Confidence")
            
            for i, turn in enumerate(self.session.history[-20:], 1):
                conf = f"{turn.confidence:.1%}" if turn.confidence else "-"
                table.add_row(str(i), turn.role, turn.content[:60] + "...", conf)
            
            console.print(table)
        else:
            print(f"\nSession History ({self.session.session_id}):")
            print("-" * 60)
            for i, turn in enumerate(self.session.history[-20:], 1):
                role = "You" if turn.role == "user" else "PHI"
                print(f"{i}. [{role}] {turn.content[:60]}...")
    
    def _show_trace(self):
        """Show reasoning trace visualization."""
        try:
            from progress import visualize_confidence_tree
            
            with open("chat_trace.jsonl", "r") as f:
                trace = [json.loads(line) for line in f if line.strip()]
            
            if trace:
                visualize_confidence_tree(trace, use_rich=RICH_AVAILABLE)
            else:
                self._print_info("No trace available. Run a query first.")
        except Exception as e:
            self._print_error(f"Error showing trace: {e}")
    
    def _handle_context(self, arg: str):
        """Handle /context command."""
        if not arg:
            self._print_error("Usage: /context <file.json>")
            return
        
        try:
            path = self._validate_file_path(arg)
            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            if not isinstance(chunks, list):
                raise ValueError("Context file must contain a JSON array")
            self.context_chunks = chunks
            self._reinit_rlm()
            self._print_info(f"✓ Loaded {len(self.context_chunks)} chunks from {path.name}")
        except Exception as e:
            self._print_error(f"Error: {e}")
    
    def run(self):
        """Main chat loop."""
        if RICH_AVAILABLE and console:
            console.print(Panel.fit(
                "[bold blue]PHI-ENHANCED RLM INTERACTIVE CHAT v2.0[/]\n"
                "[dim]Recursive reasoning with φ-Separation Mathematics[/]",
                border_style="blue"
            ))
        else:
            print()
            print("=" * 60)
            print("  PHI-ENHANCED RLM INTERACTIVE CHAT v2.0")
            print("=" * 60)
        
        if not self.initialize():
            return
        
        print()
        print("Ready! Type your questions or /help for commands.")
        print("Press Ctrl+C or type /quit to exit.")
        print()
        
        while True:
            try:
                if RICH_AVAILABLE and console:
                    query = console.input("[cyan]You: [/]").strip()
                else:
                    query = input(f"{COLOR_QUERY}You: {COLOR_RESET}").strip()
                
                if not query:
                    continue
                
                if query.startswith("/"):
                    if not self.handle_command(query):
                        break
                    continue
                
                # Process regular query
                self._print_info("Thinking...")
                answer, confidence, _ = self.process_query(query)
                self._print_answer(answer, confidence)
                print()
                
            except KeyboardInterrupt:
                print()
                self._print_info("Goodbye!")
                break
            except EOFError:
                break
            except Exception as e:
                self._print_error(f"Error: {e}")
                logger.exception("Query processing error")


def main():
    chat = InteractiveChat()
    chat.run()


if __name__ == "__main__":
    main()
