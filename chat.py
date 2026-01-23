#!/usr/bin/env python3
"""
PHI-ENHANCED RLM INTERACTIVE CHAT
==================================
Start once, ask unlimited questions.

Usage:
    python chat.py
    
Then just type your questions. Special commands:
    /file path.txt   - Load query from file
    /repo owner/repo - Analyze a GitHub repo
    /url https://... - Fetch and analyze URL
    /local ./path    - Analyze local directory
    /depth N         - Set recursion depth (default: 3)
    /help            - Show commands
    /quit or /exit   - Exit chat
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from contextlib import contextmanager
from html.parser import HTMLParser
from io import StringIO

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add rich colors if available
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


class HTMLTextExtractor(HTMLParser):
    """Extract clean text from HTML, skipping script/style/nav elements."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {'script', 'style', 'nav', 'header', 'footer', 'iframe', 'noscript'}
        self.current_skip_tag = None

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self.current_skip_tag = tag

    def handle_endtag(self, tag):
        if tag == self.current_skip_tag:
            self.current_skip_tag = None

    def handle_data(self, data):
        if self.current_skip_tag is None:
            text = data.strip()
            if text:
                self.text_parts.append(text)

    def get_text(self):
        return ' '.join(self.text_parts)


class InteractiveChat:
    """Interactive chat interface for PHI-Enhanced RLM."""

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
    MAX_CHUNKS = 15
    FALLBACK_CHUNK_SIZE = 2000

    # File safety
    ALLOWED_FILE_EXTENSIONS = {'.txt', '.md', '.json', '.py', '.js', '.html', '.css', '.yml', '.yaml', '.xml', '.csv', '.log'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

    def __init__(self):
        self.backend = None
        self.rlm = None
        self.depth = self.DEFAULT_DEPTH
        self.context_chunks = self._default_context()
        
    def _default_context(self):
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
    
    def initialize(self):
        """Initialize backend and RLM."""
        print(f"{COLOR_INFO}Initializing PHI-Enhanced RLM...{COLOR_RESET}")
        
        try:
            self.backend = OpenRouterBackend()
            print(f"{COLOR_INFO}✓ Backend ready: {self.backend.config.model}{COLOR_RESET}")
        except Exception as e:
            print(f"{COLOR_ERROR}✗ Backend error: {e}{COLOR_RESET}")
            print(f"{COLOR_ERROR}Make sure .env contains OPENROUTER_API_KEY{COLOR_RESET}")
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
        print(f"{COLOR_INFO}✓ RLM ready ({len(self.context_chunks)} chunks, depth={self.depth}){COLOR_RESET}")

    @contextmanager
    def _temporary_context(self, new_chunks):
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
        """
        Validate file path for security.

        Args:
            file_path: Path to validate

        Returns:
            Validated Path object

        Raises:
            ValueError: If path is invalid or unsafe
        """
        try:
            path = Path(file_path).resolve()
        except (OSError, RuntimeError) as e:
            logger.error(f"Invalid path: {file_path} - {e}")
            raise ValueError(f"Invalid file path: {file_path}")

        # Check if file exists
        if not path.exists():
            raise ValueError(f"File not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        # Check file extension
        if path.suffix.lower() not in self.ALLOWED_FILE_EXTENSIONS:
            raise ValueError(f"File type not allowed: {path.suffix}. Allowed: {', '.join(sorted(self.ALLOWED_FILE_EXTENSIONS))}")

        # Check file size
        file_size = path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f} MB (max: {self.MAX_FILE_SIZE / 1024 / 1024:.1f} MB)")

        # Check for path traversal attempts
        cwd = Path.cwd().resolve()
        try:
            path.relative_to(cwd)
        except ValueError:
            # File is outside current directory - warn but allow
            logger.warning(f"File outside current directory: {path}")

        return path

    def process_query(self, query: str) -> str:
        """Process a query and return the answer."""
        # Clear trace file for fresh trace
        try:
            open("chat_trace.jsonl", "w").close()
        except (IOError, OSError) as e:
            logger.warning(f"Could not clear trace file: {e}")
        
        result = self.rlm.recursive_solve(query, max_depth=self.depth)
        
        # Extract answer from JSON if present
        answer = result.value
        try:
            parsed = json.loads(answer)
            if "answer" in parsed:
                answer = parsed["answer"]
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.debug(f"Answer is not JSON or missing 'answer' key: {e}")

        return answer, result.confidence
    
    def handle_command(self, cmd: str) -> bool:
        """Handle special commands. Returns True to continue, False to exit."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if command in ["/quit", "/exit", "/q"]:
            print(f"{COLOR_INFO}Goodbye!{COLOR_RESET}")
            return False
        
        elif command == "/help":
            print(f"""
{COLOR_INFO}PHI-Enhanced RLM Chat Commands:{COLOR_RESET}
  /file <path>      Load query from file
  /repo <owner/repo> Analyze GitHub repository
  /url <url>        Fetch and analyze URL
  /local <path>     Analyze local directory
  /depth <n>        Set recursion depth (current: {self.depth})
  /context <file>   Load context chunks from JSON file
  /reset            Reset to default context
  /model            Show current model
  /help             Show this help
  /quit             Exit chat
""")
        
        elif command == "/depth":
            try:
                depth = int(arg)
                if depth < self.MIN_DEPTH or depth > self.MAX_DEPTH:
                    print(f"{COLOR_ERROR}Depth must be between {self.MIN_DEPTH} and {self.MAX_DEPTH}{COLOR_RESET}")
                else:
                    self.depth = depth
                    print(f"{COLOR_INFO}✓ Depth set to {self.depth}{COLOR_RESET}")
            except ValueError:
                print(f"{COLOR_ERROR}Usage: /depth <number> (range: {self.MIN_DEPTH}-{self.MAX_DEPTH}){COLOR_RESET}")
        
        elif command == "/model":
            print(f"{COLOR_INFO}Model: {self.backend.config.model}{COLOR_RESET}")
        
        elif command == "/reset":
            self.context_chunks = self._default_context()
            self._reinit_rlm()
            print(f"{COLOR_INFO}✓ Reset to default context{COLOR_RESET}")
        
        elif command == "/file":
            if not arg:
                print(f"{COLOR_ERROR}Usage: /file <path>{COLOR_RESET}")
            else:
                try:
                    path = self._validate_file_path(arg)
                    with open(path, "r", encoding="utf-8-sig") as f:
                        query = " ".join(f.read().split())
                    print(f"{COLOR_INFO}✓ Loaded {len(query)} chars from {path.name}{COLOR_RESET}")
                    answer, conf = self.process_query(query)
                    print(f"\n{COLOR_ANSWER}{answer}{COLOR_RESET}")
                    print(f"\n{COLOR_INFO}[Confidence: {conf:.2%}]{COLOR_RESET}")
                except ValueError as e:
                    print(f"{COLOR_ERROR}{e}{COLOR_RESET}")
                except (IOError, OSError, UnicodeDecodeError) as e:
                    logger.error(f"Error reading file {arg}: {e}")
                    print(f"{COLOR_ERROR}Error reading file: {e}{COLOR_RESET}")
        
        elif command == "/repo":
            if not arg:
                print(f"{COLOR_ERROR}Usage: /repo <owner/repo>{COLOR_RESET}")
            else:
                self._analyze_repo(arg)
        
        elif command == "/url":
            if not arg:
                print(f"{COLOR_ERROR}Usage: /url <url>{COLOR_RESET}")
            else:
                self._analyze_url(arg)
        
        elif command == "/local":
            if not arg:
                print(f"{COLOR_ERROR}Usage: /local <path>{COLOR_RESET}")
            else:
                self._analyze_local(arg)
        
        elif command == "/context":
            if not arg:
                print(f"{COLOR_ERROR}Usage: /context <file.json>{COLOR_RESET}")
            else:
                try:
                    path = self._validate_file_path(arg)
                    with open(path, "r", encoding="utf-8") as f:
                        chunks = json.load(f)
                    if not isinstance(chunks, list):
                        raise ValueError("Context file must contain a JSON array")
                    self.context_chunks = chunks
                    self._reinit_rlm()
                    print(f"{COLOR_INFO}✓ Loaded {len(self.context_chunks)} chunks from {path.name}{COLOR_RESET}")
                except ValueError as e:
                    print(f"{COLOR_ERROR}{e}{COLOR_RESET}")
                except (json.JSONDecodeError, IOError, OSError) as e:
                    logger.error(f"Error loading context from {arg}: {e}")
                    print(f"{COLOR_ERROR}Error loading context: {e}{COLOR_RESET}")
        
        else:
            print(f"{COLOR_ERROR}Unknown command: {command}. Type /help for help.{COLOR_RESET}")
        
        return True
    
    def _analyze_repo(self, source: str, query: str = None):
        """Analyze a GitHub repo."""
        try:
            from repo_analyzer import RepoAnalyzer
            print(f"{COLOR_INFO}Analyzing {source}...{COLOR_RESET}")
            analyzer = RepoAnalyzer(self.backend, verbose=True)  # Show cloned files
            if query is None:
                query = "Provide a comprehensive analysis of this project: What does it do? What are its main components? What is its architecture?"
            result = analyzer.analyze(source, query, max_depth=self.depth)
            if "error" in result:
                print(f"{COLOR_ERROR}{result['error']}{COLOR_RESET}")
            else:
                # Ensure full answer is printed
                answer = result['answer']
                print(f"\n{COLOR_ANSWER}{'=' * 60}\nANALYSIS RESULT\n{'=' * 60}{COLOR_RESET}")
                print(f"{COLOR_ANSWER}{answer}{COLOR_RESET}")
                print(f"\n{COLOR_INFO}[Files: {result['files_analyzed']}, Confidence: {result['confidence']:.2%}]{COLOR_RESET}")
        except ImportError:
            print(f"{COLOR_ERROR}repo_analyzer.py not found{COLOR_RESET}")
        except Exception as e:
            print(f"{COLOR_ERROR}Error: {e}{COLOR_RESET}")
    
    def _analyze_url(self, url: str):
        """Analyze a URL with proper resource limits and HTML parsing."""
        try:
            import urllib.request
            import urllib.error

            print(f"{COLOR_INFO}Fetching {url}...{COLOR_RESET}")

            # Validate URL scheme
            if not url.startswith(('http://', 'https://')):
                print(f"{COLOR_ERROR}Invalid URL scheme. Use http:// or https://{COLOR_RESET}")
                return

            # Fetch with resource limits
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )

            try:
                with urllib.request.urlopen(req, timeout=self.URL_TIMEOUT_SECONDS) as resp:
                    # Check content type
                    content_type = resp.headers.get('Content-Type', '')
                    if 'text/html' not in content_type.lower():
                        logger.warning(f"Non-HTML content type: {content_type}")

                    # Read with size limit
                    raw_data = resp.read(self.MAX_URL_CONTENT_SIZE + 1)
                    if len(raw_data) > self.MAX_URL_CONTENT_SIZE:
                        logger.warning(f"Content truncated at {self.MAX_URL_CONTENT_SIZE} bytes")
                        raw_data = raw_data[:self.MAX_URL_CONTENT_SIZE]

                    html = raw_data.decode('utf-8', errors='ignore')
            except urllib.error.HTTPError as e:
                print(f"{COLOR_ERROR}HTTP Error {e.code}: {e.reason}{COLOR_RESET}")
                logger.error(f"HTTP error fetching {url}: {e}")
                return
            except urllib.error.URLError as e:
                print(f"{COLOR_ERROR}URL Error: {e.reason}{COLOR_RESET}")
                logger.error(f"URL error fetching {url}: {e}")
                return
            except TimeoutError:
                print(f"{COLOR_ERROR}Request timed out after {self.URL_TIMEOUT_SECONDS}s{COLOR_RESET}")
                return

            # Extract text using proper HTML parser
            parser = HTMLTextExtractor()
            try:
                parser.feed(html)
                text = parser.get_text()
            except Exception as e:
                logger.error(f"HTML parsing error: {e}")
                print(f"{COLOR_ERROR}Error parsing HTML content{COLOR_RESET}")
                return

            # Validate content
            if len(text) < self.MIN_CONTENT_LENGTH:
                print(f"{COLOR_ERROR}Could not extract sufficient text content from URL{COLOR_RESET}")
                return

            print(f"{COLOR_INFO}✓ Extracted {len(text)} chars of text{COLOR_RESET}")

            # Split content into chunks for better φ-Gram selection
            content = text[:self.MAX_URL_PROCESSED_SIZE]

            # Split by sentences and paragraphs
            paragraphs = re.split(r'\.\s+|\n\n+', content)

            # Group into chunks
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

            # Limit chunks
            chunks = chunks[:self.MAX_CHUNKS]

            if not chunks:
                chunks = [content[:self.FALLBACK_CHUNK_SIZE]]

            print(f"{COLOR_INFO}Analyzing content (this may take 30-60 seconds)...{COLOR_RESET}")
            sys.stdout.flush()

            # Use context manager for temporary context
            try:
                with self._temporary_context(chunks):
                    answer, conf = self.process_query(f"Analyze and summarize the main content from this URL: {url}")
                    print(f"\n{COLOR_ANSWER}{'=' * 60}\nSUMMARY\n{'=' * 60}{COLOR_RESET}")
                    print(f"{COLOR_ANSWER}{answer}{COLOR_RESET}")
                    print(f"\n{COLOR_INFO}[Confidence: {conf:.2%}]{COLOR_RESET}")
            except KeyboardInterrupt:
                print(f"\n{COLOR_ERROR}Analysis cancelled by user.{COLOR_RESET}")
                return
            except Exception as e:
                logger.error(f"Error during analysis: {e}")
                print(f"\n{COLOR_ERROR}Analysis error: {e}{COLOR_RESET}")
                return

        except Exception as e:
            logger.error(f"Unexpected error in _analyze_url: {e}")
            print(f"{COLOR_ERROR}Error: {e}{COLOR_RESET}")
    
    def _analyze_local(self, path: str):
        """Analyze local directory."""
        try:
            from repo_analyzer import RepoAnalyzer
            print(f"{COLOR_INFO}Analyzing {path}...{COLOR_RESET}")
            analyzer = RepoAnalyzer(self.backend, verbose=False)
            result = analyzer.analyze(path, "Analyze this codebase and explain what it does.", max_depth=self.depth)
            if "error" in result:
                print(f"{COLOR_ERROR}{result['error']}{COLOR_RESET}")
            else:
                print(f"\n{COLOR_ANSWER}{result['answer']}{COLOR_RESET}")
                print(f"\n{COLOR_INFO}[Files: {result['files_analyzed']}, Confidence: {result['confidence']:.2%}]{COLOR_RESET}")
        except Exception as e:
            print(f"{COLOR_ERROR}Error: {e}{COLOR_RESET}")
    
    def run(self):
        """Main chat loop."""
        print()
        print("=" * 60)
        print("  PHI-ENHANCED RLM INTERACTIVE CHAT")
        print("=" * 60)
        print()
        
        if not self.initialize():
            return
        
        print()
        print(f"Ready! Type your questions or /help for commands.")
        print(f"Press Ctrl+C or type /quit to exit.")
        print()
        
        while True:
            try:
                # Get input
                query = input(f"{COLOR_QUERY}You: {COLOR_RESET}").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.startswith("/"):
                    if not self.handle_command(query):
                        break
                    continue
                
                # Process regular query
                print(f"{COLOR_INFO}Thinking...{COLOR_RESET}")
                answer, confidence = self.process_query(query)
                
                print(f"\n{COLOR_ANSWER}PHI-RLM: {answer}{COLOR_RESET}")
                print(f"\n{COLOR_INFO}[Confidence: {confidence:.2%}]{COLOR_RESET}\n")
                
            except KeyboardInterrupt:
                print(f"\n{COLOR_INFO}Goodbye!{COLOR_RESET}")
                break
            except EOFError:
                break
            except Exception as e:
                print(f"{COLOR_ERROR}Error: {e}{COLOR_RESET}")


def main():
    chat = InteractiveChat()
    chat.run()


if __name__ == "__main__":
    main()
