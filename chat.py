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


class InteractiveChat:
    """Interactive chat interface for PHI-Enhanced RLM."""
    
    def __init__(self):
        self.backend = None
        self.rlm = None
        self.depth = 3
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
            total_budget_tokens=16384,  # Increased for longer answers
            trace_file="chat_trace.jsonl"
        )
        print(f"{COLOR_INFO}✓ RLM ready ({len(self.context_chunks)} chunks, depth={self.depth}){COLOR_RESET}")
    
    def process_query(self, query: str) -> str:
        """Process a query and return the answer."""
        # Clear trace file for fresh trace
        try:
            open("chat_trace.jsonl", "w").close()
        except:
            pass
        
        result = self.rlm.recursive_solve(query, max_depth=self.depth)
        
        # Extract answer from JSON if present
        answer = result.value
        try:
            parsed = json.loads(answer)
            if "answer" in parsed:
                answer = parsed["answer"]
        except:
            pass
        
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
                self.depth = int(arg)
                print(f"{COLOR_INFO}✓ Depth set to {self.depth}{COLOR_RESET}")
            except:
                print(f"{COLOR_ERROR}Usage: /depth <number>{COLOR_RESET}")
        
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
                    with open(arg, "r", encoding="utf-8-sig") as f:
                        query = " ".join(f.read().split())
                    print(f"{COLOR_INFO}✓ Loaded {len(query)} chars from {arg}{COLOR_RESET}")
                    answer, conf = self.process_query(query)
                    print(f"\n{COLOR_ANSWER}{answer}{COLOR_RESET}")
                    print(f"\n{COLOR_INFO}[Confidence: {conf:.2%}]{COLOR_RESET}")
                except FileNotFoundError:
                    print(f"{COLOR_ERROR}File not found: {arg}{COLOR_RESET}")
                except Exception as e:
                    print(f"{COLOR_ERROR}Error: {e}{COLOR_RESET}")
        
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
                    with open(arg, "r", encoding="utf-8") as f:
                        self.context_chunks = json.load(f)
                    self._reinit_rlm()
                    print(f"{COLOR_INFO}✓ Loaded {len(self.context_chunks)} chunks from {arg}{COLOR_RESET}")
                except Exception as e:
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
        """Analyze a URL."""
        try:
            import urllib.request
            import re
            print(f"{COLOR_INFO}Fetching {url}...{COLOR_RESET}")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode('utf-8', errors='ignore')
            
            # Extract text from HTML
            # Remove script and style elements
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove HTML tags but keep content
            text = re.sub(r'<[^>]+>', ' ', html)
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            # Decode HTML entities
            text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            
            content = text[:30000]  # Limit size
            
            if len(content) < 100:
                print(f"{COLOR_ERROR}Could not extract text content from URL{COLOR_RESET}")
                return
            
            print(f"{COLOR_INFO}✓ Extracted {len(content)} chars of text{COLOR_RESET}")
            
            # Split content into multiple chunks for better φ-Gram selection
            content = content[:20000]  # Limit total size
            
            # Split by paragraphs (double newlines or periods followed by space)
            import re
            paragraphs = re.split(r'\.\s+|\n\n+', content)
            
            # Group into chunks of ~500-1000 chars each
            chunks = []
            current_chunk = ""
            for para in paragraphs:
                para = para.strip()
                if not para or len(para) < 20:
                    continue
                if len(current_chunk) + len(para) < 800:
                    current_chunk += " " + para
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Limit to 15 chunks max
            chunks = chunks[:15]
            
            if not chunks:
                chunks = [content[:2000]]  # Fallback
            
            self.context_chunks = chunks
            self._reinit_rlm()
            
            print(f"{COLOR_INFO}Analyzing content (this may take 30-60 seconds)...{COLOR_RESET}")
            answer, conf = self.process_query(f"Analyze and summarize the main content from this URL: {url}")
            print(f"\n{COLOR_ANSWER}{'=' * 60}\nSUMMARY\n{'=' * 60}{COLOR_RESET}")
            print(f"{COLOR_ANSWER}{answer}{COLOR_RESET}")
            print(f"\n{COLOR_INFO}[Confidence: {conf:.2%}]{COLOR_RESET}")
            
            # Restore default context
            self.context_chunks = self._default_context()
            self._reinit_rlm()
        except Exception as e:
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
