#!/usr/bin/env python3
"""
GITHUB/URL REPOSITORY ANALYZER
==============================
Fetches and analyzes content from GitHub repos, URLs, or local files
using PHI-Enhanced RLM for intelligent analysis.

Usage:
    python repo_analyzer.py https://github.com/owner/repo "What does this do?"
    python repo_analyzer.py owner/repo "Assess the architecture"
    python repo_analyzer.py https://example.com/page.html "Summarize this"
    python repo_analyzer.py ./local/path "Analyze this codebase"
"""

import os
import sys
import json
import argparse
import tempfile
import shutil
import subprocess
import re
import html
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import urllib.request


def html_to_text(html_content: str) -> str:
    """Extract readable text from HTML content."""
    # Remove script and style elements
    text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<head[^>]*>.*?</head>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Replace common block elements with newlines
    text = re.sub(r'<(p|div|h[1-6]|li|tr|br)[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</(p|div|h[1-6]|li|tr)>', '\n', text, flags=re.IGNORECASE)
    
    # Remove remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Clean up whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

# PHI-Enhanced RLM imports
from openrouter_backend import OpenRouterBackend
from phi_enhanced_rlm import PhiEnhancedRLM

# =============================================================================
# CONSTANTS
# =============================================================================

# File extensions to analyze
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
    '.sql', '.sh', '.bash', '.ps1', '.yml', '.yaml', '.json', '.xml', '.toml',
    '.md', '.txt', '.rst', '.html', '.css', '.scss', '.less'
}

# Files to always include
IMPORTANT_FILES = {
    'README.md', 'README.rst', 'README.txt', 'README',
    'package.json', 'requirements.txt', 'setup.py', 'pyproject.toml',
    'Cargo.toml', 'go.mod', 'pom.xml', 'build.gradle',
    'Makefile', 'Dockerfile', 'docker-compose.yml',
    '.gitignore', 'LICENSE', 'CONTRIBUTING.md'
}

# Directories to skip
SKIP_DIRS = {
    '.git', 'node_modules', '__pycache__', '.venv', 'venv', 'env',
    'dist', 'build', 'target', '.idea', '.vscode', '.pytest_cache',
    'coverage', '.nyc_output', 'vendor', 'packages'
}

MAX_FILE_SIZE = 100_000  # 100KB max per file
MAX_TOTAL_SIZE = 500_000  # 500KB total context


# =============================================================================
# CONTENT FETCHERS
# =============================================================================

class ContentFetcher:
    """Base class for content fetchers."""
    
    def fetch(self, source: str) -> Dict[str, str]:
        """Fetch content and return dict of {path: content}."""
        raise NotImplementedError


class GitHubFetcher(ContentFetcher):
    """Fetch content from GitHub repositories."""
    
    def __init__(self, temp_dir: Optional[str] = None, keep_repo: bool = False):
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="repo_analyzer_")
        self.keep_repo = keep_repo
        self.clone_path = None
    
    def fetch(self, source: str) -> Dict[str, str]:
        """Clone repo and extract relevant files."""
        # Parse GitHub URL or owner/repo format
        if source.startswith("https://github.com/"):
            repo_url = source
            parts = source.replace("https://github.com/", "").split("/")
            repo_name = parts[1] if len(parts) > 1 else "repo"
        elif "/" in source and not source.startswith(("http://", "https://", "/", ".")):
            # owner/repo format
            repo_url = f"https://github.com/{source}"
            repo_name = source.split("/")[1]
        else:
            raise ValueError(f"Invalid GitHub source: {source}")
        
        # Clone repository
        clone_path = os.path.join(self.temp_dir, repo_name)
        print(f"  Cloning {repo_url}...")
        
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, clone_path],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                raise RuntimeError(f"Git clone failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Git clone timed out")
        except FileNotFoundError:
            raise RuntimeError("Git not found. Please install git.")
        
        # Extract files
        return self._extract_files(clone_path)
    
    def _extract_files(self, path: str) -> Dict[str, str]:
        """Extract relevant files from directory."""
        files = {}
        total_size = 0
        
        for root, dirs, filenames in os.walk(path):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            
            for filename in filenames:
                filepath = os.path.join(root, filename)
                relpath = os.path.relpath(filepath, path)
                ext = Path(filename).suffix.lower()
                
                # Check if file should be included
                if filename in IMPORTANT_FILES or ext in CODE_EXTENSIONS:
                    try:
                        size = os.path.getsize(filepath)
                        if size > MAX_FILE_SIZE:
                            continue
                        if total_size + size > MAX_TOTAL_SIZE:
                            continue
                        
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            files[relpath] = content
                            total_size += size
                    except Exception:
                        continue
        
        return files
    
    def cleanup(self):
        """Remove temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)


class URLFetcher(ContentFetcher):
    """Fetch content from URLs."""
    
    def fetch(self, source: str) -> Dict[str, str]:
        """Download content from URL."""
        print(f"  Fetching {source}...")
        
        try:
            req = urllib.request.Request(
                source,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                raw_content = response.read().decode('utf-8', errors='ignore')
                
                # Extract text from HTML
                content_type = response.headers.get('Content-Type', '')
                if 'html' in content_type.lower() or raw_content.strip().startswith('<!'):
                    content = html_to_text(raw_content)
                    print(f"  [OK] Extracted {len(content)} chars of text from HTML")
                else:
                    content = raw_content
                
                # Truncate if too large
                if len(content) > MAX_TOTAL_SIZE:
                    content = content[:MAX_TOTAL_SIZE] + "\n[... truncated ...]"
                
                return {"content": content}
        except Exception as e:
            raise RuntimeError(f"Failed to fetch URL: {e}")


class LocalFetcher(ContentFetcher):
    """Fetch content from local files/directories."""
    
    def fetch(self, source: str) -> Dict[str, str]:
        """Read local files."""
        path = Path(source)
        
        if path.is_file():
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return {str(path.name): f.read()[:MAX_FILE_SIZE]}
        elif path.is_dir():
            return self._extract_dir(path)
        else:
            raise ValueError(f"Path not found: {source}")
    
    def _extract_dir(self, path: Path) -> Dict[str, str]:
        """Extract files from directory."""
        files = {}
        total_size = 0
        
        for item in path.rglob("*"):
            if item.is_file():
                relpath = str(item.relative_to(path))
                
                # Skip unwanted dirs
                if any(skip in relpath for skip in SKIP_DIRS):
                    continue
                
                ext = item.suffix.lower()
                if item.name in IMPORTANT_FILES or ext in CODE_EXTENSIONS:
                    try:
                        size = item.stat().st_size
                        if size > MAX_FILE_SIZE or total_size + size > MAX_TOTAL_SIZE:
                            continue
                        
                        content = item.read_text(encoding='utf-8', errors='ignore')
                        files[relpath] = content
                        total_size += size
                    except Exception:
                        continue
        
        return files


# =============================================================================
# ANALYZER
# =============================================================================

class RepoAnalyzer:
    """Main analyzer using PHI-Enhanced RLM."""
    
    def __init__(self, backend: OpenRouterBackend, verbose: bool = True):
        self.backend = backend
        self.verbose = verbose
    
    def analyze(self, source: str, query: str, max_depth: int = 3) -> Dict[str, Any]:
        """
        Analyze content from source with given query.
        
        Args:
            source: GitHub repo, URL, or local path
            query: Analysis question
            max_depth: RLM recursion depth
        
        Returns:
            Analysis result dict
        """
        if self.verbose:
            print("=" * 70)
            print("REPO/URL ANALYZER (PHI-Enhanced RLM)")
            print("=" * 70)
            print()
        
        # Determine source type and fetch content
        if self.verbose:
            print(f"Source: {source}")
            print(f"Query: {query}")
            print()
            print("Fetching content...")
        
        fetcher = self._get_fetcher(source)
        try:
            files = fetcher.fetch(source)
        finally:
            if hasattr(fetcher, 'cleanup'):
                fetcher.cleanup()
        
        if not files:
            return {"error": "No analyzable content found"}
        
        if self.verbose:
            print(f"  [OK] Extracted {len(files)} files")
            for name in list(files.keys())[:10]:
                print(f"    - {name}")
            if len(files) > 10:
                print(f"    ... and {len(files) - 10} more")
            print()
        
        # Create context chunks from files
        chunks = self._create_chunks(files)
        
        if self.verbose:
            print(f"  [OK] Created {len(chunks)} context chunks")
            print()
        
        # Initialize RLM
        if self.verbose:
            print("Running PHI-Enhanced RLM analysis...")
            print()
        
        rlm = PhiEnhancedRLM(
            base_llm_callable=self.backend,
            context_chunks=chunks,
            total_budget_tokens=4096,
            trace_file="analyzer_trace.jsonl"
        )
        
        # Run analysis
        result = rlm.recursive_solve(query, max_depth=max_depth)
        
        # Build response
        response = {
            "source": source,
            "query": query,
            "answer": result.value,
            "confidence": result.confidence,
            "metadata": result.metadata,
            "files_analyzed": len(files),
            "chunks_used": len(chunks)
        }
        
        if self.verbose:
            print("=" * 70)
            print("ANALYSIS RESULT")
            print("=" * 70)
            print()
            print(f"Answer:\n{result.value}")
            print()
            print(f"Confidence: {result.confidence:.4f}")
            print(f"Files analyzed: {len(files)}")
            print(f"Chunks created: {len(chunks)}")
            print()
        
        return response
    
    def _get_fetcher(self, source: str) -> ContentFetcher:
        """Get appropriate fetcher for source."""
        if source.startswith("https://github.com/") or \
           ("/" in source and not source.startswith(("http://", "https://", "/", "."))):
            return GitHubFetcher()
        elif source.startswith(("http://", "https://")):
            return URLFetcher()
        else:
            return LocalFetcher()
    
    def _create_chunks(self, files: Dict[str, str]) -> List[str]:
        """Create context chunks from files."""
        chunks = []
        
        # Add file structure overview
        structure = "Repository structure:\n" + "\n".join(f"- {name}" for name in sorted(files.keys()))
        chunks.append(structure[:2000])
        
        # Add important files first
        for name in IMPORTANT_FILES:
            for filepath, content in files.items():
                if filepath.endswith(name) or filepath == name:
                    chunk = f"=== {filepath} ===\n{content[:3000]}"
                    chunks.append(chunk)
        
        # Add other files
        for filepath, content in files.items():
            if not any(filepath.endswith(imp) for imp in IMPORTANT_FILES):
                chunk = f"=== {filepath} ===\n{content[:2000]}"
                chunks.append(chunk)
                
                if len(chunks) >= 20:  # Limit chunks
                    break
        
        return chunks


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze GitHub repos, URLs, or local files with PHI-Enhanced RLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python repo_analyzer.py https://github.com/owner/repo "What does this project do?"
    python repo_analyzer.py owner/repo "Assess the code quality"
    python repo_analyzer.py https://example.com/page "Summarize this content"
    python repo_analyzer.py ./my/project "Find security issues"
    python repo_analyzer.py owner/repo --output report.json
        """
    )
    parser.add_argument("source", help="GitHub repo (owner/repo or URL), URL, or local path")
    parser.add_argument("query", nargs="?", default="Analyze this project and explain what it does, its architecture, and key features.",
                        help="Analysis query")
    parser.add_argument("--depth", type=int, default=3, help="RLM recursion depth (default: 3)")
    parser.add_argument("--output", "-o", type=str, help="Save output to JSON file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    # Initialize backend
    try:
        backend = OpenRouterBackend()
    except Exception as e:
        print(f"[FAIL] Failed to initialize backend: {e}")
        print("Make sure .env contains OPENROUTER_API_KEY")
        return 1
    
    # Run analysis
    analyzer = RepoAnalyzer(backend, verbose=not args.quiet)
    
    try:
        result = analyzer.analyze(args.source, args.query, max_depth=args.depth)
    except Exception as e:
        print(f"[FAIL] Analysis failed: {e}")
        return 1
    
    # Save output if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"[OK] Output saved to {args.output}")
    
    # Print for quiet mode
    if args.quiet and "answer" in result:
        print(result["answer"])
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
