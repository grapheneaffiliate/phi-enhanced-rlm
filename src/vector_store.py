#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VECTOR STORE INTEGRATION
========================
ChromaDB-backed vector store for large-scale document retrieval.
Provides the retrieval layer that feeds into RLM for deep analysis.

Usage:
    from vector_store import VectorStore
    
    # Initialize
    store = VectorStore("my_collection")
    
    # Ingest documents
    store.add_documents([
        {"id": "doc1", "text": "...", "metadata": {"source": "file1.pdf"}},
        {"id": "doc2", "text": "...", "metadata": {"source": "file2.pdf"}},
    ])
    
    # Query
    results = store.query("What is the main argument?", top_k=20)
    
    # Feed to RLM
    chunks = [r["text"] for r in results]
    rlm_result = rlm.recursive_solve(query, context_chunks=chunks)
"""

import os
import sys
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

logger = logging.getLogger(__name__)

# Try to import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not installed. Run: pip install chromadb")

# Try to import sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class Document:
    """A document with text and metadata."""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class QueryResult:
    """Result from a vector query."""
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float  # similarity score (higher = more similar)


class VectorStore:
    """
    ChromaDB-backed vector store for document retrieval.
    
    Features:
    - Persistent storage (survives restarts)
    - Automatic chunking of large documents
    - Metadata filtering
    - Hybrid search (semantic + keyword)
    """
    
    DEFAULT_CHUNK_SIZE = 1000  # characters
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality
    
    def __init__(
        self,
        collection_name: str = "default",
        persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the collection
            persist_dir: Directory for persistent storage (default: ~/.cache/phi_rlm/vectordb)
            embedding_model: Sentence transformer model name
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not installed. Run: pip install chromadb")
        
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Set up persistence directory
        if persist_dir is None:
            persist_dir = Path.home() / ".cache" / "phi_rlm" / "vectordb"
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            model_name = embedding_model or self.DEFAULT_MODEL
            try:
                self.embedding_model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        
        # Stats
        self._stats = {
            "documents_added": 0,
            "chunks_added": 0,
            "queries": 0,
        }
    
    def _generate_id(self, text: str, prefix: str = "") -> str:
        """Generate a unique ID for a chunk."""
        hash_val = hashlib.md5(text.encode()).hexdigest()[:12]
        return f"{prefix}_{hash_val}" if prefix else hash_val
    
    def _chunk_text(self, text: str, doc_id: str) -> List[Tuple[str, str]]:
        """
        Split text into overlapping chunks.
        
        Returns:
            List of (chunk_id, chunk_text) tuples
        """
        chunks = []
        start = 0
        chunk_num = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > self.chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunk_id = f"{doc_id}_chunk_{chunk_num}"
            chunks.append((chunk_id, chunk.strip()))
            
            start = end - self.chunk_overlap
            chunk_num += 1
        
        return chunks
    
    def _embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for texts."""
        if self.embedding_model is None:
            return None
        
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk: bool = True,
        show_progress: bool = True,
    ) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of dicts with 'id', 'text', and optional 'metadata'
            chunk: Whether to chunk large documents
            show_progress: Show progress indicator
            
        Returns:
            Number of chunks added
        """
        all_chunks = []
        all_ids = []
        all_metadatas = []
        
        for doc in documents:
            doc_id = doc.get("id", self._generate_id(doc["text"]))
            text = doc["text"]
            metadata = doc.get("metadata", {})
            
            if chunk and len(text) > self.chunk_size:
                chunks = self._chunk_text(text, doc_id)
                for chunk_id, chunk_text in chunks:
                    all_ids.append(chunk_id)
                    all_chunks.append(chunk_text)
                    all_metadatas.append({
                        **metadata,
                        "doc_id": doc_id,
                        "chunk": True,
                    })
            else:
                all_ids.append(doc_id)
                all_chunks.append(text)
                all_metadatas.append({
                    **metadata,
                    "doc_id": doc_id,
                    "chunk": False,
                })
        
        if not all_chunks:
            return 0
        
        # Generate embeddings
        embeddings = self._embed(all_chunks)
        
        # Add to collection
        if embeddings:
            self.collection.add(
                ids=all_ids,
                documents=all_chunks,
                embeddings=embeddings,
                metadatas=all_metadatas,
            )
        else:
            # Let ChromaDB handle embedding
            self.collection.add(
                ids=all_ids,
                documents=all_chunks,
                metadatas=all_metadatas,
            )
        
        self._stats["documents_added"] += len(documents)
        self._stats["chunks_added"] += len(all_chunks)
        
        if show_progress:
            print(f"  Added {len(documents)} documents ({len(all_chunks)} chunks)")
        
        return len(all_chunks)
    
    def add_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add a file to the vector store.
        
        Supports: .txt, .md, .py, .js, .json, .pdf, .docx
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_metadata = metadata or {}
        file_metadata["source"] = str(path)
        file_metadata["filename"] = path.name
        
        # Read content based on file type
        ext = path.suffix.lower()
        
        if ext in ['.txt', '.md', '.py', '.js', '.ts', '.json', '.yaml', '.yml', '.csv']:
            text = path.read_text(encoding='utf-8', errors='ignore')
        elif ext == '.pdf':
            try:
                from extractors import extract_pdf_content
                result = extract_pdf_content(str(path))
                text = result.text
                file_metadata["title"] = result.title
                file_metadata["pages"] = result.metadata.get("page_count", 0)
            except ImportError:
                raise ImportError("PDF extraction requires PyMuPDF. Run: pip install PyMuPDF")
        elif ext == '.docx':
            try:
                from extractors import extract_docx_content
                result = extract_docx_content(str(path))
                text = result.text
                file_metadata["title"] = result.title
            except ImportError:
                raise ImportError("DOCX extraction requires python-docx. Run: pip install python-docx")
        else:
            # Try to read as text
            text = path.read_text(encoding='utf-8', errors='ignore')
        
        return self.add_documents([{
            "id": str(path),
            "text": text,
            "metadata": file_metadata,
        }])
    
    def add_directory(
        self,
        dir_path: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
        show_progress: bool = True,
    ) -> int:
        """
        Add all files from a directory.
        
        Args:
            dir_path: Directory path
            extensions: File extensions to include (default: common text/code files)
            recursive: Search subdirectories
            show_progress: Show progress
            
        Returns:
            Number of chunks added
        """
        if extensions is None:
            extensions = ['.txt', '.md', '.py', '.js', '.ts', '.json', '.pdf', '.docx']
        
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        pattern = "**/*" if recursive else "*"
        files = [f for f in path.glob(pattern) if f.is_file() and f.suffix.lower() in extensions]
        
        if show_progress:
            print(f"Found {len(files)} files to index")
        
        total_chunks = 0
        for i, file_path in enumerate(files):
            try:
                chunks = self.add_file(str(file_path), show_progress=False)
                total_chunks += chunks
                if show_progress and (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(files)} files...")
            except Exception as e:
                logger.warning(f"Failed to add {file_path}: {e}")
        
        if show_progress:
            print(f"Indexed {len(files)} files ({total_chunks} chunks)")
        
        return total_chunks
    
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include_distances: bool = True,
    ) -> List[QueryResult]:
        """
        Query the vector store.
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            where: Metadata filter (ChromaDB where clause)
            include_distances: Include similarity scores
            
        Returns:
            List of QueryResult objects
        """
        self._stats["queries"] += 1
        
        # Generate query embedding
        query_embedding = None
        if self.embedding_model:
            query_embedding = self._embed([query_text])[0]
        
        # Query collection
        if query_embedding:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        else:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        
        # Convert to QueryResult objects
        query_results = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else [""] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
            
            for i, doc_id in enumerate(ids):
                # Convert distance to similarity score (1 - distance for cosine)
                score = 1 - distances[i] if distances[i] else 1.0
                query_results.append(QueryResult(
                    id=doc_id,
                    text=documents[i],
                    metadata=metadatas[i],
                    score=score,
                ))
        
        return query_results
    
    def query_for_rlm(
        self,
        query_text: str,
        top_k: int = 20,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Query and return just the text chunks for RLM processing.
        
        This is the bridge between retrieval and reasoning.
        """
        results = self.query(query_text, top_k=top_k, where=where)
        return [r.text for r in results]
    
    def count(self) -> int:
        """Return the number of items in the collection."""
        return self.collection.count()
    
    def clear(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._stats = {"documents_added": 0, "chunks_added": 0, "queries": 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            **self._stats,
            "total_chunks": self.count(),
            "collection": self.collection_name,
            "persist_dir": str(self.persist_dir),
        }
    
    def delete(self, ids: List[str]):
        """Delete documents by ID."""
        self.collection.delete(ids=ids)


class RLMPipeline:
    """
    Complete pipeline: Vector Store -> RLM Analysis
    
    Usage:
        pipeline = RLMPipeline()
        pipeline.ingest_directory("./documents")
        result = pipeline.analyze("What are the main themes?")
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        persist_dir: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the pipeline."""
        self.store = VectorStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
        )
        
        # Initialize RLM backend
        self.backend = None
        self.rlm = None
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    
    def _init_rlm(self, chunks: List[str]):
        """Initialize RLM with context chunks."""
        if self.backend is None:
            from openrouter_backend import OpenRouterBackend
            self.backend = OpenRouterBackend(api_key=self._api_key)
        
        from phi_enhanced_rlm import PhiEnhancedRLM
        self.rlm = PhiEnhancedRLM(
            base_llm_callable=self.backend,
            context_chunks=chunks,
            total_budget_tokens=8192,
        )
    
    def ingest(self, documents: List[Dict[str, Any]]) -> int:
        """Ingest documents."""
        return self.store.add_documents(documents)
    
    def ingest_file(self, file_path: str) -> int:
        """Ingest a single file."""
        return self.store.add_file(file_path)
    
    def ingest_directory(self, dir_path: str, **kwargs) -> int:
        """Ingest all files from a directory."""
        return self.store.add_directory(dir_path, **kwargs)
    
    def analyze(
        self,
        query: str,
        top_k: int = 20,
        max_depth: int = 3,
        where: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Full pipeline: retrieve relevant chunks, then analyze with RLM.
        
        Args:
            query: Analysis query
            top_k: Number of chunks to retrieve
            max_depth: RLM recursion depth
            where: Metadata filter for retrieval
            verbose: Show progress
            
        Returns:
            Analysis result with answer, confidence, and metadata
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"RLM PIPELINE ANALYSIS")
            print(f"{'='*60}")
            print(f"Query: {query}")
            print(f"Collection: {self.store.collection_name} ({self.store.count()} chunks)")
        
        # Step 1: Retrieve relevant chunks
        if verbose:
            print(f"\n[1] Retrieving top {top_k} relevant chunks...")
        
        results = self.store.query(query, top_k=top_k, where=where)
        chunks = [r.text for r in results]
        
        if verbose:
            print(f"    Retrieved {len(chunks)} chunks")
            print(f"    Top scores: {[f'{r.score:.2f}' for r in results[:5]]}")
        
        if not chunks:
            return {
                "answer": "No relevant documents found in the collection.",
                "confidence": 0.0,
                "chunks_retrieved": 0,
            }
        
        # Step 2: Initialize RLM with chunks
        if verbose:
            print(f"\n[2] Initializing RLM with {len(chunks)} chunks...")
        
        self._init_rlm(chunks)
        
        # Step 3: Run recursive analysis
        if verbose:
            print(f"\n[3] Running recursive analysis (max_depth={max_depth})...")
        
        result = self.rlm.recursive_solve(query, max_depth=max_depth)
        
        # Step 4: Format output
        output = {
            "answer": result.value if hasattr(result, 'value') else str(result),
            "confidence": result.confidence if hasattr(result, 'confidence') else 0.5,
            "chunks_retrieved": len(chunks),
            "top_sources": [
                {"id": r.id, "score": r.score, "metadata": r.metadata}
                for r in results[:5]
            ],
        }
        
        if hasattr(result, 'metadata'):
            output["metadata"] = result.metadata
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"RESULT")
            print(f"{'='*60}")
            print(f"Answer: {output['answer'][:500]}...")
            print(f"Confidence: {output['confidence']:.1%}")
            print(f"Sources: {len(results)} chunks from {len(set(r.metadata.get('doc_id', r.id) for r in results))} documents")
        
        return output
    
    def compare(
        self,
        query: str,
        filter1: Dict[str, Any],
        filter2: Dict[str, Any],
        top_k: int = 15,
    ) -> Dict[str, Any]:
        """
        Compare two subsets of the collection.
        
        Args:
            query: Comparison query
            filter1: Metadata filter for first subset
            filter2: Metadata filter for second subset
            
        Returns:
            Comparison result
        """
        # Get chunks from each subset
        results1 = self.store.query(query, top_k=top_k, where=filter1)
        results2 = self.store.query(query, top_k=top_k, where=filter2)
        
        # Combine with markers
        combined_chunks = []
        combined_chunks.append("=== SOURCE 1 ===")
        combined_chunks.extend([r.text for r in results1])
        combined_chunks.append("=== SOURCE 2 ===")
        combined_chunks.extend([r.text for r in results2])
        
        # Analyze comparison
        comparison_query = f"""Compare the two sources based on this question: {query}

Identify:
1. Key similarities
2. Key differences  
3. Which source is stronger for what aspects
4. Overall assessment"""
        
        self._init_rlm(combined_chunks)
        result = self.rlm.recursive_solve(comparison_query, max_depth=2)
        
        return {
            "answer": result.value if hasattr(result, 'value') else str(result),
            "confidence": result.confidence if hasattr(result, 'confidence') else 0.5,
            "source1_chunks": len(results1),
            "source2_chunks": len(results2),
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for vector store operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RLM Vector Store Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", help="File or directory path")
    ingest_parser.add_argument("--collection", "-c", default="default", help="Collection name")
    ingest_parser.add_argument("--recursive", "-r", action="store_true", help="Recursive directory scan")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the store")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--collection", "-c", default="default", help="Collection name")
    query_parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of results")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Full RLM analysis")
    analyze_parser.add_argument("query", help="Analysis query")
    analyze_parser.add_argument("--collection", "-c", default="default", help="Collection name")
    analyze_parser.add_argument("--top-k", "-k", type=int, default=20, help="Chunks to retrieve")
    analyze_parser.add_argument("--depth", "-d", type=int, default=3, help="RLM recursion depth")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show collection stats")
    stats_parser.add_argument("--collection", "-c", default="default", help="Collection name")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        store = VectorStore(collection_name=args.collection)
        path = Path(args.path)
        if path.is_file():
            chunks = store.add_file(str(path))
        else:
            chunks = store.add_directory(str(path), recursive=args.recursive)
        print(f"\nTotal: {store.count()} chunks in collection '{args.collection}'")
    
    elif args.command == "query":
        store = VectorStore(collection_name=args.collection)
        results = store.query(args.query, top_k=args.top_k)
        print(f"\nTop {len(results)} results:")
        for i, r in enumerate(results):
            print(f"\n[{i+1}] Score: {r.score:.3f}")
            print(f"    ID: {r.id}")
            print(f"    Text: {r.text[:200]}...")
    
    elif args.command == "analyze":
        pipeline = RLMPipeline(collection_name=args.collection)
        result = pipeline.analyze(args.query, top_k=args.top_k, max_depth=args.depth)
        print(f"\n{'='*60}")
        print(json.dumps(result, indent=2, default=str))
    
    elif args.command == "stats":
        store = VectorStore(collection_name=args.collection)
        stats = store.get_stats()
        print(f"\nCollection: {stats['collection']}")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Persist dir: {stats['persist_dir']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
