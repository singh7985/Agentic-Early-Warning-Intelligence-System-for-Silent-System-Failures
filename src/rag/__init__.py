"""
RAG Pipeline Module for Early Warning System

This module implements a complete Retrieval-Augmented Generation (RAG) pipeline
for retrieving similar historical failures and generating contextual explanations.

Components:
-----------
- DocumentChunker: Split documents into semantic chunks
- Embedder: Generate embeddings using sentence-transformers
- VectorStore: FAISS-based vector database with persistence
- Retriever: Similarity search with citation tracking
- KnowledgeBase: Build knowledge base from PHASE 5 degradation data

Example Usage:
-------------
    from src.rag import KnowledgeBase, Retriever
    
    # Build knowledge base
    kb = KnowledgeBase()
    kb.build_from_degradation_data(degradation_periods)
    
    # Retrieve similar incidents
    retriever = Retriever(kb.vector_store)
    results = retriever.search(
        query="High temperature deviation with gradual RUL decrease",
        top_k=5,
        include_citations=True
    )
    
    for result in results:
        print(f"Similarity: {result['score']:.3f}")
        print(f"Incident: {result['text']}")
        print(f"Citation: {result['citation']}")
"""

from src.rag.document_chunker import DocumentChunker
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.rag.knowledge_base import KnowledgeBase

__all__ = [
    'DocumentChunker',
    'Embedder',
    'VectorStore',
    'Retriever',
    'KnowledgeBase',
]

__version__ = '1.0.0'
