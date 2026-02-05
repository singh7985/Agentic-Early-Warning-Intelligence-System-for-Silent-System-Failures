"""
Vector Store for RAG Pipeline

FAISS-based vector database with persistence and metadata filtering.
Supports efficient similarity search and CRUD operations.
"""

import logging
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store for embeddings with metadata.
    
    Attributes:
        embedding_dim (int): Dimension of embeddings
        index_type (str): FAISS index type
        metric (str): Distance metric ('cosine', 'l2', 'ip')
        normalize (bool): Normalize vectors before indexing
    """
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = 'Flat',
        metric: str = 'cosine',
        normalize: bool = True
    ):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: FAISS index type
                - 'Flat': Exact search (default, most accurate)
                - 'IVFFlat': Inverted file index (faster for large datasets)
                - 'HNSW': Hierarchical navigable small world (fast approximate)
            metric: Distance metric
                - 'cosine': Cosine similarity (default)
                - 'l2': Euclidean distance
                - 'ip': Inner product
            normalize: Normalize vectors to unit length (recommended for cosine)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        self.normalize = normalize
        
        # Initialize FAISS index
        try:
            import faiss
            self.faiss = faiss
            
            # Create index based on type and metric
            if index_type == 'Flat':
                if metric == 'cosine' or metric == 'ip':
                    self.index = faiss.IndexFlatIP(embedding_dim)
                elif metric == 'l2':
                    self.index = faiss.IndexFlatL2(embedding_dim)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
            
            elif index_type == 'HNSW':
                # HNSW index for fast approximate search
                self.index = faiss.IndexHNSWFlat(embedding_dim, 32)
                if metric == 'cosine' or metric == 'ip':
                    self.index.metric_type = faiss.METRIC_INNER_PRODUCT
                elif metric == 'l2':
                    self.index.metric_type = faiss.METRIC_L2
            
            elif index_type == 'IVFFlat':
                # IVF index requires training
                quantizer = faiss.IndexFlatL2(embedding_dim)
                nlist = 100  # Number of clusters
                self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
                self._needs_training = True
            
            else:
                raise ValueError(f"Unknown index type: {index_type}")
            
            logger.info(
                f"Initialized VectorStore: dim={embedding_dim}, "
                f"type={index_type}, metric={metric}"
            )
        
        except ImportError:
            logger.error(
                "faiss-cpu not installed. "
                "Install with: pip install faiss-cpu"
            )
            raise
        
        # Storage for metadata
        self.documents = []  # List of original documents
        self.metadata = []   # List of metadata dicts
        self._id_to_idx = {}  # Map document IDs to indices
        self._next_id = 0
    
    def add(
        self,
        embeddings: np.ndarray,
        documents: List[Dict],
        ids: Optional[List[int]] = None
    ) -> List[int]:
        """
        Add embeddings and documents to the store.
        
        Args:
            embeddings: Embedding array of shape (n, embedding_dim)
            documents: List of document dictionaries
            ids: Optional list of document IDs (generated if not provided)
        
        Returns:
            List of assigned document IDs
        """
        if len(embeddings) != len(documents):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and documents "
                f"({len(documents)}) must have same length"
            )
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match "
                f"store dimension ({self.embedding_dim})"
            )
        
        # Normalize if needed
        if self.normalize and (self.metric == 'cosine' or self.metric == 'ip'):
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms
        
        # Convert to float32 for FAISS
        embeddings = embeddings.astype('float32')
        
        # Train index if needed (IVFFlat)
        if hasattr(self, '_needs_training') and self._needs_training:
            if not self.index.is_trained:
                logger.info("Training IVFFlat index...")
                self.index.train(embeddings)
                self._needs_training = False
        
        # Generate IDs if not provided
        if ids is None:
            ids = list(range(self._next_id, self._next_id + len(documents)))
            self._next_id += len(documents)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents and metadata
        start_idx = len(self.documents)
        for i, (doc, doc_id) in enumerate(zip(documents, ids)):
            self.documents.append(doc)
            self.metadata.append(doc.get('metadata', {}))
            self._id_to_idx[doc_id] = start_idx + i
        
        logger.info(f"Added {len(documents)} documents to vector store")
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> Tuple[List[float], List[int], List[Dict]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (dict of key-value pairs)
        
        Returns:
            Tuple of (scores, indices, documents)
        """
        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension ({query_embedding.shape[0]}) "
                f"does not match store dimension ({self.embedding_dim})"
            )
        
        # Normalize if needed
        if self.normalize and (self.metric == 'cosine' or self.metric == 'ip'):
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
        
        # Convert to float32 and reshape for FAISS
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Search
        if filter_metadata:
            # Filter by metadata (post-filtering for now)
            # For large datasets, consider pre-filtering with FAISS metadata
            k_candidate = min(top_k * 10, self.index.ntotal)  # Get more candidates
            scores, indices = self.index.search(query_embedding, k_candidate)
            
            # Filter by metadata
            filtered_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                
                doc_metadata = self.metadata[idx]
                match = all(
                    doc_metadata.get(key) == value
                    for key, value in filter_metadata.items()
                )
                
                if match:
                    filtered_results.append((float(score), int(idx), self.documents[idx]))
                
                if len(filtered_results) >= top_k:
                    break
            
            if filtered_results:
                scores, indices, documents = zip(*filtered_results)
                return list(scores), list(indices), list(documents)
            else:
                return [], [], []
        
        else:
            # No filtering
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Extract documents
            results_scores = []
            results_indices = []
            results_documents = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                results_scores.append(float(score))
                results_indices.append(int(idx))
                results_documents.append(self.documents[idx])
            
            return results_scores, results_indices, results_documents
    
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search for multiple queries.
        
        Args:
            query_embeddings: Query embeddings of shape (n_queries, embedding_dim)
            top_k: Number of results per query
        
        Returns:
            Tuple of (scores, indices) arrays
        """
        # Normalize if needed
        if self.normalize and (self.metric == 'cosine' or self.metric == 'ip'):
            norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            query_embeddings = query_embeddings / norms
        
        # Convert to float32
        query_embeddings = query_embeddings.astype('float32')
        
        # Search
        scores, indices = self.index.search(query_embeddings, top_k)
        
        return scores, indices
    
    def get_by_id(self, doc_id: int) -> Optional[Dict]:
        """Get document by ID."""
        idx = self._id_to_idx.get(doc_id)
        if idx is not None:
            return self.documents[idx]
        return None
    
    def get_by_ids(self, doc_ids: List[int]) -> List[Optional[Dict]]:
        """Get multiple documents by IDs."""
        return [self.get_by_id(doc_id) for doc_id in doc_ids]
    
    def delete(self, doc_ids: List[int]):
        """
        Delete documents by IDs.
        Note: FAISS doesn't support deletion, so we mark as deleted.
        For persistent deletion, rebuild the index.
        """
        logger.warning(
            "FAISS does not support deletion. "
            "Documents marked as deleted but still in index."
        )
        for doc_id in doc_ids:
            if doc_id in self._id_to_idx:
                idx = self._id_to_idx[doc_id]
                self.documents[idx] = {'_deleted': True}
                self.metadata[idx] = {'_deleted': True}
    
    def size(self) -> int:
        """Get number of documents in store."""
        return self.index.ntotal
    
    def get_statistics(self) -> Dict:
        """Get vector store statistics."""
        return {
            'n_documents': self.size(),
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'metric': self.metric,
            'normalize': self.normalize,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }
    
    def save(self, directory: str):
        """
        Save vector store to directory.
        
        Args:
            directory: Directory to save to
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = Path(directory) / 'faiss_index.bin'
        self.faiss.write_index(self.index, str(index_path))
        
        # Save documents and metadata
        docs_path = Path(directory) / 'documents.pkl'
        with open(docs_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'id_to_idx': self._id_to_idx,
                'next_id': self._next_id
            }, f)
        
        # Save config
        config_path = Path(directory) / 'config.json'
        with open(config_path, 'w') as f:
            json.dump({
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'metric': self.metric,
                'normalize': self.normalize
            }, f, indent=2)
        
        logger.info(f"Saved vector store to {directory}")
    
    @classmethod
    def load(cls, directory: str) -> 'VectorStore':
        """
        Load vector store from directory.
        
        Args:
            directory: Directory to load from
        
        Returns:
            Loaded VectorStore instance
        """
        import faiss
        
        # Load config
        config_path = Path(directory) / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create instance
        store = cls(**config)
        
        # Load FAISS index
        index_path = Path(directory) / 'faiss_index.bin'
        store.index = faiss.read_index(str(index_path))
        
        # Load documents and metadata
        docs_path = Path(directory) / 'documents.pkl'
        with open(docs_path, 'rb') as f:
            data = pickle.load(f)
            store.documents = data['documents']
            store.metadata = data['metadata']
            store._id_to_idx = data['id_to_idx']
            store._next_id = data['next_id']
        
        logger.info(f"Loaded vector store from {directory}")
        return store


def create_vector_store_from_chunks(
    chunks: List[Dict],
    embeddings: np.ndarray,
    embedding_dim: int,
    index_type: str = 'Flat',
    metric: str = 'cosine'
) -> VectorStore:
    """
    Create and populate vector store from chunks and embeddings.
    
    Args:
        chunks: List of chunk dictionaries
        embeddings: Chunk embeddings
        embedding_dim: Embedding dimension
        index_type: FAISS index type
        metric: Distance metric
    
    Returns:
        Populated VectorStore instance
    """
    store = VectorStore(
        embedding_dim=embedding_dim,
        index_type=index_type,
        metric=metric
    )
    
    store.add(embeddings, chunks)
    
    logger.info(f"Created vector store with {len(chunks)} chunks")
    return store
