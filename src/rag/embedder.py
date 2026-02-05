"""
Embedder for RAG Pipeline

Generate embeddings for text chunks using sentence-transformers.
Supports multiple embedding models with caching and batch processing.
"""

import logging
from typing import List, Union, Optional, Dict
import numpy as np
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class Embedder:
    """
    Generate embeddings for text using sentence-transformers.
    
    Attributes:
        model_name (str): Name of sentence-transformers model
        embedding_dim (int): Dimension of embeddings
        device (str): 'cpu' or 'cuda'
        cache_embeddings (bool): Whether to cache embeddings
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: str = 'cpu',
        cache_embeddings: bool = True,
        normalize: bool = True
    ):
        """
        Initialize embedder.
        
        Args:
            model_name: Sentence-transformers model name
                - 'all-MiniLM-L6-v2': Fast, 384-dim (default)
                - 'all-mpnet-base-v2': High quality, 768-dim
                - 'multi-qa-MiniLM-L6-cos-v1': Optimized for Q&A
            device: 'cpu' or 'cuda'
            cache_embeddings: Cache embeddings for repeated texts
            normalize: Normalize embeddings to unit length
        """
        self.model_name = model_name
        self.device = device
        self.cache_embeddings = cache_embeddings
        self.normalize = normalize
        self._embedding_cache = {}
        
        # Import sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Loaded embedding model '{model_name}' with "
                f"dimension {self.embedding_dim} on {device}"
            )
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_text(
        self,
        text: Union[str, List[str]],
        show_progress: bool = False,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            text: Single text string or list of strings
            show_progress: Show progress bar for batch encoding
            batch_size: Batch size for encoding
        
        Returns:
            Embedding array of shape (n_texts, embedding_dim)
        """
        # Handle single text
        if isinstance(text, str):
            # Check cache
            if self.cache_embeddings and text in self._embedding_cache:
                return self._embedding_cache[text]
            
            embedding = self.model.encode(
                [text],
                show_progress_bar=False,
                batch_size=1,
                normalize_embeddings=self.normalize
            )[0]
            
            # Cache
            if self.cache_embeddings:
                self._embedding_cache[text] = embedding
            
            return embedding
        
        # Handle list of texts
        # Check cache for all texts
        if self.cache_embeddings:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, t in enumerate(text):
                if t in self._embedding_cache:
                    cached_embeddings.append((i, self._embedding_cache[t]))
                else:
                    uncached_texts.append(t)
                    uncached_indices.append(i)
            
            # Embed uncached texts
            if uncached_texts:
                new_embeddings = self.model.encode(
                    uncached_texts,
                    show_progress_bar=show_progress,
                    batch_size=batch_size,
                    normalize_embeddings=self.normalize
                )
                
                # Cache new embeddings
                for t, emb in zip(uncached_texts, new_embeddings):
                    self._embedding_cache[t] = emb
                
                # Combine cached and new embeddings in correct order
                all_embeddings = [None] * len(text)
                for i, emb in cached_embeddings:
                    all_embeddings[i] = emb
                for i, emb in zip(uncached_indices, new_embeddings):
                    all_embeddings[i] = emb
                
                return np.array(all_embeddings)
            else:
                # All cached
                return np.array([emb for _, emb in sorted(cached_embeddings)])
        else:
            # No caching
            embeddings = self.model.encode(
                text,
                show_progress_bar=show_progress,
                batch_size=batch_size,
                normalize_embeddings=self.normalize
            )
            return embeddings
    
    def embed_chunks(
        self,
        chunks: List[Dict],
        text_field: str = 'text',
        show_progress: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of chunk dictionaries
            text_field: Field containing text to embed
            show_progress: Show progress bar
            batch_size: Batch size for encoding
        
        Returns:
            Embedding array of shape (n_chunks, embedding_dim)
        """
        texts = [chunk[text_field] for chunk in chunks]
        embeddings = self.embed_text(
            texts,
            show_progress=show_progress,
            batch_size=batch_size
        )
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return embeddings
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ('cosine', 'dot', 'euclidean')
        
        Returns:
            Similarity score
        """
        if metric == 'cosine':
            # Cosine similarity
            if self.normalize:
                # Already normalized, just dot product
                return float(np.dot(embedding1, embedding2))
            else:
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
        
        elif metric == 'dot':
            # Dot product
            return float(np.dot(embedding1, embedding2))
        
        elif metric == 'euclidean':
            # Negative Euclidean distance (higher = more similar)
            return float(-np.linalg.norm(embedding1 - embedding2))
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def compute_similarity_matrix(
        self,
        embeddings1: np.ndarray,
        embeddings2: Optional[np.ndarray] = None,
        metric: str = 'cosine'
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix.
        
        Args:
            embeddings1: First set of embeddings (n1, dim)
            embeddings2: Second set of embeddings (n2, dim), or None for self-similarity
            metric: Similarity metric
        
        Returns:
            Similarity matrix of shape (n1, n2) or (n1, n1)
        """
        if embeddings2 is None:
            embeddings2 = embeddings1
        
        if metric == 'cosine':
            if self.normalize:
                # Already normalized, just matrix multiply
                return embeddings1 @ embeddings2.T
            else:
                # Normalize first
                norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
                norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
                norm1[norm1 == 0] = 1
                norm2[norm2 == 0] = 1
                embeddings1_norm = embeddings1 / norm1
                embeddings2_norm = embeddings2 / norm2
                return embeddings1_norm @ embeddings2_norm.T
        
        elif metric == 'dot':
            return embeddings1 @ embeddings2.T
        
        elif metric == 'euclidean':
            # Negative squared Euclidean distance
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            norm1_sq = np.sum(embeddings1 ** 2, axis=1, keepdims=True)
            norm2_sq = np.sum(embeddings2 ** 2, axis=1, keepdims=True)
            distances_sq = norm1_sq + norm2_sq.T - 2 * (embeddings1 @ embeddings2.T)
            return -np.sqrt(np.maximum(distances_sq, 0))
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def clear_cache(self):
        """Clear embedding cache."""
        cache_size = len(self._embedding_cache)
        self._embedding_cache.clear()
        logger.info(f"Cleared embedding cache ({cache_size} entries)")
    
    def get_cache_size(self) -> int:
        """Get number of cached embeddings."""
        return len(self._embedding_cache)
    
    def get_statistics(self) -> Dict:
        """Get embedder statistics."""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'cache_enabled': self.cache_embeddings,
            'cache_size': self.get_cache_size(),
            'normalize': self.normalize
        }
    
    def save(self, filepath: str):
        """Save embedder configuration and cache."""
        config = {
            'model_name': self.model_name,
            'device': self.device,
            'cache_embeddings': self.cache_embeddings,
            'normalize': self.normalize,
            'embedding_cache': self._embedding_cache if self.cache_embeddings else {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(
            f"Saved embedder config to {filepath} "
            f"(cache size: {len(self._embedding_cache)})"
        )
    
    @classmethod
    def load(cls, filepath: str) -> 'Embedder':
        """Load embedder configuration and cache."""
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        # Extract cache
        embedding_cache = config.pop('embedding_cache', {})
        
        # Create embedder
        embedder = cls(**config)
        embedder._embedding_cache = embedding_cache
        
        logger.info(
            f"Loaded embedder config from {filepath} "
            f"(cache size: {len(embedding_cache)})"
        )
        return embedder


def batch_embed_documents(
    documents: List[Dict],
    embedder: Embedder,
    text_field: str = 'text',
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Batch embed multiple documents efficiently.
    
    Args:
        documents: List of document dictionaries
        embedder: Embedder instance
        text_field: Field containing text
        batch_size: Batch size for encoding
        show_progress: Show progress bar
    
    Returns:
        Embedding array of shape (n_documents, embedding_dim)
    """
    texts = [doc[text_field] for doc in documents]
    embeddings = embedder.embed_text(
        texts,
        show_progress=show_progress,
        batch_size=batch_size
    )
    
    logger.info(f"Batch embedded {len(documents)} documents")
    return embeddings


def find_duplicate_chunks(
    chunks: List[Dict],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.95,
    text_field: str = 'text'
) -> List[List[int]]:
    """
    Find duplicate or near-duplicate chunks based on embedding similarity.
    
    Args:
        chunks: List of chunk dictionaries
        embeddings: Chunk embeddings
        similarity_threshold: Threshold for considering chunks as duplicates
        text_field: Field containing text
    
    Returns:
        List of duplicate groups (list of chunk indices)
    """
    n = len(chunks)
    if n == 0:
        return []
    
    # Compute similarity matrix
    from scipy.spatial.distance import cosine
    
    visited = set()
    duplicate_groups = []
    
    for i in range(n):
        if i in visited:
            continue
        
        # Find similar chunks
        group = [i]
        for j in range(i + 1, n):
            if j in visited:
                continue
            
            # Compute cosine similarity
            similarity = 1 - cosine(embeddings[i], embeddings[j])
            
            if similarity >= similarity_threshold:
                group.append(j)
                visited.add(j)
        
        if len(group) > 1:
            duplicate_groups.append(group)
            visited.add(i)
    
    logger.info(
        f"Found {len(duplicate_groups)} duplicate groups "
        f"(threshold={similarity_threshold})"
    )
    return duplicate_groups
