"""
Retriever for RAG Pipeline

Similarity search with citation tracking and relevance scoring.
Implements query expansion, re-ranking, and metadata filtering.
"""

import logging
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieval system with similarity search and citation tracking.
    
    Attributes:
        vector_store: VectorStore instance
        embedder: Embedder instance
        top_k (int): Default number of results
        min_similarity (float): Minimum similarity threshold
        include_citations (bool): Whether to include citation information
    """
    
    def __init__(
        self,
        vector_store,
        embedder,
        top_k: int = 5,
        min_similarity: float = 0.3,
        include_citations: bool = True,
        rerank: bool = False
    ):
        """
        Initialize retriever.
        
        Args:
            vector_store: VectorStore instance
            embedder: Embedder instance
            top_k: Default number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            include_citations: Include citation tracking
            rerank: Enable re-ranking (cross-encoder, if available)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.include_citations = include_citations
        self.rerank = rerank
        
        # Query history for analytics
        self.query_history = []
        
        logger.info(
            f"Initialized Retriever: top_k={top_k}, "
            f"min_similarity={min_similarity}"
        )
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        filter_metadata: Optional[Dict] = None,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Query string
            top_k: Number of results (uses default if None)
            min_similarity: Minimum similarity threshold
            filter_metadata: Metadata filters
            return_scores: Include similarity scores in results
        
        Returns:
            List of result dictionaries with text, metadata, scores, citations
        """
        top_k = top_k or self.top_k
        min_similarity = min_similarity or self.min_similarity
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search vector store (get more candidates for filtering)
        k_search = top_k * 2 if filter_metadata else top_k
        scores, indices, documents = self.vector_store.search(
            query_embedding,
            top_k=k_search,
            filter_metadata=filter_metadata
        )
        
        # Filter by minimum similarity
        results = []
        for score, idx, doc in zip(scores, indices, documents):
            # Convert distance to similarity if needed
            if self.vector_store.metric == 'l2':
                # L2 distance: lower is better, convert to similarity
                similarity = 1.0 / (1.0 + score)
            elif self.vector_store.metric == 'cosine' or self.vector_store.metric == 'ip':
                # Inner product/cosine: higher is better
                similarity = float(score)
            else:
                similarity = float(score)
            
            if similarity < min_similarity:
                continue
            
            # Build result
            result = {
                'text': doc.get('text', ''),
                'metadata': doc.get('metadata', {}),
                'index': idx
            }
            
            if return_scores:
                result['score'] = similarity
                result['rank'] = len(results) + 1
            
            if self.include_citations:
                result['citation'] = self._build_citation(doc, similarity)
            
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        # Re-rank if enabled
        if self.rerank and len(results) > 1:
            results = self._rerank_results(query, results)
        
        # Log query
        self._log_query(query, len(results), scores[0] if scores else 0.0)
        
        logger.info(
            f"Retrieved {len(results)} results for query: '{query[:50]}...'"
        )
        
        return results
    
    def search_by_example(
        self,
        example_text: str,
        top_k: Optional[int] = None,
        exclude_self: bool = True
    ) -> List[Dict]:
        """
        Search for similar documents using an example document.
        
        Args:
            example_text: Example document text
            top_k: Number of results
            exclude_self: Exclude the example itself from results
        
        Returns:
            List of similar documents
        """
        top_k = top_k or self.top_k
        
        # Get slightly more results to account for self-exclusion
        k_search = top_k + 1 if exclude_self else top_k
        
        results = self.search(
            query=example_text,
            top_k=k_search,
            return_scores=True
        )
        
        # Exclude exact matches if requested
        if exclude_self:
            results = [
                r for r in results
                if r['text'].strip() != example_text.strip()
            ][:top_k]
        
        return results
    
    def search_with_context(
        self,
        query: str,
        context: Dict,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Search with additional context (e.g., sensor values, RUL).
        
        Args:
            query: Query string
            context: Context dictionary with sensor data, RUL, etc.
            top_k: Number of results
        
        Returns:
            List of results with context-aware scoring
        """
        # Expand query with context
        expanded_query = self._expand_query_with_context(query, context)
        
        # Search
        results = self.search(
            query=expanded_query,
            top_k=top_k
        )
        
        # Add context to results
        for result in results:
            result['query_context'] = context
        
        return results
    
    def batch_search(
        self,
        queries: List[str],
        top_k: Optional[int] = None
    ) -> List[List[Dict]]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
        
        Returns:
            List of result lists (one per query)
        """
        top_k = top_k or self.top_k
        
        # Batch embed queries
        query_embeddings = self.embedder.embed_text(queries, show_progress=True)
        
        # Batch search
        scores_batch, indices_batch = self.vector_store.batch_search(
            query_embeddings,
            top_k=top_k
        )
        
        # Build results for each query
        all_results = []
        for query, scores, indices in zip(queries, scores_batch, indices_batch):
            results = []
            for score, idx in zip(scores, indices):
                if idx < 0 or idx >= len(self.vector_store.documents):
                    continue
                
                doc = self.vector_store.documents[idx]
                similarity = float(score)
                
                if similarity < self.min_similarity:
                    continue
                
                result = {
                    'text': doc.get('text', ''),
                    'metadata': doc.get('metadata', {}),
                    'score': similarity,
                    'rank': len(results) + 1,
                    'index': int(idx)
                }
                
                if self.include_citations:
                    result['citation'] = self._build_citation(doc, similarity)
                
                results.append(result)
            
            all_results.append(results)
        
        logger.info(f"Batch searched {len(queries)} queries")
        return all_results
    
    def validate_retrieval(
        self,
        test_cases: List[Dict],
        query_field: str = 'query',
        expected_field: str = 'expected_results',
        top_k: int = 5
    ) -> Dict:
        """
        Validate retrieval quality on test cases.
        
        Args:
            test_cases: List of test case dictionaries
            query_field: Field containing query
            expected_field: Field containing expected result indices/texts
            top_k: Number of results to retrieve
        
        Returns:
            Validation metrics dictionary
        """
        n_test_cases = len(test_cases)
        n_correct = 0
        n_in_top_k = 0
        mrr_scores = []  # Mean Reciprocal Rank
        
        for test_case in test_cases:
            query = test_case[query_field]
            expected = test_case[expected_field]
            
            # Search
            results = self.search(query, top_k=top_k)
            
            # Check if expected result is in top-k
            result_texts = [r['text'] for r in results]
            result_indices = [r['index'] for r in results]
            
            # Handle expected as text or index
            if isinstance(expected, str):
                found_idx = next(
                    (i for i, text in enumerate(result_texts) if expected in text),
                    None
                )
            elif isinstance(expected, int):
                found_idx = next(
                    (i for i, idx in enumerate(result_indices) if idx == expected),
                    None
                )
            else:
                found_idx = None
            
            if found_idx is not None:
                n_in_top_k += 1
                if found_idx == 0:
                    n_correct += 1
                # Reciprocal rank
                mrr_scores.append(1.0 / (found_idx + 1))
            else:
                mrr_scores.append(0.0)
        
        # Compute metrics
        metrics = {
            'n_test_cases': n_test_cases,
            'top_1_accuracy': n_correct / n_test_cases if n_test_cases > 0 else 0,
            f'top_{top_k}_recall': n_in_top_k / n_test_cases if n_test_cases > 0 else 0,
            'mrr': np.mean(mrr_scores) if mrr_scores else 0
        }
        
        logger.info(
            f"Validation: Top-1 Accuracy={metrics['top_1_accuracy']:.2%}, "
            f"Top-{top_k} Recall={metrics[f'top_{top_k}_recall']:.2%}, "
            f"MRR={metrics['mrr']:.3f}"
        )
        
        return metrics
    
    def _build_citation(self, doc: Dict, similarity: float) -> Dict:
        """Build citation information for a document."""
        metadata = doc.get('metadata', {})
        
        citation = {
            'similarity': float(similarity),
            'retrieved_at': datetime.now().isoformat(),
        }
        
        # Add metadata fields to citation
        if 'engine_id' in metadata:
            citation['engine_id'] = metadata['engine_id']
        if 'start_cycle' in metadata and 'end_cycle' in metadata:
            citation['cycle_range'] = f"{metadata['start_cycle']}-{metadata['end_cycle']}"
        if 'failure_type' in metadata:
            citation['failure_type'] = metadata['failure_type']
        if 'source' in metadata:
            citation['source'] = metadata['source']
        
        # Build citation string
        parts = []
        if 'engine_id' in citation:
            parts.append(f"Engine {citation['engine_id']}")
        if 'cycle_range' in citation:
            parts.append(f"Cycles {citation['cycle_range']}")
        if 'failure_type' in citation:
            parts.append(f"Type: {citation['failure_type']}")
        parts.append(f"Similarity: {similarity:.3f}")
        
        citation['citation_string'] = ' | '.join(parts)
        
        return citation
    
    def _expand_query_with_context(self, query: str, context: Dict) -> str:
        """Expand query with context information."""
        expansions = []
        
        if 'rul' in context:
            rul = context['rul']
            expansions.append(f"remaining useful life {rul} cycles")
        
        if 'sensor_deviations' in context:
            for sensor, deviation in context['sensor_deviations'].items():
                expansions.append(f"{sensor} deviation {deviation:.2f}")
        
        if 'alert_level' in context:
            level = context['alert_level']
            expansions.append(f"alert level {level}")
        
        if expansions:
            expanded = f"{query}. Context: {', '.join(expansions)}"
        else:
            expanded = query
        
        return expanded
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Re-rank results using cross-encoder (if available).
        Falls back to original ranking if cross-encoder not available.
        """
        try:
            from sentence_transformers import CrossEncoder
            
            # Load cross-encoder model (cache it)
            if not hasattr(self, '_cross_encoder'):
                self._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            # Prepare pairs
            pairs = [[query, r['text']] for r in results]
            
            # Get cross-encoder scores
            ce_scores = self._cross_encoder.predict(pairs)
            
            # Add to results and re-sort
            for result, ce_score in zip(results, ce_scores):
                result['rerank_score'] = float(ce_score)
            
            results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
            
            # Update ranks
            for i, result in enumerate(results):
                result['rank'] = i + 1
            
            logger.info("Re-ranked results using cross-encoder")
        
        except ImportError:
            logger.warning("Cross-encoder not available for re-ranking")
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}")
        
        return results
    
    def _log_query(self, query: str, n_results: int, top_score: float):
        """Log query for analytics."""
        self.query_history.append({
            'query': query,
            'n_results': n_results,
            'top_score': top_score,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_query_history(self) -> List[Dict]:
        """Get query history."""
        return self.query_history
    
    def get_statistics(self) -> Dict:
        """Get retriever statistics."""
        stats = {
            'n_queries': len(self.query_history),
            'top_k': self.top_k,
            'min_similarity': self.min_similarity,
            'include_citations': self.include_citations,
            'rerank_enabled': self.rerank
        }
        
        if self.query_history:
            scores = [q['top_score'] for q in self.query_history]
            n_results = [q['n_results'] for q in self.query_history]
            
            stats['mean_top_score'] = np.mean(scores)
            stats['mean_n_results'] = np.mean(n_results)
        
        return stats


def create_sensor_deviation_query(
    sensor_values: Dict[str, float],
    rul: Optional[float] = None,
    anomaly_score: Optional[float] = None
) -> str:
    """
    Create a query string from current sensor deviations.
    
    Args:
        sensor_values: Dictionary of sensor names to deviation values
        rul: Current RUL prediction
        anomaly_score: Current anomaly score
    
    Returns:
        Formatted query string
    """
    parts = ["Find past incidents similar to current sensor deviation pattern:"]
    
    # Sort sensors by absolute deviation
    sorted_sensors = sorted(
        sensor_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    # Add top sensor deviations
    for sensor, value in sorted_sensors[:5]:
        if abs(value) > 0.1:  # Only include significant deviations
            direction = "increase" if value > 0 else "decrease"
            parts.append(f"{sensor} {direction} by {abs(value):.2f}")
    
    # Add RUL if available
    if rul is not None:
        parts.append(f"remaining useful life {rul:.0f} cycles")
    
    # Add anomaly score if available
    if anomaly_score is not None:
        parts.append(f"anomaly score {anomaly_score:.3f}")
    
    return ". ".join(parts) + "."
