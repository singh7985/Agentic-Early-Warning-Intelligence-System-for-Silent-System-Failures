"""
Retrieval Agent: Queries VectorDB for historical context

Responsibilities:
- Query knowledge base for similar failures
- Retrieve relevant historical incidents
- Filter by sensor patterns and failure types
- Return citations and relevance scores
- Manage retrieval confidence
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievedIncident:
    """A retrieved historical incident"""
    text: str
    similarity_score: float
    engine_id: int
    cycle_range: str
    failure_type: str
    severity: str
    sensors_affected: List[str]
    citation: Dict[str, Any]
    retrieved_at: str


@dataclass
class RetrievalResult:
    """Result from knowledge base retrieval"""
    query: str
    results: List[RetrievedIncident]
    total_results: int
    mean_score: float
    retrieval_confidence: float
    query_type: str  # 'text', 'pattern', 'sensor'
    retrieval_time_ms: float
    timestamp: str


class RetrievalAgent:
    """
    Agent responsible for querying historical failure data.
    
    Features:
    - Text-based semantic search
    - Sensor pattern matching
    - Failure type filtering
    - Citation tracking
    - Retrieval confidence scoring
    - Query history
    """

    def __init__(
        self,
        knowledge_base: Optional[Any] = None,
        top_k: int = 5,
        min_similarity: float = 0.3,
        retrieval_confidence_threshold: float = 0.5,
    ):
        """
        Initialize Retrieval Agent.
        
        Args:
            knowledge_base: KnowledgeBase instance for queries
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            retrieval_confidence_threshold: Minimum confidence for results
        """
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.retrieval_confidence_threshold = retrieval_confidence_threshold
        
        # Query history
        self.query_history = []
        
        logger.info(
            f"RetrievalAgent initialized. "
            f"top_k={top_k}, min_similarity={min_similarity}"
        )

    def search_by_text(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Search for similar failures using text query.
        
        Args:
            query: Natural language query
            top_k: Override default top_k
        
        Returns:
            RetrievalResult with retrieved incidents
        """
        import time
        start_time = time.time()
        
        top_k = top_k or self.top_k

        try:
            if self.knowledge_base is None:
                logger.warning("No knowledge base available for retrieval")
                return RetrievalResult(
                    query=query,
                    results=[],
                    total_results=0,
                    mean_score=0.0,
                    retrieval_confidence=0.0,
                    query_type='text',
                    retrieval_time_ms=0.0,
                    timestamp=datetime.now().isoformat(),
                )

            # Search knowledge base
            raw_results = self.knowledge_base.search(query, top_k=top_k)

            # Convert to RetrievedIncident format
            incidents = []
            scores = []
            
            for result in raw_results:
                score = result.get('score', 0.5)
                
                if score >= self.min_similarity:
                    incident = RetrievedIncident(
                        text=result.get('text', ''),
                        similarity_score=score,
                        engine_id=result.get('metadata', {}).get('engine_id', 0),
                        cycle_range=result.get('metadata', {}).get('cycle_range', 'unknown'),
                        failure_type=result.get('metadata', {}).get('failure_type', 'unknown'),
                        severity=self._infer_severity(score),
                        sensors_affected=result.get('metadata', {}).get('affected_sensors', []),
                        citation=result.get('citation', {}),
                        retrieved_at=datetime.now().isoformat(),
                    )
                    incidents.append(incident)
                    scores.append(score)

            # Calculate retrieval confidence
            if scores:
                mean_score = float(np.mean(scores))
                # Confidence based on scores and number of results
                retrieval_confidence = min(
                    mean_score,
                    len(incidents) / max(1, top_k),  # Bonus for finding more results
                )
            else:
                mean_score = 0.0
                retrieval_confidence = 0.0

            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

            result = RetrievalResult(
                query=query,
                results=incidents,
                total_results=len(incidents),
                mean_score=mean_score,
                retrieval_confidence=retrieval_confidence,
                query_type='text',
                retrieval_time_ms=elapsed_time,
                timestamp=datetime.now().isoformat(),
            )

            self.query_history.append(result)
            return result

        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return RetrievalResult(
                query=query,
                results=[],
                total_results=0,
                mean_score=0.0,
                retrieval_confidence=0.0,
                query_type='text',
                retrieval_time_ms=0.0,
                timestamp=datetime.now().isoformat(),
            )

    def search_by_sensor_pattern(
        self,
        sensor_deviations: Dict[str, float],
        rul: float,
        anomaly_score: float = 0.5,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Search for similar failures by sensor deviation pattern.
        
        Args:
            sensor_deviations: Dict of {sensor_name: deviation_value}
            rul: Current Remaining Useful Life estimate
            anomaly_score: Current anomaly score
            top_k: Override default top_k
        
        Returns:
            RetrievalResult with similar failure patterns
        """
        import time
        start_time = time.time()
        
        top_k = top_k or self.top_k

        try:
            if self.knowledge_base is None:
                logger.warning("No knowledge base available")
                return RetrievalResult(
                    query=str(sensor_deviations),
                    results=[],
                    total_results=0,
                    mean_score=0.0,
                    retrieval_confidence=0.0,
                    query_type='sensor',
                    retrieval_time_ms=0.0,
                    timestamp=datetime.now().isoformat(),
                )

            # Build query from sensor pattern
            query = self._build_sensor_query(sensor_deviations, rul, anomaly_score)

            # Use knowledge base search_similar_failures if available
            if hasattr(self.knowledge_base, 'search_similar_failures'):
                raw_results = self.knowledge_base.search_similar_failures(
                    sensor_deviations=sensor_deviations,
                    rul=rul,
                    anomaly_score=anomaly_score,
                    top_k=top_k,
                )
            else:
                # Fallback to text search
                raw_results = self.knowledge_base.search(query, top_k=top_k)

            # Convert results
            incidents = []
            scores = []
            
            for result in raw_results:
                score = result.get('score', 0.5)
                
                if score >= self.min_similarity:
                    incident = RetrievedIncident(
                        text=result.get('text', ''),
                        similarity_score=score,
                        engine_id=result.get('metadata', {}).get('engine_id', 0),
                        cycle_range=result.get('metadata', {}).get('cycle_range', 'unknown'),
                        failure_type=result.get('metadata', {}).get('failure_type', 'unknown'),
                        severity=self._infer_severity(score),
                        sensors_affected=result.get('metadata', {}).get('affected_sensors', []),
                        citation=result.get('citation', {}),
                        retrieved_at=datetime.now().isoformat(),
                    )
                    incidents.append(incident)
                    scores.append(score)

            # Calculate retrieval confidence
            if scores:
                mean_score = float(np.mean(scores))
                retrieval_confidence = min(
                    mean_score,
                    len(incidents) / max(1, top_k),
                )
            else:
                mean_score = 0.0
                retrieval_confidence = 0.0

            elapsed_time = (time.time() - start_time) * 1000

            result = RetrievalResult(
                query=query,
                results=incidents,
                total_results=len(incidents),
                mean_score=mean_score,
                retrieval_confidence=retrieval_confidence,
                query_type='sensor',
                retrieval_time_ms=elapsed_time,
                timestamp=datetime.now().isoformat(),
            )

            self.query_history.append(result)
            return result

        except Exception as e:
            logger.error(f"Error in sensor pattern search: {e}")
            return RetrievalResult(
                query=str(sensor_deviations),
                results=[],
                total_results=0,
                mean_score=0.0,
                retrieval_confidence=0.0,
                query_type='sensor',
                retrieval_time_ms=0.0,
                timestamp=datetime.now().isoformat(),
            )

    def search_by_failure_type(
        self,
        failure_type: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Search for incidents of a specific failure type.
        
        Args:
            failure_type: Type of failure to search for
            top_k: Override default top_k
        
        Returns:
            RetrievalResult with matching incidents
        """
        query = f"Find incidents with {failure_type} failure type"
        return self.search_by_text(query, top_k)

    def filter_results(
        self,
        results: RetrievalResult,
        min_similarity: Optional[float] = None,
        failure_types: Optional[List[str]] = None,
        severity_levels: Optional[List[str]] = None,
    ) -> RetrievalResult:
        """
        Filter retrieval results by criteria.
        
        Args:
            results: Original retrieval result
            min_similarity: Minimum similarity threshold
            failure_types: Acceptable failure types
            severity_levels: Acceptable severity levels
        
        Returns:
            Filtered RetrievalResult
        """
        min_similarity = min_similarity or self.min_similarity
        
        filtered_incidents = []
        
        for incident in results.results:
            # Check similarity
            if incident.similarity_score < min_similarity:
                continue
            
            # Check failure type
            if failure_types and incident.failure_type not in failure_types:
                continue
            
            # Check severity
            if severity_levels and incident.severity not in severity_levels:
                continue
            
            filtered_incidents.append(incident)

        # Recalculate metrics
        scores = [i.similarity_score for i in filtered_incidents]
        mean_score = float(np.mean(scores)) if scores else 0.0

        return RetrievalResult(
            query=results.query,
            results=filtered_incidents,
            total_results=len(filtered_incidents),
            mean_score=mean_score,
            retrieval_confidence=results.retrieval_confidence,
            query_type=results.query_type,
            retrieval_time_ms=results.retrieval_time_ms,
            timestamp=datetime.now().isoformat(),
        )

    def _build_sensor_query(
        self,
        sensor_deviations: Dict[str, float],
        rul: float,
        anomaly_score: float,
    ) -> str:
        """Build natural language query from sensor pattern."""
        # Sort by absolute deviation
        sorted_sensors = sorted(
            sensor_deviations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        sensor_desc = ", ".join([
            f"{s} ({'+' if v > 0 else ''}{v:.2f})"
            for s, v in sorted_sensors[:3]
        ])

        query = (
            f"Find similar failures with sensor deviations in {sensor_desc}, "
            f"RUL around {rul:.0f} cycles, anomaly score {anomaly_score:.2f}"
        )
        
        return query

    def _infer_severity(self, similarity_score: float) -> str:
        """Infer severity from similarity score."""
        if similarity_score >= 0.8:
            return 'high'
        elif similarity_score >= 0.6:
            return 'medium'
        else:
            return 'low'

    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        if not self.query_history:
            return {
                'n_queries': 0,
                'avg_results': 0.0,
                'avg_confidence': 0.0,
                'avg_retrieval_time_ms': 0.0,
            }

        return {
            'n_queries': len(self.query_history),
            'avg_results': np.mean([r.total_results for r in self.query_history]),
            'avg_confidence': np.mean([r.retrieval_confidence for r in self.query_history]),
            'avg_retrieval_time_ms': np.mean([r.retrieval_time_ms for r in self.query_history]),
            'query_types': self._count_query_types(),
        }

    def _count_query_types(self) -> Dict[str, int]:
        """Count queries by type."""
        counts = {}
        for result in self.query_history:
            query_type = result.query_type
            counts[query_type] = counts.get(query_type, 0) + 1
        return counts

    def clear_history(self):
        """Clear query history."""
        self.query_history = []
        logger.info("Query history cleared")
