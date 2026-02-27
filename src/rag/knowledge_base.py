"""
Knowledge Base Builder for RAG Pipeline

Build knowledge base from PHASE 5 degradation data and historical failures.
Integrates document chunking, embedding, and vector store creation.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

from src.rag.document_chunker import DocumentChunker, create_failure_document
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Knowledge base for failure history and degradation patterns.
    
    Integrates:
    - Document chunking
    - Embedding generation
    - Vector storage
    - Similarity search
    
    Built from PHASE 5 degradation labels and sensor data.
    """
    
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        chunker: Optional[DocumentChunker] = None,
        embedding_model: str = 'all-MiniLM-L6-v2',
        chunk_size: int = 500,
        chunk_strategy: str = 'sentence'
    ):
        """
        Initialize knowledge base.
        
        Args:
            embedder: Pre-configured Embedder (creates new if None)
            chunker: Pre-configured DocumentChunker (creates new if None)
            embedding_model: Sentence-transformers model name
            chunk_size: Maximum characters per chunk
            chunk_strategy: Chunking strategy
        """
        # Initialize embedder
        if embedder is None:
            logger.info(f"Creating embedder with model: {embedding_model}")
            self.embedder = Embedder(
                model_name=embedding_model,
                device='cpu',
                cache_embeddings=True
            )
        else:
            self.embedder = embedder
        
        # Initialize chunker
        if chunker is None:
            logger.info(
                f"Creating chunker: size={chunk_size}, strategy={chunk_strategy}"
            )
            self.chunker = DocumentChunker(
                chunk_size=chunk_size,
                chunk_overlap=50,
                strategy=chunk_strategy
            )
        else:
            self.chunker = chunker
        
        # Vector store (created during build)
        self.vector_store = None
        self.retriever = None
        
        # Metadata
        self.documents = []
        self.chunks = []
        self.embeddings = None
        
        logger.info("Initialized KnowledgeBase")
    
    def build_from_degradation_data(
        self,
        degradation_periods: List[Dict],
        sensor_data: Optional[pd.DataFrame] = None,
        warnings_df: Optional[pd.DataFrame] = None,
        save_dir: Optional[str] = None
    ):
        """
        Build knowledge base from PHASE 5 degradation data.
        
        Args:
            degradation_periods: List of degradation period dicts from DegradationLabeler
            sensor_data: DataFrame with sensor readings (columns: engine_id, cycle, sensor values)
            warnings_df: DataFrame with warnings from EarlyWarningSystem
            save_dir: Directory to save KB (optional)
        """
        logger.info(f"Building KB from {len(degradation_periods)} degradation periods")
        
        # Create documents
        self.documents = []
        for period in degradation_periods:
            engine_id = period.get('engine_id', 0)
            
            # Extract sensor data for this period if available
            sensor_stats = None
            if sensor_data is not None:
                sensor_stats = self._extract_sensor_stats(
                    sensor_data,
                    engine_id,
                    period.get('start', 0),
                    period.get('end', 0)
                )
            
            # Extract warnings for this engine/period
            warnings = None
            if warnings_df is not None:
                warnings = self._extract_warnings(
                    warnings_df,
                    engine_id,
                    period.get('start', 0),
                    period.get('end', 0)
                )
            
            # Create document
            doc = create_failure_document(
                engine_id=engine_id,
                degradation_period=period,
                sensor_data=sensor_stats,
                warnings=warnings
            )
            self.documents.append(doc)
        
        logger.info(f"Created {len(self.documents)} documents")
        
        # Chunk documents
        self.chunks = self.chunker.chunk_documents(
            self.documents,
            text_field='text',
            metadata_fields=['engine_id', 'failure_type', 'duration', 'severity']
        )
        
        logger.info(f"Created {len(self.chunks)} chunks")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        self.embeddings = self.embedder.embed_chunks(
            self.chunks,
            show_progress=True,
            batch_size=32
        )
        
        logger.info(f"Generated embeddings: {self.embeddings.shape}")
        
        # Create vector store
        self.vector_store = VectorStore(
            embedding_dim=self.embedder.embedding_dim,
            index_type='Flat',
            metric='cosine'
        )
        
        self.vector_store.add(
            embeddings=self.embeddings,
            documents=self.chunks
        )
        
        logger.info(f"Built vector store with {self.vector_store.size()} entries")
        
        # Create retriever
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            top_k=5,
            min_similarity=0.3,
            include_citations=True
        )
        
        logger.info("Created retriever")
        
        # Save if directory provided
        if save_dir:
            self.save(save_dir)
    
    def build_from_documents(
        self,
        documents: List[Dict],
        text_field: str = 'text',
        metadata_fields: Optional[List[str]] = None,
        save_dir: Optional[str] = None
    ):
        """
        Build knowledge base from custom documents.
        
        Args:
            documents: List of document dictionaries
            text_field: Field containing text
            metadata_fields: Metadata fields to preserve
            save_dir: Directory to save KB
        """
        logger.info(f"Building KB from {len(documents)} custom documents")
        
        self.documents = documents
        
        # Chunk documents
        self.chunks = self.chunker.chunk_documents(
            documents,
            text_field=text_field,
            metadata_fields=metadata_fields or []
        )
        
        # Generate embeddings
        self.embeddings = self.embedder.embed_chunks(
            self.chunks,
            text_field='text',
            show_progress=True
        )
        
        # Create vector store
        self.vector_store = VectorStore(
            embedding_dim=self.embedder.embedding_dim,
            index_type='Flat',
            metric='cosine'
        )
        
        self.vector_store.add(
            embeddings=self.embeddings,
            documents=self.chunks
        )
        
        # Create retriever
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            top_k=5,
            min_similarity=0.3
        )
        
        logger.info(
            f"Built KB: {len(self.documents)} docs, "
            f"{len(self.chunks)} chunks, "
            f"{self.vector_store.size()} vectors"
        )
        
        if save_dir:
            self.save(save_dir)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search knowledge base.
        
        Args:
            query: Query string
            top_k: Number of results
            filter_metadata: Metadata filters
        
        Returns:
            List of search results with text, metadata, scores, citations
        """
        if self.retriever is None:
            raise ValueError("Knowledge base not built. Call build_from_* first.")
        
        return self.retriever.search(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
    
    def search_similar_failures(
        self,
        sensor_deviations: Dict[str, float],
        rul: Optional[float] = None,
        anomaly_score: Optional[float] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for similar past failures based on sensor pattern.
        
        Args:
            sensor_deviations: Dictionary of sensor names to deviation values
            rul: Current RUL prediction
            anomaly_score: Current anomaly score
            top_k: Number of similar failures to return
        
        Returns:
            List of similar failure incidents with citations
        """
        # Build query from sensor pattern
        from src.rag.retriever import create_sensor_deviation_query
        
        query = create_sensor_deviation_query(
            sensor_values=sensor_deviations,
            rul=rul,
            anomaly_score=anomaly_score
        )
        
        logger.info(f"Searching for similar failures: {query[:100]}...")
        
        # Search
        results = self.search(query, top_k=top_k)
        
        return results
    
    def add_documents(
        self,
        new_documents: List[Dict],
        text_field: str = 'text'
    ):
        """
        Add new documents to existing knowledge base.
        
        Args:
            new_documents: List of new document dictionaries
            text_field: Field containing text
        """
        if self.vector_store is None:
            raise ValueError("Knowledge base not built. Call build_from_* first.")
        
        logger.info(f"Adding {len(new_documents)} new documents")
        
        # Chunk new documents
        new_chunks = self.chunker.chunk_documents(
            new_documents,
            text_field=text_field
        )
        
        # Generate embeddings
        new_embeddings = self.embedder.embed_chunks(
            new_chunks,
            show_progress=True
        )
        
        # Add to vector store
        self.vector_store.add(
            embeddings=new_embeddings,
            documents=new_chunks
        )
        
        # Update internal storage
        self.documents.extend(new_documents)
        self.chunks.extend(new_chunks)
        if self.embeddings is not None:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        else:
            self.embeddings = new_embeddings
        
        logger.info(
            f"Added {len(new_documents)} documents "
            f"(total: {len(self.documents)} docs, {len(self.chunks)} chunks)"
        )
    
    def validate(
        self,
        test_cases: List[Dict],
        query_field: str = 'query',
        expected_field: str = 'expected_results'
    ) -> Dict:
        """
        Validate retrieval quality on test cases.
        
        Args:
            test_cases: List of test case dictionaries
            query_field: Field containing query
            expected_field: Field containing expected results
        
        Returns:
            Validation metrics
        """
        if self.retriever is None:
            raise ValueError("Knowledge base not built.")
        
        return self.retriever.validate_retrieval(
            test_cases=test_cases,
            query_field=query_field,
            expected_field=expected_field
        )
    
    def get_statistics(self) -> Dict:
        """Get knowledge base statistics."""
        stats = {
            'n_documents': len(self.documents),
            'n_chunks': len(self.chunks),
            'embedding_model': self.embedder.model_name,
            'embedding_dim': self.embedder.embedding_dim,
            'chunk_strategy': self.chunker.strategy,
            'chunk_size': self.chunker.chunk_size
        }
        
        if self.vector_store:
            stats['vector_store_size'] = self.vector_store.size()
            stats.update(self.vector_store.get_statistics())
        
        if self.retriever:
            stats.update(self.retriever.get_statistics())
        
        if self.chunks:
            chunk_stats = self.chunker.get_statistics(self.chunks)
            stats.update({
                'mean_chunk_size': chunk_stats['mean_chunk_size'],
                'total_chars': chunk_stats['total_chars']
            })
        
        return stats
    
    def save(self, directory: str):
        """
        Save knowledge base to directory.
        
        Args:
            directory: Directory to save to
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Save vector store
        vs_dir = Path(directory) / 'vector_store'
        if self.vector_store:
            self.vector_store.save(str(vs_dir))
        
        # Save embedder
        embedder_path = Path(directory) / 'embedder.pkl'
        self.embedder.save(str(embedder_path))
        
        # Save chunker
        chunker_path = Path(directory) / 'chunker.pkl'
        self.chunker.save(str(chunker_path))
        
        # Save documents and chunks
        import pickle
        docs_path = Path(directory) / 'documents.pkl'
        with open(docs_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'chunks': self.chunks
            }, f)
        
        logger.info(f"Saved knowledge base to {directory}")
    
    @classmethod
    def load(cls, directory: str) -> 'KnowledgeBase':
        """
        Load knowledge base from directory.
        
        Args:
            directory: Directory to load from
        
        Returns:
            Loaded KnowledgeBase instance
        """
        directory = Path(directory)
        
        # Load embedder
        embedder_path = directory / 'embedder.pkl'
        embedder = Embedder.load(str(embedder_path))
        
        # Load chunker
        chunker_path = directory / 'chunker.pkl'
        chunker = DocumentChunker.load(str(chunker_path))
        
        # Create KB instance
        kb = cls(embedder=embedder, chunker=chunker)
        
        # Load vector store
        vs_dir = directory / 'vector_store'
        if vs_dir.exists():
            kb.vector_store = VectorStore.load(str(vs_dir))
        
        # Load documents and chunks
        import pickle
        docs_path = directory / 'documents.pkl'
        with open(docs_path, 'rb') as f:
            data = pickle.load(f)
            kb.documents = data['documents']
            kb.chunks = data['chunks']
        
        # Recreate retriever
        if kb.vector_store:
            kb.retriever = Retriever(
                vector_store=kb.vector_store,
                embedder=kb.embedder,
                top_k=5,
                min_similarity=0.3,
                include_citations=True
            )
        
        logger.info(f"Loaded knowledge base from {directory}")
        return kb
    
    def _extract_sensor_stats(
        self,
        sensor_data: pd.DataFrame,
        engine_id: int,
        start_cycle: int,
        end_cycle: int
    ) -> Optional[Dict]:
        """Extract sensor statistics for a period."""
        # Resolve column names (support both CMAPSSDataLoader and legacy formats)
        id_col = 'engine_id' if 'engine_id' in sensor_data.columns else 'unit_number'
        cycle_col = 'cycle' if 'cycle' in sensor_data.columns else 'time_cycles'

        # Filter data for this engine and period
        mask = (
            (sensor_data[id_col] == engine_id) &
            (sensor_data[cycle_col] >= start_cycle) &
            (sensor_data[cycle_col] <= end_cycle)
        )
        period_data = sensor_data[mask]
        
        if period_data.empty:
            return None
        
        # Compute statistics for sensor columns
        sensor_cols = [col for col in period_data.columns 
                      if col.startswith('sensor') or col.startswith('setting')]
        
        stats = {}
        for col in sensor_cols[:10]:  # Limit to first 10 sensors
            if col in period_data.columns:
                stats[col] = {
                    'mean': float(period_data[col].mean()),
                    'std': float(period_data[col].std()),
                    'min': float(period_data[col].min()),
                    'max': float(period_data[col].max())
                }
        
        return stats
    
    def _extract_warnings(
        self,
        warnings_df: pd.DataFrame,
        engine_id: int,
        start_cycle: int,
        end_cycle: int
    ) -> Optional[List[Dict]]:
        """Extract warnings for a period."""
        # Filter warnings for this engine and period
        if 'engine_id' not in warnings_df.columns:
            return None
        
        mask = (
            (warnings_df['engine_id'] == engine_id) &
            (warnings_df['cycle'] >= start_cycle) &
            (warnings_df['cycle'] <= end_cycle)
        )
        period_warnings = warnings_df[mask]
        
        if period_warnings.empty:
            return None
        
        # Convert to list of dicts
        warnings = period_warnings.to_dict('records')
        
        return warnings


def create_test_cases_from_degradation(
    degradation_periods: List[Dict],
    n_test_cases: int = 10
) -> List[Dict]:
    """
    Create test cases for retrieval validation.
    
    Args:
        degradation_periods: List of degradation period dictionaries
        n_test_cases: Number of test cases to create
    
    Returns:
        List of test case dictionaries with query and expected results
    """
    import random
    
    if len(degradation_periods) < n_test_cases:
        n_test_cases = len(degradation_periods)
    
    # Sample random periods
    sampled = random.sample(degradation_periods, n_test_cases)
    
    test_cases = []
    for period in sampled:
        # Create query from period characteristics
        query = f"Silent degradation lasting {period.get('duration', 0)} cycles"
        if 'mean_degradation_score' in period:
            score = period['mean_degradation_score']
            query += f" with severity {score:.3f}"
        
        # Expected result is the document for this period
        test_case = {
            'query': query,
            'expected_results': period.get('engine_id', 0)
        }
        
        test_cases.append(test_case)
    
    logger.info(f"Created {len(test_cases)} test cases")
    return test_cases
