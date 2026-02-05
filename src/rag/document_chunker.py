"""
Document Chunker for RAG Pipeline

Splits documents into semantic chunks for embedding and retrieval.
Supports multiple chunking strategies:
- Fixed-size chunking with overlap
- Sentence-based chunking
- Paragraph-based chunking
- Semantic chunking (sentence similarity)
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Split documents into chunks for efficient embedding and retrieval.
    
    Attributes:
        chunk_size (int): Maximum characters per chunk
        chunk_overlap (int): Overlap between consecutive chunks
        strategy (str): Chunking strategy ('fixed', 'sentence', 'paragraph', 'semantic')
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: str = 'sentence'
    ):
        """
        Initialize document chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap characters between chunks
            strategy: Chunking strategy
                - 'fixed': Fixed-size chunks with overlap
                - 'sentence': Split by sentences, group to target size
                - 'paragraph': Split by paragraphs
                - 'semantic': Group semantically similar sentences
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        
        logger.info(
            f"Initialized DocumentChunker: size={chunk_size}, "
            f"overlap={chunk_overlap}, strategy={strategy}"
        )
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
        
        Returns:
            List of chunk dictionaries with 'text', 'metadata', 'chunk_id'
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        # Clean text
        text = self._clean_text(text)
        
        # Choose chunking strategy
        if self.strategy == 'fixed':
            chunks = self._chunk_fixed_size(text)
        elif self.strategy == 'sentence':
            chunks = self._chunk_sentences(text)
        elif self.strategy == 'paragraph':
            chunks = self._chunk_paragraphs(text)
        elif self.strategy == 'semantic':
            chunks = self._chunk_semantic(text)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Add metadata and chunk IDs
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_dict = {
                'text': chunk_text,
                'chunk_id': i,
                'metadata': metadata or {}
            }
            result.append(chunk_dict)
        
        logger.info(f"Created {len(result)} chunks from text of length {len(text)}")
        return result
    
    def chunk_documents(
        self,
        documents: List[Dict],
        text_field: str = 'text',
        metadata_fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dictionaries
            text_field: Field containing text to chunk
            metadata_fields: Fields to include in chunk metadata
        
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        metadata_fields = metadata_fields or []
        
        for doc_id, doc in enumerate(documents):
            if text_field not in doc:
                logger.warning(f"Document {doc_id} missing '{text_field}' field")
                continue
            
            # Extract metadata
            metadata = {'doc_id': doc_id}
            for field in metadata_fields:
                if field in doc:
                    metadata[field] = doc[field]
            
            # Chunk document
            chunks = self.chunk_text(doc[text_field], metadata=metadata)
            all_chunks.extend(chunks)
        
        logger.info(
            f"Chunked {len(documents)} documents into {len(all_chunks)} chunks"
        )
        return all_chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        return text.strip()
    
    def _chunk_fixed_size(self, text: str) -> List[str]:
        """Fixed-size chunking with overlap."""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < text_len:
                # Look for sentence end within next 100 chars
                chunk_text = text[start:end + 100]
                match = re.search(r'[.!?]\s', chunk_text)
                if match:
                    end = start + match.end()
            
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap
        
        return [c for c in chunks if c]
    
    def _chunk_sentences(self, text: str) -> List[str]:
        """Sentence-based chunking grouped to target size."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_size + sentence_len > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append(' '.join(current_chunk))
                # Keep overlap (last sentence)
                current_chunk = [current_chunk[-1]] if current_chunk else []
                current_size = len(current_chunk[0]) if current_chunk else 0
            
            current_chunk.append(sentence)
            current_size += sentence_len
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _chunk_paragraphs(self, text: str) -> List[str]:
        """Paragraph-based chunking."""
        # Split by double newlines or multiple spaces
        paragraphs = re.split(r'\n\s*\n|\s{4,}', text)
        
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph too large, split further
            if len(para) > self.chunk_size:
                sub_chunks = self._chunk_sentences(para)
                chunks.extend(sub_chunks)
            else:
                chunks.append(para)
        
        return chunks
    
    def _chunk_semantic(self, text: str) -> List[str]:
        """
        Semantic chunking (placeholder - requires embeddings).
        Falls back to sentence chunking for now.
        """
        logger.warning("Semantic chunking not fully implemented, using sentence chunking")
        return self._chunk_sentences(text)
    
    def get_statistics(self, chunks: List[Dict]) -> Dict:
        """
        Get statistics about chunks.
        
        Args:
            chunks: List of chunk dictionaries
        
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                'n_chunks': 0,
                'total_chars': 0,
                'mean_chunk_size': 0,
                'median_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
        
        chunk_sizes = [len(c['text']) for c in chunks]
        
        return {
            'n_chunks': len(chunks),
            'total_chars': sum(chunk_sizes),
            'mean_chunk_size': np.mean(chunk_sizes),
            'median_chunk_size': np.median(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'std_chunk_size': np.std(chunk_sizes)
        }
    
    def save(self, filepath: str):
        """Save chunker configuration."""
        import pickle
        config = {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'strategy': self.strategy
        }
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)
        logger.info(f"Saved chunker config to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DocumentChunker':
        """Load chunker configuration."""
        import pickle
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        logger.info(f"Loaded chunker config from {filepath}")
        return cls(**config)


def create_failure_document(
    engine_id: int,
    degradation_period: Dict,
    sensor_data: Optional[Dict] = None,
    warnings: Optional[List[Dict]] = None
) -> Dict:
    """
    Create a structured failure document from PHASE 5 outputs.
    
    Args:
        engine_id: Engine identifier
        degradation_period: Degradation period dictionary from DegradationLabeler
        sensor_data: Optional sensor readings during degradation
        warnings: Optional list of warnings generated
    
    Returns:
        Structured document dictionary ready for chunking
    """
    # Build text description
    text_parts = []
    
    # Header
    text_parts.append(f"Engine {engine_id} Silent Degradation Incident")
    text_parts.append(f"Duration: {degradation_period.get('duration', 0)} cycles")
    text_parts.append(
        f"Cycles: {degradation_period.get('start', 0)} to "
        f"{degradation_period.get('end', 0)}"
    )
    
    # Degradation characteristics
    if 'mean_degradation_score' in degradation_period:
        score = degradation_period['mean_degradation_score']
        text_parts.append(f"Mean degradation score: {score:.3f}")
    
    if 'anomaly_rate' in degradation_period:
        rate = degradation_period['anomaly_rate']
        text_parts.append(f"Anomaly rate: {rate:.1%}")
    
    if 'n_change_points' in degradation_period:
        n_cp = degradation_period['n_change_points']
        text_parts.append(f"Change points detected: {n_cp}")
    
    # Sensor patterns
    if sensor_data:
        text_parts.append("Sensor deviation patterns:")
        for sensor, stats in sensor_data.items():
            if isinstance(stats, dict):
                mean_val = stats.get('mean', 0)
                std_val = stats.get('std', 0)
                text_parts.append(f"  {sensor}: mean={mean_val:.2f}, std={std_val:.2f}")
    
    # Warning information
    if warnings:
        text_parts.append(f"Warnings generated: {len(warnings)}")
        if warnings:
            first_warning = warnings[0]
            if 'lead_time' in first_warning:
                lead = first_warning['lead_time']
                text_parts.append(f"First warning lead-time: {lead} cycles before failure")
    
    # Combine into document
    text = '. '.join(text_parts) + '.'
    
    # Metadata
    metadata = {
        'engine_id': engine_id,
        'failure_type': 'silent_degradation',
        'duration': degradation_period.get('duration', 0),
        'start_cycle': degradation_period.get('start', 0),
        'end_cycle': degradation_period.get('end', 0),
        'severity': degradation_period.get('mean_degradation_score', 0),
        'source': 'phase5_degradation_labeler'
    }
    
    return {
        'text': text,
        'metadata': metadata
    }
