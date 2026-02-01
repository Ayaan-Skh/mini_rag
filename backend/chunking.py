# Used for chunking the data symentically and overlap bits to not to lose the context

import re
from typing import List,Dict,Optional,Any
import logging

from config import settings

logger=logging.getLogger(__name__)

class TextChunk:
    """
    Represents a chunk of text with its metadata.
    
    Attributes:
        text: The actual text content
        metadata: Dictionary containing source, position, etc.
        chunk_id: Unique identifier for this chunk
    """
    
    def __init__(self, text: str, metadata: Dict[str, Any], chunk_id: str):
        self.text = text
        self.metadata = metadata
        self.chunk_id = chunk_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id
        }

class DocumentChunker:
    """
    Handles intelligent splitting
    """
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        logger.info("Chunker initialized")
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        This is used to split text into sentences
        
        """
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks while preserving sentence boundaries.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        metadata = metadata or {}
        text = text.strip()
        
        # If text is smaller than chunk size, create a single chunk
        if len(text) <= self.chunk_size:
            logger.info(f"Text is small ({len(text)} chars), creating single chunk")
            chunk_metadata = {
                **metadata,
                "position": 0,
                "char_start": 0,
                "char_end": len(text)
            }
            chunk_id = f"{metadata.get('source', 'unknown')}_0"
            return [TextChunk(text, chunk_metadata, chunk_id)]
        
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        logger.info(f"Chunking text: {len(text)} chars, {len(sentences)} sentences")
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Join sentences into chunk text
                chunk_text = " ".join(current_chunk)
                
                # Create chunk with position metadata
                chunk_metadata = {
                    **metadata,
                    "position": len(chunks),
                    "char_start": max(0, sum(len(c.text) for c in chunks) - self.chunk_overlap),
                    "char_end": sum(len(c.text) for c in chunks) + len(chunk_text)
                }
                
                chunk_id = f"{metadata.get('source', 'unknown')}_{len(chunks)}"
                chunks.append(TextChunk(chunk_text, chunk_metadata, chunk_id))
                
                # Calculate overlap: keep last N characters worth of sentences
                overlap_length = 0
                overlap_sentences = []
                
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_metadata = {
                **metadata,
                "position": len(chunks),
                "char_start": max(0, sum(len(c.text) for c in chunks) - self.chunk_overlap),
                "char_end": sum(len(c.text) for c in chunks) + len(chunk_text)
            }
            chunk_id = f"{metadata.get('source', 'unknown')}_{len(chunks)}"
            chunks.append(TextChunk(chunk_text, chunk_metadata, chunk_id))
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def chunk_documents(
        self,
        documents: List[Dict[str, str]]
    ) -> List[TextChunk]:
        """
        Chunk multiple documents at once.
        
        Args:
            documents: List of dicts with 'text' and optional 'metadata' keys
        
        Returns:
            List of all chunks from all documents
        
        This is useful for batch processing during initial indexing.
        """
        all_chunks = []
        
        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            # Ensure source is set
            if "source" not in metadata:
                metadata["source"] = f"document_{i}"
            
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks


# Global instance
_chunker = None


def get_chunker() -> DocumentChunker:
    """Get or create the global chunker instance."""
    global _chunker
    if _chunker is None:
        _chunker = DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )  # ✅ Now has arguments
    return _chunker  
        
        
        
        
        
        
        
        
        