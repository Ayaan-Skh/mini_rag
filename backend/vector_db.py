"""
Vector Database Module
======================
This module handles all interactions with Qdrant vector database.
It provides functionality for:
- Creating and managing collections
- Upserting document chunks with embeddings
- Performing similarity search
- Managing vector indices

Qdrant Choice:
- Cloud-hosted for easy deployment
- Fast approximate nearest neighbor (ANN) search
- Built-in filtering and metadata support
- Excellent Python SDK
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from typing import List, Dict, Any, Optional
import logging
import uuid

from config import settings
from embeddings import get_embedding_generator

logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    Manages vector database operations with Qdrant.
    
    This class handles:
    - Collection initialization
    - Document indexing with embeddings
    - Similarity search (retrieval)
    - Metadata filtering
    """
    
    def __init__(
        self,
        url: str = None,
        api_key: str = None,
        collection_name: str = None
    ):
        """
        Initialize connection to Qdrant cloud instance.
        
        Args:
            url: Qdrant cloud URL
            api_key: Qdrant API key
            collection_name: Name of the collection to use
        
        Connection Details:
        - Uses HTTPS for secure communication
        - API key authentication
        - Persistent connection for all operations
        """
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.collection_name = collection_name or settings.collection_name
        
        logger.info(f"Connecting to Qdrant at {self.url}")
        
        try:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=30
            )
            logger.info("Successfully connected to Qdrant")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
        
        self.embedding_generator = get_embedding_generator()
    
    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create the vector collection with proper configuration.
        
        Args:
            recreate: If True, delete existing collection and create new one
        
        Returns:
            True if collection was created or already exists
        
        Collection Configuration:
        - Distance: Cosine similarity (range: -1 to 1, higher is more similar)
        - Dimension: 384 for all-MiniLM-L6-v2
        - On-disk storage: False (use RAM for faster search)
        
        Technical Notes:
        - Cosine distance is ideal for sentence embeddings
        - RAM storage trades memory for speed (good for small-medium datasets)
        - HNSW index used by default (fast approximate search)
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if exists and recreate:
                logger.info(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                exists = False
            
            if not exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.vector_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection created successfully")
                return True
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                return True
                
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Insert or update document chunks in the vector database.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
            batch_size: Number of chunks to upload at once
        
        Returns:
            Number of chunks successfully upserted
        
        Process:
        1. Generate embeddings for all chunk texts (batched for efficiency)
        2. Create Qdrant points with embeddings and metadata
        3. Upsert to database in batches
        
        Why Batching:
        - Reduces network overhead
        - Qdrant can optimize batch operations
        - Prevents timeout on large uploads
        
        Metadata Storage:
        - All chunk metadata is stored as payload
        - Can be used for filtering during search
        - Enables accurate citation generation
        """
        try:
            logger.info(f"Upserting {len(chunks)} chunks to vector database")
            
            # Extract texts for embedding
            texts = [chunk["text"] for chunk in chunks]
            
            # Generate embeddings in batch (much faster than one-by-one)
            logger.info("Generating embeddings...")
            embeddings = self.embedding_generator.embed_batch(texts,batch_size=batch_size)
            
            # Create Qdrant points
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=str(uuid.uuid4()),  # Generate unique ID
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "metadata": chunk.get("metadata", {}),
                        "chunk_id": chunk.get("chunk_id", f"chunk_{i}")
                    }
                )
                points.append(point)
            
            # Upsert in batches
            logger.info(f"Uploading points in batches of {batch_size}...")
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.info(f"Uploaded batch {i // batch_size + 1}/{(len(points) + batch_size - 1) // batch_size}")
            
            logger.info(f"Successfully upserted {len(chunks)} chunks")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error upserting chunks: {e}")
            raise
    
    def search(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query: The search query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"source": "doc1.pdf"})
        
        Returns:
            List of matching chunks with scores and metadata
        
        Search Process:
        1. Generate embedding for query text
        2. Perform approximate nearest neighbor search in Qdrant
        3. Apply metadata filters if provided
        4. Return top K results with similarity scores
        
        Similarity Scores:
        - Range: 0 to 1 (cosine similarity)
        - Higher score = more similar
        - Typical good matches: > 0.7
        - Threshold depends on use case
        """
        top_k = top_k or settings.top_k_results
        
        try:
            logger.info(f"Searching for: '{query[:100]}...' (top_k={top_k})")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.embed_text(query)
            
            # Build filter if metadata provided
            query_filter = None
            if filter_metadata:
                conditions = []
                for key, value in filter_metadata.items():
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    query_filter = Filter(must=conditions)
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=query_filter
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "text": result.payload["text"],
                    "metadata": result.payload.get("metadata", {}),
                    "chunk_id": result.payload.get("chunk_id", ""),
                    "score": result.score
                })
            
            logger.info(f"Found {len(results)} matching chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection stats (count, config, etc.)
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}


# Global instance
_vector_db = None


def get_vector_database() -> VectorDatabase:
    """Get or create the global vector database instance."""
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDatabase()
    return _vector_db