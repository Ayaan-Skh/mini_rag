"""
Retriever Module
================
This module implements the retrieval and reranking pipeline for RAG.
It retrieves relevant chunks from the vector DB and reranks them for diversity.

Two-Stage Retrieval:
1. Initial retrieval: Get top-K chunks by vector similarity
2. Reranking: Apply Maximal Marginal Relevance (MMR) for diversity

Why MMR?
- Reduces redundancy in retrieved chunks
- Balances relevance with diversity
- Prevents multiple similar chunks from dominating context
- Improves LLM's ability to synthesize different perspectives
"""

import numpy as np
from typing import List, Dict, Any
import logging

from config import settings
from vector_db import get_vector_database
from embeddings import get_embedding_generator

logger = logging.getLogger(__name__)


class Retriever:
    """
    Handles document retrieval with reranking for RAG systems.
    
    This implements a two-stage process:
    1. Retrieve more chunks than needed from vector DB
    2. Rerank using MMR to select diverse, relevant subset
    """
    
    def __init__(
        self,
        top_k: int = None,
        rerank_top_k: int = None,
        diversity_lambda: float = 0.5
    ):
        """
        Initialize the retriever.
        
        Args:
            top_k: Number of chunks to retrieve initially
            rerank_top_k: Number of chunks to return after reranking
            diversity_lambda: Balance between relevance and diversity (0-1)
                            0 = max diversity, 1 = max relevance
        
        Technical Notes:
        - Retrieve 2-3x more chunks than needed for reranking
        - Lambda of 0.5 balances relevance and diversity well
        - Adjust lambda based on use case (higher for factual QA)
        """
        self.top_k = top_k or settings.top_k_results
        self.rerank_top_k = rerank_top_k or settings.rerank_top_k
        self.diversity_lambda = diversity_lambda
        
        self.vector_db = get_vector_database()
        self.embedding_generator = get_embedding_generator()
        
        logger.info(
            f"Initialized retriever: top_k={self.top_k}, "
            f"rerank_top_k={self.rerank_top_k}, lambda={self.diversity_lambda}"
        )
    
    def _calculate_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1, vec2: Embedding vectors
        
        Returns:
            Similarity score between 0 and 1
        
        Formula: cosine_similarity = dot(A, B) / (||A|| * ||B||)
        
        Since our embeddings are already normalized by sentence-transformers,
        this simplifies to just the dot product.
        """
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        # Compute cosine similarity
        similarity = np.dot(vec1_np, vec2_np) / (
            np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np) + 1e-10
        )
        
        return float(similarity)
    
    def _mmr_rerank(
        self,
        query_embedding: List[float],
        candidates: List[Dict[str, Any]],
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using Maximal Marginal Relevance (MMR).
        
        Args:
            query_embedding: Embedding of the query
            candidates: List of candidate chunks with embeddings
            k: Number of chunks to select
        
        Returns:
            Reranked list of k chunks
        
        MMR Algorithm:
        1. Start with empty result set
        2. Iteratively select next chunk that maximizes:
           MMR = λ * sim(chunk, query) - (1-λ) * max(sim(chunk, selected))
        3. This balances relevance to query with diversity from selected chunks
        
        Intuition:
        - First chunk: most relevant to query
        - Subsequent chunks: relevant to query BUT different from already selected
        - Lambda controls the trade-off
        
        Example:
        Query: "What is machine learning?"
        Without MMR: 5 similar chunks all saying "ML is a subset of AI..."
        With MMR: Chunks covering ML definition, types, applications, history, etc.
        """
        if len(candidates) <= k:
            return candidates
        
        logger.info(f"Applying MMR reranking to {len(candidates)} candidates")
        
        # Generate embeddings for all candidate texts
        candidate_texts = [c["text"] for c in candidates]
        candidate_embeddings = self.embedding_generator.embed_batch(candidate_texts,batch_size=100)
        
        # Calculate initial relevance scores (similarity to query)
        relevance_scores = [
            self._calculate_similarity(query_embedding, emb)
            for emb in candidate_embeddings
        ]
        
        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        
        for _ in range(min(k, len(candidates))):
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance component: similarity to query
                relevance = relevance_scores[idx]
                
                # Diversity component: max similarity to already selected chunks
                if selected_indices:
                    max_similarity = max(
                        self._calculate_similarity(
                            candidate_embeddings[idx],
                            candidate_embeddings[selected_idx]
                        )
                        for selected_idx in selected_indices
                    )
                else:
                    max_similarity = 0
                
                # MMR score = λ * relevance - (1-λ) * diversity
                mmr_score = (
                    self.diversity_lambda * relevance -
                    (1 - self.diversity_lambda) * max_similarity
                )
                mmr_scores.append((idx, mmr_score))
            
            # Select chunk with highest MMR score
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Return reranked chunks
        reranked = [candidates[idx] for idx in selected_indices]
        
        logger.info(f"Reranked to {len(reranked)} diverse chunks")
        return reranked
    
    def retrieve(
        self,
        query: str,
        use_reranking: bool = True,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: The search query
            use_reranking: Whether to apply MMR reranking
            filter_metadata: Optional metadata filters
        
        Returns:
            List of relevant chunks with text, metadata, and scores
        
        Pipeline:
        1. Generate query embedding
        2. Search vector DB for top-K similar chunks
        3. (Optional) Apply MMR reranking for diversity
        4. Return final chunks with scores and metadata
        
        Usage Example:
        >>> retriever = Retriever()
        >>> chunks = retriever.retrieve("What is deep learning?")
        >>> for chunk in chunks:
        ...     print(f"Score: {chunk['score']:.3f}")
        ...     print(f"Source: {chunk['metadata']['source']}")
        ...     print(f"Text: {chunk['text'][:100]}...")
        """
        try:
            logger.info(f"Retrieving chunks for query: '{query[:100]}...'")
            
            # Generate query embedding for reranking
            query_embedding = self.embedding_generator.embed_text(query)
            
            # Retrieve initial candidates from vector DB
            candidates = self.vector_db.search(
                query=query,
                top_k=self.top_k,
                filter_metadata=filter_metadata
            )
            
            if not candidates:
                logger.warning("No candidates found for query")
                return []
            
            # Apply MMR reranking if enabled
            if use_reranking and len(candidates) > self.rerank_top_k:
                results = self._mmr_rerank(
                    query_embedding,
                    candidates,
                    self.rerank_top_k
                )
            else:
                # Just take top-K by score
                results = candidates[:self.rerank_top_k]
            
            logger.info(f"Retrieved {len(results)} final chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise


# Global instance
_retriever = None


def get_retriever() -> Retriever:
    """Get or create the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever