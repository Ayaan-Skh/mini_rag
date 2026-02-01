"""
LLM Module
==========
This module handles interaction with the Claude API for answer generation.
It formats retrieved chunks, generates prompts, and extracts inline citations.

Key Features:
- Structured prompts with retrieved context
- Inline citation generation using [1], [2], etc.
- Streaming support for real-time responses
- Token usage tracking for cost monitoring
"""

from anthropic import Anthropic
from typing import List, Dict, Any, Optional, Generator
import logging
import re
from groq import Groq
from config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Manages Claude API interactions for RAG-based question answering.
    
    This service:
    1. Formats retrieved chunks into context
    2. Constructs prompts with citations
    3. Generates answers with inline citations
    4. Handles streaming responses
    """
    
    def __init__(self, api_key: str = None , model: str = "openai/gpt-oss-120b"):
        self.api_key = api_key or settings.groq_api_key
        self.model = model
        self.client = Groq(api_key=self.api_key)

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into numbered context for the LLM.
        
        Args:
            chunks: List of retrieved chunks with text and metadata
        
        Returns:
            Formatted context string with source citations
        
        Format:
        [1] Source: document.pdf, Position: 2
        Text: "Content of first chunk..."
        
        [2] Source: document.pdf, Position: 5
        Text: "Content of second chunk..."
        
        This format:
        - Numbers chunks for easy citation ([1], [2], etc.)
        - Includes source information for user reference
        - Clearly separates chunks for LLM clarity
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "Unknown")
            position = metadata.get("position", "Unknown")
            
            context_parts.append(
                f"[{i}] Source: {source}, Position: {position}\n"
                f"Text: {chunk['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def _construct_prompt(self, query: str, context: str) -> str:
        """
        Construct the prompt for Claude with context and instructions.
        
        Args:
            query: User's question
            context: Formatted retrieved chunks
        
        Returns:
            Complete prompt string
        
        Prompt Structure:
        1. System-like instructions (role, task, rules)
        2. Retrieved context with citations
        3. User query
        4. Output format instructions
        
        Citation Instructions:
        - Use [1], [2], etc. for inline citations
        - Cite specific chunks that support each claim
        - Multiple citations allowed: [1,2]
        - No-answer cases handled gracefully
        """
        prompt = f"""You are a helpful AI assistant answering questions based on provided context.

CONTEXT:
{context}

INSTRUCTIONS:
1. Answer the user's query using ONLY the information from the context above
2. Use inline citations like [1], [2] to reference specific sources
3. If a claim is supported by multiple sources, cite all: [1,2]
4. If the context doesn't contain enough information to answer, say so clearly
5. Be concise but complete
6. Maintain a professional, informative tone

USER QUERY:
{query}

Answer the query based on the context provided. Remember to use inline citations [1], [2], etc."""
        
        return prompt
    
    def generate_answer(self, query: str, chunks: List[Dict[str, Any]], max_tokens: int = 1000):
        try:
            context = self._format_context(chunks)
            prompt = self._construct_prompt(query, context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }
            
            sources = self._extract_sources(answer, chunks)
            return {"answer": answer, "sources": sources, "usage": usage}            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def _extract_sources(
        self,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract citations from answer and map to source chunks.
        
        Args:
            answer: Generated answer with citations
            chunks: Original context chunks
        
        Returns:
            List of cited sources with metadata
        
        Citation Extraction:
        - Regex pattern: \[(\d+(?:,\d+)*)\]
        - Matches: [1], [2,3], [1,2,4]
        - Extracts all unique citation numbers
        - Maps to corresponding chunks
        
        Example:
        Answer: "ML is a field of AI [1,2] that enables computers to learn [3]."
        Extracts: [1], [2], [3]
        Maps to: chunks[0], chunks[1], chunks[2]
        """
        # Find all citation patterns like [1], [2], [1,2]
        citation_pattern = r'\[(\d+(?:,\d+)*)\]'
        matches = re.findall(citation_pattern, answer)
        
        # Extract unique citation numbers
        cited_indices = set()
        for match in matches:
            indices = match.split(',')
            cited_indices.update(int(idx.strip()) for idx in indices)
        
        # Map to source chunks
        sources = []
        for idx in sorted(cited_indices):
            # Convert to 0-indexed
            chunk_idx = idx - 1
            if 0 <= chunk_idx < len(chunks):
                chunk = chunks[chunk_idx]
                metadata = chunk.get("metadata", {})
                
                sources.append({
                    "citation": f"[{idx}]",
                    "source": metadata.get("source", "Unknown"),
                    "position": metadata.get("position", "Unknown"),
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "score": chunk.get("score", 0.0)
                })
        
        logger.info(f"Extracted {len(sources)} cited sources from answer")
        return sources
    
    def generate_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        max_tokens: int = 1000
    ) -> Generator[str, None, None]:
        """
        Generate answer with streaming for real-time display.
        
        Args:
            query: User's question
            chunks: Retrieved context chunks
            max_tokens: Maximum tokens in response
        
        Yields:
            Text chunks as they're generated
        
        Streaming Benefits:
        - Better user experience (see answer as it's generated)
        - Lower perceived latency
        - Can cancel long responses early
        
        Usage:
        >>> for chunk in llm.generate_answer_stream(query, chunks):
        ...     print(chunk, end='', flush=True)
        """
        try:
            if not chunks:
                yield "I don't have enough information to answer this query."
                return
            
            context = self._format_context(chunks)
            prompt = self._construct_prompt(query, context)
            
            logger.info(f"Starting streaming response for query: '{query[:100]}...'")
            
            # Use streaming API
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            ) as stream:
                for text in stream.text_stream:
                    yield text
            
            logger.info("Completed streaming response")
            
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            yield f"\n\nError: {str(e)}"


# Global instance
_llm_service = None


def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service