"""
Configuration Module
====================
This module handles all application configuration using Pydantic Settings.
It loads environment variables and provides type-safe access to configuration.

Key Features:
- Type validation for all config values
- Default values for development
- Support for .env files
- Immutable configuration (frozen=True)
"""

from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """
    Application Settings
    
    This class defines all configuration parameters for the RAG application.
    Values are loaded from environment variables with fallback to defaults.
    """
    
    # Anthropic API Configuration
    anthropic_api_key: str = Field(
        default="",
        description="API key for Anthropic Claude API"
    )
        # Groq API Configuration
    groq_api_key: str = Field(
        default="",
        description="API key for Groq API"
    )
    
    qdrant_url: str = Field(
        default="",
        description="URL for Qdrant cloud instance"
    )
    qdrant_api_key: str = Field(
        default="",
        description="API key for Qdrant cloud"
    )
    # Collection Configuration
    collection_name: str = Field(
        default="mini_rag_documents",
        description="Name of the Qdrant collection"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model name"
    )
    vector_dimension: int = Field(
        default=384,
        description="Dimension of embedding vectors"
    )
    
    # Chunking Configuration
    chunk_size: int = Field(
        default=1000,
        description="Maximum size of text chunks in characters"
    )
    chunk_overlap: int = Field(
        default=150,
        description="Overlap between consecutive chunks"
    )
    
    # Retrieval Configuration
    top_k_results: int = Field(
        default=5,
        description="Number of chunks to retrieve from vector DB"
    )
    rerank_top_k: int = Field(
        default=3,
        description="Number of top chunks after reranking"
    )
    
    # Server Configuration
    host: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    port: int = Field(
        default=8000,
        description="Server port number"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        frozen = True  # Make settings immutable
        extra = "ignore"  # Ignore extra fields


# Global settings instance
settings = Settings()