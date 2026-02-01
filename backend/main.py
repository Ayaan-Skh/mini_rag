"""
FastAPI Backend Application
===========================
This is the main backend server for the Mini RAG application.

API Endpoints:
- POST /upload: Upload and process documents
- POST /query: Query the RAG system
- GET /health: Health check
- GET /stats: Get collection statistics

Technical Stack:
- FastAPI: Modern, fast web framework
- CORS: Enable frontend communication
- File uploads: Handle multipart form data
- Error handling: Comprehensive exception handling
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import tempfile
import os
from pathlib import Path
import asyncio

# Import our modules
from config import settings
from document_processor import get_document_processor
from chunking import get_chunker
from vector_db import get_vector_database
from retriever import get_retriever
from llm import get_llm_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Mini RAG API",
    description="A production-ready RAG system with document upload and question answering",
    version="1.0.0"
)

# Configure CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Pydantic Models ====================

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="The question to ask", min_length=1)
    use_reranking: bool = Field(True, description="Whether to use MMR reranking")
    top_k: Optional[int] = Field(None, description="Number of chunks to retrieve")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "use_reranking": True,
                "top_k": 5
            }
        }


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="Generated answer with citations")
    sources: List[dict] = Field(..., description="List of cited sources")
    usage: dict = Field(..., description="Token usage statistics")
    retrieval_time: float = Field(..., description="Time taken for retrieval (seconds)")
    generation_time: float = Field(..., description="Time taken for generation (seconds)")


class UploadResponse(BaseModel):
    """Response model for upload endpoint."""
    message: str
    filename: str
    chunks_created: int
    status: str


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""
    collection_name: str
    vectors_count: int
    points_count: int
    status: str


# ==================== Startup Event ====================

@app.on_event("startup")
async def startup_event():
    """
    Initialize services on application startup.
    
    This ensures:
    1. Vector database connection is established
    2. Collection is created if it doesn't exist
    3. ML models are loaded
    4. All services are ready before accepting requests
    """
    logger.info("Starting Mini RAG API...")
    
    try:
        # Initialize vector database
        logger.info("Initializing vector database...")
        vector_db = get_vector_database()
        vector_db.create_collection(recreate=False)
        
        # Initialize embedding model (downloads if first time)
        logger.info("Loading embedding model...")
        get_retriever()
        
        # Initialize LLM service
        logger.info("Initializing LLM service...")
        get_llm_service()
        
        logger.info("✓ All services initialized successfully")
        logger.info(f"✓ Using collection: {settings.collection_name}")
        logger.info(f"✓ Server running on {settings.host}:{settings.port}")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Mini RAG API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "upload": "/upload",
            "query": "/query",
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Verifies:
    - API is running
    - Vector database is accessible
    - Services are initialized
    
    Returns:
    - 200 if healthy
    - 500 if any issues
    """
    try:
        # Try to get collection info to verify DB connection
        vector_db = get_vector_database()
        info = vector_db.get_collection_info()
        
        return {
            "status": "healthy",
            "database": "connected",
            "collection": info.get("name", "unknown"),
            "vectors_count": info.get("vectors_count", 0)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get collection statistics.
    
    Returns:
    - Collection name
    - Number of vectors/documents
    - Collection status
    
    Useful for:
    - Monitoring document count
    - Verifying uploads
    - Debugging
    """
    try:
        vector_db = get_vector_database()
        info = vector_db.get_collection_info()
        
        return StatsResponse(
            collection_name=info.get("name", settings.collection_name),
            vectors_count=info.get("vectors_count", 0),
            points_count=info.get("points_count", 0),
            status=info.get("status", "unknown")
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (PDF, DOCX, TXT, MD)")
):
    """
    Upload and process a document.
    
    Process:
    1. Validate file type
    2. Save to temporary location
    3. Extract text using document processor
    4. Split into chunks
    5. Generate embeddings
    6. Store in vector database
    
    Args:
        file: Uploaded file (multipart/form-data)
    
    Returns:
        Upload status with chunk count
    
    Supported Formats:
    - PDF (.pdf)
    - Word (.docx)
    - Plain text (.txt)
    - Markdown (.md)
    
    Example:
    ```bash
    curl -X POST "http://localhost:8000/upload" \
         -F "file=@document.pdf"
    ```
    """
    try:
        logger.info(f"Received file upload: {file.filename}")
        
        # Check file type
        logger.info("Befor calling get_doc_processor")
        doc_processor = get_document_processor()
        if not doc_processor.is_supported(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: {', '.join(doc_processor.get_supported_extensions())}"
            )
        
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Process document
            logger.info(f"Processing document: {file.filename}")
            doc_result = doc_processor.process_file(temp_path, file.filename)
            
            # Chunk the text
            logger.info(f"processed file {doc_result}")
            logger.info("Chunking document...")
            chunker = get_chunker()
            chunks = chunker.chunk_text(
                text=doc_result['text'],
                metadata=doc_result['metadata']
            )
            
            # Convert chunks to dict format
            chunk_dicts = [chunk.to_dict() for chunk in chunks]
            
            # Store in vector database
            logger.info(f"Storing {len(chunk_dicts)} chunks in vector database...")
            vector_db = get_vector_database()
            chunks_stored = vector_db.upsert_chunks(chunk_dicts)
            
            logger.info(f"✓ Successfully processed {file.filename}: {chunks_stored} chunks")
            
            return UploadResponse(
                message="Document uploaded and processed successfully",
                filename=file.filename,
                chunks_created=chunks_stored,
                status="success"
            )
            
        finally:
            # Clean up temp file
            os.unlink(temp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    Process:
    1. Retrieve relevant chunks from vector DB
    2. (Optional) Rerank chunks using MMR
    3. Generate answer using Claude
    4. Extract and return citations
    
    Args:
        request: QueryRequest with query text and options
    
    Returns:
        QueryResponse with answer, sources, and timing info
    
    Example Request:
    ```json
    {
        "query": "What is machine learning?",
        "use_reranking": true,
        "top_k": 5
    }
    ```
    
    Example Response:
    ```json
    {
        "answer": "Machine learning is a subset of AI [1] that enables...",
        "sources": [
            {
                "citation": "[1]",
                "source": "ml_intro.pdf",
                "text": "Machine learning is...",
                "score": 0.89
            }
        ],
        "usage": {
            "input_tokens": 450,
            "output_tokens": 120
        },
        "retrieval_time": 0.245,
        "generation_time": 1.823
    }
    ```
    """
    try:
        import time
        
        logger.info(f"Received query: '{request.query[:100]}...'")
        
        # Retrieve relevant chunks
        retrieval_start = time.time()
        retriever = get_retriever()
        
        # Override top_k if provided
        if request.top_k:
            retriever.top_k = request.top_k
        
        chunks = retriever.retrieve(
            query=request.query,
            use_reranking=request.use_reranking
        )
        retrieval_time = time.time() - retrieval_start
        
        logger.info(f"Retrieved {len(chunks)} chunks in {retrieval_time:.3f}s")
        
        if not chunks:
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your query. Please try uploading relevant documents first.",
                sources=[],
                usage={},
                retrieval_time=retrieval_time,
                generation_time=0.0
            )
        
        # Generate answer
        generation_start = time.time()
        llm_service = get_llm_service()
        result = llm_service.generate_answer(
            query=request.query,
            chunks=chunks
        )
        generation_time = time.time() - generation_start
        
        logger.info(f"Generated answer in {generation_time:.3f}s")
        
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'],
            usage=result['usage'],
            retrieval_time=retrieval_time,
            generation_time=generation_time
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/query/stream")
async def query_rag_stream(request: QueryRequest):
    """
    Query the RAG system with streaming response.
    
    This endpoint streams the answer as it's being generated,
    providing better UX for long responses.
    
    Returns:
        Server-Sent Events (SSE) stream of answer chunks
    
    Example:
    ```javascript
    const response = await fetch('/query/stream', {
        method: 'POST',
        body: JSON.stringify({query: "What is AI?"})
    });
    
    const reader = response.body.getReader();
    while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        console.log(new TextDecoder().decode(value));
    }
    ```
    """
    try:
        logger.info(f"Received streaming query: '{request.query[:100]}...'")
        
        # Retrieve chunks
        retriever = get_retriever()
        chunks = retriever.retrieve(
            query=request.query,
            use_reranking=request.use_reranking
        )
        
        if not chunks:
            async def empty_stream():
                yield "I couldn't find any relevant information to answer your query."
            return StreamingResponse(empty_stream(), media_type="text/plain")
        
        # Stream answer generation
        llm_service = get_llm_service()
        
        async def generate():
            for chunk in llm_service.generate_answer_stream(request.query, chunks):
                yield chunk
                await asyncio.sleep(0.01)  # Small delay for smooth streaming
        
        return StreamingResponse(generate(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Error in streaming query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# ==================== Error Handlers ====================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.
    
    Provides consistent error responses and logging.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred. Please try again or contact support.",
            "error": str(exc)
        }
    )


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Mini RAG API server...")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )