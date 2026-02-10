from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """
    Request schema for /ask endpoint (RAG).
    
    Why needed:
    - Validates incoming JSON
    - Ensures question exists
    - Allows customizing top_k (flexibility)
    """
    question: str = Field(..., min_length=1, max_length=1000, description="Question to answer")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of chunks to retrieve")


class SearchRequest(BaseModel):
    """
    Request schema for /search endpoint (semantic search).
    
    Why separate from /ask?
    - /search: Just find information (retrieval only)
    - /ask: Find + answer (retrieval + generation)
    - Users may want search without synthesis
    """
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold (0-1)")


class SearchResult(BaseModel):
    """
    Single search result.
    
    Why this structure:
    - rank: Show position (relevance order)
    - content: The actual chunk text
    - score: Confidence 0-1 (1=perfect match)
    - distance: For debugging/analysis
    """
    rank: int = Field(..., description="Result rank (1-based)")
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0=different, 1=identical)")
    distance: Optional[float] = Field(None, description="Raw cosine distance (for debugging)")


class SearchResponse(BaseModel):
    """
    Response schema for /search endpoint.
    
    Structure:
    - query: Echo back user query
    - total_results: Count of results returned
    - results: List of SearchResult objects
    
    Why this format?
    - Consistent API responses
    - Clients know what to expect
    - Easy to parse and handle
    """
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., ge=0, description="Number of results")
    results: List[SearchResult] = Field(default=[], description="List of search results")


class AnswerResponse(BaseModel):
    """
    Response schema for /ask endpoint (RAG).
    
    Why these fields?
    - question: Echo user's question
    - answer: Synthesized answer from chunks
    - context_used: Sources for answer (transparency)
    - confidence: How sure we are (0-1 scale)
    
    This is RAG:
    1. Retrieval: Find context_used chunks
    2. Augmentation: Add to prompt as context
    3. Generation: Generate answer from context
    """
    question: str = Field(..., description="User's question")
    answer: str = Field(..., description="Generated answer based on documents")
    context_used: List[str] = Field(default=[], description="Document chunks used to generate answer")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score (0-1)")