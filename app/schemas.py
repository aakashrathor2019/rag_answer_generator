from pydantic import BaseModel
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    top_k: int
    threshold: Optional[float] = 0.3  # optional, default 0.3

class QueryRequest(BaseModel):
    question: str
    top_k: int
    threshold: Optional[float] = 0.3

class SearchResult(BaseModel):
    rank: int
    content: str
    score: float
    distance: float

class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResult]

class AnswerResponse(BaseModel):
    question: str
    answer: str
    context_used: List[str]
