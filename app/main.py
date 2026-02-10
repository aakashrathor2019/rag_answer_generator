from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import shutil
import os
import logging
from .database import init_db, get_db
from . import models, services, schemas
from typing import List

# ============================================================================
# SETUP
# ============================================================================

logger = logging.getLogger(__name__)
app = FastAPI(title="Document Intelligence API")

# Configuration (should move to .env in production)
SIMILARITY_THRESHOLD = 0.7  # Cosine distance threshold (0.7 means 30% similarity)


@app.on_event("startup")
def startup_event():
    """Initialize database on startup."""
    logger.info("Starting up Document Intelligence API")
    init_db()
    logger.info("Database initialized successfully")


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint.
    
    Returns:
    - status: "healthy" if running
    - database: "connected" if DB is accessible
    
    Why needed: Load balancers use this to verify service is alive
    """
    try:
        db.execute("SELECT 1")
        logger.debug("Health check passed")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Database connection failed")


# ============================================================================
# DOCUMENT UPLOAD ENDPOINT
# ============================================================================

@app.post("/documents/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload and process PDF documents.
    
    Process:
    1. Save PDF to disk
    2. Extract text with PyMuPDF
    3. Split into chunks (1500 chars each, 200 char overlap)
    4. Generate embeddings for each chunk
    5. Store in PostgreSQL with pgvector
    
    Returns:
    - id: Document UUID
    - filename: Original filename
    - chunks: Number of chunks created
    
    Why chunks?
    - Embeddings work better on smaller text (~375 tokens)
    - Allows precise retrieval (full document too coarse)
    - Enables context preservation with overlap
    """
    logger.info(f"Upload started for file: {file.filename}")
    
    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"Non-PDF file rejected: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are currently supported for RAG.")
    
    temp_name = f"temp_{file.filename}"
    with open(temp_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    logger.debug(f"Temporary file created: {temp_name}")

    try:
        # Extract text from PDF
        content = services.get_pdf_text(temp_name)
        if not content.strip():
            logger.warning(f"Empty PDF uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="PDF appears to be empty or contains only images.")
        
        logger.info(f"PDF text extracted: {len(content)} characters")

        # Create document record
        new_doc = models.Document(name=file.filename)
        db.add(new_doc)
        db.flush()  # Get the ID without committing
        logger.debug(f"Document record created: {new_doc.id}")

        # Create chunks and embeddings
        text_chunks = services.create_chunks(content)
        logger.info(f"Created {len(text_chunks)} chunks from PDF")
        
        for chunk_idx, chunk_text in enumerate(text_chunks):
            vector = services.embed_text(chunk_text)
            chunk_entry = models.Chunk(
                document_id=new_doc.id,
                content=chunk_text,
                embedding=vector
            )
            db.add(chunk_entry)
            if chunk_idx % 10 == 0:
                logger.debug(f"Processed chunk {chunk_idx}/{len(text_chunks)}")
        
        db.commit()
        logger.info(f"Document upload complete: {len(text_chunks)} chunks stored")
        return {
            "id": str(new_doc.id),
            "filename": file.filename,
            "chunks": len(text_chunks)
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)
            logger.debug(f"Temporary file cleaned up: {temp_name}")


# ============================================================================
# SEARCH ENDPOINT (Retrieval Only)
# ============================================================================

@app.post("/search", response_model=schemas.SearchResponse)
def search_documents(
    request: schemas.SearchRequest,
    db: Session = Depends(get_db)
) -> schemas.SearchResponse:
    """
    Pure semantic search endpoint (retrieval only, no synthesis).
    
    Difference from /ask:
    - /search: Find relevant chunks (what is this about?)
    - /ask: Answer a question (what does this mean?)
    
    Process:
    1. Convert query to embedding vector
    2. Search database using cosine distance
    3. Filter by similarity threshold
    4. Return ranked results
    
    Why separate endpoint?
    - Different use cases require different outputs
    - Users may want sources without synthesized answer
    - Simpler performance profile (no LLM call)
    
    Args:
    - query: Search query string
    - top_k: Max number of results (1-20)
    - threshold: Min similarity score (0-1)
    
    Returns:
    - query: Echo of user query
    - total_results: Count of results
    - results: List of SearchResult objects with rank, content, score
    """
    logger.info(f"Search started: query='{request.query}', top_k={request.top_k}, threshold={request.threshold}")
    
    try:
        # Generate embedding for query
        query_vector = services.embed_text(request.query)
        logger.debug(f"Query embedding generated (384 dims)")
        
        # Search database using cosine distance
        results = db.query(
            models.Chunk,
            models.Chunk.embedding.cosine_distance(query_vector).label("distance")
        ).order_by("distance").limit(request.top_k).all()
        
        logger.info(f"Database query returned {len(results)} results")
        
        if not results:
            logger.warning("No chunks found in database")
            return schemas.SearchResponse(
                query=request.query,
                total_results=0,
                results=[]
            )
        
        # Filter by threshold and convert to response format
        search_results = []
        for rank, (chunk, distance) in enumerate(results, 1):
            # Convert distance to similarity score (0-1, where 1 is identical)
            # Cosine distance: 0 = identical, 2 = opposite
            # Similarity = 1 - (distance / 2)
            similarity_score = max(0, 1 - (distance / 2))
            
            if similarity_score >= request.threshold:
                search_results.append(
                    schemas.SearchResult(
                        rank=rank,
                        content=chunk.content,
                        score=similarity_score,
                        distance=float(distance)
                    )
                )
                logger.debug(f"Result #{rank}: score={similarity_score:.3f}, distance={distance:.3f}")
        
        logger.info(f"Search complete: {len(search_results)} results passed threshold")
        return schemas.SearchResponse(
            query=request.query,
            total_results=len(search_results),
            results=search_results
        )
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# ============================================================================
# RAG ENDPOINT (Retrieval + Synthesis)
# ============================================================================

@app.post("/ask", response_model=schemas.AnswerResponse)
def ask_question(
    query: schemas.QueryRequest,
    db: Session = Depends(get_db)
) -> schemas.AnswerResponse:
    """
    RAG (Retrieval-Augmented Generation) endpoint.
    
    Process:
    1. RETRIEVAL: Find relevant chunks via semantic search
    2. AUGMENTATION: Prepare context from chunks
    3. GENERATION: Synthesize answer from context
    
    Why RAG?
    - Answers grounded in actual documents (not hallucinated)
    - Shows sources (context_used) for verification
    - More reliable than general LLMs for factual questions
    
    Current limitation:
    - Uses rule-based synthesis (combine chunks)
    - Future: Could use LLM like GPT-4, Llama, etc.
    
    Args:
    - question: User's question
    - top_k: Number of chunks to retrieve
    
    Returns:
    - question: Echo of user question
    - answer: Synthesized answer from document chunks
    - context_used: Source chunks used (for transparency)
    - confidence: How confident in answer (0-1)
    """
    logger.info(f"Question received: '{query.question}'")
    
    try:
        # RETRIEVAL: Generate embedding and search
        query_vector = services.embed_text(query.question)
        logger.debug("Query embedding generated")
        
        results = db.query(
            models.Chunk,
            models.Chunk.embedding.cosine_distance(query_vector).label("distance")
        ).order_by("distance").limit(query.top_k).all()
        
        logger.info(f"Retrieved {len(results)} chunks from database")
        
        # Check if any results found
        if not results:
            logger.warning("No documents found - empty database")
            return schemas.AnswerResponse(
                question=query.question,
                answer="I don't have any documents in my memory yet. Please upload a PDF.",
                context_used=[],
                confidence=0.0
            )
        
        # Check if best match meets threshold
        best_match, best_distance = results[0]
        best_similarity = max(0, 1 - (best_distance / 2))  # Convert distance to similarity
        
        if best_distance > SIMILARITY_THRESHOLD:
            logger.warning(f"Best match below threshold: distance={best_distance:.3f}")
            return schemas.AnswerResponse(
                question=query.question,
                answer="I couldn't find any information in the uploaded documents that relates to your question.",
                context_used=[],
                confidence=0.0
            )
        
        # AUGMENTATION: Collect relevant chunks as context
        context_chunks = []
        context_list = []
        for chunk, distance in results:
            if distance <= SIMILARITY_THRESHOLD:
                context_chunks.append(chunk.content)
                context_list.append(chunk.content)
                logger.debug(f"Context chunk added: distance={distance:.3f}")
        
        logger.info(f"Augmented context: {len(context_chunks)} chunks, {sum(len(c) for c in context_chunks)} chars")
        
        # GENERATION: Synthesize answer from context
        if not context_chunks:
            synthesized_answer = "I found relevant documents but couldn't extract useful information."
            confidence = 0.3
        else:
            # Combine chunks into coherent context
            combined_context = " ".join(context_chunks)
            
            # Rule-based synthesis (future: replace with LLM)
            # Extract key sentences from context related to question
            synthesized_answer = _synthesize_answer(
                question=query.question,
                context=combined_context,
                chunks=context_chunks
            )
            
            # Confidence based on match quality and context length
            chunk_count = len(context_chunks)
            avg_similarity = 1 - sum(d for _, d in results[:chunk_count]) / (2 * chunk_count)
            confidence = min(1.0, best_similarity * (chunk_count / 3))  # Boost confidence with more chunks
        
        logger.info(f"Answer synthesized: confidence={confidence:.2f}")
        
        return schemas.AnswerResponse(
            question=query.question,
            answer=synthesized_answer,
            context_used=context_list,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Question answering failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _synthesize_answer(question: str, context: str, chunks: List[str]) -> str:
    """
    Synthesize an answer from document chunks.
    
    Current approach: Rule-based synthesis (combines chunks intelligently)
    
    Why synthesize instead of just return chunks?
    - Chunks are raw text fragments
    - Synthesis creates coherent narrative
    - Answers user's specific question
    
    Example:
    Q: "What is FastAPI?"
    Chunks: ["FastAPI is...", "It uses...", "FastAPI includes..."]
    Synthesized: "Based on the documents, FastAPI is... It includes... Additionally..."
    
    Future improvement:
    - Replace with LLM (GPT-4, Llama, etc.)
    - Prompt engineering for better answers
    - Multi-turn conversation support
    
    Args:
    - question: User's question
    - context: Combined text from all relevant chunks
    - chunks: Individual chunks (for ordering)
    
    Returns:
    - Synthesized answer string
    """
    # Prefix with relevance marker
    answer_prefix = "Based on the documents, "
    
    # If only one chunk, use it as primary answer
    if len(chunks) == 1:
        chunk = chunks[0]
        # Truncate to 500 chars if too long
        if len(chunk) > 500:
            chunk = chunk[:500] + "..."
        return answer_prefix + chunk
    
    # Multiple chunks: combine intelligently
    # Strategy: Use first two sentences of each chunk
    sentences = []
    for chunk in chunks[:3]:  # Use top 3 chunks max
        # Split into sentences (simple approach)
        chunk_sentences = chunk.split(". ")
        for i, sent in enumerate(chunk_sentences[:2]):  # First 2 sentences per chunk
            if sent.strip():
                sentences.append(sent.strip())
                if not sent.endswith("."):
                    sentences[-1] += "."
    
    # Join sentences into answer
    if sentences:
        synthesized = " ".join(sentences[:5])  # Max 5 sentences
        return answer_prefix + synthesized
    else:
        return answer_prefix + "I found relevant information but couldn't extract a clear answer."