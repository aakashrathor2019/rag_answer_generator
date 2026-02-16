from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
import logging
import shutil
import os
from sqlalchemy.orm import Session
from sqlalchemy import text
from .database import init_db, get_db
from . import models, services, schemas
import app.llm as llm

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")

DEFAULT_SIMILARITY_THRESHOLD = 0.3

@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/")
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "200", "database": "connected"}
    except Exception as e:
        logger.exception("Database connection failed")
        raise HTTPException(status_code=500, detail="Database unreachable")

@app.post("/documents/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = file.filename or "uploaded"
    temp_name = f"temp_{filename}"
    
    try:
        with open(temp_name, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if filename.lower().endswith(".pdf"):
            content = services.get_pdf_text(temp_name)
        else:
            with open(temp_name, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        
        if not content.strip():
            raise HTTPException(status_code=400, detail="No extractable text found")

        new_doc = models.Document(name=filename)
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)

        text_chunks = services.create_chunks(content, chunk_size_tokens=450, overlap_percent=0.15)
        for chunk_text in text_chunks:
            vector = services.embed_text(chunk_text)
            db.add(models.Chunk(document_id=new_doc.id, content=chunk_text, embedding=vector))
        
        db.commit()
        return {"id": str(new_doc.id), "filename": filename, "chunks": len(text_chunks)}

    except Exception as e:
        db.rollback()
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_name): os.remove(temp_name)

@app.post("/search", response_model=schemas.SearchResponse)
def search_documents(request: schemas.SearchRequest, db: Session = Depends(get_db)):

    query_vector = services.embed_text(request.query)

    results = (
        db.query(
            models.Chunk,
            models.Chunk.embedding.cosine_distance(query_vector).label("distance")
        )
        .order_by(text("distance ASC"))
        .limit(request.top_k)
        .all()
    )

    search_results = []

    for rank, (chunk, distance) in enumerate(results, 1):
        similarity = 1 - float(distance)
        if similarity < request.threshold:
            continue

        search_results.append(
            schemas.SearchResult(
                rank=rank,
                content=chunk.content,
                score=round(similarity, 4),
                distance=float(distance)
            )
        )

    return schemas.SearchResponse(
        query=request.query,
        total_results=len(search_results),
        results=search_results
    )


@app.post("/ask", response_model=schemas.AnswerResponse)
def ask_question(query: schemas.QueryRequest, db: Session = Depends(get_db)):

    threshold = getattr(query, "threshold", DEFAULT_SIMILARITY_THRESHOLD)

    query_vector = services.embed_text(query.question)

    results = (
        db.query(
            models.Chunk,
            models.Chunk.embedding.cosine_distance(query_vector).label("distance")
        )
        .order_by(text("distance ASC"))
        .limit(query.top_k)
        .all()
    )

    contexts = []

    for chunk, distance in results:
        similarity = 1 - float(distance)

        if similarity >= threshold:
            contexts.append(chunk.content)

    # Case 1: Context found → RAG Answer
    if contexts:

        combined_context = "\n\n".join(contexts)

        try:
            answer = llm.generate_answer(
                context=combined_context,
                question=query.question
            )

        except Exception as e:
            logger.error(f"Ollama Error: {e}")
            answer = "Local LLM is unavailable."

    # Case 2: No context → Pure LLM
    else:

        try:
            answer = llm.generate_answer(
                context="",
                question=query.question
            )

        except Exception as e:
            logger.error(f"Ollama Error: {e}")
            answer = "Local LLM is unavailable."


    return schemas.AnswerResponse(
        question=query.question,
        answer=answer,
        context_used=contexts
    )
