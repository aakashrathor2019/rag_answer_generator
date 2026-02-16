# Updated Readme file 
Document Intelligence API (RAG-Based System)
Overview

This project is a Document Intelligence & Question Answering API built using FastAPI, PostgreSQL (pgvector), and Docker.
It allows users to upload documents, perform semantic search using vector embeddings, and ask questions based on uploaded content using a Retrieval-Augmented Generation (RAG) approach.
The system follows a modular and scalable backend architecture and is fully containerized for easy deployment.

# Features
Upload PDF and text documents
Automatic document chunking and embedding
Vector similarity search using PostgreSQL (pgvector)
Semantic search API
Question answering based on document content
Dockerized setup for easy execution
Clean and modular code structure

# Architecture
The project follows a layered architecture:
Client
   ↓
FastAPI (API Layer)
   ↓
Service Layer (Embedding / Search / QA)
   ↓
PostgreSQL + pgvector (Storage Layer)


Data Flow:
User uploads a document
Document is parsed and split into chunks
Chunks are converted into embeddings
Embeddings are stored in PostgreSQL
Search/QA retrieves relevant chunks
LLM generates final response (RAG)

# Project Structure
app/
 ├── main.py         # API entry point
 ├── models.py       # Database models
 ├── schemas.py      # Pydantic schemas
 ├── services.py     # Business logic
 ├── database.py     # DB connection
 ├── llm.py          # LLM integration
Dockerfile
docker-compose.yml
requirements.txt
README.md

# Tech Stack
Backend: FastAPI (Python)
Database: PostgreSQL + pgvector
Embeddings: SentenceTransformers
LLM: OpenAI / Local LLM (Optional)
Containerization: Docker, Docker Compose

# Setup & Installation
Prerequisites
Docker
Docker Compose

Step 1: Clone Repository
git clone repo url
switch on project directory with the help of cd command

Step 2: Configure Environment
Create .env file:
DATABASE_URL=postgresql://postgres:postgres@db:5432/ragdb

Step 3: Run with Docker
docker-compose up --build


API will be available at:

http://localhost:8000

# API Endpoints
Health Check
GET /

Upload Document
POST /documents/upload

Form Data:
file: PDF or TXT

Semantic Search
POST /search


Body:
{
  "query": "machine learning concepts",
  "top_k": 5
}

Ask Question (RAG)
POST /ask
Body:

{
  "question": "What is deep learning?",
  "top_k": 5
}

# LLM Integration (RAG)
The system uses a Retrieval-Augmented Generation approach:
Retrieve relevant document chunks
Pass context to LLM
Generate final answer
Supported LLM options:
OpenAI GPT Models
Local LLM (Ollama)
HuggingFace Models
LLM integration is implemented in llm.py.

# Database Schema
documents
Field	Type
id	UUID
filename	String
created_at	Timestamp
chunks
Field	Type
id	UUID
document_id	UUID
content	Text
embedding	Vector

# Design Decisions
PostgreSQL with pgvector for vector storage
Modular service-based architecture
Chunk-based indexing for better retrieval
Docker for reproducibility
RAG pattern for QA

# Limitations
Linear vector search 
Basic chunking strategy
No UI frontend
LLM depends on external API