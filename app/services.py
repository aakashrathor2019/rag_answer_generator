import fitz
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import re
from app.llm import generate_answer



_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(_EMBED_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)



def synthesize_answer(question: str, contexts: list) -> str:

    if not contexts:
        return "No relevant context found in this document."

    combined_context = "\n\n".join(contexts[:5])

    answer = generate_answer(
        context=combined_context,
        question=question
    )

    return answer

def get_pdf_text(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text") + " "
    return text.strip()

def create_chunks(text: str, chunk_size_tokens: int = 600, overlap_percent: float = 0.15):
    if not text:
        return []
    
    overlap_tokens = max(1, int(chunk_size_tokens * overlap_percent))
    step = chunk_size_tokens - overlap_tokens
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    chunks = []
    for start in range(0, len(token_ids), step):
        end = start + chunk_size_tokens
        slice_ids = token_ids[start:end]
        if not slice_ids:
            break
        chunk_text = tokenizer.decode(slice_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
        if end >= len(token_ids):
            break
    return chunks

def embed_text(text_or_list):
    single = isinstance(text_or_list, str)
    if single:
        text_or_list = [text_or_list]
    
    embeddings = model.encode(text_or_list)
    return embeddings[0].tolist() if single else [e.tolist() for e in embeddings]