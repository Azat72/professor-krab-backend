from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from qdrant_retriever import search_qdrant
from llm_handler import generate_answer
import os

app = FastAPI(title="KRAB API", version="1.0")

DOCX_FOLDER = "source_docs"  # Папка, где хранятся оригинальные .docx документы

class ChatMessage(BaseModel):
    user: str
    bot: str

class AskRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatMessage]] = []

class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    debug: Optional[List[dict]] = None

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    context_chunks = search_qdrant(request.question, top_k=10)

    print("\n🟢 Qdrant context:")
    for chunk in context_chunks:
        print(f"→ {chunk['title']} | {chunk['text'][:80]}...")

    context_text = "\n\n".join(
        [f"[Источник: {chunk['title']}\nФайл: {chunk['source_file']}\n]\n{chunk['text']}" for chunk in context_chunks if chunk.get('text')]
    )

    print("\n📦 GPT CONTEXT START:\n")
    print(context_text[:1500])
    print("...CONTEXT END\n")

    titles = list({chunk['title'] for chunk in context_chunks if chunk.get('title')})

    answer = generate_answer(
        question=request.question,
        context=context_text,
        chat_history=request.chat_history or []
    )

    return AskResponse(answer=answer, sources=titles, debug=context_chunks)

@app.get("/download")
def download_docx(file: str):
    filepath = os.path.join(DOCX_FOLDER, file)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(path=filepath, filename=file, media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
