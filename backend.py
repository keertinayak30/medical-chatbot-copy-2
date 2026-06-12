"""
backend.py — MediBot FastAPI backend

Exposes the LangGraph agent and helper functions as HTTP/JSON endpoints.
The full RAG pipeline (embeddings, Pinecone, LLM, memory, RAG chain) is
initialized exactly once inside the FastAPI lifespan startup event — never
on a per-request basis.

This module is launched as a background daemon thread from streamlit_app.py
via uvicorn.run("backend:app", host="127.0.0.1", port=8000).
It is never run directly by the user.
"""

import os
import tempfile
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel

from src.agent import run_agent
from src.helper import download_hugging_face_embeddings, text_to_speech, transcribe_audio
from src.prompt import system_prompt
from src.tools import init_tools


# ── Pydantic request / response models ───────────────────────────────────────────

class ChatRequest(BaseModel):
    msg: str

class SpeakRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    response: str
    status_updates: List[str]


# ── Module-level state (populated during lifespan startup, never touched after) ──

_memory = None  # ConversationBufferWindowMemory — set once at startup


# ── Lifespan: build the full RAG pipeline before accepting any requests ───────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager. Everything before `yield` runs at startup;
    everything after `yield` runs at shutdown. The `yield` is what lets the app
    start serving requests — so by the time any route is called, initialization
    is guaranteed complete.

    This is the correct replacement for Flask's module-level globals. The RAG
    chain runs once here, and route handlers access it via _memory (module-level).
    """
    global _memory

    load_dotenv()
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    os.environ["GROQ_API_KEY"] = groq_api_key

    print("[MediBot backend] Downloading HuggingFace embeddings…", flush=True)
    embeddings = download_hugging_face_embeddings()

    print("[MediBot backend] Connecting to Pinecone…", flush=True)
    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medchatbot2",
        embedding=embeddings,
    )
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.3,
        streaming=True,
    )

    _memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Inject shared objects into src/tools.py — identical call to the old app.py
    init_tools(rag_chain, llm, _memory)

    print("[MediBot backend] RAG pipeline ready. Accepting requests.", flush=True)
    yield
    # Shutdown: nothing to clean up


app = FastAPI(title="MediBot API", version="2.0", lifespan=lifespan)


# ── Health check ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Streamlit polls this endpoint during startup to know when FastAPI is ready.
    Because lifespan `yield` happens before routes are served, a 200 here means
    the RAG chain is fully initialized.
    """
    return {"status": "ok"}


# ── Routes ────────────────────────────────────────────────────────────────────────

@app.post("/get", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Text-only message → LangGraph agent → grounded answer."""
    if not request.msg.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    result = run_agent(user_message=request.msg, image_path=None)
    return ChatResponse(
        response=result.get("final_answer") or "",
        status_updates=result.get("status_updates") or [],
    )


@app.post("/reset-memory")
def reset_memory():
    """Clear the server-side conversation memory. Called on new Streamlit session."""
    if _memory is not None:
        _memory.chat_memory.clear()
    return {"status": "ok"}


@app.post("/speak")
def speak(request: SpeakRequest, background_tasks: BackgroundTasks):
    """
    Text → gTTS MP3. Returns the file as a streaming response.
    BackgroundTasks deletes the temp file AFTER the bytes are fully sent to the
    client — this is FastAPI's idiomatic replacement for Flask's @after_this_request.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")
    tmp_path = text_to_speech(text)
    if tmp_path is None:
        raise HTTPException(status_code=500, detail="TTS generation failed. Check server logs.")
    background_tasks.add_task(os.remove, tmp_path)
    return FileResponse(tmp_path, media_type="audio/mpeg")


@app.post("/analyze-image", response_model=ChatResponse)
async def analyze_image(
    image: UploadFile = File(...),
    question: str = Form(""),
):
    """
    Multipart upload: image file + optional question text.
    Validates extension and size, writes to a temp file, passes the path to
    run_agent() (which calls vision_tool → analyze_image in src/helper.py).
    The temp file is deleted in the finally block regardless of outcome.
    """
    filename = image.filename or "upload.jpg"
    ext = os.path.splitext(filename)[1].lower()
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Upload JPG, PNG, or WebP.",
        )

    contents = await image.read()
    if len(contents) > 20 * 1024 * 1024:  # Groq vision API enforces 20 MB
        raise HTTPException(status_code=400, detail="Image too large. Maximum size is 20 MB.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        result = run_agent(user_message=question.strip(), image_path=tmp_path)
        if result.get("final_answer") is None:
            raise HTTPException(status_code=500, detail="Agent could not generate a response.")
        return ChatResponse(
            response=result["final_answer"],
            status_updates=result.get("status_updates") or [],
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Multipart upload: audio file (WAV from audio-recorder-streamlit).
    Writes to temp file, calls transcribe_audio() (Whisper via Groq), returns text.
    """
    contents = await audio.read()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        text = transcribe_audio(tmp_path)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
