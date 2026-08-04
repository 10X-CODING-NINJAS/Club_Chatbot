"""
Club Chatbot API
=================
FastAPI server exposing the RAG chatbot as a REST API.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import importlib
import logging
import uuid
import os
import shutil
from contextlib import asynccontextmanager
from typing import Optional, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("club-chatbot")

# ---------------------------------------------------------------------------
# Dynamic imports (filenames start with numbers)
# ---------------------------------------------------------------------------
feeding_pipeline = importlib.import_module("1_data_feeding_pipeline")
fallback_pipeline = importlib.import_module("5_fallback_with_ollama")

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
sessions_history: Dict[str, List[Dict[str, str]]] = {}
db = None


def init_db():
    """Load or create the ChromaDB vector store."""
    global db
    logger.info("Initializing vector store...")
    try:
        db = feeding_pipeline.main()
        logger.info("✅ Vector store initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    yield
    logger.info("Shutting down Club Chatbot API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Club Chatbot API",
    description="RAG Chatbot API for the Coding Ninjas 10X Club — Spider-Bot 🕷️",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Web team can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

class ResetRequest(BaseModel):
    session_id: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        if not request.session_id:
            session_id = str(uuid.uuid4())
        else:
            session_id = request.session_id

        if session_id not in sessions_history:
            sessions_history[session_id] = []

        chat_history = sessions_history[session_id]
        user_question = request.message

        if not db:
            raise HTTPException(status_code=500, detail="Vector store is not initialized")

        # Step 1: Rewrite question if history exists
        if chat_history:
            messages = [
                {"role": "system", "content": "Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."}
            ] + chat_history + [
                {"role": "user", "content": f"New question: {user_question}"}
            ]
            search_question = fallback_pipeline.call_llm_with_fallback(messages).strip()
            logger.info(f"Rewrote question to: {search_question}")
        else:
            search_question = user_question

        # Step 2: Retrieve documents
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(search_question)

        # Step 3: Build final prompt
        combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

        system_prompt = (
            "You are 'Spider-Bot', the friendly neighborhood assistant for the Coding Ninjas 10X Club."
            "You speak enthusiastically, just like Peter Parker / Spider-Man. Use mild Spider-Man slang "
            "(like 'web-slinging', 'spidey-sense', 'thwip', or 'with great power comes great code'). "
            "CRITICAL RULE: You must ONLY answer questions using the provided documents and conversation history. "
            "If the context doesn't have the answer, just say your spidey-sense is tingling but you don't have enough info in your web-shooters to answer that right now."
        )

        messages = [
            {"role": "system", "content": system_prompt}
        ] + chat_history + [
            {"role": "user", "content": combined_input}
        ]

        # Step 4: Call LLM
        answer = fallback_pipeline.call_llm_with_fallback(messages)

        # Step 5: Update history
        chat_history.append({"role": "user", "content": user_question})
        chat_history.append({"role": "assistant", "content": answer})

        return ChatResponse(response=answer, session_id=session_id)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/reset")
def reset_chat(request: ResetRequest):
    try:
        if request.session_id in sessions_history:
            sessions_history[request.session_id] = []
        return {"message": "Session reset", "session_id": request.session_id}
    except Exception as e:
        logger.error(f"Error resetting chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "vector_store": db is not None
    }

@app.post("/ingest")
def ingest_data():
    global db
    try:
        embedding_model = feeding_pipeline.get_embedding_model()
        persist_dir = feeding_pipeline.DEFAULT_PERSIST_DIRECTORY
        
        # To truly rebuild, we clear the old DB directory
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
            
        documents = feeding_pipeline.load_info(feeding_pipeline.DEFAULT_INFO_PATH)
        chunks = feeding_pipeline.split_documents(documents)
        db = feeding_pipeline.create_vector_store(chunks, embedding_model, persist_dir)
        
        return {"message": "Ingestion complete", "chunks": len(chunks)}
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
