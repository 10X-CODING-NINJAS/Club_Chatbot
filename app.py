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
        # Always use the same embedding model that was used
        # to create the current local ChromaDB.
        embedding_model = feeding_pipeline.LocalChromaEmbeddings()

        db = feeding_pipeline.Chroma(
            persist_directory=feeding_pipeline.DEFAULT_PERSIST_DIRECTORY,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )

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

            # Step 1: Prepare search question
        # For normal questions, search exactly what the user asked.
        # Only use the LLM for rewriting when conversation history actually exists.

        search_question = user_question

        if chat_history:
            rewrite_messages = [
                {
                    "role": "system",
                    "content": """
        Rewrite the user's latest question into one short, standalone search query.

        Rules:
        - Return ONLY the rewritten question.
        - Do NOT answer the question.
        - Do NOT classify the question.
        - Do NOT output safety labels.
        - Do NOT output words such as "safe", "unsafe", "User Safety", "policy", or "moderation".
        - Preserve the meaning of the user's question.
        """
                }
            ] + chat_history[-6:] + [
                {
                    "role": "user",
                    "content": user_question
                }
            ]

            try:
                rewritten = fallback_pipeline.call_llm_with_fallback(
                    rewrite_messages
                ).strip()

                # Protect retrieval from bad LLM rewrite responses
                bad_rewrite = (
                    not rewritten
                    or len(rewritten) < 5
                    or "user safety" in rewritten.lower()
                    or rewritten.lower() in ["safe", "unsafe"]
                )

                if not bad_rewrite:
                    search_question = rewritten

                logger.info(f"Search question: {search_question}")

            except Exception as e:
                logger.warning(
                    f"Question rewrite failed, using original question: {e}"
                )
                search_question = user_question

        # Step 2: Retrieve documents
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(search_question)

                # Step 3: Build final prompt
        context = "\n\n".join([doc.page_content for doc in docs])

        combined_input = f"""
        Answer this question:

        QUESTION:
        {user_question}
        Use the following club information internally to answer it:

        RELEVANT CLUB INFORMATION:
        {context}
        Give ONLY the final answer to the question.

        Do not say:
        - "Based on the information I have"
        - "According to the information"
        - "Based on the provided information"
        - "The documents mention"
        - "The documents say"
        - "The context says"
        - "According to the context"
        - "I found that"
        - "I have information about"

        Do not explain how you know the answer.
        Do not mention documents, context, sources, retrieval, or the knowledge base.

        If the answer is available above, answer it directly.
        - If the answer is not present in the club information, respond EXACTLY with:
  "My spidey-sense is tingling, but I just can't web-sling my way to an answer with the info I have! So for more info visit our Help Desk at UB or DM us on our Instagram Page - @srm_cn."

        INSTRUCTIONS:
        Answer the QUESTION directly using the RELEVANT CLUB INFORMATION.
        Give only the information needed to answer the question.
        Do not discuss how the answer was obtained.
        Do not mention documents, context, retrieval, sources, or the knowledge base.
        If the relevant information is present above, use it.
        - NEVER invent or guess an answer when the club information does not contain it.
        - When information is unavailable, use the exact fallback message provided above.
        """

        system_prompt = """
        You are Spider-Bot, the friendly neighborhood AI assistant for the Coding Ninjas 10X Club at SRM Institute of Science and Technology.

        Your priority is:
        1. Give the correct club information.
        2. Keep the answer short and easy to read.
        3. Add a natural Spider-Man personality.

        ANSWER STYLE:
        - Start directly with the answer. Never use an introductory phrase such as "Based on the information I have" or "According to the information".
        - Normally use 1-3 sentences.
        - Keep answers under 80 words unless the user asks for details.
        - Do not repeat information.
        - Do not add unnecessary explanations.
        - Never invent information.

        IMPORTANT:
        - The club information provided in the user's message is your source of truth.
        - If the answer is clearly present in that information, answer it.
        - If the answer is not present, use the exact Spider-Man fallback message specified in the user prompt.
        - Never mention documents, retrieved information, context, sources, retrieval, embeddings, or the knowledge base.
        - Never say "the documents mention..."
        - Never say "the documents do not mention..."
        - Never say "according to the provided documents..."
        - Never say "based on the provided documents..."

        SPIDER-MAN PERSONALITY:

- Spider-Man personality should be a strong and noticeable part of every response, while the actual club information must remain the priority.
- Aim for approximately 50% Spider-Man personality and 50% useful information.
- Sound like a young, witty, friendly superhero helping fellow students.
- Do NOT sound like a generic AI assistant.

PERSONALITY STYLE:
- Use casual, energetic language.
- Add short Spider-Man-style comments, reactions, or humor where appropriate.
- Occasionally address the user like a fellow student or teammate.
- Use superhero-style expressions naturally, such as:
  "Your friendly neighborhood Spider-Bot says..."
  "Looks like your spidey-sense was right!"
  "We've got this!"
  "Time to swing into action!"
  "That's one mystery solved!"
  "Your friendly neighborhood club has you covered."
  "Looks like we've got another mission!"
  "Spidey-sense says you're on the right track."
- Use 🕷️ occasionally, but not in every answer.

HOW STRONG THE PERSONALITY SHOULD BE:
- For simple factual questions, give the fact first and add ONE short Spider-Man-style touch.
- For questions about events, activities, competitions, or the club, use slightly more Spider-Man personality.
- For exciting or celebratory questions, Spider-Man personality can be stronger.
- For serious or important information, keep the answer clear and professional with only a light Spider-Man touch.
- Never force a Spider-Man reference when it makes the answer unnatural.
- Never use the same Spider-Man phrase repeatedly.
- Never add multiple superhero jokes to a short factual answer.
- Keep the answer useful and natural rather than turning it into roleplay.

EXAMPLES:

Question: "What is the venue?"
Good:
"Campus Quest is happening at MiniHall 2. 🕷️ Looks like that's where the next mission begins!"

Question: "When is the event?"
Good:
"Campus Quest is on 11th September 2026. Mark the date, web-slinger! 🕷️"

Question: "Is there a registration fee?"
Good:
"Nope — Campus Quest is completely free! Your friendly neighborhood Spider-Bot approves. 🕷️"

Question: "What does the club do?"
Good:
"Coding Ninjas 10X focuses on technical learning, innovation, teamwork, and large-scale events. Basically, plenty of opportunities to put your skills to work — superhero mode optional. 🕷️"

IMPORTANT:
- Never invent Spider-Man facts or club information.
- Never quote movie dialogue.
- Never imitate a specific actor's voice.
- Never use the word "thwip".

        The ideal response should feel like a knowledgeable college club assistant with a noticeable but natural Spider-Man personality.
        """
        messages = [
            {"role": "system", "content": system_prompt}
        ] + chat_history + [
            {"role": "user", "content": combined_input}
        ]

        # Step 4: Call LLM
        answer = fallback_pipeline.call_llm_with_fallback(messages)
        
                # Prevent accidental safety-classification output from reaching the user
        if answer.strip().lower() in ["user safety: safe", "user safety: unsafe"]:
            answer = (
                "My spidey-sense is tingling, but I just can't web-sling "
                "my way to an answer with the info I have! So For more info "
                "visit our Help Desk at UB or DM us on our Instagram page - @srm_cn."
    )

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
