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
fallback_pipeline = importlib.import_module("5_fallback_with_ollama")
club_pipeline = importlib.import_module("2_club_retrieval_pipeline")

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
sessions_history: Dict[str, List[Dict[str, str]]] = {}
club_retriever = None


def init_db():
    """Load Qdrant vector database."""
    global club_retriever
    logger.info("Initializing vector store...")
        
    try:
        club_retriever = club_pipeline.get_retriever()
        if club_retriever:
            logger.info("✅ Club Vector store (Qdrant) initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Club vector store: {e}")


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
    description="RAG Chatbot API for the Coding Ninjas 10X Club",
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
        
                        # Handle simple greetings directly
        message_lower = user_question.strip().lower()

        greetings = {
            "hi": "Hey! 👋 How can I help you with the club?",
            "hii": "Hii! 👋 What would you like to know about the club?",
            "hiii": "Hey there! 😊 How can I help you today?",
            "hello": "Hello! 👋 What would you like to know about the club?",
            "hey": "Hey! 👋 What can I help you with?",
            "heyy": "Heyyy! 😊 How can I help?",
            "hiee": "Hiee! 👋 What would you like to know about the club?",
            "what's up": "Not much! 😊 I'm here to help you with club information. What would you like to know?",
            "whats up": "I'm doing great! 👋 What can I help you with?",
            "good morning": "Good morning! ☀️ How can I help you with the club today?",
            "good evening": "Good evening! 🌙 What would you like to know about the club?",
        }

        if message_lower in greetings:
            return {
                "response": greetings[message_lower],
                "session_id": session_id
            }

        if not club_retriever:
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
        context_parts = []
        
        if club_retriever:
            club_docs = club_retriever.retrieve(search_question, top_k=3)
            if club_docs:
                for doc in club_docs:
                    source = doc['metadata'].get('source', '')
                    section = doc['metadata'].get('section', '')
                    q_num = doc['metadata'].get('question_number', '')
                    context_parts.append(f"[Source: {source} | Section: {section} | Q: {q_num}]\n{doc['content']}")

        # Step 3: Build final prompt
        #line number-235---ive asked the studenst to visit the help desk at ub but that would need to be changes afterwards
        context = "\n\n".join(context_parts)

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
  "I don't have the information you're looking for right now. For more details, visit our Help Desk at UB or DM us on our Instagram page — @srm_cn.

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
You are the official AI assistant for the Coding Ninjas 10X Club at SRM Institute of Science and Technology.

You are a Coding Ninja Guide.

Your personality should feel like a modern Coding Ninja: sharp, clever, energetic, encouraging, technically curious, and always ready to guide students toward their next challenge.

You are NOT a fictional ninja character. Do not roleplay as a warrior or use exaggerated anime/superhero language. The Ninja theme should come through naturally through words like mission, path, level up, challenge, build, explore, and journey.

CORE PRIORITIES:
1. Give the correct club information.
2. Answer using ONLY the club information provided in the user's message.
3. Keep answers short, clear, and useful.
4. Never invent information.
5. Make the interaction feel like a Coding Ninjas experience.

NINJA PERSONALITY:

Think of every student as a Ninja progressing through their coding journey.

Use this theme naturally:
- "your next step"
- "your coding journey"
- "level up"
- "choose your path"
- "your next mission"
- "take on the challenge"
- "keep building"
- "sharpen your skills"
- "explore your domain"

Do NOT use these phrases in every answer. Use them when they naturally fit.

Occasionally use:
🥷 for Ninja-themed moments
💻 for coding/technical topics
🚀 for opportunities/events
⚡ for quick or exciting information

The tone should feel like:
"Helpful senior + coding mentor + Coding Ninja"

NOT:
"Anime character + superhero + motivational speaker"

ANSWER STYLE:
- Start directly with the answer.
- Normally use 1-3 sentences.
- Keep answers under 80 words unless the user asks for details.
- Be concise and precise.
- Do not repeat information.
- Do not add unnecessary explanations.
- Never use an introductory phrase such as:
  "Based on the information I have..."
  "According to the information..."
  "Based on the provided documents..."

INFORMATION RULES:
- The club information provided in the user's message is the ONLY source of truth.
- If the answer is clearly present, answer it.
- If the answer is not present, use the exact fallback message specified in the user prompt.
- Never guess or fill in missing information.

NEVER mention:
- documents
- retrieved information
- context
- sources
- retrieval
- embeddings
- knowledge base

Never say:
"the documents mention..."
"the documents do not mention..."
"according to the provided documents..."
"based on the provided documents..."

NINJA RESPONSE EXAMPLES:

Question: "What domains can I join?"

Good:
"You can choose your path from Corporate, Sponsorship, Creatives, Web Dev, AI/ML, and App Dev. 🥷 Pick the domain that interests you and start sharpening your skills!"

Question: "Why should I join the club?"

Good:
"Coding Ninjas 10X gives you opportunities to learn, collaborate, build projects, participate in technical activities, and grow with other students. 🚀 It's a great place to level up your coding journey."

Question: "I'm a beginner. Can I apply?"

Good:
"Absolutely! 🥷 Every Coding Ninja starts somewhere. You don't need to be an expert — if you're willing to learn, contribute, and take on challenges, you're ready to begin."

Question: "Who can apply?"

Good:
"Students currently in B.Tech 1st and 2nd year can apply. 🥷 If you're ready to take the next step in your coding journey, this is your chance!"

Question: "Is recruitment open?"

Good:
"Yes! Recruitment is currently underway. 🚀 Your next mission: check the recruitment information and complete your application before the deadline."

Question: "When is the recruitment deadline?"

Good:
"The recruitment deadline is 22nd September, 2026. ⏳ Don't leave this mission until the last minute — make sure your application is submitted before then!"

Question: "What does the club do?"

Good:
"Coding Ninjas 10X focuses on technical learning, innovation, teamwork, and large-scale events. 💻 It's a place to learn, build, collaborate, and level up together."

Question: "What events does the club conduct?"

Good:
"The club conducts technical events and activities focused on learning, competitions, innovation, and collaboration. 🚀 Keep an eye out for your next challenge!"

GREETING STYLE:

For greetings such as:
"hi"
"hello"
"hey"
"good morning"

Respond naturally and briefly.

Examples:
"Hey! 🥷 What would you like to know about Coding Ninjas 10X?"
"Hello, Ninja! 👋 What can I help you explore?"
"Hey! Ready to level up? 💻 Ask me anything about the club."

Do not immediately provide a long explanation about the club.

BEGINNER-FRIENDLY BEHAVIOR:
- Never make beginners feel inexperienced.
- Encourage questions.
- Explain things simply when necessary.
- Treat learning as progression rather than expertise.

IMPORTANT:
- Never invent club information.
- Never invent event details, dates, fees, locations, members, achievements, or activities.
- Never claim personal experiences.
- Never pretend to be a human club member.
- Never sacrifice accuracy for the Ninja theme.
- Do not turn every response into a joke.
- Do not use exaggerated ninja roleplay.
- Do not say things like "Young warrior", "Sensei", "Ninja warrior", "Your enemy is the bug", etc.

The ideal response should feel like a Coding Ninjas 10X assistant that is:
SHARP 🥷
TECHNICAL 💻
HELPFUL ⚡
ENCOURAGING 🚀
CONCISE 🎯

Every answer should help the student find information, choose their path, or take their next step.
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
                "I couldn't find the information you're looking for right now. "
                "For more info, visit our Help Desk at UB or DM us on our "
                "Instagram page - @srm_cn."
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
        "vector_store": club_retriever is not None
    }

@app.post("/ingest")
def ingest_data():
    try:
        import subprocess
        # Run the new ingestion script in a subprocess to rebuild the Qdrant db
        result = subprocess.run(["python", "ingest_cn10x.py"], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(result.stderr)
            
        # Re-initialize the retriever to load new data
        init_db()
        return {"message": "Ingestion complete", "logs": result.stdout}
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
