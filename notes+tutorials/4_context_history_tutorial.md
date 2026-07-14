# Building the Context History Pipeline

In the previous tutorials you built retrieval — the AI fetches relevant chunks and generates answers. But there's a problem: **it has no memory**. If you ask "what is coding ninjas?" and then follow up with "who leads it?", the AI has no idea what "it" refers to.

In this tutorial, you will build `4_context_history.py` which adds conversation memory to your RAG chatbot.

---

## Step 1: Setup and Import

You need to connect to the same ChromaDB database using the same embedding model from your feeding pipeline.

```python
import os
import importlib
import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma

# Challenge:
# 1. Use importlib.import_module() to import "1_data_feeding_pipeline".
# 2. Extract the OpenRouterEmbeddings class from it.
# 3. Call load_dotenv().
# 4. Create the embedding model instance and open the Chroma database (same as file 2).
# 5. Create an empty list called 'chat_history' — this will store the conversation.
```

---

## Step 2: Build the LLM Caller

You need a reusable function that sends a list of messages to OpenRouter and returns the AI's response.

```python
def call_llm(messages: list[dict]) -> str:
    """
    Goal: Send a list of chat messages to OpenRouter and return the response text.

    Instructions:
    1. Set up your HTTP headers with Authorization (Bearer token) and Content-Type.
    2. Build the JSON payload with:
       - "model": "openrouter/free" (auto-routes to a free model)
       - "messages": the messages list passed into this function
    3. POST to "https://openrouter.ai/api/v1/chat/completions".
    4. Parse the JSON response and return data["choices"][0]["message"]["content"].
    """
    pass
```

---

## Step 3: The Question Rewriter

This is the key part that makes context history work. When the user asks a vague follow-up like "who leads it?", you need to rewrite that into a standalone question like "Who leads the Coding Ninjas 10X Club?" before searching the database.

```python
def ask_question(user_question: str) -> str:
    """
    Goal: Process a question with conversation history, retrieve docs, and generate an answer.

    Instructions:

    STEP 1 — Rewrite the question (only if there's chat history):
    1. Check if chat_history is not empty.
    2. If it has history, build a messages list:
       - A system message telling the AI: "Given the chat history, rewrite the new question 
         to be standalone and searchable. Just return the rewritten question."
       - Append the full chat_history list.
       - Append a user message: "New question: {user_question}"
    3. Call call_llm() with that messages list. The response is your rewritten search_question.
    4. If there's no history, just use user_question directly as search_question.

    STEP 2 — Retrieve relevant documents:
    1. Create a retriever from the database with search_kwargs={"k": 3}.
    2. Call retriever.invoke(search_question) to get the relevant docs.

    STEP 3 — Generate the answer:
    1. Combine all retrieved doc contents into a single prompt string.
    2. Build a messages list:
       - A system message setting the AI's role.
       - Append the full chat_history for conversational context.
       - A user message containing the documents + the original user_question.
    3. Call call_llm() to get the answer.

    STEP 4 — Save to history:
    1. Append {"role": "user", "content": user_question} to chat_history.
    2. Append {"role": "assistant", "content": answer} to chat_history.
    3. Return the answer.
    """
    pass
```

---

## Step 4: The Chat Loop

Finally, create an interactive loop so the user can keep asking questions.

```python
def start_chat() -> None:
    """
    Goal: Run an interactive chat loop.

    Instructions:
    1. Print a welcome message.
    2. Start a while True loop.
    3. Take user input with input().
    4. If the user types 'quit', print goodbye and break.
    5. Otherwise, call ask_question() with the input.
    """
    pass

if __name__ == "__main__":
    start_chat()
```

---

### How Context History Works

Here is what happens when a user has a multi-turn conversation:

```
Turn 1: "What is Coding Ninjas?"
  → Search query: "What is Coding Ninjas?"  (no history, used as-is)
  → AI answers based on retrieved docs.

Turn 2: "What events do they organize?"
  → AI rewrites to: "What events does the Coding Ninjas 10X Club organize?"
  → This standalone question is now searchable in the vector database.
  → AI answers based on newly retrieved docs + remembers Turn 1.
```

Without this rewriting step, the database would try to search for "What events do they organize?" — and since it doesn't know who "they" is, it would return garbage results.

### Next Steps
Once you fill in the `pass` blocks, run `uv run 4_context_history.py` and have a multi-turn conversation to test that the history rewriting works correctly.
