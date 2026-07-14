import os
import importlib
import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma

# Import the custom embeddings class from our feeding pipeline
feeding_pipeline = importlib.import_module("1_data_feeding_pipeline")
OpenRouterEmbeddings = feeding_pipeline.OpenRouterEmbeddings

# Load environment variables
load_dotenv()

# Connect to your document database
persistent_directory = "db/chroma_db"
embeddings = OpenRouterEmbeddings(
    model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Store our conversation as messages
chat_history = []


def call_llm(messages: list[dict]) -> str:
    """Send messages to the LLM via OpenRouter and return the response."""
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openrouter/free",
        "messages": messages
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=payload,
        headers=headers
    )
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


def ask_question(user_question: str) -> str:
    """Process a user question with context history and retrieved documents."""
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to rewrite the question as standalone so it's searchable
        messages = [
            {"role": "system", "content": "Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."}
        ] + chat_history + [
            {"role": "user", "content": f"New question: {user_question}"}
        ]

        search_question = call_llm(messages).strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")

    # Step 3: Create final prompt with context
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    # Step 4: Get the answer (include chat history for context)
    messages = [
        {"role": "system", "content": "You are a helpful assistant for the Coding Ninjas 10X Club. Answer questions based on provided documents and conversation history."}
    ] + chat_history + [
        {"role": "user", "content": combined_input}
    ]

    answer = call_llm(messages)

    # Step 5: Remember this conversation
    chat_history.append({"role": "user", "content": user_question})
    chat_history.append({"role": "assistant", "content": answer})

    print(f"Answer: {answer}")
    return answer


# Simple chat loop
def start_chat() -> None:
    """Interactive chat loop."""
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == 'quit':
            print("Goodbye!")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()
