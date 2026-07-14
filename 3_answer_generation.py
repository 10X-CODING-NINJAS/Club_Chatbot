# THIS FILE IS JUST TO KNOW THE BASIC INDERSTANDING WE ARE NOT USING IT IG
import os
import importlib
import requests
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Import the custom embeddings class from our feeding pipeline
feeding_pipeline = importlib.import_module("1_data_feeding_pipeline")
OpenRouterEmbeddings = feeding_pipeline.OpenRouterEmbeddings

load_dotenv()


def get_retriever() -> object:
    """Load the vector store and return a retriever."""
    persistent_directory = "db/chroma_db"

    embedding_model = OpenRouterEmbeddings(
        model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    return db.as_retriever(search_kwargs={"k": 5})


def generate_answer(query: str, context: str) -> str:
    """Send the query + retrieved context to an LLM via OpenRouter and return the answer."""
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openrouter/free",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for the Coding Ninjas 10X Club. "
                    "Answer the user's question using ONLY the provided context. "
                    "If the context does not contain the answer, say you don't have that information. "
                    "Keep your answers concise and friendly."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=payload,
        headers=headers
    )
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


def main() -> None:
    """Retrieve relevant docs and generate an AI answer."""
    query = input("Ask a question about the club: ")

    # 1. Retrieve relevant chunks
    retriever = get_retriever()
    relevant_docs = retriever.invoke(query)

    # 2. Combine chunks into a single context string
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    # 3. Generate an answer from the LLM using only the retrieved context
    print("\n--- Answer ---")
    answer = generate_answer(query, context)
    print(answer)


if __name__ == "__main__":
    main()
