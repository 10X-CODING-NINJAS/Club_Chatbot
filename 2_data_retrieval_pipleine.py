"""Retrieve relevant documents from the existing ChromaDB database.

This script connects to the vector database created by
`1_data_feeding_pipeline.py`, performs a similarity search,
and prints the retrieved documents for a sample query.
"""

import importlib
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma

# Python module names cannot start with a number, so import the ingestion module
# dynamically to reuse its custom embedding class.
feeding_pipeline = importlib.import_module("1_data_feeding_pipeline")
OpenRouterEmbeddings = feeding_pipeline.OpenRouterEmbeddings

# Settings that are safe to adjust while experimenting with retrieval quality.
PERSISTENT_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL_NAME = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
QUERY = "what is coding ninjas?"
RETRIEVAL_K = 2


def create_vector_store() -> Chroma:
    """Connect to the persisted Chroma database using the configured embeddings."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set. Add it to your .env file.")

    embedding_model = OpenRouterEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        api_key=api_key,
    )
    return Chroma(
        persist_directory=PERSISTENT_DIRECTORY,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )


def retrieve_documents(vector_store: Chroma, query: str, k: int):
    """Return the k documents most relevant to query."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def display_documents(query: str, documents) -> None:
    """Print a query and its retrieved document contents in a readable format."""
    print(f"User Query: {query}")
    print("--- Context ---")

    if not documents:
        print("No relevant documents were found.")
        return

    for index, document in enumerate(documents, start=1):
        print(f"Document {index}:\n{document.page_content}\n")


def main() -> None:
    """Load configuration, retrieve documents for the sample query, and print them."""
    load_dotenv()
    vector_store = create_vector_store()
    documents = retrieve_documents(vector_store, QUERY, RETRIEVAL_K)
    display_documents(QUERY, documents)


if __name__ == "__main__":
    main()
