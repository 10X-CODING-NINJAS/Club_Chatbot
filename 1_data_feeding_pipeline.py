"""
RAG Document Ingestion Pipeline
================================

Loads text documents from a directory, splits them into smaller
chunks, generates embeddings using the OpenRouter API, and stores the
resulting vectors in a persistent ChromaDB database.

The resulting vector store can later be queried to retrieve relevant
context for a Retrieval-Augmented Generation (RAG) chatbot.

Usage:
    Place your `.txt` source files in an `info/` folder next to this
    script, set the OPENROUTER_API_KEY environment variable (e.g. in
    a `.env` file), then run:

        python rag_ingest.py

    The first run builds the vector store in `db/chroma_db/`.
    Subsequent runs simply load the existing store instead of
    rebuilding it.
"""

import os
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

# Embedding model used for turning text into vectors.
EMBEDDING_MODEL_NAME = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

# Where source .txt files live, and where the vector database is stored.
DEFAULT_INFO_PATH = "info"
DEFAULT_PERSIST_DIRECTORY = "db/chroma_db"


class OpenRouterEmbeddings(Embeddings):
    """
    Embedding model that calls the OpenRouter API.

    Implements LangChain's `Embeddings` interface so it can be used
    as a drop-in embedding function for LangChain vector stores
    (e.g. Chroma).
    """

    def __init__(self, model: str, api_key: str,
                 base_url: str = "https://openrouter.ai/api/v1"):
        """
        Args:
            model: OpenRouter embedding model identifier.
            api_key: OpenRouter API key.
            base_url: Base URL for the OpenRouter API.
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """
        Send a batch of texts to the OpenRouter embedding endpoint.

        Args:
            texts: List of input strings to embed.

        Returns:
            List of embedding vectors, one per input string.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": texts}

        response = requests.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of documents."""
        return self._call_api(texts)

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query string."""
        return self._call_api([text])[0]


def get_embedding_model() -> OpenRouterEmbeddings:
    """
    Build the shared OpenRouter embedding model instance.

    Returns:
        Configured OpenRouterEmbeddings instance.

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )
    return OpenRouterEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=api_key)


def load_info(info_path: str = DEFAULT_INFO_PATH) -> list[Document]:
    """
    Load all .txt documents from the given directory.

    Args:
        info_path: Directory containing .txt files.

    Returns:
        List of LangChain Document objects, one per file.

    Raises:
        FileNotFoundError: If the directory is missing or contains
            no .txt files.
    """
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Directory '{info_path}' does not exist.")

    documents = []
    for filename in os.listdir(info_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(info_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                documents.append(
                    Document(
                        page_content=f.read(),
                        metadata={"source": filepath},
                    )
                )

    if not documents:
        raise FileNotFoundError(f"No .txt files found in '{info_path}'.")

    return documents


def split_documents(
    info: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 0,
) -> list[Document]:
    """
    Split documents into smaller chunks for embedding.

    Smaller chunks improve retrieval accuracy in RAG systems, since
    the chatbot can pull in just the relevant snippet instead of an
    entire document.

    Args:
        info: Documents to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between
            consecutive chunks (helps preserve context across chunk
            boundaries).

    Returns:
        List of chunked Document objects.
    """
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(info)


def create_vector_store(
    chunks: list[Document],
    embedding_model: OpenRouterEmbeddings,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> Chroma:
    """
    Embed document chunks and store them in a persistent ChromaDB store.

    Args:
        chunks: Document chunks to embed and store.
        embedding_model: Embedding model to use.
        persist_directory: Directory where the vector store is saved.

    Returns:
        The created Chroma vector store.
    """
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},  # cosine similarity
    )


def main() -> Chroma:
    """
    Run the RAG ingestion pipeline.

    If a vector store already exists at DEFAULT_PERSIST_DIRECTORY, it
    is loaded directly. Otherwise, documents are loaded, split into
    chunks, embedded, and saved as a new vector store.

    Returns:
        The loaded or newly created Chroma vector store.
    """
    embedding_model = get_embedding_model()

    if os.path.exists(DEFAULT_PERSIST_DIRECTORY):
        return Chroma(
            persist_directory=DEFAULT_PERSIST_DIRECTORY,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )

    documents = load_info(DEFAULT_INFO_PATH)
    chunks = split_documents(documents)
    return create_vector_store(chunks, embedding_model, DEFAULT_PERSIST_DIRECTORY)


if __name__ == "__main__":
    main()