import os
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


class OpenRouterEmbeddings(Embeddings):
    """Custom embeddings class that calls OpenRouter API directly,
    bypassing LangChain's strict OpenAI response parser."""

    def __init__(self, model: str, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Make a direct HTTP request to OpenRouter's /embeddings endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": texts
        }
        response = requests.post(f"{self.base_url}/embeddings", json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        # Extract embeddings from whatever format OpenRouter returns
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return self._call_api(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._call_api([text])[0]

def load_info(info_path: str = "info") -> list[Document]:
    """this loads all the files from the info dir"""
    print(f"Loading documents from {info_path}...")

    if not os.path.exists(info_path):
            raise FileNotFoundError(f"The directory {info_path} does not exist. FAAAAAAAH..for gods sake add the folder")

    info = []

    # Iterate through the directory and read .txt files
    for filename in os.listdir(info_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(info_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Create a LangChain Document manually
                doc = Document(page_content=content, metadata={"source": filepath})
                info.append(doc)

    if len(info) == 0:
            raise FileNotFoundError(f"No .txt files found in {info_path}. nigga are u stupid add the files")

    for i, doc in enumerate(info[:2]):  # Show first 2 documents
            print(f"\nDocument {i+1}:")
            print(f"  Source: {doc.metadata['source']}")
            print(f"  Content length: {len(doc.page_content)} characters")
            print(f"  Content preview: {doc.page_content[:100]}...")
            print(f"  metadata: {doc.metadata}")

    return info


def split_documents(info: list[Document], chunk_size: int = 500, chunk_overlap: int = 0) -> list[Document]:
    """Split documents into smaller chunks with overlap so basically breaking into multiple dimensions and storing them"""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(info)

    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print("Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks

def create_vector_store(chunks: list[Document], persist_directory: str = "db/chroma_db") -> Chroma:
    """Create and persist ChromaDB vector store we are using openai's small embedding model to do it as pratyush (webdev cto) gave the smallest document"""
    print("Creating embeddings and storing in ChromaDB so we can store it cheaply")

    embedding_model = OpenRouterEmbeddings(
        model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    # Create ChromaDB vector store
    print("--- Creating vector store (ie: matrices) ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"} #i have no idea what this does lol use ai
    )
    print("--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main() -> Chroma | None:
    """finally making the pipeline """
    print("=== RAG Document Ingestion Pipeline ===\n")

    # Define paths
    info_path = "info"
    persistent_directory = "db/chroma_db"

    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print("are u stupid ites already done")

        embedding_model = OpenRouterEmbeddings(
            model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
        return vectorstore

    print("Persistent directory does not exist. Initializing vector store...\n")

    # 1. Loading the files
    documents = load_info(info_path=info_path)

    # 2. Chunking the files
    chunks = split_documents(documents)

    # 3. Embedding and Storing in Vector DB
    vectorstore = create_vector_store(chunks, persist_directory=persistent_directory)

    return vectorstore

if __name__ == "__main__":
    main()
