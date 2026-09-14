import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", None)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

if not QDRANT_URL:
    QDRANT_URL = "db/qdrant_db"

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "cn10x_club_knowledge")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

class ClubRetriever:
    """
    Handles retrieval of Club FAQ information using Qdrant and fastembed embeddings.
    """
    def __init__(self):
        print(f"Initializing {EMBEDDING_MODEL} embedding model...")
        self.embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
        
        if QDRANT_URL.startswith("http"):
            print(f"Connecting to Qdrant Cloud at {QDRANT_URL}...")
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            print(f"Connecting to Local Qdrant at {QDRANT_URL}...")
            self.client = QdrantClient(path=QDRANT_URL)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Embeds the query and retrieves the top_k matching chunks from Qdrant.
        """
        if not self.client.collection_exists(COLLECTION_NAME):
            print(f"Collection {COLLECTION_NAME} does not exist.")
            return []
            
        # Embed the query
        query_embedding = list(self.embedding_model.embed([query]))[0].tolist()
        
        search_result = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k
        )
        
        results = []
        for hit in search_result.points:
            payload = hit.payload
            results.append({
                "content": payload.get("text", ""),
                "score": hit.score,
                "metadata": {
                    "source": payload.get("source"),
                    "section": payload.get("section"),
                    "question_number": payload.get("question_number")
                }
            })
            
        return results

# Singleton instance for the app
retriever_instance = None

def get_retriever() -> ClubRetriever:
    global retriever_instance
    if retriever_instance is None:
        try:
            retriever_instance = ClubRetriever()
        except Exception as e:
            print(f"Failed to initialize ClubRetriever: {e}")
            retriever_instance = None
    return retriever_instance
