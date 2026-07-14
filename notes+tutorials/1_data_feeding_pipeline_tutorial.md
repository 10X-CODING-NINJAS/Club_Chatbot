# Building the Data Feeding Pipeline

Welcome to the backend of your AI. In this tutorial, you are going to build `1_data_feeding_pipeline.py` from scratch. 

Before your AI can answer questions about the club, it needs to "read" the club data. But you can't just hand an AI a giant text file and expect it to remember everything instantly. We need to break the knowledge down, translate it into math, and store it in a database optimized for AI search.

Your challenge is to implement the functions below. We've provided the starting templates and instructions on what each function needs to achieve.

---

## Step 1: Loading the Info

Before you can process data, you have to gather it. Create a new file called `1_data_feeding_pipeline.py`.

Your first task is to write a function that loads raw text files from your `info/` directory and turns them into LangChain `Document` objects.

```python
import os
from langchain_core.documents import Document

def load_info(info_path: str = "info") -> list[Document]:
    """
    Goal: Load all .txt files from the target directory.
    
    Instructions:
    1. Check if the 'info_path' directory exists. If not, raise a FileNotFoundError.
    2. Create an empty list to hold your documents.
    3. Loop through every file in the directory using os.listdir().
    4. If the file ends with '.txt', open it and read the contents.
    5. Wrap the content in a LangChain Document object:
       doc = Document(page_content=your_text_here, metadata={"source": filepath})
    6. Add the document to your list and return the list.
    """
    pass
```

---

## Step 2: Splitting Documents (Chunking)

You can't process massive files in one go. You have to slice them into smaller pieces called "chunks." When a user asks a question later, the AI will search your database for the *most relevant chunk* to answer it.

```python
from langchain_text_splitters import CharacterTextSplitter

def split_documents(info: list[Document], chunk_size: int = 100, chunk_overlap: int = 0) -> list[Document]:
    """
    Goal: Break large documents into smaller chunks.
    
    Instructions:
    1. Initialize a CharacterTextSplitter with the provided chunk_size and chunk_overlap.
    2. Call the 'split_documents' method on your text splitter, passing in the 'info' list.
    3. Return the resulting list of chunked documents.
    """
    pass
```

---

## Step 3: Embeddings (Translating to Math)

Computers don't understand English; they understand numbers. An embedding model takes a chunk of text and turns it into a massive array of numbers. 

We are going to use **OpenRouter** to get access to a free embedding model. You need to build a custom LangChain class to handle the API calls.

```python
import requests
from langchain_core.embeddings import Embeddings

class OpenRouterEmbeddings(Embeddings):
    """Custom class that calls OpenRouter API directly."""

    def __init__(self, model: str, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """
        Goal: Hit the OpenRouter /embeddings endpoint.
        
        Instructions:
        1. Set up your HTTP headers. You need an 'Authorization' header containing your api_key (Bearer token) and 'Content-Type' set to 'application/json'.
        2. Set up your JSON payload containing the 'model' and the 'input' (which is the 'texts' list).
        3. Use requests.post() to send the request to self.base_url + "/embeddings".
        4. Parse the JSON response. OpenRouter returns a dictionary containing a 'data' array.
        5. Extract the 'embedding' array from each item in the 'data' array.
        6. Return the list of embeddings.
        """
        pass

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # This one is easy: just return self._call_api(texts)
        pass

    def embed_query(self, text: str) -> list[float]:
        # Wrap the single 'text' in a list, call the API, and return the first element.
        pass
```

---

## Step 4: The Vector Database

Now that you have your text chunks translated into number arrays, you need to store them. You are going to use **ChromaDB**, a local, open-source vector database.

```python
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def create_vector_store(chunks: list[Document], persist_directory: str = "db/chroma_db") -> Chroma:
    """
    Goal: Create and persist a ChromaDB vector store.
    
    Instructions:
    1. Instantiate your custom OpenRouterEmbeddings class. Pass it the model "nvidia/llama-nemotron-embed-vl-1b-v2:free" and your API key (use os.getenv("OPENROUTER_API_KEY")).
    2. Call Chroma.from_documents(). You will need to pass in:
       - documents=chunks
       - embedding=your_embedding_model
       - persist_directory=persist_directory
       - collection_metadata={"hnsw:space": "cosine"}
    3. Return the newly created vectorstore.
    """
    pass
```

---

## Step 5: Tying It Together

Finally, create the `main()` function to run your pipeline sequentially.

```python
def main() -> Chroma | None:
    """
    Goal: Run the ingestion pipeline.
    
    Instructions:
    1. Define your info_path ("info") and persistent_directory ("db/chroma_db").
    2. Check if the persistent_directory already exists using os.path.exists(). If it does, you can skip the ingestion to save time and API calls.
    3. If it doesn't exist, call your functions in order:
       - documents = load_info()
       - chunks = split_documents(documents)
       - vectorstore = create_vector_store(chunks)
    4. Return the vectorstore.
    """
    pass

if __name__ == "__main__":
    main()
```

### Next Steps
Once you have filled out the `pass` blocks, run your script! If everything is set up correctly, it will generate your `db/chroma_db` folder. From there, you can move on to creating the Retriever.
