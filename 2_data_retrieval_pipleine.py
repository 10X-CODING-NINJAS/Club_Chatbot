import os
import importlib
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Python can't import modules starting with a number, so we use importlib i leart this the hard way
# open ai is rly shit use claude even Xai
feeding_pipeline = importlib.import_module("1_data_feeding_pipeline")
OpenRouterEmbeddings = feeding_pipeline.OpenRouterEmbeddings

load_dotenv()

persistent_directory = "db/chroma_db"


# Load embeddings and vector store using our custom OpenRouter class
embedding_model = OpenRouterEmbeddings(
    model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Search for relevant documents
query = "what is coding ninjas?"

retriever = db.as_retriever(search_kwargs={"k": 2}) # change these chunks to get the best response

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
