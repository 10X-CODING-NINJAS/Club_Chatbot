# Building the Data Retrieval Pipeline

Now that you've successfully fed the club data into ChromaDB, the next step is to retrieve it! 

In this tutorial, you will build `2_data_retrieval_pipleine.py`. The goal of this script is to take a user's question, turn it into an embedding vector, search the vector database for the top 5 most similar chunks of text, and print them out.

Your challenge is to construct the script using the instructions and structure below.

---

## Step 1: Handling the Python Import Trick

Since Python does not allow direct imports from files starting with a number (like `1_data_feeding_pipeline.py`), you need to import the custom embedding class dynamically.

Start your `2_data_retrieval_pipleine.py` file with this setup:

```python
import os
import importlib
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Challenge: Dynamically import your OpenRouterEmbeddings class from the feeding pipeline.
# 1. Use importlib.import_module() to load "1_data_feeding_pipeline".
# 2. Extract the 'OpenRouterEmbeddings' class from that module.
```

---

## Step 2: Initialize the Vector Database connection

To query the database, you must open it with the exact same embedding model used to save it. If the embedding model is different, your search math won't align and you'll get gibberish.

```python
load_dotenv()

persistent_directory = "db/chroma_db"

# Challenge:
# 1. Instantiate the OpenRouterEmbeddings class using the free model:
#    "nvidia/llama-nemotron-embed-vl-1b-v2:free" and your OpenRouter API key from the environment.
# 2. Open the Chroma database using the persist_directory, the embedding_model, 
#    and the collection_metadata {"hnsw:space": "cosine"}.
```

---

## Step 3: Perform the Search and Retrieve

Once the database connection is open, you can configure it as a **retriever** and search for relevant documents.

```python
# Challenge:
# 1. Define a user query (e.g., "what is coding ninjas?")
# 2. Convert the database to a retriever using `db.as_retriever()`. 
#    Pass the search parameter {"k": 5} to tell it to fetch the top 5 closest matching chunks.
# 3. Call `.invoke(query)` on your retriever to fetch the relevant documents.
```

---

## Step 4: Display the Results

Once you get the documents back, you need to loop through them and show the text content to the user.

```python
# Challenge:
# 1. Print the user's original query.
# 2. Loop through the list of retrieved documents and print the content (`doc.page_content`) of each.
```

---

### Next Steps
Run your completed script using `uv run 2_data_retrieval_pipleine.py`! You should see the top 5 chunks of text from `ClubQuestions.txt` that are mathematically closest to your query. Once this works, you've successfully completed the Retrieval part of RAG (Retrieval-Augmented Generation)!
