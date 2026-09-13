import os
import re
import uuid
import PyPDF2
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", None)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# If no URL is provided, fallback to local path for development only
if not QDRANT_URL:
    QDRANT_URL = "db/qdrant_db" # Local SQLite mode

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "cn10x_club_knowledge")
EMBEDDING_MODEL = "BAAI/bge-m3"

def parse_pdf(file_path):
    print(f"Reading PDF from {file_path}...")
    reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Sections define logical separation of the FAQ
    sections = ["ABOUT THE CLUB", "RECRUITMENT", "CAMPUS QUEST 5.0"]
    
    current_section = "GENERAL"
    
    # Find all question blocks starting with number)
    parts = re.split(r'\b(\d+)\)\s+', text)
    
    # parts[0] is the intro text
    intro_text = parts[0]
    for s in sections:
        if s in intro_text:
            current_section = s
            
    qa_pairs = []
    
    for i in range(1, len(parts), 2):
        q_num = parts[i]
        block = parts[i+1]
        
        # Check if the block contains a section title transition for the NEXT questions
        for s in sections:
            if s in block:
                # The section title usually appears at the end of the answer text
                block_parts = block.split(s)
                block = block_parts[0].strip()
                next_section = s
                break
        else:
            next_section = current_section
            
        # Try to separate Question and Answer. Question ends with '?'
        q_split = block.split('?', 1)
        if len(q_split) == 2:
            question = q_split[0].strip() + '?'
            answer = q_split[1].strip()
        else:
            # Fallback if no question mark
            question = block[:100]
            answer = block
            
        qa_pairs.append({
            "question_number": int(q_num),
            "question": question,
            "answer": answer,
            "section": current_section,
            "source": os.path.basename(file_path)
        })
        
        current_section = next_section

    return qa_pairs

def main():
    pdf_path = "CN10X QUESTIONS-1.pdf"
    if not os.path.exists(pdf_path):
        pdf_path = os.path.join(os.path.dirname(__file__), "..", "CN10X QUESTIONS-1.pdf")
        if not os.path.exists(pdf_path):
            print("Cannot find CN10X QUESTIONS-1.pdf")
            return
            
    qa_pairs = parse_pdf(pdf_path)
    print(f"Extracted {len(qa_pairs)} QA pairs.")
    
    print("Initializing embedding model (BAAI/bge-m3)...")
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    
    # Initialize Qdrant Client
    if QDRANT_URL.startswith("http"):
        print(f"Connecting to Qdrant Cloud at {QDRANT_URL}...")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        print(f"Initializing Local Qdrant at {QDRANT_URL}...")
        os.makedirs(os.path.dirname(QDRANT_URL), exist_ok=True)
        client = QdrantClient(path=QDRANT_URL)
    
    # Recreate collection
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    
    points = []
    print("Embedding QA pairs and uploading to Qdrant...")
    for qa in qa_pairs:
        text_to_embed = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
        embedding = list(embedding_model.embed([text_to_embed]))[0]
        
        payload = {
            "source": qa["source"],
            "section": qa["section"],
            "question_number": qa["question_number"],
            "question": qa["question"],
            "answer": qa["answer"],
            "text": text_to_embed,
            "document_type": "faq"
        }
        
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=payload
            )
        )
        
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    print(f"✅ Successfully ingested {len(points)} vectors into {COLLECTION_NAME}.")

if __name__ == "__main__":
    main()
