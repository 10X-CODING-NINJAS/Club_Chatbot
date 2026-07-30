import os
import importlib
import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma

# Dynamically import the embedding pipeline
feeding_pipeline = importlib.import_module("1_data_feeding_pipeline")
OpenRouterEmbeddings = feeding_pipeline.OpenRouterEmbeddings

# Dynamically import the fallback routing logic
fallback_pipeline = importlib.import_module("5_fallback_with_ollama")
call_llm_with_fallback = fallback_pipeline.call_llm_with_fallback

# Load environment variables
load_dotenv()

# Connect to your document database
persistent_directory = "db/chroma_db"
embeddings = OpenRouterEmbeddings(
    model="nvidia/llama-nemotron-embed-vl-1b-v2:free", # free nvidia model
    api_key=os.getenv("OPENROUTER_API_KEY")
)
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# This is the AI's memory.
chat_history = []

# NOTE: The hardcoded call_llm() function has been completely removed.
# All LLM calls now route through call_llm_with_fallback() from file 5.

def ask_question(user_question: str) -> str:
    """The brain of the chatbot. Rewrites follow-up questions then fetches answers."""
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Make the question clear using conversation history
    if chat_history:
        messages = [
            {"role": "system", "content": "Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."}
        ] + chat_history + [
            {"role": "user", "content": f"New question: {user_question}"}
        ]
        
        # Swapped to the fallback router here
        search_question = call_llm_with_fallback(messages).strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3}) 
    docs = retriever.invoke(search_question)

    # removed the documents print statement that were used to answer qs

    # Step 3: Create final prompt with context
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    # Step 4: Get the answer (include chat history for context)
    messages = [
        {
            "role": "system", 
            "content": (
                "You are 'Spider-Bot', the friendly neighborhood assistant for the Coding Ninjas 10X Club."
                "You speak enthusiastically, just like Peter Parker / Spider-Man. Use mild Spider-Man slang "
                "(like 'web-slinging', 'spidey-sense', 'thwip', or 'with great power comes great code'). "
                "CRITICAL RULE: You must ONLY answer questions using the provided documents and conversation history. "
                "If the context doesn't have the answer, just say your spidey-sense is tingling but you don't have enough info in your web-shooters to answer that right now."
            )
        }
    ] + chat_history + [
        {"role": "user", "content": combined_input}
    ]

    # Swapped to the fallback router here
    answer = call_llm_with_fallback(messages)

    # Step 5: Remember this conversation
    chat_history.append({"role": "user", "content": user_question})
    chat_history.append({"role": "assistant", "content": answer})

    print(f"\nAnswer: {answer}")
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