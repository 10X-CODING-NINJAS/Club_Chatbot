# 🕷️ Club Chatbot API

RAG-powered chatbot API for the **Coding Ninjas 10X Club**. Meet **Spider-Bot** — your friendly neighborhood assistant that answers questions about the club using verified club data.

## Architecture

```
Website (Frontend)  →  FastAPI Server (app.py)  →  ChromaDB (Vector Store)
                                ↓
                    LLM Fallback Chain (5_fallback_with_ollama.py)
                    ├── Tier 1: Mistral
                    ├── Tier 2: Groq
                    ├── Tier 3: OpenRouter
                    └── Tier 4: Ollama
```

**How it works:**
1. User asks a question via the `/chat` API
2. The question is embedded and searched against club data in ChromaDB
3. Relevant chunks + conversation history are sent to the LLM
4. Spider-Bot responds in character 🕸️

## Project Structure

```
├── app.py                      # FastAPI server (main entry point)
├── 1_data_feeding_pipeline.py  # Document ingestion → ChromaDB
├── 5_fallback_with_ollama.py   # LLM provider cascade
├── info/                       # Source .txt files for the knowledge base
│   └── ClubQuestions.txt
├── pyproject.toml              # Dependencies
└── .env                        # API keys (not in git)
```

## Setup

### 1. Clone & install dependencies

```bash
git clone git@github.com:10X-CODING-NINJAS/Club_Chatbot.git
cd Club_Chatbot
uv sync
```

### 2. Create your `.env` file

```env
# Required — used for embeddings + Tier 1 LLM
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openrouter/free

# Optional fallbacks (configure whichever you have)
MISTRAL_API_KEY=your_mistral_api_key
GROQ_API_KEY=your_groq_api_key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Timeout (seconds)
REQUEST_TIMEOUT=30
```

### 3. Run the server

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server will automatically ingest data from `info/` on first startup.

**API docs** will be available at: `http://localhost:8000/docs`

## API Endpoints

### `POST /chat` — Send a message

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Coding Ninjas 10X Club?"}'
```

**Request:**
```json
{
  "message": "What is Coding Ninjas 10X Club?",
  "session_id": "optional-uuid-for-multi-turn-chat"
}
```

**Response:**
```json
{
  "response": "Hey there, web-slinger! The Coding Ninjas 10X Club is...",
  "session_id": "a1b2c3d4-e5f6-..."
}
```

> 💡 **Multi-turn chat:** Save the `session_id` from the first response and send it with subsequent messages to maintain conversation context.

### `POST /chat/reset` — Reset a session

```bash
curl -X POST http://localhost:8000/chat/reset \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id"}'
```

### `GET /health` — Health check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "vector_store": true
}
```

### `POST /ingest` — Re-ingest documents

Rebuilds the vector store from `info/` directory. Call this after updating the `.txt` files.

```bash
curl -X POST http://localhost:8000/ingest
```

## Website Integration

### JavaScript (Fetch API)

```javascript
// Generate a session ID once per user/conversation
const sessionId = crypto.randomUUID();

async function sendMessage(message) {
  const response = await fetch('http://YOUR_SERVER:8000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: message,
      session_id: sessionId
    })
  });

  const data = await response.json();
  return data.response;
}

// Usage
const answer = await sendMessage("What is Coding Ninjas?");
console.log(answer);
```

### React Example

```jsx
import { useState, useRef } from 'react';

const API_URL = 'http://YOUR_SERVER:8000';

function ChatBot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const sessionId = useRef(crypto.randomUUID());

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg = input;
    setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMsg,
          session_id: sessionId.current
        })
      });

      const data = await res.json();
      setMessages(prev => [...prev, { role: 'bot', text: data.response }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'bot', text: 'Oops! Something went wrong.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={msg.role}>{msg.text}</div>
        ))}
        {loading && <div className="bot">Thinking...</div>}
      </div>
      <input
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={e => e.key === 'Enter' && sendMessage()}
        placeholder="Ask Spider-Bot..."
      />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
}
```

## Deployment

### Deploy on Railway (Recommended)
1. Go to [railway.app](https://railway.app) and sign in with GitHub.
2. Click **New Project** → **Deploy from GitHub Repo**.
3. Select the `Club_Chatbot` repository. Railway will automatically detect the `Procfile`.
4. Once deployed, click on your service, go to the **Variables** tab, and click **Raw Editor**. Paste your `.env` content:
   ```env
   OPENROUTER_API_KEY=your_key_here
   OPENROUTER_MODEL=openrouter/free
   MISTRAL_API_KEY=your_key_here
   MISTRAL_MODEL=mistral-large-latest
   GROQ_API_KEY=your_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   REQUEST_TIMEOUT=30
   ```
5. Click **Update Variables**.
6. Go to **Settings** → **Networking** → **Generate Domain** to get your public API URL.

### Deploy on a VPS (DigitalOcean, AWS, etc.)
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone git@github.com:10X-CODING-NINJAS/Club_Chatbot.git
cd Club_Chatbot
uv sync

# Create .env with your API keys
nano .env

# Run in background
nohup uv run uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2 &
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI |
| Vector DB | ChromaDB (local, persistent) |
| Embeddings | `nvidia/llama-nemotron-embed-vl-1b-v2:free` via OpenRouter |
| LLM | 4-tier fallback: Mistral → Groq → OpenRouter → Ollama |
| Orchestration | LangChain |
| Package Manager | uv |

## Team

Built by the **10X Coding Ninjas** 🥷
