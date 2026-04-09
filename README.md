# Brand-Aware AI Marketing Agent

A **production-ready, Task-Oriented AI Agent** that automatically generates
premium marketing content using **Grok (xAI)** as the reasoning engine and
**FAISS** for retrieval-augmented generation (RAG).

> **Pipeline:** Input → Think → RAG Action → Structured Output

---

## Agent Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (main.py)                    │
│                                                                  │
│   POST /generate                                                 │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              LangChain AgentExecutor (agent.py)          │   │
│   │                                                          │   │
│   │  ┌──────────────────────────────────────────────────┐   │   │
│   │  │  STEP 1 — THINK                                  │   │   │
│   │  │  Analyze brand identity, audience, industry,     │   │   │
│   │  │  tone, and campaign goal.                        │   │   │
│   │  └──────────────────────┬───────────────────────────┘   │   │
│   │                         │                                │   │
│   │  ┌──────────────────────▼───────────────────────────┐   │   │
│   │  │  STEP 2 — ACT (RAG Tool Call)                    │   │   │
│   │  │  fetch_similar_campaigns(industry, tone)         │   │   │
│   │  │        │                                         │   │   │
│   │  │        ▼                                         │   │   │
│   │  │  FAISS Vector DB ← HuggingFace Embeddings        │   │   │
│   │  │  Returns top-2 semantically similar campaigns    │   │   │
│   │  └──────────────────────┬───────────────────────────┘   │   │
│   │                         │                                │   │
│   │  ┌──────────────────────▼───────────────────────────┐   │   │
│   │  │  STEP 3 — GENERATE                               │   │   │
│   │  │  Grok LLM synthesizes brand context + RAG data   │   │   │
│   │  │  → returns raw JSON                              │   │   │
│   │  └──────────────────────┬───────────────────────────┘   │   │
│   │                         │                                │   │
│   └─────────────────────────┼──────────────────────────────-┘   │
│                             │                                    │
│        ┌────────────────────▼─────────────────────┐             │
│        │  Pydantic v2 Validation (schemas.py)      │             │
│        │  ContentResponse → JSON response          │             │
│        └──────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
BrandAwareAgent/
├── main.py           # FastAPI server — routes, middleware, lifespan hooks
├── agent.py          # LangChain agent — Grok LLM + RAG tool + pipeline logic
├── vector_db.py      # FAISS setup — embeddings, index build, retrieval
├── schemas.py        # Pydantic v2 — input validation + output models
├── requirements.txt  # All Python dependencies (pinned versions)
├── .env.example      # Environment variable template (copy → .env)
├── .env              # Your actual API key (DO NOT commit — in .gitignore)
├── .gitignore        # Git ignore rules
├── faiss_index/      # Auto-generated FAISS index (created on first run)
│   ├── index.faiss
│   └── index.pkl
└── README.md         # This file
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend | FastAPI + Uvicorn | REST API server |
| AI Agent | LangChain AgentExecutor | THINK → ACT → GENERATE pipeline |
| LLM | Grok (xAI) `grok-beta` | Language reasoning + generation |
| LLM API | xAI API (`https://api.x.ai/v1`) | OpenAI-compatible endpoint |
| Vector DB | FAISS (CPU, local) | Semantic similarity search |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Offline embeddings (no API key) |
| Validation | Pydantic v2 | Input/output schema enforcement |
| Config | python-dotenv | `.env` file loading |

---

## How to Run the Project

### Prerequisites

Make sure you have the following installed before starting:

| Tool | Version | Download |
|------|---------|----------|
| Python | 3.10 or higher | https://python.org/downloads |
| VS Code | Latest | https://code.visualstudio.com |
| Git | Latest | https://git-scm.com |
| Postman | Latest | https://postman.com/downloads |

---

### Step 1 — Get the Project

**Option A — If you have the files already:**
```bash
# Navigate to where you placed the folder
cd path/to/BrandAwareAgent
```

**Option B — Clone from Git:**
```bash
git clone https://github.com/your-username/BrandAwareAgent.git
cd BrandAwareAgent
```

---

### Step 2 — Set Up Python Virtual Environment

A virtual environment isolates the project's dependencies from your system Python.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the beginning of your terminal prompt, confirming
the virtual environment is active.

> **Tip:** In VS Code, press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac), search for
> "Python: Select Interpreter", and choose the `venv` interpreter.

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, LangChain, Grok client, FAISS, and all other dependencies.

> **First-time note:** On the first run, `sentence-transformers` will download the
> `all-MiniLM-L6-v2` model (~90MB). This is a one-time download; subsequent runs
> use the cached version.

Verify the installation:
```bash
python -c "import fastapi, langchain, faiss; print('All dependencies installed!')"
```

---

### Step 4 — Configure Your API Key

**4a. Copy the environment template:**

Windows:
```bash
copy .env.example .env
```

macOS / Linux:
```bash
cp .env.example .env
```

**4b. Get your Grok API key:**
1. Go to **https://console.x.ai/**
2. Sign in or create a free account
3. Navigate to **API Keys**
4. Click **Create API Key** and copy it

**4c. Paste your key into `.env`:**

Open `.env` in any text editor:
```env
XAI_API_KEY=xai-your_real_grok_api_key_here
```

Replace `xai-your_real_grok_api_key_here` with your actual key.

> **Important:** Never share your `.env` file or commit it to Git. It's already
> in `.gitignore`.

---

### Step 5 — (Optional) Pre-Build the FAISS Index

The FAISS index is built automatically on the first `/generate` request, but you
can pre-build it for a faster first response:

```bash
python vector_db.py
```

Expected output:
```
2025-01-15 10:00:00 — INFO — Building FAISS index with 14 campaign documents...
2025-01-15 10:00:05 — INFO — FAISS knowledge base successfully built and saved to 'faiss_index/'.

FAISS index is ready at 'faiss_index/'.
Total documents indexed: 14
```

---

### Step 6 — Start the Server

```bash
uvicorn main:app --reload
```

Expected output:
```
2025-01-15 10:00:00 | BrandAwareAgent | INFO | ============================================================
2025-01-15 10:00:00 | BrandAwareAgent | INFO |   Brand-Aware AI Marketing Agent — Starting Up
2025-01-15 10:00:00 | BrandAwareAgent | INFO | ============================================================
2025-01-15 10:00:00 | BrandAwareAgent | INFO | XAI_API_KEY detected.
2025-01-15 10:00:00 | BrandAwareAgent | INFO | FAISS knowledge base is ready.
2025-01-15 10:00:00 | BrandAwareAgent | INFO | Server is ready. Visit http://127.0.0.1:8000/docs
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
```

The server is live at **http://127.0.0.1:8000**

---

## Testing the API

### Option A — Swagger UI (Easiest)

1. Open your browser and navigate to: **http://127.0.0.1:8000/docs**
2. Click on `POST /generate`
3. Click **"Try it out"**
4. Paste the sample JSON below into the request body
5. Click **"Execute"**

---

### Option B — Postman

1. Open Postman → Click **New** → **HTTP Request**
2. Set method to **POST**
3. URL: `http://127.0.0.1:8000/generate`
4. Click the **Body** tab → Select **raw** → Choose **JSON** from the dropdown
5. Paste this payload:

```json
{
  "brand_name": "Lumina Kicks",
  "target_audience": "Gen Z sneaker enthusiasts aged 18-25",
  "industry": "Fashion",
  "tone": "Fun",
  "campaign_goal": "Launch summer streetwear collection"
}
```

6. Click **Send**

---

### Option C — cURL (Terminal)

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "brand_name": "Lumina Kicks",
    "target_audience": "Gen Z sneaker enthusiasts aged 18-25",
    "industry": "Fashion",
    "tone": "Fun",
    "campaign_goal": "Launch summer streetwear collection"
  }'
```

---

## Sample Request Payloads

**Fashion — Fun:**
```json
{
  "brand_name": "Lumina Kicks",
  "target_audience": "Gen Z sneaker enthusiasts aged 18-25",
  "industry": "Fashion",
  "tone": "Fun",
  "campaign_goal": "Launch summer streetwear collection"
}
```

**Tech — Formal:**
```json
{
  "brand_name": "NexaCloud",
  "target_audience": "Enterprise CTOs and IT decision-makers",
  "industry": "Tech",
  "tone": "Formal",
  "campaign_goal": "Drive SaaS platform adoption among Fortune 500 companies"
}
```

**Fitness — Motivational:**
```json
{
  "brand_name": "IronForge Gym",
  "target_audience": "Adults aged 25-40 looking to transform their fitness",
  "industry": "Fitness",
  "tone": "Motivational",
  "campaign_goal": "Increase new gym memberships by 30% in Q1"
}
```

**Food & Beverage — Casual:**
```json
{
  "brand_name": "Zesto Kitchen",
  "target_audience": "Food lovers aged 22-35 who order delivery 3x per week",
  "industry": "Food & Beverage",
  "tone": "Casual",
  "campaign_goal": "Launch new fusion menu and increase repeat orders"
}
```

---

## Expected Output

```json
{
  "social_media_caption": "Step into summer with the freshest kicks in the game! 🌞👟 Lumina Kicks just dropped and your wardrobe will never be the same.",
  "hashtags": [
    "#LuminaKicks",
    "#SummerStreetWear",
    "#SneakerHead",
    "#FreshKicks",
    "#GenZFashion"
  ],
  "ad_copy": "This summer, your style speaks before you do. Introducing Lumina Kicks — the streetwear drop Gen Z has been waiting for. Bold silhouettes, vibrant colorways, and limitless energy. Every pair tells a story. Yours starts now — grab your pair before they sell out.",
  "blog_ideas": [
    "Top 5 Ways to Style Lumina Kicks This Summer",
    "Why Gen Z is Redefining Streetwear Culture in 2025",
    "Sneaker Care Tips to Keep Your Fresh Kicks Looking New"
  ]
}
```

---

## API Reference

### `GET /`

Health check endpoint.

**Response:**
```json
{
  "status": "online",
  "agent": "Brand-Aware AI Marketing Agent",
  "llm": "Grok (xAI) — grok-beta",
  "retrieval": "FAISS + HuggingFace all-MiniLM-L6-v2 (local, offline)",
  "faiss_ready": true,
  "docs": "http://127.0.0.1:8000/docs"
}
```

---

### `POST /generate`

Executes the full AI Agent pipeline.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `brand_name` | string | Yes | Brand's official name (2–100 chars) |
| `target_audience` | string | Yes | Target audience description (5–300 chars) |
| `industry` | string | Yes | Industry vertical (e.g. Fashion, Tech) |
| `tone` | string | Yes | One of: Formal, Casual, Motivational, Fun, Energetic, Inspirational |
| `campaign_goal` | string | Yes | Primary campaign objective (5–300 chars) |

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| `social_media_caption` | string | On-brand social media caption |
| `hashtags` | string[] | 4–6 campaign-relevant hashtags |
| `ad_copy` | string | Conversion-optimized ad copy paragraph |
| `blog_ideas` | string[] or null | 3 SEO-optimized blog title ideas |

---

## Prompt Engineering Logic

The agent uses a structured multi-step system prompt:

1. **Role priming** — Grok is positioned as an "elite AI Marketing Director" to
   activate domain-specific knowledge patterns.

2. **Mandatory pipeline** — The prompt enforces STEP 1 → STEP 2 → STEP 3 with
   explicit labels, preventing the model from skipping the RAG tool call.

3. **Tool constraint** — "You MUST call `fetch_similar_campaigns`. This is
   NON-NEGOTIABLE" ensures the RAG step always executes.

4. **Output contract** — Raw JSON only, specific keys, field-level descriptions.
   Markdown code fences are explicitly forbidden (with defensive stripping as backup).

5. **Anti-plagiarism instruction** — "Do NOT copy the retrieved examples verbatim"
   ensures original content is generated, not template copying.

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `XAI_API_KEY is not configured` | Missing or empty `.env` file | Create `.env` from `.env.example` and add your key |
| `Connection refused` or `Network error` | Wrong API key or no internet | Verify key at https://console.x.ai, check network |
| `ModuleNotFoundError` | Dependencies not installed | Run `pip install -r requirements.txt` inside venv |
| `FAISS init issue` | Embedding model download failed | Check internet; delete `faiss_index/` folder and restart |
| `422 Unprocessable Entity` | Invalid request payload or tone value | Check all 5 fields are present; tone must be from allowed list |
| `The agent returned malformed JSON` | Grok output formatting issue | Retry the request; intermittent LLM formatting errors |
| `(venv)` not showing in terminal | Virtual env not activated | Run `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows) |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `XAI_API_KEY` | Yes | Your Grok (xAI) API key from https://console.x.ai/ |

---

## Knowledge Base

The FAISS knowledge base contains **14 curated campaign templates** spanning:
Fashion (Fun, Formal, Motivational), Tech (Formal, Casual), Fitness (Motivational, Energetic),
Food & Beverage (Casual), Beverages (Energetic), Healthcare (Formal), Education (Motivational),
E-commerce (Casual), Real Estate (Formal), and Travel (Inspirational).

At runtime, the agent retrieves the **top-2 most semantically similar** documents
using cosine similarity on HuggingFace embeddings — providing relevant creative
context without restricting the LLM's originality.

To add more documents, edit the `SAMPLE_DOCUMENTS` list in `vector_db.py`,
delete the `faiss_index/` folder, and restart the server.
