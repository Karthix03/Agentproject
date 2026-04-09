# 🚀 Brand AI — Marketing Content Generator

A **full-stack AI-powered marketing content generator** that creates:
- Social media captions
- Hashtags
- Ad copy
- Blog ideas  

Built using **FastAPI + Groq (LLaMA 3.1) + LangChain + Custom UI**

---

## 🧠 How It Works
# 🚀 Brand AI — Marketing Content Generator

A **full-stack AI-powered marketing content generator** that creates:
- Social media captions
- Hashtags
- Ad copy
- Blog ideas  

Built using **FastAPI + Groq (LLaMA 3.1) + LangChain + Custom UI**

---

## 🧠 How It Works
Frontend (HTML UI)
│
▼
FastAPI Backend (main.py)
│
▼
LangChain + Groq LLM (agent.py)
│
▼
Structured Output (JSON)
│
▼
Rendered in UI (cards)

Frontend (HTML UI)
│
▼
FastAPI Backend (main.py)
│
▼
LangChain + Groq LLM (agent.py)
│
▼
Structured Output (JSON)
│
▼
Rendered in UI (cards)

📁 Project Structure

BrandAI/
├── main.py # FastAPI backend (API routes)
├── agent.py # LLM logic (Groq + prompt + parsing)
├── schemas.py # Request & response models
├── .env # API key (NOT committed)
├── requirements.txt # Dependencies
└── frontend.html # Your UI file


---

## ⚙️ Tech Stack

| Layer | Tech |
|------|------|
| Frontend | HTML, CSS, JavaScript |
| Backend | FastAPI |
| AI Model | Groq (LLaMA 3.1) |
| Framework | LangChain |
| Validation | Pydantic |

---

## 🚀 Setup Instructions

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Set Up Environment Variables

Create a `.env` file and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3️⃣ Run the FastAPI Server

```bash
uvicorn main:app --reload
```