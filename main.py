import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from schemas import ContentRequest, ContentResponse, HealthResponse
from agent import generate_brand_content

# ─────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BrandAI")

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="Brand AI (Groq)",
    version="5.0"
)

# ─────────────────────────────────────────────
# CORS (allow frontend access)
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
def health():
    return {
        "status": "online",
        "agent": "Brand AI",
        "llm": "Groq (LLaMA 3.1)",
        "docs": "http://127.0.0.1:8000/docs"
    }

# ─────────────────────────────────────────────
# Generate endpoint
# ─────────────────────────────────────────────
@app.post("/generate", response_model=ContentResponse)
def generate(req: ContentRequest):

    # 🔴 Check API key
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="Missing GROQ_API_KEY"
        )

    try:
        # 🧠 Call agent (IMPORTANT: must return structured JSON)
        result = generate_brand_content(req)

        # ✅ Ensure valid response
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Empty response from AI agent"
            )

        return result

    except Exception as e:
        logger.error(f"Error in /generate: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ─────────────────────────────────────────────
# Run server
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)