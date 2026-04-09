"""
main.py — FastAPI server entry point for the Brand-Aware AI Marketing Agent.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from schemas import ContentRequest, ContentResponse, HealthResponse
from agent import generate_brand_content
from vector_db import initialize_vector_db

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BrandAwareAgent")

# Load env
load_dotenv()


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("=" * 60)
    logger.info("  Brand-Aware AI Marketing Agent — Starting Up")
    logger.info("=" * 60)

    if not os.environ.get("GROQ_API_KEY"):
        logger.warning(
            "GROQ_API_KEY is not set. The /generate endpoint will fail until configured."
        )
    else:
        logger.info("GROQ_API_KEY detected.")
        logger.info("Pre-building FAISS knowledge base...")

        try:
            initialize_vector_db()
            logger.info("FAISS ready.")
        except Exception as e:
            logger.warning(f"FAISS init issue: {e}")

    logger.info("Server ready → http://127.0.0.1:8000/docs")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down...")


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Brand-Aware AI Marketing Agent (Groq)",
    description="AI Agent using Groq + LangChain + FAISS",
    version="4.0.0",
    lifespan=lifespan,
)

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Exception Handler ───────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Server error occurred"},
    )


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
def read_root():
    return {
        "status": "online",
        "agent": "Brand-Aware AI Marketing Agent",
        "llm": "Groq (LLaMA3)",
        "docs": "http://127.0.0.1:8000/docs",
    }


@app.post("/generate", response_model=ContentResponse)
def generate_content(request: ContentRequest):

    api_key = os.environ.get("GROQ_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY missing. Add it to .env file."
        )

    try:
        logger.info(f"Generating content for: {request.brand_name}")
        return generate_brand_content(request)

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agent pipeline error: {str(e)}"
        )


# ─── Run Server ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)