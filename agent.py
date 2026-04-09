"""
agent.py — Core AI Agent logic for Brand-Aware Content Generation.
"""

import json
import re
import os
import logging

# ✅ Load environment variables
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from vector_db import get_vector_db
from schemas import ContentRequest, ContentResponse

logger = logging.getLogger(__name__)

# ─── GROQ CONFIGURATION (UPDATED ✅) ─────────────────────────────────────────
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama3-70b-8192"   # 🔥 Fast + Free


# ─── RAG Tool ────────────────────────────────────────────────────────────────

@tool
def fetch_similar_campaigns(industry: str, tone: str) -> str:
    """
    Fetch similar marketing campaigns using FAISS vector similarity search
    based on the given industry and tone.
    """
    logger.info(f"[RAG] Querying FAISS — Industry: '{industry}' | Tone: '{tone}'")

    try:
        vectorstore = get_vector_db()
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2},
        )

        query = f"Industry: {industry}. Tone: {tone}. Marketing campaign content."
        docs = retriever.invoke(query)

    except Exception as e:
        logger.error(f"[RAG] FAISS retrieval failed: {e}")
        return "Knowledge base unavailable. Generate original content."

    if not docs:
        return "No past campaigns found. Generate fresh content."

    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"[Reference Campaign {i}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


# ─── LLM Builder (UPDATED ✅) ────────────────────────────────────────────────

def _build_llm() -> ChatOpenAI:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()

    print("Loaded GROQ API KEY:", api_key[:10] if api_key else "None")

    if not api_key:
        raise EnvironmentError(
            "❌ GROQ_API_KEY is missing. Add it to .env file."
        )

    return ChatOpenAI(
        model=GROQ_MODEL,
        api_key=api_key,
        base_url=GROQ_BASE_URL,
        temperature=0.75,
        max_tokens=1500,
    )


# ─── Prompt Builder ──────────────────────────────────────────────────────────

def _build_prompt() -> ChatPromptTemplate:
    system_prompt = (
        "You are an elite AI Marketing Director Agent.\n\n"

        "STEP 1: Analyze brand deeply.\n"
        "STEP 2: MUST call fetch_similar_campaigns tool.\n"
        "STEP 3: Generate high-quality marketing content.\n\n"

        "OUTPUT FORMAT:\n"
        "Return ONLY valid JSON:\n"

        "{{\n"
        '  "social_media_caption": "...",\n'
        '  "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"],\n'
        '  "ad_copy": "...",\n'
        '  "blog_ideas": ["idea1", "idea2", "idea3"]\n'
        "}}\n"
    )

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", (
            "Brand: {brand_name}\n"
            "Audience: {target_audience}\n"
            "Industry: {industry}\n"
            "Tone: {tone}\n"
            "Goal: {campaign_goal}"
        )),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


# ─── MAIN FUNCTION ───────────────────────────────────────────────────────────

def generate_brand_content(request: ContentRequest) -> ContentResponse:

    logger.info(f"🚀 Starting pipeline for: {request.brand_name}")

    llm = _build_llm()
    tools = [fetch_similar_campaigns]

    prompt = _build_prompt()
    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )

    result = agent_executor.invoke({
        "brand_name": request.brand_name,
        "target_audience": request.target_audience,
        "industry": request.industry,
        "tone": request.tone,
        "campaign_goal": request.campaign_goal,
    })

    output_text = result["output"].strip()

    # Remove markdown if model adds it
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", output_text, re.DOTALL)
    if fence_match:
        output_text = fence_match.group(1).strip()

    try:
        data = json.loads(output_text)
        return ContentResponse(**data)

    except Exception as e:
        print("❌ RAW OUTPUT:", output_text)
        raise ValueError("Invalid JSON from model") from e