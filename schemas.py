from pydantic import BaseModel, Field
from typing import List


# ── Request मॉडल ─────────────────────────────────────
class ContentRequest(BaseModel):
    brand_name: str = Field(..., min_length=2)
    target_audience: str = Field(..., min_length=5)
    industry: str
    tone: str
    campaign_goal: str


# ── Response मॉडल (FIXED) ───────────────────────────
class ContentResponse(BaseModel):
    social_media_caption: str
    hashtags: List[str]
    ad_copy: str
    blog_ideas: List[str]


# ── Health मॉडल ─────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    agent: str
    llm: str
    docs: str