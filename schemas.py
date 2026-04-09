"""
schemas.py — Pydantic v2 request/response models for the Brand-Aware AI Agent.

All input validation and output serialization is handled here.
FastAPI auto-generates OpenAPI docs from these models.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


# ---------------------------------------------------------------------------
# Allowed tone values — enforced at validation time
# ---------------------------------------------------------------------------
VALID_TONES = {"Formal", "Casual", "Motivational", "Fun", "Energetic", "Inspirational"}


class ContentRequest(BaseModel):
    """
    Input payload for /generate endpoint.
    All five fields are required. Tone is validated against the allowed set.
    """
    brand_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="The brand's official name.",
    )
    target_audience: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description="Describe who the brand is targeting (age, interests, demographics).",
    )
    industry: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Industry vertical (e.g. Fashion, Tech, Food & Beverage, Healthcare).",
    )
    tone: str = Field(
        ...,
        description=f"Communication tone. Allowed values: {', '.join(sorted(VALID_TONES))}.",
    )
    campaign_goal: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description="Primary objective of the marketing campaign.",
    )

    @field_validator("tone")
    @classmethod
    def validate_tone(cls, v: str) -> str:
        # Normalize casing before comparison
        normalized = v.strip().title()
        if normalized not in VALID_TONES:
            raise ValueError(
                f"Invalid tone '{v}'. Must be one of: {', '.join(sorted(VALID_TONES))}."
            )
        return normalized

    model_config = {
        "json_schema_extra": {
            "example": {
                "brand_name": "Lumina Kicks",
                "target_audience": "Gen Z sneaker enthusiasts aged 18–25",
                "industry": "Fashion",
                "tone": "Fun",
                "campaign_goal": "Launch summer streetwear collection",
            }
        }
    }


class ContentResponse(BaseModel):
    """
    Structured output returned by the AI Agent after the full pipeline completes.
    Every field maps directly to a concrete marketing deliverable.
    """
    social_media_caption: str = Field(
        ...,
        description="A punchy, on-brand social media caption (Instagram / X / LinkedIn).",
    )
    hashtags: List[str] = Field(
        ...,
        min_length=3,
        description="4–6 relevant, campaign-specific hashtags.",
    )
    ad_copy: str = Field(
        ...,
        description="Conversion-optimized advertisement copy paragraph.",
    )
    blog_ideas: Optional[List[str]] = Field(
        default=None,
        description="3 SEO-optimized blog post ideas related to the campaign (optional).",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "social_media_caption": "Step into summer with the freshest kicks in the game! 🌞👟",
                "hashtags": ["#LuminaKicks", "#SummerStreetWear", "#SneakerHead", "#FreshKicks"],
                "ad_copy": "This summer, your style speaks before you do. Introducing Lumina Kicks — the streetwear drop Gen Z has been waiting for.",
                "blog_ideas": [
                    "Top 5 Ways to Style Lumina Kicks This Summer",
                    "Why Gen Z is Redefining Streetwear Culture in 2025",
                    "Sneaker Care Tips to Keep Your Fresh Kicks Looking New",
                ],
            }
        }
    }


class HealthResponse(BaseModel):
    """Response model for the GET / health-check endpoint."""
    status: str
    agent: str
    llm: str
    retrieval: str
    faiss_ready: bool
    docs: str
