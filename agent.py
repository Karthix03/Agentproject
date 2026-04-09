import os
import logging
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from schemas import ContentRequest, ContentResponse

load_dotenv()
logger = logging.getLogger(__name__)

# ✅ GROQ CONFIG
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"


def _build_llm():
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise Exception("GROQ_API_KEY missing")

    return ChatOpenAI(
        model=GROQ_MODEL,
        api_key=api_key,
        base_url=GROQ_BASE_URL,
        temperature=0.7,
        max_tokens=800,
    )


def _build_prompt():
    return ChatPromptTemplate.from_template("""
You are a creative AI marketing expert.

Generate marketing content STRICTLY in this format:

Caption:
<caption>

Hashtags:
#tag1 #tag2 #tag3 #tag4 #tag5

Ad Copy:
<ad copy>

Blog Ideas:
1. ...
2. ...
3. ...

---

Brand: {brand_name}
Audience: {target_audience}
Industry: {industry}
Tone: {tone}
Goal: {campaign_goal}
""")


# ✅ PARSER FUNCTION
def parse_output(text: str):
    try:
        lines = text.split("\n")

        caption = ""
        hashtags = []
        ad_copy = ""
        blog_ideas = []

        mode = None

        for line in lines:
            line = line.strip()

            if line.startswith("Caption:"):
                mode = "caption"
                continue
            elif line.startswith("Hashtags:"):
                mode = "hashtags"
                continue
            elif line.startswith("Ad Copy:"):
                mode = "ad"
                continue
            elif line.startswith("Blog Ideas:"):
                mode = "blog"
                continue

            if mode == "caption" and line:
                caption += line + " "

            elif mode == "hashtags" and line:
                hashtags = line.split()

            elif mode == "ad" and line:
                ad_copy += line + " "

            elif mode == "blog" and line:
                blog_ideas.append(line)

        return {
            "caption": caption.strip(),
            "hashtags": hashtags,
            "ad_copy": ad_copy.strip(),
            "blog_ideas": blog_ideas
        }

    except Exception as e:
        logger.error("Parsing failed: %s", e)
        return {
            "caption": text,
            "hashtags": [],
            "ad_copy": "",
            "blog_ideas": []
        }


# ✅ MAIN FUNCTION (FIXED)
def generate_brand_content(request: ContentRequest) -> ContentResponse:

    llm = _build_llm()
    prompt = _build_prompt()

    chain = prompt | llm

    response = chain.invoke({
        "brand_name": request.brand_name,
        "target_audience": request.target_audience,
        "industry": request.industry,
        "tone": request.tone,
        "campaign_goal": request.campaign_goal
    })

    output = response.content.strip()

    parsed = parse_output(output)

    return ContentResponse(
        social_media_caption=parsed["caption"],
        hashtags=parsed["hashtags"],
        ad_copy=parsed["ad_copy"],
        blog_ideas=parsed["blog_ideas"]
    )