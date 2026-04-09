"""
vector_db.py — FAISS vector store setup using local HuggingFace embeddings.

This module:
  1. Defines a rich library of sample marketing campaign documents.
  2. Embeds them locally using sentence-transformers (no API key needed).
  3. Persists the FAISS index to disk so it survives server restarts.
  4. Exposes get_vector_db() for the RAG tool in agent.py.

No external API key is required for embeddings — everything runs 100% offline.
"""

import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
DB_PATH = "faiss_index"

# Lightweight, fast, high-quality local embedding model (~90 MB, one-time download)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─── Knowledge Base ───────────────────────────────────────────────────────────
# Each document represents a real-world marketing campaign template.
# The agent retrieves the 2 most relevant ones via semantic similarity at runtime.
SAMPLE_DOCUMENTS = [
    # ── Fashion ──────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Industry: Fashion. Tone: Fun. Goal: Summer Sale. "
            "Caption: 'Summer vibes on point! Shop our massive sale now and stand out 🏖️✨' "
            "Hashtags: #SummerFashion #StyleVibe #OOTDSale #HotLooks "
            "Ad Copy: 'Ready for the summer sun? Grab the latest trendy fits from StyleVibe "
            "with up to 50% off! Slay your summer without breaking the bank.' "
            "Blog Ideas: ['Summer Fashion Trends to Watch', 'How to Build a Capsule Wardrobe on a Budget']"
        ),
        metadata={"id": 1, "industry": "Fashion", "tone": "Fun"},
    ),
    Document(
        page_content=(
            "Industry: Fashion. Tone: Formal. Goal: Luxury Brand Launch. "
            "Caption: 'Crafted for those who appreciate the art of refinement.' "
            "Hashtags: #LuxuryFashion #TimelessStyle #ArtisanCraft #HighEndFashion "
            "Ad Copy: 'Introducing Aurelion — where couture meets craftsmanship. "
            "Each piece is a testament to decades of artisanal heritage and modern elegance.' "
            "Blog Ideas: ['The History of Luxury Fashion', 'Sustainable Luxury — A New Paradigm']"
        ),
        metadata={"id": 2, "industry": "Fashion", "tone": "Formal"},
    ),
    Document(
        page_content=(
            "Industry: Fashion. Tone: Motivational. Goal: Athletic Wear Launch. "
            "Caption: 'Wear your ambition. Every rep, every mile, every win. 💪🔥' "
            "Hashtags: #WearYourAmbition #AthleticWear #FitLife #MoveWithPurpose "
            "Ad Copy: 'Performance meets style. Our new athletic line is engineered for champions "
            "who refuse to settle. Train harder. Look better. Win more.' "
            "Blog Ideas: ['Top Athletic Wear Trends 2025', 'How Your Outfit Affects Athletic Performance']"
        ),
        metadata={"id": 3, "industry": "Fashion", "tone": "Motivational"},
    ),

    # ── Tech ─────────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Industry: Tech. Tone: Formal. Goal: Enterprise Software Sales. "
            "Caption: 'Innovating the way you do business. Discover our enterprise solutions today.' "
            "Hashtags: #EnterpriseAI #DigitalTransformation #B2BTech #CloudFirst "
            "Ad Copy: 'Streamline your enterprise operations with OmniTech Solutions. "
            "Elevate efficiency and drive growth with cutting-edge cloud infrastructure.' "
            "Blog Ideas: ['ROI of Digital Transformation', '5 Signs Your Business Needs an ERP Upgrade']"
        ),
        metadata={"id": 4, "industry": "Tech", "tone": "Formal"},
    ),
    Document(
        page_content=(
            "Industry: Tech. Tone: Casual. Goal: Consumer App Launch. "
            "Caption: 'The app you did not know you needed — until now. Download. Love. Repeat. 📱' "
            "Hashtags: #AppLaunch #TechLife #MustHaveApp #ProductivityHack "
            "Ad Copy: 'Meet FlowApp — the productivity sidekick that fits your life, not the other way around. "
            "Simple to use, impossible to quit.' "
            "Blog Ideas: ['10 Productivity Apps That Actually Work', 'How FlowApp Saves 2 Hours Daily']"
        ),
        metadata={"id": 5, "industry": "Tech", "tone": "Casual"},
    ),

    # ── Fitness ───────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Industry: Fitness. Tone: Motivational. Goal: Gym Memberships. "
            "Caption: 'It is time to build the best version of yourself! 🏋️‍♂️💯 #FitnessGoals' "
            "Hashtags: #FitnessGoals #JoinIronForge #TransformYourself #GymLife "
            "Ad Copy: 'Stop waiting for tomorrow. Join IronForge Gym today and transform your sweat into strength. "
            "Your journey starts here — and it starts NOW.' "
            "Blog Ideas: ['How to Build a Sustainable Workout Routine', 'Nutrition Tips for Gym Beginners']"
        ),
        metadata={"id": 6, "industry": "Fitness", "tone": "Motivational"},
    ),
    Document(
        page_content=(
            "Industry: Fitness. Tone: Energetic. Goal: Supplement Launch. "
            "Caption: 'Push past every limit. Pure performance in every scoop. ⚡🔥' "
            "Hashtags: #FuelYourGains #ProteinPower #FitnessFuel #MaxPerformance "
            "Ad Copy: 'Introducing ProMax Protein — clinically dosed, athlete tested, and built to fuel "
            "your hardest sessions. Your next PR starts here.' "
            "Blog Ideas: ['Protein Timing for Maximum Muscle Growth', 'Pre-Workout vs Post-Workout Nutrition']"
        ),
        metadata={"id": 7, "industry": "Fitness", "tone": "Energetic"},
    ),

    # ── Food & Beverage ──────────────────────────────────────────────────────
    Document(
        page_content=(
            "Industry: Food & Beverage. Tone: Casual. Goal: New Menu Launch. "
            "Caption: 'Slice, slice baby! 🍕 Tag a friend to share our new pizza menu! #Foodie #PizzaTime' "
            "Hashtags: #PizzaTime #NewMenu #FoodieFinds #MustTry "
            "Ad Copy: 'Craving something new? Try our authentic wood-fired pizzas with exclusive toppings! "
            "Order now for 20% off your first delivery.' "
            "Blog Ideas: ['The Art of the Perfect Wood-Fired Pizza', 'Pairing Wines with Your Favourite Pizza']"
        ),
        metadata={"id": 8, "industry": "Food & Beverage", "tone": "Casual"},
    ),
    Document(
        page_content=(
            "Industry: Beverages. Tone: Energetic. Goal: Energy Drink Launch. "
            "Caption: 'No limits. No excuses. Pure energy. Try VoltX today ⚡ #MaxPerformance' "
            "Hashtags: #EnergyDrink #VoltX #MaxPerformance #FuelYourDay "
            "Ad Copy: 'Push past every limit. The new VoltX Energy Drink is engineered for peak performers — "
            "athletes, creators, and hustlers who refuse to slow down.' "
            "Blog Ideas: ['Natural Energy vs Energy Drinks — The Truth', 'How to Power Your Mornings Without Coffee']"
        ),
        metadata={"id": 9, "industry": "Beverages", "tone": "Energetic"},
    ),

    # ── Healthcare ────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Industry: Healthcare. Tone: Formal. Goal: Patient Awareness. "
            "Caption: 'Expert care, compassionate approach. Schedule your appointment now. #HealthFirst' "
            "Hashtags: #HealthFirst #PatientCare #MedicalExcellence #WellnessMatters "
            "Ad Copy: 'Your health is your greatest wealth. Book a consultation with our certified specialists today. "
            "We combine advanced medicine with genuine compassion.' "
            "Blog Ideas: ['Preventive Healthcare — Why It Matters', '5 Health Screenings Everyone Should Get']"
        ),
        metadata={"id": 10, "industry": "Healthcare", "tone": "Formal"},
    ),

    # ── Education ─────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Industry: Education. Tone: Motivational. Goal: Course Enrollment. "
            "Caption: 'Your dream career starts with one decision. Enroll today! 🎓 #LearnAndGrow' "
            "Hashtags: #LearnAndGrow #CareerTransformation #OnlineLearning #SkillUp "
            "Ad Copy: 'Unlock your potential with MasterMinds Academy. Join thousands who have transformed "
            "their careers with industry-led, project-based learning.' "
            "Blog Ideas: ['How Online Learning Changed My Career', 'Top In-Demand Skills for 2025']"
        ),
        metadata={"id": 11, "industry": "Education", "tone": "Motivational"},
    ),

    # ── E-commerce ────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Industry: E-commerce. Tone: Casual. Goal: Flash Sale. "
            "Caption: '24 hours only! Get up to 70% off your favorite brands 🛍️ #FlashSale' "
            "Hashtags: #FlashSale #DealAlert #ShopNow #LimitedTimeOffer "
            "Ad Copy: 'Deals so good they should be illegal! Shop our 24-hour flash sale and save big on everything. "
            "No code needed — discounts applied automatically at checkout.' "
            "Blog Ideas: ['How to Win Flash Sales — Insider Tips', 'The Psychology Behind Limited-Time Offers']"
        ),
        metadata={"id": 12, "industry": "E-commerce", "tone": "Casual"},
    ),

    # ── Real Estate ──────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Industry: Real Estate. Tone: Formal. Goal: Property Launch. "
            "Caption: 'Where luxury meets livability. Welcome home to Pinnacle Residences.' "
            "Hashtags: #LuxuryRealEstate #DreamHome #PropertyLaunch #PremiumLiving "
            "Ad Copy: 'Discover Pinnacle Residences — a landmark development redefining urban luxury. "
            "Thoughtfully designed spaces for those who demand the extraordinary.' "
            "Blog Ideas: ['Top Features in Luxury Homes for 2025', 'Why Location Still Matters Most in Real Estate']"
        ),
        metadata={"id": 13, "industry": "Real Estate", "tone": "Formal"},
    ),

    # ── Travel ───────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Industry: Travel. Tone: Inspirational. Goal: Holiday Package Sales. "
            "Caption: 'The world is calling. Are you ready to answer? ✈️🌍 #Wanderlust' "
            "Hashtags: #Wanderlust #TravelTheWorld #ExploreMore #HolidayDeals "
            "Ad Copy: 'Life is short — make every trip extraordinary. Discover our curated holiday packages "
            "built for explorers, dreamers, and adventure seekers.' "
            "Blog Ideas: ['10 Hidden Destinations You Must Visit in 2025', 'How to Travel on a Budget Without Sacrificing Luxury']"
        ),
        metadata={"id": 14, "industry": "Travel", "tone": "Inspirational"},
    ),
]


# ─── Internal Helpers ─────────────────────────────────────────────────────────

def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns a local HuggingFace embedding model.
    Runs fully offline — no API key or network access required at inference time.
    The model is cached by sentence-transformers after the first download.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def initialize_vector_db() -> None:
    """
    Builds the FAISS index from SAMPLE_DOCUMENTS and persists it to DB_PATH.
    Safe to call multiple times — skips if the index already exists on disk.
    """
    if os.path.exists(DB_PATH):
        logger.info(f"FAISS index already exists at '{DB_PATH}'. Skipping rebuild.")
        return

    logger.info(f"Building FAISS index with {len(SAMPLE_DOCUMENTS)} campaign documents...")
    logger.info(f"Using local embedding model: '{EMBEDDING_MODEL}'")

    try:
        embeddings = _get_embeddings()
        vectorstore = FAISS.from_documents(SAMPLE_DOCUMENTS, embeddings)
        vectorstore.save_local(DB_PATH)
        logger.info(f"FAISS knowledge base successfully built and saved to '{DB_PATH}'.")
    except Exception as e:
        logger.error(f"FAISS build failed: {e}")
        raise


def get_vector_db() -> FAISS:
    """
    Loads the FAISS index from disk.
    Auto-initializes the index if it does not yet exist (lazy init).

    Returns:
        FAISS: A ready-to-query vectorstore instance.
    """
    if not os.path.exists(DB_PATH):
        logger.warning("FAISS index not found — triggering lazy initialization.")
        initialize_vector_db()

    embeddings = _get_embeddings()
    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,  # Safe — we wrote this index ourselves
    )
    logger.info("FAISS index loaded successfully.")
    return vectorstore


# ─── CLI Bootstrap ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run directly to pre-build the FAISS index before starting the server.
    Usage: python vector_db.py
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
    initialize_vector_db()
    print(f"\nFAISS index is ready at '{DB_PATH}/'.")
    print(f"Total documents indexed: {len(SAMPLE_DOCUMENTS)}")
