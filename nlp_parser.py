"""
nlp_parser.py — Query Understanding Layer

Extracts structured intent from free-form user queries.
Two-tier approach:
  1. Keyword matching (fast, deterministic — runs always)
  2. LLM-based fallback intent refinement (optional, plug in later)

The output intent dict is the shared contract between NLP and Embedding/Rule layers.
"""

import re
from typing import Optional

INTENT_SCHEMA = {
    "crop_stress": False,
    "water_shortage": False,
    "vegetation_health": False,
    "water_abundance": False,
    "urban_expansion": False,
    "deforestation": False,
    "flood_risk": False,
    "fire_risk": False,
    "agriculture": False,
    "urban": False,
    "water_body": False,
    "forest": False,
    "barren": False,
}

KEYWORD_RULES = [
    # Crop stress
    (["crop stress", "plant stress", "stressed crop", "crop damage",
      "low ndvi", "low vegetation", "dying crop", "wilting"], "crop_stress"),

    # Water shortage
    (["water shortage", "water scarcity", "drought", "dry land",
      "no water", "arid", "low water", "water deficit"], "water_shortage"),

    # Vegetation health
    (["healthy vegetation", "high ndvi", "green cover", "dense vegetation",
      "lush", "vegetation health", "good crop"], "vegetation_health"),

    # Water abundance
    (["water abundance", "flooded", "flood", "high water", "wet land",
      "waterlogged", "inundated", "excess water"], "water_abundance"),

    # Urban
    (["urban", "city", "town", "built-up", "settlement",
      "infrastructure", "roads", "concrete", "urban sprawl"], "urban"),
    (["urban expansion", "urban growth", "urbanization", "city expansion"], "urban_expansion"),

   # Add "pasture" and "grassland" to agriculture
    (["agriculture", "agricultural", "farmland", "farm", "crop",
    "field", "cultivation", "irrigated", "pasture", "grassland",
    "grazing", "pasture lands"], "agriculture"),

    # Add "rainfall", "rain", "monsoon", "waterlogged" to water_abundance  
    (["water abundance", "flooded", "flood", "high water", "wet land",
    "waterlogged", "inundated", "excess water", "rainfall", "rain",
    "heavy rain", "heavy rainfall", "monsoon", "saturated"], "water_abundance"),

    # Add "drought", "dry", "arid" to water_shortage
    (["water shortage", "water scarcity", "drought", "dry land",
    "no water", "arid", "low water", "water deficit", "dry",
    "drought areas", "parched", "desiccated"], "water_shortage"),
    
    # Agriculture
    (["agriculture", "agricultural", "farmland", "farm", "crop",
      "field", "cultivation", "irrigated"], "agriculture"),

    # Forest / deforestation
    (["forest", "woodland", "trees", "forested"], "forest"),
    (["deforestation", "forest loss", "tree loss", "cleared land",
      "logging", "forest degradation"], "deforestation"),

    # Water body
    (["river", "lake", "reservoir", "water body", "water bodies", "pond",
      "stream", "wetland", "canal", "water detected"], "water_body"),

    # Hazards
    (["flood risk", "flood prone", "flood zone"], "flood_risk"),
    (["fire risk", "wildfire", "burn scar", "fire", "combustion"], "fire_risk"),

    # Bare / barren
    (["barren", "bare soil", "wasteland", "desert", "sparse vegetation"], "barren"),
]


def parse_query(query: str) -> dict:
    """
    Parse a natural language query into a structured intent dict.
    Returns a copy of INTENT_SCHEMA with matched fields set to True.
    """
    intent = dict(INTENT_SCHEMA)
    q = query.lower()
    q = re.sub(r"[^\w\s\-+&]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    for keywords, field in KEYWORD_RULES:
        for kw in keywords:
            if kw in q:
                intent[field] = True
                break

    # Compound inference: if crop_stress + water_shortage both detected,
    # that's a strong agriculture signal
    if intent["crop_stress"] and intent["water_shortage"]:
        intent["agriculture"] = True

    # Deforestation implies forest context
    if intent["deforestation"]:
        intent["forest"] = True

    return intent


def explain_intent(intent: dict) -> str:
    """Human-readable summary of parsed intent for UI display."""
    active = [k for k, v in intent.items() if v]
    if not active:
        return "No specific intent detected — using general similarity search."

    labels = {
        "crop_stress": "crop stress",
        "water_shortage": "water shortage",
        "vegetation_health": "healthy vegetation",
        "water_abundance": "water abundance",
        "urban_expansion": "urban expansion",
        "deforestation": "deforestation",
        "flood_risk": "flood risk",
        "fire_risk": "fire risk",
        "agriculture": "agricultural land",
        "urban": "urban areas",
        "water_body": "water bodies",
        "forest": "forested areas",
        "barren": "barren land",
    }
    readable = [labels.get(k, k) for k in active]
    return "Detected: " + ", ".join(readable)
