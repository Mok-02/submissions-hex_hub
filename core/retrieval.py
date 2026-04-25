"""
retrieval.py — Retrieval + Scoring Engine

For each tile in the database:
  1. Compute cosine similarity between query embedding and tile embedding
  2. Compute rule-based metadata scores (NDVI thresholds, water flags, etc.)
  3. Fuse scores with configurable weights
  4. Return top-K ranked results

Rule weights and thresholds are all externalized — easy to tune without touching logic.
"""

import numpy as np
from core.embeddings import cosine_similarity
from typing import Optional

RULE_WEIGHTS = {
    "similarity":      0.45,
    "ndvi_stress":     0.15,
    "water_shortage":  0.15,
    "land_type_match": 0.20,  # increased from 0.10
    "ndvi_health":     0.05,  # new signal for healthy vegetation
}

NDVI_STRESS_THRESHOLD = 0.35
NDVI_HEALTH_THRESHOLD = 0.60
WATER_LOW_THRESHOLD = 0.35
WATER_HIGH_THRESHOLD = 0.65


def _rule_score(intent: dict, metadata: dict) -> dict:
    """
    Compute normalized rule-based sub-scores [0, 1] for each signal.
    Returns a breakdown dict for transparency.
    """
    scores = {}
    ndvi = metadata.get("ndvi", 0.5)
    water = metadata.get("water_presence", 0.5)
    land = metadata.get("land_type", "unknown")

    # Crop stress: reward low NDVI
    if intent.get("crop_stress"):
        scores["ndvi_stress"] = max(0.0, 1.0 - ndvi / NDVI_STRESS_THRESHOLD) if ndvi < NDVI_STRESS_THRESHOLD else 0.0
    else:
        scores["ndvi_stress"] = 0.0

    # Water shortage: reward low water
    if intent.get("water_shortage"):
        scores["water_shortage"] = max(0.0, 1.0 - water / WATER_LOW_THRESHOLD) if water < WATER_LOW_THRESHOLD else 0.0
    else:
        scores["water_shortage"] = 0.0

    # Vegetation health: reward high NDVI
    if intent.get("vegetation_health"):
        scores["ndvi_health"] = max(0.0, (ndvi - NDVI_HEALTH_THRESHOLD) / (1.0 - NDVI_HEALTH_THRESHOLD)) if ndvi > NDVI_HEALTH_THRESHOLD else 0.0
    else:
        scores["ndvi_health"] = 0.0

    # Land type match
    land_intent_map = {
        "agriculture": ["agriculture"],
        "urban":       ["urban"],
        "water_body":  ["water"],
        "forest":      ["forest"],
        "barren":      ["barren"],
    }
    land_match = 0.0
    for intent_key, land_types in land_intent_map.items():
        if intent.get(intent_key) and land in land_types:
            land_match = 1.0
            break
    scores["land_type_match"] = land_match

# Penalize wrong land type for agriculture queries
    if intent.get("crop_stress") or intent.get("agriculture"):
        if land in ["water"]:
            scores["ndvi_stress"] = 0.0  # water's negative NDVI is not crop stress

    # Penalize wrong land type for water shortage queries  
    if intent.get("water_shortage"):
        if land in ["water"]:
            scores["water_shortage"] = 0.0  # water bodies don't have water shortage

    return scores


def score_tile(
    query_embedding: np.ndarray,
    tile_embedding: np.ndarray,
    intent: dict,
    metadata: dict,
) -> dict:
    """
    Full scoring pipeline for a single tile.
    Returns score breakdown for explanation layer.
    """
    sim = (cosine_similarity(query_embedding, tile_embedding) + 1.0) / 2.0  # normalize [-1,1] → [0,1]
    rules = _rule_score(intent, metadata)

    final = (
        RULE_WEIGHTS["similarity"] * sim
        + RULE_WEIGHTS["ndvi_stress"] * rules["ndvi_stress"]
        + RULE_WEIGHTS["water_shortage"] * rules["water_shortage"]
        + RULE_WEIGHTS["land_type_match"] * rules["land_type_match"]
        + RULE_WEIGHTS["ndvi_health"] * rules["ndvi_health"]
    )

    return {
        "similarity": round(sim, 4),
        "rules": {k: round(v, 4) for k, v in rules.items()},
        "final_score": round(min(final, 1.0), 4),
    }


def retrieve_top_k(
    query_embedding: np.ndarray,
    embedding_db: dict,
    intent: dict,
    k: int = 5,
) -> list[dict]:
    """
    Score all tiles, rank, return top-K with full breakdown.
    """
    results = []
    for tile_id, entry in embedding_db.items():
        score_breakdown = score_tile(
            query_embedding,
            entry["embedding"],
            intent,
            entry["metadata"],
        )
        results.append({
            "tile_id": tile_id,
            "metadata": entry["metadata"],
            "score_breakdown": score_breakdown,
            "final_score": score_breakdown["final_score"],
        })

    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results[:k]
