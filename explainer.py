"""
explainer.py — Explanation Layer

Generates human-readable explanations for each top-K result.
Template-based (deterministic, fast) with LLM plug-in hook for later.

Each explanation covers:
  - Why this tile scored high
  - Specific metadata evidence
  - Confidence qualification
"""

CONFIDENCE_LABELS = {
    (0.45, 1.01): "High confidence",
    (0.35, 0.45): "Moderate confidence",
    (0.25, 0.35): "Low confidence",
    (0.00, 0.25): "Weak match",
}

NDVI_DESCRIPTORS = [
    (0.00, 0.20, "critically stressed (NDVI {ndvi:.2f})"),
    (0.20, 0.35, "stressed (NDVI {ndvi:.2f})"),
    (0.35, 0.55, "moderate vegetation cover (NDVI {ndvi:.2f})"),
    (0.55, 0.75, "healthy vegetation (NDVI {ndvi:.2f})"),
    (0.75, 1.01, "dense healthy vegetation (NDVI {ndvi:.2f})"),
]

WATER_DESCRIPTORS = [
    (0.00, 0.25, "severe water deficit"),
    (0.25, 0.45, "low water availability"),
    (0.45, 0.65, "moderate water presence"),
    (0.65, 0.85, "good water availability"),
    (0.85, 1.01, "abundant water presence"),
]

LAND_DESCRIPTIONS = {
    "agriculture": "agricultural land",
    "urban":       "urban/built-up area",
    "water":       "water body",
    "forest":      "forested region",
    "barren":      "barren/bare soil",
    "unknown":     "unclassified land",
}


def _get_confidence(score: float) -> str:
    for (lo, hi), label in CONFIDENCE_LABELS.items():
        if lo <= score < hi:
            return label
    return "Unknown confidence"


def _ndvi_desc(ndvi: float) -> str:
    for lo, hi, tmpl in NDVI_DESCRIPTORS:
        if lo <= ndvi < hi:
            return tmpl.format(ndvi=ndvi)
    return f"NDVI {ndvi:.2f}"


def _water_desc(water: float) -> str:
    for lo, hi, label in WATER_DESCRIPTORS:
        if lo <= water < hi:
            return label
    return f"water presence {water:.2f}"


def generate_explanation(
    tile_id: str,
    metadata: dict,
    score_breakdown: dict,
    intent: dict,
    rank: int,
) -> str:
    ndvi = metadata.get("ndvi", 0.5)
    water = metadata.get("water_presence", 0.5)
    land = metadata.get("land_type", "unknown")
    lat = metadata.get("lat", None)
    lon = metadata.get("lon", None)

    confidence = _get_confidence(score_breakdown["final_score"])
    ndvi_str = _ndvi_desc(ndvi)
    water_str = _water_desc(water)
    land_str = LAND_DESCRIPTIONS.get(land, land)

    parts = [f"{confidence} match (rank #{rank})."]
    parts.append(f"Land type: {land_str}.")
    parts.append(f"Vegetation is {ndvi_str}; {water_str} detected.")

    # Intent-specific reasoning
    rules = score_breakdown.get("rules", {})

    if intent.get("crop_stress") and rules.get("ndvi_stress", 0) > 0:
        parts.append("Low NDVI directly indicates crop stress conditions.")

    if intent.get("water_shortage") and rules.get("water_shortage", 0) > 0:
        parts.append("Water availability is below critical threshold, consistent with drought/shortage query.")

    if intent.get("vegetation_health") and rules.get("ndvi_health", 0) > 0:
        parts.append("High NDVI confirms healthy, active vegetation cover.")

    if rules.get("land_type_match", 0) > 0:
        parts.append(f"Land classification ({land_str}) matches the query target.")

    sim = score_breakdown.get("similarity", 0)
    if sim > 0.75:
        parts.append("Semantic embedding similarity is very high.")
    elif sim > 0.55:
        parts.append("Good embedding-level semantic alignment with query.")

    if lat is not None and lon is not None:
        parts.append(f"Location: ({lat:.3f}°, {lon:.3f}°).")

    return " ".join(parts)


def annotate_results(results: list[dict], intent: dict) -> list[dict]:
    """
    Add explanations to all ranked results in-place.
    """
    for i, r in enumerate(results):
        r["explanation"] = generate_explanation(
            tile_id=r["tile_id"],
            metadata=r["metadata"],
            score_breakdown=r["score_breakdown"],
            intent=intent,
            rank=i + 1,
        )
    return results