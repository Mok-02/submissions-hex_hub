"""
pipeline.py — Satellite Intelligence System (SIS)
"ChatGPT for a satellite" — ask a question, get a direct answer.
TerraMind processes each tile independently, downlinks only the answer.
"""

import os
import time
import json
import pickle
from core.nlp_parser import parse_query, explain_intent
from data.dataset import load_eurosat_dataset, dataset_summary

CLASS_RULES = {
    "flood_risk":         lambda m: m["water_presence"] > 0.55 and m["ndvi"] < 0.3,
    "crop_stress":        lambda m: m["ndvi"] < 0.25 and m["land_type"] == "agriculture",
    "healthy_vegetation": lambda m: m["ndvi"] > 0.6 and m["land_type"] in ["forest", "agriculture"],
    "urban_expansion":    lambda m: m["land_type"] == "urban" and m["ndvi"] < 0.4,
    "water_body":         lambda m: m["land_type"] == "water",
    "agriculture":        lambda m: m["land_type"] == "agriculture",
    "water_shortage":     lambda m: m["water_presence"] < 0.3,
    "normal":             lambda m: False,  # ← changed to False — no match on unknown queries
}


INTENT_TO_CLASS = {
    "water_abundance":   "flood_risk",
    "flood_risk":        "flood_risk",
    "crop_stress":       "crop_stress",
    "vegetation_health": "healthy_vegetation",
    "urban_expansion":   "urban_expansion",
    "urban":             "urban_expansion",
    "water_body":        "water_body",
    "forest":            "healthy_vegetation",
    "agriculture":       "agriculture",       # ← new
    "water_shortage":    "water_shortage",    # ← new
}

ANSWER_TEMPLATES = {
    "flood_risk":         "⚠️  FLOOD RISK detected in {n} tile(s).",
    "crop_stress":        "🌾 CROP STRESS detected in {n} tile(s).",
    "healthy_vegetation": "🌿 HEALTHY VEGETATION confirmed in {n} tile(s).",
    "urban_expansion":    "🏙️  URBAN EXPANSION detected in {n} tile(s).",
    "water_body":         "💧 WATER BODIES identified in {n} tile(s).",
    "agriculture":        "🌱 AGRICULTURAL LAND identified in {n} tile(s).",
    "water_shortage":     "🔴 LOW WATER / DROUGHT conditions in {n} tile(s).",
    "normal":             "❓ Query not understood — no specific pattern matched.",  # ← honest fallback
}

class SISPipeline:
    def __init__(self, n_tiles: int = 60, seed: int = 0):
        cache_path = "data/embeddings_cache.pkl"

        if os.path.exists(cache_path):
            print("[SIS] Loading from cache (instant)...")
            with open(cache_path, "rb") as f:
                self.tile_db = pickle.load(f)
            print("[SIS] Cache loaded.")
        else:
            print(f"[SIS] First run — embedding {n_tiles} tiles (one time only)...")
            self.tile_db = load_eurosat_dataset(n_tiles=n_tiles, seed=seed)
            with open(cache_path, "wb") as f:
                pickle.dump(self.tile_db, f)
            print("[SIS] Cache saved — future runs will be instant.")

        self.stats = dataset_summary(self.tile_db)
        print(f"[SIS] Ready: {self.stats}")

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()

        intent         = parse_query(question)
        intent_summary = explain_intent(intent)

        target_class = "normal"
        for intent_key, cls in INTENT_TO_CLASS.items():
            if intent.get(intent_key):
                target_class = cls
                break

        matched_tiles = []
        for tile_id, entry in self.tile_db.items():
            meta    = entry["metadata"]
            rule_fn = CLASS_RULES.get(target_class, CLASS_RULES["normal"])
            if rule_fn(meta):
                matched_tiles.append({
                    "tile_id":    tile_id,
                    "lat":        meta["lat"],
                    "lon":        meta["lon"],
                    "region":     meta["region"],
                    "land_type":  meta["land_type"],
                    "ndvi":       meta["ndvi"],
                    "water":      meta["water_presence"],
                    "class":      target_class,
                    "confidence": _confidence(meta, target_class),
                })

        n          = len(matched_tiles)
        answer_str = ANSWER_TEMPLATES.get(
            target_class, ANSWER_TEMPLATES["normal"]
        ).format(n=n)

        downlink = {
            "question":      question,
            "answer":        answer_str,
            "target_class":  target_class,
            "tiles_matched": n,
            # FIND this in ask():
            "locations": [
            {
                "lat":    t["lat"],
                "lon":    t["lon"],
                "region": t["region"],
                "conf":   t["confidence"],
            }
                for t in matched_tiles[:5]
                ],
        }

        downlink_bytes = len(json.dumps(downlink).encode())
        elapsed        = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "question":       question,
            "intent_summary": intent_summary,
            "answer":         answer_str,
            "matched_tiles":  matched_tiles,
            "downlink":       downlink,
            "meta": {
                "total_tiles":         len(self.tile_db),
                "tiles_matched":       n,
                "downlink_bytes":      downlink_bytes,
                "raw_imagery_mb":      len(self.tile_db) * 700,
                "bandwidth_reduction": f"{round((1 - downlink_bytes / (len(self.tile_db) * 700 * 1024)) * 100, 4)}%",
                "latency_ms":          elapsed,
            },
        }

    def get_stats(self) -> dict:
        return self.stats


def _confidence(meta: dict, target_class: str) -> float:
    ndvi  = meta.get("ndvi", 0.5)
    water = meta.get("water_presence", 0.5)
    if target_class == "flood_risk":
        return round(min(0.99, 0.6 + water * 0.4), 2)
    elif target_class == "crop_stress":
        return round(min(0.99, 0.6 + (0.35 - ndvi)), 2)
    elif target_class == "healthy_vegetation":
        return round(min(0.99, 0.5 + ndvi * 0.5), 2)
    elif target_class == "water_body":
        return round(min(0.99, 0.6 + water * 0.35), 2)
    return 0.75