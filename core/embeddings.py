"""
embeddings.py — TerraMind embedding layer

generate_image_embedding()  — fake fallback (used if TerraMind unavailable)
generate_query_embedding()  — builds query vector from intent dict
cosine_similarity()         — safe cosine similarity for any array shape
"""

import numpy as np
import json
from pathlib import Path

EMBEDDING_DIM = 768


def _seed_from_id(tile_id: str) -> int:
    return sum(ord(c) for c in tile_id)


def generate_image_embedding(tile_id: str, metadata: dict) -> np.ndarray:
    """
    Fake TerraMind embedding — used only as fallback.
    Semantically informed by metadata so cosine similarity still works.
    """
    rng  = np.random.default_rng(_seed_from_id(tile_id))
    base = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)

    ndvi  = metadata.get("ndvi", 0.5)
    water = metadata.get("water_presence", 0.5)
    land  = metadata.get("land_type", "unknown")

    base[0:64]   += (1.0 - ndvi)  * 3.0
    base[64:128] += (1.0 - water) * 3.0
    base[128:192] += ndvi  * 2.0
    base[192:256] += water * 2.0

    land_offsets = {
        "agriculture": ([256, 320], 2.5),
        "urban":       ([320, 384], 2.5),
        "water":       ([384, 448], 2.5),
        "forest":      ([448, 512], 2.5),
        "barren":      ([512, 576], 2.5),
    }
    if land in land_offsets:
        span, strength = land_offsets[land]
        base[span[0]:span[1]] += strength

    norm = np.linalg.norm(base)
    return base / norm if norm > 0 else base


def generate_query_embedding(intent: dict) -> np.ndarray:
    """
    Build a query vector from structured intent.
    Projects into same semantic subspace as image embeddings.
    """
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    if intent.get("crop_stress"):
        vec[0:64] += 3.0
    if intent.get("water_shortage"):
        vec[64:128] += 3.0
    if intent.get("vegetation_health"):
        vec[128:192] += 3.0
    if intent.get("water_abundance"):
        vec[192:256] += 3.0

    land_map = {
        "agriculture": [256, 320],
        "urban":       [320, 384],
        "water_body":  [384, 448],
        "forest":      [448, 512],
        "barren":      [512, 576],
    }
    for key, span in land_map.items():
        if intent.get(key):
            vec[span[0]:span[1]] += 2.5

    noise = np.random.default_rng(42).standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.05
    vec += noise

    # Ensure same dim as TerraMind embeddings
    if vec.shape[0] < EMBEDDING_DIM:
        vec = np.pad(vec, (0, EMBEDDING_DIM - vec.shape[0]))
    else:
        vec = vec[:EMBEDDING_DIM]

    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Safe cosine similarity — handles any array shape."""
    a = np.array(a).flatten().astype(np.float32)
    b = np.array(b).flatten().astype(np.float32)
    dot = float(np.dot(a, b))
    na  = float(np.linalg.norm(a))
    nb  = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def save_embeddings(db: dict, path: str) -> None:
    serializable = {}
    for tile_id, entry in db.items():
        serializable[tile_id] = {
            "embedding": entry["embedding"].tolist(),
            "metadata":  entry["metadata"],
        }
    with open(path, "w") as f:
        json.dump(serializable, f)


def load_embeddings(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    db = {}
    for tile_id, entry in raw.items():
        db[tile_id] = {
            "embedding": np.array(entry["embedding"], dtype=np.float32),
            "metadata":  entry["metadata"],
        }
    return db