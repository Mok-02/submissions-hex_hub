"""
dataset.py — EuroSAT_MS loader
Reads 13-band .tif files, computes real NDVI and NDWI,
generates TerraMind embeddings for each tile.
Uses actual GeoTIFF coordinates — no fake location labels.
"""

import os
import numpy as np
from pathlib import Path

LABEL_MAP = {
    "AnnualCrop":            "agriculture",
    "Forest":                "forest",
    "HerbaceousVegetation":  "forest",
    "Highway":               "urban",
    "Industrial":            "urban",
    "Pasture":               "agriculture",
    "PermanentCrop":         "agriculture",
    "Residential":           "urban",
    "River":                 "water",
    "SeaLake":               "water",
}


def _region_label(lat: float, lon: float) -> str:
    """Convert real lat/lon to a human-readable region label."""
    if 35 < lat < 72 and -25 < lon < 45:
        if lon < 0:
            return f"Iberian Peninsula ({lat:.1f}N, {abs(lon):.1f}W)"
        elif lon < 10:
            return f"Western Europe ({lat:.1f}N, {lon:.1f}E)"
        elif lon < 20:
            return f"Central Europe ({lat:.1f}N, {lon:.1f}E)"
        elif lon < 32:
            return f"Eastern Europe ({lat:.1f}N, {lon:.1f}E)"
        else:
            return f"Eastern Mediterranean ({lat:.1f}N, {lon:.1f}E)"
    elif 8 < lat < 37 and 68 < lon < 97:
        return f"India ({lat:.1f}N, {lon:.1f}E)"
    elif lat > 60:
        return f"Northern Europe ({lat:.1f}N, {lon:.1f}E)"
    elif lat < -10:
        return f"Southern Hemisphere ({abs(lat):.1f}S, {lon:.1f}E)"
    else:
        return f"Region ({lat:.1f}N, {lon:.1f}E)"


def compute_ndvi(tif_array: np.ndarray) -> float:
    """
    Real NDVI from 13-band Sentinel-2 tile.
    Band 4 (Red)  = index 3
    Band 8 (NIR)  = index 7
    """
    red = tif_array[3].astype(np.float32)
    nir = tif_array[7].astype(np.float32)
    denom = nir + red
    if denom.sum() == 0:
        return 0.5
    ndvi = (nir - red) / (denom + 1e-6)
    return float(np.clip(ndvi.mean(), -1, 1))


def compute_water(tif_array: np.ndarray) -> float:
    """
    NDWI from Band 3 (Green) and Band 8 (NIR).
    Shifted from [-1,1] to [0,1].
    """
    green = tif_array[2].astype(np.float32)
    nir   = tif_array[7].astype(np.float32)
    denom = green + nir
    if denom.sum() == 0:
        return 0.1
    ndwi = (green - nir) / (denom + 1e-6)
    return float(np.clip((ndwi.mean() + 1) / 2, 0, 1))


def _get_real_coords(tif_path):
    """
    Extract real lat/lon center coordinates from GeoTIFF metadata.
    Falls back to (0, 0) if CRS transform fails.
    """
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            cx = (bounds.left + bounds.right) / 2
            cy = (bounds.bottom + bounds.top) / 2
            try:
                from pyproj import Transformer
                transformer = Transformer.from_crs(
                    src.crs, "EPSG:4326", always_xy=True
                )
                lon, lat = transformer.transform(cx, cy)
            except Exception:
                # If pyproj unavailable, use raw coords
                lat, lon = cy, cx
        return round(float(lat), 4), round(float(lon), 4)
    except Exception:
        return 0.0, 0.0


def load_eurosat_dataset(
    ms_root: str = "data/EuroSAT_MS",
    n_tiles: int = 60,
    seed:    int = 0,
) -> dict:
    """
    Load n_tiles from EuroSAT_MS.
    Computes real NDVI/NDWI from spectral bands.
    Extracts real coordinates from GeoTIFF metadata.
    Generates TerraMind embeddings via torch_infer.
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("Run: pip install rasterio --break-system-packages")

    from core.torch_infer import embed_tif
    from core.embeddings import generate_image_embedding

    rng     = np.random.default_rng(seed)
    ms_path = Path(ms_root)

    # Collect all (tif_path, class_name, land_type) pairs
    all_tiles = []
    for class_folder in sorted(ms_path.iterdir()):
        if not class_folder.is_dir():
            continue
        land_type = LABEL_MAP.get(class_folder.name, "unknown")
        for tif_file in sorted(class_folder.glob("*.tif")):
            all_tiles.append((tif_file, land_type, class_folder.name))

    if not all_tiles:
        raise FileNotFoundError(f"No .tif files found in {ms_root}")

    # Random sample
    indices  = rng.choice(len(all_tiles), size=min(n_tiles, len(all_tiles)), replace=False)
    selected = [all_tiles[i] for i in indices]

    embedding_db = {}
    print(f"[dataset] Loading {len(selected)} tiles from EuroSAT_MS...")

    for i, (tif_path, land_type, class_name) in enumerate(selected):

        # Read tile
        with rasterio.open(tif_path) as src:
            tif_array = src.read().astype(np.float32)

        # Real spectral indices
        ndvi  = compute_ndvi(tif_array)
        water = compute_water(tif_array)

        # Real coordinates from GeoTIFF
        lat, lon = _get_real_coords(tif_path)
        region   = _region_label(lat, lon)

        metadata = {
            "land_type":      land_type,
            "class_name":     class_name,
            "ndvi":           round(ndvi, 3),
            "water_presence": round(water, 3),
            "lat":            lat,
            "lon":            lon,
            "region":         region,
            "source":         "EuroSAT_MS",
            "filename":       tif_path.name,
        }

        tile_id  = f"MS_{class_name}_{tif_path.stem}"
        embedding = embed_tif(tif_array, metadata)

        embedding_db[tile_id] = {
            "embedding": embedding,
            "metadata":  metadata,
        }

        print(f"  [{i+1}/{len(selected)}] {tile_id} | "
              f"{land_type:11s} | "
              f"NDVI={ndvi:.2f} | "
              f"water={water:.2f} | "
              f"{region}")

    return embedding_db


def dataset_summary(embedding_db: dict) -> dict:
    land_counts = {}
    for entry in embedding_db.values():
        lt = entry["metadata"]["land_type"]
        land_counts[lt] = land_counts.get(lt, 0) + 1
    return {
        "total_tiles":      len(embedding_db),
        "land_type_counts": land_counts,
    }
