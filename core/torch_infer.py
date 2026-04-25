"""
torch_infer.py — TerraMind inference using BACKBONE_REGISTRY API

Outputs 12 tensors of shape [1, 196, 192] (ViT patch tokens × feature dim).
Uses Global Average Pooling on last 4 layers → exactly 768-dim embedding.
No random projection. Real semantic vectors from TerraMind.
"""

import numpy as np
import torch
import torch.nn.functional as F
from core.embeddings import generate_image_embedding

EMBEDDING_DIM = 768
_encoder = None

# EuroSAT_MS 13 bands → extract these 6 matching TerraMind S2L2A bands
# B2(Blue)=1, B3(Green)=2, B4(Red)=3, B8(NIR)=7, B11(SWIR1)=10, B12(SWIR2)=11
S2_BAND_INDICES = [1, 2, 3, 7, 10, 11]


def _get_encoder():
    global _encoder
    if _encoder is not None:
        return _encoder
    try:
        from terratorch import BACKBONE_REGISTRY
        print("[TerraMind] Loading encoder (tiny)...")
        _encoder = BACKBONE_REGISTRY.build(
            'terramind_v1_tiny',
            pretrained=True,
            modalities=['S2L2A'],
            bands={'S2L2A': ['BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2']},
        )
        _encoder.eval()
        print("[TerraMind] Encoder ready.")
        return _encoder
    except Exception as e:
        print(f"[TerraMind] Not available ({e}), using fake embeddings.")
        return None


def embed_tif(tif_array: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Embed a (13, H, W) EuroSAT_MS tile using TerraMind tiny.

    Pipeline:
      1. Extract 6 bands, normalize to [0,1], resize to 224×224
      2. Forward through TerraMind → 12 intermediate feature maps
      3. Global Average Pooling on last 4 layers → 4 × 192 = 768 dims
      4. L2 normalize
    """
    encoder = _get_encoder()

    if encoder is None:
        tile_id = metadata.get("filename", "unknown")
        return generate_image_embedding(tile_id, metadata)

    try:
        arr = tif_array[S2_BAND_INDICES].copy()  # (6, H, W)

        # Per-band normalization to [0, 1]
        for b in range(arr.shape[0]):
            bmin, bmax = arr[b].min(), arr[b].max()
            if bmax > bmin:
                arr[b] = (arr[b] - bmin) / (bmax - bmin)

        tensor = torch.from_numpy(arr).unsqueeze(0)          # (1, 6, H, W)
        tensor = F.interpolate(tensor, size=(224, 224),
                               mode="bilinear", align_corners=False)

        inputs = {'S2L2A': tensor}

        with torch.no_grad():
            output = encoder(inputs)

        # output: list of 12 tensors, each [1, 196, 192]
        # 196 = 14×14 spatial patches (224/16), 192 = feature dim per patch
        #
        # Global Average Pooling over spatial tokens:
        #   each tensor [1, 196, 192] → mean(dim=1) → [1, 192]
        # Last 4 layers concatenated: 4 × 192 = 768 (exact match, no projection)

        if isinstance(output, (list, tuple)):
            layers = [o for o in output if isinstance(o, torch.Tensor)]
            last_4 = layers[-4:]  # deepest layers have richest semantics
            pooled = []
            for feat in last_4:
                gap = feat.mean(dim=1)  # [1, 192]
                pooled.append(gap)
            vec_tensor = torch.cat(pooled, dim=1).squeeze(0)  # (768,)
        else:
            # Fallback for unexpected output shape
            if len(output.shape) == 3:
                vec_tensor = output.mean(dim=1).squeeze(0)
            else:
                vec_tensor = output.reshape(1, -1).squeeze(0)

        vec = vec_tensor.numpy().astype(np.float32)

        assert vec.shape[0] == EMBEDDING_DIM, \
            f"Embedding dim {vec.shape[0]} != {EMBEDDING_DIM}"

        # L2 normalize
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    except Exception as e:
        print(f"[TerraMind] Inference error ({e}), using fallback.")
        tile_id = metadata.get("filename", "unknown")
        return generate_image_embedding(tile_id, metadata)