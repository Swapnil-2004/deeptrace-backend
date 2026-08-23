"""
================================================================
DeepTrace — detector/vit_model.py
ViT (Vision Transformer) AI image detector.

This is Model 2 — Organika/sdxl-detector from HuggingFace.
Best for detecting AI-generated images:
MidJourney, Stable Diffusion, DALL-E, SDXL, Flux.

NOT used for face-swap deepfakes — that is EfficientNet's job.
Together they cover both major categories of fake media.
================================================================
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple

# ── Model ID ──────────────────────────────────────────────────
VIT_MODEL_ID = "Organika/sdxl-detector"

# ── Global model state ────────────────────────────────────────
_pipeline = None
_device   = None


def _get_device() -> str:
    """Get best available device string for HuggingFace pipeline."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_vit() -> bool:
    """
    Load ViT model from HuggingFace.
    Downloads automatically on first run (~300MB).
    Cached locally after first download.

    Returns:
        True if loaded successfully, False otherwise
    """
    global _pipeline, _device

    _device = _get_device()
    print(f"[ViT] Using device: {_device}")
    print(f"[ViT] Loading model: {VIT_MODEL_ID}")
    print("[ViT] First run will download ~300MB — please wait...")

    try:
        from transformers import pipeline

        _pipeline = pipeline(
            "image-classification",
            model     = VIT_MODEL_ID,
            device    = 0 if _device == "cuda" else -1,
        )

        print(f"[OK] ViT model loaded: {VIT_MODEL_ID}")
        return True

    except Exception as e:
        print(f"[ERROR] ViT loading failed: {e}")
        import traceback
        traceback.print_exc()
        _pipeline = None
        return False


def predict(img: Image.Image) -> float:
    """
    Run ViT prediction on an image.

    The model returns labels like:
    - "artificial" or "AI-generated" → fake
    - "real" or "human" → real

    Args:
        img: PIL Image in RGB mode

    Returns:
        fake_score: float 0-1 (1 = definitely AI generated)
    """
    if _pipeline is None:
        print("[WARNING] ViT not loaded — returning 0.5")
        return 0.5

    try:
        # Run inference
        results = _pipeline(img)

        # results is a list of dicts like:
        # [{"label": "artificial", "score": 0.92},
        #  {"label": "real",       "score": 0.08}]

        fake_score = _extract_fake_score(results)
        return fake_score

    except Exception as e:
        print(f"[ERROR] ViT prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.5


def _extract_fake_score(results: list) -> float:
    """
    Extract fake probability from ViT pipeline results.

    The Organika/sdxl-detector model uses these labels:
    - "artificial" = AI generated
    - "real"       = real image

    Handles variations in label naming across model versions.

    Args:
        results: list of {"label": str, "score": float}

    Returns:
        fake_score: float 0-1
    """
    if not results:
        return 0.5

    # Keywords that indicate AI/fake
    FAKE_KEYWORDS = {
        "artificial", "ai", "generated", "fake",
        "synthetic", "aigc", "sdxl", "diffusion"
    }

    # Keywords that indicate real
    REAL_KEYWORDS = {
        "real", "human", "authentic", "natural", "photo"
    }

    for result in results:
        label = result.get("label", "").lower().strip()
        score = float(result.get("score", 0.5))

        # Check if this is a fake label
        if any(kw in label for kw in FAKE_KEYWORDS):
            return score

        # Check if this is a real label
        if any(kw in label for kw in REAL_KEYWORDS):
            return 1.0 - score

    # Fallback — use first result score
    return float(results[0].get("score", 0.5))


def is_loaded() -> bool:
    """Check if model is loaded and ready."""
    return _pipeline is not None
