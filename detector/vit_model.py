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

import os
if os.environ.get("DEEPTRACE_BACKEND", "").strip().lower() != "onnx":
    import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple

# ── Model ID ──────────────────────────────────────────────────
VIT_MODEL_ID = "Organika/sdxl-detector"

# ── Global model state ────────────────────────────────────────
_pipeline = None
_device   = None

# -- ONNX backend (Phase 3) ------------------------------------
# Only active when the environment variable DEEPTRACE_BACKEND=onnx is set.
# Without it the torch/HuggingFace code below runs exactly as before
# (Colab, main).
import os
USE_ONNX      = os.environ.get("DEEPTRACE_BACKEND", "").strip().lower() == "onnx"
ONNX_PATH     = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models", "vit_fp16w.onnx"
)
_onnx_session = None
FAKE_INDEX    = 0   # Organika/sdxl-detector id2label: 0 = artificial (fake)


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

    if USE_ONNX:
        return _load_onnx()

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
    if USE_ONNX:
        return _predict_onnx(img)

    if _pipeline is None:
        print("[ViT] Not loaded yet — loading now (lazy load)...")
        if not load_vit():
            print("[WARNING] ViT lazy load failed — returning 0.5")
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


def unload_vit() -> None:
    """
    Free this model's memory after use, to stay within the
    Render free tier's 512MB RAM limit. No accuracy impact --
    the pipeline reloads fresh (same HuggingFace model) next
    time predict() is called.
    """
    global _pipeline
    if USE_ONNX:
        _unload_onnx()
        return
    _pipeline = None

    import gc
    gc.collect()

    # On Linux (Render), ask glibc to actually return the memory gc just
    # freed back to the OS, instead of glibc quietly keeping it reserved
    # for reuse within this process. No effect on Windows/Mac (dev
    # machine) or non-glibc Linux - silently no-ops there via try/except.
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def is_loaded() -> bool:
    """Check if model is loaded and ready."""
    if USE_ONNX:
        return _onnx_session is not None
    return _pipeline is not None


# -- ONNX backend (Phase 3) ------------------------------------
# Used only when DEEPTRACE_BACKEND=onnx. The HuggingFace code above is unchanged.
# Preprocessing is the exact numpy emulation of the torch uint8 bicubic
# resize (detector/vit_preproc.py), so no torch/transformers is needed.
def _load_onnx() -> bool:
    global _onnx_session
    if _onnx_session is not None:
        return True
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        if os.environ.get("DEEPTRACE_ORT_LEAN", "1") != "0":
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            so.enable_cpu_mem_arena = False
            so.enable_mem_pattern = False
        _onnx_session = ort.InferenceSession(
            ONNX_PATH, so, providers=["CPUExecutionProvider"]
        )
        print(f"[ViT] ONNX session loaded: {ONNX_PATH}")
        return True
    except Exception as e:
        print(f"[ERROR] ViT ONNX loading failed: {e}")
        import traceback
        traceback.print_exc()
        _onnx_session = None
        return False


def _predict_onnx(img: Image.Image) -> float:
    if _onnx_session is None:
        print("[ViT] ONNX not loaded yet - loading now (lazy load)...")
        if not _load_onnx():
            print("[WARNING] ViT ONNX lazy load failed -- returning 0.5")
            return 0.5
    try:
        from detector.vit_preproc import resize_u8, to_input
        x = to_input(resize_u8(img))
        inp = _onnx_session.get_inputs()[0]
        out = _onnx_session.get_outputs()[0]
        dt = np.float16 if "float16" in inp.type else np.float32
        x = np.ascontiguousarray(x, dtype=dt)
        logits = _onnx_session.run([out.name], {inp.name: x})[0]
        lg = np.asarray(logits[0], dtype=np.float64)
        e = np.exp(lg - lg.max())
        return float(e[FAKE_INDEX] / e.sum())
    except Exception as e:
        print(f"[ERROR] ViT ONNX prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.5


def _unload_onnx() -> None:
    global _onnx_session
    _onnx_session = None
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
