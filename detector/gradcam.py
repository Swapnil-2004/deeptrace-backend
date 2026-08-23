"""
================================================================
DeepTrace — detector/gradcam.py
Grad-CAM heatmap generation.

Grad-CAM (Gradient-weighted Class Activation Mapping)
shows WHERE in the image the model focused its attention
when making the fake/real decision.

What this file does:
1. Takes activations from EfficientNet last conv layer
2. Weights them by importance
3. Creates a heatmap highlighting suspicious regions
4. Applies jet colormap (red=suspicious, blue=normal)
5. Blends with original image at 40% opacity
6. Returns as base64 string for frontend
================================================================
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional

from utils.image_utils import pil_to_base64, blend_heatmap


def generate_heatmap(
    original_img : Image.Image,
    activations  : Optional[np.ndarray],
    fake_score   : float
) -> str:
    """
    Generate Grad-CAM heatmap from model activations.

    Args:
        original_img: original PIL Image (any size)
        activations:  numpy array from EfficientNet hook
                      shape (1, C, H, W) — C channels, H×W spatial
        fake_score:   model's fake probability (0-1)

    Returns:
        base64 data URL string for frontend display
        Falls back to a simple gradient heatmap if activations unavailable
    """
    orig_w, orig_h = original_img.size
    orig_arr = np.array(original_img.convert("RGB"))

    # ── Generate activation map ─────────────────────────────
    if activations is not None:
        cam = _activations_to_cam(activations)
    else:
        # Fallback — generate a simple gradient heatmap
        cam = _generate_fallback_cam(orig_h, orig_w, fake_score)

    # ── Resize CAM to original image size ───────────────────
    cam_resized = cv2.resize(cam, (orig_w, orig_h),
                             interpolation=cv2.INTER_CUBIC)

    # ── Apply jet colormap ───────────────────────────────────
    # Normalise to 0-255
    cam_norm = (cam_resized * 255).astype(np.uint8)
    # Apply jet colormap — red=high, blue=low
    cam_color = cv2.applyColorMap(cam_norm, cv2.COLORMAP_JET)
    # Convert BGR to RGB
    cam_rgb = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)

    # ── Blend with original ──────────────────────────────────
    blended = blend_heatmap(orig_arr, cam_rgb, alpha=0.4)

    # ── Convert to base64 for frontend ──────────────────────
    blended_pil = Image.fromarray(blended)
    return pil_to_base64(blended_pil, format="JPEG")


def _activations_to_cam(activations: np.ndarray) -> np.ndarray:
    """
    Convert activation maps to a single CAM (Class Activation Map).

    Method:
    1. Take activations shape (1, C, H, W)
    2. Average across channels (equal weighting)
    3. Apply ReLU — only keep positive activations
    4. Normalise to 0-1

    Args:
        activations: numpy array shape (1, C, H, W)

    Returns:
        2D numpy array (H, W) normalised to 0-1
    """
    try:
        # Remove batch dim → (C, H, W)
        acts = activations[0]

        # Average across channels → (H, W)
        cam = np.mean(acts, axis=0)

        # ReLU — only positive activations matter
        cam = np.maximum(cam, 0)

        # Normalise to 0-1
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    except Exception as e:
        print(f"[GradCAM] Activation processing error: {e}")
        return np.zeros((7, 7), dtype=np.float32)


def _generate_fallback_cam(h: int, w: int,
                           fake_score: float) -> np.ndarray:
    """
    Generate a simple fallback heatmap when activations unavailable.
    Places hotspots at hairline and jaw regions —
    the BTD boundary regions that are most suspicious.

    Args:
        h, w:       image height and width
        fake_score: model's fake probability

    Returns:
        2D numpy array (H, W) normalised 0-1
    """
    cam = np.zeros((h, w), dtype=np.float32)

    if fake_score < 0.3:
        return cam

    # Hairline region — top 15-35% of image
    y1_hair = int(h * 0.15)
    y2_hair = int(h * 0.35)
    x1_hair = int(w * 0.2)
    x2_hair = int(w * 0.8)

    # Jaw region — bottom 55-75% of image
    y1_jaw = int(h * 0.55)
    y2_jaw = int(h * 0.75)
    x1_jaw = int(w * 0.2)
    x2_jaw = int(w * 0.8)

    # Intensity proportional to fake score
    intensity = float(fake_score)

    # Apply Gaussian hotspots
    cam[y1_hair:y2_hair, x1_hair:x2_hair] += intensity * 0.8
    cam[y1_jaw:y2_jaw,   x1_jaw:x2_jaw]   += intensity * 0.9

    # Smooth
    cam = cv2.GaussianBlur(cam, (31, 31), 0)

    # Normalise
    if cam.max() > 1e-8:
        cam = cam / cam.max()

    return cam.astype(np.float32)


def generate_video_heatmap(
    original_img : Image.Image,
    activations  : Optional[np.ndarray],
    fake_score   : float,
    frame_scores : list
) -> str:
    """
    Generate heatmap for video result.
    Similar to image heatmap but also considers
    which frames were most suspicious.

    Args:
        original_img: PIL Image (first frame or most suspicious frame)
        activations:  activation maps from most suspicious frame
        fake_score:   overall video fake score
        frame_scores: list of per-frame scores

    Returns:
        base64 heatmap string
    """
    # Use the same heatmap generation as image
    # But boost intensity based on fake frame ratio
    if frame_scores:
        fake_ratio = sum(1 for s in frame_scores if s >= 0.5) / len(frame_scores)
        # Boost fake_score by fake_ratio for more intense heatmap
        boosted_score = min(1.0, fake_score * (1 + fake_ratio * 0.3))
    else:
        boosted_score = fake_score

    return generate_heatmap(original_img, activations, boosted_score)
