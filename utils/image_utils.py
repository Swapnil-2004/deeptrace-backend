"""
================================================================
DeepTrace — utils/image_utils.py
Image preprocessing utilities.
Handles decoding, resizing, normalizing images for models,
and encoding results back to base64 for the frontend.
================================================================
"""

import io
import base64
import numpy as np
from PIL import Image


# ── Model input size ─────────────────────────────────────────
MODEL_INPUT_SIZE = (224, 224)

# ── ImageNet normalization values ─────────────────────────────
# Used by EfficientNet and ViT — both pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def decode_image_bytes(image_bytes: bytes) -> Image.Image:
    """
    Decode raw image bytes into a PIL Image.
    Converts to RGB — removes alpha channel if PNG has transparency.

    Args:
        image_bytes: raw bytes from uploaded file

    Returns:
        PIL Image in RGB mode
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Could not decode image: {str(e)}")


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array.
    Output shape: (H, W, 3) — uint8 values 0-255

    Args:
        img: PIL Image in RGB mode

    Returns:
        numpy array shape (H, W, 3)
    """
    return np.array(img)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    Input: (H, W, 3) uint8 or float32

    Args:
        arr: numpy array

    Returns:
        PIL Image
    """
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def resize_for_model(img: Image.Image,
                     size: tuple = MODEL_INPUT_SIZE) -> Image.Image:
    """
    Resize image to model input size using high quality resampling.

    Args:
        img:  PIL Image
        size: target (width, height) — default (224, 224)

    Returns:
        Resized PIL Image
    """
    return img.resize(size, Image.LANCZOS)


def normalize_for_model(img: Image.Image) -> np.ndarray:
    """
    Prepare image for model inference:
    1. Resize to 224x224
    2. Convert to float32 in range [0, 1]
    3. Apply ImageNet normalization
    4. Add batch dimension → shape (1, 3, 224, 224)

    Args:
        img: PIL Image (any size)

    Returns:
        numpy array shape (1, 3, 224, 224) — ready for PyTorch
    """
    # Resize
    img_resized = resize_for_model(img)

    # To numpy float32 in [0, 1]
    arr = np.array(img_resized, dtype=np.float32) / 255.0

    # Apply ImageNet normalization
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    arr  = (arr - mean) / std

    # HWC → CHW → NCHW (add batch dim)
    arr = arr.transpose(2, 0, 1)          # (3, 224, 224)
    arr = np.expand_dims(arr, axis=0)     # (1, 3, 224, 224)

    return arr


def pil_to_tensor(img: Image.Image):
    """
    Convert PIL Image directly to PyTorch tensor.
    Shape: (1, 3, 224, 224) — normalized, on CPU.

    Args:
        img: PIL Image

    Returns:
        torch.Tensor shape (1, 3, 224, 224)
    """
    import torch
    arr = normalize_for_model(img)
    return torch.from_numpy(arr).float()


def numpy_to_base64(arr: np.ndarray, format: str = "PNG") -> str:
    """
    Convert numpy array (image) to base64 string for frontend.
    Frontend can use this directly as <img src="...">

    Args:
        arr:    numpy array (H, W, 3) uint8
        format: image format — "PNG" or "JPEG"

    Returns:
        base64 data URL string like "data:image/png;base64,..."
    """
    img = numpy_to_pil(arr)
    return pil_to_base64(img, format)


def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """
    Convert PIL Image to base64 data URL string.

    Args:
        img:    PIL Image
        format: "PNG" or "JPEG"

    Returns:
        base64 data URL string
    """
    buffer    = io.BytesIO()
    mime_type = "image/png" if format.upper() == "PNG" else "image/jpeg"
    img.save(buffer, format=format, optimize=True)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def get_image_info(img: Image.Image) -> dict:
    """
    Get basic information about an image.

    Args:
        img: PIL Image

    Returns:
        dict with width, height, mode, aspect_ratio
    """
    w, h = img.size
    return {
        "width"        : w,
        "height"       : h,
        "mode"         : img.mode,
        "aspect_ratio" : round(w / h, 3)
    }


def crop_with_padding(img: Image.Image,
                      bbox: tuple,
                      padding: float = 0.25) -> Image.Image:
    """
    Crop a region from image with percentage padding around it.
    Used to crop face region with some context around it.

    Args:
        img:     PIL Image
        bbox:    (x_min, y_min, x_max, y_max) in pixel coordinates
        padding: fraction of bbox size to add as padding (default 25%)

    Returns:
        Cropped PIL Image
    """
    w, h = img.size
    x1, y1, x2, y2 = bbox

    # Calculate padding in pixels
    box_w   = x2 - x1
    box_h   = y2 - y1
    pad_x   = int(box_w * padding)
    pad_y   = int(box_h * padding)

    # Apply padding and clamp to image boundaries
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return img.crop((x1, y1, x2, y2))


def blend_heatmap(original: np.ndarray,
                  heatmap:  np.ndarray,
                  alpha:    float = 0.4) -> np.ndarray:
    """
    Blend a heatmap overlay onto the original image.
    Used for Grad-CAM visualization.

    Args:
        original: original image (H, W, 3) uint8
        heatmap:  heatmap (H, W, 3) uint8 — jet colormap applied
        alpha:    heatmap opacity (0=invisible, 1=fully opaque)

    Returns:
        Blended image (H, W, 3) uint8
    """
    original_f = original.astype(np.float32)
    heatmap_f  = heatmap.astype(np.float32)

    # Resize heatmap to match original if needed
    if original.shape[:2] != heatmap.shape[:2]:
        heatmap_pil = Image.fromarray(heatmap)
        heatmap_pil = heatmap_pil.resize(
            (original.shape[1], original.shape[0]),
            Image.LANCZOS
        )
        heatmap_f = np.array(heatmap_pil, dtype=np.float32)

    blended = (1 - alpha) * original_f + alpha * heatmap_f
    return blended.clip(0, 255).astype(np.uint8)
