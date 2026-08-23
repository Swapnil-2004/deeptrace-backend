"""
================================================================
DeepTrace — detector/metadata.py
EXIF metadata and file-level analysis.

What this file does:
1. Reads EXIF data from image files
2. Checks for AI software signatures
3. Checks for missing camera data (AI images have none)
4. Analyses PNG chunks for generation markers
5. Returns meta_score and human-readable flags
================================================================
"""

import io
import struct
from PIL import Image
from PIL.ExifTags import TAGS
from typing import Dict, List, Tuple


# ── Known AI software signatures in metadata ─────────────────
AI_SOFTWARE_SIGNATURES = [
    "stable diffusion", "midjourney", "dall-e", "dalle",
    "firefly", "imagen", "sdxl", "flux", "novai",
    "comfyui", "automatic1111", "invoke", "diffusers",
    "generative", "ai generated", "artificial intelligence",
    "neural network", "gan", "vae"
]

# ── Camera maker strings — real cameras always have these ─────
REAL_CAMERA_MAKERS = [
    "canon", "nikon", "sony", "fujifilm", "olympus",
    "panasonic", "leica", "hasselblad", "pentax", "apple",
    "samsung", "google", "xiaomi", "huawei", "oneplus",
    "nokia", "lg", "motorola", "oppo", "vivo"
]


def analyse_metadata(image_bytes: bytes,
                     filename: str = "") -> Dict:
    """
    Main function — analyse image metadata for AI signatures.

    Args:
        image_bytes: raw image file bytes
        filename:    original filename (used to determine type)

    Returns:
        Dict with:
            meta_score: float 0-1 (1 = definitely AI metadata)
            meta_flags: list of human-readable flag strings
            exif_data:  dict of found EXIF fields (for logging)
    """
    flags     = []
    score     = 0.0
    exif_data = {}

    # Run all checks
    exif_result  = _check_exif(image_bytes, exif_data)
    flags.extend(exif_result["flags"])
    score = max(score, exif_result["score"])

    png_result = _check_png_chunks(image_bytes, filename)
    flags.extend(png_result["flags"])
    score = max(score, png_result["score"])

    software_result = _check_software_signature(exif_data)
    flags.extend(software_result["flags"])
    score = max(score, software_result["score"])

    # Cap at 1.0
    score = min(1.0, score)

    if flags:
        print(f"[Meta] Score: {score:.3f} | Flags: {len(flags)}")
        for f in flags:
            print(f"  {f}")
    else:
        print(f"[Meta] Score: {score:.3f} | No suspicious metadata")

    return {
        "meta_score" : score,
        "meta_flags" : flags,
        "exif_data"  : exif_data
    }


def _check_exif(image_bytes: bytes,
                exif_data: dict) -> Dict:
    """
    Check EXIF data for camera info and AI signatures.

    Real photos ALWAYS have:
    - Camera make and model
    - Datetime
    - Some camera settings (ISO, aperture, etc.)

    AI images have:
    - No camera make/model
    - No datetime or generic datetime
    - No camera settings
    """
    flags = []
    score = 0.0

    try:
        img = Image.open(io.BytesIO(image_bytes))

        # Try to get EXIF data
        exif_raw = img._getexif() if hasattr(img, '_getexif') else None

        if exif_raw is None:
            # No EXIF at all — common for AI images and PNGs
            if image_bytes[:3] == b'\xff\xd8\xff':
                # It's a JPEG — real JPEGs always have EXIF
                flags.append("No EXIF data in JPEG — real camera photos always contain EXIF")
                score = max(score, 0.4)
                print("[Meta] No EXIF in JPEG")
            return {"flags": flags, "score": score}

        # Parse EXIF tags
        for tag_id, value in exif_raw.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="ignore").strip("\x00")
                except Exception:
                    value = str(value)
            exif_data[tag_name] = str(value)

        # Check for camera make
        make  = exif_data.get("Make",  "").lower().strip()
        model = exif_data.get("Model", "").lower().strip()

        if not make and not model:
            flags.append("No EXIF camera data found — AI generated images lack camera metadata")
            score = max(score, 0.35)
        else:
            # Check if it is a known real camera
            is_real_camera = any(cam in make or cam in model
                                 for cam in REAL_CAMERA_MAKERS)
            if is_real_camera:
                print(f"[Meta] Real camera detected: {make} {model}")
            else:
                print(f"[Meta] Unknown camera: {make} {model}")

        # Check for Software tag — may contain AI tool name
        software = exif_data.get("Software", "").lower()
        if software:
            exif_data["Software"] = software

    except Exception as e:
        print(f"[Meta] EXIF read error: {e}")

    return {"flags": flags, "score": score}


def _check_png_chunks(image_bytes: bytes,
                      filename: str) -> Dict:
    """
    Check PNG file chunks for AI generation markers.

    Many AI tools embed metadata in PNG text chunks:
    - ComfyUI embeds workflow JSON
    - Automatic1111 embeds generation parameters
    - MidJourney embeds job ID
    - Stable Diffusion embeds prompt and model info
    """
    flags = []
    score = 0.0

    # Only check PNG files
    is_png = (image_bytes[:8] == b'\x89PNG\r\n\x1a\n' or
              filename.lower().endswith(".png"))

    if not is_png:
        return {"flags": flags, "score": score}

    try:
        # Parse PNG chunks manually
        offset    = 8   # Skip PNG signature
        ai_chunks = []

        while offset < len(image_bytes) - 12:
            # Read chunk: 4 bytes length + 4 bytes type
            chunk_len  = struct.unpack(">I", image_bytes[offset:offset+4])[0]
            chunk_type = image_bytes[offset+4:offset+8].decode("ascii",
                                                                errors="ignore")
            chunk_data = image_bytes[offset+8:offset+8+chunk_len]

            # tEXt and iTXt chunks contain text metadata
            if chunk_type in ("tEXt", "iTXt", "zTXt"):
                try:
                    text = chunk_data.decode("utf-8", errors="ignore").lower()

                    # Check for AI signatures in text
                    for sig in AI_SOFTWARE_SIGNATURES:
                        if sig in text:
                            ai_chunks.append(sig)
                            break

                    # Check for common AI generation keywords
                    ai_keywords = [
                        "steps:", "sampler:", "cfg scale:", "seed:",
                        "model hash:", "clip skip:", "negative prompt:",
                        "controlnet", "lora", "workflow"
                    ]
                    for kw in ai_keywords:
                        if kw in text:
                            ai_chunks.append(f"AI param: {kw}")
                            break

                except Exception:
                    pass

            # Move to next chunk (data + CRC)
            offset += 12 + chunk_len

            if chunk_len > len(image_bytes):
                break

        if ai_chunks:
            unique_sigs = list(set(ai_chunks))[:3]  # Max 3 flags
            flags.append(f"PNG generation metadata detected: {', '.join(unique_sigs)}")
            score = max(score, 0.7)

    except Exception as e:
        print(f"[Meta] PNG chunk error: {e}")

    return {"flags": flags, "score": score}


def _check_software_signature(exif_data: dict) -> Dict:
    """
    Check Software EXIF field for known AI tool names.
    """
    flags = []
    score = 0.0

    software = exif_data.get("Software", "").lower()
    comment  = exif_data.get("UserComment", "").lower()
    artist   = exif_data.get("Artist", "").lower()

    all_text = f"{software} {comment} {artist}"

    for sig in AI_SOFTWARE_SIGNATURES:
        if sig in all_text:
            flags.append(f"AI software signature in metadata: '{sig}'")
            score = 0.9
            break

    return {"flags": flags, "score": score}
