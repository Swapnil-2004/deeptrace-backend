"""
================================================================
DeepTrace — detector/pipeline.py
Main orchestrator — connects all modules in the right order.

This is what app.py calls for every image and video request.

CHANGES IN THIS VERSION:
- Video path now uses effnet_video_model (the dedicated,
  face-crop-trained video model, 90.64% Real / 96.85% Fake on
  held-out test) instead of effnet_model (the image model).
- Video path now explicitly crops faces with padding=0.30 to
  exactly match how the video model was trained (face_utils'
  default is 0.25, which would have been a mismatch).
- Image path (analyse_image_bytes) is COMPLETELY UNCHANGED --
  still uses effnet_model on the full image, exactly as before.
================================================================
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Optional

# ── Module imports ────────────────────────────────────────────
from detector import face_utils
from detector import btd
from detector import effnet_model
from detector import effnet_video_model
from detector import gradcam
from detector import vit_model
from detector import metadata
from detector import ensemble
from detector import temporal_btd
from detector import optical_flow
from utils   import image_utils
from utils   import video_utils

# Padding used when training the video model -- must match exactly
VIDEO_FACE_CROP_PADDING = 0.30


# ================================================================
# MODEL LOADING
# ================================================================
_models_loaded = False
_video_model_loaded = False


def load_all_models() -> None:
    """
    Startup hook — kept for backward compatibility with app.py's lifespan.

    IMPORTANT: This no longer eagerly loads models. On memory-constrained
    hosts (like Render's free 512MB tier), loading all 3 heavy models
    (2x EfficientNet-B4 + ViT transformer) at once during startup both:
    1. Blocks the port from opening in time (Render's health check times out)
    2. Exceeds the 512MB RAM limit immediately

    Instead, each model now lazy-loads itself the first time predict()
    is called on it (see effnet_model.py, effnet_video_model.py,
    vit_model.py). This means a pure-image request only ever loads
    effnet_model + vit_model, not the unused video model -- reducing
    typical peak memory.
    """
    global _models_loaded, _video_model_loaded

    print("[Pipeline] Startup: models will lazy-load on first use "
          "(not loaded eagerly, to fit within memory limits).")

    # Report as "ready" immediately since lazy loading means the app
    # can serve requests right away; individual predict() calls handle
    # loading and will report failures at that point if any occur.
    _models_loaded = True
    _video_model_loaded = True


def models_loaded() -> bool:
    """Check if core (image) models are loaded."""
    return _models_loaded


def video_model_loaded() -> bool:
    """Check if the dedicated video model is loaded."""
    return _video_model_loaded


# ================================================================
# IMAGE ANALYSIS PIPELINE  (UNCHANGED from before)
# ================================================================
def analyse_image_bytes(image_bytes: bytes,
                        filename: str = "image.jpg") -> Dict:
    """
    Full image analysis pipeline.
    Called by app.py POST /analyse/image

    Args:
        image_bytes: raw bytes of uploaded image file
        filename:    original filename

    Returns:
        Complete result dict matching API contract
    """
    print(f"\n{'='*50}")
    print(f"[Pipeline] Analysing image: {filename}")
    print(f"{'='*50}")

    # ── Step 1: Decode image ─────────────────────────────────
    img = image_utils.decode_image_bytes(image_bytes)
    print(f"[Pipeline] Image size: {img.size}")

    # ── Step 2: Face detection + landmarks ───────────────────
    print("[Pipeline] Step 2: Face detection...")
    face_data        = face_utils.detect_and_crop(img)
    face_found       = face_data["face_found"]
    face_crop        = face_data["face_crop"]
    skip_face_signals= not face_found

    print(f"[Pipeline] Face found: {face_found}")

    # ── Step 3: BTD forensic signals (mam's algorithm) ───────
    print("[Pipeline] Step 3: BTD forensic analysis...")
    btd_result = btd.run_all_signals(
        img, face_data, skip_face_signals
    )
    btd_score  = btd_result["btd_score"]
    btd_flags  = btd_result["btd_flags"]

    # ── Step 4: EfficientNet prediction + activations ────────
    print("[Pipeline] Step 4: EfficientNet-B4...")
    # Use full image — model was trained on full Kaggle images,
    # not tight face crops. Tight crops changed input distribution
    # after MediaPipe fix, causing false positives on real photos.
    effnet_score, activations = effnet_model.predict(img)
    print(f"[Pipeline] EfficientNet score: {effnet_score:.3f}")

    # Free EfficientNet from memory before loading ViT -- keeps peak
    # RAM low enough for Render's free 512MB tier. No accuracy impact;
    # activations are already captured above and heatmap uses only
    # that captured array, not the live model.
    effnet_model.unload_effnet()

    # ── Step 5: Grad-CAM heatmap ─────────────────────────────
    print("[Pipeline] Step 5: Generating Grad-CAM heatmap...")
    heatmap_b64 = gradcam.generate_heatmap(
        img, activations, effnet_score
    )

    # ── Step 6: ViT prediction ───────────────────────────────
    print("[Pipeline] Step 6: ViT detector...")
    # Use full image consistently — same reasoning as EfficientNet
    vit_score = vit_model.predict(img)
    print(f"[Pipeline] ViT score: {vit_score:.3f}")

    # Free ViT from memory now that this request's inference is done
    vit_model.unload_vit()

    # ── Step 7: Metadata analysis ────────────────────────────
    print("[Pipeline] Step 7: Metadata analysis...")
    meta_result = metadata.analyse_metadata(image_bytes, filename)
    meta_score  = meta_result["meta_score"]
    meta_flags  = meta_result["meta_flags"]

    # ── Step 8: Ensemble ─────────────────────────────────────
    print("[Pipeline] Step 8: Computing ensemble...")
    ens_result = ensemble.compute_image_ensemble(
        effnet_score = effnet_score,
        vit_score    = vit_score,
        btd_score    = btd_score,
        meta_score   = meta_score,
        face_found   = face_found
    )

    # ── Step 9: Build response ───────────────────────────────
    all_flags = btd_flags + meta_flags

    response = {
        "prediction"     : ens_result["prediction"],
        "confidence"     : ens_result["confidence"],
        "final_score"    : ens_result["final_score"],
        "face_found"     : face_found,
        "heatmap_base64" : heatmap_b64,
        "signals": {
            "effnet_score" : ens_result["individual"]["effnet"],
            "effnet_label" : "Fake" if effnet_score >= 0.5 else "Real",
            "vit_score"    : ens_result["individual"]["vit"],
            "vit_label"    : "Fake" if vit_score >= 0.5 else "Real",
            "btd_score"    : ens_result["individual"]["btd"],
            "btd_flags"    : btd_flags,
            "meta_score"   : ens_result["individual"]["meta"],
            "meta_flags"   : meta_flags,
        },
        "all_flags"      : all_flags,
        "override"       : ens_result["override"],
        "override_reason": ens_result["override_reason"],
    }

    print(f"[Pipeline] DONE: {response['prediction']} "
          f"({response['confidence']}% confident)")
    print(f"{'='*50}\n")

    return response


# ================================================================
# VIDEO ANALYSIS PIPELINE
# ================================================================
def analyse_video_file(video_path: str,
                       filename: str = "video.mp4") -> Dict:
    """
    Full video analysis pipeline.
    Called by app.py POST /analyse/video

    Args:
        video_path: path to temp video file on disk
        filename:   original filename

    Returns:
        Complete result dict matching API contract
    """
    print(f"\n{'='*50}")
    print(f"[Pipeline] Analysing video: {filename}")
    print(f"{'='*50}")

    # ── Step 1: Get video info ───────────────────────────────
    print("[Pipeline] Step 1: Reading video info...")
    video_info = video_utils.get_video_info(video_path)
    print(f"[Pipeline] Video: {video_info['fps']}fps "
          f"{video_info['total_frames']} frames "
          f"{video_info['duration_sec']}s")

    # ── Step 2: Sample frames ────────────────────────────────
    print("[Pipeline] Step 2: Sampling frames...")
    frames, frame_indices = video_utils.sample_frames(
        video_path, max_frames=30
    )
    print(f"[Pipeline] Sampled {len(frames)} frames")

    # ── Step 3: Per-frame analysis ───────────────────────────
    print("[Pipeline] Step 3: Per-frame analysis...")
    frame_scores   = []
    frame_effnet_scores = []
    frame_vit_scores    = []
    all_contours   = []
    all_face_masks = []
    best_frame_img = None
    best_activations = None
    best_score     = 0.0

    for i, frame_bgr in enumerate(frames):
        print(f"  Frame {i+1}/{len(frames)}...", end=" ")

        # Convert BGR to PIL
        frame_pil = video_utils.bgr_to_pil(frame_bgr)

        # Face detection -- padding=0.30 to EXACTLY match how the
        # video model was trained (extract_frames_facecrop_v2.py
        # used CROP_PADDING=0.30). face_utils' own default is 0.25,
        # which would silently mismatch training if not overridden here.
        face_data  = face_utils.detect_and_crop(frame_pil, padding=VIDEO_FACE_CROP_PADDING)
        face_found = face_data["face_found"]
        face_crop  = face_data["face_crop"]

        # Store contour and mask for temporal BTD
        all_contours.append(face_data.get("contour"))
        all_face_masks.append(face_data.get("face_mask"))

        # Video-specific EfficientNet on this frame (NOT the image model --
        # this is the model trained specifically on face-cropped video
        # frames: 90.64% Real / 96.85% Fake on held-out test)
        frame_effnet, frame_acts = effnet_video_model.predict(face_crop)
        frame_effnet_scores.append(frame_effnet)

        # ViT on this frame (still the general-purpose ViT -- not
        # retrained specifically for video in this project)
        frame_vit = vit_model.predict(
            face_crop if face_found else frame_pil
        )
        frame_vit_scores.append(frame_vit)

        # BTD on this frame (lightweight — skip slow signals)
        frame_btd_result = btd.run_all_signals(
            frame_pil, face_data,
            skip_face_signals=not face_found
        )
        frame_btd = frame_btd_result["btd_score"]

        # Frame ensemble (no metadata for video frames)
        frame_ens = ensemble.compute_image_ensemble(
            effnet_score = frame_effnet,
            vit_score    = frame_vit,
            btd_score    = frame_btd,
            meta_score   = 0.0,
            face_found   = face_found
        )
        frame_score = frame_ens["final_score"]
        frame_scores.append(frame_score)

        print(f"score={frame_score:.3f}")

        # Track most suspicious frame for heatmap
        if frame_score > best_score:
            best_score       = frame_score
            best_frame_img   = frame_pil
            best_activations = frame_acts

    print(f"[Pipeline] Frame scores: "
          f"mean={np.mean(frame_scores):.3f} "
          f"max={max(frame_scores):.3f}")

    # Free both models now that all frames are processed -- keeps
    # peak RAM low for Render's free 512MB tier. Not unloaded per-frame
    # since these two models are needed together on every frame;
    # unloading mid-loop would force 30 reloads for no benefit.
    effnet_video_model.unload_effnet_video()
    vit_model.unload_vit()

    # ── Step 4: Temporal BTD (mam's algorithm) ───────────────
    print("[Pipeline] Step 4: Temporal BTD analysis...")
    temporal_result = temporal_btd.run_temporal_btd(
        frames       = frames,
        contours     = all_contours,
        face_masks   = all_face_masks,
        frame_scores = frame_scores
    )

    # ── Step 5: Optical flow analysis ────────────────────────
    print("[Pipeline] Step 5: Optical flow analysis...")
    flow_result = optical_flow.compute_flow_score(
        frames     = frames,
        face_masks = all_face_masks
    )

    # ── Step 6: Metadata analysis on video file ──────────────
    print("[Pipeline] Step 6: Metadata check...")
    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read(65536)  # Read first 64KB only
        meta_result = metadata.analyse_metadata(video_bytes, filename)
        meta_flags  = meta_result["meta_flags"]
    except Exception:
        meta_flags = []

    # ── Step 7: Video ensemble ───────────────────────────────
    print("[Pipeline] Step 7: Video ensemble...")
    video_ens = ensemble.compute_video_ensemble(
        frame_scores   = frame_scores,
        temporal_score = temporal_result["score"],
        flow_score     = flow_result["flow_score"]
    )

    # ── Step 8: Generate heatmap from most suspicious frame ──
    print("[Pipeline] Step 8: Generating heatmap...")
    if best_frame_img is not None:
        heatmap_b64 = gradcam.generate_video_heatmap(
            original_img = best_frame_img,
            activations  = best_activations,
            fake_score   = best_score,
            frame_scores = frame_scores
        )
    else:
        heatmap_b64 = ""

    # ── Step 9: Build response ───────────────────────────────
    all_flags = (temporal_result.get("flags", []) +
                 flow_result.get("flags", []) +
                 meta_flags)

    response = {
        "prediction"      : video_ens["prediction"],
        "confidence"      : video_ens["confidence"],
        "final_score"     : video_ens["final_score"],
        "heatmap_base64"  : heatmap_b64,
        "video_info": {
            "fps"             : video_info["fps"],
            "total_frames"    : video_info["total_frames"],
            "frames_analysed" : len(frames),
            "duration_sec"    : video_info["duration_sec"],
        },
        "per_frame_scores" : [round(s, 4) for s in frame_scores],
        "temporal_btd": {
            "score"                  : temporal_result["score"],
            "boundary_too_stable"    : temporal_result["boundary_too_stable"],
            "flow_mismatch_detected" : temporal_result["flow_mismatch_detected"],
            "flags"                  : temporal_result["flags"],
        },
        "flow_score"       : flow_result["flow_score"],
        "consistency_score": temporal_result.get("consistency_score", 0.0),
        "fake_frame_ratio" : video_ens["fake_frame_ratio"],
        "signals": {
            "effnet_score" : round(float(np.mean(frame_effnet_scores)), 4),
            "effnet_label" : "Fake" if float(np.mean(frame_effnet_scores)) >= 0.5 else "Real",
            "vit_score"    : round(float(np.mean(frame_vit_scores)), 4),
            "vit_label"    : "Fake" if float(np.mean(frame_vit_scores)) >= 0.5 else "Real",
            "btd_score"    : temporal_result["score"],
            "btd_flags"    : temporal_result["flags"],
            "meta_score"   : 0.0,
            "meta_flags"   : meta_flags,
        },
        "all_flags"        : all_flags,
    }

    print(f"[Pipeline] VIDEO DONE: {response['prediction']} "
          f"({response['confidence']}% confident)")
    print(f"{'='*50}\n")

    return response
