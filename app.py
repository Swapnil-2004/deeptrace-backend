"""
================================================================
DeepTrace — app.py
Main FastAPI application.
Handles all routes, CORS, startup model loading,
and request/response for image and video analysis.
================================================================
"""

import os
import time
import tempfile
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── File size limits ──────────────────────────────────────────
MAX_IMAGE_BYTES = 10 * 1024 * 1024   # 10MB
MAX_VIDEO_BYTES = 50 * 1024 * 1024   # 50MB

# ── Allowed file types ────────────────────────────────────────
ALLOWED_IMAGE_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp"
}
ALLOWED_VIDEO_TYPES = {
    "video/mp4", "video/avi", "video/quicktime",
    "video/x-msvideo", "video/x-matroska"
}
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


# ================================================================
# LIFESPAN — runs on startup and shutdown
# Loads all models into memory once when server starts.
# This way every request is fast — no model loading per request.
# ================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    print("=" * 60)
    print("DeepTrace Backend Starting...")
    print("=" * 60)

    try:
        from detector.pipeline import load_all_models
        load_all_models()
        print("[OK] All models loaded successfully")
    except Exception as e:
        print(f"[WARNING] Model loading failed: {e}")
        print("[INFO] Server will start but detections may fail")
        traceback.print_exc()

    print("=" * 60)
    print("DeepTrace Backend Ready!")
    print("=" * 60)

    yield  # Server runs here

    print("DeepTrace Backend Shutting down...")


# ================================================================
# CREATE APP
# ================================================================
app = FastAPI(
    title       = "DeepTrace API",
    description = "AI & Deepfake Detection using Boundary-Topology Analysis",
    version     = "1.0.0",
    lifespan    = lifespan
)


# ================================================================
# CORS — allows frontend to call this backend
# Without this, the browser blocks all requests.
# ================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins       = [
        "http://localhost:3000",      # VS Code Live Server
        "http://127.0.0.1:3000",
        "http://localhost:5500",      # Another common Live Server port
        "http://127.0.0.1:5500",
    ],
    # Matches any Vercel deployment URL, including preview-branch URLs
    # (e.g. https://deeptrace-frontend.vercel.app,
    #  https://deeptrace-frontend-git-main-yourname.vercel.app)
    allow_origin_regex = r"https://.*\.vercel\.app",
    allow_credentials   = True,
    allow_methods       = ["*"],
    allow_headers       = ["*"],
)


# ================================================================
# HELPER FUNCTIONS
# ================================================================
def validate_image_file(file: UploadFile, content: bytes) -> None:
    """Validate image file type and size. Raises HTTPException if invalid."""
    # Check size
    if len(content) > MAX_IMAGE_BYTES:
        size_mb = len(content) / 1024 / 1024
        raise HTTPException(
            status_code = 413,
            detail      = f"File too large ({size_mb:.1f}MB). Maximum is 10MB."
        )
    # Check extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_IMAGE_EXT:
        raise HTTPException(
            status_code = 400,
            detail      = f"Invalid file type '{ext}'. Accepted: {', '.join(ALLOWED_IMAGE_EXT)}"
        )


def validate_video_file(file: UploadFile, content: bytes) -> None:
    """Validate video file type and size. Raises HTTPException if invalid."""
    # Check size
    if len(content) > MAX_VIDEO_BYTES:
        size_mb = len(content) / 1024 / 1024
        raise HTTPException(
            status_code = 413,
            detail      = f"File too large ({size_mb:.1f}MB). Maximum is 50MB."
        )
    # Check extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(
            status_code = 400,
            detail      = f"Invalid file type '{ext}'. Accepted: {', '.join(ALLOWED_VIDEO_EXT)}"
        )


# ================================================================
# ROUTES
# ================================================================

# ── Health check ─────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """
    Quick health check endpoint.
    Frontend calls this to verify backend is alive.
    """
    import torch
    try:
        from detector.pipeline import models_loaded
        loaded = models_loaded()
    except Exception:
        loaded = False

    return {
        "status"        : "ok",
        "gpu"           : torch.cuda.is_available(),
        "gpu_name"      : torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models_loaded" : loaded,
        "version"       : "1.0.0"
    }


# ── Root ─────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name"       : "DeepTrace API",
        "version"    : "1.0.0",
        "status"     : "running",
        "endpoints"  : ["/health", "/analyse/image", "/analyse/video"]
    }


# ── Analyse Image ─────────────────────────────────────────────
@app.post("/analyse/image")
async def analyse_image(file: UploadFile = File(...)):
    """
    Analyse an image for AI generation or deepfake manipulation.

    Accepts: JPG, JPEG, PNG, WEBP (max 10MB)
    Returns: prediction, confidence, heatmap, signal scores
    """
    start_time = time.time()

    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

    # Validate
    validate_image_file(file, content)

    # Run detection pipeline
    try:
        from detector.pipeline import analyse_image_bytes
        result = analyse_image_bytes(content, file.filename or "image.jpg")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code = 500,
            detail      = f"Analysis failed: {str(e)}"
        )

    # Add processing time
    result["processing_time_ms"] = int((time.time() - start_time) * 1000)
    result["filename"]           = file.filename

    return JSONResponse(content=result)


# ── Analyse Video ─────────────────────────────────────────────
@app.post("/analyse/video")
async def analyse_video(file: UploadFile = File(...)):
    """
    Analyse a video for deepfake manipulation.
    Uses temporal BTD (Boundary-Topology Detector) across frames.

    Accepts: MP4, AVI, MOV, MKV (max 50MB)
    Returns: prediction, confidence, per-frame scores, temporal BTD
    """
    start_time = time.time()

    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

    # Validate
    validate_video_file(file, content)

    # Save to temp file — OpenCV needs a file path not bytes
    ext      = os.path.splitext(file.filename or "video.mp4")[1].lower()
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete = False,
            suffix = ext,
            dir    = tempfile.gettempdir()
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        from detector.pipeline import analyse_video_file
        result = analyse_video_file(tmp_path, file.filename or "video.mp4")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code = 500,
            detail      = f"Video analysis failed: {str(e)}"
        )
    finally:
        # Always clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # Add processing time
    result["processing_time_ms"] = int((time.time() - start_time) * 1000)
    result["filename"]           = file.filename

    return JSONResponse(content=result)


# ================================================================
# RUN — for local development
# Command: python app.py
# Or:      uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# ================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host     = "0.0.0.0",
        port     = 8000,
        reload   = True,    # Auto-reload when code changes
        log_level= "info"
    )
