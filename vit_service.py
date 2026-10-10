"""
DeepTrace - vit_service.py  (Phase 4)
Tiny service that holds ONLY the ViT (ONNX fp16 weights, lean session).
The API service resizes the image (exact, detector/vit_preproc.py) and sends
the 224x224x3 uint8 picture here; this service returns the fake score.
Run:  uvicorn vit_service:app --host 0.0.0.0 --port 8001
"""
import os
os.environ.setdefault("DEEPTRACE_BACKEND", "onnx")   # never load torch here
import threading
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

EXPECTED_BYTES = 224 * 224 * 3

@asynccontextmanager
async def _lifespan(_app):
    # Load the model BEFORE the service answers anything, so Render holds
    # the waking request until the model is ready (a load inside the first
    # request starves the 0.1 CPU and the health check restarts the instance).
    # Same model, same session settings: scores cannot change.
    from detector import vit_model
    if not vit_model._load_onnx():
        raise RuntimeError("ViT ONNX model could not be loaded at startup")
    vit_model.predict_u8(np.zeros((224, 224, 3), dtype=np.uint8))  # warm-up
    print("[ViT service] model loaded and warmed up", flush=True)
    yield


app = FastAPI(title="DeepTrace ViT service", version="1.0.0", lifespan=_lifespan)
_lock = threading.Lock()   # one prediction at a time (keeps memory low)


def _run(body: bytes) -> float:
    from detector import vit_model
    u8 = np.frombuffer(body, dtype=np.uint8).reshape(224, 224, 3)
    with _lock:
        return vit_model.predict_u8(u8)


@app.get("/health")
def health():
    return {"status": "ok", "service": "vit"}


@app.post("/predict")
async def predict(request: Request):
    body = await request.body()
    if len(body) != EXPECTED_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_BYTES} bytes (224x224x3 uint8), got {len(body)}")
    try:
        score = await run_in_threadpool(_run, body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ViT prediction failed: {e}")
    return {"fake_score": score}
