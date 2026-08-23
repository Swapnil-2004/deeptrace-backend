# DeepTrace — Dockerfile
# For Render.com deployment

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgles2 \
    libegl1 \
    libgstreamer1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker cache optimization)
COPY requirements.txt .

# Install PyTorch CPU version (Render free tier has no GPU)
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# ── Pre-download all models during BUILD, not at first user request ──
# This moves the slow download+cache step to deploy time (when Render
# builds the image) instead of making the first real user wait for it.
# Same weights, same models -- just fetched earlier in the process.
RUN mkdir -p models && \
    python -c "\
from huggingface_hub import hf_hub_download; \
import shutil; \
p1 = hf_hub_download(repo_id='Swapnil05092004/deeptrace-models', filename='efficientnet_deeptrace.pth'); \
shutil.copyfile(p1, 'models/efficientnet_deeptrace.pth'); \
p2 = hf_hub_download(repo_id='Swapnil05092004/deeptrace-models', filename='efficientnet_video.pth'); \
shutil.copyfile(p2, 'models/efficientnet_video.pth'); \
print('[Build] EfficientNet weights pre-downloaded successfully')"

# Pre-download the ViT (Organika/sdxl-detector) model + face_landmarker.task
# so these are cached in the image too, not fetched on first request.
RUN python -c "\
from transformers import pipeline; \
pipeline('image-classification', model='Organika/sdxl-detector', device=-1); \
print('[Build] ViT model pre-downloaded successfully')"

RUN mkdir -p models && \
    python -c "\
import urllib.request; \
url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'; \
urllib.request.urlretrieve(url, 'models/face_landmarker.task'); \
print('[Build] face_landmarker.task pre-downloaded successfully')"

# Render assigns its own port via $PORT — do not hardcode 7860
EXPOSE 10000

# Start the server, binding to whatever port Render provides
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
