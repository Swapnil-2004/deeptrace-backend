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

# Render assigns its own port via $PORT — do not hardcode 7860
EXPOSE 10000

# Start the server, binding to whatever port Render provides
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
