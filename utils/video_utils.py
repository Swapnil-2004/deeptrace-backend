"""
================================================================
DeepTrace — utils/video_utils.py
Video processing utilities.
Handles frame extraction, sampling strategy,
video metadata reading, and cleanup.
================================================================
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict


def get_video_info(video_path: str) -> Dict:
    """
    Read basic metadata from a video file.

    Args:
        video_path: path to video file

    Returns:
        dict with fps, total_frames, width, height, duration_sec
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    duration_sec = total_frames / fps if fps > 0 else 0

    return {
        "fps"          : round(fps, 2),
        "total_frames" : total_frames,
        "width"        : width,
        "height"       : height,
        "duration_sec" : round(duration_sec, 2)
    }


def sample_frames(video_path: str,
                  max_frames: int = 30) -> Tuple[List[np.ndarray], List[int]]:
    """
    Sample frames evenly from a video.

    Sampling strategy:
    - Short video (<=10s): sample every frame up to max_frames
    - Medium video (10-60s): sample max_frames evenly
    - Long video (>60s): sample max_frames evenly distributed

    Args:
        video_path: path to video file
        max_frames: maximum number of frames to extract

    Returns:
        Tuple of:
        - list of frames as numpy arrays (H, W, 3) BGR
        - list of frame indices that were sampled
    """
    info         = get_video_info(video_path)
    total        = info["total_frames"]
    fps          = info["fps"]
    duration_sec = info["duration_sec"]

    if total <= 0:
        raise ValueError("Video has no frames or could not be read")

    # Determine frame indices to sample
    if duration_sec <= 10.0:
        # Short video — sample every frame up to max_frames
        step    = max(1, total // max_frames)
        indices = list(range(0, total, step))[:max_frames]
    else:
        # Longer video — evenly distributed
        indices = [
            int(i * (total - 1) / (max_frames - 1))
            for i in range(max_frames)
        ]
        # Remove duplicates while preserving order
        seen    = set()
        indices = [x for x in indices if not (x in seen or seen.add(x))]

    # Extract frames
    cap    = cv2.VideoCapture(video_path)
    frames = []
    sampled_indices = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
            sampled_indices.append(idx)

    cap.release()

    if len(frames) == 0:
        raise ValueError("Could not extract any frames from video")

    return frames, sampled_indices


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    """
    Convert OpenCV BGR frame to PIL Image in RGB.

    Args:
        frame: numpy array (H, W, 3) in BGR format

    Returns:
        PIL Image in RGB
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Convert BGR numpy array to RGB numpy array.

    Args:
        frame: (H, W, 3) BGR

    Returns:
        (H, W, 3) RGB
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def compute_optical_flow(frame1: np.ndarray,
                         frame2: np.ndarray) -> np.ndarray:
    """
    Compute dense optical flow between two consecutive frames.
    Uses Farneback method — good balance of speed and quality.

    Args:
        frame1: first frame (H, W, 3) BGR
        frame2: second frame (H, W, 3) BGR

    Returns:
        flow array (H, W, 2) — x and y flow components
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,
        pyr_scale  = 0.5,   # pyramid scale
        levels     = 3,     # pyramid levels
        winsize    = 15,    # averaging window size
        iterations = 3,     # iterations per level
        poly_n     = 5,     # polynomial neighbourhood size
        poly_sigma = 1.2,   # Gaussian sigma for polynomial
        flags      = 0
    )

    return flow


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """
    Compute magnitude of optical flow vectors.

    Args:
        flow: (H, W, 2) flow array

    Returns:
        (H, W) magnitude array
    """
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag


def mean_flow_in_region(flow: np.ndarray,
                        mask: np.ndarray) -> float:
    """
    Calculate mean flow magnitude within a masked region.

    Args:
        flow: (H, W, 2) optical flow
        mask: (H, W) binary mask — 1 inside region, 0 outside

    Returns:
        Mean flow magnitude inside the region
    """
    mag = flow_magnitude(flow)

    # Apply mask
    if mask is not None and mask.sum() > 0:
        region_flow = mag[mask > 0]
        return float(np.mean(region_flow))

    return float(np.mean(mag))


def resize_frame(frame: np.ndarray,
                 max_size: int = 640) -> np.ndarray:
    """
    Resize frame if it is larger than max_size on any dimension.
    Maintains aspect ratio.

    Args:
        frame:    (H, W, 3) BGR frame
        max_size: maximum dimension size

    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]

    if max(h, w) <= max_size:
        return frame

    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def cleanup_temp_file(path: str) -> None:
    """
    Safely delete a temporary file.

    Args:
        path: file path to delete
    """
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            print(f"Warning: Could not delete temp file {path}: {e}")
