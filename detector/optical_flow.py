"""
================================================================
DeepTrace — detector/optical_flow.py
Optical flow analysis for video deepfake detection.

What this file does:
1. Computes dense optical flow between consecutive frames
2. Calculates mean flow magnitude across the video
3. Detects frozen background + moving face (deepfake signal)
4. Works together with temporal_btd.py for video analysis
================================================================
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple


def compute_flow_score(frames: List[np.ndarray],
                       face_masks: List[Optional[np.ndarray]]) -> dict:
    """
    Compute optical flow analysis score for a video.

    Deepfake signal: very low global flow + high face score variance
    = frozen/looping background with pasted face

    Args:
        frames:     list of BGR frames
        face_masks: list of binary face masks (one per frame)

    Returns:
        Dict with flow_score, mean_flow, and analysis details
    """
    if len(frames) < 2:
        return {
            "flow_score"       : 0.0,
            "mean_flow"        : 0.0,
            "mean_bg_flow"     : 0.0,
            "mean_face_flow"   : 0.0,
            "frozen_background": False
        }

    flow_magnitudes    = []
    bg_flow_values     = []
    face_flow_values   = []

    for i in range(1, min(len(frames), 20)):
        f1   = frames[i - 1]
        f2   = frames[i]
        mask = face_masks[i] if face_masks and i < len(face_masks) else None

        try:
            gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0
            )

            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_magnitudes.append(float(np.mean(mag)))

            # Separate face vs background flow
            if mask is not None and mask.sum() > 0:
                bg_mask = (1 - mask).astype(bool)
                if bg_mask.sum() > 0:
                    bg_flow_values.append(float(np.mean(mag[bg_mask])))
                face_flow_values.append(float(np.mean(mag[mask > 0])))

        except Exception as e:
            print(f"[Flow] Error at frame {i}: {e}")
            continue

    if not flow_magnitudes:
        return {
            "flow_score"       : 0.0,
            "mean_flow"        : 0.0,
            "mean_bg_flow"     : 0.0,
            "mean_face_flow"   : 0.0,
            "frozen_background": False
        }

    mean_flow    = float(np.mean(flow_magnitudes))
    mean_bg_flow = float(np.mean(bg_flow_values))   if bg_flow_values   else mean_flow
    mean_face_flow = float(np.mean(face_flow_values)) if face_flow_values else mean_flow

    # Frozen background detection
    # Real videos: background and face move together naturally
    # Deepfakes: background is often static/looping while face moves
    frozen_background = False
    flow_score        = 0.0

    if bg_flow_values and face_flow_values:
        bg_mean   = mean_bg_flow   + 1e-8
        face_mean = mean_face_flow + 1e-8
        ratio     = face_mean / bg_mean

        # If face moves much more than background = suspicious
        if ratio > 3.0 and mean_bg_flow < 0.5:
            frozen_background = True
            flow_score = min(1.0, (ratio - 3.0) / 3.0)
            print(f"[Flow] Frozen background detected: ratio={ratio:.2f}")

    print(f"[Flow] mean={mean_flow:.3f} bg={mean_bg_flow:.3f} "
          f"face={mean_face_flow:.3f} score={flow_score:.3f}")

    return {
        "flow_score"       : flow_score,
        "mean_flow"        : mean_flow,
        "mean_bg_flow"     : mean_bg_flow,
        "mean_face_flow"   : mean_face_flow,
        "frozen_background": frozen_background
    }
