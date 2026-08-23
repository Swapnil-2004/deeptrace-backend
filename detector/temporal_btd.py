"""
================================================================
DeepTrace — detector/temporal_btd.py
Temporal Boundary-Topology Detector (Temporal BTD)
This is mam's key algorithm — the academic highlight.

Core idea:
Face swaps and AI generation struggle at topological
boundaries (hairline, ears, jaw) ESPECIALLY under motion.

What this file does:
1. Tracks face boundary contour across video frames
2. Measures boundary curvature evolution over time
3. Detects:
   - Too stable boundary = AI generation loop = FAKE
   - Too erratic boundary = face-swap artifact = FAKE
   - Natural variation = REAL
4. Compares boundary motion vs optical flow field
   - Mismatch = face boundary moves differently from rest = FAKE
================================================================
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Tuple


# ================================================================
# THRESHOLDS — calibrated for temporal analysis
# ================================================================
# Boundary curvature std_dev thresholds
STABILITY_LOW  = 0.020  # Below this = too stable = AI loop = FAKE
STABILITY_HIGH = 0.250  # Above this = too erratic = swap artifact = FAKE

# Optical flow mismatch threshold
FLOW_MISMATCH_RATIO = 2.5  # boundary flow / face centre flow > this = FAKE


# ================================================================
# CONTOUR CURVATURE
# ================================================================
def compute_curvature(contour: np.ndarray,
                      n_points: int = 50) -> np.ndarray:
    """
    Compute curvature at evenly sampled points along a contour.

    Method:
    - Resample contour to n_points evenly spaced points
    - At each point, compute angle change between
      incoming and outgoing vectors
    - This angle change = curvature at that point

    Args:
        contour:  (N, 2) array of boundary points
        n_points: number of points to sample

    Returns:
        (n_points,) array of curvature values
    """
    if contour is None or len(contour) < 3:
        return np.zeros(n_points)

    try:
        # Resample to n_points evenly spaced
        n      = len(contour)
        idx    = np.linspace(0, n - 1, n_points).astype(int)
        pts    = contour[idx]  # (n_points, 2)

        curvatures = []
        for i in range(n_points):
            prev_pt = pts[(i - 1) % n_points]
            curr_pt = pts[i]
            next_pt = pts[(i + 1) % n_points]

            # Vectors
            v1 = curr_pt - prev_pt
            v2 = next_pt - curr_pt

            # Lengths
            l1 = np.linalg.norm(v1) + 1e-8
            l2 = np.linalg.norm(v2) + 1e-8

            # Unit vectors
            u1 = v1 / l1
            u2 = v2 / l2

            # Angle between them (dot product)
            dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
            angle = np.arccos(dot)

            curvatures.append(angle)

        return np.array(curvatures, dtype=np.float32)

    except Exception as e:
        print(f"[TempBTD] Curvature error: {e}")
        return np.zeros(n_points)


# ================================================================
# BOUNDARY TRACKING
# ================================================================
def track_boundary_evolution(contours: List[Optional[np.ndarray]]) -> Dict:
    """
    Track how the face boundary curvature evolves across frames.

    Args:
        contours: list of contour arrays (one per frame)
                  each is (N, 2) or None if face not detected

    Returns:
        Dict with:
            temporal_btd_score:    float 0-1 (1=fake)
            boundary_too_stable:   bool
            boundary_too_erratic:  bool
            mean_curvature_change: float
            std_curvature_change:  float
            flags:                 list of flag strings
    """
    # Filter out None contours
    valid_contours = [c for c in contours if c is not None]

    if len(valid_contours) < 2:
        return {
            "temporal_btd_score"   : 0.0,
            "boundary_too_stable"  : False,
            "boundary_too_erratic" : False,
            "mean_curvature_change": 0.0,
            "std_curvature_change" : 0.0,
            "flags"                : []
        }

    # Compute curvature at each frame
    curvature_arrays = [compute_curvature(c) for c in valid_contours]

    # Compute frame-to-frame curvature changes
    changes = []
    for i in range(1, len(curvature_arrays)):
        diff = np.abs(curvature_arrays[i] - curvature_arrays[i-1])
        changes.append(float(np.mean(diff)))

    if not changes:
        return {
            "temporal_btd_score"   : 0.0,
            "boundary_too_stable"  : False,
            "boundary_too_erratic" : False,
            "mean_curvature_change": 0.0,
            "std_curvature_change" : 0.0,
            "flags"                : []
        }

    changes = np.array(changes)
    mean_change = float(np.mean(changes))
    std_change  = float(np.std(changes))

    # Decision logic
    too_stable  = std_change < STABILITY_LOW
    too_erratic = std_change > STABILITY_HIGH
    flags       = []
    score       = 0.0

    if too_stable:
        # AI generation loops often have unnaturally stable boundaries
        score = 1.0 - (std_change / STABILITY_LOW)
        score = min(1.0, max(0.0, score))
        flags.append(
            f"Boundary curvature unnaturally stable across frames "
            f"(std={std_change:.4f}) — consistent with AI generation loop"
        )
        print(f"  [TempBTD] Too stable: std={std_change:.4f}")

    elif too_erratic:
        # Face swap artifacts cause erratic boundary changes
        score = min(1.0, (std_change - STABILITY_HIGH) / STABILITY_HIGH)
        flags.append(
            f"Boundary curvature too erratic across frames "
            f"(std={std_change:.4f}) — consistent with face-swap artifact"
        )
        print(f"  [TempBTD] Too erratic: std={std_change:.4f}")
    else:
        print(f"  [TempBTD] Natural boundary evolution: std={std_change:.4f}")

    return {
        "temporal_btd_score"   : float(score),
        "boundary_too_stable"  : bool(too_stable),
        "boundary_too_erratic" : bool(too_erratic),
        "mean_curvature_change": float(mean_change),
        "std_curvature_change" : float(std_change),
        "flags"                : flags
    }


# ================================================================
# OPTICAL FLOW MISMATCH
# ================================================================
def compute_flow_mismatch(frames: List[np.ndarray],
                          face_masks: List[Optional[np.ndarray]]) -> Dict:
    """
    Compare optical flow at face boundary vs face centre.

    In deepfakes, the face boundary often moves differently
    from the surrounding facial area — a key forensic signal.

    Args:
        frames:     list of BGR frames
        face_masks: list of binary face masks (one per frame)

    Returns:
        Dict with:
            flow_mismatch_score:    float 0-1
            flow_mismatch_detected: bool
            mean_boundary_flow:     float
            mean_centre_flow:       float
            flags:                  list of flag strings
    """
    if len(frames) < 2:
        return {
            "flow_mismatch_score"   : 0.0,
            "flow_mismatch_detected": False,
            "mean_boundary_flow"    : 0.0,
            "mean_centre_flow"      : 0.0,
            "flags"                 : []
        }

    ratios = []
    flags  = []

    for i in range(1, len(frames)):
        f1 = frames[i-1]
        f2 = frames[i]
        mask = face_masks[i] if face_masks and i < len(face_masks) else None

        if mask is None or mask.sum() == 0:
            continue

        try:
            # Compute dense optical flow
            gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0
            )

            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

            # Boundary region — erode mask to get centre,
            # subtract from dilated to get boundary
            kernel    = np.ones((9, 9), np.uint8)
            dilated   = cv2.dilate(mask, kernel)
            eroded    = cv2.erode(mask,  kernel)
            boundary  = dilated - eroded
            centre    = eroded

            if boundary.sum() == 0 or centre.sum() == 0:
                continue

            boundary_flow = float(np.mean(mag[boundary > 0]))
            centre_flow   = float(np.mean(mag[centre   > 0])) + 1e-8

            ratio = boundary_flow / centre_flow
            ratios.append(ratio)

        except Exception as e:
            print(f"[TempBTD] Flow error at frame {i}: {e}")
            continue

    if not ratios:
        return {
            "flow_mismatch_score"   : 0.0,
            "flow_mismatch_detected": False,
            "mean_boundary_flow"    : 0.0,
            "mean_centre_flow"      : 0.0,
            "flags"                 : []
        }

    mean_ratio = float(np.mean(ratios))
    mismatch_detected = mean_ratio > FLOW_MISMATCH_RATIO

    if mismatch_detected:
        score = min(1.0, (mean_ratio - FLOW_MISMATCH_RATIO) / FLOW_MISMATCH_RATIO)
        flags.append(
            f"Face boundary motion does not match surrounding optical flow "
            f"(ratio={mean_ratio:.2f}) — consistent with pasted face region"
        )
        print(f"  [TempBTD] Flow mismatch detected: ratio={mean_ratio:.2f}")
    else:
        score = 0.0
        print(f"  [TempBTD] Flow normal: ratio={mean_ratio:.2f}")

    return {
        "flow_mismatch_score"   : float(score),
        "flow_mismatch_detected": bool(mismatch_detected),
        "mean_ratio"            : float(mean_ratio),
        "flags"                 : flags
    }


# ================================================================
# TEMPORAL CONSISTENCY
# ================================================================
def compute_temporal_consistency(frame_scores: List[float]) -> Dict:
    """
    Measure consistency of per-frame fake scores.

    High variance between consecutive frame scores = flickering
    = deepfake artifact where frame quality varies.

    Args:
        frame_scores: list of per-frame fake scores (0-1)

    Returns:
        Dict with consistency_score and flag
    """
    if len(frame_scores) < 2:
        return {
            "consistency_score": 0.0,
            "flag"             : ""
        }

    diffs = [abs(frame_scores[i] - frame_scores[i-1])
             for i in range(1, len(frame_scores))]

    mean_diff = float(np.mean(diffs))

    # High variance = flickering = fake signal
    FLICKER_THRESHOLD = 0.12

    if mean_diff > FLICKER_THRESHOLD:
        score = min(1.0, (mean_diff - FLICKER_THRESHOLD) / FLICKER_THRESHOLD)
        flag  = (f"High frame-to-frame score variance ({mean_diff:.3f}) — "
                 "flickering detection scores consistent with deepfake artifact")
        print(f"  [TempBTD] Flickering detected: mean_diff={mean_diff:.3f}")
        return {"consistency_score": score, "flag": flag}

    print(f"  [TempBTD] Consistent scores: mean_diff={mean_diff:.3f}")
    return {"consistency_score": 0.0, "flag": ""}


# ================================================================
# MAIN FUNCTION — Run full temporal BTD analysis
# ================================================================
def run_temporal_btd(frames: List[np.ndarray],
                     contours: List[Optional[np.ndarray]],
                     face_masks: List[Optional[np.ndarray]],
                     frame_scores: List[float]) -> Dict:
    """
    Run complete temporal BTD analysis on video frames.
    Called by pipeline.py after per-frame analysis is done.

    Args:
        frames:       list of BGR frames
        contours:     list of face contour arrays (one per frame)
        face_masks:   list of face mask arrays (one per frame)
        frame_scores: list of per-frame fake scores from image pipeline

    Returns:
        Dict with all temporal analysis results
    """
    print(f"[TempBTD] Analysing {len(frames)} frames...")

    all_flags = []

    # 1. Track boundary curvature evolution
    boundary_result = track_boundary_evolution(contours)
    all_flags.extend(boundary_result.get("flags", []))

    # 2. Compute optical flow mismatch
    flow_result = compute_flow_mismatch(frames, face_masks)
    all_flags.extend(flow_result.get("flags", []))

    # 3. Temporal consistency of frame scores
    consistency_result = compute_temporal_consistency(frame_scores)
    if consistency_result.get("flag"):
        all_flags.append(consistency_result["flag"])

    # 4. Combined temporal BTD score
    # Weighted combination of all temporal signals
    boundary_score    = boundary_result.get("temporal_btd_score", 0.0)
    flow_score        = flow_result.get("flow_mismatch_score",    0.0)
    consistency_score = consistency_result.get("consistency_score", 0.0)

    temporal_score = (
        0.50 * boundary_score    +   # Boundary evolution — most important
        0.30 * flow_score        +   # Flow mismatch — strong signal
        0.20 * consistency_score     # Flickering — supporting signal
    )
    temporal_score = float(min(1.0, temporal_score))

    print(f"[TempBTD] Scores: boundary={boundary_score:.3f} "
          f"flow={flow_score:.3f} consistency={consistency_score:.3f}")
    print(f"[TempBTD] Final temporal score: {temporal_score:.3f}")

    return {
        "score"                    : temporal_score,
        "boundary_too_stable"      : boundary_result.get("boundary_too_stable",   False),
        "boundary_too_erratic"     : boundary_result.get("boundary_too_erratic",  False),
        "flow_mismatch_detected"   : flow_result.get("flow_mismatch_detected",    False),
        "mean_curvature_change"    : boundary_result.get("mean_curvature_change", 0.0),
        "std_curvature_change"     : boundary_result.get("std_curvature_change",  0.0),
        "consistency_score"        : consistency_score,
        "flow_score"               : flow_score,
        "flags"                    : all_flags
    }
