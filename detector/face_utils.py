"""
================================================================
DeepTrace — detector/face_utils.py  (FIXED — MediaPipe Tasks API)
Face detection, landmark extraction, and face cropping.

IMPORTANT FIX (this version):
Google removed the legacy `mp.solutions.*` API from recent
MediaPipe releases (0.10.32+). Your installed version 0.10.35
no longer supports it, causing every detection to silently fail
with: "module 'mediapipe' has no attribute 'solutions'".

This version migrates to the official replacement:
the MediaPipe Tasks API (FaceLandmarker), which uses a
downloadable .task model bundle instead of the old namespace.

This file ONLY changes face detection plumbing.
BTD math, EfficientNet, ensemble weights — untouched.
================================================================
"""

import os
import cv2
import urllib.request
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, List

# ── Model file location ─────────────────────────────────────
# Downloaded automatically on first run, cached after that.
_MODEL_DIR  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models"
)
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# Lazy-loaded global landmarker instance
_landmarker = None


def _ensure_model_downloaded():
    """
    Download the face_landmarker.task model bundle if not present.
    Only runs once — cached in models/ folder afterward.
    """
    os.makedirs(_MODEL_DIR, exist_ok=True)

    if os.path.exists(_MODEL_PATH) and os.path.getsize(_MODEL_PATH) > 1_000_000:
        return  # Already downloaded and looks valid

    print(f"[FaceUtils] Downloading face_landmarker.task (~3.7MB)...")
    print(f"[FaceUtils] Source: {_MODEL_URL}")
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"[FaceUtils] Saved to: {_MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Could not download face_landmarker.task: {e}")
        print("[ERROR] Face detection will be unavailable.")


def _init_mediapipe():
    """
    Initialise MediaPipe FaceLandmarker (new Tasks API) on first use.
    Replaces the old, now-removed mp.solutions.face_mesh API.
    """
    global _landmarker

    if _landmarker is not None:
        return  # Already initialised

    _ensure_model_downloaded()

    if not os.path.exists(_MODEL_PATH):
        print("[ERROR] face_landmarker.task missing — cannot init MediaPipe")
        _landmarker = None
        return

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)

        options = mp_vision.FaceLandmarkerOptions(
            base_options                 = base_options,
            running_mode                 = mp_vision.RunningMode.IMAGE,
            num_faces                    = 1,
            min_face_detection_confidence= 0.4,
            min_face_presence_confidence = 0.4,
            min_tracking_confidence      = 0.4,
            output_face_blendshapes      = False,
            output_facial_transformation_matrixes = False,
        )

        _landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        print("[OK] MediaPipe FaceLandmarker (Tasks API) loaded")

    except Exception as e:
        print(f"[WARNING] MediaPipe Tasks API failed to load: {e}")
        import traceback
        traceback.print_exc()
        _landmarker = None


def _run_landmarker(img: Image.Image):
    """
    Run the FaceLandmarker on a PIL image.
    Returns the raw MediaPipe detection result, or None on failure.
    """
    _init_mediapipe()

    if _landmarker is None:
        return None

    try:
        import mediapipe as mp
        img_rgb = np.array(img.convert("RGB"))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = _landmarker.detect(mp_image)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None

        return result

    except Exception as e:
        print(f"[WARNING] Face landmark detection error: {e}")
        return None


def detect_face(img: Image.Image) -> Optional[Dict]:
    """
    Detect the primary face in an image.
    Derives a bounding box from the 468 landmarks
    (Tasks API does not return a separate bbox directly).

    Returns:
        Dict with bbox, confidence, center — or None if no face.
    """
    result = _run_landmarker(img)
    if result is None:
        return None

    w, h = img.size

    try:
        landmarks = result.face_landmarks[0]  # list of NormalizedLandmark

        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]

        x1, x2 = max(0, int(min(xs))), min(w, int(max(xs)))
        y1, y2 = max(0, int(min(ys))), min(h, int(max(ys)))

        if x2 <= x1 or y2 <= y1:
            return None

        return {
            "bbox"      : (x1, y1, x2, y2),
            "confidence": 0.9,  # Tasks API doesn't expose a single score here
            "center"    : ((x1 + x2) // 2, (y1 + y2) // 2)
        }

    except Exception as e:
        print(f"[WARNING] Face bbox derivation error: {e}")
        return None


def get_landmarks(img: Image.Image,
                  bbox: Optional[Tuple] = None) -> Optional[np.ndarray]:
    """
    Extract 468 facial landmarks from image using Tasks API.

    Returns:
        numpy array (468, 3) — x, y in pixel coords, z relative depth
        Or None if landmarks not found.
    """
    result = _run_landmarker(img)
    if result is None:
        return None

    w, h = img.size

    try:
        landmarks = result.face_landmarks[0]
        points = np.array([
            [lm.x * w, lm.y * h, lm.z]
            for lm in landmarks
        ], dtype=np.float32)
        return points  # shape (468, 3)

    except Exception as e:
        print(f"[WARNING] Landmark extraction error: {e}")
        return None


def crop_face(img: Image.Image,
              bbox: Tuple,
              padding: float = 0.25) -> Image.Image:
    """Crop face from image with padding around bounding box."""
    w, h            = img.size
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return img.crop((x1, y1, x2, y2))


# Face oval landmark indices — unchanged, same MediaPipe topology
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58,  132,
    93,  234, 127, 162, 21,  54,  103, 67,  109
]


def create_face_mask(img_shape: Tuple,
                     landmarks: np.ndarray) -> np.ndarray:
    """Create a binary mask of the face region using landmarks."""
    H, W = img_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    if landmarks is None:
        return mask

    try:
        oval_pts = landmarks[FACE_OVAL_INDICES, :2].astype(np.int32)
        oval_pts = oval_pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [oval_pts], 1)
    except Exception as e:
        print(f"[WARNING] Face mask error: {e}")

    return mask


def get_face_contour(img: Image.Image,
                     landmarks: np.ndarray) -> Optional[np.ndarray]:
    """Extract the face boundary contour as ordered points."""
    if landmarks is None:
        return None

    try:
        contour = landmarks[FACE_OVAL_INDICES, :2].astype(np.float32)
        return contour  # shape (36, 2)
    except Exception as e:
        print(f"[WARNING] Contour extraction error: {e}")
        return None


def measure_facial_symmetry(landmarks: np.ndarray) -> float:
    """
    Measure facial symmetry deviation.
    BTD Signal I — AI faces are often too symmetrical.
    Returns deviation score: low = too symmetric = AI signal.
    """
    if landmarks is None:
        return 0.5

    SYMMETRIC_PAIRS = [
        (33,  263), (133, 362), (70,  300),
        (105, 334), (61,  291), (234, 454),
        (127, 356), (93,  323), (132, 361),
    ]

    deviations = []

    try:
        centre_x = landmarks[4, 0]

        for left_idx, right_idx in SYMMETRIC_PAIRS:
            if left_idx >= len(landmarks) or right_idx >= len(landmarks):
                continue

            left_pt  = landmarks[left_idx,  :2]
            right_pt = landmarks[right_idx, :2]

            dist_left  = abs(left_pt[0]  - centre_x)
            dist_right = abs(right_pt[0] - centre_x)

            if dist_left + dist_right > 0:
                deviation = abs(dist_left - dist_right) / (dist_left + dist_right)
                deviations.append(deviation)

    except Exception as e:
        print(f"[WARNING] Symmetry measurement error: {e}")
        return 0.5

    if not deviations:
        return 0.5

    mean_deviation = float(np.mean(deviations))
    return min(1.0, mean_deviation / 0.15)


def detect_and_crop(img: Image.Image,
                    padding: float = 0.25) -> Dict:
    """
    Main entry point — detect face, get landmarks, crop.
    Returns everything needed by the detection pipeline.

    Single landmarker call internally (not two) — faster than
    the previous version, since Tasks API gives both bbox-source
    landmarks and full mesh in one detect() call.
    """
    result_dict = {
        "face_found" : False,
        "face_crop"  : img,
        "bbox"       : None,
        "landmarks"  : None,
        "face_mask"  : None,
        "contour"    : None,
        "symmetry"   : 0.5
    }

    raw_result = _run_landmarker(img)

    if raw_result is None:
        print("[INFO] No face detected — using full image")
        return result_dict

    w, h = img.size

    try:
        landmarks_raw = raw_result.face_landmarks[0]
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z]
            for lm in landmarks_raw
        ], dtype=np.float32)
    except Exception as e:
        print(f"[WARNING] Landmark conversion error: {e}")
        return result_dict

    # Derive bbox from landmarks
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    x1, x2 = max(0, int(xs.min())), min(w, int(xs.max()))
    y1, y2 = max(0, int(ys.min())), min(h, int(ys.max()))

    if x2 <= x1 or y2 <= y1:
        print("[INFO] Invalid face bbox — using full image")
        return result_dict

    result_dict["face_found"] = True
    result_dict["bbox"]       = (x1, y1, x2, y2)
    result_dict["landmarks"]  = landmarks

    face_crop = crop_face(img, (x1, y1, x2, y2), padding)
    result_dict["face_crop"] = face_crop

    img_arr   = np.array(img)
    face_mask = create_face_mask(img_arr.shape, landmarks)
    result_dict["face_mask"] = face_mask

    contour = get_face_contour(img, landmarks)
    result_dict["contour"] = contour

    symmetry = measure_facial_symmetry(landmarks)
    result_dict["symmetry"] = symmetry

    print(f"[FaceUtils] Face detected successfully — bbox={x1},{y1},{x2},{y2}")

    return result_dict
