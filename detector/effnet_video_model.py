"""
================================================================
DeepTrace -- detector/effnet_video_model.py
Loads and runs the VIDEO-specific EfficientNet-B4 model
(trained on face-cropped video frames, v4 -- 90.64% Real /
96.85% Fake on combined FF++/Celeb-DF/SDFVD held-out test).

SEPARATE module from effnet_model.py (the image model).
Nothing here touches efficientnet_deeptrace.pth or effnet_model.py
in any way. This only loads/uses efficientnet_video.pth.

Matches effnet_model.py's interface exactly (including Grad-CAM
hook support), so pipeline.py can use it as a drop-in replacement
for the video path.

Expects an already face-cropped PIL image as input (30% padding,
matching training) -- crop the face BEFORE calling predict() here.
================================================================
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Optional, Tuple

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models", "efficientnet_video.pth"
)

# ── HuggingFace source (used if the file isn't on disk) ────────
HF_REPO_ID  = "Swapnil05092004/deeptrace-models"
HF_FILENAME = "efficientnet_video.pth"


def _ensure_model_downloaded() -> None:
    """Download the weight file from HuggingFace Hub if not present locally."""
    if os.path.exists(MODEL_PATH):
        return
    try:
        from huggingface_hub import hf_hub_download
        print(f"[EffNetVideo] Weights not found locally — downloading from "
              f"HuggingFace ({HF_REPO_ID})...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        downloaded_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
        import shutil
        shutil.copyfile(downloaded_path, MODEL_PATH)
        print("[OK] Weights downloaded and placed at", MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Could not download video model weights from HuggingFace: {e}")

IMG_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_model = None
_device = None
_hooks = []
_activations = {}


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_effnet_video() -> bool:
    global _model, _device

    _device = _get_device()
    print(f"[EffNetVideo] Using device: {_device}")

    _ensure_model_downloaded()

    try:
        import timm

        _model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=2)

        if not os.path.exists(MODEL_PATH):
            print(f"[EffNetVideo] Video model not found at {MODEL_PATH}")
            print("[EffNetVideo] Video analysis will not be available until this model exists.")
            _model = None
            return False

        print(f"[EffNetVideo] Loading video model from {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=_device, weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        _model.load_state_dict(state_dict, strict=True)
        _model = _model.to(_device)
        _model.eval()

        print("[OK] Video EfficientNet-B4 loaded")

        _register_gradcam_hook()

        return True

    except Exception as e:
        print(f"[ERROR] Video EfficientNet loading failed: {e}")
        import traceback
        traceback.print_exc()
        _model = None
        return False


def _register_gradcam_hook():
    global _hooks, _activations

    for hook in _hooks:
        hook.remove()
    _hooks = []

    if _model is None:
        return

    def save_activation(name):
        def hook(module, input, output):
            _activations[name] = output.detach()
        return hook

    last_conv = None
    for name, module in _model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = (name, module)

    if last_conv:
        name, module = last_conv
        hook = module.register_forward_hook(save_activation("last_conv"))
        _hooks.append(hook)
        print(f"[EffNetVideo] Grad-CAM hook registered on: {name}")


def _preprocess(img: Image.Image) -> torch.Tensor:
    img_resized = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    tensor = torch.from_numpy(arr).float().unsqueeze(0)
    return tensor


def predict(face_crop_img: Image.Image) -> Tuple[float, Optional[np.ndarray]]:
    """
    Run the video model on an already face-cropped image.
    Matches effnet_model.predict()'s interface exactly.
    """
    if _model is None:
        print("[WARNING] Video EfficientNet not loaded -- returning 0.5")
        return 0.5, None

    try:
        tensor = _preprocess(face_crop_img).to(_device)

        with torch.no_grad():
            logits = _model(tensor)

        probs = torch.softmax(logits, dim=1)
        fake_prob = float(probs[0, 1].cpu())

        activations = None
        if "last_conv" in _activations:
            activations = _activations["last_conv"].cpu().numpy()

        return fake_prob, activations

    except Exception as e:
        print(f"[ERROR] Video EfficientNet prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.5, None


def is_loaded() -> bool:
    return _model is not None
