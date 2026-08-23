"""
================================================================
DeepTrace — detector/effnet_model.py
EfficientNet-B4 deepfake detection model.

This is Model 1 — our own fine-tuned model.
Loaded via timm library with pretrained ImageNet weights,
then fine-tuned on deepfake dataset.

Also registers forward hook for Grad-CAM heatmap generation.
================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import os

# ── Model path ────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models", "efficientnet_deeptrace.pth"
)

# ── HuggingFace source (used if the file isn't on disk) ────────
HF_REPO_ID  = "Swapnil05092004/deeptrace-models"
HF_FILENAME = "efficientnet_deeptrace.pth"


def _ensure_model_downloaded() -> None:
    """Download the weight file from HuggingFace Hub if not present locally."""
    if os.path.exists(MODEL_PATH):
        return
    try:
        from huggingface_hub import hf_hub_download
        print(f"[EffNet] Weights not found locally — downloading from "
              f"HuggingFace ({HF_REPO_ID})...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        downloaded_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
        import shutil
        shutil.copyfile(downloaded_path, MODEL_PATH)
        print("[OK] Weights downloaded and placed at", MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Could not download model weights from HuggingFace: {e}")

# ── Global model state ────────────────────────────────────────
_model       = None
_device      = None
_hooks       = []
_activations = {}


def _get_device() -> torch.device:
    """Get best available device — CUDA if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_effnet() -> bool:
    """
    Load EfficientNet-B4 model.

    Priority:
    1. Load fine-tuned weights from models/efficientnet_deeptrace.pth
    2. If not found — load pretrained ImageNet weights from timm
       and use as-is (less accurate but functional)

    Returns:
        True if loaded successfully, False otherwise
    """
    global _model, _device

    _device = _get_device()
    print(f"[EffNet] Using device: {_device}")

    _ensure_model_downloaded()

    try:
        import timm

        # Create EfficientNet-B4 architecture
        _model = timm.create_model(
            "efficientnet_b4",
            pretrained   = False,   # We load weights manually
            num_classes  = 2        # Binary: real (0) or fake (1)
        )

        if os.path.exists(MODEL_PATH):
            # Load our fine-tuned weights
            print(f"[EffNet] Loading fine-tuned weights from {MODEL_PATH}")
            state_dict = torch.load(MODEL_PATH, map_location=_device)

            # Handle different save formats
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            _model.load_state_dict(state_dict, strict=False)
            print("[OK] Fine-tuned EfficientNet-B4 loaded")

        else:
            # Fine-tuned weights not found — use pretrained ImageNet
            print(f"[EffNet] Fine-tuned weights not found at {MODEL_PATH}")
            print("[EffNet] Loading pretrained ImageNet weights as fallback")
            _model = timm.create_model(
                "efficientnet_b4",
                pretrained  = True,
                num_classes = 2
            )
            print("[OK] EfficientNet-B4 (ImageNet pretrained) loaded")

        _model = _model.to(_device)
        _model.eval()

        # Register Grad-CAM hook on last conv layer
        _register_gradcam_hook()

        return True

    except Exception as e:
        print(f"[ERROR] EfficientNet loading failed: {e}")
        import traceback
        traceback.print_exc()
        _model = None
        return False


def _register_gradcam_hook():
    """
    Register forward hook on last convolutional layer.
    This captures activations needed for Grad-CAM heatmap.
    """
    global _hooks, _activations

    # Remove existing hooks
    for hook in _hooks:
        hook.remove()
    _hooks = []

    if _model is None:
        return

    def save_activation(name):
        def hook(module, input, output):
            _activations[name] = output.detach()
        return hook

    # Find last conv layer in EfficientNet-B4
    # In timm EfficientNet, this is typically the last block's conv_pwl
    last_conv = None
    for name, module in _model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = (name, module)

    if last_conv:
        name, module = last_conv
        hook = module.register_forward_hook(save_activation("last_conv"))
        _hooks.append(hook)
        print(f"[EffNet] Grad-CAM hook registered on: {name}")


def predict(img: Image.Image) -> Tuple[float, Optional[np.ndarray]]:
    """
    Run EfficientNet-B4 prediction on an image.

    Args:
        img: PIL Image in RGB mode (any size — resized internally)

    Returns:
        Tuple of:
        - fake_score: float 0-1 (1 = definitely fake)
        - activations: numpy array of last conv activations for Grad-CAM
                       or None if hook not available
    """
    if _model is None:
        print("[WARNING] EfficientNet not loaded — returning 0.5")
        return 0.5, None

    try:
        from utils.image_utils import pil_to_tensor

        # Prepare input tensor
        tensor = pil_to_tensor(img).to(_device)  # shape (1, 3, 224, 224)

        # Forward pass
        with torch.no_grad():
            logits = _model(tensor)  # shape (1, 2)

        # Convert to probability
        probs = torch.softmax(logits, dim=1)
        fake_prob = float(probs[0, 1].cpu())  # index 1 = fake class

        # Get activations for Grad-CAM
        activations = None
        if "last_conv" in _activations:
            activations = _activations["last_conv"].cpu().numpy()

        return fake_prob, activations

    except Exception as e:
        print(f"[ERROR] EfficientNet prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.5, None


def is_loaded() -> bool:
    """Check if model is loaded and ready."""
    return _model is not None
