"""
DeepTrace - ViT preprocessing, numpy only (no torch), memory-light.

Reproduces EXACTLY the torch uint8 antialiased bicubic resize used by the
HuggingFace ViTImageProcessor (same algorithm as onnx_work\\preproc_vit.py),
but the horizontal pass runs in row bands so the whole image is never
converted to float64.

Sums are exact integers in float64, so band order cannot change the result.
"""
import functools
import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BAND_ROWS = 64

# EXIF orientation -> transpose method (same mapping as PIL ImageOps.exif_transpose)
try:
    _T = Image.Transpose
except AttributeError:       # very old Pillow
    _T = Image
_EXIF_OPS = {2: _T.FLIP_LEFT_RIGHT, 3: _T.ROTATE_180, 4: _T.FLIP_TOP_BOTTOM,
             5: _T.TRANSPOSE, 6: _T.ROTATE_270, 7: _T.TRANSVERSE, 8: _T.ROTATE_90}


def _bic(x, a=-0.5):
    x = np.abs(x)
    return np.where(x < 1, ((a + 2) * x - (a + 3)) * x * x + 1,
           np.where(x < 2, (((x - 5) * x + 8) * x - 4) * a, 0.0))


@functools.lru_cache(maxsize=2)
def _mat_q(insize, outsize=224):
    scale = insize / outsize
    fs = max(scale, 1.0)
    support = 2.0 * fs
    M = np.zeros((outsize, insize))
    for i in range(outsize):
        c = (i + 0.5) * scale
        lo = max(int(c - support + 0.5), 0)
        hi = min(int(c + support + 0.5), insize)
        xs = np.arange(lo, hi)
        w = _bic((xs - c + 0.5) / fs)
        M[i, lo:hi] = w / w.sum()
    maxw = np.abs(M).max()
    p = 0
    while p < 22:
        if int(0.5 + maxw * (2 ** (p + 1))) >= 2 ** 15:
            break
        p += 1
    s = 2.0 ** p
    Q = np.where(M >= 0, np.trunc(M * s + 0.5), np.trunc(M * s - 0.5))
    return Q, p


def _q(x, p):
    return np.clip(np.floor((x + 2.0 ** (p - 1)) / 2.0 ** p), 0, 255)


def resize_u8(img):
    """PIL RGB image (or HxWx3 uint8 array) -> 224x224x3 uint8, exact."""
    if not hasattr(img, "crop"):
        img = Image.fromarray(np.asarray(img))
    orient = img.getexif().get(0x0112, 1)
    if img.mode != "RGB":
        img = img.convert("RGB")
    # The HuggingFace pipeline (original path) applies the EXIF orientation to PIL
    # inputs; do the same, copying the image only when a rotation/flip is needed.
    if orient in _EXIF_OPS:
        img = img.transpose(_EXIF_OPS[orient])
    W, H = img.size
    Qw, pw = _mat_q(W)
    Qh, ph = _mat_q(H)
    x = np.empty((H, 224, 3), dtype=np.uint8)
    for r0 in range(0, H, BAND_ROWS):
        r1 = min(r0 + BAND_ROWS, H)
        band = np.asarray(img.crop((0, r0, W, r1))).astype(np.float64)
        t = np.tensordot(band, Qw, axes=([1], [1])).transpose(0, 2, 1)
        x[r0:r1] = _q(t, pw).astype(np.uint8)
        del band, t
    y = np.tensordot(Qh, x.astype(np.float64), axes=([1], [0]))
    return _q(y, ph).astype(np.uint8)


def to_input(u8):
    a = (u8.astype(np.float32) / 255.0 - MEAN) / STD
    return np.expand_dims(a.transpose(2, 0, 1), 0).astype(np.float32)
