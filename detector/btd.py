"""
================================================================
DeepTrace — detector/btd.py
Boundary-Topology Detector (BTD)
Implements all 9 forensic signals from mentor's algorithm.

Core idea: AI generation and face-swapping always struggle
at topological boundaries — hairline, ears, jaw.
These 9 signals examine those exact regions.

Signals:
  A — Face oval curvature
  B — Boundary gradient mismatch
  C — Edge persistence (gradient uniformity)
  D — ELA (Error Level Analysis)
  E — Noise residual
  F — Chromatic aberration
  G — Colour kurtosis
  H — High-frequency content
  I — Facial landmark symmetry (from face_utils)
================================================================
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
from scipy import stats


# ================================================================
# THRESHOLDS
# Calibrated across multiple image types:
# MidJourney, Stable Diffusion, DALL-E, FaceSwap, real photos
# ================================================================
THRESHOLDS = {
    # Signal A — curvature residual: AI = too perfect oval (low residual)
    "curvature_real_min"  : 0.08,   # below this = too perfect = AI

    # Signal B — gradient mismatch: AI = abrupt boundary jump
    "gradient_mismatch_fake": 0.45, # above this = AI signal

    # Signal C — edge uniformity: AI = very uniform (low CV)
    "edge_cv_real_min"    : 0.25,   # below this = too uniform = AI

    # Signal D — ELA variance: AI = very uniform blocks (low variance)
    "ela_var_real_min"    : 8.0,    # below this = no editing history = AI

    # Signal E — noise level: AI = near zero noise
    "noise_real_min"      : 0.003,  # below this = impossible in camera = AI

    # Signal F — chromatic aberration: AI = missing fringing
    "chroma_real_min"     : 0.01,   # below this = no lens aberration = AI

    # Signal G — colour kurtosis: AI = flat distribution
    "kurtosis_real_min"   : 0.0,    # below this = platykurtic = AI

    # Signal H — HF content: AI = too smooth
    "hf_ratio_real_min"   : 0.08,   # below this = too smooth = AI

    # Signal I — symmetry deviation: AI = too symmetric
    "symmetry_real_min"   : 0.04,   # below this = too symmetric = AI
}

# Signal weights for final BTD score
SIGNAL_WEIGHTS = {
    "A": 0.20,  # curvature — increased, gets some of B's weight
    "B": 0.00,  # DISABLED — always returns 0 — see run_all_signals()
    "C": 0.13,  # edge uniformity
    "D": 0.12,  # ELA
    "E": 0.20,  # noise — proven reliable, gets B's weight
    "F": 0.15,  # chromatic aberration — proven reliable
    "G": 0.08,  # colour kurtosis
    "H": 0.07,  # HF content
    "I": 0.05,  # symmetry
}


# ================================================================
# SIGNAL A — Face Oval Curvature
# AI faces fit a near-perfect polynomial oval.
# Real faces have natural irregular asymmetry.
# ================================================================
def signal_a_curvature(img: Image.Image,
                       face_mask: Optional[np.ndarray]) -> Tuple[float, str]:
    """
    Analyse face boundary curvature.
    Fits a polynomial to the face silhouette width per row.
    Low residual = too perfect = AI signal.

    Returns:
        (score 0-1, flag message or "")
        score: 1 = definitely AI, 0 = definitely real
    """
    if face_mask is None or face_mask.sum() == 0:
        return 0.0, ""

    try:
        h, w = face_mask.shape[:2]

        # For each row, find leftmost and rightmost face pixel
        widths = []
        rows   = []
        for y in range(h):
            row = face_mask[y]
            xs  = np.where(row > 0)[0]
            if len(xs) >= 2:
                widths.append(xs[-1] - xs[0])
                rows.append(y)

        if len(widths) < 10:
            return 0.0, ""

        rows   = np.array(rows,   dtype=np.float32)
        widths = np.array(widths, dtype=np.float32)

        # Normalise
        rows_n   = rows   / h
        widths_n = widths / w

        # Fit degree-3 polynomial
        coeffs  = np.polyfit(rows_n, widths_n, 3)
        fitted  = np.polyval(coeffs, rows_n)
        residuals = np.abs(widths_n - fitted)
        mean_res  = float(np.mean(residuals))

        # Low residual = perfect oval = AI signal
        threshold = THRESHOLDS["curvature_real_min"]
        if mean_res < threshold:
            score = 1.0 - (mean_res / threshold)
            score = min(1.0, score)
            flag  = (f"Near-perfect face oval shape ({mean_res:.3f}) — "
                     "AI faces fit a smooth polynomial curve")
            return score, flag

        return 0.0, ""

    except Exception as e:
        print(f"[BTD-A] Error: {e}")
        return 0.0, ""


# ================================================================
# SIGNAL B — Boundary Gradient Mismatch
# AI images have abrupt edge jumps at hairline/jaw boundary.
# Real images have smooth gradient transitions.
# ================================================================
def signal_b_gradient_mismatch(img: Image.Image,
                                face_mask: Optional[np.ndarray]) -> Tuple[float, str]:
    """
    Compare fine vs coarse Laplacian response at face boundary.
    High mismatch = abrupt edge = AI signal.
    """
    if face_mask is None or face_mask.sum() == 0:
        return 0.0, ""

    try:
        img_arr = np.array(img.convert("L"), dtype=np.float32)

        # Fine Laplacian (sharp edges)
        lap_fine   = np.abs(cv2.Laplacian(img_arr, cv2.CV_32F, ksize=1))
        # Coarse Laplacian (broad transitions)
        lap_coarse = np.abs(cv2.Laplacian(img_arr, cv2.CV_32F, ksize=5))

        # Boundary region = dilated mask edge
        kernel   = np.ones((7, 7), np.uint8)
        boundary = cv2.dilate(face_mask, kernel) - face_mask

        if boundary.sum() == 0:
            return 0.0, ""

        # Mismatch at boundary
        fine_vals   = lap_fine[boundary > 0]
        coarse_vals = lap_coarse[boundary > 0]

        # Normalise difference using a symmetric ratio with a sane
        # floor based on the overall edge scale of this image — not
        # a tiny fixed epsilon. A fixed 1e-6 floor lets near-flat
        # boundary pixels (common in real hair/skin/shadow regions)
        # explode the ratio to absurd values, which then silently
        # clip to the 1.0 ceiling and pin the signal permanently on.
        scale = max(float(np.mean(coarse_vals)), 1e-3)
        denom = np.maximum(coarse_vals, 0.05 * scale)

        per_pixel_mismatch = np.abs(fine_vals - coarse_vals) / denom
        # Clip per-pixel outliers BEFORE averaging, so a handful of
        # extreme pixels can't dominate the whole boundary's score.
        per_pixel_mismatch = np.clip(per_pixel_mismatch, 0.0, 3.0)

        mismatch = float(np.mean(per_pixel_mismatch)) / 3.0  # back to 0-1
        mismatch = min(1.0, mismatch)

        threshold = THRESHOLDS["gradient_mismatch_fake"]
        if mismatch > threshold:
            score = (mismatch - threshold) / (1.0 - threshold)
            score = min(1.0, score)
            flag  = (f"Boundary gradient mismatch at hairline/jaw "
                     f"({mismatch:.3f}) — abrupt edge inconsistent with real hair")
            return score, flag

        return 0.0, ""

    except Exception as e:
        print(f"[BTD-B] Error: {e}")
        return 0.0, ""


# ================================================================
# SIGNAL C — Edge Persistence (Gradient Uniformity)
# AI images have unnaturally uniform edge strength.
# Real images have natural variation in edge intensity.
# ================================================================
def signal_c_edge_persistence(img: Image.Image) -> Tuple[float, str]:
    """
    Measure coefficient of variation (CV) of edge magnitudes.
    Low CV = edges too uniform = AI signal.
    """
    try:
        gray  = np.array(img.convert("L"), dtype=np.float32)

        # Sobel gradient magnitude
        gx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)

        # Only look at strong edges
        strong = mag[mag > np.percentile(mag, 75)]

        if len(strong) < 10:
            return 0.0, ""

        mean_val = float(np.mean(strong))
        std_val  = float(np.std(strong))

        if mean_val < 1e-6:
            return 0.0, ""

        cv = std_val / mean_val  # Coefficient of variation

        threshold = THRESHOLDS["edge_cv_real_min"]
        if cv < threshold:
            score = 1.0 - (cv / threshold)
            score = min(1.0, score)
            flag  = (f"Unnaturally uniform edge strength (CV={cv:.3f}) — "
                     "real images have natural variation in edge intensity")
            return score, flag

        return 0.0, ""

    except Exception as e:
        print(f"[BTD-C] Error: {e}")
        return 0.0, ""


# ================================================================
# SIGNAL D — ELA (Error Level Analysis)
# Re-saves image at known JPEG quality, measures block variance.
# AI images have no editing history — perfectly uniform blocks.
# ================================================================
def signal_d_ela(img: Image.Image) -> Tuple[float, str]:
    """
    Error Level Analysis.
    Low block variance after re-save = no editing history = AI signal.
    """
    try:
        import io

        # Save at known JPEG quality
        buffer = io.BytesIO()
        img_rgb = img.convert("RGB")
        img_rgb.save(buffer, format="JPEG", quality=75)
        buffer.seek(0)

        # Reload and compute difference
        resaved = Image.open(buffer).convert("RGB")
        orig_arr = np.array(img_rgb,  dtype=np.float32)
        res_arr  = np.array(resaved,  dtype=np.float32)

        ela_map = np.abs(orig_arr - res_arr)

        # Block variance — divide into 8x8 blocks
        h, w = ela_map.shape[:2]
        block_vars = []
        for y in range(0, h - 8, 8):
            for x in range(0, w - 8, 8):
                block = ela_map[y:y+8, x:x+8]
                block_vars.append(float(np.var(block)))

        if not block_vars:
            return 0.0, ""

        mean_var = float(np.mean(block_vars))

        threshold = THRESHOLDS["ela_var_real_min"]
        if mean_var < threshold:
            score = 1.0 - (mean_var / threshold)
            score = min(1.0, score)
            flag  = (f"Very uniform ELA blocks (var={mean_var:.2f}) — "
                     "no editing history detected, consistent with AI generation")
            return score, flag

        return 0.0, ""

    except Exception as e:
        print(f"[BTD-D] Error: {e}")
        return 0.0, ""


# ================================================================
# SIGNAL E — Noise Residual
# Real cameras always introduce sensor noise.
# AI images have near-zero sensor noise — physically impossible.
# ================================================================
def signal_e_noise(img: Image.Image) -> Tuple[float, str]:
    """
    Measure sensor noise residual.
    Very low noise = impossible in real cameras = AI signal.
    """
    try:
        gray  = np.array(img.convert("L"), dtype=np.float32) / 255.0

        # Compare to Gaussian-blurred version
        blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = np.abs(gray - blurred)

        mean_noise = float(np.mean(residual))

        threshold = THRESHOLDS["noise_real_min"]
        if mean_noise < threshold:
            score = 1.0 - (mean_noise / threshold)
            score = min(1.0, score)
            noise_pct = mean_noise * 1000  # scale for readability
            flag  = (f"Very low sensor noise ({noise_pct:.3f}‰) — "
                     "impossible in real photographic capture")
            return score, flag

        return 0.0, ""

    except Exception as e:
        print(f"[BTD-E] Error: {e}")
        return 0.0, ""


# ================================================================
# SIGNAL F — Chromatic Aberration
# Real lenses always produce colour fringing at edges.
# AI generation completely misses this optical phenomenon.
# ================================================================
def signal_f_chromatic_aberration(img: Image.Image) -> Tuple[float, str]:
    """
    Detect chromatic aberration (colour fringing at edges).
    Missing aberration = AI generation = AI signal.
    """
    try:
        img_arr = np.array(img.convert("RGB"), dtype=np.float32)

        r_channel = img_arr[:, :, 0]
        b_channel = img_arr[:, :, 2]

        # Laplacian of red vs blue — different for real lenses
        lap_r = np.abs(cv2.Laplacian(r_channel, cv2.CV_32F))
        lap_b = np.abs(cv2.Laplacian(b_channel, cv2.CV_32F))

        # Chromatic aberration = difference between channels at edges
        chroma = float(np.mean(np.abs(lap_r - lap_b)))

        # Normalise by mean edge strength
        mean_edge = float(np.mean(lap_r + lap_b)) + 1e-6
        chroma_n  = chroma / mean_edge

        threshold = THRESHOLDS["chroma_real_min"]
        if chroma_n < threshold:
            score = 1.0 - (chroma_n / threshold)
            score = min(1.0, score)
            flag  = (f"Missing chromatic aberration ({chroma_n:.4f}) — "
                     "real camera lenses always produce colour fringing at edges")
            return score, flag

        return 0.0, ""

    except Exception as e:
        print(f"[BTD-F] Error: {e}")
        return 0.0, ""


# ================================================================
# SIGNAL G — Colour Kurtosis
# AI images have unnaturally flat colour distributions.
# Real images have peaked, rich colour distributions.
# ================================================================
def signal_g_colour_kurtosis(img: Image.Image) -> Tuple[float, str]:
    """
    Measure kurtosis of LAB colour channels.
    Low kurtosis = flat distribution = platykurtic = AI signal.
    """
    try:
        # Convert to LAB colour space
        img_rgb = img.convert("RGB")
        img_arr = np.array(img_rgb, dtype=np.float32) / 255.0

        # Simple LAB approximation using OpenCV
        img_bgr = cv2.cvtColor(
            (img_arr * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR
        )
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Kurtosis of L, A, B channels
        kurtoses = []
        for ch in range(3):
            channel = lab[:, :, ch].flatten()
            if len(channel) > 0:
                k = float(stats.kurtosis(channel))
                kurtoses.append(k)

        if not kurtoses:
            return 0.0, ""

        mean_kurtosis = float(np.mean(kurtoses))

        threshold = THRESHOLDS["kurtosis_real_min"]
        if mean_kurtosis < threshold:
            score = max(0.0, 1.0 - ((mean_kurtosis + 3) / (threshold + 3)))
            score = min(1.0, score)
            flag  = (f"Flat LAB colour distribution (kurtosis={mean_kurtosis:.2f}) — "
                     "AI colour distributions are unnaturally uniform")
            return score, flag

        return 0.0, ""

    except Exception as e:
        print(f"[BTD-G] Error: {e}")
        return 0.0, ""


# ================================================================
# SIGNAL H — High-Frequency Content
# AI images are too smooth in the frequency domain.
# Real images have rich high-frequency texture.
# ================================================================
def signal_h_high_frequency(img: Image.Image) -> Tuple[float, str]:
    """
    Analyse frequency domain using FFT.
    Low HF ratio = too smooth = AI signal.
    """
    try:
        gray = np.array(img.convert("L"), dtype=np.float32)

        # FFT magnitude spectrum
        fft  = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        mag  = np.abs(fft_shift)

        h, w = mag.shape
        cy, cx = h // 2, w // 2

        # Define LF and HF regions by radius
        lf_r = min(h, w) // 8   # Low frequency: centre circle
        hf_r = min(h, w) // 4   # High frequency: outer ring

        Y, X = np.ogrid[:h, :w]
        dist  = np.sqrt((X - cx)**2 + (Y - cy)**2)

        lf_mask = dist < lf_r
        hf_mask = (dist >= lf_r) & (dist < hf_r)

        lf_energy = float(np.mean(mag[lf_mask])) + 1e-6
        hf_energy = float(np.mean(mag[hf_mask]))

        hf_ratio  = hf_energy / lf_energy

        threshold = THRESHOLDS["hf_ratio_real_min"]
        if hf_ratio < threshold:
            score = 1.0 - (hf_ratio / threshold)
            score = min(1.0, score)
            flag  = (f"Low high-frequency content (ratio={hf_ratio:.4f}) — "
                     "AI images are unnaturally smooth in the frequency domain")
            return score, flag

        return 0.0, ""

    except Exception as e:
        print(f"[BTD-H] Error: {e}")
        return 0.0, ""


# ================================================================
# SIGNAL I — Facial Landmark Symmetry
# Imported from face_utils.measure_facial_symmetry()
# Handled in run_all_signals() below
# ================================================================


# ================================================================
# MAIN FUNCTION — Run all 9 BTD signals
# ================================================================
def run_all_signals(img: Image.Image,
                    face_data: Dict,
                    skip_face_signals: bool = False) -> Dict:
    """
    Run all 9 BTD forensic signals on an image.

    Args:
        img:               PIL Image in RGB
        face_data:         output from face_utils.detect_and_crop()
        skip_face_signals: True if no face found (skips A, B, I)

    Returns:
        Dict with:
            btd_score:  weighted final score (0=real, 1=fake)
            btd_flags:  list of human-readable flag strings
            signals:    dict of individual signal scores
    """
    signals = {}
    flags   = []

    face_mask  = face_data.get("face_mask")
    symmetry   = face_data.get("symmetry", 0.5)

    # Use face crop if available, else full image
    face_crop  = face_data.get("face_crop", img)

    print("[BTD] Running forensic signal analysis...")

    # ── Signal A — Face oval curvature ──────────────────────
    if not skip_face_signals and face_mask is not None:
        score_a, flag_a = signal_a_curvature(img, face_mask)
    else:
        score_a, flag_a = 0.0, ""
    signals["A"] = score_a
    if flag_a:
        flags.append(flag_a)
        print(f"  [A] Curvature: {score_a:.3f} — {flag_a[:40]}...")
    else:
        print(f"  [A] Curvature: {score_a:.3f} — normal")

    # ── Signal B — Boundary gradient mismatch ───────────────
    # Disabled: this signal's Laplacian ratio math produces
    # ceiling values (1.000) on real faces due to near-zero
    # coarse Laplacian at flat boundary pixels (hair/skin/shadow).
    # It has never discriminated real from fake reliably.
    # Documented as future work. Weight redistributed below.
    score_b, flag_b = 0.0, ""
    signals["B"] = score_b
    if flag_b:
        flags.append(flag_b)
        print(f"  [B] Gradient:  {score_b:.3f} — {flag_b[:40]}...")
    else:
        print(f"  [B] Gradient:  {score_b:.3f} — normal")

    # ── Signal C — Edge persistence ──────────────────────────
    score_c, flag_c = signal_c_edge_persistence(face_crop)
    signals["C"] = score_c
    if flag_c:
        flags.append(flag_c)
        print(f"  [C] Edge CV:   {score_c:.3f} — {flag_c[:40]}...")
    else:
        print(f"  [C] Edge CV:   {score_c:.3f} — normal")

    # ── Signal D — ELA ───────────────────────────────────────
    score_d, flag_d = signal_d_ela(img)
    signals["D"] = score_d
    if flag_d:
        flags.append(flag_d)
        print(f"  [D] ELA:       {score_d:.3f} — {flag_d[:40]}...")
    else:
        print(f"  [D] ELA:       {score_d:.3f} — normal")

    # ── Signal E — Noise residual ────────────────────────────
    score_e, flag_e = signal_e_noise(face_crop)
    signals["E"] = score_e
    if flag_e:
        flags.append(flag_e)
        print(f"  [E] Noise:     {score_e:.3f} — {flag_e[:40]}...")
    else:
        print(f"  [E] Noise:     {score_e:.3f} — normal")

    # ── Signal F — Chromatic aberration ─────────────────────
    score_f, flag_f = signal_f_chromatic_aberration(face_crop)
    signals["F"] = score_f
    if flag_f:
        flags.append(flag_f)
        print(f"  [F] Chroma:    {score_f:.3f} — {flag_f[:40]}...")
    else:
        print(f"  [F] Chroma:    {score_f:.3f} — normal")

    # ── Signal G — Colour kurtosis ───────────────────────────
    score_g, flag_g = signal_g_colour_kurtosis(face_crop)
    signals["G"] = score_g
    if flag_g:
        flags.append(flag_g)
        print(f"  [G] Kurtosis:  {score_g:.3f} — {flag_g[:40]}...")
    else:
        print(f"  [G] Kurtosis:  {score_g:.3f} — normal")

    # ── Signal H — High-frequency content ───────────────────
    score_h, flag_h = signal_h_high_frequency(face_crop)
    signals["H"] = score_h
    if flag_h:
        flags.append(flag_h)
        print(f"  [H] HF ratio:  {score_h:.3f} — {flag_h[:40]}...")
    else:
        print(f"  [H] HF ratio:  {score_h:.3f} — normal")

    # ── Signal I — Facial symmetry ───────────────────────────
    if not skip_face_signals:
        threshold_i = THRESHOLDS["symmetry_real_min"]
        if symmetry < threshold_i:
            score_i = 1.0 - (symmetry / threshold_i)
            score_i = min(1.0, score_i)
            flag_i  = (f"Facial symmetry too perfect (deviation={symmetry:.4f}) — "
                       "real faces always have natural micro-asymmetry")
            flags.append(flag_i)
            print(f"  [I] Symmetry:  {score_i:.3f} — {flag_i[:40]}...")
        else:
            score_i = 0.0
            print(f"  [I] Symmetry:  {score_i:.3f} — normal")
    else:
        score_i = 0.0

    signals["I"] = score_i

    # ── Weighted final BTD score ─────────────────────────────
    btd_score = (
        SIGNAL_WEIGHTS["A"] * signals["A"] +
        SIGNAL_WEIGHTS["B"] * signals["B"] +
        SIGNAL_WEIGHTS["C"] * signals["C"] +
        SIGNAL_WEIGHTS["D"] * signals["D"] +
        SIGNAL_WEIGHTS["E"] * signals["E"] +
        SIGNAL_WEIGHTS["F"] * signals["F"] +
        SIGNAL_WEIGHTS["G"] * signals["G"] +
        SIGNAL_WEIGHTS["H"] * signals["H"] +
        SIGNAL_WEIGHTS["I"] * signals["I"]
    )

    btd_score = float(min(1.0, btd_score))

    print(f"[BTD] Final score: {btd_score:.3f} | Flags: {len(flags)}")

    return {
        "btd_score"  : btd_score,
        "btd_flags"  : flags,
        "signals"    : signals
    }
