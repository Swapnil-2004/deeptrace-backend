"""
================================================================
DeepTrace — detector/ensemble.py
Weighted ensemble — combines all detector scores into
one final verdict with confidence.

Image weights:
  EfficientNet : 35%  (deepfake specialist)
  ViT          : 25%  (AI image specialist)
  BTD signals  : 30%  (mam's forensic algorithm)
  Metadata     : 10%  (supporting signal)

Video weights:
  Mean frame score  : 35%
  Max frame score   : 15%
  Fake frame ratio  : 20%
  Temporal BTD      : 20%
  Flow consistency  : 10%

Override rules:
  - EfficientNet > 0.90 alone → final = 0.90 FAKE
  - Definitive AI metadata    → final = 0.95 FAKE
  - No face + ViT high        → boost ViT weight
================================================================
"""

from typing import Dict, List, Optional


# ================================================================
# IMAGE ENSEMBLE WEIGHTS
# ================================================================
IMAGE_WEIGHTS = {
    "effnet" : 0.35,
    "vit"    : 0.25,
    "btd"    : 0.30,
    "meta"   : 0.10,
}

# Adjusted weights when no face is found
NO_FACE_WEIGHTS = {
    "effnet" : 0.35,
    "vit"    : 0.35,   # ViT gets more weight — better for full images
    "btd"    : 0.20,   # BTD gets less — face signals unavailable
    "meta"   : 0.10,
}

# Override thresholds
EFFNET_OVERRIDE_THRESHOLD = 0.90   # Very high effnet alone = FAKE
META_OVERRIDE_THRESHOLD   = 0.80   # Definitive metadata = FAKE
FAKE_THRESHOLD            = 0.50   # Final score above this = FAKE


# ================================================================
# VIDEO ENSEMBLE WEIGHTS
# ================================================================
VIDEO_WEIGHTS = {
    "mean_score"    : 0.35,
    "max_score"     : 0.15,
    "fake_ratio"    : 0.20,
    "temporal_btd"  : 0.20,
    "flow"          : 0.10,
}


# ================================================================
# IMAGE ENSEMBLE
# ================================================================
def compute_image_ensemble(
    effnet_score : float,
    vit_score    : float,
    btd_score    : float,
    meta_score   : float,
    face_found   : bool = True
) -> Dict:
    """
    Combine all image detection scores into final verdict.

    Args:
        effnet_score: EfficientNet fake probability 0-1
        vit_score:    ViT fake probability 0-1
        btd_score:    BTD forensic score 0-1
        meta_score:   Metadata analysis score 0-1
        face_found:   whether face was detected

    Returns:
        Dict with:
            final_score:  float 0-1
            prediction:   "Fake" or "Real"
            confidence:   float 0-100
            override:     bool — whether override rule fired
            override_reason: str
            weights_used: dict of actual weights used
    """
    override        = False
    override_reason = ""

    # ── Override Rule 1 — Very high EfficientNet alone ──────
    if effnet_score > EFFNET_OVERRIDE_THRESHOLD:
        final_score     = 0.90
        override        = True
        override_reason = (f"EfficientNet very high confidence "
                          f"({effnet_score:.3f}) — override to FAKE")
        print(f"[Ensemble] Override 1: effnet={effnet_score:.3f}")

    # ── Override Rule 2 — Definitive metadata signature ─────
    elif meta_score > META_OVERRIDE_THRESHOLD:
        final_score     = 0.95
        override        = True
        override_reason = (f"Definitive AI metadata signature "
                          f"({meta_score:.3f}) — override to FAKE")
        print(f"[Ensemble] Override 2: meta={meta_score:.3f}")

    else:
        # ── Normal weighted ensemble ─────────────────────────
        weights = IMAGE_WEIGHTS if face_found else NO_FACE_WEIGHTS

        final_score = (
            weights["effnet"] * effnet_score +
            weights["vit"]    * vit_score    +
            weights["btd"]    * btd_score    +
            weights["meta"]   * meta_score
        )
        final_score = float(min(1.0, max(0.0, final_score)))

        print(f"[Ensemble] Weighted: effnet={effnet_score:.3f} "
              f"vit={vit_score:.3f} btd={btd_score:.3f} "
              f"meta={meta_score:.3f} → {final_score:.3f}")

    # ── Verdict ──────────────────────────────────────────────
    prediction = "Fake" if final_score >= FAKE_THRESHOLD else "Real"

    # ── Confidence — how far from 0.5 ────────────────────────
    confidence = abs(final_score - 0.50) * 200
    confidence = min(99.0, max(1.0, confidence))

    print(f"[Ensemble] Final: {prediction} | "
          f"score={final_score:.3f} | confidence={confidence:.1f}%")

    return {
        "final_score"    : round(final_score, 4),
        "prediction"     : prediction,
        "confidence"     : round(confidence, 1),
        "override"       : override,
        "override_reason": override_reason,
        "weights_used"   : IMAGE_WEIGHTS if face_found else NO_FACE_WEIGHTS,
        "individual"     : {
            "effnet": round(effnet_score, 4),
            "vit"   : round(vit_score,    4),
            "btd"   : round(btd_score,    4),
            "meta"  : round(meta_score,   4),
        }
    }


# ================================================================
# VIDEO ENSEMBLE
# ================================================================
def compute_video_ensemble(
    frame_scores   : List[float],
    temporal_score : float,
    flow_score     : float
) -> Dict:
    """
    Combine all video detection scores into final verdict.

    Args:
        frame_scores:   list of per-frame fake scores
        temporal_score: temporal BTD score 0-1
        flow_score:     optical flow score 0-1

    Returns:
        Dict with final_score, prediction, confidence,
        fake_frame_ratio, and component scores
    """
    if not frame_scores:
        return {
            "final_score"     : 0.5,
            "prediction"      : "Real",
            "confidence"      : 1.0,
            "fake_frame_ratio": 0.0,
            "mean_score"      : 0.5,
            "max_score"       : 0.5,
        }

    # Component scores
    mean_score  = float(sum(frame_scores) / len(frame_scores))
    max_score   = float(max(frame_scores))
    fake_ratio  = float(
        sum(1 for s in frame_scores if s >= FAKE_THRESHOLD) / len(frame_scores)
    )

    # Override — very high temporal BTD alone
    if temporal_score > 0.85:
        final_score     = 0.88
        override        = True
        override_reason = f"Temporal BTD very high ({temporal_score:.3f})"
        print(f"[VideoEnsemble] Temporal override: {temporal_score:.3f}")
    else:
        # Weighted combination
        final_score = (
            VIDEO_WEIGHTS["mean_score"]   * mean_score    +
            VIDEO_WEIGHTS["max_score"]    * max_score     +
            VIDEO_WEIGHTS["fake_ratio"]   * fake_ratio    +
            VIDEO_WEIGHTS["temporal_btd"] * temporal_score +
            VIDEO_WEIGHTS["flow"]         * flow_score
        )
        final_score     = float(min(1.0, max(0.0, final_score)))
        override        = False
        override_reason = ""

        print(f"[VideoEnsemble] mean={mean_score:.3f} max={max_score:.3f} "
              f"ratio={fake_ratio:.3f} temporal={temporal_score:.3f} "
              f"flow={flow_score:.3f} → {final_score:.3f}")

    prediction = "Fake" if final_score >= FAKE_THRESHOLD else "Real"
    confidence = abs(final_score - 0.50) * 200
    confidence = min(99.0, max(1.0, confidence))

    print(f"[VideoEnsemble] Final: {prediction} | "
          f"score={final_score:.3f} | confidence={confidence:.1f}%")

    return {
        "final_score"     : round(final_score, 4),
        "prediction"      : prediction,
        "confidence"      : round(confidence, 1),
        "fake_frame_ratio": round(fake_ratio,  4),
        "mean_score"      : round(mean_score,  4),
        "max_score"       : round(max_score,   4),
        "temporal_score"  : round(temporal_score, 4),
        "flow_score"      : round(flow_score,  4),
        "override"        : override,
        "override_reason" : override_reason,
    }
