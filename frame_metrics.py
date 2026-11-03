"""frame_metrics.py — per-frame sharpness/quality metrics shared by Cull and Dedup.

Moved out of photo_curator.py unchanged, so that fast_pipeline's worker
processes can use them without importing the Flask app.
"""

import cv2
import numpy as np


def sharpness_score(gray):
    """Whole-frame contrast-normalized focus measure (used by Dedup's quality
    key). Haze/low contrast does NOT read as blur."""
    h, w = gray.shape[:2]
    long_edge = max(h, w)
    if long_edge > 1024:
        sc = 1024.0 / long_edge
        gray = cv2.resize(gray, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    img_var = float(gray.astype('float32').var()) + 1e-6
    return lap_var / img_var * 1000.0


def region_sharpness(gray, tiles=8):
    """REGION-based sharpness for Cull: tile the frame, score each tile's
    contrast-normalized focus, and return a high percentile. This measures the
    SHARPEST meaningful region, so a tack-sharp subject against soft bokeh/sky
    is correctly recognized as in-focus instead of being dragged down by the
    smooth background (the key v3.1 accuracy fix)."""
    h, w = gray.shape[:2]
    long_edge = max(h, w)
    if long_edge > 1024:
        sc = 1024.0 / long_edge
        gray = cv2.resize(gray, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(gray, cv2.CV_64F)          # uint8 -> CV_64F (supported)
    g = gray.astype('float32')
    H, W = g.shape
    th, tw = max(1, H // tiles), max(1, W // tiles)
    scores = []
    for y in range(0, H, th):
        for x in range(0, W, tw):
            tg = g[y:y + th, x:x + tw]
            if tg.size < 64:
                continue
            v = float(tg.var()) + 1e-6
            scores.append(float(lap[y:y + th, x:x + tw].var()) / v * 1000.0)
    if not scores:
        return 0.0
    return float(np.percentile(np.array(scores), 90))


def quick_quality(bgr, gray):
    """Lightweight 0–100 'is this an interesting, detailed frame' proxy used
    only to RESCUE soft-but-well-composed shots from culling. Cheap on purpose;
    the real aesthetic judgement happens in the Rank step."""
    edges = cv2.Canny(gray, 60, 160)
    ed = float(np.count_nonzero(edges)) / edges.size
    contrast = float(gray.std())
    b, gg, r = (bgr[:, :, i].astype('float32') for i in range(3))
    rg, yb = r - gg, 0.5 * (r + gg) - b
    cf = (rg.std() ** 2 + yb.std() ** 2) ** 0.5 + 0.3 * (rg.mean() ** 2 + yb.mean() ** 2) ** 0.5
    q = 0.5 * min(1, ed / 0.12) + 0.25 * min(1, contrast / 70) + 0.25 * min(1, cf / 80)
    return float(max(0.0, min(1.0, q)) * 100)
