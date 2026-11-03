#!/usr/bin/env python3
"""
fast_pipeline.py — one-decode, multi-core analysis for Cull + Dedup + Rank.

The stock app runs each step as its own single-threaded loop, and each step
re-decodes every photo (Dedup alone decodes a RAW 3–4 times, and builds its
8x8 hash from a FULL-resolution decode). This module:

  • decodes each photo ONCE, at the same 1/2 scale the app already uses
    (IMREAD_REDUCED_COLOR_2), reading EXIF time from the header. A 1/4 decode
    was tried: on 8 MB camera JPEGs it is barely faster (entropy decoding
    dominates, not pixel output) and it shifts the Laplacian sharpness score
    by ~15 points, so it isn't worth it;
  • computes every per-photo metric from that one decode — Cull's region
    sharpness + quick quality, Dedup's signature + ORB features, Rank's full
    v3 breakdown — reusing the app's own functions so the math is unchanged;
  • fans the photos out over a process pool (one per core);
  • then runs the app's own FastBatchDeduplicator clustering serially on the
    precomputed signatures, which is cheap (vectorized XOR/popcount).
"""

import io
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from raw_loader import is_raw, is_heif, _embedded_jpeg, imread_bgr
from photo_dedup_batch import FastBatchDeduplicator
from photo_ranking_v3 import AdvancedPhotoAnalyzer

WORK = 1024        # long edge every metric is computed at (matches the app)
ORB_H = 480        # ORB confirmation works at this height (matches the app)


def _exif_time(fp):
    """Same rule as FastBatchDeduplicator._capture_time: DateTimeOriginal,
    else DateTime, never mtime. Image.open only parses the header here."""
    try:
        exif = Image.open(fp).getexif()
        for tag in (36867, 306):
            val = exif.get(tag)
            if val:
                return time.mktime(time.strptime(str(val), "%Y:%m:%d %H:%M:%S"))
    except Exception:
        pass
    return None


def _load_small(path):
    """(bgr, capture_time) from a single reduced decode."""
    if is_raw(path):
        data = _embedded_jpeg(path)
        if data is not None:
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
            if bgr is not None:
                return bgr, _exif_time(io.BytesIO(data))
        return imread_bgr(path), _exif_time(path)        # demosaic fallback
    if is_heif(path):
        return imread_bgr(path), _exif_time(path)        # no DCT scaling for HEIC
    bgr = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    if bgr is None:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    return bgr, _exif_time(path)


def _fit(img, long_edge):
    h, w = img.shape[:2]
    if max(h, w) <= long_edge:
        return img
    sc = long_edge / max(h, w)
    return cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)


def _signature(gray):
    """Same 192-bit ahash + dhash_h + dhash_v as FastBatchDeduplicator, built
    from the already-decoded frame instead of a fresh full-res decode."""
    img = Image.fromarray(gray)
    a = np.asarray(img.resize((8, 8), Image.BILINEAR), dtype=np.float32)
    dh = np.asarray(img.resize((9, 8), Image.BILINEAR), dtype=np.float32)
    dv = np.asarray(img.resize((8, 9), Image.BILINEAR), dtype=np.float32)
    return np.concatenate([(a > a.mean()).flatten(),
                           (dh[:, 1:] > dh[:, :-1]).flatten(),
                           (dv[1:, :] > dv[:-1, :]).flatten()]).astype(bool)


class _Preloaded(AdvancedPhotoAnalyzer):
    """The stock v3 analyzer, fed an already-decoded frame."""

    def __init__(self, bgr, gray):
        super().__init__(work_long_edge=WORK)
        self._arrays = (bgr, gray)

    def _load(self, image_path):
        return self._arrays


_orb = None


def analyze_one(path):
    """Every per-photo metric the three steps need, from one decode."""
    # Imported here so worker processes don't pay for it until first use.
    from photo_curator import region_sharpness, quick_quality, sharpness_score
    global _orb
    if _orb is None:
        cv2.setNumThreads(1)          # parallelism comes from the pool instead
        _orb = cv2.ORB_create(nfeatures=300)
    big, ts = _load_small(path)
    if big is None:
        return {'path': path, 'ok': False}
    big_gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    # Cull's metrics run on the half-size frame, exactly as run_cull does
    # (region_sharpness downsizes to 1024 itself; quick_quality doesn't).
    region_s = region_sharpness(big_gray)
    q = quick_quality(big, big_gray)
    bgr = _fit(big, WORK)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    orb_gray = _fit(gray, ORB_H) if gray.shape[0] > ORB_H else gray
    k, d = _orb.detectAndCompute(orb_gray, None)
    return {
        'path': path, 'ok': True, 'ts': ts,
        'region_s': region_s,
        'q': q,
        'sharp': sharpness_score(gray),
        'sig': _signature(gray),
        'orb': (len(k) if k is not None else 0, d),
        'rank': _Preloaded(bgr, gray).analyze_image(path),
    }


def analyze_all(paths, workers=None, progress=None):
    """Analyze in parallel; results come back in input order."""
    paths = [str(p) for p in paths]
    out = []
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(analyze_one, paths, chunksize=4)):
            out.append(r)
            if progress:
                progress(i + 1, len(paths))
    return out


def dedup(results, threshold=0.8):
    """Run the app's own clustering over precomputed signatures/ORB/time."""
    from photo_curator import _LiteScore
    dd = FastBatchDeduplicator(threshold=threshold)
    dd.reset()
    for r in results:
        if not r['ok']:
            continue
        p = r['path']
        dd._sig_cache[p] = r['sig']
        dd._ts_cache[p] = r['ts']
        dd._orb_desc_cache[p] = r['orb']
        dd.add_photo(_LiteScore(p, Path(p).name, r['sharp'], r['sharp']))
    return dd
