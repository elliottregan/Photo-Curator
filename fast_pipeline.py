#!/usr/bin/env python3
"""
fast_pipeline.py — one-decode, multi-core analysis for Cull + Dedup + Rank.

The stock app ran each step as its own single-threaded loop, and each step
re-decoded every photo (Dedup alone decoded a RAW 3–4 times, and built its
8x8 hash from a FULL-resolution decode). This module:

  • decodes each photo ONCE, at the same 1/2 scale the app already used
    (IMREAD_REDUCED_COLOR_2), reading EXIF time from the header. A 1/4 decode
    was tried: on 8 MB camera JPEGs it is barely faster (entropy decoding
    dominates, not pixel output) and it shifts the Laplacian sharpness score
    by ~15 points, so it isn't worth it;
  • computes every per-photo metric from that one decode — Cull's region
    sharpness + quick quality, Dedup's signature + ORB features, Rank's full
    v3 breakdown — with the app's own functions, so the math is unchanged;
  • fans the photos out over a process pool (one worker per core) that lives
    for the whole session, and caches each result in memory. Whichever step
    runs first pays for the decode; the later steps reuse it;
  • leaves Dedup's clustering to the app's own FastBatchDeduplicator, run
    serially over the precomputed signatures (cheap: vectorized XOR/popcount).
"""

import dataclasses
import io
import logging
import os
import threading
import time
from concurrent.futures import (ProcessPoolExecutor, CancelledError,
                                FIRST_COMPLETED, wait)
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from raw_loader import is_raw, is_heif, _embedded_jpeg, imread_bgr
from frame_metrics import region_sharpness, quick_quality, sharpness_score
from photo_ranking_v3 import AdvancedPhotoAnalyzer

logger = logging.getLogger(__name__)

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


def _init_worker():
    global _orb
    cv2.setNumThreads(1)          # parallelism comes from the pool instead
    _orb = cv2.ORB_create(nfeatures=300)


def analyze_one(path):
    """Every per-photo metric the three steps need, from one decode."""
    try:
        if _orb is None:
            _init_worker()
        big, ts = _load_small(path)
        if big is None:
            return {'path': path, 'ok': False}
        big_gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        # Cull's metrics run on the half-size frame, exactly as run_cull did
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
    except Exception as e:
        logger.warning(f"analyze failed {path}: {e}")
        return {'path': path, 'ok': False}


def prime_dedup(dd, r):
    """Hand a result's signature / capture time / ORB features to a
    FastBatchDeduplicator, so its add_photo() decodes nothing. A photo that
    failed to decode gets no signature, so it stays its own cluster — what
    the deduplicator did when its own decode failed."""
    p = r['path']
    ok = r['ok']
    dd._sig_cache[p] = r['sig'] if ok else None
    dd._ts_cache[p] = r['ts'] if ok else None
    dd._orb_desc_cache[p] = r['orb'] if ok else (0, None)


def _key(path):
    """Cache key that survives the app moving a file to Blurred/ and back
    (shutil.move keeps size and mtime)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (Path(path).name, st.st_size, st.st_mtime_ns)


def _at_path(r, path):
    """A cached result re-pointed at the file's current path."""
    if r['path'] == path:
        return r
    r = dict(r, path=path)
    if r.get('rank') is not None:
        r['rank'] = dataclasses.replace(r['rank'], path=path,
                                        filename=Path(path).name)
    return r


class Analyzer:
    """Process pool + in-memory result cache shared by Cull, Dedup and Rank.

    Safe to call from several step threads at once: a photo already being
    analyzed for one step is waited on, not analyzed twice.
    """

    def __init__(self, workers=None):
        self.workers = workers or os.cpu_count() or 4
        self._pool = None
        # Re-entrant: a future that is already done runs its done-callback
        # (_store) immediately, inside _submit, with the lock held.
        self._lock = threading.RLock()
        self._done = {}        # key -> result
        self._inflight = {}    # key -> Future

    def _submit(self, path):
        """Future for path (reusing an in-flight one). Caller holds _lock."""
        key = _key(path)
        f = self._inflight.get(key) if key is not None else None
        if f is None or f.cancelled():
            if self._pool is None:
                self._pool = ProcessPoolExecutor(max_workers=self.workers,
                                                 initializer=_init_worker)
            f = self._pool.submit(analyze_one, path)
            f.pool = self._pool
            if key is not None:
                self._inflight[key] = f
                f.add_done_callback(lambda f, key=key: self._store(key, f))
        return f

    def _store(self, key, f):
        with self._lock:
            if self._inflight.get(key) is f:
                del self._inflight[key]
            if not f.cancelled() and f.exception() is None and f.result()['ok']:
                self._done[key] = f.result()

    def _reset_pool(self, broken):
        """Drop a broken pool. Every future it held fails at once, so only
        the first report replaces it; later ones see a newer pool and keep it."""
        with self._lock:
            if self._pool is not broken:
                return
            broken.shutdown(wait=False, cancel_futures=True)
            self._pool = None
            self._inflight = {k: f for k, f in self._inflight.items()
                              if f.pool is not broken}

    def run(self, paths, on_result, cancelled=lambda: False):
        """Analyze paths, calling on_result(index, result) as each finishes —
        cached photos first and instantly, the rest in completion order.

        Returns True when every photo was delivered, False if cancelled()
        turned true first (photos this call queued but hadn't started are
        dropped; any that finish anyway are still cached)."""
        paths = [str(p) for p in paths]
        futs, ready = {}, []
        with self._lock:
            for i, p in enumerate(paths):
                r = self._done.get(_key(p))
                if r is not None:
                    ready.append((i, r))
                else:
                    futs[self._submit(p)] = i
        mine = set(futs)
        for i, r in ready:
            on_result(i, _at_path(r, paths[i]))
        retried = set()
        pending = set(futs)
        while pending:
            if cancelled():
                for f in pending & mine:
                    f.cancel()
                return False
            done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            for f in done:
                i = futs.pop(f)
                try:
                    r = f.result()
                except CancelledError:
                    # Another step queued this photo, then was stopped.
                    with self._lock:
                        nf = self._submit(paths[i])
                    futs[nf] = i
                    mine.add(nf)
                    pending.add(nf)
                    continue
                except BrokenProcessPool:
                    # A worker died (e.g. a decoder crash on a corrupt file).
                    # Start a fresh pool and retry each affected photo once.
                    self._reset_pool(f.pool)
                    if i not in retried:
                        retried.add(i)
                        with self._lock:
                            nf = self._submit(paths[i])
                        futs[nf] = i
                        mine.add(nf)
                        pending.add(nf)
                        continue
                    logger.warning(f"analysis crashed twice, skipping {paths[i]}")
                    r = {'path': paths[i], 'ok': False}
                on_result(i, _at_path(r, paths[i]))
        return True
