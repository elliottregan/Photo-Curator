#!/usr/bin/env python3
"""
bench.py — time the stock Cull → Dedup → Rank chain against fast_pipeline,
and check the two agree.

    .venv/bin/python bench.py FOLDER [--skip-stock] [--workers N]

The fast path only reads FOLDER, but the stock dedup step writes
.dedup_sig_cache.json next to the photos (and bench.py deletes it afterwards).
To leave a photo folder untouched, point this at a folder of symlinks to it.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import photo_curator as pc
import fast_pipeline as fp


def cull_tiers(items, strictness=1.0, adaptive=True, rescue_on=True):
    """run_cull's final classify_all(), lifted out of its closure."""
    if adaptive and items:
        M = float(np.median([it['region_s'] for it in items]))
        blur_lo = max(30.0, M * 0.25)
        sharp_hi = max(blur_lo * 1.5, M * 0.70)
    else:
        blur_lo, sharp_hi = pc.BASE_BLUR, pc.BASE_SHARP
    blur_lo *= strictness
    sharp_hi *= strictness
    qs = [it['q'] for it in items]
    q_rescue = float(np.percentile(qs, 70)) if qs else 65.0
    return {it['path']: pc.classify_sharpness(it['region_s'], it['q'], blur_lo,
                                              sharp_hi, q_rescue, rescue_on)[0]
            for it in items}


def run_stock(folder):
    t = {}
    cache = Path(folder) / '.dedup_sig_cache.json'
    cache.unlink(missing_ok=True)
    t0 = time.time(); pc.run_cull(folder, 1.0, True, True); t['cull'] = time.time() - t0
    t0 = time.time(); pc.run_dedup(folder, 0.8); t['dedup'] = time.time() - t0
    t0 = time.time(); pc.run_rank(folder); t['rank'] = time.time() - t0
    cache.unlink(missing_ok=True)
    st = pc.state
    for step in ('cull', 'dedup', 'rank'):
        if st[step]['status'].startswith('Error'):
            raise SystemExit(f"stock {step}: {st[step]['status']}")
    return t, {
        'tiers': {p['path']: p['tier'] for p in st['cull']['photos']},
        'cull_kept': st['cull']['sharp_paths'],
        'dedup_kept': st['dedup']['kept_paths'],
        'rank': {s.path: s for s in st['rank']['scores']},
    }


def run_fast(folder, workers):
    paths = [str(p) for p in pc.list_images(folder)]
    t0 = time.time()
    res = fp.analyze_all(paths, workers=workers)
    t_an = time.time() - t0
    ok = [r for r in res if r['ok']]
    t0 = time.time()
    tiers = cull_tiers(ok)
    kept = [r for r in ok if tiers[r['path']] != 'blurry']
    dd = fp.dedup(kept)
    survivors = [s.path for s in dd.current_survivors()]
    t_cl = time.time() - t0
    return {'analyze': t_an, 'cluster': t_cl}, {
        'tiers': tiers,
        'cull_kept': [r['path'] for r in kept],
        'dedup_kept': survivors,
        'rank': {r['path']: r['rank'] for r in ok if r['rank']},
        'n': len(paths),
    }


def compare(a, b, topn=50):
    common = set(a['tiers']) & set(b['tiers'])
    tier_agree = sum(a['tiers'][p] == b['tiers'][p] for p in common) / max(len(common), 1)
    keep_a, keep_b = set(a['cull_kept']), set(b['cull_kept'])
    dk_a, dk_b = set(a['dedup_kept']), set(b['dedup_kept'])
    rp = sorted(set(a['rank']) & set(b['rank']))
    ov_a = np.array([a['rank'][p].overall_score for p in rp])
    ov_b = np.array([b['rank'][p].overall_score for p in rp])
    # Top-N overlap restricted to photos both versions ranked.
    top_a = set(sorted(rp, key=lambda p: -a['rank'][p].overall_score)[:topn])
    top_b = set(sorted(rp, key=lambda p: -b['rank'][p].overall_score)[:topn])
    return {
        'cull_tier_agreement': round(tier_agree, 3),
        'cull_kept': [len(keep_a), len(keep_b)],
        'cull_kept_jaccard': round(len(keep_a & keep_b) / max(len(keep_a | keep_b), 1), 3),
        'dedup_unique': [len(dk_a), len(dk_b)],
        'dedup_kept_jaccard': round(len(dk_a & dk_b) / max(len(dk_a | dk_b), 1), 3),
        'rank_overall_mean_abs_diff': round(float(np.abs(ov_a - ov_b).mean()), 2),
        'rank_overall_max_abs_diff': round(float(np.abs(ov_a - ov_b).max()), 2),
        'rank_overall_pearson': round(float(np.corrcoef(ov_a, ov_b)[0, 1]), 4),
        f'rank_top{topn}_overlap': len(top_a & top_b),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--skip-stock', action='store_true')
    ap.add_argument('--workers', type=int, nargs='+', default=[os.cpu_count()])
    args = ap.parse_args()
    folder = str(Path(args.folder).resolve())
    report = {'folder': folder}

    fast = None
    for w in args.workers:
        t, fast = run_fast(folder, w)
        report[f'fast_w{w}'] = {k: round(v, 2) for k, v in t.items()}
        print(f"fast  workers={w:<2} analyze {t['analyze']:.1f}s  cluster {t['cluster']:.2f}s"
              f"  → {fast['n'] / (t['analyze'] + t['cluster']):.1f} photos/s", flush=True)

    if not args.skip_stock:
        t, stock = run_stock(folder)
        report['stock'] = {k: round(v, 2) for k, v in t.items()}
        n = len(stock['tiers'])
        print(f"stock cull {t['cull']:.1f}s ({n / t['cull']:.1f}/s)  "
              f"dedup {t['dedup']:.1f}s ({len(stock['cull_kept']) / t['dedup']:.1f}/s)  "
              f"rank {t['rank']:.1f}s ({len(stock['dedup_kept']) / t['rank']:.1f}/s)  "
              f"total {sum(t.values()):.1f}s", flush=True)
        report['agreement'] = compare(stock, fast)
        print(json.dumps(report['agreement'], indent=2))

    print(json.dumps(report))


if __name__ == '__main__':
    main()
