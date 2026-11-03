#!/usr/bin/env python3
"""
bench.py — time the app's real Cull → Dedup → Rank steps, and compare two
versions of the app (e.g. this branch against upstream) on the same photos.

    # this checkout
    .venv/bin/python bench.py run FOLDER --out fast.json
    # a checkout of upstream, e.g. `git worktree add ../stock upstream/main`
    .venv/bin/python bench.py run FOLDER --app ../stock --out stock.json
    .venv/bin/python bench.py compare stock.json fast.json

Stock Dedup writes .dedup_sig_cache.json next to the photos (deleted again
afterwards). To leave a photo folder untouched, point FOLDER at a folder of
symlinks to it.
"""

import argparse
import json
import sys
import time
from pathlib import Path


RANK_FIELDS = ['overall_score', 'composition', 'technical', 'sharpness', 'color',
               'aesthetic', 'exposure', 'dynamic_range', 'noise', 'harmony']


def run(folder, app_dir, out):
    sys.path.insert(0, str(Path(app_dir).resolve()))
    import photo_curator as pc
    print(f"app: {Path(pc.__file__).parent}", flush=True)

    cache = Path(folder) / '.dedup_sig_cache.json'
    had_cache = cache.exists()
    t = {}
    t0 = time.time(); pc.run_cull(folder, 1.0, True, True); t['cull'] = time.time() - t0
    t0 = time.time(); pc.run_dedup(folder, 0.8); t['dedup'] = time.time() - t0
    t0 = time.time(); pc.run_rank(folder); t['rank'] = time.time() - t0
    if not had_cache:
        cache.unlink(missing_ok=True)
    st = pc.state
    for step in ('cull', 'dedup', 'rank'):
        if st[step]['status'].startswith('Error'):
            raise SystemExit(f"{step}: {st[step]['status']}")

    res = {
        'app': str(Path(pc.__file__).parent), 'folder': folder,
        'times': {k: round(v, 2) for k, v in t.items()},
        'tiers': {p['path']: p['tier'] for p in st['cull']['photos']},
        'cull_kept': st['cull']['sharp_paths'],
        'dedup_kept': st['dedup']['kept_paths'],
        'rank': {s.path: {f: getattr(s, f) for f in RANK_FIELDS}
                 for s in st['rank']['scores']},
    }
    n = len(res['tiers'])
    print(f"cull  {t['cull']:6.1f}s  ({n} photos)\n"
          f"dedup {t['dedup']:6.1f}s  ({len(res['cull_kept'])} in, {len(res['dedup_kept'])} unique)\n"
          f"rank  {t['rank']:6.1f}s  ({len(res['rank'])} ranked)\n"
          f"total {sum(t.values()):6.1f}s  → {n / sum(t.values()):.1f} photos/s")
    Path(out).write_text(json.dumps(res))


def compare(a_file, b_file, topn=50):
    a, b = (json.loads(Path(f).read_text()) for f in (a_file, b_file))
    ta, tb = sum(a['times'].values()), sum(b['times'].values())
    print(f"A {a['app']}\n  {a['times']}  total {ta:.1f}s")
    print(f"B {b['app']}\n  {b['times']}  total {tb:.1f}s")
    print(f"speedup A→B: {ta / tb:.1f}x\n")

    common = set(a['tiers']) & set(b['tiers'])
    same = sum(a['tiers'][p] == b['tiers'][p] for p in common)
    print(f"cull tiers identical:  {same}/{len(common)}")
    ka, kb = set(a['cull_kept']), set(b['cull_kept'])
    print(f"cull kept:             {len(ka)} vs {len(kb)} ({len(ka ^ kb)} differ)")
    da, db = set(a['dedup_kept']), set(b['dedup_kept'])
    print(f"dedup unique:          {len(da)} vs {len(db)} ({len(da ^ db)} differ)")
    rp = sorted(set(a['rank']) & set(b['rank']))
    if rp:
        diffs = {f: max(abs(a['rank'][p][f] - b['rank'][p][f]) for p in rp)
                 for f in RANK_FIELDS}
        print(f"rank, max abs diff over {len(rp)} photos: "
              + ", ".join(f"{f} {d:.2f}" for f, d in diffs.items()))
        top = [set(sorted(rp, key=lambda p, r=r: -r[p]['overall_score'])[:topn])
               for r in (a['rank'], b['rank'])]
        print(f"rank top-{topn} overlap:   {len(top[0] & top[1])}/{min(topn, len(rp))}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('folder')
    r.add_argument('--app', default=str(Path(__file__).parent),
                   help='checkout whose photo_curator.py to run (default: this one)')
    r.add_argument('--out', required=True)
    c = sub.add_parser('compare')
    c.add_argument('a')
    c.add_argument('b')
    args = ap.parse_args()
    if args.cmd == 'run':
        run(str(Path(args.folder).resolve()), args.app, args.out)
    else:
        compare(args.a, args.b)


if __name__ == '__main__':
    main()
