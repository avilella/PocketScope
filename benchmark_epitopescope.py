#!/usr/bin/env python3
"""Retrospective validation of EpitopeScope against solved nanobody-antigen complexes.

Section 4.5 of the workflow preprint asks for a retrospective step that quantifies ranking
enrichment of known references rather than only reporting the pipeline's own scores. The
reference set here is `datasets/nanobody_bench`: antigen chains taken out of solved nanobody
complexes, where the true epitope is whatever the nanobody actually contacts. The nanobody is
stripped from the input, so nothing about the answer reaches the detector.

Two separable questions are reported, because they can fail independently:

  detection - does the site enumerator propose the true epitope at all, anywhere in its
              candidate list?
  ranking   - given that it does, how high does the scoring push it, against the null of
              ranking the same candidates at random?

    python benchmark_epitopescope.py --embed esm2 --device cpu
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_truth(tsv: Path):
    truth = {}
    for row in csv.DictReader(tsv.open(), delimiter="\t"):
        key = f"{row['pdb_id']}_{row['antigen_chain']}"
        truth[key] = {int(re.sub(r"[^0-9-]", "", t)) for t in row["epitope_resseq"].split()
                      if re.sub(r"[^0-9-]", "", t)}
    return truth


def predicted_residues(cell: str):
    return {int(m) for m in re.findall(r"-?\d+", cell)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 0


def binomial_ci(hits: int, n: int, replicates: int = BOOTSTRAP_REPLICATES,
                seed: int = BOOTSTRAP_SEED):
    """Deterministic 95 % percentile interval for a proportion.

    n is 30 here, so a bare point estimate is misleading: recall@1 of 23 % carries an
    interval roughly twice its own width. Seeded so the reported bounds are reproducible,
    following the same convention as Odin-Multi's summary figures.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.binomial(n, hits / n, size=replicates) / n
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--inputdir", default=str(HERE / "datasets" / "nanobody_bench"))
    ap.add_argument("--truth", default=None)
    ap.add_argument("--outdir", default=None, help="where the CSVs go (default: alongside)")
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--embed", default="esm2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-patches", type=int, default=12)
    ap.add_argument("--recall-cut", type=float, default=0.25,
                    help="fraction of true epitope residues a candidate must recover to count")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--skip-run", action="store_true", help="score existing CSVs only")
    ap.add_argument("--script", default=None,
                    help="path to the epitopescope build to test; defaults to the one beside "
                         "this script. Use it to A/B two versions against the same truth")
    ap.add_argument("--rank-by", default="rank",
                    help="column to rank candidates by; 'rank' uses the tool's own order")
    ap.add_argument("--descending", action="store_true",
                    help="with --rank-by, treat a larger value as better")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    ap.add_argument("--extra-args", default="",
                    help="extra flags passed through to epitopescope, e.g. "
                         "\"--specificity-ratio 0\"")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    indir = Path(a.inputdir).resolve()
    truth = load_truth(Path(a.truth) if a.truth else indir / "true_epitopes.tsv")
    outdir = Path(a.outdir).resolve() if a.outdir else indir
    log(f"{len(truth)} reference complexes in {indir}")

    if not a.skip_run:
        cmd = [sys.executable, a.script or str(HERE / "epitopescope.py"), "-d", str(indir),
               "--outdir", str(outdir), "--tag", a.tag, "--embed", a.embed,
               "--device", a.device, "--max-patches", str(a.max_patches), "--no-panel"]
        if a.extra_args:
            cmd += shlex.split(a.extra_args)
        if a.refresh:
            cmd.append("--refresh")
        log("running: " + " ".join(cmd))
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if r.returncode != 0:
            log(f"epitopescope exited {r.returncode}")
            return r.returncode

    rng = np.random.default_rng(a.seed)
    rows, rand_ranks = [], []
    for key, true_res in sorted(truth.items()):
        csv_path = outdir / f"{key}.{a.tag}.csv"
        if not csv_path.exists():
            log(f"  ! no output for {key}")
            continue
        cands = [c for c in csv.DictReader(csv_path.open())
                 if str(c.get("rank", "")).strip() != ""]
        if not cands:
            continue
        if a.rank_by != "rank":
            def key(c):
                try:
                    return float(c[a.rank_by])
                except (ValueError, KeyError):
                    return 0.0
            cands.sort(key=key, reverse=a.descending)
            for i, c in enumerate(cands, 1):
                c["rank"] = i
        best_j, best_rec, best_id = 0.0, 0.0, ""
        hit_ranks = []
        for c in cands:
            pred = predicted_residues(c["epitope_residues"])
            rec = len(pred & true_res) / max(len(true_res), 1)
            j = jaccard(pred, true_res)
            if j > best_j:
                best_j, best_rec, best_id = j, rec, c["epitope_id"]
            if rec >= a.recall_cut:
                hit_ranks.append(int(c["rank"]))
        detected = bool(hit_ranks)
        first_hit = min(hit_ranks) if detected else None
        # null model: the same candidate set, ranked at random
        n, k = len(cands), len(hit_ranks)
        if k:
            draw = rng.permutation(n)[:k]
            rand_ranks.append(int(draw.min()) + 1)
        rows.append(dict(antigen=key, n_candidates=n, n_true=len(true_res),
                         detected=detected, first_hit_rank=first_hit,
                         best_jaccard=round(best_j, 3), best_recall=round(best_rec, 3),
                         best_id=best_id, n_hits=k))

    if not rows:
        log("no scored antigens")
        return 1

    det = [r for r in rows if r["detected"]]
    n = len(rows)
    if not a.quiet:
      print(f"{'antigen':<10} {'cands':>5} {'true':>5} {'hits':>5} {'rank':>5} "
          f"{'jaccard':>8} {'recall':>7}  best")
      for r in sorted(rows, key=lambda x: (x["first_hit_rank"] is None,
                                         x["first_hit_rank"] or 99)):
        print(f"{r['antigen']:<10} {r['n_candidates']:>5} {r['n_true']:>5} {r['n_hits']:>5} "
              f"{str(r['first_hit_rank'] or '-'):>5} {r['best_jaccard']:>8.3f} "
              f"{r['best_recall']:>7.3f}  {r['best_id']}")

    print()
    print(f"complexes scored                 {n}")
    dlo, dhi = binomial_ci(len(det), n)
    print(f"detection (true epitope proposed) {len(det)}/{n} = {100*len(det)/n:.0f} % "
          f"[95% CI {100*dlo:.0f}-{100*dhi:.0f}]")
    for topn in (1, 3, 5):
        hit = sum(1 for r in det if r["first_hit_rank"] <= topn)
        exp = np.mean([1.0 - math.comb(max(r["n_candidates"] - r["n_hits"], 0), topn)
                       / math.comb(r["n_candidates"], topn)
                       if r["n_candidates"] >= topn else 1.0 for r in det]) if det else 0.0
        ef = (hit / len(det)) / exp if det and exp > 0 else float("nan")
        lo, hi = binomial_ci(hit, n)
        print(f"recall@{topn}  {hit}/{n} = {100*hit/n:>3.0f} % "
              f"[95% CI {100*lo:.0f}-{100*hi:.0f}]   "
              f"(random expectation {100*exp:.0f} %, enrichment factor {ef:.2f}x)")
    if det:
        ranks = [r["first_hit_rank"] for r in det]
        print(f"median rank of first true-epitope hit   {int(np.median(ranks))} "
              f"of {int(np.median([r['n_candidates'] for r in det]))} candidates")
        print(f"random-ranking control, median          "
              f"{int(np.median(rand_ranks)) if rand_ranks else '-'}")
        print(f"mean best Jaccard against true epitope  "
              f"{np.mean([r['best_jaccard'] for r in rows]):.3f}")
        print(f"mean best recall of true epitope        "
              f"{np.mean([r['best_recall'] for r in rows]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
