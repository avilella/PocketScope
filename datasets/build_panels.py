#!/usr/bin/env python3
"""Build the EpitopeScope benchmark panels from public databases.

Resolves each protein in datasets/panel_targets.tsv to its best experimental structure with
the PDBe SIFTS `best_structures` API, downloads that entry from RCSB, and writes the mapped
chain out as a single-chain PDB file. One file is one antigen, which is what EpitopeScope
expects.

The three panels are nested (10 subset of 48 subset of 96) so that the effect of panel size on
cross-talk can be read directly: the same antigen appears in all three, with more competitors
around it each time.

    python datasets/build_panels.py                 # builds panel10, panel48, panel96
    python datasets/build_panels.py --sizes 10      # just the pilot
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIFTS = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{acc}"
RCSB = "https://files.rcsb.org/download/{pdb}.pdb"
CACHE = HERE / ".cache"

AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
       "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE", "SEC"}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def fetch(url: str, dest: Path, tries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as fh:
                shutil.copyfileobj(r, fh)
            if dest.stat().st_size > 0:
                return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            if k == tries - 1:
                log(f"  ! {url}: {exc}")
            time.sleep(1.5 * (k + 1))
    dest.unlink(missing_ok=True)
    return False


def best_structure(acc: str, min_res: float, min_cov: float):
    """-> (pdb_id, chain, resolution, coverage) for the best-resolved, best-covering entry."""
    dest = CACHE / "sifts" / f"{acc}.json"
    if not fetch(SIFTS.format(acc=acc), dest):
        return None
    try:
        data = json.loads(dest.read_text())
    except json.JSONDecodeError:
        dest.unlink(missing_ok=True)
        return None
    entries = data.get(acc) or []
    cand = []
    for e in entries:
        res, cov = e.get("resolution"), e.get("coverage") or 0.0
        if res is None or res > min_res or cov < min_cov:
            continue
        cand.append((-cov, res, e["pdb_id"].upper(), e["chain_id"]))
    if not cand:
        for e in entries:                                   # relax: take anything mapped
            if e.get("resolution"):
                cand.append((-(e.get("coverage") or 0.0), e["resolution"],
                             e["pdb_id"].upper(), e["chain_id"]))
    if not cand:
        return None
    cand.sort()
    negcov, res, pdb, chain = cand[0]
    return pdb, chain, res, -negcov


def extract_chain(src: Path, chain: str, dest: Path, gene: str) -> int:
    """Write the first model of one chain, standard residues only. Returns residue count."""
    keep, seen = [], set()
    with open(src, errors="replace") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if line.startswith(("ATOM  ", "HETATM")):
                if line[21] != chain:
                    continue
                if line[17:20].strip().upper() not in AA3:
                    continue
                keep.append(line)
                seen.add(line[22:27])
            elif line.startswith("SSBOND") and (line[15] == chain or line[29] == chain):
                keep.append(line)
    if len(seen) < 25:
        return len(seen)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as fh:
        fh.write(f"HEADER    {gene:<40}\n")
        fh.writelines(sorted(keep, key=lambda l: l.startswith("ATOM  ") or l.startswith("HETATM")))
        fh.write("END\n")
    return len(seen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default=str(HERE / "panel_targets.tsv"))
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 48, 96])
    ap.add_argument("--max-resolution", type=float, default=3.0)
    ap.add_argument("--min-coverage", type=float, default=0.30)
    ap.add_argument("--min-residues", type=int, default=40)
    a = ap.parse_args()

    rows = []
    for line in Path(a.targets).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            rows.append(tuple(parts[:3]))
    log(f"{len(rows)} candidate antigens; need {max(a.sizes)}")

    out = Path(a.outdir)
    staged, manifest = [], []
    for gene, acc, family in rows:
        if len(staged) >= max(a.sizes):
            break
        hit = best_structure(acc, a.max_resolution, a.min_coverage)
        if not hit:
            log(f"  - {gene} ({acc}): no mapped structure")
            continue
        pdb, chain, res, cov = hit
        raw = CACHE / "pdb" / f"{pdb}.pdb"
        if not fetch(RCSB.format(pdb=pdb), raw):
            log(f"  - {gene}: download failed for {pdb}")
            continue
        name = f"{gene}_{pdb}_{chain}.pdb"
        tmp = CACHE / "chains" / name
        n = extract_chain(raw, chain, tmp, gene)
        if n < a.min_residues or not tmp.exists():
            log(f"  - {gene}: {pdb}_{chain} has only {n} residues")
            continue
        staged.append((name, tmp))
        manifest.append((len(staged), gene, acc, family, pdb, chain, f"{res:.2f}",
                         f"{cov:.2f}", str(n)))
        log(f"  {len(staged):>3}. {gene:<10} {pdb}_{chain}  {res:.2f} A  "
            f"cov {cov:.2f}  {n} res")

    if len(staged) < max(a.sizes):
        log(f"warning: only {len(staged)} antigens resolved, wanted {max(a.sizes)}")

    for size in sorted(a.sizes):
        if size > len(staged):
            log(f"skipping panel{size}: only {len(staged)} antigens available")
            continue
        d = out / f"panel{size}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for name, tmp in staged[:size]:
            shutil.copy2(tmp, d / name)
        log(f"panel{size}: {size} PDB files in {d}")

    man = out / "panel_manifest.tsv"
    with open(man, "w") as fh:
        fh.write("n\tgene\tuniprot\tfamily\tpdb_id\tchain\tresolution_A\tunp_coverage\tn_residues\n")
        for r in manifest:
            fh.write("\t".join(map(str, r)) + "\n")
    log(f"manifest: {man}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
