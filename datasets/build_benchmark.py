#!/usr/bin/env python3
"""Build the retrospective validation set: real nanobody-antigen complexes.

Section 4.5 of the workflow preprint asks for a retrospective validation step that shows
ranking enrichment of literature-supported references. The equivalent here is a set of solved
nanobody-antigen complexes: the true epitope is whatever the nanobody actually touches, so a
detector can be scored on whether it ranks that patch highly when shown the antigen alone.

For each complex this writes the isolated antigen chain (no nanobody) plus the observed
epitope residues, so nothing about the answer leaks into the input EpitopeScope sees.
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

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{pdb}"
ENTITY = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb}/{eid}"
RCSB = "https://files.rcsb.org/download/{pdb}.pdb"

AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
       "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE", "SEC"}

NB_WORDS = ("nanobody", "vhh", "single-domain", "single domain", "sdab", "camelid",
            "sybody", "nanobodies", "llama", "alpaca", "megabody", "variable domain of heavy"
            " chain", "heavy chain antibody", "immunoglobulin heavy chain variable")
AB_WORDS = ("fab ", "fab,", "light chain", "kappa", "lambda", "scfv", "antibody heavy",
            "immunoglobulin g", "fragment antigen")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def get(url: str, dest: Path, tries: int = 3):
    if dest.exists() and dest.stat().st_size > 0:
        return dest.read_bytes()
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if data:
                dest.write_bytes(data)
                return data
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            if k == tries - 1:
                log(f"  ! {url}: {exc}")
            time.sleep(1.0 * (k + 1))
    return None


def search_entries(rows: int, max_res: float):
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "full_text", "parameters": {"value": "nanobody"}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.polymer_entity_count_protein",
            "operator": "greater_or_equal", "value": 2}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined",
            "operator": "less", "value": max_res}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.experimental_method",
            "operator": "exact_match", "value": "X-ray"}}]},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": rows},
                            "results_content_type": ["experimental"],
                            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined",
                                      "direction": "asc"}]}}
    dest = CACHE / "search" / f"nb_{rows}_{max_res}.json"
    if not (dest.exists() and dest.stat().st_size > 0):
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(SEARCH, data=json.dumps(q).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            dest.write_bytes(r.read())
    return [x["identifier"] for x in json.loads(dest.read_text()).get("result_set", [])]


def entity_table(pdb: str):
    """-> [(chain_ids, name, length)] for every protein entity of an entry."""
    raw = get(ENTRY.format(pdb=pdb), CACHE / "entry" / f"{pdb}.json")
    if not raw:
        return []
    ids = json.loads(raw).get("rcsb_entry_container_identifiers", {}).get(
        "polymer_entity_ids", [])
    out = []
    for eid in ids:
        raw = get(ENTITY.format(pdb=pdb, eid=eid), CACHE / "entity" / f"{pdb}_{eid}.json")
        if not raw:
            continue
        e = json.loads(raw)
        poly = e.get("entity_poly", {})
        ptype = (poly.get("rcsb_entity_polymer_type") or "").lower()
        ptype2 = (poly.get("type") or "").lower()
        if "protein" not in ptype and "polypeptide" not in ptype2:
            continue
        name = (e.get("rcsb_polymer_entity", {}).get("pdbx_description") or "").lower()
        chains = e.get("rcsb_polymer_entity_container_identifiers", {}).get(
            "auth_asym_ids", [])
        out.append((chains, name, int(poly.get("rcsb_sample_sequence_length") or 0)))
    return out


def classify(entities):
    """-> (nanobody chains, antigen chain) or None when the pair is ambiguous."""
    nb, ag = [], []
    for chains, name, length in entities:
        is_nb = any(w in name for w in NB_WORDS) or (100 <= length <= 145 and "antibody" in name)
        is_ab = any(w in name for w in AB_WORDS)
        if is_nb and not is_ab:
            nb.extend(chains)
        elif is_ab:
            return None                       # conventional Fab present: not a clean case
        else:
            ag.append((length, chains, name))
    if not nb or not ag:
        return None
    ag.sort(reverse=True)
    length, chains, name = ag[0]
    if length < 50 or not chains:
        return None
    return nb, chains[0], name


def parse_atoms(path: Path):
    """-> {chain: [(reskey, resname, xyz)]} heavy atoms of the first model."""
    out = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[16] not in (" ", "A"):
                continue
            rn = line[17:20].strip().upper()
            if rn not in AA3:
                continue
            el = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
            if el in ("H", "D"):
                continue
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            out.setdefault(line[21], []).append((line[22:27], rn, xyz, line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(HERE / "nanobody_bench"))
    ap.add_argument("--n", type=int, default=30, help="complexes to keep")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip this many accepted complexes first, to build a disjoint set")
    ap.add_argument("--search-rows", type=int, default=150)
    ap.add_argument("--max-resolution", type=float, default=2.6)
    ap.add_argument("--contact", type=float, default=4.5,
                    help="heavy-atom distance defining an observed epitope contact")
    ap.add_argument("--min-epitope", type=int, default=8)
    ap.add_argument("--min-residues", type=int, default=60)
    a = ap.parse_args()

    out = Path(a.outdir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    ids = search_entries(a.search_rows, a.max_resolution)
    log(f"{len(ids)} candidate nanobody complexes from RCSB")
    kept, skipped = [], 0
    for pdb in ids:
        if len(kept) >= a.n:
            break
        ents = entity_table(pdb)
        cls = classify(ents)
        if not cls:
            continue
        nb_chains, ag_chain, ag_name = cls
        raw = CACHE / "pdb" / f"{pdb}.pdb"
        if not get(RCSB.format(pdb=pdb), raw):
            continue
        atoms = parse_atoms(raw)
        if ag_chain not in atoms:
            continue
        nb_xyz = [x[2] for c in nb_chains if c in atoms for x in atoms[c]]
        if not nb_xyz:
            continue
        nb_xyz = np.asarray(nb_xyz, np.float32)
        ag = atoms[ag_chain]
        res_keys, epitope = [], set()
        for key, rn, xyz, _ in ag:
            if key not in res_keys:
                res_keys.append(key)
            d = np.sqrt(((nb_xyz - np.asarray(xyz, np.float32)) ** 2).sum(1)).min()
            if d <= a.contact:
                epitope.add(key)
        if len(res_keys) < a.min_residues or len(epitope) < a.min_epitope:
            continue
        if skipped < a.skip:
            skipped += 1
            continue
        name = f"{pdb}_{ag_chain}.pdb"
        with open(out / name, "w") as fh:
            fh.write(f"HEADER    {ag_name[:60]:<60}\n")
            for _, _, _, line in ag:
                fh.write(line)
            fh.write("END\n")
        kept.append((pdb, ag_chain, ",".join(nb_chains), len(res_keys), len(epitope),
                     " ".join(sorted(k.strip() for k in epitope), ), ag_name[:60]))
        log(f"  {len(kept):>3}. {pdb}_{ag_chain}  {len(res_keys)} res, "
            f"epitope {len(epitope)} res vs Nb {','.join(nb_chains)}  {ag_name[:45]}")

    tsv = out / "true_epitopes.tsv"
    with open(tsv, "w") as fh:
        fh.write("pdb_id\tantigen_chain\tnanobody_chains\tn_residues\tn_epitope_residues"
                 "\tepitope_resseq\tantigen_name\n")
        for r in kept:
            fh.write("\t".join(map(str, r)) + "\n")
    log(f"{len(kept)} complexes written to {out}")
    log(f"ground truth: {tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
