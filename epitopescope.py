#!/usr/bin/env python3
"""EpitopeScope - rank antigen surface epitopes for orthogonal, PURE-expressible nanobody design.

Given a directory of antigen PDB files, EpitopeScope enumerates candidate surface epitope
patches on every antigen and scores each one on three axes:

  1. epitope quality   - is this patch a plausible nanobody epitope at all?
                         (exposed area, protrusion / cleft character, rigidity, composition)

  2. orthogonality     - if a nanobody were raised against this patch, how likely is it to
                         also stick to a patch on one of the OTHER antigens in the same tube?
                         Patches are compared with the PocketScope late-interaction MaxSim
                         representation (per-residue pLM embedding + 13 physicochemical
                         channels, L2-normalised, Equation 1 of the PocketScope paper).
                         Optionally also screened against the prebuilt PocketScope human
                         pocketome index for proteome-wide off-target risk.

  3. PURE expressibility - can the minimal construct that carries this epitope actually be made
                         and folded by an E. coli PURE in-vitro transcription/translation
                         reaction? Penalises disulfides (PURE is reducing), N-glycosylation
                         dependence, transmembrane / aggregation-prone segments, disorder,
                         inter-chain (quaternary) epitopes, cofactor dependence, and epitopes
                         that cannot be excised as a compact autonomous domain.

The three axes are combined into a single ranked score per candidate epitope, and one CSV is
written per input PDB.

Everything informational goes to stderr; the full paths of the output CSVs go to stdout.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------------------------
# PocketScope reuse: the pocket/patch representation of Equation 1 lives in this repository.
# ---------------------------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from pocketscope.features import residue_phys, match_feature, ALPHA
except Exception as exc:                                              # pragma: no cover
    raise SystemExit(
        f"epitopescope needs the pocketscope package that ships in this repository "
        f"(import failed: {exc}). Run it from a checkout of PocketScope, or `pip install -e .`."
    )

__version__ = "0.3.1"

# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
       "MSE": "M", "SEC": "C", "PYL": "K", "HSD": "H", "HSE": "H", "HSP": "H",
       "CSO": "C", "CME": "C", "SEP": "S", "TPO": "T", "PTR": "Y", "UNK": "X"}

WATERS = {"HOH", "WAT", "DOD", "H2O"}

# Modelled sugars: a contact here means the epitope is glycan-shielded on the native antigen,
# which is a different warning from depending on a metal or nucleotide cofactor.
GLYCANS = {"NAG", "NDG", "BMA", "MAN", "BGC", "GLC", "GAL", "FUC", "FUL", "SIA", "NGA",
           "XYP", "A2G", "GLA", "RIP"}
# Ubiquitous crystallisation additives, not real cofactors.
IGNORE_HET = {"SO4", "PO4", "GOL", "EDO", "PEG", "MPD", "ACT", "CL", "NA", "K", "TRS",
              "DMS", "IOD", "BME", "FMT", "NO3", "CIT", "MES", "EPE", "IMD", "AZI"}

# Bondi van der Waals radii, in angstrom.
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90, "H": 1.20,
       "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "ZN": 1.39, "MG": 1.73, "CA": 2.31,
       "FE": 2.00, "MN": 2.00, "NA": 2.27, "K": 2.75, "CU": 1.40, "NI": 1.63, "CO": 2.00}

# Tien et al. 2013, theoretical maximum accessible surface area per residue (angstrom^2).
MAX_ASA = {"A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0, "E": 223.0, "Q": 225.0,
           "G": 104.0, "H": 224.0, "I": 197.0, "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0,
           "P": 159.0, "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0, "X": 200.0}

# Kyte & Doolittle hydropathy.
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
      "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
      "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2, "X": 0.0}

HYDROPHOBIC = set("AVLIMFWCY")

# EMBOSS pKa values, used for the isoelectric point of the proposed construct.
PKA_SIDE = {"D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.07, "H": 6.00, "K": 10.53, "R": 12.48}
PKA_NTERM, PKA_CTERM = 8.6, 3.6

# Average residue masses, in Da.
MW = {"A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14, "E": 129.12, "Q": 128.13,
      "G": 57.05, "H": 137.14, "I": 113.16, "L": 113.16, "K": 128.17, "M": 131.19, "F": 147.18,
      "P": 97.12, "S": 87.08, "T": 101.10, "W": 186.21, "Y": 163.18, "V": 99.13, "X": 110.0}

PDB_SUFFIXES = (".pdb", ".ent", ".pdb.gz", ".ent.gz", ".cif", ".mmcif", ".cif.gz")


def log(msg: str) -> None:
    """Always shown: headline results, warnings and errors."""
    print(msg, file=sys.stderr, flush=True)


def vlog(msg: str, verbose: bool) -> None:
    """Shown only under --verbose: per-structure and per-stage detail."""
    if verbose:
        print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------
# Structure parsing
# ---------------------------------------------------------------------------------------------

@dataclass
class Residue:
    chain: str
    resseq: int
    icode: str
    resname: str
    aa: str
    idx: int = -1                       # global index within the Structure
    chain_pos: int = -1                 # position within its chain's residue list
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    radii: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    names: list = field(default_factory=list)
    bfac: float = 0.0
    ca: np.ndarray = None
    cb: np.ndarray = None
    sasa_complex: float = 0.0
    sasa_monomer: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.chain}{self.resseq}{self.icode.strip()}"


@dataclass
class Structure:
    path: Path
    name: str
    residues: list
    chains: dict                        # chain id -> [Residue]
    het: list                           # (resname, chain, resseq, coords) non-water HETATM groups
    is_plddt: bool = False
    bfac_pct: dict = None               # residue idx -> within-structure B-factor percentile

    def chain_seq(self, ch: str) -> str:
        return "".join(r.aa for r in self.chains[ch])


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        import gzip
        return gzip.open(path, "rt", errors="replace")
    return open(path, "rt", errors="replace")


def _element_of(atom_name: str, raw_element: str) -> str:
    e = raw_element.strip().upper()
    if e:
        return e
    nm = atom_name.strip()
    return (nm[1] if nm[:1].isdigit() else nm[:1]).upper()


def parse_pdb(path: Path, bfactor_mode: str = "auto") -> Structure:
    """Parse the first model of a PDB file into per-residue heavy-atom records."""
    residues, order, het = {}, [], []
    header_af = False
    with _open_text(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec.startswith(("REMARK", "TITLE", "HEADER", "DBREF")) and "ALPHAFOLD" in line.upper():
                header_af = True
            if rec == "ENDMDL":
                break
            if rec not in ("ATOM  ", "HETATM"):
                continue
            altloc = line[16]
            if altloc not in (" ", "A", "1"):
                continue
            resname = line[17:20].strip().upper()
            if resname in WATERS:
                continue
            atom = line[12:16].strip()
            elem = _element_of(atom, line[76:78])
            if elem in ("H", "D"):
                continue
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            chain = (line[21].strip() or "_")
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26]
            try:
                b = float(line[60:66])
            except ValueError:
                b = 0.0

            if resname not in AA3:
                het.append((resname, chain, resseq, np.array(xyz, np.float32)))
                continue

            key = (chain, resseq, icode)
            r = residues.get(key)
            if r is None:
                r = Residue(chain=chain, resseq=resseq, icode=icode, resname=resname,
                            aa=AA3.get(resname, "X"))
                r._xyz, r._rad, r._b = [], [], []
                residues[key] = r
                order.append(key)
            r._xyz.append(xyz)
            r._rad.append(VDW.get(elem, 1.70))
            r._b.append(b)
            r.names.append(atom)
            if atom == "CA":
                r.ca = np.array(xyz, np.float32)
            elif atom == "CB":
                r.cb = np.array(xyz, np.float32)
            elif atom == "N":
                r._n = np.array(xyz, np.float32)
            elif atom == "C":
                r._c = np.array(xyz, np.float32)

    reslist = []
    for key in order:
        r = residues[key]
        if r.ca is None:
            continue
        r.coords = np.asarray(r._xyz, np.float32)
        r.radii = np.asarray(r._rad, np.float32)
        r.bfac = float(np.mean(r._b)) if r._b else 0.0
        if r.cb is None:
            r.cb = _pseudo_cb(r)
        del r._xyz, r._rad, r._b
        reslist.append(r)

    if not reslist:
        raise ValueError(f"no standard amino-acid residues with a CA atom in {path}")

    chains: dict = {}
    for i, r in enumerate(reslist):
        r.idx = i
        chains.setdefault(r.chain, []).append(r)
    for ch, rs in chains.items():
        for p, r in enumerate(rs):
            r.chain_pos = p

    bvals = np.array([r.bfac for r in reslist], np.float32)
    if bfactor_mode == "plddt":
        is_plddt = True
    elif bfactor_mode in ("bfactor", "none"):
        is_plddt = False
    else:                                       # auto
        is_plddt = bool(header_af or (bvals.min() >= 0.0 and bvals.max() <= 100.0
                                      and bvals.mean() > 40.0 and np.median(bvals) > 50.0
                                      and "AF-" in path.name.upper()))
        if not is_plddt and bvals.max() <= 100.0 and bvals.mean() > 60.0 and bvals.min() > 15.0:
            # AlphaFold-like distribution even without an AF- filename
            is_plddt = bool(np.percentile(bvals, 75) > 80.0)

    st = Structure(path=path, name=path.name, residues=reslist, chains=chains, het=het,
                   is_plddt=is_plddt)
    # Crystallographic B-factors are on a per-structure scale: across this repository's panels
    # the mean runs from 17 to 119 A^2. Mapping them onto one absolute rigidity scale makes a
    # well-refined antigen look uniformly rigid and a poorly-refined one uniformly floppy, which
    # is a property of the experiment, not the epitope, and it corrupts every cross-antigen
    # comparison the tool exists to make. Rank them within their own structure instead.
    # pLDDT needs no such treatment: it is already comparable between models.
    if not is_plddt and len(bvals) > 1:
        order = np.argsort(np.argsort(bvals))
        pct = order / max(len(bvals) - 1, 1)
        st.bfac_pct = {r.idx: float(pct[i]) for i, r in enumerate(reslist)}
    return st


def _pseudo_cb(r: Residue) -> np.ndarray:
    """Ideal CB position from N, CA, C; falls back to CA."""
    n, c = getattr(r, "_n", None), getattr(r, "_c", None)
    if n is None or c is None or r.ca is None:
        return r.ca
    b, cc = r.ca - n, c - r.ca
    a = np.cross(b, cc)
    return (-0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + r.ca).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Solvent accessible surface area (Shrake-Rupley)
# ---------------------------------------------------------------------------------------------

def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1).astype(np.float32)


def _neighbour_lists(xyz: np.ndarray, cutoff: float):
    try:
        from scipy.spatial import cKDTree
        return cKDTree(xyz).query_ball_point(xyz, cutoff)
    except Exception:
        n = len(xyz)
        out = []
        for s in range(0, n, 512):
            d2 = ((xyz[s:s + 512, None, :] - xyz[None, :, :]) ** 2).sum(-1)
            for row in (d2 <= cutoff * cutoff):
                out.append(np.flatnonzero(row).tolist())
        return out


def atom_sasa(xyz: np.ndarray, radii: np.ndarray, probe: float = 1.4,
              n_points: int = 92) -> np.ndarray:
    """Per-atom SASA in angstrom^2."""
    if len(xyz) == 0:
        return np.zeros(0, np.float32)
    sphere = _fibonacci_sphere(n_points)
    rp = radii + probe
    nbrs = _neighbour_lists(xyz, 2.0 * float(rp.max()))
    out = np.empty(len(xyz), np.float32)
    for i in range(len(xyz)):
        pts = xyz[i] + rp[i] * sphere
        acc = np.ones(n_points, bool)
        for j in nbrs[i]:
            if j == i:
                continue
            d2 = ((pts - xyz[j]) ** 2).sum(1)
            acc &= d2 > rp[j] * rp[j]
            if not acc.any():
                break
        out[i] = 4.0 * math.pi * rp[i] * rp[i] * acc.sum() / n_points
    return out


def compute_sasa(st: Structure, probe: float = 1.4, n_points: int = 92) -> None:
    """Fill sasa_complex (all chains present) and sasa_monomer (chain in isolation)."""
    xyz = np.concatenate([r.coords for r in st.residues])
    rad = np.concatenate([r.radii for r in st.residues])
    bounds, k = [], 0
    for r in st.residues:
        bounds.append((k, k + len(r.coords)))
        k += len(r.coords)
    a = atom_sasa(xyz, rad, probe, n_points)
    for r, (s, e) in zip(st.residues, bounds):
        r.sasa_complex = float(a[s:e].sum())

    if len(st.chains) == 1:
        for r in st.residues:
            r.sasa_monomer = r.sasa_complex
        return
    for ch, rs in st.chains.items():
        cxyz = np.concatenate([r.coords for r in rs])
        crad = np.concatenate([r.radii for r in rs])
        b, k = [], 0
        for r in rs:
            b.append((k, k + len(r.coords)))
            k += len(r.coords)
        ca = atom_sasa(cxyz, crad, probe, n_points)
        for r, (s, e) in zip(rs, b):
            r.sasa_monomer = float(ca[s:e].sum())


def sasa_of(r: Residue, mode: str = "monomer") -> float:
    """Exposed area of a residue, either on the isolated chain or in the deposited assembly.

    `monomer` is the default: PURE makes one chain at a time, and a nanobody would be raised
    against the isolated antigen, so an epitope occluded by a partner chain in the crystal is
    still a real epitope. `interface_buried_A2` reports how much of it the assembly hides.
    """
    return r.sasa_monomer if mode == "monomer" else r.sasa_complex


def rsasa(r: Residue, mode: str = "monomer") -> float:
    return float(sasa_of(r, mode) / MAX_ASA.get(r.aa, 200.0))


# ---------------------------------------------------------------------------------------------
# Contacts, disulfides, cofactors
# ---------------------------------------------------------------------------------------------

def residue_contacts(st: Structure, cutoff: float = 5.0, seq_sep: int = 3):
    """Symmetric residue contact sets keyed by global residue index (heavy atoms < cutoff)."""
    xyz = np.concatenate([r.coords for r in st.residues])
    owner = np.concatenate([np.full(len(r.coords), r.idx, np.int32) for r in st.residues])
    pairs = set()
    try:
        from scipy.spatial import cKDTree
        t = cKDTree(xyz)
        for i, j in t.query_pairs(cutoff):
            a, b = owner[i], owner[j]
            if a != b:
                pairs.add((min(a, b), max(a, b)))
    except Exception:
        for s in range(0, len(xyz), 512):
            d2 = ((xyz[s:s + 512, None, :] - xyz[None, :, :]) ** 2).sum(-1)
            ii, jj = np.nonzero(d2 <= cutoff * cutoff)
            for i, j in zip(ii + s, jj):
                a, b = owner[i], owner[j]
                if a != b:
                    pairs.add((min(a, b), max(a, b)))
    cmap = {r.idx: set() for r in st.residues}
    for a, b in pairs:
        ra, rb = st.residues[a], st.residues[b]
        if ra.chain == rb.chain and abs(ra.chain_pos - rb.chain_pos) < seq_sep:
            continue
        cmap[a].add(b)
        cmap[b].add(a)
    return cmap


def find_disulfides(st: Structure, cutoff: float = 2.5):
    """[(idx_a, idx_b)] for SG-SG pairs closer than cutoff."""
    sg = [(r.idx, r.coords[r.names.index("SG")])
          for r in st.residues if r.aa == "C" and "SG" in r.names]
    out = []
    for a in range(len(sg)):
        for b in range(a + 1, len(sg)):
            if np.linalg.norm(sg[a][1] - sg[b][1]) <= cutoff:
                out.append((sg[a][0], sg[b][0]))
    return out


def cofactor_contacts(st: Structure, cutoff: float = 4.5):
    """Global residue indices in contact with a non-water HETATM group -> {idx: resname}.

    Crystallisation additives are ignored; sugars are reported so a glycan-shielded epitope can
    be told apart from one that needs a metal or nucleotide the PURE mix does not supply.
    """
    het = [h for h in st.het if h[0] not in IGNORE_HET]
    if not het:
        return {}
    hxyz = np.stack([h[3] for h in het])
    hname = [h[0] for h in het]
    out = {}
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(hxyz)
        for r in st.residues:
            d, j = tree.query(r.coords, distance_upper_bound=cutoff)
            k = int(np.argmin(d))
            if np.isfinite(d[k]) and j[k] < len(hname):
                out[r.idx] = hname[int(j[k])]
    except ImportError:
        for r in st.residues:
            d = np.sqrt(((r.coords[:, None, :] - hxyz[None, :, :]) ** 2).sum(-1))
            m = d.min(0)
            hit = np.flatnonzero(m <= cutoff)
            if len(hit):
                out[r.idx] = hname[int(hit[np.argmin(m[hit])])]
    return out


# ---------------------------------------------------------------------------------------------
# Sequence-level liabilities
# ---------------------------------------------------------------------------------------------

def nglyc_sequons(seq: str):
    """0-based positions of N in an N-X-[ST] sequon with X != P."""
    return [i for i in range(len(seq) - 2)
            if seq[i] == "N" and seq[i + 1] != "P" and seq[i + 2] in "ST"]


def max_hydropathy_window(seq: str, window: int = 19) -> float:
    if len(seq) < window:
        window = max(5, len(seq))
    v = np.array([KD.get(a, 0.0) for a in seq], np.float32)
    if len(v) < window:
        return float(v.mean()) if len(v) else 0.0
    c = np.convolve(v, np.ones(window, np.float32) / window, mode="valid")
    return float(c.max())


def gravy(seq: str) -> float:
    return float(np.mean([KD.get(a, 0.0) for a in seq])) if seq else 0.0


def isoelectric_point(seq: str) -> float:
    counts = {a: seq.count(a) for a in PKA_SIDE}

    def charge(ph: float) -> float:
        q = 1.0 / (1.0 + 10 ** (ph - PKA_NTERM)) - 1.0 / (1.0 + 10 ** (PKA_CTERM - ph))
        for aa, pka in PKA_SIDE.items():
            if not counts[aa]:
                continue
            if aa in "KRH":
                q += counts[aa] / (1.0 + 10 ** (ph - pka))
            else:
                q -= counts[aa] / (1.0 + 10 ** (pka - ph))
        return q

    lo, hi = 0.0, 14.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if charge(mid) > 0:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 2)


def molecular_weight(seq: str) -> float:
    return round(sum(MW.get(a, 110.0) for a in seq) + 18.02, 1)


# ---------------------------------------------------------------------------------------------
# Candidate epitope patches
# ---------------------------------------------------------------------------------------------

@dataclass
class Patch:
    antigen: str                    # input file stem
    pid: str                        # antigen:chain:seed label
    chain: str
    members: list                   # global residue indices, ordered by descending exposed ASA
    seed: int
    seq: str = ""
    # geometry / exposure
    area: float = 0.0               # summed exposed ASA of the patch, angstrom^2
    mean_rsasa: float = 0.0
    protrusion: float = 0.0
    concavity: float = 0.0
    planarity: float = 0.0
    contiguity: float = 0.0
    rigidity: float = 0.0
    mean_bfac: float = 0.0
    net_charge: float = 0.0
    hydrophobic_frac: float = 0.0
    interface_buried: float = 0.0   # ASA buried by other chains, angstrom^2
    cofactor: str = ""
    # construct
    con_lo: int = 0
    con_hi: int = 0
    con_seq: str = ""
    con_range: str = ""
    dangling: float = 0.0
    n_cys: int = 0
    ss_internal: int = 0
    ss_broken: int = 0
    n_sequon_construct: int = 0
    n_sequon_patch: int = 0
    tm_hydropathy: float = 0.0
    con_gravy: float = 0.0
    con_pi: float = 0.0
    con_mw: float = 0.0
    con_disorder: float = 0.0
    exposed_hyd_frac: float = 0.0
    # scores
    s_epitope: float = 0.0
    s_pure: float = 0.0
    s_ortho: float = 0.0
    total: float = 0.0
    epi_parts: dict = field(default_factory=dict)
    pure_parts: dict = field(default_factory=dict)
    # cross-talk
    xt_max: float = 0.0
    xt_partner: str = ""
    xt_top3: float = 0.0
    xt_pct: float = 0.0
    xt_self: float = 0.0
    xt_z: float = 0.0
    off_max: float = 0.0
    off_hit: str = ""
    feats: np.ndarray = None
    w: np.ndarray = None            # per-residue exposed-area weights, aligned with feats
    panel_sel: bool = False
    panel_worst: float = 0.0
    panel_worst_partner: str = ""
    # site detection provenance (Concavity-style clustering vs exposed-surface seeding)
    source: str = "surface_seed"
    buriedness: float = 0.0
    centre: np.ndarray = None
    box: float = 0.0
    # thermodynamic and efficiency layer
    dg: float = 0.0
    kd: float = 0.0
    pkd: float = 0.0
    le: float = 0.0
    lle: float = 0.0
    fq: float = 0.0
    apolar_index: float = 0.0
    bsa: float = 0.0
    eci: float = 0.0
    # rule set
    rules: dict = field(default_factory=dict)
    zone: str = ""
    # multivariate
    cluster: int = -1
    cluster_size: int = 0
    cluster_antigens: int = 0
    pcs: tuple = (0.0, 0.0, 0.0)
    # specificity, status and provenance
    spec_ratio: float = 0.0
    status: str = "ranked"
    redundant_with: str = ""
    offtarget_contexts: str = ""


def _bell(x: float, lo: float, opt_lo: float, opt_hi: float, hi: float) -> float:
    """1.0 inside [opt_lo, opt_hi], tapering linearly to 0 at lo and hi."""
    if x <= lo or x >= hi:
        return 0.0
    if x < opt_lo:
        return (x - lo) / max(opt_lo - lo, 1e-9)
    if x > opt_hi:
        return (hi - x) / max(hi - opt_hi, 1e-9)
    return 1.0


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 below lo, 1 above hi, linear in between."""
    if hi <= lo:
        return 1.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _geomean(vals, floor: float = 0.05) -> float:
    v = [max(float(x), floor) for x in vals if x is not None]
    return float(np.exp(np.mean(np.log(v)))) if v else 0.0


def residue_buriedness(st: Structure, surface_idx, n_rays: int = 42,
                       reach: float = 11.0) -> np.ndarray:
    """Concavity of each surface residue, by ray casting into the outward hemisphere.

    The workflow preprint detects sites as surface depressions rather than at a hand-picked
    catalytic centre, which is what removes site-selection bias. The same idea in one pass:
    fire rays outward from each surface residue and count how many are blocked again by the
    protein within `reach`. A residue on a convex bulge blocks almost nothing; one at the
    bottom of a groove is walled in on most sides.

    -> buriedness in [0, 1], one value per entry of `surface_idx`.
    """
    cb_all = np.stack([r.cb for r in st.residues])
    centre = cb_all.mean(0)
    dirs = _fibonacci_sphere(n_rays)
    xyz = np.concatenate([r.coords for r in st.residues])
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(xyz)
    except Exception:
        tree = None

    steps = np.arange(3.0, reach, 1.5, dtype=np.float32)
    out = np.zeros(len(surface_idx), np.float32)
    for k, gi in enumerate(surface_idx):
        origin = st.residues[gi].cb
        outward = origin - centre
        n = np.linalg.norm(outward)
        outward = outward / n if n > 1e-6 else np.array([0, 0, 1.0], np.float32)
        # only rays pointing away from the protein core
        keep = dirs[dirs @ outward > 0.25]
        if len(keep) == 0:
            continue
        probes = (origin[None, None, :] + keep[:, None, :] * steps[None, :, None]
                  ).reshape(-1, 3)
        if tree is not None:
            hit = tree.query_ball_point(probes, 2.2, return_length=True) > 0
        else:
            d2 = ((probes[:, None, :] - xyz[None, :, :]) ** 2).sum(-1)
            hit = (d2 <= 2.2 * 2.2).any(1)
        blocked = hit.reshape(len(keep), len(steps)).any(1)
        out[k] = blocked.mean()
    return out


def concavity_clusters(st: Structure, surface_idx, buried, args):
    """Group high-concavity surface residues into cohesive sites by spatial proximity.

    Mirrors step 4.4.2 of the preprint: residues forming high-scoring concave regions are
    clustered without manual curation, and each cluster's Ca centroid becomes the site centre
    (there, the docking grid centre; here, the centre of a candidate epitope).
    """
    hot = [i for i, b in enumerate(buried) if b >= args.concavity_cut]
    if not hot:
        return []
    pts = np.stack([st.residues[surface_idx[i]].cb for i in hot])
    n = len(hot)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    try:
        from scipy.spatial import cKDTree
        pairs = cKDTree(pts).query_pairs(args.cluster_link)
    except Exception:
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        pairs = {(i, j) for i in range(n) for j in range(i + 1, n)
                 if d[i, j] <= args.cluster_link}
    for i, j in pairs:
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(hot[i])
    return [g for g in groups.values() if len(g) >= args.patch_min]


def enumerate_patches(st: Structure, args, cmap, plddt_ok: bool) -> list:
    """Enumerate candidate epitopes from concavity clusters and from exposed-surface seeds.

    Two detectors, deliberately: the concavity clustering of the preprint finds grooves and
    depressions without site-selection bias, but a nanobody epitope is not always a groove, so
    exposed-surface seeding keeps flat and convex patches in the running. Both feed one
    non-maximum-suppression pass, and `site_source` records which detector proposed each.
    """
    surface = [r for r in st.residues if rsasa(r, args.surface) >= args.min_rsasa]
    if len(surface) < args.patch_min:
        return []
    cb_all = np.stack([r.cb for r in st.residues])
    centre_all = cb_all.mean(0)
    rg = float(np.sqrt(((cb_all - centre_all) ** 2).sum(1).mean()))

    cb_s = np.stack([r.cb for r in surface])
    idx_s = np.array([r.idx for r in surface])
    surf_gidx = [r.idx for r in surface]

    try:
        from scipy.spatial import cKDTree
        tree_s, tree_all = cKDTree(cb_s), cKDTree(cb_all)
        near_s = tree_s.query_ball_point(cb_s, args.patch_radius)
        dens = np.array([len(x) for x in tree_all.query_ball_point(cb_s, 12.0)], np.float32)
    except Exception:
        d = np.sqrt(((cb_s[:, None, :] - cb_s[None, :, :]) ** 2).sum(-1))
        near_s = [np.flatnonzero(row <= args.patch_radius).tolist() for row in d]
        da = np.sqrt(((cb_s[:, None, :] - cb_all[None, :, :]) ** 2).sum(-1))
        dens = (da <= 12.0).sum(1).astype(np.float32)

    buried = residue_buriedness(st, surf_gidx, args.concavity_rays, args.concavity_reach)
    local_of = {g: k for k, g in enumerate(surf_gidx)}

    def build(mem_local, seed_local, source):
        mem_local = [m for m in mem_local if surface[m].chain == surface[seed_local].chain]
        if len(mem_local) < args.patch_min:
            return None
        anchor = np.stack([cb_s[m] for m in mem_local]).mean(0)
        d = np.linalg.norm(cb_s[mem_local] - anchor, axis=1)
        mem_local = [mem_local[i] for i in np.argsort(d)][:args.patch_max]
        members = sorted((int(idx_s[m]) for m in mem_local),
                         key=lambda gi: -sasa_of(st.residues[gi], args.surface))
        p = Patch(antigen=st.path.stem, pid="", chain=surface[seed_local].chain,
                  members=members, seed=int(idx_s[seed_local]), source=source)
        p.buriedness = float(np.mean([buried[local_of[g]] for g in members
                                      if g in local_of] or [0.0]))
        _measure_patch(p, st, cb_all, centre_all, rg,
                       float(np.mean([dens[local_of[g]] for g in members
                                      if g in local_of] or [0.0])), plddt_ok, args.surface)
        return p

    cand = []
    # (1) Concavity clusters: sites found as surface depressions, no hand-picked centre.
    for grp in concavity_clusters(st, surf_gidx, buried, args):
        seed = max(grp, key=lambda i: buried[i])
        near = set(grp)
        for i in grp:
            near.update(near_s[i])
        q = build(sorted(near), seed, "concavity_cluster")
        if q is not None:
            cand.append(q)
    # (2) Exposed-surface seeds: keeps flat and convex epitopes in the running.
    for k, r in enumerate(surface):
        q = build(list(near_s[k]), k, "surface_seed")
        if q is not None:
            cand.append(q)

    cand.sort(key=lambda p: (-p.epi_parts["prelim"], p.source != "concavity_cluster"))
    # Non-maximum suppression, but a suppressed candidate is recorded rather than dropped.
    # Odin-Multi keeps duplicate designs in its tables with a `duplicate_sequence` status and
    # a `duplicate_of_design_id` pointer; silently discarding them loses the audit trail of
    # what the detector actually proposed.
    kept, redundant = [], []
    for p in cand:
        ms = set(p.members)
        dup = next((q for q in kept
                    if len(ms & set(q.members)) / len(ms | set(q.members)) > args.max_overlap),
                   None)
        if dup is not None:
            if len(redundant) < args.emit_redundant:
                p.status, p._dup = "redundant", dup
                redundant.append(p)
            continue
        if len(kept) >= args.max_patches:
            continue
        kept.append(p)

    for n, p in enumerate(kept, 1):
        p.pid = f"{st.path.stem}|{p.chain}|E{n:02d}"
    for n, p in enumerate(redundant, 1):
        p.pid = f"{st.path.stem}|{p.chain}|R{n:02d}"
        p.redundant_with = p._dup.pid
        del p._dup
    for p in kept + redundant:
        X = np.stack([st.residues[i].cb for i in p.members])
        p.centre = X.mean(0)
        p.box = float(2.0 * np.abs(X - p.centre).max() + 6.0)
    return kept + redundant


def _measure_patch(p: Patch, st: Structure, cb_all, centre_all, rg, density, plddt_ok,
                   surface: str = "monomer") -> None:
    rs = [st.residues[i] for i in p.members]
    p.seq = "".join(r.aa for r in rs)
    p.area = float(sum(sasa_of(r, surface) for r in rs))
    p.mean_rsasa = float(np.mean([rsasa(r, surface) for r in rs]))
    p.interface_buried = float(sum(max(r.sasa_monomer - r.sasa_complex, 0.0) for r in rs))

    X = np.stack([r.cb for r in rs])
    c = X.mean(0)
    p.protrusion = float(np.linalg.norm(c - centre_all) / max(rg, 1e-6))
    p.concavity = (float(np.clip(p.buriedness, 0.0, 1.0)) if p.buriedness > 0
                   else float(np.clip((density - 20.0) / 22.0, 0.0, 1.0)))
    Xc = X - c
    if len(X) >= 3:
        w = np.linalg.svd(Xc, compute_uv=False) ** 2 / max(len(X) - 1, 1)
        p.planarity = float(np.sqrt(max(w[-1], 0.0)))
    p.mean_bfac = float(np.mean([r.bfac for r in rs]))
    if plddt_ok:
        p.rigidity = float(np.clip(p.mean_bfac / 100.0, 0, 1))
    elif st.bfac_pct:
        # 1 - the patch's mean B-factor percentile within its own structure
        p.rigidity = float(np.clip(
            1.0 - np.mean([st.bfac_pct.get(r.idx, 0.5) for r in rs]), 0.0, 1.0))
    else:
        p.rigidity = float(np.clip(1.0 - (p.mean_bfac - 15.0) / 60.0, 0, 1))

    pos = sorted(r.chain_pos for r in rs)
    runs, run = [], 1
    for a, b in zip(pos, pos[1:]):
        run = run + 1 if b - a <= 2 else 1
        runs.append(run)
    p.contiguity = float(max(runs or [1]) / len(pos))
    p.net_charge = float(sum({"D": -1, "E": -1, "K": 1, "R": 1}.get(r.aa, 0) for r in rs))
    p.hydrophobic_frac = float(np.mean([r.aa in HYDROPHOBIC for r in rs]))
    p.epi_parts["prelim"] = p.area * p.mean_rsasa * (0.5 + 0.5 * p.rigidity)


def score_epitope(p: Patch, args) -> None:
    """Is this patch a good nanobody epitope, ignoring cross-talk and expression?

    Calibrated against `datasets/nanobody_bench`, 30 solved nanobody-antigen complexes where
    the observed epitope is known. Two priors that looked reasonable did not survive that
    check and were changed: raw patch area separates real epitopes monotonically rather than
    peaking in a window, so the term is a ramp and not a bell; and protrusion runs the wrong
    way (real epitopes are slightly *less* protruding than the average candidate), so it is
    replaced by surface topography, which was the single most discriminating descriptor.
    """
    parts = {
        # bigger is simply better here: a VHH goes for the largest contiguous surface it can
        # reach, and no upper penalty is supported by the benchmark (AUC 0.64)
        "area": _ramp(p.area, 400.0, 1400.0),
        # three-dimensional topography, the strongest single descriptor in the benchmark
        # (AUC 0.68): a paratope needs something to grip
        "topography": _ramp(p.planarity, 1.2, 3.2),
        # ordered epitopes are strongly preferred (AUC 0.67)
        "rigidity": max(p.rigidity, 0.05),
        # a mild preference for concave sites survives (AUC 0.60), so it is kept but weighted
        # gently rather than treated as a requirement
        "cleft": 0.70 + 0.30 * p.concavity,
        # exposure is a gate, not a reward: below --min-rsasa a patch is unreachable, but
        # among reachable patches more exposure is not better
        "exposure": _bell(p.mean_rsasa, 0.10, 0.25, 0.65, 1.01),
        # a very hydrophobic patch is a sticky, poorly specific epitope
        "polarity": _bell(p.hydrophobic_frac, -0.01, 0.10, 0.45, 0.85),
    }
    p.epi_parts.update(parts)
    p.s_epitope = _geomean(parts.values())


# ---------------------------------------------------------------------------------------------
# Minimal PURE construct around an epitope
# ---------------------------------------------------------------------------------------------

def construct_span(st: Structure, p: Patch, cmap, args):
    """Greedily grow a contiguous window in the patch's chain that closes the most contacts.

    Returns (lo, hi) chain positions inclusive plus the fraction of contacts left dangling.
    """
    chain = st.chains[p.chain]
    n = len(chain)
    pos = [st.residues[i].chain_pos for i in p.members if st.residues[i].chain == p.chain]
    lo, hi = max(min(pos) - 4, 0), min(max(pos) + 4, n - 1)
    gidx = [r.idx for r in chain]

    def cut_cost(lo, hi):
        inside = set(gidx[lo:hi + 1])
        cut = tot = 0
        for gi in inside:
            for gj in cmap[gi]:
                tot += 1
                if gj not in inside:
                    cut += 1
        return cut, tot

    cut, tot = cut_cost(lo, hi)
    while (hi - lo + 1) < args.max_construct and (lo > 0 or hi < n - 1):
        best, move = 0.0, None
        for nl, nh in ((lo - 1, hi), (lo, hi + 1)):
            if nl < 0 or nh > n - 1:
                continue
            c2, _ = cut_cost(nl, nh)
            gain = cut - c2
            if gain > best:
                best, move, newcut = gain, (nl, nh), c2
        if move is None or best < args.grow_gain:
            break
        lo, hi, cut = move[0], move[1], newcut
    cut, tot = cut_cost(lo, hi)
    return lo, hi, (cut / tot if tot else 0.0)


def score_pure(p: Patch, st: Structure, cmap, ss_pairs, cofac, args, plddt_ok) -> None:
    """Can a PURE E. coli IVTT reaction make and fold the construct carrying this epitope?"""
    lo, hi, dangle = construct_span(st, p, cmap, args)
    chain = st.chains[p.chain]
    seg = chain[lo:hi + 1]
    p.con_lo, p.con_hi = seg[0].resseq, seg[-1].resseq
    p.con_range = f"{p.chain}:{p.con_lo}-{p.con_hi}"
    p.con_seq = "".join(r.aa for r in seg)
    p.dangling = round(dangle, 4)

    inside = set(r.idx for r in seg)
    p.n_cys = p.con_seq.count("C")
    p.ss_internal = sum(1 for a, b in ss_pairs if a in inside and b in inside)
    p.ss_broken = sum(1 for a, b in ss_pairs if (a in inside) != (b in inside))

    p.n_sequon_construct = len(nglyc_sequons(p.con_seq))
    patch_pos = {st.residues[i].chain_pos for i in p.members}
    full = st.chain_seq(p.chain)
    p.n_sequon_patch = sum(1 for q in nglyc_sequons(full) if q in patch_pos)

    p.tm_hydropathy = round(max_hydropathy_window(p.con_seq), 3)
    p.con_gravy = round(gravy(p.con_seq), 3)
    p.con_pi = isoelectric_point(p.con_seq)
    p.con_mw = molecular_weight(p.con_seq)
    if plddt_ok:
        p.con_disorder = round(float(np.mean([r.bfac < 70.0 for r in seg])), 3)
    else:
        p.con_disorder = 0.0

    hyd = sum(r.sasa_monomer for r in seg if r.aa in HYDROPHOBIC)
    tot = sum(r.sasa_monomer for r in seg) or 1.0   # the PURE product is always a monomer
    p.exposed_hyd_frac = round(hyd / tot, 3)

    p.cofactor = cofac.get(p.seed, "")
    if not p.cofactor:
        for i in p.members:
            if i in cofac:
                p.cofactor = cofac[i]
                break

    L = len(p.con_seq)
    ss_pen = 1.0
    if not args.allow_disulfides:
        ss_pen *= 0.55 ** p.ss_internal
    ss_pen *= 0.35 ** p.ss_broken
    ss_pen *= 0.97 ** max(p.n_cys - 2 * (p.ss_internal + p.ss_broken), 0)

    parts = {
        # PURE yields fall off hard above ~60 kDa; short compact domains fold best
        "length": _bell(L, 25.0, 55.0, 260.0, 600.0),
        # PURExpress is a reducing environment unless DTT is omitted and DsbC supplied
        "redox": float(np.clip(ss_pen, 0.02, 1.0)),
        # E. coli cannot glycosylate: a sequon inside the epitope means a different surface
        "glycan": (0.55 ** p.n_sequon_patch) * (0.93 ** max(p.n_sequon_construct
                                                            - p.n_sequon_patch, 0)),
        # a transmembrane-like stretch aggregates without a membrane or chaperone
        "membrane": _bell(p.tm_hydropathy, -9.0, -9.0, 1.10, 2.20),
        # excising the fragment must not shred its hydrophobic core
        "excisable": float(np.clip(1.0 - dangle / 0.45, 0.02, 1.0)),
        # exposed hydrophobic surface is the main aggregation driver in a crowded lysate-free mix
        "aggregation": _bell(p.exposed_hyd_frac, -0.01, 0.0, 0.34, 0.62),
        "order": 1.0 - 0.85 * p.con_disorder,
        # an epitope that only exists in an oligomer will not exist on a PURE monomer
        "monomeric": 1.0 if p.interface_buried < 60.0 else
                     float(np.clip(1.0 - (p.interface_buried - 60.0) / 400.0, 0.15, 1.0)),
        # PURE supplies no metals, hemes or nucleotide cofactors
        # PURE supplies no metals, hemes or nucleotides; a modelled glycan at the epitope means
        # the E. coli product presents a different surface there
        "cofactor_free": 1.0 if not p.cofactor else (0.70 if p.cofactor in GLYCANS else 0.55),
        "solubility": 0.75 + 0.25 * float(np.clip(abs(p.con_pi - 7.0) / 1.5, 0.0, 1.0)),
    }
    p.pure_parts = parts
    p.s_pure = _geomean(parts.values())


# ---------------------------------------------------------------------------------------------
# Thermodynamic layer and efficiency metrics
#
# The workflow preprint converts a docking score into interpretable affinity units (Kd, pKd)
# and then into size-corrected efficiency metrics (LE, LLE, FQ), so that a large ligand cannot
# win on raw score alone. The same discipline is applied here. There is no docking step, so
# the free energy is estimated from the surface a nanobody paratope would bury, and the
# "ligand size" that efficiency divides by is the number of epitope residues.
#
# These numbers are estimates from geometry, not measurements. They exist to make candidates
# comparable on an interpretable axis and to expose the greasy-binder failure mode that raw
# scores hide; they are not predicted affinities of any particular designed binder.
# ---------------------------------------------------------------------------------------------

R_GAS = 0.0019872041          # kcal / (mol K)
T_PHYS = 310.15               # K, 37 C, the fixed temperature the preprint uses for pKd/LLE
RT_PHYS = R_GAS * T_PHYS      # 0.6163 kcal/mol

# Effective solvation coefficients, kcal/mol per angstrom^2 of total buried area. Chosen so a
# typical nanobody interface (~1700 A^2 total, 40 % apolar) lands near -12 kcal/mol, i.e. a
# low-nanomolar Kd, and a marginal 800 A^2 interface lands in the high-micromolar range.
K_APOLAR, K_POLAR = 0.0095, 0.0055


def patch_apolar_fraction(p: Patch, st: Structure, surface: str) -> float:
    """Fraction of the patch's exposed area contributed by apolar side chains."""
    tot = ap = 0.0
    for i in p.members:
        r = st.residues[i]
        a = sasa_of(r, surface)
        tot += a
        if r.aa in HYDROPHOBIC:
            ap += a
    return ap / tot if tot > 0 else 0.0


def score_thermo(p: Patch, st: Structure, args) -> None:
    """Estimated interface free energy, Kd, pKd and the LE / LLE / FQ analogues."""
    p.bsa = float(min(p.area, args.paratope_area))
    apolar = patch_apolar_fraction(p, st, args.surface)
    total = 2.0 * p.bsa                       # both sides of the interface bury area
    dg = -(K_APOLAR * total * apolar + K_POLAR * total * (1.0 - apolar))
    dg += args.dg_flex * (1.0 - p.rigidity)   # a floppy epitope pays an ordering penalty
    p.dg = float(np.clip(dg, -20.0, -0.5))
    p.kd = float(np.exp(p.dg / RT_PHYS))
    p.pkd = float(-np.log10(max(p.kd, 1e-300)))

    # LE analogue: free energy per epitope residue. The preprint divides by heavy-atom count;
    # the size of a binding determinant here is how many residues it takes to make it.
    p.le = float(-p.dg / max(len(p.members), 1))

    # LLE analogue: affinity discounted by lipophilicity. The exposed apolar fraction is mapped
    # onto a logP-like 0-5 scale, so an epitope whose predicted affinity is bought entirely
    # with greasy surface scores low -- the failure mode the preprint calls out explicitly.
    p.apolar_index = float(5.0 * apolar)
    p.lle = float(p.pkd - p.apolar_index)

    # Epitope Complexity Index, the Structural Complexity Index analogue: size, chemical
    # variety and discontinuity in one number.
    counts = np.array([p.seq.count(a) for a in set(p.seq)], np.float64)
    probs = counts / counts.sum() if counts.sum() else np.array([1.0])
    entropy = float(-(probs * np.log(probs)).sum() / np.log(20.0))
    p.eci = float(np.clip(0.40 * min(len(p.members) / 22.0, 1.0)
                          + 0.30 * entropy
                          + 0.30 * (1.0 - p.contiguity), 0.0, 1.0))


def fit_efficiency_baseline(patches: list, verbose: bool = False):
    """Expected LE as a function of epitope size, fitted across the whole collection.

    Reynolds showed ligand efficiency falls with ligand size, so fit quality compares a ligand
    against what is typical for its size rather than against a flat threshold. Here the same
    correction is fitted from the run itself: LE ~ a + b / n_residues by least squares, which
    is the right shape because free energy grows sublinearly with epitope size.
    """
    n = np.array([len(p.members) for p in patches], np.float64)
    le = np.array([p.le for p in patches], np.float64)
    ok = (n > 0) & np.isfinite(le)
    if ok.sum() < 8 or len(np.unique(n[ok])) < 3:
        return None
    A = np.stack([np.ones(ok.sum()), 1.0 / n[ok]], axis=1)
    coef, *_ = np.linalg.lstsq(A, le[ok], rcond=None)
    a, b = float(coef[0]), float(coef[1])
    vlog(f"efficiency baseline: expected LE = {a:.4f} + {b:.4f}/n_residues", verbose)
    return a, b


def apply_fit_quality(patches: list, baseline) -> None:
    """FQ analogue: observed efficiency over the efficiency expected for that epitope size."""
    for p in patches:
        if baseline is None:
            exp = 0.0715 + 7.5328 / max(len(p.members), 1)      # Reynolds-style fallback
        else:
            a, b = baseline
            exp = a + b / max(len(p.members), 1)
        p.fq = float(p.le / exp) if exp > 1e-6 else 0.0


# ---------------------------------------------------------------------------------------------
# Rule set and developability zone
#
# The preprint scores compounds against named, independently checkable rules (Lipinski, Veber,
# Ghose, Egan) and against the BOILED-Egg absorption regions, rather than folding everything
# into one opaque number. The equivalent for a PURE reaction is a set of named pass/fail
# conditions on the construct, plus a coarse three-colour zone.
# ---------------------------------------------------------------------------------------------

RULES = ["size", "redox", "glyco", "membrane", "domain", "monomer", "order"]


def apply_rules(p: Patch, args) -> None:
    L = len(p.con_seq)
    p.rules = {
        # PURE yields fall away above roughly 60 kDa; very short fragments do not fold alone
        "size": bool(45 <= L <= 300 and p.con_mw <= 35000.0),
        # PURExpress is reducing unless DTT is left out and DsbC supplied
        "redox": bool(p.ss_broken == 0 and (args.allow_disulfides or p.ss_internal == 0)),
        # E. coli does not glycosylate, so a sequon in the epitope changes the surface
        "glyco": bool(p.n_sequon_patch == 0),
        # a transmembrane-like stretch aggregates without a membrane
        "membrane": bool(p.tm_hydropathy < 1.6),
        # the fragment has to survive being cut out of its parent
        "domain": bool(p.dangling <= 0.30),
        # an epitope that only exists in an oligomer will not exist on a PURE monomer
        "monomer": bool(p.interface_buried < 200.0),
        # disorder is neither expressible nor a usable epitope
        "order": bool(p.con_disorder <= 0.20),
    }
    n_pass = sum(p.rules.values())
    if n_pass >= 6 and L <= 200 and p.exposed_hyd_frac <= 0.34:
        p.zone = "GREEN"
    elif n_pass >= 4 and L <= 320 and p.exposed_hyd_frac <= 0.45:
        p.zone = "YELLOW"
    else:
        p.zone = "RED"


# ---------------------------------------------------------------------------------------------
# Multivariate analysis over the whole collection
# ---------------------------------------------------------------------------------------------

def resolve_cluster_cut(patches: list, S: np.ndarray, spec: str) -> float:
    """`auto` puts the cut at a high quantile of the observed cross-antigen similarities.

    An absolute similarity threshold is not portable: the physicochemical-only representation
    saturates near 1.0 while a language model spreads over a much wider range, so a fixed cut
    would either merge everything or nothing. Calibrating on the run's own background keeps
    "these two epitopes are unusually alike" meaning the same thing in both.
    """
    if spec != "auto":
        return float(spec)
    n = len(patches)
    other = np.array([[patches[i].antigen != patches[j].antigen for j in range(n)]
                      for i in range(n)])
    bg = np.asarray(S)[np.triu(other, 1) > 0]
    bg = bg[np.isfinite(bg)]
    return float(np.percentile(bg, 95.0)) if len(bg) else 0.75


def cluster_epitopes(patches: list, S: np.ndarray, cut: float, verbose: bool = False) -> int:
    """Group mutually confusable epitopes by average-linkage clustering on the cross-talk matrix.

    The preprint clusters its affinity matrix to expose which targets and compounds behave
    alike. Clustering the epitope-by-epitope similarity matrix answers the multiplexing
    question in the same idiom: epitopes in one cluster present a similar surface to a
    paratope, so a panel should draw at most one nanobody from each cluster.
    """
    n = len(patches)
    if n < 2:
        for p in patches:
            p.cluster, p.cluster_size, p.cluster_antigens = 0, n, 1
        return max(n, 1)
    D = np.clip(1.0 - np.asarray(S, np.float64), 0.0, None)
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    labels = None
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        Z = linkage(squareform(D, checks=False), method="average")
        labels = fcluster(Z, t=1.0 - cut, criterion="distance")
    except Exception:
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(n):
            for j in range(i + 1, n):
                if S[i, j] >= cut:
                    a, b = find(i), find(j)
                    if a != b:
                        parent[a] = b
        labels = np.array([find(i) for i in range(n)])
    remap, out = {}, []
    for lab in labels:
        remap.setdefault(int(lab), len(remap) + 1)
        out.append(remap[int(lab)])
    members = {}
    for p, lab in zip(patches, out):
        members.setdefault(lab, []).append(p)
    for lab, ps in members.items():
        for p in ps:
            p.cluster = lab
            p.cluster_size = len(ps)
            p.cluster_antigens = len({q.antigen for q in ps})
    vlog(f"cross-talk clustering at similarity {cut}: {len(members)} clusters over "
         f"{n} epitopes", verbose)
    return len(members)


PCA_DESCRIPTORS = ["area", "mean_rsasa", "protrusion", "concavity", "planarity", "contiguity",
                   "rigidity", "hydrophobic_frac", "net_charge", "eci", "pkd", "le", "lle",
                   "dangling", "exposed_hyd_frac", "con_disorder"]


def pca_epitopes(patches: list, verbose: bool = False) -> None:
    """PC1-PC3 of the standardised descriptor matrix, so the epitope landscape can be plotted."""
    if len(patches) < 4:
        return
    X = np.array([[float(getattr(p, d, 0.0) if d != "area" else p.area / 100.0)
                   for d in PCA_DESCRIPTORS] for p in patches], np.float64)
    X = X[:, np.nanstd(X, axis=0) > 1e-9]
    if X.shape[1] < 3:
        return
    X = (X - X.mean(0)) / X.std(0)
    X = np.nan_to_num(X)
    U, sv, _ = np.linalg.svd(X, full_matrices=False)
    scores = U * sv
    var = sv ** 2 / max((sv ** 2).sum(), 1e-12)
    for k, p in enumerate(patches):
        p.pcs = tuple(round(float(scores[k, j]), 4) if j < scores.shape[1] else 0.0
                      for j in range(3))
    vlog("PCA on epitope descriptors: PC1-PC3 explain "
         f"{100 * var[:3].sum():.1f} % of variance "
         f"({', '.join(f'{100 * v:.1f} %' for v in var[:3])})", verbose)


# ---------------------------------------------------------------------------------------------
# Per-residue representations and the cross-talk (MaxSim) engine
# ---------------------------------------------------------------------------------------------

ESMC_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / \
    "models--EvolutionaryScale--esmc-600m-2024-12"


class Embedder:
    """Per-residue chain embeddings. Backends: none (physicochemical only), esm2, esmc."""

    ESM2_DEFAULT = "facebook/esm2_t33_650M_UR50D"

    def __init__(self, backend: str, device: str = "cuda", verbose: bool = False,
                 model_name: str = None):
        self.backend, self.device, self.verbose = backend, device, verbose
        self.model_name = model_name or self.ESM2_DEFAULT
        self._m = self._t = None
        self.dim = 0

    @staticmethod
    def resolve(requested: str, verbose: bool = False) -> str:
        if requested != "auto":
            return requested
        try:
            import torch  # noqa: F401
        except Exception:
            vlog("embed auto -> none (torch not importable)", verbose)
            return "none"
        try:
            import esm  # noqa: F401
            if ESMC_CACHE.exists():
                return "esmc"
        except Exception:
            pass
        try:
            import transformers  # noqa: F401
            return "esm2"
        except Exception:
            return "none"

    def _ensure(self):
        if self._m is not None or self.backend == "none":
            return
        import torch
        dev = self.device
        if dev.startswith("cuda") and not torch.cuda.is_available():
            log("cuda unavailable, embedding on cpu")
            dev = self.device = "cpu"
        if self.backend == "esmc":
            from pocketscope.encoder import _load
            self._t, self._m = _load(dev)
            self.dim = 1152
        else:
            if not self.verbose:
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
                import transformers
                transformers.logging.set_verbosity_error()
                try:
                    transformers.utils.logging.disable_progress_bar()
                except Exception:
                    pass
            from transformers import AutoTokenizer, AutoModel
            name = self.model_name
            vlog(f"loading {name} on {dev}", self.verbose)
            self._t = AutoTokenizer.from_pretrained(name)
            self._m = AutoModel.from_pretrained(name).eval().to(dev)
            self.dim = int(self._m.config.hidden_size)

    def embed(self, seq: str) -> np.ndarray:
        """(len(seq), dim) or (len(seq), 0) when the backend is `none`."""
        if self.backend == "none":
            return np.zeros((len(seq), 0), np.float32)
        try:
            return self._embed(seq)
        except Exception as exc:
            if self.device == "cpu":
                raise
            log(f"warning: {type(exc).__name__} on {self.device} ({exc}); retrying on cpu")
            self._m = self._t = None
            if self.backend == "esmc":
                from pocketscope import encoder as _enc
                _enc._state["model"] = _enc._state["tok"] = None
            self.device = "cpu"
            return self._embed(seq)

    def _embed(self, seq: str) -> np.ndarray:
        self._ensure()
        win, ov = 900, 100
        if len(seq) <= win:
            return self._forward(seq)
        out = np.zeros((len(seq), self.dim), np.float32)
        cov = np.zeros(len(seq), np.float32)
        step = win - ov
        for s in range(0, len(seq), step):
            sub = seq[s:s + win]
            if not sub:
                break
            e = self._forward(sub)
            out[s:s + len(sub)] += e
            cov[s:s + len(sub)] += 1.0
            if s + win >= len(seq):
                break
        return out / np.maximum(cov, 1.0)[:, None]

    def _forward(self, seq: str) -> np.ndarray:
        import torch
        if self.backend == "esmc":
            enc = self._t([seq], return_tensors="pt", padding=True)
            with torch.no_grad():
                o = self._m.forward(sequence_tokens=enc["input_ids"].to(self.device),
                                    sequence_id=enc["attention_mask"].to(self.device))
            return o.embeddings[0, 1:1 + len(seq)].float().cpu().numpy()
        enc = self._t(seq, return_tensors="pt", add_special_tokens=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            o = self._m(**enc)
        return o.last_hidden_state[0, 1:1 + len(seq)].float().cpu().numpy()


def patch_features(p: Patch, st: Structure, chain_emb: dict, alpha: float) -> np.ndarray:
    """(L, D) unit-norm per-residue patch tensor, PocketScope Equation 1."""
    rs = [st.residues[i] for i in p.members]
    phys = residue_phys("".join(r.aa for r in rs))
    emb = chain_emb.get(p.chain)
    if emb is None or emb.shape[1] == 0:
        v = phys.astype(np.float32)
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        n[n == 0] = 1.0
        return (v / n).astype(np.float32)
    rows = emb[[r.chain_pos for r in rs]]
    return match_feature(rows, phys, alpha)


def patch_weights(p: Patch, st: Structure, surface: str = "monomer") -> np.ndarray:
    """Per-residue weights for MaxSim: how much of the patch surface each residue presents."""
    a = np.array([sasa_of(st.residues[i], surface) for i in p.members], np.float32)
    return a / max(float(a.sum()), 1e-9)


def maxsim(a: np.ndarray, b: np.ndarray, wa: np.ndarray = None) -> float:
    """Asymmetric late-interaction MaxSim of patch a against patch b.

    Query residues are weighted by their exposed area: a nanobody only ever sees the solvent
    exposed face of the patch, so a buried residue should not drive a cross-talk call.
    """
    m = (a @ b.T).max(1)
    if wa is None:
        return float(m.mean())
    return float((wa * m).sum() / max(wa.sum(), 1e-9))


def sym_maxsim(pa: "Patch", pb: "Patch") -> float:
    return 0.5 * (maxsim(pa.feats, pb.feats, pa.w) + maxsim(pb.feats, pa.feats, pb.w))


def _stack_patches(patches: list):
    """Pad every patch tensor to a common length -> (T, W, mask)."""
    n = len(patches)
    L = max(len(p.feats) for p in patches)
    D = patches[0].feats.shape[1]
    T = np.zeros((n, L, D), np.float32)
    W = np.zeros((n, L), np.float32)
    M = np.zeros((n, L), bool)
    for i, p in enumerate(patches):
        k = len(p.feats)
        T[i, :k] = p.feats
        W[i, :k] = p.w if p.w is not None else 1.0 / k
        M[i, :k] = True
    return T, W, M


def crosstalk_matrix(patches: list, chunk: int = 48, direction: str = "max") -> np.ndarray:
    """All-pairs weighted MaxSim, batched.

    The pairwise Python loop is fine for a handful of antigens and hopeless for a 96-well
    panel: 96 antigens give ~1150 patches and 660k pairs. Padding the patches into one tensor
    turns the whole matrix into a few large einsums instead.

    MaxSim is asymmetric, so the two directions must be combined. `direction`:

      max   (default) the pessimistic corner. Cross-talk is a risk, and a paratope raised on
            either patch could pick up the other, so the larger of the two readings is the
            one that matters. Odin-Multi applies the same discipline throughout, aggregating
            contexts as min(on-target) against max(off-target) and taking iPSAE as the min of
            its two directional scores.
      mean  the symmetric average used before v0.3.0. Softer, and it can hide a strongly
            one-directional resemblance.
      min   the optimistic corner. Only useful for diagnostics.
    """
    T, W, M = _stack_patches(patches)
    n = len(patches)
    S = np.zeros((n, n), np.float32)
    denom = np.maximum(W.sum(1), 1e-9)
    for s0 in range(0, n, chunk):
        e0 = min(s0 + chunk, n)
        sims = np.einsum("iqd,jkd->ijqk", T[s0:e0], T, optimize=True)
        sims = np.where(M[None, :, None, :], sims, -np.inf)
        mx = sims.max(3)                                      # (i, j, q)
        del sims
        wq = np.where(M[s0:e0], W[s0:e0], 0.0)[:, None, :]
        mx = np.where(np.isfinite(mx), mx, 0.0)
        S[s0:e0] = (mx * wq).sum(2) / denom[s0:e0, None]
    if direction == "mean":
        return 0.5 * (S + S.T)
    if direction == "min":
        return np.minimum(S, S.T)
    return np.maximum(S, S.T)


def crosstalk(patches: list, verbose: bool = False, direction: str = "max",
              top_partners: int = 3) -> np.ndarray:
    """Fill xt_* on every patch and return the full patch-by-patch similarity matrix.

    `crosstalk_max` is the raw similarity to the best-matching patch on a DIFFERENT antigen.
    Because that is a maximum over many patches it is always high on an absolute scale, so the
    orthogonality axis uses the rank of that maximum within the candidate pool: the patch whose
    worst cross-antigen match is the mildest scores 1.0, the patch with the worst match scores 0.
    `crosstalk_z` keeps the absolute reading, as a robust z-score against the background of all
    cross-antigen patch pairs.
    """
    n = len(patches)
    S = crosstalk_matrix(patches, direction=direction)
    np.fill_diagonal(S, -np.inf)
    if n < 2:
        for p in patches:
            p.s_ortho = 1.0
        return S

    other = np.array([[patches[i].antigen != patches[j].antigen for j in range(n)]
                      for i in range(n)])
    bg = S[np.triu(other, 1) > 0]
    bg = bg[np.isfinite(bg)]
    med = float(np.median(bg)) if len(bg) else 0.0
    mad = float(np.median(np.abs(bg - med))) * 1.4826 if len(bg) else 0.0

    for i, p in enumerate(patches):
        row = np.where(other[i], S[i], -np.inf)
        if np.isfinite(row).any():
            order = np.argsort(-row)
            j = int(order[0])
            p.xt_max = round(float(row[j]), 4)
            p.xt_partner = patches[j].pid
            top = [float(row[k]) for k in order[:3] if np.isfinite(row[k])]
            p.xt_top3 = round(float(np.mean(top)), 4)
            p.xt_z = round((p.xt_max - med) / mad, 3) if mad > 1e-9 else 0.0
        else:
            p.xt_max, p.xt_top3, p.xt_partner, p.xt_z = 0.0, 0.0, "", 0.0
        same = np.where(~other[i], S[i], -np.inf)
        p.xt_self = round(float(same.max()), 4) if np.isfinite(same).any() else 0.0

    if not len(bg):
        # only one antigen in the collection: there is nothing to cross-talk with
        for p in patches:
            p.xt_pct, p.s_ortho = 0.0, 1.0
        return S
    # Odin-Multi qualifies a design on an absolute specificity ratio rather than on its rank
    # among the other designs (summary.py: min(off-target i_pae) / max(target i_pae) >= 1.5).
    # The analogue in similarity space is a margin ratio: how far this epitope's nearest
    # look-alike sits, in units of how far a TYPICAL epitope's nearest look-alike sits.
    #
    # The reference has to be the median of the per-patch maxima, not the median of all
    # pairs. crosstalk_max is a maximum over every patch on every other antigen, so it always
    # sits far out in the pairwise distribution; dividing by the pairwise median would put
    # almost everything below 1.0 by construction and say nothing. Comparing like with like --
    # one order statistic against the same order statistic -- makes the ratio mean what it
    # claims: above 1.0, this epitope's worst look-alike is further away than usual.
    for i, p in enumerate(patches):
        # the k most similar DISTINCT other antigens, for off-target counter-selection
        row = np.where(other[i], S[i], -np.inf)
        seen, ctx = set(), []
        for j in np.argsort(-row):
            if not np.isfinite(row[j]):
                break
            ag = patches[int(j)].antigen
            if ag in seen or ag == p.antigen:
                continue
            seen.add(ag)
            ctx.append(f"{ag}:{row[int(j)]:.4f}")
            if len(ctx) >= top_partners:
                break
        p.offtarget_contexts = ";".join(ctx)

    m = np.array([p.xt_max for p in patches], np.float64)
    ref = 1.0 - float(np.median(m[np.isfinite(m)])) if np.isfinite(m).any() else 0.0
    for p in patches:
        p.spec_ratio = round(float((1.0 - p.xt_max) / ref), 4) if ref > 1e-9 else 1.0

    rank = np.empty(n, np.float64)
    rank[np.argsort(m, kind="stable")] = np.arange(n)
    for i, p in enumerate(patches):
        p.xt_pct = round(float(rank[i] / max(n - 1, 1)), 4)
        p.s_ortho = float(1.0 - p.xt_pct)
    return S


def assign_status(patches: list, args, verbose: bool = False) -> dict:
    """Label every candidate with an explicit status instead of silently ranking all of them.

    Odin-Multi records why a design did not become a candidate (`incomplete_contexts`,
    `below_specificity_ratio`, `duplicate_sequence`, ...) rather than dropping it. The same
    vocabulary here: `redundant` is already set by the site enumerator, and everything else
    is gated on the absolute specificity ratio.
    """
    counts: dict = {}
    for p in patches:
        if p.status != "redundant":
            p.status = ("ranked" if p.spec_ratio >= args.specificity_ratio
                        else "below_specificity_ratio")
        counts[p.status] = counts.get(p.status, 0) + 1
    vlog("candidate status: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
         verbose)
    return counts


def select_panel(patches: list, S: np.ndarray, args, verbose: bool = False):
    """Pick one epitope per antigen so that the worst pairwise cross-talk in the panel is small.

    This is the multiplexing question directly: if every antigen in the tube gets its own
    nanobody, which set of epitopes gives the least chance that any nanobody also binds a
    neighbour. Solved by iterated best-response from several starts, which is exact enough for
    the handful of candidates per antigen that survive the quality filters.
    """
    idx_of = {id(p): i for i, p in enumerate(patches)}
    by_ag: dict = {}
    for p in patches:
        if p.status == "ranked":
            by_ag.setdefault(p.antigen, []).append(p)
    if len(by_ag) < 2:
        for ps in by_ag.values():
            best = max(ps, key=lambda p: _geomean([p.s_epitope, p.s_pure]))
            best.panel_sel = True
        return 0.0

    pool = {}
    for ag, ps in by_ag.items():
        ok = [p for p in ps if p.s_epitope >= args.panel_min_epitope
              and p.s_pure >= args.panel_min_pure]
        ranked = sorted(ok or ps, key=lambda p: -_geomean([p.s_epitope, p.s_pure]))
        pool[ag] = ranked[:max(args.panel_pool, 1)]
    ags = sorted(pool)
    Sm = np.asarray(S, np.float64).copy()
    np.fill_diagonal(Sm, -np.inf)

    cand_idx = {a: np.array([idx_of[id(p)] for p in pool[a]]) for a in ags}
    cand_q = {a: np.array([_geomean([p.s_epitope, p.s_pure]) for p in pool[a]]) for a in ags}
    qw = args.panel_quality_weight
    m = len(ags)

    def total_cost(pick):
        idx = np.array([pick[a] for a in ags])
        sub = Sm[np.ix_(idx, idx)]
        iu = np.triu_indices(m, 1)
        worst = float(sub[iu].max()) if m > 1 else 0.0
        q = float(np.mean([cand_q[a][list(cand_idx[a]).index(pick[a])] for a in ags]))
        return worst + qw * (1.0 - q), worst

    rng = np.random.default_rng(0)
    restarts = max(1, min(args.panel_restarts, max(4, 400 // max(m, 1))))
    starts = [{a: int(cand_idx[a][0]) for a in ags}]
    for _ in range(restarts):
        starts.append({a: int(rng.choice(cand_idx[a])) for a in ags})

    best_pick, best_cost = None, math.inf
    for pick in starts:
        pick = dict(pick)
        for _ in range(30):
            moved = False
            for a in ags:
                others = np.array([pick[b] for b in ags if b != a])
                if len(others) == 0:
                    continue
                sub = Sm[np.ix_(others, others)]
                base = float(sub[np.triu_indices(len(others), 1)].max()) \
                    if len(others) > 1 else 0.0
                ci = cand_idx[a]
                vs = Sm[np.ix_(ci, others)].max(1)          # each candidate vs the rest
                worst = np.maximum(base, vs)
                # mean quality with this antigen's contribution swapped in
                qothers = sum(cand_q[b][list(cand_idx[b]).index(pick[b])]
                              for b in ags if b != a)
                qmean = (qothers + cand_q[a]) / m
                costs = worst + qw * (1.0 - qmean)
                k = int(np.argmin(costs))
                if int(ci[k]) != pick[a] and costs[k] < costs[list(ci).index(pick[a])] - 1e-12:
                    pick[a] = int(ci[k])
                    moved = True
            if not moved:
                break
        c, _ = total_cost(pick)
        if c < best_cost:
            best_pick, best_cost = dict(pick), c

    best_sel = {}
    for a in ags:
        gi = best_pick[a]
        best_sel[a] = next(p for p in pool[a] if idx_of[id(p)] == gi)
    ii = {a: idx_of[id(best_sel[a])] for a in ags}
    worst_overall = 0.0
    for a in ags:
        p = best_sel[a]
        p.panel_sel = True
        pairs = [(float(S[ii[a]][ii[b]]), best_sel[b].pid) for b in ags if b != a]
        v, who = max(pairs)
        p.panel_worst, p.panel_worst_partner = round(v, 4), who
        worst_overall = max(worst_overall, v)
    log(f"panel: one epitope per antigen, worst pairwise cross-talk {worst_overall:.4f}")
    for a in ags:
        p = best_sel[a]
        log(f"    {p.pid:<26} epitope={p.s_epitope:.2f} pure={p.s_pure:.2f} "
            f"spec={p.spec_ratio:.2f} worst-vs-panel={p.panel_worst:.4f} "
            f"vs {p.panel_worst_partner}")
    weak = [a for a in ags if best_sel[a].spec_ratio < args.specificity_ratio]
    missing = sorted({p.antigen for p in patches} - set(ags))
    if weak or missing:
        log(f"panel verdict: NOT SEPARABLE at specificity ratio "
            f"{args.specificity_ratio:g}")
        if missing:
            log(f"    no candidate cleared the gate for: {', '.join(missing[:8])}"
                + (" ..." if len(missing) > 8 else ""))
        if weak:
            log(f"    selected but under the gate: {', '.join(weak[:8])}"
                + (" ..." if len(weak) > 8 else ""))
    else:
        log(f"panel verdict: separable, every member clears specificity ratio "
            f"{args.specificity_ratio:g}")
    return worst_overall


def proteome_screen(patches: list, index_dir: str, device: str, top_k: int,
                    verbose: bool = False) -> None:
    """Optional: score every patch against the prebuilt PocketScope human pocketome."""
    from pocketscope.index import PocketIndex
    ix = PocketIndex(index_dir, device=device)
    vlog(f"pocketome index: {ix.n} pockets x {ix.n_max} x {ix.dim}", verbose)
    for p in patches:
        if p.feats.shape[1] != ix.dim:
            log(f"skipping proteome screen: patch dim {p.feats.shape[1]} != index dim {ix.dim} "
                f"(needs --embed esmc)")
            return
        hits = ix.search(p.feats, top_k=top_k)
        if hits:
            p.off_max = hits[0]["score"]
            p.off_hit = f"{hits[0]['accession']}:{hits[0]['pocket_id']}"


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------

FIELDS = [
    "rank", "epitope_id", "antigen", "chain", "n_epitope_residues",
    "nanobody_target_score", "epitope_score", "orthogonality_score", "pure_ivtt_score",
    "status", "redundant_with", "specificity_ratio",
    "pure_zone", "n_pure_rules_passed", "pure_rules_all_pass", "flags",
    "site_source", "site_buriedness", "site_center_x", "site_center_y", "site_center_z",
    "site_box_A",
    "dg_est_kcal_mol", "kd_est_M", "pkd_est", "buried_area_est_A2", "apolar_index",
    "interface_nres", "interface_dsasa_est_A2", "interface_dg_dsasa_ratio",
    "interface_hydrophobicity",
    "epitope_le", "epitope_lle", "epitope_fq", "epitope_complexity_index",
    "target_residues", "target_hotspot_residues", "odin_hotspots_core",
    "epitope_residues", "rfdiffusion_hotspots", "epitope_sequence",
    "crosstalk_max", "crosstalk_partner", "crosstalk_top3", "crosstalk_percentile",
    "crosstalk_within_antigen", "crosstalk_z", "offtarget_contexts",
    "panel_selected", "panel_worst_crosstalk", "panel_worst_partner",
    "crosstalk_cluster", "cluster_size", "cluster_n_antigens", "pc1", "pc2", "pc3",
    "proteome_offtarget_max", "proteome_offtarget_hit",
    "patch_area_A2", "mean_rel_sasa", "protrusion", "concavity", "planarity_A",
    "seq_contiguity", "rigidity", "mean_bfactor_or_plddt", "net_charge",
    "hydrophobic_frac", "interface_buried_A2", "cofactor_contact",
    "construct_range", "construct_len", "construct_mw_da", "construct_pI", "construct_gravy",
    "construct_disorder_frac", "dangling_contact_frac", "n_cys", "ss_internal",
    "ss_broken_by_excision", "nglyc_sequons_construct", "nglyc_sequons_in_epitope",
    "max_hydropathy_w19", "exposed_hydrophobic_frac",
    "rule_size_pass", "rule_redox_pass", "rule_glyco_pass", "rule_membrane_pass",
    "rule_domain_pass", "rule_monomer_pass", "rule_order_pass",
    "epi_area", "epi_topography", "epi_rigidity", "epi_cleft", "epi_exposure", "epi_polarity",
    "pure_length", "pure_redox", "pure_glycan", "pure_membrane", "pure_excisable",
    "pure_aggregation", "pure_order", "pure_monomeric", "pure_cofactor_free", "pure_solubility",
    "construct_sequence",
]


def flags_for(p: Patch) -> str:
    f = []
    if p.ss_broken:
        f.append(f"SS_BROKEN_BY_EXCISION({p.ss_broken})")
    if p.ss_internal:
        f.append(f"DISULFIDE_IN_CONSTRUCT({p.ss_internal})")
    if p.n_sequon_patch:
        f.append(f"NGLYC_IN_EPITOPE({p.n_sequon_patch})")
    if p.tm_hydropathy > 1.6:
        f.append("TM_LIKE_SEGMENT")
    elif p.tm_hydropathy > 1.1:
        f.append("HYDROPHOBIC_SEGMENT")
    if p.interface_buried >= 60.0:
        f.append(f"QUATERNARY_EPITOPE({int(p.interface_buried)}A2)")
    if p.cofactor in GLYCANS:
        f.append(f"GLYCAN_SHIELDED({p.cofactor})")
    elif p.cofactor:
        f.append(f"COFACTOR({p.cofactor})")
    if p.con_disorder > 0.25:
        f.append("DISORDERED_CONSTRUCT")
    if p.dangling > 0.30:
        f.append("POORLY_EXCISABLE")
    if len(p.con_seq) > 300:
        f.append("LARGE_CONSTRUCT")
    if p.xt_pct >= 0.90:
        f.append("HIGH_CROSSTALK")
    if p.cluster_size > 1 and p.cluster_antigens > 1:
        f.append(f"SHARED_CLUSTER({p.cluster_antigens}_antigens)")
    if p.lle < 2.0:
        f.append("LOW_LLE_GREASY")
    if p.fq and p.fq < 0.8:
        f.append("LOW_FIT_QUALITY")
    if p.status == "below_specificity_ratio":
        f.append(f"BELOW_SPECIFICITY({p.spec_ratio:.2f})")
    if p.status == "redundant":
        f.append(f"REDUNDANT_WITH({p.redundant_with})")
    return ";".join(f) or "-"


def row_for(p: Patch, st: Structure, rank: int, surface: str = "monomer") -> dict:
    rs = [st.residues[i] for i in p.members]
    top = sorted(rs, key=lambda r: -sasa_of(r, surface))[:5]
    return {
        "rank": rank,
        "epitope_id": p.pid,
        "antigen": p.antigen,
        "chain": p.chain,
        "n_epitope_residues": len(p.members),
        "nanobody_target_score": round(p.total, 4),
        "epitope_score": round(p.s_epitope, 4),
        "orthogonality_score": round(p.s_ortho, 4),
        "pure_ivtt_score": round(p.s_pure, 4),
        "status": p.status,
        "redundant_with": p.redundant_with,
        "specificity_ratio": p.spec_ratio,
        "pure_zone": p.zone,
        "n_pure_rules_passed": sum(p.rules.values()),
        "pure_rules_all_pass": int(all(p.rules.values())),
        "flags": flags_for(p),
        "site_source": p.source,
        "site_buriedness": round(p.buriedness, 4),
        "site_center_x": round(float(p.centre[0]), 3) if p.centre is not None else "",
        "site_center_y": round(float(p.centre[1]), 3) if p.centre is not None else "",
        "site_center_z": round(float(p.centre[2]), 3) if p.centre is not None else "",
        "site_box_A": round(p.box, 2),
        "dg_est_kcal_mol": round(p.dg, 3),
        "kd_est_M": f"{p.kd:.3e}",
        "pkd_est": round(p.pkd, 3),
        "buried_area_est_A2": round(p.bsa, 1),
        "apolar_index": round(p.apolar_index, 3),
        # Rosetta InterfaceAnalyzer vocabulary, so these join against a design pipeline's
        # own interface table. Estimated from geometry here, not from a scored complex.
        "interface_nres": len(p.members),
        "interface_dsasa_est_A2": round(2.0 * p.bsa, 1),
        "interface_dg_dsasa_ratio": round(100.0 * p.dg / max(2.0 * p.bsa, 1e-9), 4),
        "interface_hydrophobicity": round(100.0 * p.hydrophobic_frac, 1),
        "epitope_le": round(p.le, 4),
        "epitope_lle": round(p.lle, 3),
        "epitope_fq": round(p.fq, 4),
        "epitope_complexity_index": round(p.eci, 4),
        **{f"rule_{k}_pass": int(v) for k, v in p.rules.items()},
        "target_residues": ",".join(f"{r.chain}:{r.resseq}{r.icode.strip()}"
                                    for r in sorted(rs, key=lambda x: (x.chain, x.resseq,
                                                                       x.icode))),
        "target_hotspot_residues": collapse_residues(
            ",".join(f"{r.chain}:{r.resseq}{r.icode.strip()}"
                     for r in sorted(rs, key=lambda x: x.resseq))),
        "odin_hotspots_core": collapse_residues(
            ",".join(f"{r.chain}:{r.resseq}{r.icode.strip()}"
                     for r in sorted(top, key=lambda x: x.resseq))),
        "epitope_residues": " ".join(f"{r.aa}{r.resseq}" for r in sorted(rs, key=lambda x: x.resseq)),
        "rfdiffusion_hotspots": ",".join(f"{r.chain}{r.resseq}" for r in top),
        "epitope_sequence": "".join(r.aa for r in sorted(rs, key=lambda x: x.resseq)),
        "crosstalk_max": p.xt_max,
        "crosstalk_partner": p.xt_partner or "-",
        "crosstalk_top3": p.xt_top3,
        "crosstalk_percentile": p.xt_pct,
        "crosstalk_within_antigen": p.xt_self,
        "crosstalk_z": p.xt_z,
        "offtarget_contexts": p.offtarget_contexts,
        "panel_selected": 1 if p.panel_sel else 0,
        "panel_worst_crosstalk": p.panel_worst if p.panel_sel else "",
        "panel_worst_partner": p.panel_worst_partner if p.panel_sel else "",
        "crosstalk_cluster": p.cluster,
        "cluster_size": p.cluster_size,
        "cluster_n_antigens": p.cluster_antigens,
        "pc1": p.pcs[0], "pc2": p.pcs[1], "pc3": p.pcs[2],
        "proteome_offtarget_max": p.off_max or "",
        "proteome_offtarget_hit": p.off_hit or "",
        "patch_area_A2": round(p.area, 1),
        "mean_rel_sasa": round(p.mean_rsasa, 4),
        "protrusion": round(p.protrusion, 4),
        "concavity": round(p.concavity, 4),
        "planarity_A": round(p.planarity, 3),
        "seq_contiguity": round(p.contiguity, 3),
        "rigidity": round(p.rigidity, 4),
        "mean_bfactor_or_plddt": round(p.mean_bfac, 2),
        "net_charge": p.net_charge,
        "hydrophobic_frac": round(p.hydrophobic_frac, 3),
        "interface_buried_A2": round(p.interface_buried, 1),
        "cofactor_contact": p.cofactor or "-",
        "construct_range": p.con_range,
        "construct_len": len(p.con_seq),
        "construct_mw_da": p.con_mw,
        "construct_pI": p.con_pi,
        "construct_gravy": p.con_gravy,
        "construct_disorder_frac": p.con_disorder,
        "dangling_contact_frac": p.dangling,
        "n_cys": p.n_cys,
        "ss_internal": p.ss_internal,
        "ss_broken_by_excision": p.ss_broken,
        "nglyc_sequons_construct": p.n_sequon_construct,
        "nglyc_sequons_in_epitope": p.n_sequon_patch,
        "max_hydropathy_w19": p.tm_hydropathy,
        "exposed_hydrophobic_frac": p.exposed_hyd_frac,
        **{f"epi_{k}": round(v, 4) for k, v in p.epi_parts.items() if k != "prelim"},
        **{f"pure_{k}": round(v, 4) for k, v in p.pure_parts.items()},
        "construct_sequence": p.con_seq,
    }


def write_csv(path: Path, patches: list, st: Structure, surface: str = "monomer") -> None:
    """Ranked candidates first and numbered; everything else follows with an empty rank.

    A redundant or below-threshold row must not consume a rank, or it displaces a real
    candidate and quietly corrupts any recall@N computed downstream.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    ranked = [p for p in patches if p.status == "ranked"]
    others = [p for p in patches if p.status != "ranked"]
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for rank, p in enumerate(ranked, 1):
            w.writerow(row_for(p, st, rank, surface))
        for p in others:
            w.writerow(row_for(p, st, "", surface))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------
# Odin-Multi export
#
# Odin-Multi (https://github.com/DigBioLab/odin_multi) designs one shared binder sequence
# against several fixed contexts at once, with attractive losses on targets and repulsive
# losses on off-targets. That is the natural consumer of a panel: EpitopeScope decides which
# epitope each nanobody should aim at and which neighbours it must avoid, and Odin-Multi turns
# that into a design run. Only the small JSON contract is reproduced here -- none of
# Odin-Multi's AlphaFold dependencies are needed to write it.
# ---------------------------------------------------------------------------------------------

def collapse_residues(target_residues: str) -> str:
    """`A:42,A:43,A:44,A:57` -> `A42-44,A57`, the target_hotspot_residues syntax."""
    by_chain: dict = {}
    for tok in target_residues.split(","):
        if ":" not in tok:
            continue
        c, n = tok.split(":", 1)
        digits = "".join(ch for ch in n if ch.isdigit() or ch == "-")
        if digits.lstrip("-").isdigit():
            by_chain.setdefault(c, []).append(int(digits))
    out = []
    for c, nums in by_chain.items():
        nums = sorted(set(nums))
        i = 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            out.append(f"{c}{nums[i]}" if i == j else f"{c}{nums[i]}-{nums[j]}")
            i = j + 1
    return ",".join(out)


def odin_hotspots(p: Patch, st: Structure, mode: str, n_core: int, surface: str) -> str:
    """Hotspot string for one epitope.

    `core` takes the n most exposed residues. A whole 22-residue patch as hotspots
    over-constrains the design: the shipped Odin-Multi example uses nine, and RFdiffusion
    practice is three to six.
    """
    rs = [st.residues[i] for i in p.members]
    if mode == "core":
        rs = sorted(rs, key=lambda r: -sasa_of(r, surface))[:max(n_core, 1)]
    return collapse_residues(",".join(f"{r.chain}:{r.resseq}{r.icode.strip()}"
                                      for r in sorted(rs, key=lambda x: x.resseq)))


def _portable_path(target: Path, base: Path) -> str:
    rel = os.path.relpath(target, base)
    return rel if rel.count("..") <= 3 else str(target)


def write_odin_export(outdir: Path, patches: list, structures: dict, args) -> Path:
    """Write settings_target JSONs and the matching `odin_multi.py design` command lines."""
    import json

    outdir.mkdir(parents=True, exist_ok=True)
    tdir = outdir / "settings_target"
    tdir.mkdir(exist_ok=True)
    st_of = {st.path.stem: st for st in structures.values()}
    # The panel pick per antigen, plus the next few ranked sites when --odin-top > 1.
    # Benchmarks put the validated epitope first only about half the time, so committing a
    # design campaign to rank 1 alone throws away sites that are in the list; Odin-Multi's
    # own re-evaluation stage is the right place to arbitrate between them.
    ranked_by_ag: dict = {}
    for p in sorted((q for q in patches if q.status == "ranked"), key=lambda q: -q.total):
        ranked_by_ag.setdefault(p.antigen, []).append(p)
    by_ag, extra = {}, []
    for ag, lst in ranked_by_ag.items():
        head = next((q for q in lst if q.panel_sel), lst[0])
        by_ag[ag] = head
        for q in lst:
            if q is not head and len(extra) < 10 ** 6:
                extra.append(q)
    top_extra: dict = {}
    for ag, lst in ranked_by_ag.items():
        head = by_ag[ag]
        top_extra[ag] = [q for q in lst if q is not head][:max(args.odin_top - 1, 0)]

    # A JSON is needed for every antigen that appears as a target OR as somebody's
    # off-target, otherwise a --context line points at a file that was never written.
    needed = set(by_ag)
    for p in by_ag.values():
        for c in p.offtarget_contexts.split(";"):
            if c:
                needed.add(c.split(":")[0])
    # An off-target context does not have to clear the specificity gate: the point is to
    # tell the design run what surface to avoid, and an antigen with no qualifying epitope
    # of its own is exactly the one worth avoiding. Any non-redundant site will do.
    best_of = {}
    for p in patches:
        if p.status == "redundant":
            continue
        if p.antigen not in best_of or p.total > best_of[p.antigen].total:
            best_of[p.antigen] = p

    written, targets = [], []
    for ag in sorted(needed):
        st = st_of.get(ag)
        p = by_ag.get(ag) or best_of.get(ag)
        if st is None or p is None:
            continue
        cfg = {
            "binder_name": f"nb_{ag}",
            # relative when that stays readable, absolute otherwise; Odin-Multi accepts
            # either, and a path with eight levels of `..` helps nobody
            "starting_pdb": _portable_path(st.path.resolve(), tdir.resolve()),
            "chains": ",".join(sorted(st.chains)),
            "target_hotspot_residues": odin_hotspots(p, st, args.odin_hotspot_mode,
                                                     args.odin_hotspot_n, args.surface),
        }
        f = tdir / f"{ag}.json"
        f.write_text(json.dumps(cfg, indent=2) + "\n")
        written.append((ag, p, f))
        if ag in by_ag:
            targets.append((ag, p, f))

    lines = ["#!/usr/bin/env bash",
             "# Generated by epitopescope " + __version__ + ".",
             "# One Odin-Multi design run per antigen: the panel epitope is the target context,",
             "# and the antigens it most resembles become off-target contexts to counter-select.",
             "# Pair each target JSON with settings_loss/target.json or offtarget.json.",
             "set -euo pipefail", ""]
    # alternates get their own target file and their own run
    for ag, alts in sorted(top_extra.items()):
        st = st_of.get(ag)
        if st is None:
            continue
        for k, q in enumerate(alts, 2):
            cfg = {
                "binder_name": f"nb_{ag}_alt{k}",
                "starting_pdb": _portable_path(st.path.resolve(), tdir.resolve()),
                "chains": ",".join(sorted(st.chains)),
                "target_hotspot_residues": odin_hotspots(q, st, args.odin_hotspot_mode,
                                                         args.odin_hotspot_n, args.surface),
            }
            f2 = tdir / f"{ag}_alt{k}.json"
            f2.write_text(json.dumps(cfg, indent=2) + "\n")
            written.append((f"{ag}_alt{k}", q, f2))
            targets.append((f"{ag}_alt{k}", q, f2))

    have = {ag for ag, _, _ in written}
    for ag, p, f in targets:
        offs = [c.split(":")[0] for c in p.offtarget_contexts.split(";") if c]
        base = ag.split("_alt")[0]
        offs = [o for o in offs if o in have and o != base][:args.odin_offtargets]
        lines.append(f"# {p.pid}  spec_ratio={p.spec_ratio:.2f}  "
                     f"worst-vs-panel={p.panel_worst:.4f}")
        cmd = ["python -u odin_multi.py design",
               f"  --run-dir outputs/{ag}",
               f"  --context settings_target/{ag}.json settings_loss/target.json"]
        for o in offs:
            cmd.append(f"  --context settings_target/{o}.json settings_loss/offtarget.json")
        cmd += ["  --advanced settings_advanced/general.json",
                "  --base-seed 42 --num-designs 100"]
        lines.append(" \\\n".join(cmd))
        lines.append("")
    sh = outdir / "odin_design_commands.sh"
    sh.write_text("\n".join(lines))
    sh.chmod(0o755)
    return outdir


def check_environment(args) -> int:
    """Preflight the things that silently degrade a run, in the style of Odin-Multi's
    validate_install.py: report each check, and fail only on what actually blocks."""
    ok = True
    log(f"epitopescope {__version__}")
    log(f"  python        {sys.version.split()[0]}")
    log(f"  numpy         {np.__version__}")
    try:
        import scipy
        log(f"  scipy         {scipy.__version__}")
    except Exception:
        log("  scipy         MISSING (slower fallbacks will be used)")
    backend = Embedder.resolve(args.embed, True)
    log(f"  --embed       {args.embed} -> {backend}")
    if backend == "none":
        log("  WARNING       no protein language model; cross-talk will be weak")
    for mod in {"esm2": ["torch", "transformers"], "esmc": ["torch", "esm"]}.get(backend, []):
        try:
            m = __import__(mod)
            log(f"  {mod:<13} {getattr(m, '__version__', 'ok')}")
        except Exception as exc:
            log(f"  {mod:<13} MISSING ({exc})")
            ok = False
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                d = torch.cuda.get_device_properties(i)
                log(f"  cuda:{i}        {d.name}, {d.total_memory / 2 ** 30:.1f} GiB")
        else:
            log(f"  cuda          NOT AVAILABLE. torch {torch.__version__} is built for CUDA "
                f"{torch.version.cuda}; --device cuda will silently fall back to cpu. "
                f"Install a torch whose CUDA version your driver supports.")
    except Exception:
        pass
    log("environment OK" if ok else "environment INCOMPLETE")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------
# Metric dictionary
#
# Every column carries a description plus what a high and a low value actually mean, so a CSV
# can be read without the source. `colour` follows the convention of the example schema:
# RdBu for numeric columns, Spectral for identifiers and categorical text.
# ---------------------------------------------------------------------------------------------

#   name: (description, meaning of a HIGH value, meaning of a LOW value, direction)
# direction: "higher_is_better", "lower_is_better", "context" (neither is inherently good),
#            or "" for identifiers and categorical fields.
METRIC_DOCS = {
 "rank": ("Position of this epitope within its antigen, 1 = best nanobody_target_score.",
   "Ranked far down; better candidates exist on the same antigen.",
   "Top candidate for this antigen.", "lower_is_better"),
 "epitope_id": ("Unique epitope label, <antigen>|<chain>|E<nn>.", "", "", ""),
 "antigen": ("Input PDB file stem this epitope belongs to.", "", "", ""),
 "chain": ("Chain of the antigen carrying the epitope.", "", "", ""),
 "n_epitope_residues": ("Residues in the epitope patch.",
   "A large patch; more contact surface but harder for one VHH paratope to cover fully.",
   "A small patch; easy to cover but offers little to bind.", "context"),
 "nanobody_target_score": ("Overall ranking score: weighted geometric mean of the epitope, "
   "orthogonality and PURE axes, so a failure on any one axis cannot be masked.",
   "Good epitope, orthogonal to the other antigens, and expressible in PURE.",
   "Fails on at least one axis; read the three sub-scores to see which.", "higher_is_better"),
 "epitope_score": ("How plausible this patch is as a nanobody epitope from geometry alone.",
   "Well exposed, right size for a VHH paratope, rigid, chemically balanced.",
   "Too small, too flat, too floppy or too greasy to be a good epitope.", "higher_is_better"),
 "orthogonality_score": ("Rank of this epitope's worst cross-antigen match within the "
   "candidate pool; 1 = the mildest worst-case match in the run.",
   "Distinctive: nothing on the other antigens looks like it.",
   "Its closest look-alike on another antigen is among the most similar pairs in the run.",
   "higher_is_better"),
 "pure_ivtt_score": ("How likely the minimal construct carrying this epitope is to be made "
   "and folded by an E. coli PURE in-vitro transcription/translation reaction.",
   "Compact, cysteine-light, ordered, excisable, monomeric: a good PURE construct.",
   "Blocked by disulfides, glycosylation, hydrophobic segments, disorder or size.",
   "higher_is_better"),
 "status": ("Why this row is or is not a candidate: `ranked` (a real candidate), "
   "`redundant` (a strongly overlapping site suppressed in favour of the one named in "
   "redundant_with, kept for the audit trail rather than discarded), or "
   "`below_specificity_ratio` (too confusable with another antigen to qualify).",
   "", "", ""),
 "redundant_with": ("For a redundant row, the epitope_id it overlaps. Empty otherwise.",
   "", "", ""),
 "specificity_ratio": ("How far this epitope's nearest look-alike on another antigen sits, "
   "in units of how far a typical cross-antigen pair sits. The absolute qualification rule, "
   "in the style of Odin-Multi's min(off-target i_pae) / max(target i_pae) gate.",
   "Much better separated from the rest of the panel than an average pair: safe to design "
   "against.",
   "Below 1.0 means more confusable than an average cross-antigen pair. This is the number "
   "that can declare a whole panel unusable, which a rank-based score cannot.",
   "higher_is_better"),
 "target_hotspot_residues": ("The whole epitope in Odin-Multi / BindCraft hotspot syntax, "
   "<chain><resnum> with ranges collapsed, e.g. A10,A14-18. Paste into a settings_target "
   "JSON.", "", "", ""),
 "odin_hotspots_core": ("The most exposed residues only, same syntax. Passing a whole "
   "22-residue epitope as hotspots tends to over-constrain a design run; this is the "
   "three-to-six residue core that design tools expect.", "", "", ""),
 "offtarget_contexts": ("The most similar DISTINCT other antigens, as antigen:similarity "
   "pairs. These are the counter-selection contexts for a multi-target design run.",
   "", "", ""),
 "interface_nres": ("Epitope residue count, under the Rosetta InterfaceAnalyzer name so "
   "this table joins against a design pipeline's interface metrics.",
   "A large interface.", "A small interface.", "context"),
 "interface_dsasa_est_A2": ("Estimated total buried area of the interface, both sides, the "
   "dSASA analogue. Estimated from geometry, not from a scored complex.",
   "A large contact area is available.", "Little area to bury.", "higher_is_better"),
 "interface_dg_dsasa_ratio": ("Estimated dG per 100 A^2 of buried area, the Rosetta "
   "dG_dSASA_ratio analogue and the same normalisation idea as epitope_le.",
   "Close to zero: poor energy return per unit area buried.",
   "Strongly negative: efficient use of the interface.", "lower_is_better"),
 "interface_hydrophobicity": ("Percentage of epitope residues with apolar side chains, the "
   "Rosetta interface_hydrophobicity analogue.",
   "A greasy interface: sticky and often non-specific.",
   "A polar interface, usually more selectively recognisable.", "context"),
 "pure_zone": ("Three-colour developability call, the BOILED-Egg analogue: GREEN, YELLOW "
   "or RED from construct length, exposed hydrophobic surface and rules passed.",
   "GREEN: short, polar-surfaced and clearing at least 6 of the 7 rules. Order it.",
   "RED: too long, too greasy, or failing several rules. Redesign or pick another epitope.",
   "context"),
 "n_pure_rules_passed": ("How many of the 7 named PURE rules this construct passes.",
   "Clears nearly every named liability.",
   "Trips several independent liabilities at once.", "higher_is_better"),
 "pure_rules_all_pass": ("1 when all 7 PURE rules pass, else 0.",
   "No named liability at all.", "At least one rule fails.", "higher_is_better"),
 "flags": ("Human-readable list of the specific liabilities detected, '-' when none.",
   "", "", ""),
 "site_source": ("Which detector proposed this site: concavity_cluster (a surface depression "
   "found by ray-cast concavity clustering) or surface_seed (an exposed-residue seed).",
   "", "", ""),
 "site_buriedness": ("Fraction of outward rays from the patch that are blocked again by the "
   "protein within reach; the concavity measure.",
   "A groove or pocket, walled in on most sides; the site type nanobody CDR3 loops favour.",
   "A convex bulge sticking out into solvent.", "context"),
 "site_center_x": ("x of the patch centroid, angstrom, in the input file's frame.", "", "", ""),
 "site_center_y": ("y of the patch centroid, angstrom, in the input file's frame.", "", "", ""),
 "site_center_z": ("z of the patch centroid, angstrom, in the input file's frame.", "", "", ""),
 "site_box_A": ("Edge length of a cubic box enclosing the patch, angstrom; the design or "
   "docking box for this site.",
   "A large, spread-out site.", "A tight, compact site.", "context"),
 "dg_est_kcal_mol": ("Estimated interface free energy from the area a VHH paratope would "
   "bury, split into apolar and polar terms, plus a penalty for a flexible epitope. An "
   "estimate from geometry, not a prediction for any particular designed binder.",
   "Near zero: little burial available, weak binding expected.",
   "Strongly negative: a large, well-formed interface is available.", "lower_is_better"),
 "kd_est_M": ("Dissociation constant implied by dg_est at 310.15 K, in molar.",
   "Weak: a high Kd means little binding.", "Tight binding implied.", "lower_is_better"),
 "pkd_est": ("-log10(kd_est_M); 9 is nanomolar, 6 is micromolar.",
   "Tight binding implied by the available interface.",
   "Only weak binding is geometrically available here.", "higher_is_better"),
 "buried_area_est_A2": ("Antigen-side area a paratope could bury, capped at --paratope-area.",
   "A large contact surface is available.",
   "Little surface to bind; affinity will be hard to reach.", "higher_is_better"),
 "apolar_index": ("Exposed apolar fraction of the patch mapped onto a logP-like 0-5 scale.",
   "A greasy patch: sticky and prone to non-specific binding.",
   "A polar, more selectively recognisable patch.", "lower_is_better"),
 "epitope_le": ("Ligand-efficiency analogue: -dg_est per epitope residue, kcal/mol/residue.",
   "Every residue of the epitope earns its place.",
   "A large epitope contributing little binding per residue.", "higher_is_better"),
 "epitope_lle": ("Lipophilic-efficiency analogue: pkd_est minus apolar_index. Discounts "
   "affinity that is bought with greasy surface rather than specific contacts.",
   "Affinity comes from specific, polar contacts: the selective kind.",
   "Affinity is mostly hydrophobic stickiness; expect poor specificity.", "higher_is_better"),
 "epitope_fq": ("Fit-quality analogue: epitope_le over the efficiency expected for an epitope "
   "of this size, fitted across the whole run. 1.0 is exactly typical.",
   "More efficient than other epitopes of the same size.",
   "Less efficient than its size would predict.", "higher_is_better"),
 "epitope_complexity_index": ("Structural-complexity analogue combining patch size, residue "
   "variety and sequence discontinuity, 0-1.",
   "A large, chemically varied, conformational epitope: distinctive but harder to mimic.",
   "A small, uniform, largely linear epitope: simple but less distinctive.", "context"),
 "target_residues": ("The epitope as chain-qualified residue coordinates, "
   "<chain>:<resseq>[<icode>] separated by commas and ordered by residue number, e.g. "
   "A:42,A:43,A:44. This is the unambiguous machine-readable identifier for the site: unlike "
   "epitope_residues it names the chain, so it stays correct for multi-chain inputs and can "
   "be pasted straight into a selection or design specification.", "", "", ""),
 "epitope_residues": ("Epitope residues as <aa><resseq>, in sequence order. Convenient to "
   "read; use target_residues when the chain matters.", "", "", ""),
 "rfdiffusion_hotspots": ("Five most exposed epitope residues, formatted for the RFdiffusion "
   "ppi.hotspot_res argument.", "", "", ""),
 "epitope_sequence": ("One-letter sequence of the epitope residues in sequence order.",
   "", "", ""),
 "crosstalk_max": ("Raw MaxSim similarity to the most similar patch on a DIFFERENT antigen. "
   "Absolute scale depends on --embed, so compare within a run.",
   "A patch on another antigen looks very much like this one: cross-binding risk.",
   "Nothing on the other antigens resembles it.", "lower_is_better"),
 "crosstalk_partner": ("The epitope_id on another antigen that this one most resembles.",
   "", "", ""),
 "crosstalk_top3": ("Mean similarity to the three most similar patches on other antigens.",
   "Resembles several other antigens, not just one.",
   "Distinctive against the whole collection.", "lower_is_better"),
 "crosstalk_percentile": ("Rank of crosstalk_max among all candidates in the run, 0-1.",
   "Among the worst cross-talk in the run.",
   "Among the most distinctive epitopes in the run.", "lower_is_better"),
 "crosstalk_within_antigen": ("Similarity to the most similar patch on the SAME antigen.",
   "The antigen presents this surface twice; one nanobody may bind two sites on it.",
   "The epitope is unique on its own antigen.", "lower_is_better"),
 "crosstalk_z": ("Robust z-score of crosstalk_max against all cross-antigen pairs in the run.",
   "Far above the typical cross-antigen similarity: a real outlier pairing.",
   "At or below the background level of similarity.", "lower_is_better"),
 "panel_selected": ("1 if the panel optimiser picked this epitope as its antigen's "
   "representative in the mutually orthogonal set.",
   "1: this is the recommended epitope for this antigen in a multiplexed panel.",
   "0: not chosen; another epitope on this antigen separates better from the rest.",
   "higher_is_better"),
 "panel_worst_crosstalk": ("For a selected epitope, its worst similarity to another panel "
   "member. Blank for unselected rows.",
   "Even the chosen panel still has a confusable pair here.",
   "The chosen panel is well separated at this epitope.", "lower_is_better"),
 "panel_worst_partner": ("The panel member this one is most confusable with.", "", "", ""),
 "crosstalk_cluster": ("Average-linkage cluster of the cross-talk matrix. Epitopes sharing a "
   "cluster present similar surfaces, so take at most one nanobody per cluster.", "", "", ""),
 "cluster_size": ("Number of epitopes in this cluster.",
   "Sits in a crowded cluster of look-alike surfaces.",
   "A singleton cluster: a distinctive surface.", "lower_is_better"),
 "cluster_n_antigens": ("Distinct antigens represented in this cluster.",
   "The look-alike surface recurs on several antigens: real multiplexing risk.",
   "Confined to one antigen.", "lower_is_better"),
 "pc1": ("First principal component of the standardised epitope descriptor matrix; the "
   "dominant axis of variation across every epitope in the run.",
   "At one extreme of the main descriptor axis, typically the large, buried, high-affinity "
   "end.",
   "At the opposite extreme, typically the small, exposed, low-affinity end.", "context"),
 "pc2": ("Second principal component of the epitope descriptor matrix.",
   "One extreme of the second axis of variation; inspect the loadings to interpret it.",
   "The opposite extreme of that axis.", "context"),
 "pc3": ("Third principal component of the epitope descriptor matrix.",
   "One extreme of the third axis of variation.", "The opposite extreme.", "context"),
 "proteome_offtarget_max": ("Best MaxSim against the prebuilt PocketScope human pocketome, "
   "when --index is used. Blank otherwise.",
   "Something in the human proteome presents a similar constellation: investigate.",
   "No close match in the indexed pocketome.", "lower_is_better"),
 "proteome_offtarget_hit": ("Accession and pocket id of that best proteome match.", "", "", ""),
 "patch_area_A2": ("Summed exposed area of the epitope residues, angstrom^2.",
   "A large surface; more than one paratope can cover at once.",
   "Too small to support a high-affinity interface.", "context"),
 "mean_rel_sasa": ("Mean relative solvent accessibility of the epitope residues, 0-1.",
   "Fully solvent exposed and reachable.",
   "Partly buried; a paratope may not reach it.", "higher_is_better"),
 "protrusion": ("Distance of the patch centroid from the protein centroid, in radii of "
   "gyration.",
   "Sits on a protruding feature, sterically easy to approach.",
   "Sits in a recessed region of the surface.", "context"),
 "concavity": ("Concavity of the patch, 0-1, from ray-cast buriedness.",
   "A cleft or groove; the site type nanobody CDR3 loops reach into.",
   "A flat or convex surface, better suited to a broad flat paratope.", "context"),
 "planarity_A": ("RMS spread of patch residues along their least-varying axis, angstrom.",
   "A rough, three-dimensional surface with real topography.",
   "A flat surface offering little shape complementarity.", "context"),
 "seq_contiguity": ("Fraction of the epitope in its largest contiguous sequence run, 0-1.",
   "Close to a linear epitope: easier to transplant as a peptide.",
   "A conformational epitope needing the folded domain.", "context"),
 "rigidity": ("Conformational rigidity from pLDDT or B-factors, 0-1.",
   "Well ordered: a stable target for design.",
   "Mobile or disordered: an unreliable epitope.", "higher_is_better"),
 "mean_bfactor_or_plddt": ("Mean B-factor, or mean pLDDT when the input is a predicted "
   "model. The two scales run in opposite directions, which is why `rigidity` exists.",
   "For a pLDDT column, confidently predicted and rigid. For a B-factor column, mobile.",
   "For a pLDDT column, poorly predicted and likely disordered. For a B-factor column, "
   "well ordered.", "context"),
 "net_charge": ("Formal net charge of the epitope residues at neutral pH.",
   "Strongly basic patch; a binder will need acidic complementarity.",
   "Strongly acidic patch.", "context"),
 "hydrophobic_frac": ("Fraction of epitope residues with apolar side chains.",
   "Greasy epitope: sticky and often non-specific.",
   "Polar epitope: usually more selectively recognisable.", "context"),
 "interface_buried_A2": ("Epitope area buried by another chain in the deposited assembly.",
   "The epitope is occluded in the assembly and may not exist on a PURE monomer.",
   "Fully available on the isolated chain.", "lower_is_better"),
 "cofactor_contact": ("Non-water heteroatom group contacting the epitope, or '-'.", "", "", ""),
 "construct_range": ("Chain and residue range of the minimal construct carrying the epitope.",
   "", "", ""),
 "construct_len": ("Length of that construct in residues.",
   "A long construct; PURE yield and folding both fall away with size.",
   "A short construct, but too short may not fold autonomously.", "context"),
 "construct_mw_da": ("Molecular weight of the construct, daltons.",
   "Heavy; approaching or past the practical PURE ceiling.",
   "Small and well within PURE's comfortable range.", "lower_is_better"),
 "construct_pI": ("Isoelectric point of the construct.",
   "Basic construct.", "Acidic construct. Either is fine; a pI near the working pH is not.",
   "context"),
 "construct_gravy": ("Grand average of hydropathy over the construct.",
   "Hydrophobic overall: aggregation risk in a chaperone-poor mix.",
   "Hydrophilic overall: generally better behaved.", "lower_is_better"),
 "construct_disorder_frac": ("Fraction of the construct below pLDDT 70. 0 when the input is "
   "not a predicted model.",
   "Largely disordered: will not fold to a defined target.",
   "Well ordered throughout.", "lower_is_better"),
 "dangling_contact_frac": ("Fraction of the construct's residue contacts severed by cutting "
   "it out of the parent chain.",
   "Excision shreds the domain's core; the fragment will not fold alone.",
   "The fragment is a self-contained structural unit.", "lower_is_better"),
 "n_cys": ("Cysteines in the construct.",
   "Many cysteines: misoxidation and aggregation risk in PURE.",
   "Few or none: no redox liability.", "lower_is_better"),
 "ss_internal": ("Disulfide bonds fully inside the construct.",
   "The fold depends on disulfides that a reducing PURE reaction will not form.",
   "No disulfide dependence.", "lower_is_better"),
 "ss_broken_by_excision": ("Disulfides with exactly one partner inside the construct.",
   "Excision leaves unpaired cysteines from a broken disulfide: the worst redox case.",
   "No disulfide is cut by the construct boundary.", "lower_is_better"),
 "nglyc_sequons_construct": ("N-X-S/T sequons in the construct.",
   "Many native glycosylation sites; the E. coli product differs from the native antigen.",
   "Few or none.", "lower_is_better"),
 "nglyc_sequons_in_epitope": ("N-X-S/T sequons inside the epitope itself.",
   "The epitope is glycan-dependent natively and will present differently from PURE.",
   "The epitope surface is glycan-free.", "lower_is_better"),
 "max_hydropathy_w19": ("Highest mean Kyte-Doolittle hydropathy over any 19-residue window.",
   "A transmembrane-like segment: aggregates without a membrane.",
   "No hydrophobic stretch long enough to cause trouble.", "lower_is_better"),
 "exposed_hydrophobic_frac": ("Fraction of the construct's exposed area from apolar residues.",
   "A greasy exterior: the main aggregation driver in PURE.",
   "A well-behaved polar exterior.", "lower_is_better"),
 "rule_size_pass": ("PURE rule 1, size: 45-300 residues and at most 35 kDa.",
   "1: the construct is inside the size window PURE handles well.",
   "0: too short to fold alone, or heavy enough that PURE yield falls away.",
   "higher_is_better"),
 "rule_redox_pass": ("PURE rule 2, redox: no disulfide required by the fold, and none severed "
   "by the construct boundary.",
   "1: nothing depends on oxidation, so a reducing PURE reaction is fine.",
   "0: the fold needs a disulfide PURE will not form, or excision leaves an unpaired "
   "cysteine.", "higher_is_better"),
 "rule_glyco_pass": ("PURE rule 3, glycosylation: no N-X-S/T sequon inside the epitope.",
   "1: the epitope surface is the same glycosylated or not.",
   "0: the native epitope is glycan-modified, so the E. coli product presents a different "
   "surface there.", "higher_is_better"),
 "rule_membrane_pass": ("PURE rule 4, membrane: no 19-residue window above 1.6 mean "
   "Kyte-Doolittle hydropathy.",
   "1: no transmembrane-like stretch.",
   "0: a membrane-like segment that will aggregate with no membrane to insert into.",
   "higher_is_better"),
 "rule_domain_pass": ("PURE rule 5, domain: at most 30 % of the construct's residue contacts "
   "are severed by cutting it out.",
   "1: the fragment is a self-contained structural unit.",
   "0: excision shreds the hydrophobic core; the fragment will not fold alone.",
   "higher_is_better"),
 "rule_monomer_pass": ("PURE rule 6, monomer: less than 200 A^2 of the epitope is buried by "
   "another chain in the deposited assembly.",
   "1: the epitope exists on the isolated chain PURE will make.",
   "0: a quaternary epitope that may not exist on a monomer.", "higher_is_better"),
 "rule_order_pass": ("PURE rule 7, order: at most 20 % of the construct below pLDDT 70.",
   "1: well ordered throughout.",
   "0: substantially disordered, so neither expressible nor a stable design target.",
   "higher_is_better"),
 "construct_sequence": ("Amino-acid sequence of the proposed PURE construct.", "", "", ""),
}

_SUBSCORE_DOCS = {
 "epi_area": "exposed-area term, rising monotonically with patch area",
 "epi_topography": "surface-topography term, rewarding three-dimensional relief",
 "epi_exposure": "solvent-exposure term",
 "epi_cleft": "cleft-character term, rewarding concave sites",

 "epi_rigidity": "rigidity term from pLDDT or B-factors",
 "epi_polarity": "composition term, penalising all-hydrophobic patches",
 "pure_length": "construct-size term against the PURE ceiling",
 "pure_redox": "disulfide term for a reducing PURE reaction",
 "pure_glycan": "N-glycosylation term",
 "pure_membrane": "transmembrane-segment term",
 "pure_excisable": "term for how cleanly the construct cuts out of its parent",
 "pure_aggregation": "exposed-hydrophobic-surface term",
 "pure_order": "disorder term",
 "pure_monomeric": "term for epitopes that only exist in an oligomer",
 "pure_cofactor_free": "cofactor and glycan dependence term",
 "pure_solubility": "term for constructs whose pI sits on the working pH",
}


def _yaml_quote(text: str) -> str:
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_metric_yaml(path: Path, args, n_antigens: int, n_epitopes: int,
                      backend: str, cut: float) -> None:
    """Write the column dictionary: what each metric is, and what high and low mean."""
    numeric_hint = ("rank", "score", "pass", "frac", "_A2", "_len", "_A", "n_", "pc",
                    "count", "cys", "charge", "pkd", "dg_", "kd_", "le", "lle", "fq",
                    "index", "size", "cluster", "pI", "gravy", "mw", "buriedness",
                    "percentile", "_z", "crosstalk_max", "crosstalk_top3", "rigidity",
                    "concavity", "protrusion", "planarity", "contiguity", "sasa",
                    "hydropathy", "center", "box", "offtarget_max")
    lines = [
        "# EpitopeScope metric dictionary",
        f"# generated by epitopescope {__version__}",
        "#",
        "# One entry per column of the <antigen>.<tag>.csv files. `high` and `low` say what an",
        "# extreme value means in practice; `direction` says which end is desirable, or",
        "# `context` when neither end is inherently good.",
        "run:",
        f"  epitopescope_version: {__version__}",
        f"  embedding_backend: {backend}",
        f"  surface_mode: {args.surface}",
        f"  n_antigens: {n_antigens}",
        f"  n_epitopes: {n_epitopes}",
        f"  crosstalk_cluster_cut: {cut:.4f}",
        f"  temperature_K: {T_PHYS}",
        f"  weights: {{epitope: {args.w_epitope}, orthogonality: {args.w_ortho}, "
        f"pure: {args.w_pure}}}",
        "columns:",
    ]
    for name in FIELDS:
        doc = METRIC_DOCS.get(name)
        if doc is None and name in _SUBSCORE_DOCS:
            which = "epitope" if name.startswith("epi_") else "PURE"
            doc = (f"Sub-score in [0,1]: the {_SUBSCORE_DOCS[name]}. One factor of the "
                   f"{which} axis geometric mean.",
                   "This factor is not a problem for this candidate.",
                   "This factor is what is dragging the candidate down.", "higher_is_better")
        if doc is None:
            doc = (name.replace("_", " ").capitalize(), "", "", "")
        desc, hi, lo, direction = doc
        is_num = any(h in name for h in numeric_hint) and name not in (
            "site_source", "pure_zone", "flags", "epitope_id", "antigen", "chain",
            "crosstalk_partner", "panel_worst_partner", "proteome_offtarget_hit",
            "construct_range", "construct_sequence", "epitope_residues",
            "target_residues",
            "epitope_sequence", "rfdiffusion_hotspots", "cofactor_contact")
        lines.append(f"  - name: {name}")
        lines.append(f"    description: {_yaml_quote(desc)}")
        if hi:
            lines.append(f"    high: {_yaml_quote(hi)}")
        if lo:
            lines.append(f"    low: {_yaml_quote(lo)}")
        if direction:
            lines.append(f"    direction: {direction}")
        lines.append(f"    colour: {'RdBu' if is_num else 'Spectral'}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------

def analyse(st: Structure, args, verbose: bool):
    vlog(f"[{st.name}] {len(st.residues)} residues, {len(st.chains)} chain(s), "
         f"{'pLDDT' if st.is_plddt else 'B-factor'} column", verbose)
    compute_sasa(st, args.probe, args.sasa_points)
    cmap = residue_contacts(st, args.contact_cutoff)
    ss = find_disulfides(st)
    cof = cofactor_contacts(st)
    if ss:
        vlog(f"[{st.name}] {len(ss)} disulfide(s)", verbose)
    patches = enumerate_patches(st, args, cmap, st.is_plddt)
    for p in patches:
        score_epitope(p, args)
        score_pure(p, st, cmap, ss, cof, args, st.is_plddt)
        score_thermo(p, st, args)
        apply_rules(p, args)
    vlog(f"[{st.name}] {len(patches)} candidate epitope patch(es)", verbose)
    return patches


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="epitopescope",
        description="Rank antigen surface epitopes for orthogonal, PURE-expressible nanobody design.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-d", "--inputdir", required=True,
                   help="directory holding the antigen PDB files")
    p.add_argument("--tag", default="tool",
                   help="output files are named <input stem>.<tag>.csv")
    p.add_argument("--outdir", default=None,
                   help="where to write the CSVs (default: alongside the input files)")
    p.add_argument("--glob", default="*.pdb", help="which files in --inputdir to read")
    p.add_argument("--verbose", action="store_true", help="more detail on stderr")
    p.add_argument("--refresh", action="store_true",
                   help="recompute even when a non-empty output file already exists")

    g = p.add_argument_group("epitope patches")
    g.add_argument("--surface", choices=["monomer", "complex"], default="monomer",
                   help="define the antigen surface on the isolated chain (what PURE makes) or "
                        "on the deposited assembly")
    g.add_argument("--min-rsasa", type=float, default=0.20,
                   help="minimum relative SASA for a residue to count as surface")
    # Real nanobody epitopes in datasets/nanobody_* span a median of 26.7 A, so a 10 A
    # radius could not physically cover one. Widening to 12 A lifts how much of the true
    # epitope the best candidate captures on both benchmark sets (0.42 -> 0.49 held out,
    # 0.46 -> 0.56 on calibration). Its effect on recall@1 is within noise, so this is a
    # coverage fix, not a ranking one. 10.0 reproduces pre-0.3.1 behaviour.
    g.add_argument("--patch-radius", type=float, default=12.0,
                   help="CB radius, in angstrom, around a seed residue")
    g.add_argument("--patch-min", type=int, default=6, help="minimum residues in a patch")
    g.add_argument("--patch-max", type=int, default=22, help="maximum residues in a patch")
    g.add_argument("--concavity-cut", type=float, default=0.55,
                   help="buriedness above which a surface residue seeds a concavity cluster")
    g.add_argument("--concavity-rays", type=int, default=42,
                   help="rays cast per residue when measuring concavity")
    g.add_argument("--concavity-reach", type=float, default=11.0,
                   help="how far, in angstrom, a concavity ray looks for a blocking atom")
    g.add_argument("--cluster-link", type=float, default=8.0,
                   help="single-linkage distance joining concave residues into one site")
    g.add_argument("--max-overlap", type=float, default=0.40,
                   help="Jaccard overlap above which two patches are treated as redundant")
    g.add_argument("--max-patches", type=int, default=12,
                   help="ranked candidate epitopes kept per antigen")
    g.add_argument("--emit-redundant", type=int, default=6,
                   help="also emit up to this many suppressed overlapping candidates per "
                        "antigen, labelled status=redundant with a redundant_with pointer; "
                        "0 discards them as before v0.3.0")
    g.add_argument("--probe", type=float, default=1.4, help="SASA probe radius")
    g.add_argument("--sasa-points", type=int, default=92, help="Shrake-Rupley sphere points")
    g.add_argument("--contact-cutoff", type=float, default=5.0,
                   help="heavy-atom distance defining a residue contact")
    g.add_argument("--bfactor", choices=["auto", "plddt", "bfactor"], default="auto",
                   help="how to read the B-factor column: AlphaFold pLDDT (higher is better) "
                        "or crystallographic B-factors (lower is better)")

    g = p.add_argument_group("PURE construct")
    g.add_argument("--max-construct", type=int, default=280,
                   help="longest fragment considered expressible in PURE")
    g.add_argument("--grow-gain", type=int, default=1,
                   help="minimum contacts closed per added residue while growing a construct")
    g.add_argument("--allow-disulfides", action="store_true",
                   help="do not penalise internal disulfides (PURExpress minus DTT, plus DsbC)")

    g = p.add_argument_group("representation and cross-talk")
    g.add_argument("--embed", choices=["auto", "none", "esm2", "esmc"], default="auto",
                   help="per-residue representation: none = 13 physicochemical channels only; "
                        "esm2 = ESM-2 650M via transformers; esmc = ESM-C 600M, the PocketScope "
                        "model, required for --index")
    g.add_argument("--device", default="cuda", help="torch device for the language model")
    g.add_argument("--esm2-model", default=Embedder.ESM2_DEFAULT,
                   help="HuggingFace model id used by --embed esm2")
    g.add_argument("--alpha", type=float, default=ALPHA,
                   help="weight of the physicochemical block in Equation 1")
    g.add_argument("--index", default=None,
                   help="PocketScope pocketome index directory, for proteome-wide off-target "
                        "screening (needs --embed esmc)")
    g.add_argument("--index-device", default="cuda", help="torch device for the index")
    g.add_argument("--index-top-k", type=int, default=5, help="off-target hits to consider")

    g = p.add_argument_group("thermodynamics and efficiency")
    g.add_argument("--paratope-area", type=float, default=850.0,
                   help="largest antigen-side area, in angstrom^2, a VHH paratope can bury")
    g.add_argument("--dg-flex", type=float, default=2.0,
                   help="free-energy penalty, kcal/mol, for a fully flexible epitope")
    g.add_argument("--crosstalk-direction", choices=["max", "mean", "min"], default="max",
                   help="how to combine the two asymmetric MaxSim readings of a patch pair. "
                        "max is the conservative corner and the default from v0.3.0; mean "
                        "reproduces the earlier symmetric average")
    g.add_argument("--specificity-ratio", type=float, default=0.8,
                   help="minimum margin ratio for a candidate to be status=ranked: how far "
                        "its nearest look-alike sits, in units of how far a typical "
                        "epitope's nearest look-alike sits. 1.0 is exactly typical by "
                        "construction, so the default 0.8 rejects only the clearly "
                        "worse-than-average sites. Raise it for a stricter panel; set 0 to "
                        "report without gating")
    g.add_argument("--top-partners", type=int, default=3,
                   help="distinct off-target antigens recorded per epitope")
    g.add_argument("--cluster-cut", default="auto",
                   help="cross-talk similarity joining two epitopes into one cluster; "
                        "'auto' uses the 95th percentile of this run's cross-antigen "
                        "similarities, which keeps the cut comparable across --embed backends")

    g = p.add_argument_group("ranking")
    # Odin-Multi gates on the specificity ratio and then ranks the qualifying designs by
    # target quality alone, not by specificity again. Keeping orthogonality at 1.5 in the
    # composite double-counted it against its own gate. Re-weighting to lead with epitope
    # quality raises held-out recall@1 from 30 % to 40 % over 30 unseen nanobody complexes
    # (and 47 % to 70 % on the calibration set) with detection unchanged. Orthogonality keeps
    # a small weight so it still breaks ties among qualified sites.
    g.add_argument("--w-epitope", type=float, default=2.0, help="weight of the epitope axis")
    g.add_argument("--w-ortho", type=float, default=0.5, help="weight of the orthogonality axis")
    g.add_argument("--w-pure", type=float, default=1.5, help="weight of the PURE axis")
    g.add_argument("--top", type=int, default=0, help="rows written per antigen (0 = all)")

    g = p.add_argument_group("panel selection")
    g.add_argument("--no-panel", action="store_true",
                   help="skip choosing one mutually orthogonal epitope per antigen")
    g.add_argument("--panel-pool", type=int, default=6,
                   help="candidate epitopes per antigen offered to the panel optimiser")
    g.add_argument("--panel-min-epitope", type=float, default=0.45,
                   help="minimum epitope score for a panel candidate")
    g.add_argument("--panel-min-pure", type=float, default=0.45,
                   help="minimum PURE score for a panel candidate")
    g.add_argument("--panel-quality-weight", type=float, default=0.15,
                   help="how much panel quality is traded against worst-case cross-talk")
    g.add_argument("--panel-restarts", type=int, default=24,
                   help="random restarts for the panel optimiser")
    g = p.add_argument_group("Odin-Multi export")
    g.add_argument("--emit-odin", default=None, metavar="DIR",
                   help="write Odin-Multi settings_target JSONs and design command lines for "
                        "the selected panel into DIR")
    g.add_argument("--odin-hotspot-mode", choices=["core", "full"], default="core",
                   help="hotspots per target: the most exposed --odin-hotspot-n residues, or "
                        "the whole epitope (which tends to over-constrain the design)")
    g.add_argument("--odin-hotspot-n", type=int, default=6,
                   help="core hotspot count")
    g.add_argument("--odin-offtargets", type=int, default=2,
                   help="off-target contexts per design run")
    g.add_argument("--odin-top", type=int, default=1,
                   help="design runs per antigen: the panel epitope plus this many next-best "
                        "ranked sites. Benchmarks rank the validated epitope first only about "
                        "half the time, so 3-6 is a realistic campaign")

    p.add_argument("--check-env", action="store_true",
                   help="report the environment and exit, including whether torch can "
                        "actually see the GPU")
    p.add_argument("--no-yaml", action="store_true",
                   help="do not write the epitopescope.<tag>.yaml metric dictionary")
    p.add_argument("--version", action="version", version=f"epitopescope {__version__}")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    if "--check-env" in argv and not any(a in ("-d", "--inputdir") for a in argv):
        argv = list(argv) + ["-d", "."]
    args = parser.parse_args(argv)
    if args.check_env:
        return check_environment(args)
    v = args.verbose

    indir = Path(args.inputdir).expanduser().resolve()
    if not indir.is_dir():
        log(f"error: --inputdir {indir} is not a directory")
        return 2
    inputs = sorted(q for q in indir.glob(args.glob) if q.is_file())
    if not inputs:
        log(f"error: no files matching {args.glob!r} in {indir}")
        return 2
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    out_of = {q: ((outdir or q.parent) / f"{q.stem}.{args.tag}.csv").resolve() for q in inputs}
    todo = [q for q in inputs
            if args.refresh or not (out_of[q].exists() and out_of[q].stat().st_size > 0)]

    log(f"epitopescope {__version__}: {len(inputs)} antigen file(s) in {indir}, "
        f"{len(todo)} to compute")
    if not todo:
        vlog("all outputs present and non-empty; use --refresh to recompute", v)
        for q in inputs:
            print(out_of[q])
        ypath = (outdir or inputs[0].parent) / f"epitopescope.{args.tag}.yaml"
        if ypath.exists() and ypath.stat().st_size > 0 and not args.no_yaml:
            print(ypath.resolve())
        return 0

    # Cross-talk is defined against the whole collection, so every antigen is analysed even
    # when only some outputs are being written.
    structures, all_patches = {}, []
    for q in inputs:
        try:
            st = parse_pdb(q, args.bfactor)
        except Exception as exc:
            log(f"warning: skipping {q.name}: {exc}")
            continue
        structures[q] = st
        all_patches.extend(analyse(st, args, v))
    if not structures:
        log("error: no parsable structures")
        return 1
    if not all_patches:
        log("error: no candidate epitope patches found; try lowering --min-rsasa")
        return 1

    backend = Embedder.resolve(args.embed, v)
    need = {"esm2": ["torch", "transformers"], "esmc": ["torch", "esm"]}.get(backend, [])
    missing = []
    for mod in need:
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        log(f"error: --embed {backend} needs {', '.join(missing)}. "
            f"Install them (see envs/requirements-plm.txt) or run with --embed none.")
        return 1
    if args.index and backend != "esmc":
        log(f"note: --index needs --embed esmc; backend resolved to {backend!r}, "
            f"proteome screening disabled")
    vlog(f"representation: {backend} + {len(residue_phys('A')[0])} physicochemical channels", v)
    if backend == "none":
        log("note: --embed none compares patches on 13 physicochemical channels only. That "
            "signal is weak and saturates: cross-talk ranking is only meaningful relative to "
            "the other candidates in this run. Use --embed esm2 or --embed esmc for a "
            "sequence-aware comparison.")
    emb = Embedder(backend, args.device, v, args.esm2_model)

    for q, st in structures.items():
        used = {p.chain for p in all_patches if p.antigen == st.path.stem}
        chain_emb = {}
        for ch in sorted(used):
            seq = st.chain_seq(ch)
            vlog(f"[{st.name}] embedding chain {ch} ({len(seq)} aa)", v)
            chain_emb[ch] = emb.embed(seq)
        for p in all_patches:
            if p.antigen == st.path.stem:
                p.feats = patch_features(p, st, chain_emb, args.alpha)
                p.w = patch_weights(p, st, args.surface)

    vlog(f"cross-talk: {len(all_patches)} patches over "
         f"{len({p.antigen for p in all_patches})} antigens", v)
    S = crosstalk(all_patches, v, args.crosstalk_direction, args.top_partners)

    baseline = fit_efficiency_baseline(all_patches, v)
    apply_fit_quality(all_patches, baseline)
    assign_status(all_patches, args, v)
    cut = resolve_cluster_cut(all_patches, S, args.cluster_cut)
    cluster_epitopes(all_patches, S, cut, v)
    pca_epitopes(all_patches, v)

    if args.index and backend == "esmc":
        try:
            proteome_screen(all_patches, args.index, args.index_device, args.index_top_k, v)
        except Exception as exc:
            log(f"warning: proteome screen failed: {exc}")

    w = np.array([args.w_epitope, args.w_ortho, args.w_pure], np.float64)
    for p in all_patches:
        vals = np.array([max(p.s_epitope, 1e-3), max(p.s_ortho, 1e-3), max(p.s_pure, 1e-3)])
        p.total = float(np.exp((w * np.log(vals)).sum() / w.sum()))

    if not args.no_panel:
        select_panel(all_patches, S, args, v)

    written = []
    for q in inputs:
        st = structures.get(q)
        if st is None:
            continue
        if q not in todo:
            vlog(f"[{q.name}] up to date", v)
            written.append(out_of[q])
            continue
        ps = sorted((p for p in all_patches if p.antigen == st.path.stem),
                    key=lambda p: -p.total)
        if args.top > 0:
            ps = ps[:args.top]
        write_csv(out_of[q], ps, st, args.surface)
        if ps:
            b = ps[0]
            vlog(f"[{q.name}] {len(ps)} epitopes; best {b.pid} score={b.total:.3f} "
                 f"(epitope {b.s_epitope:.2f} / ortho {b.s_ortho:.2f} / pure {b.s_pure:.2f}) "
                 f"construct {b.con_range} {len(b.con_seq)} aa", v)
        written.append(out_of[q])

    if args.emit_odin:
        od = write_odin_export(Path(args.emit_odin).expanduser().resolve(),
                               all_patches, structures, args)
        vlog(f"Odin-Multi export written to {od}", v)
        written.append((od / "odin_design_commands.sh").resolve())

    if not args.no_yaml:
        ypath = (outdir or inputs[0].parent) / f"epitopescope.{args.tag}.yaml"
        write_metric_yaml(ypath, args, len(structures), len(all_patches), backend, cut)
        written.append(ypath.resolve())

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
