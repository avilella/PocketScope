# EpitopeScope

**v0.2.0** - revised against *An Open-Source Reproducible Workflow for Pocket-Oriented Virtual
Screening and ADME-Integrated Chemoinformatics* (bioRxiv 2026.04.28.721199). What that
workflow does for compounds, EpitopeScope now does for epitopes: sites are found by concavity
clustering rather than picked by hand, scores are converted into interpretable affinity units,
efficiency metrics divide out size, named rules replace opaque penalties, and the collection is
analysed multivariately. See [What changed in v0.2.0](#what-changed-in-v020).

Given a folder of antigen PDB files, rank the surface epitopes you should target with
*in silico* designed nanobodies, so that

1. each nanobody has a good epitope to bind,
2. the panel has **minimal cross-talk** — no nanobody is likely to also stick to another
   antigen in the same tube, and
3. the antigen fragment carrying that epitope can realistically be **made and folded by an
   E. coli PURE in-vitro transcription/translation reaction**.

One CSV of ranked candidate epitopes is written per input PDB. The full paths of those CSVs are
the only thing the script writes to stdout; everything else goes to stderr.

EpitopeScope reuses the PocketScope representation in this repository — per-residue protein
language model embeddings concatenated with 13 physicochemical channels, L2-normalised
(Equation 1 in `pocketscope/features.py`), compared by exact late-interaction MaxSim — but
applies it to *solvent-exposed surface patches* rather than P2Rank cavities, because that is
what a nanobody actually sees.

---

## What changed in v0.2.0

The preprint's discipline, mapped onto epitopes:

| preprint | EpitopeScope v0.2.0 |
|---|---|
| Concavity detects surface depressions from residue centroids; residues are clustered by spatial proximity and each cluster's Ca centroid becomes the docking box centre (§4.4) | `residue_buriedness()` casts rays into the outward hemisphere of every surface residue; high-concavity residues are single-linkage clustered into sites. `site_source`, `site_buriedness`, `site_center_x/y/z`, `site_box_A`. Exposed-surface seeding is kept in parallel so flat and convex epitopes are not lost |
| ΔG from docking is converted to Kd and pKd at 310.15 K (§5.6, §6.4) | `dg_est_kcal_mol` from the area a paratope would bury, split apolar/polar, plus a flexibility penalty, then `kd_est_M` and `pkd_est` at the same fixed temperature |
| LE, LLE and FQ stop a big ligand winning on raw score (§7.3.2) | `epitope_le` (ΔG per epitope residue), `epitope_lle` (pKd minus a logP-like apolar index, which exposes the greasy-binder failure mode the preprint calls out), `epitope_fq` (efficiency over the efficiency expected for that epitope size, fitted across the run) |
| Named drug-likeness rules and BOILED-Egg regions, checked independently (§8.2, §8.4) | Seven named PURE rules (`rule_size_pass` … `rule_order_pass`), `n_pure_rules_passed`, and a `pure_zone` of GREEN / YELLOW / RED |
| Structural Complexity Index combining size, heteroatom richness and rigidity (§9.9) | `epitope_complexity_index`: patch size, residue variety and sequence discontinuity |
| PCA and hierarchical clustering of the affinity matrix (§9.7, §9.14) | `pc1`-`pc3` from the standardised epitope descriptor matrix, and `crosstalk_cluster` from average-linkage clustering of the cross-talk matrix - epitopes in one cluster present a similar surface, so take at most one nanobody per cluster |
| Retrospective validation with reference compounds, reporting recall@N and enrichment (§4.5) | `benchmark_epitopescope.py` against solved nanobody-antigen complexes, where the observed epitope is ground truth. This found and fixed two wrong-signed priors - see [Validation](#validation) |

Also new: a `epitopescope.<tag>.yaml` metric dictionary written beside the CSVs, giving every
column a description plus what a high and a low value mean.

## Install

Everything lives in one self-contained environment. Pick either.

### conda

```bash
conda env create -f envs/environment.yml
conda activate epitopescope
```

### python -m venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r envs/requirements.txt          # numpy + scipy: enough to run --embed none

# recommended: add a protein language model for a meaningful cross-talk signal.
# install the torch build that matches your machine FIRST
pip install torch --index-url https://download.pytorch.org/whl/cpu     # or .../cu124 for CUDA
pip install -r envs/requirements-plm.txt
```

`esm>=3.0` in the PLM requirements is only needed for `--embed esmc` and `--index`; drop that
line if you do not plan to search the prebuilt PocketScope pocketome.

The script has no install step of its own — run `python epitopescope.py` from the repository
root (it imports `pocketscope.features` from the checkout).

### Which embedding backend

| `--embed` | needs | what it does |
|---|---|---|
| `auto` (default) | — | `esmc` if the ESM-C weights are already cached, else `esm2` if `transformers` is importable, else `none` |
| `none` | numpy | 13 physicochemical channels only. **Weak**: patch similarities saturate near 1.0 and the ranking is only meaningful relative to the other candidates in the same run. Fine for a smoke test, not for a real panel. |
| `esm2` | torch, transformers | ESM-2 650M (`facebook/esm2_t33_650M_UR50D`, ~2.5 GB, cached in `~/.cache/huggingface`). The practical default. |
| `esmc` | torch, esm>=3.0 | ESM-C 600M — the exact model PocketScope uses. The only backend whose 1165-d features match the prebuilt pocketome index, so the only one that works with `--index`. |

---

## Usage

```bash
python epitopescope.py -d antigens/ --embed esm2
```

```
epitopescope 0.1.0: 6 antigen file(s) in /data/antigens, 6 to compute
panel: one epitope per antigen, worst pairwise cross-talk 0.7194
    1DPX|A|E03                 epitope=0.82 pure=0.81 worst-vs-panel=0.7135 vs 1REX|A|E02
    1EMA|A|E10                 epitope=0.82 pure=0.97 worst-vs-panel=0.7194 vs 6M0J|A|E09
    1REX|A|E02                 epitope=0.91 pure=0.79 worst-vs-panel=0.7135 vs 1DPX|A|E03
    1TEN|A|E04                 epitope=0.87 pure=0.91 worst-vs-panel=0.7128 vs 1EMA|A|E10
    1UBQ|A|E02                 epitope=0.87 pure=0.98 worst-vs-panel=0.5980 vs 1TEN|A|E04
    6M0J|A|E09                 epitope=0.62 pure=0.80 worst-vs-panel=0.7194 vs 1EMA|A|E10
```
and on stdout:
```
/data/antigens/1DPX.tool.csv
/data/antigens/1EMA.tool.csv
...
```

### The interface you asked for

| flag | meaning |
|---|---|
| `-d`, `--inputdir` | folder of antigen PDB files (required) |
| `--tag` | output is `<input stem>.<tag>.csv`; default `tool` |
| `--outdir` | where the CSVs go; unset by default, meaning next to the inputs |
| `--verbose` | more processing detail on stderr; off by default |
| `--refresh` | recompute even if the output exists; off by default. When off, an existing non-empty output is reused and its path is simply printed |

`--refresh` is evaluated per file. If every output is already present the script prints the
paths and exits immediately without parsing a single structure.

> **Note on caching.** Cross-talk is a property of the *collection*, not of one antigen. If
> even one output needs computing, every antigen is parsed and scored so the comparison is
> complete — only the stale CSVs are rewritten. Adding a new antigen to the folder therefore
> changes the cross-talk columns of the others, and you should pass `--refresh` after doing so.

Run `python epitopescope.py --help` for the full list. The ones worth knowing:

```
--surface {monomer,complex}   define the surface on the isolated chain (default, what PURE
                              makes) or on the deposited assembly
--min-rsasa 0.20              relative SASA above which a residue counts as surface
--patch-radius 10.0           CB radius of a candidate epitope patch
--patch-max 22                residues per patch
--max-patches 12              candidate epitopes kept per antigen
--max-construct 280           longest fragment the construct grower will reach for
--allow-disulfides            stop penalising disulfides (PURExpress minus DTT, plus DsbC)
--embed {auto,none,esm2,esmc} per-residue representation
--index DIR                   also screen against the PocketScope human pocketome (needs esmc)
--w-epitope/--w-ortho/--w-pure   axis weights, default 1.0 / 1.5 / 1.5
--no-panel                    skip the panel optimiser
```

---

## How the score is built

Every candidate patch gets three sub-scores in `[0, 1]`. The final
`nanobody_target_score` is their weighted geometric mean, so a patch that fails badly on any
one axis cannot be rescued by the other two.

### 1. `epitope_score` — is this a plausible nanobody epitope?

Geometric mean of six factors, **calibrated against 30 solved nanobody–antigen complexes**
(`datasets/nanobody_bench`). Two of the original priors did not survive that check and were
changed:

| factor | what it rewards | benchmark AUC |
|---|---|---|
| `epi_area` | exposed patch area, **monotonically** — a VHH goes for the largest surface it can reach. The original bell curve peaking at 600–900 Å² actively destroyed the signal | 0.64 |
| `epi_topography` | three-dimensional relief (`planarity_A`) — the single most discriminating descriptor. Replaces `epi_protrusion`, which ran the **wrong way**: real epitopes are slightly *less* protruding than the average candidate | 0.68 |
| `epi_rigidity` | order, from pLDDT or B-factors | 0.67 |
| `epi_cleft` | concavity — a real but mild preference, so it is weighted gently rather than treated as a requirement | 0.60 |
| `epi_exposure` | a gate, not a reward: below `--min-rsasa` a patch is unreachable, but among reachable patches more exposure is not better | 0.36 (inverted) |
| `epi_polarity` | penalises all-hydrophobic patches — a specificity and aggregation concern, not an epitope-likeness one | 0.59 |

### 2. `orthogonality_score` — will this nanobody stay off the other antigens?

Each patch becomes an `(L, D)` tensor: per-residue language-model embedding and the 13
PocketScope physicochemical channels, each block L2-normalised, the chemical block scaled by
`--alpha`, the concatenation renormalised. Two patches are compared by symmetric MaxSim, with
each query residue weighted by the surface it actually presents — a residue buried inside the
patch should not drive a cross-talk call.

`crosstalk_max` is the raw similarity to the best-matching patch on a **different** antigen.
Because that is a maximum over many patches it always sits high on an absolute scale, so the
orthogonality axis uses its **rank within the candidate pool**: the patch whose worst
cross-antigen match is the mildest scores 1.0, the worst scores 0. `crosstalk_z` keeps the
absolute reading as a robust z-score against the background of all cross-antigen pairs, and
`crosstalk_partner` names the offending patch so you can look at it.

**Sanity check.** On a test set of hen lysozyme (1DPX), human lysozyme (1REX), GFP (1EMA),
fibronectin III (1TEN), ubiquitin (1UBQ) and the SARS-CoV-2 RBD–ACE2 complex (6M0J), with
`--embed esm2` all 12/12 epitopes of each lysozyme name the *other* lysozyme as their top
cross-talk partner — the two homologues are found without any sequence alignment step. With
`--embed none` the same comparison is essentially random, which is why that backend is only a
fallback.

### 3. `pure_ivtt_score` — can PURE make it?

First the tool works out the **minimal construct**: starting from the patch's sequence span it
greedily extends a contiguous window in that chain for as long as extending closes more
residue–residue contacts than it opens, which stops naturally at domain boundaries. Then that
construct is scored on:

| factor | why |
|---|---|
| `pure_length` | PURE yields fall away above roughly 60 kDa; short compact domains fold best |
| `pure_redox` | PURExpress is a reducing environment. Internal disulfides are penalised, and a disulfide **severed** by the excision is penalised hard. `--allow-disulfides` turns the first penalty off if you run minus-DTT plus DsbC |
| `pure_glycan` | E. coli does not glycosylate. An N-X-S/T sequon *inside the epitope* means the PURE product presents a different surface there than the native antigen |
| `pure_membrane` | a transmembrane-like hydrophobic stretch (max 19-residue Kyte–Doolittle window) aggregates without a membrane |
| `pure_excisable` | fraction of the fragment's contacts left dangling by the cut — how much of its hydrophobic core the excision destroys |
| `pure_aggregation` | exposed hydrophobic surface fraction, the main aggregation driver in a chaperone-poor mix |
| `pure_order` | fraction of the construct below pLDDT 70 |
| `pure_monomeric` | an epitope buried by another chain in the deposited assembly may not exist on a PURE-made monomer |
| `pure_cofactor_free` | PURE supplies no metals, hemes or nucleotides; a modelled glycan at the epitope is flagged separately |
| `pure_solubility` | constructs whose pI sits on top of the working pH are the least soluble |

### 4. Affinity units, efficiency and rules

`dg_est_kcal_mol` comes from the antigen-side area a paratope could bury (capped at
`--paratope-area`), split into apolar and polar terms at 0.0095 and 0.0055 kcal/mol/Å² of
total buried area, plus `--dg-flex` × (1 − rigidity) for ordering a floppy epitope. The
coefficients are set so a typical nanobody interface lands near −12 kcal/mol, i.e. low
nanomolar, and a marginal one lands in the high micromolar range. `kd_est_M` and `pkd_est`
follow at 310.15 K, the temperature the preprint fixes for its efficiency metrics.

**These are estimates from geometry, not predicted affinities of any designed binder.** They
exist so candidates are comparable on an interpretable axis, and so that `epitope_lle` can
expose the failure mode raw scores hide: an epitope whose predicted affinity is bought
entirely with greasy surface scores low, exactly the "highly lipophilic binders with low LLE"
case the preprint flags.

`epitope_fq` divides observed efficiency by the efficiency expected for that epitope size,
with `expected LE = a + b/n` fitted by least squares across the whole run — the Reynolds
size-correction idea, calibrated on the run itself rather than a fixed curve. FQ ≈ 1 is
typical for its size; above 1 is better than its size predicts.

The seven PURE rules are checked independently and reported both individually
(`rule_*_pass`) and as a count, so you can see exactly which liability fired rather than
inferring it from a blended score. `pure_zone` collapses them plus construct length and
exposed hydrophobic surface into GREEN / YELLOW / RED.

### 5. The panel and the clusters

`crosstalk_max` is a per-patch worst case against *all* candidates, which is stricter than what
you need. What you actually want is one epitope per antigen such that the worst pairwise
similarity **within the chosen set** is small. That is a small combinatorial problem, solved by
iterated best-response from several starts over the candidates that clear
`--panel-min-epitope` / `--panel-min-pure`.

The result is printed to stderr and marked in the CSVs by `panel_selected`,
`panel_worst_crosstalk` and `panel_worst_partner`. In the two-kinase example above (CDK2 and
CDK1, ~65 % identical), the top-scoring epitopes on their own have a mutual cross-talk of 0.74,
while the panel optimiser finds a pair at 0.53.

`crosstalk_cluster` complements the panel: average-linkage clustering of the cross-talk
matrix, cut by default at the 95th percentile of the run's own cross-antigen similarities
(`--cluster-cut auto`). A fixed threshold is not portable, because the physicochemical-only
representation saturates near 1.0 while a language model spreads much wider. Epitopes sharing
a cluster present a similar surface to a paratope, so a panel should draw at most one
nanobody per cluster.

### 6. Optional: proteome-wide off-targets

```bash
python epitopescope.py -d antigens/ --embed esmc --index /path/to/pocket_index/
```

adds `proteome_offtarget_max` and `proteome_offtarget_hit` by scoring every patch against the
153,805-cavity PocketScope human pocketome
(https://doi.org/10.5281/zenodo.22178549, 22.9 GB, and a GPU unless you pass
`--index-device cpu`). This answers a wider question than the input collection does: is there
anything in the human proteome that presents a similar residue constellation. The caveat is
that the index holds P2Rank *cavities* while EpitopeScope produces *surface patches*, so read
those two columns as a flag to investigate, not as a calibrated probability.

---

## Benchmark datasets

Two kinds of dataset, both built from public databases by scripts in `datasets/`, both
reproducible from scratch.

### Antigen panels — 10, 48 and 96

```bash
python datasets/build_panels.py            # -> datasets/panel10, panel48, panel96
```

96 human proteins from `datasets/panel_targets.tsv`, each resolved to its best experimental
structure through the PDBe SIFTS `best_structures` API and written out as a **single chain**,
so one file is one antigen. Sizes are plate formats: a 10-antigen pilot, a half plate, a full
plate. The panels are **nested** (10 ⊂ 48 ⊂ 96), so the effect of panel size can be read
directly — the same antigen appears in all three with more competitors around it each time.

The list is chosen the way a real multiplex panel would be, and deliberately loaded with
homologous families so cross-talk has something real to find: 11 four-helix cytokines, 5 TNF
superfamily members, 5 coagulation proteases, 4 S100 proteins, 4 IL-1 family members, 3 each
of serpins, MMPs, cathepsins, CXC and CC chemokines, and paired carbonic anhydrases, FABPs,
galectins, FGFs, globins and CDKs. `datasets/panel_manifest.tsv` records the PDB ID, chain,
resolution and UniProt coverage behind every file.

Wall time on one CPU core set, `--embed esm2 --device cpu`:

| panel | antigens | epitopes | wall | peak RSS | worst pairwise cross-talk in the chosen panel |
|---|---|---|---|---|---|
| panel10 | 10 | 120 | 13 s | 3.6 GB | 0.684 |
| panel48 | 48 | 571 | 49 s | 3.8 GB | 0.870 |
| panel96 | 96 | 1147 | 103 s | 4.0 GB | 0.889 |

That last column is the point of the tool. As the tube gets more crowded, the best available
orthogonal epitope set gets measurably worse — there is simply less unoccupied surface
chemistry to go round.

### Nanobody complexes — retrospective validation

```bash
python datasets/build_benchmark.py                                        # 30 complexes
python datasets/build_benchmark.py --outdir datasets/nanobody_holdout \
                                   --n 30 --skip 30 --search-rows 400     # 30 more, disjoint
```

Solved nanobody–antigen complexes from RCSB, X-ray, better than 2.6 Å. For each, the antigen
chain is written out **without the nanobody**, and the residues the nanobody actually contacts
(heavy atoms within 4.5 Å) are recorded in `true_epitopes.tsv`. Nothing about the answer
reaches the detector.

## Validation

```bash
python benchmark_epitopescope.py --embed esm2 --device cpu
```

Two questions are reported separately, because they fail independently: does the enumerator
**propose** the true epitope at all, and does the scoring **rank** it highly against the null
of ranking the same candidates at random.

| | calibration set (n=30) | **held-out set (n=30, disjoint, no tuning)** |
|---|---|---|
| detection — true epitope proposed | 100 % | **87 %** |
| recall@1 | 47 % (EF 1.93×) | **23 % (EF 1.45×)** |
| recall@3 | 70 % (EF 1.24×) | **50 % (EF 1.26×)** |
| recall@5 | 87 % (EF 1.15×) | **67 % (EF 1.19×)** |
| median rank of first true hit | 2 of 12 (random 4) | **3 of 12 (random 5)** |

Read the held-out column. **Detection is the strong result**: the enumerator puts the real
epitope in a 12-candidate list 87 % of the time on structures it was never tuned against.
**Ranking gives a real but modest lift** — roughly 1.2–1.45× over random. It is not a
sharp epitope predictor and should not be sold as one; it moves the true epitope from a median
rank of 5 to 3.

This benchmark earned its keep by falsifying two priors. Ranking by the original
`epitope_score` was *worse than random* (EF 0.85×) while ranking by raw `patch_area_A2` was
2.54×. The bell curve on area and the reward for protrusion were both wrong; fixing them is
what produced the numbers above. `--rank-by <column>` reruns the sweep against any column.

### Does cross-talk find real homology?

On the panels, using protein family from the manifest as ground truth, with no alignment step
anywhere in the pipeline:

| panel | top cross-talk partner shares family | chance | enrichment |
|---|---|---|---|
| panel10 | 28.3 % | 4.4 % | **6.4×** |
| panel48 | 32.9 % | 2.2 % | **14.9×** |
| panel96 | 39.7 % | 2.6 % | **15.1×** |

The clusters recover real families outright: one cluster holds all four S100 proteins, another
three TNF superfamily members, others the MMP, FGF and cystatin pairs.

One honest caveat: at the default `auto` cut, panel96 also produces a single 30-antigen
cluster containing all 11 four-helix cytokines plus the IL-6 and IL-1 family members. That is
chemically defensible — all-α helical bundles genuinely present similar surfaces — but it is a
coarse call. Tighten `--cluster-cut` if you need that group resolved.

## Output columns

One row per candidate epitope, best first — 98 columns. Every one of them is documented in
the `epitopescope.<tag>.yaml` written beside the CSVs, with a description plus what a high and
a low value mean; see [The metric dictionary](#the-metric-dictionary).

**Identity** — `rank`, `epitope_id`, `antigen`, `chain`, `n_epitope_residues`,
`epitope_residues`, `epitope_sequence`, `rfdiffusion_hotspots`

**Scores and calls** — `nanobody_target_score`, `epitope_score`, `orthogonality_score`,
`pure_ivtt_score`, `pure_zone`, `n_pure_rules_passed`, `pure_rules_all_pass`, `flags`

**Site detection** — `site_source` (`concavity_cluster` or `surface_seed`), `site_buriedness`,
`site_center_x/y/z`, `site_box_A`

**Affinity and efficiency** — `dg_est_kcal_mol`, `kd_est_M`, `pkd_est`, `buried_area_est_A2`,
`apolar_index`, `epitope_le`, `epitope_lle`, `epitope_fq`, `epitope_complexity_index`

**Cross-talk and multivariate** — `crosstalk_max`, `crosstalk_partner`, `crosstalk_top3`,
`crosstalk_percentile`, `crosstalk_within_antigen`, `crosstalk_z`, `panel_selected`,
`panel_worst_crosstalk`, `panel_worst_partner`, `crosstalk_cluster`, `cluster_size`,
`cluster_n_antigens`, `pc1`, `pc2`, `pc3`, `proteome_offtarget_max`, `proteome_offtarget_hit`

**Geometry** — `patch_area_A2`, `mean_rel_sasa`, `protrusion`, `concavity`, `planarity_A`,
`seq_contiguity`, `rigidity`, `mean_bfactor_or_plddt`, `net_charge`, `hydrophobic_frac`,
`interface_buried_A2`, `cofactor_contact`

**Construct** — `construct_range`, `construct_len`, `construct_mw_da`, `construct_pI`,
`construct_gravy`, `construct_disorder_frac`, `dangling_contact_frac`, `n_cys`, `ss_internal`,
`ss_broken_by_excision`, `nglyc_sequons_construct`, `nglyc_sequons_in_epitope`,
`max_hydropathy_w19`, `exposed_hydrophobic_frac`, `construct_sequence`

**Rules** — `rule_size_pass`, `rule_redox_pass`, `rule_glyco_pass`, `rule_membrane_pass`,
`rule_domain_pass`, `rule_monomer_pass`, `rule_order_pass`

**Sub-scores** — `epi_*` and `pure_*`, so you can see exactly which factor sank a candidate
and re-weight it yourself.

`flags` is the human-readable summary: `SS_BROKEN_BY_EXCISION(n)`, `DISULFIDE_IN_CONSTRUCT(n)`,
`NGLYC_IN_EPITOPE(n)`, `TM_LIKE_SEGMENT`, `HYDROPHOBIC_SEGMENT`, `QUATERNARY_EPITOPE(nA2)`,
`GLYCAN_SHIELDED(res)`, `COFACTOR(res)`, `DISORDERED_CONSTRUCT`, `POORLY_EXCISABLE`,
`LARGE_CONSTRUCT`, `HIGH_CROSSTALK`, `SHARED_CLUSTER(n_antigens)`, `LOW_LLE_GREASY`,
`LOW_FIT_QUALITY`.

## The metric dictionary

Every run writes `epitopescope.<tag>.yaml` next to the CSVs (suppress with `--no-yaml`). Its
path is printed to stdout after the CSV paths. The schema follows `example.yaml` — `name`,
`description`, `colour` (`RdBu` numeric, `Spectral` categorical) — with `high`, `low` and
`direction` added, so a CSV can be read without the source:

```yaml
run:
  epitopescope_version: 0.2.0
  embedding_backend: esm2
  n_antigens: 96
  n_epitopes: 1147
  crosstalk_cluster_cut: 0.8019
  temperature_K: 310.15
columns:
  - name: epitope_lle
    description: "Lipophilic-efficiency analogue: pkd_est minus apolar_index. Discounts affinity that is bought with greasy surface rather than specific contacts."
    high: "Affinity comes from specific, polar contacts: the selective kind."
    low: "Affinity is mostly hydrophobic stickiness; expect poor specificity."
    direction: higher_is_better
    colour: RdBu
  - name: rule_redox_pass
    description: "PURE rule 2, redox: no disulfide required by the fold, and none severed by the construct boundary."
    high: "1: nothing depends on oxidation, so a reducing PURE reaction is fine."
    low: "0: the fold needs a disulfide PURE will not form, or excision leaves an unpaired cysteine."
    direction: higher_is_better
    colour: RdBu
```

`direction` is `higher_is_better`, `lower_is_better`, or `context` when neither end is
inherently good (`concavity`, `net_charge`, `seq_contiguity` and the principal components are
all `context`). The 18 identifier, sequence, coordinate and free-text columns carry a
description but no `high`/`low`, because they have no meaningful ordering.

## Feeding the result into a design run

`rfdiffusion_hotspots` is the five most exposed residues of the epitope, already formatted for
RFdiffusion, and `construct_range` / `construct_sequence` give you the antigen fragment to put
in the PURE reaction and to use as the design target:

```bash
# hotspots from the CSV, e.g. A57,A64,A66,A70,A72
./scripts/run_inference.py \
  inference.output_prefix=out/ag1_E03 \
  inference.input_pdb=antigens/ag1.pdb \
  'contigmap.contigs=[A166-275/0 70-100]' \
  'ppi.hotspot_res=[A57,A64,A66,A70,A72]' \
  inference.num_designs=1000
```

then ProteinMPNN for sequences and AlphaFold/Chai-1 for the ipTM and ΔSASA filters described in
`2026.08.28.747755v1.full.pdf` (that preprint is a de novo CR2 binder design campaign, not the
PocketScope paper — it is the closest worked example of this pipeline in the folder).

---

## Limitations, stated plainly

- **The epitope axis is calibrated; the PURE axis is not.** Six `epi_*` factors were tuned
  against 30 solved nanobody complexes and tested on 30 disjoint ones, and the held-out
  enrichment is a modest 1.2–1.45×. Every constant in the PURE axis, by contrast, still comes
  from published rules of thumb (Tien et al. maximum ASA, Kyte–Doolittle hydropathy, the PURE
  size ceiling, PURExpress being reducing) with **no** outcome data behind it, because none
  exists here. Treat the PURE ranking as triage that puts the obviously bad candidates last.
- **The thermodynamic layer is geometry, not physics.** `dg_est`, `kd_est` and `pkd_est` are
  a two-term solvation model over predicted buried area with coefficients chosen to land in a
  realistic range. They are an interpretable common axis and the input to the efficiency
  metrics, not predicted affinities. Do not quote a Kd from this tool.
- **The benchmark is 60 complexes.** Small, X-ray only, and biased toward whatever crystallises
  with a nanobody — the held-out set is heavy in viral spike domains. The enrichment figures
  carry real uncertainty at that n.
- **Cross-talk is measured between patch representations, not by docking.** Two patches with a
  high MaxSim present similar chemistry to a paratope. That is a real and useful risk signal,
  and it recovers known homologues, but it is not a prediction that a specific designed
  nanobody will bind both. The honest use is to *avoid* the high-scoring pairs and, after
  design, to check the actual binders against the other antigens.
- **PDB only.** mmCIF is not parsed; convert first.
- **One file is one antigen.** A multi-chain file is treated as a single antigen with
  per-chain epitopes, and cross-talk is only computed *between* files. Split a complex into
  separate files if its chains are separate antigens in your assay.
- **Conformational epitopes make long constructs.** If a patch draws residues from far apart in
  sequence, the minimal construct must span them, and `--max-construct` cannot shrink it below
  that span. Such candidates are penalised on `pure_length`, which is the intended behaviour.
- **`--embed none` is a fallback.** See the backend table above.
- **PURE expressibility is assessed from structure and sequence only.** Codon usage, mRNA
  secondary structure at the 5' end, and the specific PURE formulation are not modelled.

## Licence

MIT, same as the rest of this repository.
