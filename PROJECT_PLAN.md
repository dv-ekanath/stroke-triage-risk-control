# KG-XAI — Project Plan

**Knowledge-Graph-Guided Multi-Task Learning for Stroke Triage**
Ekanath DV (23MIA1023) · Rohan Julius Preetan (23MIA1160)

This is the single plan of record after Review 1. It supersedes
`KG_XAI_BUILD_PLAN.md` and the conformal-threshold direction in
`stroke_triage_proposal_v4.md` / `stroke_triage_v4_plan.md`. It follows the
system architecture and methodology on the original Review 1 deck (slides 8
and 10), corrected wherever the real dataset turned out to differ from what
the deck assumed.

---

## 0. The project in one paragraph

The base paper trains one model that looks at a stroke patient's CT scans and
produces three answers at once: is there a large-vessel occlusion (LVO), how
good are the collateral vessels, and where is the dead brain tissue. Each
answer is scored separately, so nothing ever checks whether the three answers
make sense *together* — the model can say "large occlusion, good collaterals,
huge infarct," a combination real strokes rarely produce, and every metric it
reports would still look fine. We add a **knowledge-graph-guided consistency
module**: a graph neural network reasons over the patient's own
Circle-of-Willis blood-vessel network, clinical guideline knowledge is
injected into the model while it decides, and a consistency loss penalises
clinically incoherent combinations. We measure the result with a new metric,
the **Joint-Decision Consistency Rate (JDCR)**, and produce a traceable
explanation for every prediction with no LLM involved.

**Novelty in one sentence (for the slide):** *We make a multi-task stroke
triage model's outputs clinically coherent with each other — by reasoning
over the patient's vascular graph with a GNN and injecting guideline
knowledge into the decoder — and introduce a metric that measures the
cross-task inconsistency existing per-task metrics cannot see.*

---

## 1. Why this direction, and what carries over

Review 1 feedback: (a) a certified decision threshold is not a large enough
novelty on its own; (b) the original knowledge-graph plan should not have
been dropped; (c) use a GNN.

This plan goes back to the deck's architecture. Nothing from Review 1 is
thrown away — it becomes the foundation:

| Built in Review 1 | Role in this plan |
|---|---|
| Real ISLES'24 data pipeline, integrity checks, perfusion clipping, caching (`src/model/dataset.py`) | Stages 1–2 of the architecture (input + preprocessing) |
| Occlusion-site derivation, geometrically verified (`src/data/occlusion_site.py`) | Labels for the LVO head, and per-patient node features for the GNN |
| Center recovery, volume audit, mRS targets (`src/data/`) | Splits, stratification, clinical graph inputs |
| Baseline segmentation, Dice 0.215 (`src/model/train_baseline.py`) | Reference point for the segmentation head |
| ISLES'24 metrics (`src/eval/metrics.py`) | Evaluation, unchanged |
| Conformal calibration layer (`src/conformal/`) | Optional component inside the Rule Evaluator (Phase 7) — **not** presented as a contribution |

---

## 2. Base paper, and how we show improvement over it

**Base paper:** Luan T. et al. *Unified multimodal learning for stroke triage:
joint detection, scoring, and segmentation of acute ischemic stroke.*
npj Digital Medicine 9:441 (2026). https://doi.org/10.1038/s41746-025-02255-0

**What it does:** NCCT + CTA + CTP → per-modality encoders → cross-modal
fusion → shared decoder → three heads (LVO detection, collateral grading,
core/penumbra segmentation). Reported: LVO AUC 0.92, collateral κ 0.72,
Dice 0.81, ECE 0.130, volume error 10.5%, unnecessary transfers 27%→12%.

**What it lacks (our entry point, from the deck's Problem Statement):** the
three heads are independent. Nothing checks cross-task clinical coherence,
there is no guideline reasoning, and no explanation.

### How improvement is demonstrated — this part must be stated carefully

**We cannot compare our numbers directly against the base paper's reported
numbers.** Their headline results come from a different label definition
(core/penumbra vs. ISLES'24's final infarct), additional datasets we are not
using, and a stated 7-center ISLES'24 evaluation that the actual public
release does not support (it has 2 centers — confirmed from the data itself).
A faculty reviewer who knows this would reject a direct numeric comparison.

The defensible method is a **controlled comparison on identical data:**

| Model | What it is |
|---|---|
| **B0** | The base paper's architecture reimplemented on ISLES'24 — 3 encoders, cross-modal fusion, shared decoder, 3 heads, **no knowledge graph** |
| **B1** | B0 + GNN over the vascular graph + guideline cross-attention injection |
| **B2 (KG-XAI, full)** | B1 + differentiable consistency loss + Rule Evaluator + rationale |

Same splits, same seeds, same preprocessing, same training budget. The
improvement claim is **B2 vs. B0**, with B1 isolating how much comes from
injection alone vs. the consistency loss. The base paper's own reported
numbers are cited as external context only, with the reasons above stated
next to them.

**What improvement we expect, stated honestly up front:**

- **JDCR** is where the knowledge graph should show a clear gain — that is
  what it is designed to do.
- **ECE** (calibration) may improve, since incoherent confident predictions
  are penalised. The base paper's 0.130 is the context figure.
- **Per-task metrics** (AUC, κ, Dice) should stay the same or improve
  modestly. If they get worse, that is a real trade-off and gets reported as
  one. The primary claim is coherence and explainability, not raw accuracy —
  this should be said before a reviewer asks.

---

## 3. Novelty

### What is new

1. **Cross-task consistency as a training objective for stroke triage.** The
   base paper treats three outputs independently. We formalize which joint
   outputs are clinically coherent and enforce it with a differentiable loss.
2. **Knowledge used *during* decision-making, not checked after it.** Guideline
   knowledge enters the decoder through cross-attention (deck Objective 2),
   rather than a rule checker run on finished outputs.
3. **GNN over the patient's real vascular anatomy.** The graph is the Circle
   of Willis itself — 12 vessel segments with fixed anatomical connections.
   Collateral circulation *is* blood rerouting through the Acom/Pcom edges of
   this graph, so message passing along those edges models the real
   mechanism behind collateral grading. This is the GNN the faculty asked for,
   applied to data that is genuinely graph-shaped.
4. **Guideline checking extended to continuous volumes (mL)** (deck Objective
   3) — thresholds on infarct/penumbra volume become soft, differentiable
   constraints instead of hard rules.
5. **A new metric, JDCR**, reported at train and test time. No prior stroke
   triage work reports cross-task consistency.
6. **Traceable, deterministic rationale** — every explanation is assembled
   from the model's actual outputs and the specific graph rules they
   satisfied or violated (deck Objective 4).

### Differentiation from prior work (the deck's slide 9, sharpened)

| Work | Has | Does not have |
|---|---|---|
| Base paper (Luan et al. 2026) | Multimodal CT, 3 joint heads | Any cross-task consistency, guideline reasoning, or explanation |
| RAD (NeurIPS 2025) | Guideline cross-attention injection | Volumetric / segmentation tasks; stroke; vascular graph |
| Graph-based knowledge VLM (MICCAI 2025) | Knowledge graph + GNN | Multi-task learning; stroke; volumetric output |
| Current stroke AI models | Strong segmentation / LVO detection | Guideline validation; clinician-traceable reasoning |
| **KG-XAI (ours)** | All of the above combined, on multi-task volumetric stroke triage, with a vascular-topology GNN and a consistency metric | — |

### What is *not* claimed as new

GAT, cross-attention, multi-task learning, multimodal fusion and template
explanations are all established techniques. The contribution is their
combination into a consistency-enforcing stroke triage system, the vascular
graph formulation, and the JDCR metric. Say this explicitly — it is the
question a reviewer asks first.

---

## 4. Dataset

### Primary: ISLES'24 (the only dataset the core claim rests on)

Location: `D:\ISLES-2024\train` (extracted from `E:\Nosql\train.7z`, 102.9 GB,
verified against the archive manifest).

| Fact | Value | Source |
|---|---|---|
| Subjects | 149 | Counted in the release |
| Centers | **2** (99 / 49, 1 blank) | Recovered from phenotype CSVs — **the deck says 7; that is wrong and must be corrected** |
| Imaging | NCCT, CTA, 4D CTP, perfusion maps (CBF, CBV, MTT, Tmax) | `raw_data/`, `derivatives/` |
| Vessel annotations | `lvo-msk` (manual occlusion), `cow-msk` (Circle of Willis, 12 TopCoW labels) | `derivatives/ses-01/` |
| Segmentation target | Final infarct on day 2–9 DWI (`lesion-msk`) | `derivatives/ses-02/` |
| Clinical | Age, sex, NIHSS, mRS (premorbid, 3-month), TICI, timings, labs | `phenotype/*.csv` |
| Infarct volume | 31.7 ± 45.1 mL, median 12.4 | Audited |

### Where each task's labels come from

| Head (per deck) | Label source | Problem found | Plan |
|---|---|---|---|
| **LVO detection** | Non-empty `lvo-msk` | ISLES'24 only includes treated LVO patients — only **4 of 149** have an empty `lvo-msk`. A binary LVO yes/no head would be trained on 4 negatives and cannot be evaluated meaningfully. | Reframe as **LVO detection & localization**: classify the occlusion as proximal-anterior (ICA/M1), other-confident, or distal/unlocalized. These classes have real variation (87 / 15 / 42). Report why the base paper's binary AUC cannot be reproduced on this dataset. |
| **Collateral grading** | Not in ISLES'24 (documented) | No ground truth exists | **Hypoperfusion Intensity Ratio (HIR)** = volume(Tmax>10s) / volume(Tmax>6s), a published collateral surrogate, computed from the Tmax maps; binned into 4 levels matching the deck's 0–L3 output. Always reported as a *proxy*. |
| **Infarct segmentation** | `lesion-msk` (final infarct) | Deck says core & penumbra; the label is final infarct | Segment final infarct (ground truth). Derive acute **core** (relative CBF < 30%) and **penumbra** (Tmax > 6s) volumes from the perfusion maps as auxiliary quantities — the guideline rules (DEFUSE-3 mismatch) need them. |

### Known data issues (already found and handled in Review 1)

- `sub-stroke0043`: truncated CBF file in the source release — excluded by the integrity check.
- `sub-stroke0016`: `cow-msk` / `lvo-msk` shape mismatch — excluded from vessel features.
- MTT/CBF/CBV contain extreme artefact values — clipped to physiological ranges (this was causing 29% NaN batches).
- `mRS premorbid` missing for 59/149.
- **No timing data at all (measured in Phase 1):** all 7 timing columns — onset-to-door, alert-to-door, door-to-imaging, door-to-groin, door-to-first-series, door-to-recanalization, time of intervention — are empty for 149/149. No time-window rule can be evaluated. Age 149/149, NIHSS 146/149 and the wake-up flag 148/149 are usable (`outputs/tables/clinical_completeness.json`).

### Supplementary datasets (deck slide 7) — recommendation: do not use for the core claim

The deck lists CPAISD (112 cases) and a second entry also labelled "CPAISD"
(397 cases — this is actually **AISD**; fix the duplicate label). Both are
**NCCT-only**, with no CTA or CTP, no vessel annotations, and different
labels. They cannot train the multimodal model or the vascular graph.
At most they could pretrain the NCCT encoder. Neither has been downloaded.
Recommended: drop them from the core pipeline and say so on the slide, or
keep NCCT-encoder pretraining as an optional stretch goal.

---

## 5. The knowledge graph — what it is and where it comes from

**There is no knowledge graph to download.** None exists for ISLES'24 or for
stroke triage. The deck's own graph box names *documents* ("AHA/ASA, DAWN,
DEFUSE-3"), not a dataset. The graph is authored from the sources below.
Draft versions exist in `kg/cow_topology.json` and `kg/guideline_rules.json`.

The knowledge graph has **two layers**, each with a different job:

### Layer 1 — Vascular graph (what the GNN runs on)

The Circle of Willis: 12 vessel segments, 12 connections, the same wiring in
every patient (anatomical variants appear as missing vessels).

```
BA ── L-PCA ── L-Pcom ── L-ICA ── L-MCA
 │                         └──── L-ACA ──┐
 │                                      Acom
 │                         ┌──── R-ACA ──┘
BA ── R-PCA ── R-Pcom ── R-ICA ── R-MCA
```

| Source | Used for |
|---|---|
| TopCoW (Yang et al., arXiv:2312.17670) | Vessel definitions and label convention — **already verified** against our real `cow-msk` data |
| Alastruey et al., *J. Biomech.* 40:1794 (2007) — 1D blood-flow model of the CoW | Published adjacency of the network (their simulation is solved on this exact graph) |
| Liu C.-F. et al., *Digital 3D Brain MRI Arterial Territories Atlas*, *Sci. Data* 10:74 (2023), PMID 36739282 | Which brain region each vessel supplies, so perfusion can be summarised per node |

**Per-patient node features:**

| Feature | From |
|---|---|
| `segment_present` | Whether that label exists in the patient's `cow-msk` |
| `occluded` | Occlusion site derivation — **only if the match is confident (≤15 mm)** |
| Territory perfusion (Tmax>6s, Tmax>10s volume, mean CBF/CBV) | Perfusion maps, summarised over the vessel's territory. Version 1: left/right hemisphere + anterior/posterior split (no registration needed). Version 2: territory atlas above (needs registration to atlas space). |

**A real trap found in the data:** of 20 patients whose nearest vessel was the
basilar artery, **19 were low-confidence matches at a median 39 mm** — not
basilar occlusions, just the vessel sitting closest to the middle when the
real occlusion is beyond the segmented circle. Feeding those in as
`occluded = 1` would have given the GNN 19 fake basilar strokes. Confidence
gating fixes it; the confident-only distribution is 70% MCA / 16% ICA / 1%
BA, which matches the clinical literature. Distal occlusions get a
graph-level `occlusion_unlocalized` flag instead of being forced onto a node.

### Layer 2 — Clinical guideline graph (what gets injected and enforced)

Concept nodes: LVO status, occlusion site, collateral grade, core volume,
penumbra volume, NIHSS, age, premorbid mRS, time window, eligibility.
Edges are rules connecting them.

| Source | What is extracted |
|---|---|
| **AHA/ASA Guideline for Early Management of Acute Ischemic Stroke** (current edition) | Endovascular therapy eligibility by occlusion site, time window, NIHSS, pre-stroke mRS; large-core recommendations |
| **DAWN** — Nogueira et al., *NEJM* 378:11–21 (2018) | Clinical–imaging mismatch groups (age / NIHSS / core volume bands), 6–24 h |
| **DEFUSE-3** — Albers et al., *NEJM* 378:708–718 (2018) | Core < 70 mL, mismatch ratio ≥ 1.8, mismatch volume ≥ 15 mL, 6–16 h |
| **Base paper** (Luan et al.) | Their implicit decision chain: LVO → collateral → core → decision (deck Objective 1) |
| **Deck Problem Statement** | Cross-task plausibility constraints, e.g. LVO-positive + small core + poor collateral is implausible |

Two kinds of edges:

- **Eligibility rules** — hard clinical criteria from the trials/guideline.
- **Plausibility constraints** — which joint outputs are coherent. These drive
  the consistency loss and JDCR. Each one is checked against real
  co-occurrence in the training data before use; a constraint the data
  contradicts is dropped, not kept.

**Verification (done in Phase 1, 2026-09-13):** every citation was checked
against a real record (PubMed via NCBI E-utilities, trial registries via the
ClinicalTrials.gov API). DAWN and DEFUSE 3 thresholds are quoted from the
registry eligibility text. The one exception: the **AHA/ASA 2026
recommendation wording** is from secondary summaries, because the full text
is blocked to automated access — confirm it before quoting. Details in
`KNOWLEDGE_GRAPH_SOURCES.md`.

**What Phase 1 found about evaluability:** with no timing data and no
ASPECTS, **no eligibility rule is fully evaluable on ISLES'24.** The Rule
Evaluator can check the imaging and clinical parts of the DAWN/DEFUSE 3
rules (after Phase 2 builds acute core and penumbra), and must say "time
window not recorded" rather than calling a patient eligible. JDCR therefore
rests on the **plausibility constraints**, not the eligibility rules.

### Not used, and why

- **UMLS / SNOMED CT / BioPortal** — give standard concept IDs, not triage
  rules. Add only if a reviewer asks for ontology grounding.
- **Generic "medical knowledge graph" datasets** (Kaggle, HuggingFace) — none
  contain the stroke triage decision chain, and their provenance usually
  can't be traced, which contradicts deck Objective 4.

---

## 6. System architecture (deck slide 8, box by box)

### 1 · Input data
NCCT, CTA, CTP. **Change:** the CTP input is the four perfusion maps
(CBF, CBV, MTT, Tmax), not raw 4D CTP — raw CTP has a different number of time
frames per patient (44–60), so it cannot feed a fixed-shape encoder. Same box,
same role.

### 2 · Preprocessing
| Step | Status |
|---|---|
| Co-registration | Done — derivatives are already in NCCT space, verified per subject |
| Skull stripping | To add (MONAI / SimpleITK threshold + morphology, or a CT brain-extraction model) |
| Intensity normalization | Done (+ perfusion clipping) |
| Vessel enhancement | To add — Frangi vesselness filter on CTA |
| Resampling | Done — 1 mm isotropic |

### 3 · Multimodal encoder & fusion
NCCT encoder, CTA encoder, CTP encoder → cross-modal attention fusion →
shared decoder, **as drawn on the deck**.
- Encoders: three lightweight 3D CNN encoders (SegResNet-style — the backbone
  already proven on this data). Swin-UNETR (the base paper's backbone) is the
  alternative; the 8 GB GPU decides — test memory in Phase 3.
- Fusion: attention across modality tokens at the bottleneck, where the
  feature map is small and attention is cheap.
- Shared decoder: upsampling decoder with skip connections from all three
  encoders.

### 4 · Task heads
LVO detection & localization · Collateral grading (HIR proxy, 0–L3) ·
Infarct segmentation — label sources in §4.

### 5 · Outputs
LVO result · collateral grade · infarct volume (+ derived core/penumbra) ·
**consistency score (0–1)** per patient · **clinical rationale**.

### 6 · KG-guided consistency module (our addition)
```
Vascular graph (Layer 1) ──► GAT encoder ──┐
                                            ├──► Cross-attention into the
Guideline graph (Layer 2) ─► embeddings ───┘     shared decoder (guideline injection)
                                                          │
                              joint outputs ──► Differentiable consistency loss
```
- **GAT encoder** (PyTorch Geometric) over the 12-node vascular graph with
  per-patient features → one embedding per vessel.
- **Cross-attention:** decoder features are the queries; vessel embeddings and
  guideline concept embeddings are the keys/values. The result is added back
  into the decoder, so the graph shapes predictions as they are made.
- **Consistency loss:** each plausibility constraint becomes a differentiable
  penalty computed from soft predictions — e.g. "LVO-positive with no
  proximal site" → P(LVO) × P(site = other). Volume conditions use a smooth
  step: sigmoid((predicted volume − threshold) / temperature), where predicted
  volume = sum of lesion probabilities × voxel volume.

*Why this avoids the problem that dropped the consistency loss in v3:* v3's
version fitted a 36-cell compatibility table on ~119 training cases (about 3
cases per cell — pure noise). Here the constraints come from published
guidelines, not from fitting, and the data is only used to *check* them.

### 7 · Training & inference pipeline
Training data → loss (multi-task + consistency) → AdamW, cosine annealing →
inference → Rule Evaluator + JDCR + rationale → clinician review.

```
L = λ_seg·(Dice + CE)            segmentation
  + λ_lvo·CE (class-weighted)    LVO detection & localization
  + λ_col·ordinal loss           collateral proxy
  + λ_cons·Σ constraint penalties  consistency (KG)
```

---

## 7. Evaluation

| Metric | What it measures | Status |
|---|---|---|
| LVO localization accuracy / macro-AUC | Head 1 | New head |
| Quadratic-weighted κ (QWK) | Collateral proxy — same metric the base paper reports | New head |
| Dice, abs. volume diff., lesion-wise F1, abs. lesion count diff. | Segmentation — all four official ISLES'24 metrics | Implemented |
| ECE | Calibration — base paper reports 0.130 | **To implement** |
| **JDCR** | Fraction of patients whose joint output satisfies all plausibility constraints | **New metric** |
| Per-patient consistency score | Output box 5 | New |
| Concept-attention alignment | Do GAT / cross-attention weights concentrate on the clinically relevant vessel (e.g. the occluded one)? | New |

**Protocol:** 5-fold cross-validation stratified by infarct volume, 3 seeds,
mean ± SD. Leave-one-center-out (deck methodology step 1) — with only 2
centers this is two runs (99→49 and 49→99), reported as a robustness check,
not a generalisation claim. Results stratified by failure mode (deck step 7):
confident vs. unlocalized occlusion, small vs. large infarct, center.

**Ablations** (each reported whether or not it flatters the model):

| Ablation | Question it answers |
|---|---|
| B0 vs. B1 vs. B2 | What the graph adds, and how much comes from injection vs. the loss |
| Vascular graph only / guideline graph only / both | Which layer does the work |
| GAT vs. GCN | Whether attention matters |
| With vs. without confidence gating | Size of the basilar-artifact effect |
| Hemisphere features vs. territory atlas | Whether finer node features help |

---

## 8. Phases — build, test, then move on

Each phase ends with a **gate**. A phase that fails its gate blocks the next
one — the same rule that caught the corrupted file, the NaN losses and the
basilar artifact in Review 1.

| # | Phase | Build | Gate (must pass to move on) |
|---|---|---|---|
| **1** | **Knowledge graph** | Finalize Layer 1 and Layer 2; verify every rule against its paper; audit `Onset to door` completeness; confidence-gated node features | Validator passes; every rule has a checked citation; confident cases land on the anatomically correct vessel |
| **2** | **Labels & preprocessing** | LVO localization labels; HIR collateral proxy; core/penumbra volumes from perfusion; skull stripping; CTA vessel enhancement | Class distributions reported; HIR is worse in proximal-occlusion patients than others; derived volumes in plausible ranges |
| **3** | **B0 — base paper reimplemented** | 3 encoders + cross-modal attention fusion + shared decoder + 3 heads, no KG; install PyTorch Geometric | Fits in 8 GB; each head beats its trivial baseline; Dice not below the Review 1 baseline (0.215) |
| **4** | **GNN standalone** | GAT over the vascular graph, trained alone to predict localization / collateral proxy from node features | Beats the same features in a plain classifier — otherwise the graph structure adds nothing yet |
| **5** | **B1 — guideline injection** | Cross-attention of graph embeddings into the decoder | Same fold/seed as B0: no head gets worse; at least one improves |
| **6** | **B2 — consistency loss** | Differentiable constraint penalties; tune λ_cons | JDCR rises vs. B1 without a real drop in task metrics |
| **7** | **Rule Evaluator, JDCR, rationale** | Post-hoc rule check, per-patient consistency score, template rationale; (optional) certified volume decision from `src/conformal/` | Rationales manually correct for 10 real patients across confident and unlocalized cases |
| **8** | **Full evaluation** | 5-fold × 3 seeds for B0 and B2, leave-one-center-out, ablations, ECE | Every number traces to an `outputs/tables/*.json` file |
| **9** | **Write-up** | Updated deck, report, results document | Deck corrections in §11 applied |

### Timeline (the deck's Gantt ends late October; ~6 weeks remain)

| Week | Dates (2026) | Phases |
|---|---|---|
| 1 | Sep 14 – Sep 20 | 1, 2 |
| 2 | Sep 21 – Sep 27 | 3 |
| 3 | Sep 28 – Oct 4 | 4, 5 |
| 4 | Oct 5 – Oct 11 | 6, 7 |
| 5 | Oct 12 – Oct 18 | 8 |
| 6 | Oct 19 – Oct 25 | 9 |

Shift to the actual Review 2 / Review 3 dates once known.

---

## 9. Compute budget

One 200-epoch run of the single-trunk baseline took ~1.9 h. Three encoders
will be slower — estimate 2.5–3 h per run.

- **Development (Phases 3–6):** fold 0, seed 0 only — about 6–10 runs.
- **Final (Phase 8):** B0 and B2 on 5 folds × 3 seeds = 30 runs ≈ 75–90 GPU
  hours; ablations on fold 0 only.

That is several days of continuous GPU time — schedule overnight runs from
Week 4. Running every ablation on the full 5 × 3 protocol does not fit.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| The KG doesn't improve per-task metrics | Expected and acceptable — the claim is consistency (JDCR) and explainability. Report task metrics honestly either way. |
| LVO negatives almost absent | Localization framing (§4); state why binary AUC can't be reproduced |
| Collateral proxy is not ground truth | Label it a proxy everywhere; cite the HIR literature |
| Time-window data absent (confirmed: 0/149) | Time conditions kept for fidelity but marked non-evaluable; eligibility reported as "criteria met, time window not recorded"; JDCR built on plausibility constraints |
| **JDCR rests on few constraints** — after Phase 1 only 1 of 4 is supported (laterality), 2 await Phase 2, 1 is unverifiable | Phase 2 adds a **territory constraint** (occluded vessel's territory vs. infarct location, using the Liu 2023 atlas) and support-checks the collateral constraints. If fewer than 3 hold, say so and narrow the JDCR claim. |
| Guideline thresholds misremembered | Done — verified against registries and PubMed in Phase 1 (AHA/ASA wording excepted) |
| 3-encoder model exceeds 8 GB | Smaller filters, smaller patches, gradient checkpointing; fall back to a shared-trunk variant and disclose it |
| Small cohort (149, 2 centers) | Cross-validation with seeds; per-fold results shown; no generalization claims |
| Base paper code/weights unavailable | B0 is reimplemented from its methods section; say so |

---

## 11. Corrections to make on the deck before the next review

| Slide | Current | Correct |
|---|---|---|
| 7 Dataset | "149 (7 centers)" | 149, **2 centers** (verified from the data) |
| 7 Dataset | Two entries both called "CPAISD" | Second is **AISD**; state whether either is used (recommended: not in the core pipeline) |
| 8 Architecture | CTP input | CTP = derived perfusion maps (CBF, CBV, MTT, Tmax) |
| 8 Architecture | Collateral grading (0–L3) | Collateral grade **proxy** from HIR (ISLES'24 ships no collateral grade) |
| 8 Architecture | LVO detection (Yes/No) | LVO detection & localization (only 4 negatives in ISLES'24) |
| 8 Architecture | Infarct segmentation (core & penumbra) | Final infarct (ground truth) + perfusion-derived core/penumbra volumes |
| 8 Architecture | Clinical KG box | Two layers: vascular graph (GNN) + guideline graph (injection & constraints) |
| 10 Methodology | "Leave-one-center-out split" | 5-fold CV primary; leave-one-center-out as a 2-center robustness check |
| 10 Methodology | "Zero additional training required for the graph" | No longer true — the GNN and cross-attention are trained |

---

## 12. References

1. Luan T. et al. Unified multimodal learning for stroke triage. *npj Digit. Med.* 9:441 (2026). — **base paper**
2. Riedel E.O. et al. The ISLES'24 Dataset. *Radiology: Artificial Intelligence* 8(3) (2026).
3. de la Rosa E. et al. ISLES'24: Final infarct prediction with multimodal imaging and clinical data. arXiv:2408.10966.
4. Yang K. et al. The TopCoW Challenge — Topology-Aware Circle of Willis Segmentation for CT and MR Angiography. arXiv:2312.17670 (2023).
5. Alastruey J. et al. Modelling the circle of Willis to assess the effects of anatomical variations and occlusions on cerebral flows. *J. Biomech.* 2007;40(8):1794–805. PMID 17045276.
6. Liu C.-F. et al. Digital 3D Brain MRI Arterial Territories Atlas. *Sci. Data* 2023;10(1):74. PMID 36739282.
7. Nogueira R.G. et al. Thrombectomy 6 to 24 Hours after Stroke with a Mismatch between Deficit and Infarct (DAWN). *NEJM* 2018;378(1):11–21. PMID 29129157. Registry NCT02142283.
8. Albers G.W. et al. Thrombectomy for Stroke at 6 to 16 Hours with Selection by Perfusion Imaging (DEFUSE 3). *NEJM* 2018;378(8):708–718. PMID 29364767. Registry NCT02586415.
9. Prabhakaran S. et al. 2026 Guideline for the Early Management of Patients With Acute Ischemic Stroke (AHA/ASA). *Stroke* 2026;57(8):e316–e436. PMID 41582814.
10. Olivot J.-M. et al. Hypoperfusion intensity ratio predicts infarct progression and functional outcome in the DEFUSE 2 Cohort. *Stroke* 2014;45(4):1018–23. PMID 24595591.
10b. Kim H. et al. Defining the Therapeutic Ceiling of Endovascular Thrombectomy in Large-Core Stroke. *Stroke* 2026;57(9):2677–2686. PMID 42403349.
11. Veličković P. et al. Graph Attention Networks. ICLR 2018.
12. Fey M., Lenssen J.E. Fast Graph Representation Learning with PyTorch Geometric. ICLR Workshop 2019.
13. Li H. et al. RAD: Towards Trustworthy Retrieval-Augmented Multi-modal Clinical Diagnosis. NeurIPS 2025.
14. Li C. et al. Fine-tuning Vision Language Models with Graph-based Knowledge for Explainable Medical Image Analysis. MICCAI 2025.

References 4–10b were checked against PubMed / arXiv / trial registry records
on 2026-09-13. The AHA/ASA 2026 citation is verified; its recommendation
wording is not (full text blocked). References 11–14 are standard ML
citations, not rechecked.
