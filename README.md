# KG-XAI — Knowledge-Graph-Guided Multi-Task Learning for Stroke Triage

Ekanath DV (23MIA1023) · Rohan Julius Preetan (23MIA1160)

A stroke-triage model that answers three questions from a patient's CT
scans — *is there a large-vessel occlusion, how good are the collateral
vessels, where is the infarct* — and, unlike the base paper, makes sure the
three answers **make clinical sense together**. A graph neural network
reasons over the patient's own Circle-of-Willis vessel network, guideline
knowledge is injected into the model while it decides, and a consistency loss
penalises incoherent combinations. We measure this with a new metric, the
**Joint-Decision Consistency Rate (JDCR)**, and every prediction comes with a
traceable explanation — no LLM involved.

**Base paper:** Luan T. et al., [*Unified multimodal learning for stroke
triage: joint detection, scoring, and segmentation of acute ischemic
stroke*](https://doi.org/10.1038/s41746-025-02255-0), npj Digital Medicine
9:441 (2026). Its three task heads are scored independently, so nothing
checks cross-task coherence — that is our entry point.

**Plan of record:** [`PROJECT_PLAN.md`](PROJECT_PLAN.md) — architecture,
novelty, dataset, knowledge graph, 9 phases with pass/fail gates.

---

## Status

| Phase | What | State |
|---|---|---|
| Review 1 | Data pipeline, audits, baseline segmentation, calibration layer | ✅ Done — see *Review 1 record* below |
| **1** | **Knowledge graph** — both layers, verified sources, per-patient vessel features | ✅ **Passed** — [`PHASE1_REPORT.md`](PHASE1_REPORT.md) |
| 2 | Labels & preprocessing — LVO localization, HIR collateral proxy, core/penumbra, location constraints | Next |
| 3 | B0 — base paper's architecture reimplemented (no graph) | |
| 4–6 | GNN, guideline injection (B1), consistency loss (B2) | |
| 7–9 | Rule evaluator + JDCR + rationale, full evaluation, write-up | |

**Phase 1 headlines** (all measured on the real ISLES'24 data):

- The vessel graph lines up with reality: confident clot locations match the
  side of the follow-up infarct in **89/94 (94.7%)** cases (pass mark 80%,
  fixed before running).
- All **7 sources verified** against PubMed, ClinicalTrials.gov or arXiv.
- ISLES'24 has **no timing data** (all 7 timing columns empty, 149/149) and
  no ASPECTS, so no eligibility rule is fully evaluable — JDCR rests on the
  plausibility constraints instead.
- Of the plausibility constraints, **1 is supported** (infarct side matches
  clot side), 1 was **tested and rejected** (upstream clots → bigger infarcts:
  p = 0.66), 2 await Phase 2, 1 is untestable. Size-based rules are weak in
  this all-reperfused cohort; **location-based rules** are the Phase 2
  priority.

## How improvement over the base paper is shown

The base paper's reported numbers aren't directly comparable (different
label definition, extra datasets, a 7-center split the public ISLES'24
release doesn't have — it has 2). So we reimplement its architecture and
compare on identical data, splits and seeds:

| Model | What it is |
|---|---|
| **B0** | Base paper's architecture on ISLES'24, no knowledge graph |
| **B1** | B0 + GNN over the vessel graph + guideline cross-attention |
| **B2** | B1 + consistency loss + rule evaluator + rationale (full KG-XAI) |

The claim is **B2 vs. B0**.

## The knowledge graph

Nothing to download — no knowledge graph exists for ISLES'24. It's authored
from verified sources ([`KNOWLEDGE_GRAPH_SOURCES.md`](KNOWLEDGE_GRAPH_SOURCES.md)):

| Layer | File | Content | Sources |
|---|---|---|---|
| 1 — Vessel graph | [`kg/cow_topology.json`](kg/cow_topology.json) | Circle of Willis: 12 vessels, 12 connections. The GNN runs on this. | TopCoW (arXiv:2312.17670), Alastruey 2007, Liu 2023 atlas |
| 2 — Guideline graph | [`kg/guideline_rules.json`](kg/guideline_rules.json) | 16 concepts, 7 eligibility rules, 5 plausibility constraints | DAWN, DEFUSE 3 (trial registries), AHA/ASA 2026, Kim 2026, Olivot 2014 |

## Dataset

**ISLES'24** public training release — 149 patients, 2 centers (99 / 49),
NCCT, CTA, CT perfusion maps, vessel masks, follow-up infarct masks, clinical
data. **Not included in this repo** (~100 GB); download it separately and pass
its path with `--root`. The release's actual layout (checked against the
archive, not assumed):

```
train/
├── raw_data/sub-strokeXXXX/ses-01/      *_ncct.nii.gz, *_cta.nii.gz, *_ctp.nii.gz, perfusion-maps/
├── derivatives/sub-strokeXXXX/
│   ├── ses-01/                          *_space-ncct_{cta,ctp,cow-msk,lvo-msk}.nii.gz, perfusion-maps/{cbf,cbv,mtt,tmax}
│   └── ses-02/                          *_space-ncct_{dwi,adc,lesion-msk}.nii.gz   (follow-up = the target)
└── phenotype/sub-strokeXXXX/
    ├── ses-01/*_demographic_baseline.csv   (center, age, NIHSS, premorbid mRS, ...)
    └── ses-02/*_outcome.csv                (mRS 3 months, TICI, ...)
```

No `participants.tsv` and no JSON sidecars; the timing columns in the
baseline CSV are empty for every patient.

## Repository layout

```
kg/                knowledge graph: vessel graph + guideline graph (JSON)
src/graph/         graph validation, per-patient vessel features, constraint support checks
src/data/          ISLES'24 audits: volumes, centers, occlusion site, mRS, clinical completeness
src/model/         data pipeline (MONAI) + baseline segmentation training
src/eval/          the four ISLES'24 metrics + decision metrics
src/conformal/     calibration layer (Review 1) — optional component of the rule evaluator
src/experiments/   synthetic calibration validation (Review 1)
outputs/tables/    every number quoted anywhere, as JSON
```

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate                  # .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU track only
```

**Phase 1 — knowledge graph** (no GPU):

```bash
python -m src.graph.build                                   # validate both graphs + provenance
python -m src.data.clinical_audit  --root /path/to/ISLES-2024/train
python -m src.graph.features       --root /path/to/ISLES-2024/train   # vessel features + laterality gate
python -m src.graph.constraint_support                      # test plausibility rules on ground truth
```

**Data audits** (Review 1, still used):

```bash
python -m src.data.audit          --root /path/to/ISLES-2024/train
python -m src.data.center_labels  --root /path/to/ISLES-2024/train
python -m src.data.occlusion_site --root /path/to/ISLES-2024/train
python -m src.data.mrs_shift      --root /path/to/ISLES-2024/train
```

**Baseline segmentation** (CUDA GPU; developed on an RTX 5060, which needs a
cu128+ PyTorch build):

```bash
python -m src.model.train_baseline --root /path/to/ISLES-2024/train \
    --fold 0 --epochs 200 --val-interval 10 \
    --cache-dir /path/to/cache_1mm --num-workers 4
```

`--cache-dir` caches preprocessing to disk (~15× faster repeat epochs). Clear
it if you change the transforms in `src/model/dataset.py`. PyTorch Geometric
is added in Phase 3.

## Review 1 record

Review 1 took a different direction — a statistically certified decision
threshold (conformal prediction + Learn-then-Test). Faculty feedback: the
threshold alone wasn't enough novelty, and the knowledge-graph plan should
return. The Review 1 documents (proposals v3/v4, the v4 plan, the prior-art
table, the progress and results write-ups, the run guide) were removed from
the working tree to avoid confusion. They remain in git history — restore
any of them with `git checkout 011a3ae -- <file>`. The Review 1 summary page
is still here as [`review1_report.html`](review1_report.html).

Its data work carries straight into KG-XAI — the baseline segmentation
(6-channel SegResNet, Dice 0.215, lesion-F1 0.278 vs. leaderboard 0.144) and
the occlusion-site derivation are the foundation of Phases 1–3.

## References

1. Luan T. et al. Unified multimodal learning for stroke triage. *npj Digit. Med.* 9:441 (2026). — base paper
2. Riedel E.O. et al. The ISLES'24 Dataset. *Radiology: Artificial Intelligence* 8(3) (2026). DOI 10.1148/ryai.250603.
3. Full verified reference list: [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §12 and [`KNOWLEDGE_GRAPH_SOURCES.md`](KNOWLEDGE_GRAPH_SOURCES.md).
