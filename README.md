# KG-XAI — Knowledge-Graph-Guided Consistency for Stroke Infarct Segmentation

Ekanath DV (23MIA1023) · Rohan Julius Preetan (23MIA1160)

A stroke-triage system built on CT scans from the public ISLES'24 release. It
segments the final infarct, injects a patient-specific Circle-of-Willis vessel
graph and guideline knowledge into the model, enforces the one clinical rule
that held up in real data (the infarct must sit on the same side as the
occluded vessel), and explains every prediction. Everything is measured on
held-out patients, including the results that did not go our way.

**Base paper:** Luan T. et al., [*Unified multimodal learning for stroke
triage: joint detection, scoring, and segmentation of acute ischemic
stroke*](https://doi.org/10.1038/s41746-025-02255-0), npj Digital Medicine
9:441 (2026). Its task heads are scored independently, so nothing checks that
the outputs agree with each other — that is our entry point.

**Documents**

| File | What it is |
|---|---|
| [`PROJECT_REPORT.md`](PROJECT_REPORT.md) | Combined project record: Part 1 final progress report (all results), Appendices A–C (Phase 1 report, Phase 2 progress, Phase 8 plan with the pass marks fixed before the runs) |
| [`PROJECT_PLAN.md`](PROJECT_PLAN.md) | Plan of record: architecture, novelty, dataset, knowledge graph, phases and gates |
| [`KNOWLEDGE_GRAPH_SOURCES.md`](KNOWLEDGE_GRAPH_SOURCES.md) | Every source behind the two graphs, with how each was verified |
| [`paper/kgxai_conference.tex`](paper/kgxai_conference.tex) | Condensed IEEEtran conference paper |
| [`paper/kgxai_paper.tex`](paper/kgxai_paper.tex) | Full-length version of the paper |
| [`demo/README.md`](demo/README.md) | The interactive demo |
| [`CLAUDE.md`](CLAUDE.md) | Working notes kept during development |

---

## Status

| Phase | What | State |
|---|---|---|
| Review 1 | Data pipeline, audits, baseline segmentation, calibration layer | Done (see *Review 1 record*) |
| 1 | Knowledge graph: both layers, verified sources, per-patient vessel features | Passed — `PROJECT_REPORT.md` Appendix A |
| 2 | Labels, perfusion features, plausibility-constraint tests | Done — Appendix B |
| 3–7 | B0 baseline, GNN, guideline injection (B1), consistency loss (B2), rule evaluator, JDCR, rationale | Built. The LVO and collateral heads failed at this data scale (negative result, Part 1 §5) |
| 8 | Full evaluation: 3 folds × 3 seeds, 21 checkpoints | Complete (2026-09-18) — Part 1 §6a, pass marks in Appendix C |
| After 8 | Mechanism ablations, failure detection vs. baselines, preprocessing benchmark | Complete — Part 1 §6b–6d |
| 9 | Write-up | Conference paper in `paper/` |
| Review 3 | Explainable AI and a local-model assistant in the compare demo | Done — see *Review 3 addition* |

## Headline results

All on held-out ISLES'24 patients, with pass marks fixed before testing.

- **One of seven guideline rules holds** in this cohort: infarct side matches
  occlusion side, 89/94 = 94.7%. The other six are rejected or untestable, most
  likely because every patient in the release was successfully reperfused.
- **The graph does not improve segmentation accuracy.** B2 vs. B0 Dice is not
  significantly different (paired Wilcoxon, p = 0.59 on the per-patient seed-0
  test, n = 148; p = 0.24 pooled over 345 paired observations). A random
  same-size graph and a forced-open attention gate gave the same Dice.
- **The JDCR claim passes on paper but not in substance.** B2's mean strict
  JDCR is 0.965 vs. B0's 0.749, but the gap comes from two B0 runs that
  collapsed to an empty prediction. When B0 trains normally the two are
  indistinguishable (p = 0.84). B2's real advantage is no collapse in 9 runs
  (0/9 vs. 2/9) and lower calibration error.
- **The graph is useful as a rule check.** A ground-truth-free flag (lesion on
  the wrong side, or under 1 mL) detects severe failure with AUROC 0.836,
  against 0.583 for ensemble disagreement. MC-dropout is competitive on subtler
  failures.
- **Auxiliary heads:** LVO localization and the collateral proxy collapsed to a
  constant answer under four regimes, but an admission-NIHSS head learned a real
  signal (Spearman 0.56–0.60), so the cause was label quality, not the
  architecture.

## Models compared

| Model | What it is |
|---|---|
| **B0** | SegResNet baseline on ISLES'24, no knowledge graph |
| **B1** | B0 + GAT over the patient's vessel graph, injected by zero-initialized cross-attention |
| **B2** | B1 + laterality-consistency loss + rule evaluator + rationale (full KG-XAI) |

The base paper's numbers are not directly comparable (different label
definition, extra datasets, a 7-center split the public release lacks), so all
three models are trained by us on identical data, splits and seeds.

## The knowledge graph

No knowledge graph exists for ISLES'24, so it is authored from verified sources
([`KNOWLEDGE_GRAPH_SOURCES.md`](KNOWLEDGE_GRAPH_SOURCES.md)):

| Layer | File | Content | Sources |
|---|---|---|---|
| 1 — Vessel graph | [`kg/cow_topology.json`](kg/cow_topology.json) | Circle of Willis: 12 vessels, 12 connections. The GNN runs on this | TopCoW (arXiv:2312.17670), Alastruey 2007, Liu 2023 atlas |
| 2 — Guideline graph | [`kg/guideline_rules.json`](kg/guideline_rules.json) | 16 concepts, 7 eligibility rules, plausibility constraints | DAWN, DEFUSE 3, AHA/ASA 2026, Kim 2026, Olivot 2014 |

## Dataset

**ISLES'24** public training release — 149 patients, 2 centers (99 / 49), NCCT,
CTA, CT perfusion maps, vessel masks, follow-up infarct masks, clinical data.
**Not included in this repo** (~100 GB); download it separately and pass its
path with `--root`. Layout, checked against the archive:

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

No `participants.tsv` and no JSON sidecars; the timing columns are empty for
every patient, and there is no ASPECTS score or collateral grade.

## Repository layout

```
kg/                knowledge graph: vessel graph + guideline graph (JSON)
src/graph/         graph validation, per-patient vessel features, constraint support checks
src/data/          ISLES'24 audits: volumes, centers, occlusion site, mRS, clinical completeness
src/model/         data pipeline (MONAI), B0/B1/B2 training, GNN head, consistency loss, NIHSS head
src/eval/          ISLES'24 metrics, rule evaluator + JDCR, Phase 8 aggregation, failure-detection baselines
src/conformal/     calibration layer (Review 1), optional part of the rule evaluator
src/experiments/   synthetic calibration validation (Review 1)
demo/              interactive compare demo: server, page, explanations, assistant, tests
paper/             conference paper, full paper, figures, make_figures.py
outputs/tables/    every number quoted anywhere, as JSON
outputs/checkpoints/  trained models (git-ignored; regenerate with the training scripts)
```

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate                  # .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU track only
```

**Knowledge graph and audits** (no GPU):

```bash
python -m src.graph.build                                          # validate both graphs + provenance
python -m src.data.clinical_audit  --root /path/to/ISLES-2024/train
python -m src.graph.features       --root /path/to/ISLES-2024/train   # vessel features + laterality gate
python -m src.graph.constraint_support                             # test plausibility rules on ground truth
python -m src.data.audit           --root /path/to/ISLES-2024/train
```

**Training** (CUDA GPU; developed on an RTX-class laptop GPU that needs a
cu128+ PyTorch build). B0 is shown; B1 and B2 use `train_graph_seg.py` and
`train_graph_seg_consistency.py`, and the full 3-fold × 3-seed grid is driven
by `src/model/run_phase8.py`:

```bash
python -m src.model.train_baseline --root /path/to/ISLES-2024/train \
    --fold 0 --epochs 200 --val-interval 10 \
    --cache-dir /path/to/cache_1mm --num-workers 4
```

`--cache-dir` caches preprocessing to disk (about 15× faster repeat epochs);
clear it if you change the transforms in `src/model/dataset.py`.

## Review 3 addition: explainable AI in the compare demo

The compare page runs two models side by side on the same patient, with the
ground-truth-free trust flag and the rule-check rationale under each. Review 3
adds three things there (details in [`demo/README.md`](demo/README.md)):

- **"Why this prediction?"** — blanks each of the six scan channels in turn and
  reports how much of the model's confidence in its own lesion disappears,
  as six bars plus a per-channel heatmap on the slice.
- **A local-model assistant** — `gemma3:4b` via Ollama answers questions about
  the patient using only numbered facts the system computed. Answers with
  invented numbers, missing citations, advice wording or misattributed facts are
  rejected and replaced by a built-in summary.
- **Tests** — `python -m demo.test_xai` (25 unit checks) and
  `python -m demo.test_xai --live` (adds 9 checks against the running server).

```bash
ollama serve                                   # once, for the assistant
python -m demo.server --root /path/to/ISLES-2024/train --cache-dir /path/to/cache --warm-explain
# open http://127.0.0.1:8765
```

The explanations describe what the model relied on, not the patient's
physiology, and have not yet passed a formal faithfulness test
(`demo/faithfulness.py` is written but was not run to completion).

## Paper

```bash
python paper/make_figures.py        # regenerates paper/figures/*.png and .pdf from the reported numbers
```

Build `paper/kgxai_conference.tex` with any LaTeX distribution (IEEEtran class).
Put the architecture diagram at `paper/figures/fig_architecture.png`.

## Review 1 record

Review 1 took a different direction — a statistically certified decision
threshold (conformal prediction + Learn-then-Test). Faculty feedback: the
threshold alone wasn't enough novelty, and the knowledge-graph plan should
return. The Review 1 documents were removed from the working tree to avoid
confusion; they remain in git history, and any can be restored with
`git checkout 011a3ae -- <file>`. Its data work carries straight into KG-XAI:
the baseline segmentation (6-channel SegResNet) and the occlusion-site
derivation are the foundation of Phases 1–3.

## References

1. Luan T. et al. Unified multimodal learning for stroke triage. *npj Digit. Med.* 9:441 (2026). — base paper
2. Riedel E.O. et al. The ISLES'24 Dataset. *Radiology: Artificial Intelligence* 8(3) (2026). DOI 10.1148/ryai.250603.
3. Full verified reference list: [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §12 and [`KNOWLEDGE_GRAPH_SOURCES.md`](KNOWLEDGE_GRAPH_SOURCES.md).
