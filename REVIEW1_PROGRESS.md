# Review 1 — Progress Report

**Project:** Risk-Controlled Threshold Triage for Ischemic Stroke
**Authors:** Ekanath DV (23MIA1023), Rohan Julius Preetan (23MIA1160)
**Rubric target:** ≥30% implementation completed for Review 1 (20 marks — Problem Understanding 3, System Design & Methodology 5, Implementation Progress 5, Technical Execution & Code Quality 4, Presentation 3)

This document explains what has actually been built and run, organised against
that rubric. Exact metric values are in [`REVIEW1_RESULTS.md`](REVIEW1_RESULTS.md);
the full technical proposal is [`stroke_triage_proposal_v4.md`](stroke_triage_proposal_v4.md).

---

## 1. Problem Understanding

The base paper (Luan et al., *Unified multimodal learning for stroke triage*,
npj Digital Medicine 9:441, 2026) is not weak because of a Dice-score gap to the
ISLES'24 leaderboard — that gap is a task-definition mismatch (their label is
core/penumbra vs. our final-infarct target) and doesn't need a novel method to
explain. The real weakness, found in the paper's full text, is its uncertainty
layer:

- Abstention is a **hand-set threshold, τ = 0.15**, justified post hoc.
- ECE = 0.130 is reported and called "excellent calibration" — 13 points of
  miscalibration by the metric's own definition.
- The clinical safety claim (unnecessary transfer 18.9% → 12.3%) is computed on
  **n = 30** patients against an unmatched historical rate.
- Their own ablation shows disabling rejection nearly doubles mis-triage (5% → 9%).

**The gap this project targets:** conformal prediction gives *coverage* — a
guarantee that a volume interval contains the truth with some probability,
averaged over the whole patient distribution. A triage decision only cares
which side of a threshold τ the truth falls on. A rule can have excellent
coverage while abstaining on everything, or poor coverage while every decision
is correct — coverage and decision correctness are different objects, and
nobody has certified the *decision* directly, at a clinically motivated
asymmetry, with a finite-sample guarantee. That is this project's claim.

## 2. System Design & Methodology

Three claims (down from an earlier four-claim draft — two of the original
claims did not survive verification against prior art, see
`stroke_triage_v4_plan.md` §0/§1 for the record):

- **N1 (primary):** a conformal predictive system gives a calibrated crossing
  probability P(V>τ) per patient; a Learn-then-Test certified decision rule
  then controls two clinically asymmetric risks simultaneously —
  `R_missed` (denying treatment to a patient who'd benefit) and `R_futile`
  (futile reperfusion). The two risks are shown to **decouple structurally**
  (R_missed depends only on the upper decision boundary, R_futile only on the
  lower one), which is what makes certification with all n=149 cases possible
  at all — the general multi-dimensional LTT machinery is not needed.
- **N2 (secondary):** decision-curve analysis extended with a deferral arm,
  using the mRS-shift head to estimate the harm ratio empirically.
- **N3 (secondary):** what n=149 can actually certify — a minimum-sample-size
  table per risk level, and an irreducible-error floor from annotation noise.

Architecture: a 6-channel (NCCT, CTA, CBF, CBV, MTT, Tmax) segmentation trunk
(Review 1's baseline) feeding, in Review 2, volume-quantile and occlusion-site
heads whose outputs the conformal layer above operates on.

## 3. Implementation Progress

**Review 1's own checklist (`stroke_triage_v4_plan.md` §3) is 7/7 complete:**

| # | Deliverable | Done |
|---|---|---|
| 1.1 | v4 proposal document | ✅ |
| 1.2 | Prior-art differentiation table | ✅ |
| 1.3 | Data audit (real data) | ✅ |
| 1.4 | Center-label recovery (real data) | ✅ |
| 1.5 | Synthetic calibration validation | ✅ |
| 1.6 | Baseline segmentation, trained and evaluated | ✅ |
| 1.7 | Occlusion-site + mRS-shift targets (real data) | ✅ |

**Concretely, in code** (`src/`), not as a plan:

| Package | Modules | What it does |
|---|---|---|
| `src/conformal/` | `cps.py`, `decision.py`, `ltt.py` | Conformal predictive system, the triage decision rule, Learn-then-Test certification. Fully implemented, validated on synthetic data (§4 below). |
| `src/data/` | `audit.py`, `center_labels.py`, `occlusion_site.py`, `mrs_shift.py` | ISLES'24 volume/τ-audit, center-label recovery, occlusion-site derivation, mRS-shift target construction. All four **run against the real downloaded dataset** (149 subjects), not simulated. |
| `src/eval/` | `metrics.py` | All four official ISLES'24 metrics + decision metrics (balanced accuracy, MCC), with majority-class baselines. |
| `src/experiments/` | `synthetic.py`, `validate_calibration.py`, `certification_limits.py` | ISLES'24-matched synthetic cohort simulator, the 4-check calibration gate, τ/n certification sweeps. |
| `src/model/` | `dataset.py`, `train_baseline.py` | MONAI data pipeline (real ISLES'24 files) + SegResNet training/evaluation loop. Trained 200 epochs on real data. |

**15 modules across 5 packages, all executed, all producing real output** (see
`outputs/tables/*.json`, `outputs/figures/*.png`) — not stubs, not
pseudocode. What remains for the full project (Review 2, not this checkpoint):
the multi-task H2–H4 heads, running the conformal layer on the trained model's
actual outputs (currently validated on synthetic data only), and the N2/N3
analyses on real model outputs.

## 4. Technical Execution & Code Quality

The pipeline was not run once and trusted — it was stress-tested against the
real archive, and every failure it surfaced was fixed, not worked around:

- **Data integrity.** One subject's perfusion map (`sub-stroke0043`, `cbf`) is
  truncated in the source release itself (confirmed against the archive's own
  manifest — not an extraction error). `src/model/dataset.py` now verifies file
  *readability*, not just existence, and cleanly excludes affected subjects
  rather than crashing a training run.
- **Numerical stability.** Real MTT perfusion maps contain catastrophic
  outliers (one subject's MTT ranges from −54,112 to +48.88, plus literal
  non-finite voxels) — a known CT-perfusion artifact. Left unclipped, this
  caused 29% of training batches (17/59) to produce a NaN loss under mixed
  precision. Fixed with physiologically-motivated intensity clipping before
  normalisation; the rate dropped to 0/59 on the next run.
- **Correctness of the multi-modal loading.** Per-subject NIfTI affines are
  not byte-identical across modalities (~1e-8 drift from independent
  resampling), which broke MONAI's naive multi-file loading; fixed by
  resampling each modality independently before concatenation.
- **Performance.** Naive data loading made one training epoch take 506
  seconds, entirely CPU-bound (0% GPU utilisation). Adding disk caching
  (MONAI `PersistentDataset`) and parallel data loading brought that to 33.5
  seconds/epoch on repeat epochs — a ~15x speedup, taking a full 200-epoch
  run from an impractical ~28 hours to under 2 hours.
- **Validation before trusting results.** The calibration layer was validated
  on synthetic data (4/4 checks passing, §4 of `REVIEW1_RESULTS.md`) *before*
  being used on anything real — exactly the gate `stroke_triage_v4_plan.md`
  §6 specifies: "if this does not hold on synthetic data, the implementation
  is wrong."

Repository hygiene: proper `.gitignore` (excludes the 5GB virtual environment
and binary checkpoints, keeps all real result JSON/PNG output), a top-level
`README.md` with setup and run instructions, one clean initial commit.

## 5. Presentation

- [`stroke_triage_proposal_v4.md`](stroke_triage_proposal_v4.md) — full technical proposal
- [`prior_art_table.md`](prior_art_table.md) — differentiation against 7 adjacent papers
- [`review1_report.html`](review1_report.html) — faculty-facing visual summary
- This document and [`REVIEW1_RESULTS.md`](REVIEW1_RESULTS.md)
