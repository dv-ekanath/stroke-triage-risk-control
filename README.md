# Risk-Controlled Threshold Triage for Ischemic Stroke

A distribution-free guarantee on the thrombectomy-eligibility *decision* — not on
the volume estimate feeding it. Extends Luan et al., [*Unified multimodal
learning for stroke triage*](https://doi.org/10.1038/s41746-025-02255-0),
npj Digital Medicine 9:441 (2026).

The base paper's safety mechanism is a hand-set rejection threshold (τ = 0.15),
called "excellent calibration" at ECE = 0.130 and validated on 30 patients. This
project replaces it with a Learn-then-Test certified decision rule carrying a
finite-sample guarantee on two clinically asymmetric risks — denying treatment
to a patient who would benefit, vs. futile reperfusion.

**Full write-up:** [`stroke_triage_proposal_v4.md`](stroke_triage_proposal_v4.md) ·
prior-art differentiation: [`prior_art_table.md`](prior_art_table.md) ·
review-by-review plan: [`stroke_triage_v4_plan.md`](stroke_triage_v4_plan.md)

## Status

Review 1 (problem framing, literature positioning, novelty statement, methodology
design) is complete — see [`stroke_triage_v4_plan.md`](stroke_triage_v4_plan.md)
§3 for the full checklist. Highlights, all run against the real downloaded
ISLES'24 release (149 subjects), not synthetic data unless noted:

- **Center-label recovery**: 99/49 split, matching the published descriptor
  (100/49) — recovered from a per-subject phenotype field, since this release
  ships no `participants.tsv`.
- **Data audit**: real volume distribution (31.7 ± 45.1 mL) and τ-certifiability
  table; τ=110mL clears the Learn-then-Test sample-size floor by a margin of one
  case, updating an earlier synthetic-only estimate that had called it infeasible.
- **Occlusion-site derivation**: nearest-CoW-distance method (direct mask
  intersection mostly fails on real data), geometrically verified, 71/144
  confident matches.
- **Baseline segmentation** (`src/model/`): 6-channel SegResNet, 200 epochs,
  real leaderboard-comparable Dice/AVD/lesion-F1/ALD (`outputs/tables/baseline_fold0.json`).
- **Calibration layer** (`src/conformal/`): validated on synthetic data — PIT
  uniformity, coverage across α, LTT risk control at 100% within-target, and the
  split-vs-cross-conformal width-instability comparison all pass.

A presentation-ready summary of Review 1 is in [`review1_report.html`](review1_report.html).

## Repository layout

```
src/conformal/     conformal predictive system, the triage decision rule, Learn-then-Test
src/data/          ISLES'24 audit, center-label recovery, occlusion-site + mRS-shift derivation
src/eval/          all four ISLES'24 metrics + decision metrics, with baselines
src/experiments/   synthetic cohort simulator, calibration validation, certification-limits sweeps
src/model/         baseline segmentation: dataset pipeline (MONAI) + SegResNet training loop
outputs/           figures/ and tables/ (JSON) written by the scripts above
```

## Setup

The dataset itself is **not** in this repo — see [`guide.md`](guide.md) for the
expected ISLES'24 directory layout. Everything below assumes it's downloaded
separately and passed via `--root`.

**Calibration + data-audit track** (no GPU, no dataset needed for calibration):

```bash
pip install -r requirements.txt
python -m src.experiments.validate_calibration --trials 300 --tau 70
python -m src.experiments.certification_limits --trials 100
python -m src.data.audit --root /path/to/ISLES-2024/train
python -m src.data.center_labels --root /path/to/ISLES-2024/train
python -m src.data.occlusion_site --root /path/to/ISLES-2024/train
python -m src.data.mrs_shift --root /path/to/ISLES-2024/train
```

**Model-training track** (needs a CUDA GPU — developed against an RTX 5060 /
Blackwell, which needs a cu128+ PyTorch build):

```bash
python -m venv .venv
.venv/Scripts/activate            # .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"   # confirm before training

python -m src.model.train_baseline --root /path/to/ISLES-2024/train \
    --fold 0 --epochs 200 --val-interval 10 \
    --cache-dir /path/to/cache_1mm --num-workers 4
```

`--cache-dir` matters: it switches to MONAI's `PersistentDataset`, caching the
resample/normalize pipeline to disk so only the first epoch pays the full
preprocessing cost (measured ~15x speedup on repeat epochs). Clear the cache
directory if you change `src/model/dataset.py`'s transforms.

## Reference

Riedel E.O. et al. The ISLES'24 Dataset: A Multimodal Stroke Imaging Dataset
with Hyperacute CT, Acute Postinterventional MRI, and 3-month Clinical Outcomes.
*Radiology: Artificial Intelligence* 8(3) (2026). DOI 10.1148/ryai.250603.
