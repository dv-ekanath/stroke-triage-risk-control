# Review 1 — Results and Metrics

Every number below is read directly from this repository's `outputs/tables/*.json`
files, produced by the scripts named in each section — nothing here is
estimated or recalled from memory. Regenerate any of it with the commands
shown.

---

## 1. Calibration layer — synthetic validation

`python -m src.experiments.validate_calibration --trials 300 --tau 70`
→ [`outputs/tables/calibration_validation.json`](outputs/tables/calibration_validation.json), [`outputs/figures/calibration_validation.png`](outputs/figures/calibration_validation.png)

Cohort: n=149, 300 trials, τ=70mL, α_missed=0.05, α_futile=0.20, δ=0.10.

**Check 1 — PIT uniformity** (want mean 0.5, sd 0.289 for a calibrated CDF):

| Statistic | Value |
|---|---|
| PIT mean | 0.4908 |
| PIT sd | 0.2903 |

**Check 2 — interval coverage across α** (want empirical ≈ nominal):

| α | Nominal coverage | Empirical mean | Empirical sd | Abs. error |
|---|---|---|---|---|
| 0.05 | 0.950 | 0.9613 | 0.0312 | 1.13 pp |
| 0.10 | 0.900 | 0.9064 | 0.0503 | 0.64 pp |
| 0.20 | 0.800 | 0.7957 | 0.0689 | 0.43 pp |
| 0.30 | 0.700 | 0.6842 | 0.0732 | 1.58 pp |

**Check 3 — LTT risk control on fresh draws** (certify on calibration half, measure realised risk on held-out half; want realised ≤ target in ≥90% of trials):

| Risk | n (certified trials) | Mean realised | Target | Within-target rate | p90 |
|---|---|---|---|---|---|
| R_missed | 7 | 0.0135 | 0.05 | **100%** | 0.0378 |
| R_futile | 7 | 0.0615 | 0.20 | **100%** | 0.1321 |

Certified rate across all 300 trials: 2.33% (this is expected and stated in
`guide.md` — this script deliberately holds out half the cohort per trial to
measure realised risk, leaving only ~74 calibration cases; the deployable
protocol uses all 149, see certification-limits below). Mean abstention at
the certified λ: 0.150.

**Check 4 — split vs. cross-conformal width stability** (α=0.10, 5-fold, n_cal≈30 for split):

| Method | Mean width (mL) | SD | Coefficient of variation |
|---|---|---|---|
| Split conformal | 62.32 | 12.82 | 0.2057 |
| Cross-conformal | 58.02 | 6.33 | 0.1091 |

Split-conformal width is **1.89× more variable** than cross-conformal —
empirically justifying the cross-conformal construction used throughout
`src/conformal/`, not just asserting it.

**All 4/4 checks pass.**

---

## 2. Certification limits — synthetic sweeps

`python -m src.experiments.certification_limits --trials 100`
→ [`outputs/tables/certification_limits.json`](outputs/tables/certification_limits.json), [`outputs/figures/certification_limits.png`](outputs/figures/certification_limits.png)

**Sample-size floor** — smallest conditioning-group size that can *ever*
certify a given risk level at δ=0.10, assuming zero observed errors:

| Risk ≤ | Minimum group size n |
|---|---|
| 0.01 | 230 |
| 0.05 | 45 |
| 0.10 | 22 |
| 0.20 | 11 |
| 0.30 | 7 |

**τ-sweep at n=149** (synthetic, ISLES'24-matched cohort):

| τ (mL) | Mean n above τ | Mean n below τ | Certified rate | Risk held (missed) | Risk held (futile) | Mean abstain |
|---|---|---|---|---|---|---|
| 10 | 108.56 | 40.44 | 0% | — | — | — |
| 20 | 71.24 | 77.76 | 100% | 99% | 99% | 0.455 |
| 30 | 49.31 | 99.69 | 100% | 99% | 98% | 0.351 |
| 40 | 35.53 | 113.47 | 100% | 98% | 99% | 0.285 |
| 50 | 26.44 | 122.56 | 99% | 100% | 99% | 0.244 |
| 70 | 16.09 | 132.91 | 72% | 98.6% | 91.7% | 0.154 |
| 90 | 10.52 | 138.48 | 17% | 100% | 94.1% | 0.095 |
| 110 | 7.15 | 141.85 | **0%** | — | — | — |
| 130 | 5.03 | 143.97 | 0% | — | — | — |
| 150 | 3.60 | 145.40 | 0% | — | — | — |

On synthetic data alone, the feasible window is τ∈[20,90]mL; τ=110 does not
certify. **Section 3 below shows this changes on the real masks.**

---

## 3. Data audit — real ISLES'24 masks (149 subjects)

`python -m src.data.audit --root <ISLES-2024>/train --lesion-counts`
→ [`outputs/tables/data_audit.json`](outputs/tables/data_audit.json), [`outputs/figures/data_audit.png`](outputs/figures/data_audit.png)

**Volume distribution:**

| Statistic | Value | Descriptor reference |
|---|---|---|
| n | 149 | 149 |
| Mean | 31.69 mL | 33.16 mL |
| SD | 45.14 mL | 48.67 mL |
| Median | 12.43 mL | — |
| Min / Max | 0.00 / 222.13 mL | <0.01 / 318.25 mL |
| Q1 / Q3 | 3.89 / 38.64 mL | — |
| Cases with ~zero infarct | 3 | — |

**τ-certifiability table (real masks)** — floor_missed=45, floor_futile=11 (δ=0.10):

| τ (mL) | n below | n above | Majority-baseline acc. | Certifiable? |
|---|---|---|---|---|
| 10 | 66 | 83 | 0.557 | ✅ |
| 20 | 96 | 53 | 0.644 | ✅ |
| 30 | 106 | 43 | 0.711 | ✅ |
| 40 | 112 | 37 | 0.752 | ✅ |
| 50 | 119 | 30 | 0.799 | ✅ |
| 60 | 124 | 25 | 0.832 | ✅ |
| 70 | 127 | 22 | 0.852 | ✅ |
| 80 | 130 | 19 | 0.872 | ✅ |
| 90 | 131 | 18 | 0.879 | ✅ |
| 100 | 134 | 15 | 0.899 | ✅ |
| **110** | **137** | **12** | 0.919 | ✅ (margin of **1 case** above the floor of 11) |
| 130 | 141 | 8 | 0.946 | ❌ (below R_futile floor) |
| 150 | 144 | 5 | 0.966 | ❌ |

**This is the key real-data finding that updates the synthetic-only estimate
in §2:** τ=110mL clears the necessary sample-size floor on the real masks,
where the synthetic sweep said it wouldn't. This is a *necessary, not
sufficient* condition — actual certification still needs a trained model's
calibrated probabilities — but the data itself no longer rules τ=110 out.

---

## 4. Center-label recovery — real data

`python -m src.data.center_labels --root <ISLES-2024>/train`
→ [`outputs/tables/center_labels.json`](outputs/tables/center_labels.json)

| | Count | Percentage | Descriptor reference |
|---|---|---|---|
| Center-1 | 99 | 66.9% | 100 (67.1%) |
| Center-2 | 49 | 33.1% | 49 (32.9%) |

Source: `phenotype-csv:center` (this release ships no `participants.tsv`; the
label was recovered from an explicit `Center` field in each subject's own
phenotype CSV). One subject (`sub-stroke0075`) has a genuinely blank field in
its own source file, accounting for the full gap to the descriptor's 100.

---

## 5. Occlusion-site derivation — real data

`python -m src.data.occlusion_site --root <ISLES-2024>/train`
→ [`outputs/tables/occlusion_site.json`](outputs/tables/occlusion_site.json)

149 subjects total; 144 processed (5 skipped: 1 real shape mismatch between
that subject's `cow-msk` and `lvo-msk`, 4 subjects with an entirely empty
`lvo-msk`).

| Site | Count | % of processed |
|---|---|---|
| M1 | 71 | 49.3% |
| Other (low-confidence) | 42 | 29.2% |
| ICA | 16 | 11.1% |
| Other | 15 | 10.4% |

Confidence threshold: 15mm nearest-distance. **102/144 (70.8%)** confident
matches; the 42 low-confidence subjects are flagged for manual spot-check, not
silently assigned.

---

## 6. mRS-shift target construction — real data

`python -m src.data.mrs_shift --root <ISLES-2024>/train`
→ [`outputs/tables/mrs_shift.json`](outputs/tables/mrs_shift.json)

| Quantity | Value |
|---|---|
| Complete premorbid+3-month pairs | 80 / 149 (53.7%) |
| mRS premorbid, mean ± SD | 0.84 ± 1.23 (v3 reference: 0.9 ± 1.2) |
| mRS 3-month, mean ± SD | 2.64 ± 2.17 |
| mRS shift, mean ± SD | 1.80 ± 1.90 |
| Good outcome rate (mRS≤2 at 3mo) | 51.25% |
| mTICI recorded, whole cohort | 100 / 149 (v3 reference: 97/149) |

---

## 7. Baseline segmentation — real data, the headline result

`python -m src.model.train_baseline --root <ISLES-2024>/train --fold 0 --epochs 200 --val-interval 10 --patch-size 128 128 64 --batch-size 2 --cache-dir <cache> --num-workers 4`
→ [`outputs/tables/baseline_fold0.json`](outputs/tables/baseline_fold0.json)

Model: 6-channel SegResNet (NCCT, CTA, CBF, CBV, MTT, Tmax). 200 epochs, fold
0 of a 5-fold volume-tercile-stratified split, 30 held-out real subjects.

| Metric | This baseline | Published leaderboard (Kurtlab) | Result |
|---|---|---|---|
| Dice | 0.2154 | 0.285 | below |
| Absolute volume difference | 21.34 mL | 21.23 mL | **matches** |
| Lesion-wise F1 | 0.2784 | 0.144 | **exceeds** |
| Absolute lesion count difference | 6.57 | 7.18 | **exceeds (lower is better)** |

Best validation Dice observed during training: 0.2263 (epoch 110). Loss
declined steadily from 0.878 (epoch 1) to 0.369 (epoch 200) with no
divergence. Training time: ~34 seconds/epoch after the first (cached) epoch,
~1.9 hours total for the full 200-epoch run — a ~15x speedup over the
uncached pipeline (506 seconds/epoch), which is what made a real 200-epoch
run practical on the available hardware (RTX 5060, 8GB VRAM).

Per-subject Dice on the 30 held-out subjects ranges from 0.000 (several
small/near-empty lesions) to 0.601 (`sub-stroke0095`) — consistent with the
known difficulty of scattered multifocal small-lesion disease across the
ISLES'24 literature, not specific to this run.
