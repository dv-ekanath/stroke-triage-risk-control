# Review 1 — execution guide

Everything here runs on **Python 3.12 with numpy + scipy + matplotlib**, which you
already have. The imaging track additionally needs `nibabel`.

---

## ⚠ Read this first — two blockers

**1. The dataset is not on this machine.** I searched `~`, `/Volumes`, and the whole
filesystem for archives over 1 GB. The only large zips are `models_final.zip`,
`bitcoin-sentiment-analysis-twitter-data.zip`, and similar — nothing stroke-related,
and no ISLES directory anywhere.

**2. You have 1.1 GiB of free disk space** (228 GiB volume, 92% full). ISLES'24 with
4D CTP is tens of GB. You cannot uncompress it until you free space — roughly 60–80 GB
to be comfortable, or use an external drive.

```bash
df -h /                      # confirm free space
du -sh ~/Downloads/* | sort -rh | head -20    # biggest offenders
```

**This does not block Review 1.** The calibration track below needs no data and no GPU,
runs in under a minute, and produces the strongest result you have. Run that first; run
the imaging track whenever the data actually lands.

---

## Part A — the calibration track (no data needed, run this tonight)

### A0. Setup

```bash
cd ~/Documents/Stroke-KG
python3 -c "import numpy, scipy, matplotlib; print('core deps OK')"
```

### A1. Validate the calibration layer — **run this first**

This is the gate. If it fails, nothing downstream can be trusted.

```bash
python3 -m src.experiments.validate_calibration --trials 300 --tau 70
```

Four checks, all of which currently pass:

| Check | What it proves | Current result |
|---|---|---|
| PIT uniformity | the predictive distribution is calibrated | mean 0.494 (want 0.5), sd 0.291 (want 0.289) |
| Interval coverage | empirical tracks nominal across α | max error 1.65 pp |
| **LTT risk control** | realised risk ≤ target on *fresh* draws | within target in **100%** of trials (need ≥90%) |
| Split vs cross | why cross-conformal is not optional | split width **1.8× more variable** |

Outputs `outputs/figures/calibration_validation.png` and
`outputs/tables/calibration_validation.json`.

> The certification rate printed here (~1.5%) looks alarming but is correct: this script
> deliberately holds out half the cohort to measure realised risk, leaving only ~74
> calibration cases. The deployable protocol uses all 149 — see A2.

### A2. Certification limits — **this is your headline result**

```bash
python3 -m src.experiments.certification_limits --trials 100
```

Produces two tables. The first is the one to put on a slide:

```
A. tau sweep at n = 149 (ISLES'24 public set)
   tau    n   n>=tau   cert%   held_m   held_f   abstain
    10  149    108.6    0.0%     --       --       --      <- too few BELOW tau
    20  149     71.2  100.0%   0.990    0.990   0.455
    30  149     49.3  100.0%   0.990    0.980   0.351
    40  149     35.5  100.0%   0.980    0.990   0.285
    50  149     26.4   99.0%   1.000    0.990   0.244
    70  149     16.1   72.0%   0.986    0.917   0.154      <- feasible, marginal
    90  149     10.5   17.0%   1.000    0.941   0.095
   110  149      7.2    0.0%     --       --       --      <- NOT certifiable
   130  149      5.0    0.0%     --       --       --
   150  149      3.6    0.0%     --       --       --
```

Three things to say about it:

1. **There is a feasible window, τ ∈ [20, 90] mL.** Outside it, no guarantee is
   obtainable at n = 149 regardless of how good the model is.
2. **τ = 110 mL cannot be certified** — only ~7 cases sit above it, and the
   Hoeffding–Bentkus floor needs 11 to certify a 0.20 risk at δ = 0.10. This
   *reverses* the τ = 110 recommendation in `stroke_triage_v4_plan.md` §2; the
   evidence-backed therapeutic ceiling is the clinically better threshold but the
   dataset cannot support a guarantee there. Say this explicitly — it is a finding,
   not an inconvenience.
3. **τ = 70 mL is the defensible primary**, certifying in 72% of trials, and
   n ≈ 250 would push that to 99%.

The floor table printed above it is reusable on its own:

```
risk <= 0.05  needs n >=  45 cases in that group
risk <= 0.10  needs n >=  22
risk <= 0.20  needs n >=  11
```

### A3. Regenerate everything

```bash
python3 -m src.experiments.validate_calibration --trials 300 --tau 70
python3 -m src.experiments.certification_limits --trials 100
ls -la outputs/figures outputs/tables
```

---

## Part B — the imaging track (needs the dataset + ~80 GB free)

### B1. Uncompress and inspect the structure

```bash
unzip -l /path/to/isles24.zip | head -40        # inspect WITHOUT extracting
unzip /path/to/isles24.zip -d ~/data/ISLES-2024
find ~/data/ISLES-2024 -maxdepth 3 -type d | head -40
find ~/data/ISLES-2024 -name "*.nii*" | head -20
find ~/data/ISLES-2024 -name "*.tsv" -o -name "*.json" | head -20
```

Expected BIDS-ish shape (the exact tokens have varied between the Zenodo release and
the challenge tarball, which is why the code globs rather than hardcodes):

```
ISLES-2024/
├── dataset_description.json
├── participants.tsv                    <- clinical table, mRS/NIHSS/labs
├── sub-stroke0001/ses-01/
│   ├── anat/  ..._ncct.nii.gz
│   ├── angio/ ..._cta.nii.gz
│   └── perf/  ..._ctp.nii.gz, cbf, cbv, mtt, tmax
└── derivatives/
    └── sub-stroke0001/ses-02/ ..._lesion-msk.nii.gz   <- FINAL INFARCT, the target
```

### B2. Install the imaging dependencies

```bash
python3 -m pip install nibabel scikit-learn pandas
```

### B3. Data audit

```bash
python3 -m src.data.audit --root ~/data/ISLES-2024
```

If it cannot find the masks, inspect the tree and pass the glob explicitly:

```bash
python3 -m src.data.audit --root ~/data/ISLES-2024 \
    --mask-pattern '**/derivatives/**/*lesion*msk*.nii.gz'
```

Add `--lesion-counts` for connected-component counts (slower; needed for ISLES'24's
fourth metric and for the multifocal-disease subgroup).

Cross-check against the descriptor: volume should be **33.16 ± 48.67 mL, range
<0.01–318.25**, lesion count **10.55 ± 11.10, max 84**. A large discrepancy means the
glob picked up the wrong masks.

The audit prints a `certifiable` column per τ, computed against the same LTT floor as
A2 — so it tells you immediately whether the real data supports the threshold you want.

### B4. Center-label recovery

```bash
python3 -m src.data.center_labels --root ~/data/ISLES-2024
```

Tries, in order: an explicit site column in any TSV → scanner vendor fields in JSON
sidecars → (manual) clustering on acquisition parameters.

Expect **Center 1 = 100 (67.1%), Center 2 = 49 (32.9%)**, Siemens and Philips only.
If a GE scanner or a third site appears, the script flags it — verify before repeating
the base paper's 7-center claim in your critique.

---

## Part C — what to put in the Review 1 deck

| Slide | Content | Source |
|---|---|---|
| Problem | Base paper's abstention is a hand-set τ=0.15, ECE=0.130 called "excellent", validated on n=30 | `stroke_triage_v4_plan.md` §2 |
| Gap | Coverage is the wrong guarantee for a threshold decision | plan §Context |
| Method | CPD → calibrated P(V>τ) → LTT-certified asymmetric abstention band | plan §1 N1 |
| **Validation** | 4/4 calibration checks pass; risk holds in 100% of trials | A1 |
| **Headline** | τ-feasibility window; τ=110 not certifiable; n≈250 needed | A2 |
| Honesty | The τ=110 reversal, stated as a finding | A2 |
| Next | Audit on real data, baseline segmentation | B3, plan §3 Review 2 |

The strongest thing you have is that **you found the sample-size limit before spending
any GPU time**, and that it changed the design. That is exactly what a Review 1 is for.

---

## Repository layout

```
src/conformal/
  cps.py        conformal predictive system — calibrated CDF over volume
  decision.py   triage rule, the two asymmetric clinical risks
  ltt.py        Learn-then-Test certification + the sample-size floor
src/experiments/
  synthetic.py               ISLES'24-matched cohort simulator
  validate_calibration.py    the 4-check gate  (A1)
  certification_limits.py    τ and n sweeps    (A2)
src/data/
  audit.py          volume distribution, class balance, certifiability  (B3)
  center_labels.py  center recovery                                     (B4)
src/eval/
  metrics.py    all four ISLES'24 metrics + decision metrics with baselines
outputs/        figures/ and tables/ (JSON), regenerated by the commands above
```

## A note on the method

`certify()` exploits a structural fact worth mentioning in the write-up: the two risks
**decouple**.

```
R_missed = P(p > lam_hi | V <  tau)   depends only on lam_hi
R_futile = P(p < lam_lo | V >= tau)   depends only on lam_lo
```

So the certified region is a product set, each risk is monotone in its own parameter,
and certification reduces to two one-dimensional fixed-sequence walks. No 2-D grid, no
data splitting to learn an ordering, and all n calibration cases contribute — which is
what makes n = 149 workable at all. The generic multi-dimensional LTT machinery is not
needed here, and saying why is a small contribution in itself.
