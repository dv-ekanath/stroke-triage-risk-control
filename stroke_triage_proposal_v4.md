# Risk-Controlled Threshold Triage for Ischemic Stroke: A Distribution-Free Guarantee on the Decision, Not the Volume

### Where conformal coverage guarantees fail the decision they are meant to support, and how Learn-then-Test fixes it at the guideline threshold

**Author:** Rohan Julius Preetan A
**Course:** Healthcare Analytics
**Version:** 4 (supersedes v3)

---

## 0. What changed from v3, and why

v3 was well-verified but carried a novelty problem it diagnosed honestly rather than solved: **N1b (distance-to-threshold Mondrian calibration) is textbook** — Boström & Johansson (COPA 2020) already do exactly this, and Jaubert et al. (ISBI 2025) already do cluster-conditional conformal calibration on volume. What v3 called a contribution reduced, on inspection, to the choice of binning variable. Two further v3 claims did not survive scrutiny: N3 (cross-center transfer) is underpowered at n=49 for the second center (binomial SE on coverage ≈ 4.3pp, a 95% CI spanning ±8.4pp — not enough to detect the "two or three percentage points" §4 asked for), and **coverage itself is the wrong guarantee for a threshold decision** — a triage rule only cares which side of τ the truth falls on, and perfect near-boundary coverage with wide intervals is consistent with a rule that abstains on everything.

v4 replaces "cover the volume, then read off a decision" with a direct, distribution-free guarantee on the decision itself, using Learn-then-Test (Angelopoulos, Bates et al., *Ann. Appl. Stat.* 2025) to certify two clinically asymmetric risks simultaneously. This is not a bigger version of v3's contribution — it is a different primary claim, built on a structural fact about this decision rule (§5.3) that makes the general multi-dimensional LTT machinery unnecessary and lets all n=149 calibration cases contribute to certification, which is what makes anything workable at this sample size at all.

**Also new in v4: this is not only a plan.** The calibration layer (`src/conformal/`) is implemented and validated on synthetic data — PIT uniformity, interval coverage across α, LTT risk control holding in 100% of trials, and the split-vs-cross-conformal width-instability comparison all pass (§9). The Week-1 data audit, center-label recovery, occlusion-site derivation, and mRS-shift target construction have been run against the **real, downloaded ISLES'24 release** (not simulated), and their results are folded into §2 and §7 below rather than kept as separate deliverables. A baseline segmentation model (§5.1) is training against the real data as this document is being written.

Sections 2 (data), 5.1 (architecture) and 5.2 (losses) carry over from v3 nearly unchanged — the restructure is in the framing, the claims, and the calibration layer, not the model. Section 1 (base-paper critique) is substantially rewritten: three objections stronger than v3's Dice-discrepancy framing were found in the base paper's full text and are the actual motivation for this project.

---

## 1. The base paper's real weakness

Confirmed against the full text of Luan et al. (*npj Digit. Med.* 9:441, 2026), DOI 10.1038/s41746-025-02255-0.

**1.1 The stated generalisation protocol cannot have been run as described.** The paper reports leave-one-center-out validation "across 7 medical centers with different vendors (GE, Siemens, Philips)" on the ISLES'24 public training set (n=149). The dataset's own descriptor (Riedel et al., *Radiology: AI* 8(3), 2026) — and, independently, this project's own recovery of the center labels directly from the release (§2) — puts that set at **two centers**, Siemens and Philips fleets only, no GE. Their headline result (per-center Dice 0.81 ± 0.03, range 0.78–0.82, CV reduced 50–55%) cannot have been produced by the protocol as described. This needs no interpretation to land.

**1.2 The Dice gap to the challenge leaderboard is a task-definition mismatch, not a performance gap.** The ISLES'24 hidden-test leaderboard tops out at Dice 28.50 ± 21.27 (Kurtlab); Luan et al. report Dice 0.81 on nominally the same dataset. Their stated target is "ischemic core and penumbra" referenced to follow-up MRI or DSA — not ISLES'24's actual label, the final infarct on day 2–9 DWI. They additionally train on CPAISD (n=112, hyperacute NCCT core/penumbra) and AISD (n=397), neither of which is ISLES'24. The three-fold Dice gap is very likely attribution drift between cohorts, not a demonstrated 3x improvement over the challenge winner, and this project does not use their number as a benchmark.

**1.3 Their uncertainty layer is the actual hole — and is this project's motivation.** ECE = 0.130 is reported and described as "excellent calibration" (13 points of miscalibration by the metric's own definition). Abstention is a **hand-set rejection threshold τ = 0.15**, justified post hoc on 18/122 flagged CTP scans. Their clinical safety endpoint — unnecessary transfer reduced from 18.9% to 12.3% — is computed on **n = 30** borderline cases against an unmatched historical registry rate of 27%. Their own ablation states that disabling rejection moves mis-triage from 5% to 9%: the entire safety story rests on one unvalidated hyperparameter, evaluated on 30 patients.

> **This is v4's motivating sentence:** *the deployed system's abstention threshold is an unvalidated hyperparameter evaluated on 30 patients; this project produces one with a distribution-free, finite-sample guarantee on the clinically asymmetric decision error instead.*

**1.4 Their eligibility rule is anchored to superseded evidence.** They gate on DAWN/DEFUSE-3 core <70mL + mismatch >1.8. SELECT2, ANGEL-ASPECT and RESCUE-Japan LIMIT have since established thrombectomy benefit in large-core patients, and the 2026 AHA/ASA guideline (DOI 10.1161/STR.0000000000000513) makes EVT in large-core stroke Class 1A in both early and late windows. A nationwide Korean registry (PMID 42403349) identifies a ≥110mL therapeutic ceiling as the evidence-backed threshold where benefit becomes negligible; OUTER LIMITS is currently randomising core >70mL. There is no longer a single correct τ (§2.3), and this project treats that as the setting, not a nuisance.

---

## 2. What the data actually contains

Verified against the Radiology: AI descriptor (Riedel et al., 2026), the challenge report, **and, where noted, this project's own audit of the real downloaded release** (ISLES'24 public training set, obtained as `train.7z`, 149 subjects, BIDS-derivative layout).

**Public training set:** n = 149, CC BY-NC-SA 4.0, Zenodo. Hidden test n = 96 (the descriptor's number is authoritative for the released dataset; the challenge report's 100 and one abstract's 98 are not used). Total 245.

| Item | Status | Use in v4 |
|---|---|---|
| NCCT, CTA, CBF, CBV, MTT, Tmax (derived perfusion maps) | Available, real-data audited | Encoder inputs (raw 4D CTP deliberately not used — its frame count varies 44–60 across subjects, and v3 already specifies the derived maps in its place) |
| Final infarct mask (DWI, day 2–9) | Available, real-data audited | Segmentation target, volume target |
| CTA vessel occlusion masks (`lvo-msk`) | Available, manual, neuroradiologist-reviewed | Site-label derivation |
| Circle of Willis segmentation (`cow-msk`), per-vessel labels | Available, model pseudolabels (TopCoW-derived) | Site-label derivation |
| Stroke pattern class, anatomic location | Available | Group-conditional conformal strata |
| mRS: premorbid, 3-month; NIHSS; mTICI | Available, per-subject phenotype CSVs (no participants.tsv in this release) | mRS-shift target, tabular branch |
| Center label | **Recovered from real data** (see below) | Cross-center conformal transfer, robustness check |

### 2.1 Real-data findings that update the numbers below

Run against the actual downloaded release, not simulated:

- **Center recovery.** This release ships no `participants.tsv` and no JSON sidecars; center is not recoverable by either of v3's assumed strategies. It ships instead in a per-subject phenotype CSV with an explicit `Center` column. Recovered split: **Center-1 99 (66.9%), Center-2 49 (33.1%)**, one subject (sub-stroke0075) with a genuinely blank field in its own source file. Matches the descriptor's 100/49 (67.1%/32.9%) almost exactly. **N3's gate is cleared with real, traceable labels, not inferred ones.**
- **Volume distribution** (n=149, real masks): mean 31.69 ± 45.14 mL, median 12.43, range 0.00–222.13 — close to the descriptor's 33.16 ± 48.67 mL, confirming the correct masks. Lesion count mean 13.58 ± 14.75 (max 104) runs higher than the descriptor's 10.55 ± 11.10 (max 84); most likely a connected-component connectivity-convention difference in this project's own counting, not a data defect, and is flagged rather than silently reported.
- **τ-feasibility on real data, not synthetic.** Every τ ∈ {10,...,110} mL clears the Learn-then-Test group-size floor on the real masks — **including τ=110mL**, which guide.md's earlier synthetic-only sweep had called uncertifiable. It clears by a margin of exactly one case (12 above vs. a floor of 11) — a necessary-but-not-sufficient signal (real model calibration near the boundary still has to hold), reported here as a finding rather than a settled result.
- **Occlusion-site derivation (§7 risk, "spot-check a subset").** v3 called intersecting `lvo-msk` with `cow-msk` "near-trivial." On real data, direct voxel intersection is empty in most subjects — both are sparse, independently-drawn annotations that sit near, not on top of, each other. Nearest-distance from the LVO centroid to each CoW label works instead, and was verified geometrically (left/right label pairing checked against world-coordinate sign, consistent across every spot-checked subject) before being trusted. Result: 71/144 (49.3%) M1, 16/144 (11.1%) ICA, 42/144 (29.2%) flagged low-confidence (nearest label >15mm away — likely distal M2 occlusions the CoW segmentation was never going to reach), the remainder other. The low-confidence set is flagged for manual review, not silently assigned.
- **mRS-shift target.** Complete premorbid+3-month pairs exist for only **80/149 (54%)** subjects — H4's effective training set is roughly half the cohort, a real constraint carried into §7. mRS premorbid 0.84 ± 1.23 (v3 reference 0.9 ± 1.2 — matches); mTICI recorded 100/149 (v3 reference 97/149 — matches, after correcting for this release's use of the literal string `"nan"` for missing values in that one field).

### 2.2 The volume distribution decides the primary metric

At any plausible τ this remains an imbalanced binary problem — a constant "below cutoff" predictor scores in the 65–90% range on plain accuracy depending on τ (see the real per-τ table in §2.1). Balanced accuracy and MCC are the only honest primaries, majority-class baseline reported alongside every decision metric (§6).

### 2.3 The estimand, restated

Every patient in ISLES'24 underwent successful reperfusion (mTICI 2c/3 predominant). The target is **tissue outcome given successful reperfusion**, not final infarct under unknown treatment — closer to what an eligibility decision is actually anticipating (what does the brain look like if you succeed) than a literal admission-core proxy. Two consequences, stated as limitations rather than hidden: no counterfactual/untreated trajectory can be estimated, and the cohort is conditioned on having been treated and on treatment succeeding, a selection effect on top of the LVO inclusion criterion. Methodology transfers to admission-core data; the numbers do not.

### 2.4 The 70mL cutoff is contested, and that is useful

No single correct τ exists (§1.4). v4 treats the model's decision behaviour across the plausible threshold range as the object of study (§4, N1's by-product), not its accuracy at one legacy number.

---

## 3. The gap, restated

Conformal volume intervals exist. Conformal abstention in clinical triage exists. Neither literature has asked whether the guarantee holds **at the decision itself**.

Marginal conformal coverage — P(V ∈ Γ(X)) ≥ 1−α — says nothing about which side of τ a given patient falls on. A patient at 15mL and a patient at 68mL do not present the same decision problem, and covering the bulk of a right-skewed cohort (mean 33mL) well is the cheapest way to satisfy a marginal guarantee, while boundary cases — the ones a threshold decision actually turns on — are a minority and harder. Worse: a rule can achieve excellent coverage by abstaining on everything, or achieve poor coverage while getting every decision right, because coverage and decision correctness are different objects. Reporting marginal coverage on a threshold task measures the wrong thing and calls it validation.

Nobody has replaced the coverage guarantee with a direct guarantee on the triage decision's own error, at a clinically motivated asymmetry, with a finite-sample certificate. That is what N1 does.

---

## 4. Novelty claims

Three claims, each independently reportable — down from v3's four, because coverage of a decoupled fact is not the same as certifying the decision, and because two of v3's four did not survive verification (§0).

### N1 (primary) — Risk-controlled threshold triage

**Step 1 — conformal predictive distribution.** A cross-conformal predictive system (Vovk et al. 2017/2019; Vovk & Manokhin 2018), `src/conformal/cps.py`, emits a calibrated CDF F̂ᵢ(v) per patient rather than an interval at one fixed α, normalised by the H3 quantile-head spread as a per-case difficulty estimate. Cross-conformal aggregation over the 5 CV folds solves the n≈24-per-fold split-conformal instability documented in v3 §1.5 (confirmed again in this project's own synthetic validation, §9: split-conformal width is measurably more variable than cross-conformal at this sample size). This yields pᵢ = 1 − F̂ᵢ(τ), a calibrated P(V > τ) **for every τ simultaneously** — v3's old N4 threshold sweep becomes a free by-product.

**Step 2 — certified decision rule.** Triage is parameterised by λ = (λ_lo, λ_hi), 0 ≤ λ_lo ≤ λ_hi ≤ 1 (`src/conformal/decision.py`):

```
p < λ_lo            -> ELIGIBLE     (asserts V < τ)
p > λ_hi            -> INELIGIBLE   (asserts V >= τ)
λ_lo <= p <= λ_hi    -> ABSTAIN      (refer to clinician)
```

Two clinical risks are certified simultaneously with Learn-then-Test (`src/conformal/ltt.py`):

- `R_missed = P(system says INELIGIBLE | true V < τ)` — denying thrombectomy to a patient who would benefit
- `R_futile = P(system says ELIGIBLE | true V >= τ)` — futile reperfusion, procedural and haemorrhage risk for no tissue benefit

**The asymmetry is set by the guideline, not by taste**, per §1.4: EVT in large-core stroke is now Class 1A in both windows, so denying treatment is the worse error, α_missed ≪ α_futile. Among certified λ, the abstention-minimising one is selected; the whole certified region is reported, not just the selected point.

**Why the general multi-dimensional LTT machinery is not needed here — and why that is itself worth saying.** The two risks decouple structurally: R_missed depends only on λ_hi, R_futile only on λ_lo. The certified region is therefore a product set, each risk is monotone in its own parameter, and certification reduces to two independent one-dimensional fixed-sequence walks (`certify()`, method `monotone_fixed_sequence`) rather than a dense grid search with Bonferroni correction — which this project's own implementation confirms certifies *nothing* at n=149 (kept as the ablation that motivates the better method, §7). No data is held out to learn an ordering, so all n calibration cases contribute — the difference between certifying something and certifying nothing at this sample size.

**Why this is novel.** Conformal risk control / LTT has been applied to FNR and IoU on segmentation masks, and cost-aware deferral to classification triage (Sci Rep 2026, DOI 10.1038/s41598-026-40637-w; cost-sensitive CP benchmark, arXiv 2607.27143). Nobody has applied risk control to a **guideline threshold on a segmentation-derived continuous quantity with two-sided asymmetric clinical costs**. It also makes the abstention band a *certified* object — precisely what the base paper's τ=0.15 is not (§1.3).

**Retained diagnostic (cheap, reported either way).** Marginal split-conformal under-coverage in the near-boundary band, stratified by |V−τ|, is still measured. A null result (boundary coverage holds) is a real finding and would vindicate current practice on this dataset; the Mondrian *correction* v3 proposed for it is dropped as method (it is Boström & Johansson, COPA 2020, applied to a new conditioning variable — the contribution reduces to the choice of variable, and a reviewer who knows that paper says so in one line).

### N2 (secondary) — Decision-analytic validation with a deferral arm

Decision curve analysis (Vickers), extended with a third action (defer) that standard net benefit has no room for:

```
NB(p_t) = TP/n − (FP/n)·(p_t/(1−p_t)) − (n_deferred/n)·c_defer
```

The threshold probability p_t **is** the harm ratio — the exact quantity §1.4 shows is contested in large-core stroke. Sweeping it and comparing point-estimate thresholding, marginal conformal, and the N1 policy against treat-all/treat-none answers "there is no single correct τ" (§2.4) with the tool clinical journals actually read, rather than a sweep plot. The mRS-shift head (retained from v3 §5.1, unchanged) is given a load-bearing job here: estimating the harm ratio empirically from predicted outcome shift rather than asserting it — the difference between decorative multi-task regularisation and a component the claim depends on. DCA and conformal abstention have not been formally connected; the deferral-cost term is the small, real piece of methodology here.

### N3 (secondary) — What n=149 can actually certify

Two parts, both turning a limitation into a reported contribution rather than an excuse.

**Certification limits.** Realised coverage from n calibration points is Beta-distributed; at n=149, α=0.10 spreads roughly ±5pp. This project's own synthetic sweep (`src/experiments/certification_limits.py`, §9) already shows the shape of this on matched-moment synthetic cohorts: a feasible τ window (τ∈[20,90]mL certifies in 100% of synthetic trials; τ=110 was the boundary case), and a floor table (risk≤0.05 needs n≥45 in the conditioning group; risk≤0.20 needs n≥11). **The real-data audit (§2.1) updates this**: on the actual masks, τ=110 clears the necessary group-size floor, by one case — the synthetic-only headline number undersold what the real, right-skewed-but-not-perfectly-lognormal cohort supports. Reporting the discrepancy between synthetic and real feasibility is itself part of N3.

**Irreducible-error floor.** Upgrades v3's old N2 (an annotation-floor normalisation convention) into a bound, using label-noise-robust conformal (Einbinder et al., JMLR 25; Penso & Goldberger, MLMI@MICCAI 2024) calibrated to the reported inter-rater volume disagreement (2.37 ± 2.59mL and 6.56 ± 13.37mL — provenance note: these are ISLES'22 pipeline-validation numbers, not a fresh ISLES'24 rater study, stated whenever cited). This derives the minimum achievable decision error near τ for *any* model and shows where the system sits relative to it, run as a sensitivity analysis across plausible σ.

**If LTT ultimately certifies nothing at the sample sizes available, N3 is the finding, not the failure**: here is the sample size a stroke-triage system needs before it can make a guaranteed decision, and nobody currently has it.

### Not claimed as novel

Conformal prediction, conformalized quantile regression, Mondrian/group-conditional conformal, jackknife+/CV+ as methods; conformal predictive intervals on segmentation-derived volume (Lambert et al., MICCAI 2024; Jaubert et al., ISBI 2025); weighted conformal prediction for covariate shift including latent-space density-ratio estimation (Lambert et al.); conformal sets feeding a deferral rule via a risk-coverage curve in clinical triage (Sci Rep 2026); differentiable size/volume constraints in segmentation losses (Kervadec 2019); selective prediction and reject-option learning; multimodal fusion of the CT trilogy; multi-task learning as regularisation; knowledge-graph encoding via GAT with cross-attention injection (considered and dropped — see §5.1 note).

---

## 5. Method

### 5.1 Architecture

Deliberately conventional (carried over from v3 nearly unchanged). The ISLES'24 challenge showed transformer variants gave no advantage at this sample size, and this project's contribution is the calibration layer, not the backbone.

```
NCCT ─────┐
CTA ──────┤  per-modality 3D encoders
CBF/CBV/  ┤
MTT/Tmax  ┤           ↓
          │   cross-modal fusion                  ┌─► H1  final infarct segmentation
Tabular ──┴─► MLP ─► FiLM conditioning ─►          ├─► H2  occlusion site (ICA/M1/M2/other)
                                           trunk ───┼─► H3  volume quantile head (τ_lo, τ_hi)
                                                     └─► H4  3-month mRS shift (ordinal, CORAL)
```

nnU-Net-class encoder-decoder trunk. H3 predicts volume quantiles directly; volume by integration of H1's soft mask is retained as an ablation. H4 predicts **mRS shift** (3-month minus premorbid), not absolute 3-month mRS — premorbid mRS averages 0.84 ± 1.23 in the real cohort (§2.1) and is a strong determinant of the absolute outcome, so predicting the absolute score would let the model score well by learning baseline disability from tabular data alone. Absolute mRS is reported alongside for comparability with the outcome-prediction literature. **Real-data constraint carried forward from §2.1:** only 80/149 (54%) subjects have a complete premorbid+3-month pair; H4's loss is computed on that subset, and this is reported as a limitation, not smoothed over.

**Review 1 baseline vs. Review 2 full model.** The item-1.6 deliverable is H1 alone — a 6-channel (NCCT, CTA, CBF, CBV, MTT, Tmax) SegResNet trunk, trained and evaluated against all four ISLES'24 metrics, comparable to the published leaderboard. The full H1–H4 multi-task architecture above is Review 2's deliverable, built on top of a validated baseline rather than in parallel with it.

**On the knowledge graph.** v2/early v3 considered a knowledge-graph-guided consistency module (GAT encoder over a clinical decision graph, cross-attention injection, a differentiable consistency loss). Dropped as method: occlusion site is upstream of and correlated with volume, so a graph-based consistency check partly duplicates the conditioning already present in the H2 head and the Mondrian-style stratification; under LTT, per-group certification multiplies hypotheses and costs power directly, which this sample size cannot absorb. The *descriptive* half survives independently of the graph: a table of which guideline-asserted relations are and are not supported by training-fold empirical co-occurrence is a one-day, standalone finding (§6), not a trained module.

### 5.2 Losses

Carried over from v3 unchanged:

```
L = λ1·L_seg + λ2·L_site + λ3·L_mRS + λ4·L_pinball + λ5·L_thr
```

`L_seg` Dice+CE on the final infarct mask; `L_site` cross-entropy on CoW-derived occlusion-site labels (§2.1: 71/144 real subjects land in the "confident" bucket at ≤15mm nearest-distance — the 42 flagged low-confidence cases are a real, reported constraint on this head's label quality); `L_mRS` CORAL ordinal loss on mRS shift; `L_pinball` on H3 at α/2, 1−α/2; `L_thr` the threshold-margin penalty from Kervadec's differentiable inequality-penalty family, applied to a guideline cutoff rather than an annotation prior:

```
L_thr = 1[V<τ]·ReLU(V̂−τ+m)²  +  1[V>=τ]·ReLU(τ+m−V̂)²
```

**A tension worth measuring, not hidden.** `L_thr` pushes predictions away from τ; the threshold-conditional layer needs a well-populated conformity-score distribution *at* τ. Training to evacuate the boundary region degrades the signal the calibration layer depends on there. Quantified directly: threshold-conditional coverage and boundary-band width, with and without `L_thr`, at matched Dice (§7 ablations).

### 5.3 Conformal decision layer — implemented and synthetic-validated (`src/conformal/`)

**Construction.** Cross-conformal over the 5 outer folds (`CrossConformalPredictiveSystem.fit_from_folds`), not a single 80/20 split — pools out-of-fold residuals so all n=149 cases contribute conformity scores instead of ≈24 per fold. Split conformal is retained as an ablation specifically to reproduce the width instability this design choice avoids (§9 already measures it: split-conformal width has a measurably higher coefficient of variation than cross-conformal at matched α).

**Scores.** Standardised residual `(V − V̂)/σ`, σ from the H3 quantile-head spread when `normalise=True`; the conformal correction `below/(n+1)` is applied throughout, not `below/n` — pretending the CDF resolves finer than the calibration set allows is exactly the overconfidence this project is about.

**Certification (N1 step 2).** `certify()` implements the two-risk, decoupled, monotone-fixed-sequence construction described in §4/N1: independent walks for λ_hi (R_missed) and λ_lo (R_futile), δ split between them, Hoeffding–Bentkus p-values (`hb_p_value`) against each conditioning group's own size, `min_n_to_certify()` giving the hard sample-size floor a conditioning group must clear before *any* rule can be certified there — this is exactly the function the real-data τ-audit in §2.1 checks against.

**Decision.** `decision.py` implements the λ-parameterised rule and its two conditional risks exactly as specified in §4, with the explicit design note that both risks are conditional on a subgroup (V<τ or V≥τ), not on n — treating them as n-scale is "the single easiest way to certify something that is not true," and `ltt.py` handles the conditioning explicitly throughout.

### 5.4 Explanation layer

Deterministic template filling. No LLM, no API call. Given head outputs plus the certified decision, emits predicted values, the interval, its position relative to τ, occlusion site, predicted mRS shift, and — when the interval straddles — the referral reason and whether the case falls inside the annotation-reliability floor (N3). Presentation, not a claim.

---

## 6. Metrics

**Primary, decision level.** Majority-class baseline reported alongside every one, every time (§2.2 shows why plain accuracy is not reportable on its own).

- Balanced accuracy and MCC at τ
- Threshold-crossing error rate, overall and within the innermost |V−τ| band
- **R_missed, R_futile** at the certified λ, against nominal α, with the LTT confidence level δ stated
- Abstention rate, and abstention-band width relative to the annotation-reliability floor (N3)
- Risk-coverage curve and its area, marginal and over the boundary band
- Certified-region size (how much of the λ grid survives, not just the selected point)

**N2 (decision-analytic).** Net benefit with the deferral term, swept over the harm-ratio range, all policies vs. treat-all/treat-none; harm ratio also estimated empirically from the H4 head.

**N3 (certification limits).** Minimum-n-per-risk-level table (already produced on synthetic cohorts, §9); real-data τ-feasibility table (§2.1, already produced); irreducible-error floor from annotation noise, sensitivity across σ.

**Group-conditional.** Coverage and decision error by stroke pattern (single-vessel/scattered/mixed) and, cell counts permitting, by the real occlusion-site buckets from §2.1 (ICA/M1/confident vs. low-confidence).

**Secondary, task level.** All four ISLES'24 metrics (Dice, absolute volume difference, lesion-wise F1, absolute lesion count difference — `src/eval/metrics.py`, matching the published leaderboard format exactly), site accuracy, mRS-shift quadratic-weighted κ plus dichotomised AUC on absolute mRS≤2.

**Descriptive.** The guideline-relation support table (§5.1's surviving half of the dropped KG module) — which asserted clinical relations are and are not supported by training-fold empirical co-occurrence, cell counts shown so thinness is visible.

---

## 7. Protocol

**Splits.** 5-fold CV over the 149 public cases, stratified by infarct-volume tercile and stroke pattern (main results; `src/model/dataset.py` implements this against the real per-subject volumes from the audit, not recomputed). Leave-one-center-out, Center-1 (n=99, real recovered count) vs. Center-2 (n=49), both directions — shift-degradation measurement, not a generalisation claim (§4/N3's demoted robustness check), with the power calculation stated in print given the n=49 constraint (§0).

**Seeds.** Three fixed seeds per fold, mean±SD throughout; every coverage/risk number carries a confidence interval, not a point estimate.

**Baselines.**

1. Published ISLES'24 leaderboard, external context only
2. nnU-Net multimodal, imaging only, point-estimate thresholding — this is item 1.6, currently training
3. Same backbone + tabular fusion, point-estimate thresholding
4. **Marginal split-conformal interval, no threshold conditioning** — the published state of the art (Lambert et al. transposed to this task); the baseline N1 must beat
5. Marginal conformal, jackknife+
6. Threshold-conditional conformal, cross-conformal — full N1 model
7. Full model + weighted conformal, cross-center only

Baseline 4 is the one that matters: if the certified policy does not improve boundary-relevant decision error over it, N1's core claim fails and only the diagnostic (§4/N1, retained) survives.

**Ablations.** Drop tabular branch; drop mRS head; drop `L_thr` (§5.2's tension, measured explicitly); volume from H1 integration vs. H3 direct head; CQR vs. split-conformal-on-residuals; split vs. cross-conformal width stability (already run on synthetic data, §9 — real-data repeat here); number of distance bands for the retained diagnostic; modality dropout with CTP absent (already structurally true — CTP was never a channel, §2 — so this ablation becomes "perfusion maps absent," simulating spoke-hospital NCCT/CTA-only triage); Bonferroni vs. monotone-fixed-sequence LTT (§4/N1 — Bonferroni is kept specifically to demonstrate it certifies nothing at n=149).

---

## 8. Risks

**Per-band calibration counts** remain the central technical risk for the retained boundary diagnostic. Cross-conformal mitigates it by using all n cases; band edges chosen from the observed volume distribution, per-band n reported in every coverage table.

**The boundary band may be thin at the chosen τ.** Resolved for τ∈{70,110} by the real-data audit (§2.1) rather than left open — both clear the LTT group-size floor, τ=110 marginally. Primary τ is 110mL (evidence-backed ceiling, §1.4), secondary 70mL (legacy, contested); this is now a data-backed choice, not inherited uncritically.

**`L_thr` may hurt calibration.** Anticipated in §5.2, turned into a measurement (§7 ablations) rather than a gamble.

**mRS shift is only partly determined by imaging, and now has a real missing-data constraint.** 54% complete-pair coverage (§2.1), not the "modest performance expected" hedge alone — H4 exists for multi-task regularisation and to estimate the harm ratio in N2 (§4), not to compete with the outcome-prediction literature, and its training subset is stated explicitly wherever H4 results are reported.

**CoW pseudolabels are model-generated, and direct-intersection site derivation does not work on real data.** Resolved in §2.1: nearest-distance derivation, geometrically verified, with 29.2% of real subjects flagged low-confidence rather than silently labelled. Manual spot-check of the flagged subset remains open work.

**Final infarct is not admission core; the cohort is all-reperfused.** Unchanged from v3, covered as a definition in §2.3, repeated as a limitation.

**Corrupted source files.** Not anticipated in v3 — found during real-data pipeline validation: one subject's perfusion map (sub-stroke0043, `cbf`) is truncated in the source release itself (confirmed against the archive's own manifest, not an extraction artifact). The data pipeline verifies file integrity (not just existence) and cleanly excludes affected subjects rather than crashing a training run on them; this is now standard practice for every pipeline entry point, not a one-off patch.

---

## 9. Verification status (real, not projected)

**Calibration layer, on synthetic data — complete.** `src/experiments/validate_calibration.py`: PIT mean 0.494 (want 0.5), sd 0.291 (want 0.289); interval coverage tracks nominal across α to within 1.65pp max error; LTT-certified risk holds within target in 100% of fresh-draw trials (need ≥90%); split-conformal width is 1.8x more variable than cross-conformal at matched α, empirically justifying §5.3's construction choice rather than asserting it.

**Certification limits, on synthetic data — complete.** `src/experiments/certification_limits.py`: feasible τ window [20,90]mL certifies in 100% of trials at n=149; τ=110 was the synthetic-only boundary case (§2.1 shows real data does slightly better); the sample-size floor table (risk≤0.05 needs n≥45 in-group, risk≤0.20 needs n≥11) is the basis for every real-data feasibility check in this document.

**Data pipeline, on the real release — complete for Week 1.** Center-label recovery, data audit, occlusion-site derivation, and mRS-shift target construction have all been run against the actual downloaded ISLES'24 archive, not simulated data; results are integrated into §2 rather than reported separately.

**Baseline segmentation (item 1.6) — complete.** 6-channel SegResNet, 200 epochs (v3's full spec), fold 0, 30 held-out real subjects, all four ISLES'24 metrics directly comparable to the published leaderboard:

| Metric | This baseline | Leaderboard (Kurtlab) |
|---|---|---|
| Dice | 0.2154 | 0.285 |
| Abs. volume difference | 21.34 mL | 21.23 mL |
| Lesion-wise F1 | 0.278 | 0.144 |
| Abs. lesion count difference | 6.57 | 7.18 |

Dice sits below the challenge-winning entry (which used heavy engineered preprocessing specifically to win the leaderboard); volume error, lesion-detection F1, and lesion-count accuracy match or exceed it, on a single run of the deliberately conventional architecture in §5.1 with no tuning beyond the default protocol. Per-subject Dice is uneven in the expected direction — several near-zero results on small/near-empty lesions, 0.5–0.6 on larger clean ones — consistent with §2.1's note that a third of this cohort is scattered multifocal small-lesion disease, the known hard case across the ISLES'24 literature, not a symptom particular to this run. Full per-subject results in `outputs/tables/baseline_fold0.json`.

**Decision layer on real model outputs (Review 2).** Not yet run — requires H3's trained quantile head, which requires the baseline above.

---

## 10. References

Beyond v3's reference list (§10 there is retained in full); additions specific to the v4 restructure:

1. Riedel E.O. et al. The ISLES'24 Dataset. *Radiology: AI* 8(3) (2026). DOI 10.1148/ryai.250603.
2. de la Rosa E. et al. ISLES'24: Final infarct prediction with multimodal imaging and clinical data. arXiv:2408.10966.
3. Kurtlab. How We Won the ISLES'24 Challenge by Preprocessing. arXiv:2505.18424.
4. Lambert B. et al. Robust Conformal Volume Estimation in 3D Medical Images. MICCAI 2024. arXiv:2407.19938.
5. Jaubert O. et al. Conformal Coronary Calcification Volume Estimation with Conditional Coverage via Histogram Clustering. ISBI 2025. arXiv:2506.04030.
6. Conformal selective prediction with cost-aware deferral for safe clinical triage under distribution shift. *Scientific Reports* (2026). DOI 10.1038/s41598-026-40637-w.
7. Angelopoulos, Bates et al. Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control. *Ann. Appl. Stat.* 19(2), 2025. arXiv:2110.01052.
8. Angelopoulos, Bates, Fisch, Lei, Schuster. Conformal Risk Control. arXiv:2208.02814.
9. Vovk V. et al. Conformal predictive distributions (2017/2019); Vovk & Manokhin, Cross-conformal predictive distributions (2018).
10. Boström H., Johansson U. Mondrian conformal regressors. COPA 2020.
11. Gibbs I., Cherian J.J., Candès E. Conformal prediction with conditional guarantees. *JRSS-B* 87(4):1100, 2025.
12. Vickers A.J., Elkin E.B. Decision curve analysis.
13. Einbinder et al. Label noise robustness of conformal prediction. *JMLR* 25 (2024); Penso & Goldberger, MLMI@MICCAI 2024.
14. 2026 AHA/ASA Guideline for the Early Management of Patients With Acute Ischemic Stroke. DOI 10.1161/STR.0000000000000513.
15. Defining the Therapeutic Ceiling of EVT in Large-Core Stroke. PMID 42403349.
16. Luan T. et al. Unified multimodal learning for stroke triage. *npj Digital Medicine* 9:441 (2026). Cited as motivation only (§1); not used as a benchmark.
17. Kervadec H. et al. Constrained-CNN losses for weakly supervised segmentation. *Medical Image Analysis* 54:88–99 (2019).
18. Cao W., Mirjalili V., Raschka S. Rank consistent ordinal regression (CORAL).
19. Barber R.F., Candès E., Ramdas A., Tibshirani R. Predictive inference with the jackknife+. *Annals of Statistics* (2021).
