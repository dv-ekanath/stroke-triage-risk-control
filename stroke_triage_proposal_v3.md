# Threshold-Conditional Conformal Calibration for Ischemic Stroke Triage Decisions

### Where volumetric uncertainty guarantees fail, and how to fix them at the decision boundary

**Author:** Rohan Julius Preetan A
**Course:** Healthcare Analytics
**Version:** 3 (supersedes v2)

---

## 1. What changed from v2, and why

Version 2 was structurally sound. Its base-paper critique survives verification. Two of its factual claims about ISLES'24 do not, one of its novelty claims is already occupied by published work, and one of its statistical choices does not work at this sample size.

### 1.1 The base-paper critique holds

Confirmed against the challenge report and the winning entry. The ISLES'24 hidden test leaderboard tops out at Dice 28.50 ± 21.27 (Kurtlab), with AMC-Axolotls at 26.27 and Ninjas at 25.46. Luan et al.'s reported Dice 0.81 on the same dataset is roughly three times the challenge winner under the challenge's own protocol. Keep v2 §1 as written. Add one correction: ISLES'24 evaluates on four metrics, not three. Absolute lesion count difference (ALD) is the fourth, and it matters here because a third of the cohort has scattered multifocal infarcts.

### 1.2 The public training set is two centers, not one

v2 §6 states "the public set is effectively single-centre" and drops leave-one-center-out on that basis. That is wrong. The dataset descriptor tabulates the public training set (n = 149) as Center 1 = 100 (67.1%) and Center 2 = 49 (32.9%). Both Munich and Zurich are represented in the public release, on different scanner fleets (Siemens Somatom Force / Xcite / AS+ versus Philips Brilliance 64 / Ingenuity).

This is the single most consequential correction, because covariate shift between calibration and test data is the known open weakness of conformal prediction in medical imaging, and this dataset supports a real test of it. v3 turns the error into a contribution (§4.3).

### 1.3 Occlusion-site labels are better supported than v2 assumed

v2 lists occlusion-site derivation as a risk requiring "registration of occlusion masks to a vascular atlas" and manual spot-checking. Unnecessary. The dataset ships two things that make this near-trivial:

- Manual CTA vessel occlusion masks delineating the final vessel segment before occlusion, drawn by two neuroradiologists in training and reviewed by an attending with 25 years of experience. Interrater agreement on 45 paired annotations: median center-of-mass displacement 1.33 mm.
- Circle of Willis segmentations with per-vessel labels (left/right ACA, MCA, ICA, PCA, anterior communicating, basilar), produced by a TopCoW-derived two-stage U-Net ensemble.

Site labels come from intersecting the occlusion mask with the labelled CoW segmentation. No atlas registration. The CoW labels are model-generated pseudolabels, so a spot-check on a subset is still warranted, but the risk drops from "may not be feasible" to "verify a sample."

### 1.4 The primary novelty claim is occupied

v2 N1 claims that conformal intervals on segmentation-derived lesion volume, feeding a threshold decision with an abstention region and risk-coverage evaluation, is unreported. Three of those four components are published:

- Lambert et al., *Robust Conformal Volume Estimation in 3D Medical Images*, MICCAI 2024. Conformal predictive intervals on volumes derived from 3D segmentation, using a multi-head architecture that emits a restrictive mask (lower bound), a permissive mask (upper bound) and a balanced mask. This is close to v2's H3-versus-H1 comparison plus the interval, already done.
- Jaubert et al., ISBI 2025. Cluster-conditional conformal intervals on coronary calcium volume, motivated explicitly by the inadequacy of marginal coverage under image-quality and cohort variation.
- Conformal selective prediction with cost-aware deferral for clinical triage (Sci Rep 2026), which establishes the conformal-set → deferral-threshold → risk-coverage-curve pipeline in a clinical setting.

What is left of N1 after subtracting these is "apply the established pipeline to a stroke guideline cutoff." That is an application. It is not defensible as a primary claim, and a reviewer who knows the MICCAI 2024 paper would say so in one line.

### 1.5 The conformal calibration budget in v2 does not work

v2 §5.3 calibrates on 20% of each training fold and runs α ∈ {0.05, 0.10, 0.20}. With 5-fold CV over 149 cases, the training split is about 119 and the calibration split about 24.

Split conformal takes the ⌈(n+1)(1−α)⌉-th order statistic of n calibration scores. At n = 24:

| α | order statistic taken | of 24 |
|---|---|---|
| 0.05 | ⌈25 × 0.95⌉ = 24 | the maximum |
| 0.10 | ⌈25 × 0.90⌉ = 23 | second-largest |
| 0.20 | ⌈25 × 0.80⌉ = 20 | fifth-largest |

At α = 0.05 the interval width for an entire fold is set by a single worst calibration case. At α = 0.10 it is set by the second worst. Coverage is still technically valid, but the width, and therefore the abstention rate, is unstable to the point of being uninterpretable across folds. v2 lists jackknife+ as a fallback. It has to be the primary method, not the fallback.

---

## 2. What the data actually contains

Verified against the Radiology: AI dataset descriptor (Riedel et al., 2026) and the challenge report.

**Public training set:** n = 149, BIDS, CC BY-NC-SA 4.0, Zenodo. Hidden test n = 96 (the challenge report says 100 and one abstract says 98; the descriptor's 96 is authoritative for the released dataset). Total 245.

| Item | Status | Use in v3 |
|---|---|---|
| NCCT, CTA, 4D CTP, perfusion maps (CBF, CBV, MTT, Tmax) | Available | Encoder inputs; perfusion maps in place of raw 4D |
| Final infarct mask (DWI, day 2–9) | Available | Segmentation target, volume target |
| CTA vessel occlusion masks | Available, manual, neuroradiologist-reviewed | Site-label derivation |
| Circle of Willis segmentation, per-vessel labels | Available, model pseudolabels | Site-label derivation |
| Stroke pattern class (4 levels) | Available | Group-conditional conformal strata |
| Anatomic location (11 regions, largest-volume assignment) | Available | Secondary strata |
| mRS: premorbid, admission, 24h, discharge, 3-month | Available | Outcome head, mRS-shift target |
| NIHSS: admission, 24h, discharge | Available | Tabular branch |
| mTICI post-intervention | Available for 97/149 in training | Cohort characterization only, see §2.2 |
| Center label | To be recovered, see §7 risk 1 | Cross-center conformal transfer |
| Labs, medications, history, time metrics | Available, ±5% noise on labs and times | Tabular branch |
| mCTA collateral grade, admission core annotation, DSA, transfer outcomes | Not present | Not modelled |

### 2.1 The volume distribution decides the primary metric

Scan-level infarct volume across both centers: 33.16 ± 48.67 mL, range <0.01 to 318.25 mL. Heavily right-skewed. Three cases have no final infarct at all.

At a 70 mL cutoff this is a badly imbalanced binary problem. A constant "below cutoff" predictor will score somewhere in the low-to-mid 80s on plain accuracy. v2's primary metric, "eligibility decision accuracy at τ = 70 mL," therefore has a trivial baseline that is close to what a real model will achieve, and reporting it alone would be misleading. v3 replaces it (§6) and makes measuring the actual class balance and the near-boundary case count the first deliverable.

Stroke pattern breakdown, both centers: single-vessel infarct 40.0%, scattered infarcts from micro-occlusions 33.1%, mixed 25.7%, no final infarct 1.2%. Unconnected infarct count averages 10.55 ± 11.10, reaching 84. A third of this cohort is multifocal small-lesion disease, which is where segmentation degrades and where volume error behaves differently. This is a labelled subgroup, so it can be conditioned on rather than averaged over.

### 2.2 The estimand, restated

v2 framed the mismatch between final infarct and admission core as an apology. It is better handled as a definition, because the inclusion criteria make the cohort narrower than v2 assumed.

Every patient in ISLES'24 underwent successful intracranial reperfusion, predominantly mTICI 2c or 3 (in the training set: 2c in 22/97, 3 in 75/97 of those with recorded mTICI). Reperfusion is close to constant by design. So the target is not "final infarct under unknown treatment." It is **tissue outcome given successful reperfusion**, estimated from pre-interventional imaging.

That is a coherent quantity, and it is closer to what an eligibility decision is actually trying to anticipate than v2 implied: the question at triage is what the brain looks like if you succeed. Two consequences follow, both of which go in the limitations rather than being hidden:

- There is no treatment variation, so nothing counterfactual can be estimated and nothing about untreated trajectory can be claimed.
- The cohort is conditioned on having been treated and on treatment succeeding, which is a selection effect on top of the LVO inclusion criterion.

The eligibility decision studied here remains a **guideline-threshold decision proxy**. Methodology transfers to admission-core data; the numbers do not.

### 2.3 The 70 mL cutoff is contested, and that is useful

v2 anchors on DAWN and DEFUSE-3, where a volume ceiling gates eligibility. That anchor has moved. SELECT2, ANGEL-ASPECT and RESCUE-Japan LIMIT established benefit for thrombectomy in large-core patients, and current guideline practice endorses selection for large cores in both early and late windows. A hard 70 mL exclusion is no longer how the decision is made. Separately, a roughly 50 mL anterior-circulation core threshold has been reported to discriminate poor 90-day prognosis with AUC 0.91.

So there is no single correct τ. v3 treats that as the setting rather than a nuisance: the object of study becomes the model's decision behaviour across the plausible threshold range, not its accuracy at one number (§4.4).

---

## 3. The gap, restated

Conformal volume intervals exist. Conformal abstention in clinical triage exists. Neither literature has asked whether the coverage guarantee holds **where the decision is actually made**.

Conformal prediction gives marginal coverage: P(V ∈ Γ(X)) ≥ 1 − α, averaged over the whole distribution of patients. The quantity a triage decision depends on is different. It is the error rate conditional on the patient sitting near the cutoff, because a patient at 15 mL and a patient at 68 mL do not present the same decision problem, and only one of them can be got wrong by a small volume error.

Marginal coverage is achieved most cheaply by covering the bulk of the distribution well. In a right-skewed cohort with mean 33 mL, the bulk sits far below any plausible cutoff. Boundary cases are a minority, they are harder, and conditional coverage is known to fail without extra structure. So the guarantee is weakest exactly where the decision is hardest, and reporting marginal coverage on a threshold task conceals this by construction.

Nobody has stated this, measured it, or corrected it.

A second gap sits underneath it. Every conformal volumetry paper reports mean interval width as its efficiency metric, and no paper says what width would be good. Width has no reference scale. ISLES'24 supplies one, because the reference standard's own volume reliability is quantified.

---

## 4. Novelty claims

Four claims, descending strength. What is not claimed is listed after.

### N1. Threshold-conditional coverage, and calibration that achieves it (primary)

Define **threshold-conditional coverage**: empirical coverage stratified by |V − τ|, reported in bands (for example 0–10, 10–25, 25–50, >50 mL from the cutoff), alongside the marginal figure.

The claim has two parts, and the first is worth reporting even if the second fails.

**N1a (diagnostic).** Marginal conformalized quantile regression under-covers in the near-boundary band on ISLES'24, relative to its nominal level and relative to the far-from-boundary bands. This is a measurement, it is cheap, and a null result is a real finding: if boundary coverage holds, the field's current practice is vindicated on this dataset and that is worth saying.

**N1b (method).** Correct it with **distance-to-threshold Mondrian conformal calibration**: partition the calibration set by predicted distance to the cutoff, |V̂ − τ|, and compute a separate conformity quantile per band. Partitioning on a function of X only, not of Y, preserves the exchangeability argument, so coverage validity is retained per band by the standard Mondrian result. The consequence is that near-boundary patients get intervals calibrated against other near-boundary patients rather than against the easy bulk.

The trade is efficiency. Splitting an already small calibration set into bands shrinks per-band n, which is why the jackknife+ / CV+ construction in §5.3 is load-bearing rather than optional.

To my knowledge, threshold-conditional coverage has not been defined or measured in medical volumetry, and no calibration scheme has been conditioned on distance to a clinical decision boundary. The Mondrian machinery is textbook; the conditioning variable and the diagnostic are the contribution.

### N2. Annotation-floor-referenced abstention (secondary)

The ISLES'24 reference standard has quantified volume reliability: against two experienced external raters, mean Dice 0.90 ± 0.09 and 0.86 ± 0.13, with mean volume differences of 2.37 ± 2.59 mL and 6.56 ± 13.37 mL. (Provenance note: the descriptor attributes these to the ISLES'22 validation of the segmentation pipeline used to generate ISLES'24 labels, so they characterise the labelling procedure rather than being a fresh ISLES'24 rater study. State this when citing.)

Two things follow that nobody currently does.

**Width normalization.** Report conformal interval width as a ratio to inter-rater volume disagreement, not in raw millilitres. A 30 mL interval means something different when the label itself is reliable to ±13 mL than when it is reliable to ±2 mL. This gives interval efficiency a reference scale it currently lacks across the whole conformal volumetry literature.

**A model-independent floor on the abstention band.** A system that resolves a case sitting 5 mL from the cutoff is asserting precision the reference standard does not have. So there is a minimum abstention band around τ, set by annotation reliability rather than by the model, inside which no automated system should claim a decision regardless of how confident it is. Report where the model's learned abstention band sits relative to that floor. A model whose band is narrower than the floor is overconfident by construction, and this is detectable without any additional data.

### N3. Cross-center conformal transfer on ISLES'24 (secondary)

Lambert et al. state that empirical investigation of weighted conformal prediction under covariate shift in the medical domain is lacking, and address it on synthetic data plus BraTS. ISLES'24's public training set is two centers with different scanner fleets, which makes a direct test available on a real multi-center clinical cohort.

Protocol: calibrate on Center 1, evaluate coverage on Center 2, and reverse. Measure coverage degradation, marginal and threshold-conditional. Then apply weighted conformal with a density ratio estimated from the segmentation model's latent representation, following Lambert et al., and measure how much coverage is recovered and at what cost in width.

The extension beyond Lambert et al. is that coverage degradation is measured **at the decision boundary**, not just marginally, and its consequence is expressed in decision flips rather than in coverage points. Cross-center coverage loss that shows up marginally as two or three percentage points may show up at the boundary as a materially different abstention rate, and that is the number a deploying hospital would care about.

### N4. Decision stability across the threshold landscape (supporting)

Because there is no single agreed cutoff (§2.3), sweep τ from roughly 10 to 150 mL and plot, as functions of τ: decision error rate, abstention rate, and threshold-conditional coverage in the innermost band. Report the τ-region in which the model is decision-reliable and the regions in which it is not.

This is a small contribution but it is an honest one, and it produces an artifact that a clinician can read directly: given a cutoff your institution uses, here is whether this model can support it.

### Not claimed as novel

- Conformal prediction, conformalized quantile regression, Mondrian/group-conditional conformal, jackknife+ and CV+ as methods.
- Conformal predictive intervals on segmentation-derived volume (Lambert et al., MICCAI 2024; Jaubert et al., ISBI 2025).
- Weighted conformal prediction for covariate shift, including latent-space density ratio estimation (Lambert et al.).
- Conformal sets feeding a deferral rule evaluated by risk-coverage curve in clinical triage (Sci Rep 2026).
- Differentiable size and volume constraints in segmentation losses (Kervadec 2019, log-barrier variant).
- Selective prediction and reject-option learning.
- Multimodal fusion of CT trilogy imaging; multi-task learning as regularisation.
- Knowledge-graph encoding via GAT with cross-attention injection.

### Dropped from v2

**The trained consistency loss `L_cons`.** v2 proposed a KL penalty between the joint predicted distribution over (site bin, volume bin, mRS bin) and a graph-derived compatibility distribution with edge weights fitted on the training fold. With four site classes, three volume bins and three mRS bins, that is 36 cells estimated from about 119 training cases per fold. Roughly three cases per cell. The estimated compatibility distribution would be almost entirely sampling noise, and the loss would be tuning against it. Drop the loss.

Keep the descriptive half: the table of which guideline-asserted relations are and are not supported by training-fold co-occurrence stands on its own as a finding, and it costs a day rather than a week. Keep the post-hoc rule-check baseline that v2 already listed.

**Explicit α = 0.05 as an operating point.** See §1.5. Report α ∈ {0.10, 0.20} with jackknife+.

---

## 5. Method

### 5.1 Architecture

Deliberately conventional. The challenge showed transformer variants gave no advantage at this sample size, and the contribution is not in the backbone.

```
NCCT ─────┐  symmetry dual-stream encoder
CTA ──────┤  3D encoder                          ┌─► H1  final infarct segmentation
Perf maps ┤  3D encoder (CBF/CBV/MTT/Tmax)       ├─► H2  occlusion site (ICA/M1/M2/other)
          │           ↓                           ├─► H3  volume quantile head (τ_lo, τ_hi)
          │   cross-modal fusion                  └─► H4  3-month mRS shift (ordinal, CORAL)
Tabular ──┴─► MLP ─► FiLM conditioning
```

nnU-Net-class encoder-decoder trunk. H3 predicts volume quantiles directly; volume by integration of H1's soft mask is retained as an ablation, following the comparison Lambert et al. already report.

H4 predicts **mRS shift** (3-month minus premorbid) rather than absolute 3-month mRS. Premorbid mRS averages 0.9 ± 1.2 in the training set and is a strong determinant of the absolute outcome, so predicting the absolute score lets the model score well by learning baseline disability from tabular data alone. Shift removes that shortcut. Absolute mRS is reported alongside for comparability with the outcome-prediction literature.

### 5.2 Losses

```
L = λ1·L_seg + λ2·L_site + λ3·L_mRS + λ4·L_pinball + λ5·L_thr
```

- `L_seg` — Dice + cross-entropy on the final infarct mask
- `L_site` — cross-entropy on CoW-derived occlusion-site labels
- `L_mRS` — CORAL ordinal loss on mRS shift
- `L_pinball` — pinball loss on H3 at α/2 and 1−α/2
- `L_thr` — threshold-margin penalty, retained from v2 N3 and positioned as machinery, not novelty:

  ```
  L_thr = 1[V < τ]·ReLU(V̂ − τ + m)²  +  1[V ≥ τ]·ReLU(τ + m − V̂)²
  ```

  This is Kervadec's differentiable inequality-penalty family applied to a guideline cutoff rather than an annotation prior. Cited as such.

**A tension worth measuring.** `L_thr` pushes predictions away from τ. Threshold-conditional conformal calibration needs a well-populated, well-behaved conformity-score distribution near τ. Training to evacuate the boundary region degrades the signal the calibration layer depends on there. These two components work against each other and nobody has said so, because nobody has run them together. Quantify it: threshold-conditional coverage and boundary-band width, with and without `L_thr`, at matched Dice. If the interaction is real, that is a small independent finding about combining decision-focused training with conformal calibration.

### 5.3 Conformal decision layer

**Construction.** Jackknife+ / CV+ over the outer folds rather than a single 80/20 split, so that all 149 cases contribute conformity scores instead of ~24 per fold. Split conformal is retained as an ablation to demonstrate the width instability quantified in §1.5, which doubles as a justification for the choice.

**Scores.** CQR conformity: `s_i = max(τ̂_lo(x_i) − V_i, V_i − τ̂_hi(x_i))`. Split-conformal-on-residuals compared as an ablation.

**Marginal layer.** Standard: `q̂` = the appropriate order statistic of the conformity scores, interval `[τ̂_lo − q̂, τ̂_hi + q̂]`.

**Threshold-conditional layer (N1b).** Partition calibration cases into bands by predicted distance |V̂ − τ|. Compute `q̂_b` per band. A test case is assigned to a band by its own predicted distance, then given that band's quantile. Because the partition depends only on X, per-band coverage validity follows from the Mondrian argument.

**Decision.** Interval entirely below τ → eligible. Entirely above → ineligible. Straddling → abstain and refer. The abstention band is compared against the annotation-reliability floor from N2.

### 5.4 Explanation layer

Deterministic template filling. No LLM, no API call. Given head outputs plus the interval, emit the predicted values, the interval, its position relative to the cutoff, the occlusion site, and the predicted mRS shift. When the interval straddles, name the referral reason and state whether the case also falls inside the annotation-reliability floor.

Example shape:

> Ineligible under 70 mL cutoff. Predicted infarct 94 mL, 90% interval 71–121 mL (threshold-conditional, boundary band), interval lies entirely above cutoff. Occlusion site M1. Predicted mRS shift +3. Interval width 50 mL, 3.7× the reference-standard volume disagreement.

This is presentation, not a claim.

---

## 6. Metrics

**Primary, decision level.** Reported with the majority-class baseline stated alongside, every time.

- Balanced accuracy and Matthews correlation coefficient at τ, not plain accuracy (§2.1)
- Threshold-crossing error rate, overall and within the innermost |V − τ| band
- **Threshold-conditional coverage** by |V − τ| band, against nominal 1 − α, with marginal coverage reported alongside for contrast
- Mean interval width **as a ratio to reference-standard volume disagreement**, per band
- Abstention rate, and abstention band width relative to the annotation-reliability floor
- Risk-coverage curve and its area, computed over the boundary band as well as marginally

**Cross-center (N3).** All of the above, calibrated on one center and evaluated on the other, in both directions, with and without weighted conformal.

**Group-conditional.** Coverage and decision error by stroke pattern (single-vessel / scattered / mixed) and, if cell counts permit, by anatomic region.

**Threshold sweep (N4).** Decision error, abstention rate and innermost-band coverage as functions of τ ∈ [10, 150] mL.

**Secondary, task level.** Dice, absolute volume difference, lesion-wise F1, **and absolute lesion count difference**, matching all four ISLES'24 metrics so results are directly comparable to the published leaderboard. Site accuracy. mRS-shift quadratic-weighted κ, plus dichotomised AUC on absolute mRS ≤ 2.

**Descriptive.** Table of guideline-asserted relations against training-fold empirical co-occurrence, with cell counts shown so the reader can see where the estimate is thin.

---

## 7. Protocol

**Splits.** Two protocols, reported separately.

1. 5-fold cross-validation over the 149 public cases, stratified by infarct-volume tercile and by stroke pattern. Main results.
2. Leave-one-center-out, Center 1 (n = 100) versus Center 2 (n = 49), both directions. Shift results only. Not presented as a generalization claim, since 49 cases is small; presented as a coverage-degradation measurement.

If the Grand Challenge portal accepts submissions, one hidden-test submission for external context on the segmentation metrics.

**Seeds.** Three fixed seeds per fold, mean ± SD throughout.

**Baselines.**

1. Published ISLES'24 leaderboard, external context only
2. nnU-Net multimodal, imaging only, point-estimate thresholding
3. Same backbone + tabular fusion, point-estimate thresholding
4. Marginal conformal interval, split conformal, no threshold conditioning — this is the published state of the art (Lambert et al. transposed to this task) and is the baseline N1 must beat
5. Marginal conformal, jackknife+
6. Threshold-conditional conformal, jackknife+ — full model
7. Full model + weighted conformal, cross-center only

Baseline 4 is the one that matters. If threshold-conditional calibration does not improve boundary coverage over it, N1b fails and only N1a survives.

**Ablations.** Drop tabular branch; drop mRS head; drop `L_thr`; volume from H1 integration versus H3 direct head; CQR versus split-conformal-on-residuals; split conformal versus jackknife+ (width stability); number of distance bands (2, 3, 4) against per-band calibration count; modality dropout with CTP absent, simulating spoke hospitals.

---

## 8. Risks

**Center labels may not ship with the public release.** This gates N3 entirely. Week 1 blocker, resolve first. Recovery order: BIDS JSON sidecars and any participants.tsv; scanner manufacturer and model fields, which differ between the sites (Siemens Somatom versus Philips Brilliance/Ingenuity); failing both, unsupervised clustering on acquisition parameters, reported as inferred rather than known. If no reliable center assignment can be recovered, N3 is dropped and N1 plus N2 carry the project. Do not fabricate a proxy and call it a center.

**Per-band calibration counts.** Splitting calibration by distance-to-threshold on 149 cases is the central technical risk of N1b. Jackknife+ mitigates it by using all cases. Mitigate further by choosing band edges from the observed volume distribution rather than round numbers, and by reporting per-band n in every coverage table. If the innermost band cannot be populated to a usable count, report N1a alone and state why N1b was not evaluable. That is a legitimate outcome.

**The boundary band may be nearly empty.** If very few of the 149 cases sit within ±20 mL of 70 mL, the entire threshold-conditional analysis has no data at that cutoff. This is why the Week 1 audit measures near-boundary counts across the τ range before anything else, and why N4's sweep exists: it identifies which τ values the dataset can actually support. Choose the primary τ where the data has mass, and justify it in print rather than inheriting 70 mL uncritically.

**Threshold-margin loss may not help, and may actively hurt calibration.** Anticipated, and §5.2 turns it into a measurement rather than a gamble.

**mRS shift is only partly determined by imaging.** Age, comorbidity and procedural factors dominate. Modest performance expected. The head exists for multi-task regularisation and as an anchor for the descriptive consistency table, not to compete with the outcome-prediction literature.

**CoW pseudolabels are model-generated.** Occlusion-site labels inherit that error. Spot-check a subset against the manual occlusion masks and report agreement.

**Final infarct is not admission core, and the cohort is all-reperfused.** Covered in §2.2 as a definition, and repeated in the limitations.

---

## 9. Schedule

| Week | Output |
|---|---|
| 1 | Data audit. Center-label recovery (gate for N3). Volume distribution, class balance at candidate τ, near-boundary case counts across τ ∈ [10, 150]. Stroke-pattern and mRS distributions. Decide primary τ from the data. |
| 2 | Preprocessing pipeline. Occlusion-site labels from occlusion mask × CoW intersection, with spot-check agreement. mRS-shift target construction. |
| 3–4 | Baseline segmentation model, all four ISLES'24 metrics, comparable to leaderboard. Tabular fusion. |
| 5 | H3 quantile head, H4 mRS-shift head, H2 site head. Marginal conformal layer (baseline 4 and 5). Split-versus-jackknife+ width stability. |
| 6 | Threshold-conditional calibration. N1a diagnostic, N1b correction, boundary-band risk-coverage curves. Annotation-floor comparison (N2). |
| 7 | Cross-center transfer and weighted conformal (N3), if Week 1 unblocked it. Threshold sweep (N4). `L_thr` ablation and the calibration-tension measurement. |
| 8 | Write-up, explanation-layer demo, guideline-relation support table. |

Code, configs and seed lists in a non-anonymised repository.

---

## 10. References

1. Riedel E.O. et al. The ISLES'24 Dataset: A Multimodal Stroke Imaging Dataset with Hyperacute CT, Acute Postinterventional MRI, and 3-month Clinical Outcomes. *Radiology: Artificial Intelligence* 8(3) (2026). DOI 10.1148/ryai.250603. — authoritative for dataset composition, center split, interrater agreement, stroke-pattern taxonomy.
2. de la Rosa E. et al. ISLES'24: Final infarct prediction with multimodal imaging and clinical data. arXiv:2408.10966. — challenge protocol, four evaluation metrics.
3. Kurtlab. How We Won the ISLES'24 Challenge by Preprocessing. arXiv:2505.18424. — leaderboard baseline, Dice 28.50 ± 21.27.
4. Lambert B. et al. Robust Conformal Volume Estimation in 3D Medical Images. MICCAI 2024, LNCS 15010:633–643. arXiv:2407.19938. — prior art for N1; open problem cited by N3.
5. Jaubert O. et al. Conformal Coronary Calcification Volume Estimation with Conditional Coverage via Histogram Clustering. ISBI 2025. arXiv:2506.04030. — prior art for conditional coverage in volumetry.
6. Conformal selective prediction with cost-aware deferral for safe clinical triage under distribution shift. *Scientific Reports* (2026). DOI 10.1038/s41598-026-40637-w. — prior art for the conformal-to-deferral pipeline.
7. Romano Y., Patterson E., Candès E. Conformalized quantile regression. NeurIPS 2019.
8. Barber R.F., Candès E., Ramdas A., Tibshirani R. Predictive inference with the jackknife+. *Annals of Statistics* (2021).
9. Vovk V. Conditional validity of inductive conformal predictors. ACML 2012. — Mondrian construction underlying N1b.
10. Angelopoulos A., Bates S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification.
11. Kervadec H. et al. Constrained-CNN losses for weakly supervised segmentation. *Medical Image Analysis* 54:88–99 (2019).
12. Cao W., Mirjalili V., Raschka S. Rank consistent ordinal regression (CORAL).
13. Albers G. et al. DEFUSE-3. *NEJM* 378:708–718 (2018).
14. Nogueira R. et al. DAWN. *NEJM* 378:11–21 (2018).
15. 2026 AHA/ASA Guideline for the Early Management of Patients With Acute Ischemic Stroke. DOI 10.1161/STR.0000000000000513. — current threshold practice, large-core indications.
16. Luan T. et al. Unified multimodal learning for stroke triage. *npj Digital Medicine* 9:441 (2026). — cited as motivation only; results not used as benchmark, see v2 §1 and §1.1 above.
