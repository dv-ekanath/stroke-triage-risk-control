# Plan: Restructure the stroke triage proposal (v3 → v4)

## Context

`stroke_triage_proposal_v3.md` is a well-verified proposal whose novelty is too weak to
carry a project, and the author knows it. Verification against the base paper and the
prior art confirms this:

- **N1b (distance-to-threshold Mondrian) is textbook.** Binning calibration data by a
  function of the point prediction is exactly Boström & Johansson, *Mondrian conformal
  regressors* (COPA 2020). Jaubert et al. (ISBI 2025) already do cluster-conditional
  conformal on volume. The contribution reduces to the choice of binning variable.
- **Coverage is the wrong guarantee for a threshold decision.** The decision only cares
  which side of τ the truth falls on. Perfect near-boundary coverage with wide intervals
  means abstaining on everything; poor coverage can coexist with perfect decisions.
  Conditional coverage is a proxy, and the proposal is optimising the proxy.
- **N3 (cross-center transfer) is underpowered.** Calibrate on Center 1 (n=100), test on
  Center 2 (n=49): binomial SE of empirical coverage at nominal 90% is
  √(0.9·0.1/49) ≈ 4.3 pp, so a 95% CI spans ±8.4 pp. §4 asks to detect "two or three
  percentage points." Not possible.
- **N2 is a normalisation convention**, built on numbers the proposal itself flags as
  imported from ISLES'22 pipeline validation. **N4 is a figure.**

Separately, the base-paper critique is aimed at the wrong target. Full text of Luan et
al. (npj Digit Med, s41746-025-02255-0) shows three objections stronger than the Dice
discrepancy v3 leads with — see §2 below. Most importantly, their safety mechanism is a
**hand-set rejection threshold τ = 0.15**, with ECE = 0.130 described as "excellent
calibration," validated on **n = 30 borderline cases**. That is the hole this project
should be aimed at, and v3 never names it.

**Intended outcome:** a v4 proposal whose primary claim is a distribution-free guarantee
on the *triage decision error*, not on volume coverage — one sentence a reviewer cannot
compress away, plus two supporting claims that stand alone if the primary fails.

**Target:** course deliverable across three reviews (30% / 50% / 20%), structured to
extend into a MICCAI / UNSURE / COPA submission. ISLES'24 is downloaded; GPU available.

---

## 1. Restructured novelty claims

Four claims become three. Each is independently reportable.

### N1 (primary) — Risk-controlled threshold triage

Replace "cover the volume, then read off a decision" with a direct, asymmetric,
distribution-free guarantee on the decision itself.

**Step 1 — conformal predictive distribution.** Instead of an interval at fixed α, fit a
cross-conformal predictive system (Vovk et al. 2017/2019; Vovk & Manokhin 2018) emitting
a calibrated CDF `F̂ᵢ(v)` per patient, normalised by the H3 quantile spread as the
difficulty estimate. Cross-conformal aggregation over the 5 folds independently solves
the n≈24-per-fold instability documented in v3 §1.5.

Yields `pᵢ = 1 − F̂ᵢ(τ)`, a calibrated P(V > τ), **for every τ simultaneously**. v3's N4
threshold sweep becomes a free by-product rather than a separate experiment.

**Step 2 — certified decision rule.** Parameterise triage by `λ = (λ_lo, λ_hi)`,
`0 ≤ λ_lo < λ_hi ≤ 1`:

```
p < λ_lo          → eligible
p > λ_hi          → ineligible
λ_lo ≤ p ≤ λ_hi   → abstain, refer to clinician
```

Certify two risks simultaneously with **Learn-then-Test** (Angelopoulos, Bates et al.,
*Ann. Appl. Stat.* 2025), which handles multi-dimensional non-monotone λ via FWER-controlled
multiple testing over a λ grid:

- `R_missed(λ)` = P(system says ineligible | true V < τ) — denying thrombectomy to a
  patient who would benefit
- `R_futile(λ)` = P(system says eligible | true V ≥ τ) — futile reperfusion, procedural
  and haemorrhage risk

Among certified λ, select the one minimising abstention rate. Report the whole certified
region, not just the selected point.

**The asymmetry is set by the guideline, not by taste.** The 2026 AHA/ASA guideline makes
EVT in large core Class 1A in both the 0–6 h and 6–24 h windows, so denying treatment is
now the worse error: `α_missed ≪ α_futile`. This is citable and defensible, and it is a
nice structural point — a guideline change determines the loss asymmetry.

**Why this is novel.** Conformal risk control / LTT has been applied to FNR and IoU on
segmentation *masks*, and cost-aware deferral to *classification* triage (Sci Rep 2026,
DOI 10.1038/s41598-026-40637-w; cost-sensitive CP benchmark, arXiv 2607.27143). Nobody
has applied risk control to a **guideline threshold on a segmentation-derived continuous
quantity with two-sided asymmetric clinical costs**. It also makes the abstention band a
*certified* object — precisely what the base paper's τ = 0.15 is not.

**Retained from old N1a (diagnostic, cheap, report either way).** Measure whether marginal
split conformal under-serves the near-boundary band, stratified by |V − τ|. A null result
is a real finding. Keep the measurement; drop the Mondrian estimator that was the "fix".

### N2 (secondary) — Decision-analytic validation with a deferral arm

Decision curve analysis (Vickers) on the triage policy, extended to three actions
(treat / don't treat / defer). Standard net benefit has no deferral action, so extend it:

```
NB(p_t) = TP/n − (FP/n)·(p_t/(1−p_t)) − (n_deferred/n)·c_defer
```

Sweep the threshold probability `p_t` — which **is** the harm ratio, the exact quantity
contested in large-core stroke — and compare: point-estimate thresholding, marginal
conformal interval, and the N1 risk-controlled policy, against treat-all / treat-none.
Report the harm-ratio range over which deferral adds net benefit at a stated cost per
clinician read.

This answers v3 §2.3 ("there is no single correct τ") far better than a sweep plot,
because DCA is the principled way to handle a contested threshold. DCA and conformal
abstention have not been formally connected; the deferral cost term is a small, real
piece of methodology, and it is the language clinical journals read in.

**Give the mRS head a job here.** Use predicted mRS shift to estimate the harm ratio
empirically rather than asserting it. This converts H4 from decorative regularisation
into a load-bearing component.

### N3 (secondary) — What n = 149 can actually certify

Two parts, both turning limitations into contributions.

**Certification limits.** Realised coverage from n calibration points is Beta-distributed;
at n = 149, α = 0.10 it has roughly ±5 pp spread. LTT certification at tight α may fail
outright at this sample size. Publish the table: for each (α_missed, α_futile) pair, the
minimum n required and the confidence achieved at n = 149. Report coverage confidence
intervals everywhere instead of point estimates — nearly no clinical conformal paper does.

**Irreducible-error floor.** Upgrade v3's N2 from a convention to a bound using
label-noise-robust conformal (Einbinder et al., JMLR 25; Penso & Goldberger, MLMI@MICCAI
2024; arXiv 2509.15120). With a noise model calibrated to the reported inter-rater volume
disagreement (2.37 ± 2.59 and 6.56 ± 13.37 mL), derive the *minimum achievable* decision
error near τ for **any** model, and show where the system sits relative to it. Run as a
sensitivity analysis across plausible σ — which also disarms the ISLES'22-provenance
caveat v3 already flags.

**If LTT certifies nothing at n = 149, N3 is the finding**, not the failure: *here is the
sample size a stroke-triage system needs before it can make a guaranteed decision, and
nobody currently has it.*

### Demoted, dropped, kept

| Item | Disposition |
|---|---|
| Old N1b (distance-to-threshold Mondrian) | **Dropped as method**, retained as the N1a diagnostic |
| Old N3 (cross-center transfer) | **Demoted to robustness check**, with the power calculation stated in print |
| Old N4 (threshold sweep) | **Absorbed** — free under the CPD |
| Knowledge graph / GAT / territory structure | **Dropped.** Occlusion site is upstream of and correlated with volume, so it partly duplicates the conditioning already in place; under LTT, per-group certification multiplies hypotheses and costs power directly. Keep anatomy as a *pre-specified evaluation stratum* (v3 §6 already has this), not as method |
| H2 occlusion-site head | **Kept** — labels are near-free from shipped CoW segmentations, earns its place as multi-task regularisation, needed by the explanation layer |
| `L_thr` × calibration tension (v3 §5.2) | **Kept and promoted.** Most genuinely original observation in v3, currently buried as a footnote to "machinery" |

---

## 2. Rewrite the base-paper critique (v4 §1)

Three objections stronger than the Dice discrepancy, all from the full text:

1. **The generalisation protocol is impossible.** The paper states ISLES'24 = "149
   multicenter training cases" evaluated by "leave-one-center-out across **7 medical
   centers** with different vendors (GE, Siemens, Philips)." Truth (v3 §1.2, verified):
   149 cases, **2** centers, Siemens and Philips fleets, no GE. Their headline result —
   per-center Dice 0.81 ± 0.03 across 0.78–0.82, CV reduced 50–55% — cannot have been
   produced as described. Needs no interpretation to land.

2. **The Dice gap is a task mismatch, not a performance gap.** Their target is "ischemic
   core and penumbra," referenced to follow-up MRI *or* DSA; ISLES'24's label is the final
   infarct on day 2–9 DWI. They also carry CPAISD (n=112, hyperacute NCCT core/penumbra)
   and AISD (n=397). Dice 0.81 vs the challenge's 0.285 is almost certainly attribution
   drift between cohorts. Say this instead of "3× the winner," which invites "different
   preprocessing."

3. **Their uncertainty layer is the actual hole — this becomes v4's motivation.**
   ECE = 0.130 called "excellent calibration" (13 points of miscalibration); abstention is
   a hand-set τ = 0.15 justified post hoc (18/122 CTP scans flagged); clinical endpoint
   (unnecessary transfer 12.3% vs 18.9%) computed on **n = 30** against an unmatched 27%
   historical registry rate. Their own ablation says disabling rejection moves mis-triage
   5% → 9%, so the entire safety story rests on one unvalidated hyperparameter.

4. **Their eligibility rule is out of date.** They gate on DAWN/DEFUSE-3 core < 70 mL +
   mismatch > 1.8. The 2026 AHA/ASA guideline (DOI 10.1161/STR.0000000000000513) makes
   large-core EVT Class 1A. Anchor v4's τ discussion on the newer evidence: a nationwide
   Korean registry identifies **≥110 mL as a therapeutic ceiling** where benefit becomes
   negligible (PMID 42403349), and **OUTER LIMITS** is randomising core > 70 mL.

**New motivating sentence for v4 §1:** *the deployed system's abstention threshold is an
unvalidated hyperparameter evaluated on 30 patients; we produce one with a distribution-free
guarantee on the clinically asymmetric error.*

**On τ.** Primary τ = 110 mL (evidence-backed ceiling), secondary τ = 70 mL (legacy,
contested). Gate on the Week-1 audit: 110 mL likely has thinner near-boundary mass than
70 mL given the cohort distribution (33.16 ± 48.67 mL, right-skewed). Because the CPD is
calibrated at every τ, only the LTT certification is τ-specific — certify where the data
has mass, report calibrated crossing probability everywhere, and justify the choice in
print.

---

## 3. Work assignment by review

Indicative weeks assume a 12-week window; shift boundaries to actual review dates. The
governing principle: **the 50%-weighted review must contain the primary novelty
demonstrably working**, not in progress.

### Review 1 — 30% (indicative weeks 1–4)

Graded on problem framing, literature positioning, novelty statement, methodology design.
This is where the novelty argument must be airtight, and where every go/no-go gate is
resolved — an empty boundary band discovered at Review 2 would be fatal.

| # | Deliverable |
|---|---|
| 1.1 | **v4 proposal document** — rewritten §1 (base-paper critique per §2 above), §3 gap, §4 three claims, §5.3 conformal decision layer, §6 metrics, §7 protocol |
| 1.2 | **Prior-art differentiation table** — one row per adjacent paper (Lambert MICCAI'24, Jaubert ISBI'25, Sci Rep 2026, Boström & Johansson 2020, Angelopoulos LTT, Gibbs–Cherian–Candès JRSS-B 2025, cost-sensitive CP arXiv 2607.27143), one column for what it does, one for what N1 does that it does not. Carries the methods venue later |
| 1.3 | **Data audit** — volume distribution; class balance at τ ∈ {70, 110}; near-boundary case counts across τ ∈ [10, 150]; stroke-pattern and mRS distributions; **primary τ chosen from the data and justified** |
| 1.4 | **Center-label recovery** — BIDS sidecars → `participants.tsv` → scanner manufacturer/model fields (Siemens Somatom vs Philips Brilliance/Ingenuity) → failing all, unsupervised clustering on acquisition parameters, reported as inferred. Gates the robustness check only, no longer a claim |
| 1.5 | **Synthetic validation of the calibration layer** — CPD + LTT implemented and verified on synthetic (V̂, V) pairs with known noise, before touching real data. See Verification below |
| 1.6 | **Baseline segmentation running** — nnU-Net-class multimodal trunk, all four ISLES'24 metrics (Dice, absolute volume difference, lesion-wise F1, absolute lesion count difference), comparable to the published leaderboard |
| 1.7 | Preprocessing pipeline; occlusion-site labels from occlusion mask ∩ CoW intersection with spot-check agreement; mRS-shift target construction |

**Go/no-go gates resolved here:** (a) is there near-boundary mass at any clinically
defensible τ? (b) are center labels recoverable? (c) does the calibration layer control
risk on synthetic data? Failing (a) at every τ is the only fatal outcome, and it is
knowable in week 1.

### Review 2 — 50% (indicative weeks 5–9)

The primary claim, end to end, with results. Highest weight, so nothing speculative goes
here.

| # | Deliverable |
|---|---|
| 2.1 | **Full model trained** — H1 segmentation, H2 site, H3 volume quantile head, H4 mRS shift (CORAL), tabular FiLM branch. 5-fold CV stratified by volume tercile and stroke pattern, 3 seeds |
| 2.2 | **Conformal predictive distribution** — cross-conformal predictive system over the folds, normalised by H3 spread. Validate: calibrated CDF, PIT histogram uniform |
| 2.3 | **LTT risk control (N1, the money result)** — λ grid, both risks certified with FWER control, certified region reported, abstention-minimising λ selected. At α_missed ∈ {0.05, 0.10}, α_futile ∈ {0.10, 0.20} |
| 2.4 | **Baseline comparison** — the one that matters is *marginal split conformal interval → threshold decision*, i.e. the published state of the art transposed. Plus point-estimate thresholding (with and without tabular fusion) and the base paper's hand-set-threshold analogue |
| 2.5 | **Boundary diagnostic (old N1a)** — coverage and decision error stratified by \|V − τ\|, marginal vs certified policy, with per-band n shown |
| 2.6 | **Primary metrics** — balanced accuracy and MCC at τ with majority-class baseline stated alongside every time; threshold-crossing error rate overall and near-boundary; abstention rate; risk-coverage curve and its area, marginally and over the boundary band |
| 2.7 | **`L_thr` tension measurement** — threshold-conditional coverage and boundary-band width, with and without `L_thr`, at matched Dice. Training to evacuate the boundary degrades the signal the certification depends on there. Small independent finding |

### Review 3 — 20% (indicative weeks 10–12)

Supporting claims, robustness, write-up. Each item here is independently droppable
without touching N1.

| # | Deliverable |
|---|---|
| 3.1 | **N2 — net benefit with deferral arm**, swept over the harm-ratio range, all policies vs treat-all/treat-none; harm ratio estimated from the H4 mRS-shift head |
| 3.2 | **N3a — certification-limits table**, minimum n per (α_missed, α_futile), confidence achieved at n = 149; coverage CIs throughout |
| 3.3 | **N3b — irreducible-error floor** from annotation noise, sensitivity across plausible σ, model position relative to the floor |
| 3.4 | **Robustness (demoted old N3)** — leave-one-center-out, both directions, **with the power calculation stated in print**. Weighted conformal with latent-space density ratio (Lambert et al.) as a secondary comparison. Presented as a coverage-degradation measurement, never as a generalisation claim |
| 3.5 | **Ablations** — drop tabular branch; drop mRS head; drop `L_thr`; volume from H1 integration vs H3 direct head; CPD vs interval-based decision; split conformal vs cross-conformal (width stability); modality dropout with CTP absent, framed as robustness only (missing-modality conformal is occupied — MC², arXiv 2608.07183, arXiv 2608.07795) |
| 3.6 | **Group-conditional evaluation** — coverage and decision error by stroke pattern (single-vessel / scattered / mixed) and by anatomic region if cell counts permit. Pre-specified, reported, not guaranteed. If heterogeneity is large, that earns anatomy a place in follow-up work |
| 3.7 | **Explanation-layer demo** (deterministic template filling, no LLM), guideline-relation support table, final write-up, paper draft skeleton |
| 3.8 | Optional: one hidden-test submission via the Grand Challenge portal for external context on segmentation metrics |

---

## 4. Files

| Path | Change |
|---|---|
| `stroke_triage_proposal_v3.md` | Source; leave in place |
| `stroke_triage_proposal_v4.md` | **New.** Full rewrite of §1, §3, §4, §5.3, §6, §7, §8, §9. §2 (data) and §5.1–5.2 (architecture, losses) carry over nearly unchanged — the restructure is in the framing and the calibration layer, not the model |
| `src/data/` | Audit scripts, BIDS parsing, center-label recovery, occlusion-site derivation |
| `src/model/` | nnU-Net-class trunk + H1–H4 heads, FiLM tabular conditioning |
| `src/conformal/` | CPD construction, LTT risk control, net benefit, certification limits. Keep separate from the model — it is the contribution and must be testable standalone |
| `src/eval/` | ISLES'24 four metrics, decision metrics, stratified reporting |

---

## 5. Implementation notes

- **`crepes`** (Boström) implements conformal predictive systems, Mondrian variants and
  difficulty normalisation directly — the natural home for N1 step 1.
- **LTT** is a grid search over λ with a multiple-testing correction; reference code at
  `aangelopoulos/conformal-risk` and the LTT paper's repo. For 2-D λ use Split Fixed
  Sequence Testing (learn an ordering on part of the calibration data, fixed-sequence test
  on the rest) rather than Bonferroni over the full grid — Bonferroni at n = 149 will
  certify nothing.
- **Conditional conformal** (`jjcherian/conditional-conformal`) is *not* in the critical
  path under this restructure. Hold it in reserve: if the boundary diagnostic (2.5) shows
  severe heterogeneity that the certified policy handles poorly, a shift class spanned by
  RBFs of |V̂ − τ| is the principled fix and slots into Review 3.
- All conformal code operates on cached `(V̂ᵢ, quantile spread, Vᵢ, group labels)` tuples,
  not on volumes recomputed from masks. Decouples the calibration work from GPU
  availability entirely.

---

## 6. Verification

**Calibration layer, before real data (Review 1, item 1.5).** This is the part that must
be right, and it is verifiable without the model:

1. Generate synthetic `(V̂, V)` pairs with known heteroscedastic noise and a right-skewed
   volume distribution matched to the cohort (mean 33 mL, SD 49 mL).
2. Build the CPD; verify the PIT histogram is uniform and empirical coverage matches
   nominal across α, over ≥1000 resamples.
3. Run LTT; verify realised `R_missed` and `R_futile` on fresh draws are ≤ their nominal
   levels at the certified λ, at the claimed FWER, across resamples. **If this does not
   hold on synthetic data, the implementation is wrong** — do not proceed to real data.
4. Sweep n ∈ {50, 100, 149, 300, 1000} on the same synthetic setup — this *is* the
   certification-limits table (3.2), obtained early and for free, and it tells you at
   Review 1 whether N1 can succeed at n = 149.

**Model (Review 2).** Reproduce leaderboard-comparable numbers on all four ISLES'24
metrics before trusting any downstream decision result. 3 fixed seeds per fold,
mean ± SD throughout. If Dice is far above ~0.30 on final-infarct labels, something is
wrong with the target — that is the base paper's error and it is easy to repeat.

**Decision layer (Review 2).** Empirical `R_missed` / `R_futile` on held-out folds must
sit at or below nominal. Report the majority-class baseline next to every decision metric,
every time — at τ = 70 mL a constant "below cutoff" predictor scores in the low-to-mid
80s on plain accuracy, so balanced accuracy and MCC are the only honest primaries.

**Reproducibility.** Code, configs and seed lists in a non-anonymised repo. Seeds fixed
and reported; every coverage and risk number carries a confidence interval.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **LTT certifies nothing at n = 149** | Known before Review 1 ends via synthetic sweep (6.4). If it fails at tight α, report at α_missed = 0.10 / α_futile = 0.20 and make N3's certification-limits table the headline. Legitimate, publishable negative result |
| **Boundary band nearly empty at the chosen τ** | Week-1 audit measures near-boundary counts across τ ∈ [10, 150] *before* anything else. CPD calibrates at all τ, so only certification is τ-specific — certify where mass exists |
| **Center labels don't ship** | Now gates only the demoted robustness check (3.4), not a claim. Recovery order in 1.4; if unrecoverable, drop 3.4 and say so. Do not fabricate a proxy |
| **`L_thr` degrades certification** | Anticipated; 2.7 turns it into a measurement rather than a gamble |
| **mRS shift only partly determined by imaging** | Expected modest. H4 exists for multi-task regularisation and to estimate the harm ratio in N2, not to compete with the outcome-prediction literature |
| **CoW pseudolabels are model-generated** | Spot-check a subset against the manual occlusion masks, report agreement (1.7) |
| **Final infarct ≠ admission core; cohort is all-reperfused** | Keep v3 §2.2 verbatim — the estimand is *tissue outcome given successful reperfusion*, stated as a definition, repeated in limitations |

---

## 8. Key references to add in v4

Beyond v3's list:

- Angelopoulos, Bates et al. *Learn then Test: Calibrating Predictive Algorithms to
  Achieve Risk Control.* Ann. Appl. Stat. 19(2), 2025. arXiv:2110.01052 — N1 step 2
- Angelopoulos, Bates, Fisch, Lei, Schuster. *Conformal Risk Control.* arXiv:2208.02814 —
  monotone alternative
- Vovk et al. *Conformal predictive distributions* (2017/2019); Vovk & Manokhin,
  *Cross-conformal predictive distributions* (2018) — N1 step 1
- Boström & Johansson. *Mondrian conformal regressors.* COPA 2020 — the prior art that
  makes old N1b textbook; cite explicitly and differentiate
- Gibbs, Cherian, Candès. *Conformal prediction with conditional guarantees.* JRSS-B
  87(4):1100, 2025 — held in reserve, cite as the principled conditional method
- Vickers & Elkin, decision curve analysis; and the continuous net benefit extension —
  N2
- Einbinder et al. *Label noise robustness of conformal prediction.* JMLR 25;
  Penso & Goldberger, MLMI@MICCAI 2024 — N3b
- 2026 AHA/ASA AIS guideline, DOI 10.1161/STR.0000000000000513 — the loss asymmetry
- *Defining the Therapeutic Ceiling of EVT in Large-Core Stroke*, PMID 42403349 — τ = 110 mL
- Sci Rep 2026 DOI 10.1038/s41598-026-40637-w; arXiv 2607.27143 — nearest prior art on
  conformal deferral in clinical triage, must be differentiated explicitly
- MC² / conformal fusion under missing modalities (arXiv 2608.07183, 2608.07795) — cite
  to establish that the modality-dropout ablation is robustness, not novelty
