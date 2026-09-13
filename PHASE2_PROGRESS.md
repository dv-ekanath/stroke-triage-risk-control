# Phase 2 Progress — Perfusion Volumes and Constraint Testing

**Status: partial — pausing here to report before the heavier atlas work.**
Built and gated: brain mask, acute core, penumbra, mismatch, HIR. Tested: 4
more plausibility constraints, on top of Phase 1's 1. **Net result: only 1 of
7 plausibility constraints in the graph is supported.** This changes the risk
picture reported at the end of Phase 1 and needs a decision before continuing
into atlas registration.

## What was built and gated

| Gate | Result |
|---|---|
| Brain mask (skull-stripped from NCCT) plausible for ≥90% of subjects | ✅ **98.6%** in [1000, 1900] mL, mean 1517 mL |
| HIR computed | ✅ 148/148, mean 0.371 |
| Acute core / penumbra / mismatch computed | ✅ 123/148 (needs a known occlusion side) |

New files: `src/graph/perfusion_volumes.py` (brain mask, core, penumbra,
mismatch, HIR — `outputs/tables/perfusion_volumes.json`), 4 new checks in
`src/graph/constraint_support.py`.

**One real gain:** the `large_core_therapeutic_ceiling` eligibility rule
(Kim 2026, ≥110 mL) is now fully evaluable on ISLES'24 — it only needed
`acute_core_volume`, which is built. It's the only eligibility rule that is.

## Every plausibility constraint tested so far

| Constraint | Status | What it means |
|---|---|---|
| Infarct side matches clot side | **Supported** | 94.7% (89/94), Phase 1 |
| Proximal (ICA/M1) clot → bigger infarct | **Rejected** | p=0.66, no difference |
| High HIR → low mismatch ratio | **Rejected** | Significant, but backwards (rho=+0.19). Also found: the check wasn't as independent as intended — both quantities share a term |
| Infarct inside admission Tmax>6s region | **Rejected** | Median only 51% overlap; 49% of subjects below half; some near 0%. Not a registration bug (affines confirmed byte-identical) — see below |
| Poor collateral (HIR) → bigger infarct, proximal only, binned | **Rejected** | Right direction (19.97 vs 11.0 mL) but p=0.134, n=9 |
| **Re-test: same, continuous, ALL occlusion-positive patients** | **Rejected** | n=143, rho=0.052, p=0.54 — essentially no relationship. Doesn't just miss significance, finds a near-zero effect. The n=9 "right direction" result does not replicate. |
| Poor collateral → non-trivial infarct (mirror) | **Rejected** | Same underlying test |
| Large infarct needs a clot | **Untestable** | Only 4 patients have no clot |

**Current count: 1 supported, 5 rejected, 1 untestable, of 7.**

## The perfusion-overlap failure — why it's not a bug

Checked, in order:
1. **Registration/alignment.** Affines for Tmax and lesion-msk are
   byte-identical for every subject checked. Ruled out.
2. **CTP's limited slab coverage** (it doesn't cover the whole head like NCCT
   does). Explains 2 of the 5 worst cases (75–84% of the lesion falls outside
   the perfusion slab). Doesn't explain the other 3 — their lesion is 99.5–100%
   inside the covered slices, and overlap is still near 0%.

For those 3, lesion and Tmax>6s centroids are **20–72 mm apart** in real
physical space. **My working explanation:** every ISLES'24 patient had
successful mechanical thrombectomy, and distal embolization of clot
fragments during the procedure — a documented complication — can infarct
tissue in a *different* territory on the same side, one the pre-procedure
perfusion scan never showed as at risk. I have not confirmed this against
procedural records; ISLES'24 doesn't provide them. It's the most consistent
explanation for the pattern, not a proven one.

## What this changes

**The pattern from Phase 1 was "location survives reperfusion, size doesn't."
That was too optimistic.** Gross laterality (left/right) survives — distal
embolization essentially never crosses to the other hemisphere. But
*precise* location — exact voxel overlap with the original perfusion deficit
— does not survive, for the same procedural reason that scrambles size.

**Where that leaves the novelty claim:** 1 of 7 constraints supported, 2
inconclusive with a plausible path to becoming supported (bigger sample,
continuous correlation instead of binned groups), 1 fully-evaluable
eligibility rule gained. JDCR today would mostly measure the laterality
check. That is real, but it's a single, comparatively simple relationship —
worth saying plainly before it's presented as a bigger result than it is.

## Update, same day: the cheap re-test is done, and it didn't help

Ran option 2 below first, since it was cheap. **It came back negative, not
inconclusive.** The continuous, whole-cohort version (n=143, rho=0.052,
p=0.54) found essentially no relationship between HIR and final infarct
volume — this isn't a sample-size problem the n=9 test had; a real effect of
any real size would have shown up in 143 patients. The earlier "right
direction" reading from the small binned test doesn't replicate and was most
likely noise. Both collateral constraints are now recorded `not_supported`
in `kg/guideline_rules.json`, not `inconclusive`.

**Count now: 1 of 7 supported, 5 rejected, 1 untestable.** Three independent
tests (this one, `collateral_hir_vs_mismatch`, and the fact that HIR itself
never validated against an independent signal) now agree that HIR is not
behaving as a collateral proxy in this cohort, and that final infarct size is
not a size measure the graph's rules can use reliably — both conclusions
independent of each other, which makes them harder to dismiss as one fluke.

## Phase 2 cleanup (2026-09-14) — closed out

Three small remaining items from `CLAUDE.md`'s task list, done:

1. **Label rename.** "distal_unlocalized" → "unlocalized" throughout the code
   and graph file (`src/graph/constraint_support.py`, `kg/guideline_rules.json`).
   Pure rename — the underlying numbers are unchanged, only the name was wrong.
2. **MCA-absent subjects checked.** Of 23 subjects with the MCA missing from
   the vessel mask on their clot's side (more than the 17 first estimated), 13
   have their clot located within 7mm of the same-side ICA — consistent with a
   real ICA-level occlusion, where the MCA is genuinely invisible on the scan
   because no blood is reaching it, not a segmentation failure. Those 13 are
   already correctly classified as ICA occlusions. 4 more have *both* ICA and
   MCA missing on that side, suggesting an even more complete blockage. Only 3
   remain unexplained this way (real distance >25mm to the ICA) — a small,
   acceptable residue.
3. **The 2 flagged subjects, manually reviewed.** Rendered NCCT slices with
   the clot location and final infarct overlaid for sub-stroke0049 and
   sub-stroke0079. Both visually confirm the known side mismatch; neither
   shows a clear, fixable labeling error — sub-stroke0049's clot sits near
   midline structures where a small side mix-up is plausible without being a
   bug, sub-stroke0079's is starker and can't be resolved from an image slice
   without a radiologist. **No relabeling done** — both stay counted as the
   real exceptions the laterality gate (94.7%, not 100%) already reflects.

Phase 2 is now fully closed except for the deferred territory rule and the
lowest-priority item (CTA vessel sharpening), neither blocking Phase 3.

## Decision needed before continuing

The next planned step (`CLAUDE.md`'s Phase 2 list) was a coarse **territory
rule** — same idea as the failed overlap check, but using broad arterial
territories (MCA/ACA/PCA regions) instead of exact Tmax>6s pixels, on the
reasoning that distal embolization usually stays inside the same broad
territory even when it changes the exact location. That reasoning still
holds, but it is now a real bet, not a safe next step — the overlap check
was expected to be one of the sturdier tests, and it wasn't, and the cheap
rescue attempt for the collateral constraints just failed too. The territory
rule needs atlas registration (Liu 2023), the heaviest single task in
Phase 2, for a graph that currently has exactly one working rule to add to.

Two ways forward, not mutually exclusive:

1. **Build the territory rule anyway.** It might hold where exact overlap
   didn't — coarser tests are more forgiving of small location shifts. Real
   cost: the atlas registration work, with no cheap way left to de-risk it
   first, and no guarantee it passes either.
2. **Accept a narrower JDCR** built mainly on the laterality constraint, and
   put more of the project's weight on the GNN and cross-attention pieces
   (Phases 3–6) rather than on having many plausibility constraints. Report
   the testing process itself — 7 candidate rules formalized from the
   literature, rigorously tested against real data, only 1 surviving — as an
   honest finding about post-thrombectomy cohorts, not a shortfall. This is
   also a defensible, presentable result on its own: it says something real
   about why collateral grading and infarct-size prediction are hard in a
   uniformly-successfully-treated population, which is itself worth a
   paragraph in the write-up.

My honest read: the collateral proxy (HIR) is not going to work as a
plausibility signal in this cohort no matter how it's re-tested, and that's
settled now, not still open. The open question is only whether the
territory rule is worth the atlas-registration cost given everything else
built on final-infarct location or size has failed. I'd want a clearer sense
of the project's time budget before recommending either way.
