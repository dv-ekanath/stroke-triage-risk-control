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
| Poor collateral (HIR) → bigger infarct, proximal only | **Inconclusive** | Right direction (19.97 vs 11.0 mL), but n=9 in the small group, p=0.134 |
| Poor collateral → non-trivial infarct (mirror) | **Inconclusive** | Same underlying test |
| Large infarct needs a clot | **Untestable** | Only 4 patients have no clot |

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

## Decision needed before continuing

The next planned step (`CLAUDE.md`'s Phase 2 list) was a coarse **territory
rule** — same idea as the failed overlap check, but using broad arterial
territories (MCA/ACA/PCA regions) instead of exact Tmax>6s pixels, on the
reasoning that distal embolization usually stays inside the same broad
territory even when it changes the exact location. That reasoning still
holds, but it is now a real bet, not a safe next step — the overlap check
was expected to be one of the sturdier tests, and it wasn't. The territory
rule also needs atlas registration (Liu 2023), the heaviest single task in
Phase 2.

Three ways to go from here, not mutually exclusive:

1. **Build the territory rule anyway.** It might hold where exact overlap
   didn't — coarser tests are more forgiving of small location shifts. Real
   cost: the atlas registration work, with no guarantee it passes either.
2. **Re-test the two inconclusive collateral constraints properly first** —
   continuous HIR-vs-volume correlation across the whole cohort instead of a
   binned, occlusion-level-restricted comparison. Cheap (no new data needed,
   just a better statistical test) and might turn "inconclusive" into
   "supported" without touching the atlas.
3. **Accept a narrower JDCR** built mainly on the laterality constraint, and
   put more of the project's weight on the GNN and cross-attention pieces
   (Phases 3–6) rather than on having many plausibility constraints. Report
   the testing process itself — 7 candidate rules formalized from the
   literature, rigorously tested against real data, only 1–3 surviving — as
   an honest finding about post-thrombectomy cohorts, not a shortfall.

My recommendation: do (2) now, since it's cheap and already scoped, then
decide on (1) vs (3) with that result in hand.
