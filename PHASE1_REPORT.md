# Phase 1 Report — Knowledge Graph

**Status: PASSED** (2026-09-13). One open item carried to Phase 2 (below).

Phase 1's job, from `PROJECT_PLAN.md` §8: finalize both layers of the
knowledge graph, verify every rule against its source, audit whether the
clinical data can support the rules, and build the per-patient vessel
features with the confidence filter.

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| Both graphs load and validate | ✅ Pass | Layer 1: 12 vessels, 12 connections, connected, both carotids reach MCA+ACA, Acom bridges both sides. Layer 2: 16 concepts, 7 rules, 4 constraints, all references resolve. `outputs/tables/kg_validation.json` |
| Every rule has a checked citation | ✅ Pass | 7/7 sources verified against PubMed (NCBI E-utilities), ClinicalTrials.gov (NCT02142283, NCT02586415) or arXiv. **Caveat:** AHA/ASA 2026 recommendation *wording* unverified — full text blocked. |
| Confident cases land on the correct vessel | ✅ Pass — **94.7%** | Clot side vs. side of the follow-up infarct, 89/94 confident front-of-brain cases. Pass mark 80%, fixed before running. PCA 6/6. `outputs/tables/kg_node_features.json` |
| Time-window data audited | ✅ Done — **absent** | All 7 timing columns empty, 149/149. `outputs/tables/clinical_completeness.json` |

## What was built

| File | Purpose |
|---|---|
| `kg/cow_topology.json` | Layer 1 — Circle of Willis graph, verified sources, measured vessel presence, leakage rule |
| `kg/guideline_rules.json` | Layer 2 — concepts, 7 eligibility rules with quoted source text, 4 plausibility constraints with support status |
| `src/graph/build.py` | Validates both graphs, checks provenance, reports what's evaluable on ISLES'24, cross-checks against real data |
| `src/graph/features.py` | Per-patient vessel features + the laterality gate |
| `src/data/clinical_audit.py` | Completeness of every clinical column |
| `KNOWLEDGE_GRAPH_SOURCES.md` | Where every node and edge came from |

Per-patient features now available for all 149 subjects: which vessels are
present, which is occluded (confident matches only), and, for unlocalized
occlusions, a flag plus the side.

## Findings

1. **ISLES'24 has no timing data.** Every time-window condition in DAWN,
   DEFUSE 3 and the AHA/ASA guideline is non-evaluable. Combined with no
   ASPECTS, **no eligibility rule is fully evaluable on this dataset.** The
   Rule Evaluator will report "imaging and clinical criteria met; time window
   not recorded" — never "eligible."
2. **JDCR must rest on the plausibility constraints, and today only 1 of 4 is
   supported** (infarct side matches occlusion side, 94.7%). Two need Phase
   2's collateral proxy; one can't be checked (4 LVO-negative patients). **This
   is the main risk to the novelty claim** — see "Carried to Phase 2."
3. **The draft rules had three errors, now fixed.** DAWN was missing the
   premorbid mRS 0–1 condition; DEFUSE 3 was missing the age limits and both
   mismatch criteria; and all trial rules were checked against the model's
   *final infarct* volume when the trials use *acute core* at admission.
4. **Two invented thresholds removed.** The collateral constraints used 100
   mL / 5 mL cut-offs with no source; they now say "set from the training
   fold."
5. **The basilar default sink.** 19 of 20 "basilar" matches were
   low-confidence at a median 39 mm. The confidence filter keeps them out;
   the validator now flags any such sink automatically.
6. **Collateral vessels are often missing** (Pcom 44–54/149, Acom 71/149).
   This is anatomically normal and useful to the GNN — a missing collateral
   route is signal.
7. **Low-confidence clots still carry a reliable side** (17/17), now a
   feature (`unlocalized_side`).
8. **Leakage guarded.** Features use admission data only; the follow-up
   infarct mask is stored separately and used only to validate.
9. **Label 15** (TopCoW's rare 13th vessel) is in 5/149 patients and never
   the clot site — excluded from the fixed graph.

## Manual review (2026-09-14) — done, closed

| Subject | Why | Finding |
|---|---|---|
| sub-stroke0049 | Confident L-MCA clot, but 81% of the infarct is on the right | Visually confirmed: infarct sits on the opposite side from the clot marker, matching the numeric finding. The clot marker sits among the small midline vessel structures, close to where left/right nearly touch anatomically — a small side mix-up there is plausible without being an obvious error. |
| sub-stroke0079 | Confident R-MCA clot, but 98% of the infarct is on the left | Visually confirmed, more starkly: infarct is large and almost entirely on the opposite side. MCA is not normally as close to the midline as the vessels in 0049's case, so this is less easily explained the same way. Cannot be adjudicated as a real label error vs. an unusual clinical pattern (e.g. watershed injury) from one image slice — that call needs a radiologist. |
| sub-stroke0003, sub-stroke0033 | L-ACA clot, right-sided infarct | Not separately re-imaged — ACA sits at the midline by anatomy, so side ambiguity here is expected, same reasoning as 0049. |

**Decision:** no relabeling or exclusion. Both are already correctly counted
among the laterality gate's known disagreements (89/94 = 94.7%, not 100%) —
this review didn't find a clear bug to fix, it confirmed the exceptions are
real and are already accounted for in that number, not hidden by it.
Diagnostic images: `sub-stroke0049_check.png`, `sub-stroke0079_check.png`
(scratchpad, not committed — regenerate via the snippet in this session's
history if needed again).

## Follow-up check (2026-09-14): a candidate size rule — not supported

Tested: *clots in upstream vessels (ICA/M1) come with bigger infarcts than
distal clots.* Pass mark fixed before running: higher median and one-sided
Mann–Whitney p < 0.05.

| Group | n | Median final infarct |
|---|---|---|
| ICA/M1 | 87 | 13.8 mL |
| Distal / beyond the circle | 42 | 13.1 mL |

**Result: fail** (p = 0.66, AUC 0.478). Recorded in
`kg/guideline_rules.json` as `not_supported` and excluded from JDCR. Source:
`outputs/tables/constraint_support.json` (`python -m src.graph.constraint_support`).

What it tells us:

- **Size rules are weak in this dataset; location rules aren't.** Every
  patient was successfully reperfused, so final infarct size reflects how fast
  flow returned more than where the clot was (reasoned, not tested). The
  location-based side rule held at 94.7%; this size-based rule failed. The
  two collateral rules are also size-based and may fail the same way.
  **Priority for Phase 2: location-based constraints — the territory rule
  first.**
- **Label noise in the "distal" class.** 11 of 42 have the clot-side MCA/ICA
  missing from the vessel mask (vs. 15/87 upstream), so some may be upstream
  clots in a vessel the CTA never showed. Rename the class
  **"unlocalized"** (location uncertain), not "distal".
- **Lead to re-test:** ICA infarcts are larger than MCA infarcts (median 25.7
  vs 11.0 mL, p = 0.035) — but it was one of two unplanned tests and doesn't
  survive correction. Re-test on training folds only.

## Carried to Phase 2

- Build **acute core** (relative CBF < 30%), **penumbra** (Tmax > 6 s),
  mismatch ratio/volume, and the **HIR collateral proxy** — these make the
  DAWN/DEFUSE 3 imaging conditions evaluable.
- **Add a territory constraint** — the occluded vessel's territory should
  match where the infarct is, using the Liu 2023 atlas. It's cross-task,
  clinically solid, and testable on real data. This is the main mitigation
  for finding 2.
- Support-check the two collateral constraints on the training fold. If
  fewer than 3 constraints hold in total, narrow the JDCR claim and say so.
- Confirm whether the 17 MCA-absent subjects are occluded MCAs not visible on
  CTA.
- Fix the final LVO label scheme (proposed: proximal-anterior 87 /
  other-confident 15 / distal-unlocalized 42 / none 4).
