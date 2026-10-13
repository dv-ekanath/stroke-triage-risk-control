# CLAUDE.md — KG-XAI project context

Read this first. Then `PROJECT_PLAN.md` (plan of record) and
`PHASE1_REPORT.md` (latest results and what's carried forward).

## What this project is

KG-XAI: Knowledge-Graph-Guided Multi-Task Learning for Stroke Triage.
Course project (Healthcare Analytics) — Ekanath DV (23MIA1023), Rohan Julius
Preetan (23MIA1160). Base paper: Luan et al., npj Digit Med 9:441 (2026).

A multi-task model (LVO detection & localization, collateral grading,
infarct segmentation) whose outputs are made clinically consistent with each
other by: a GNN over the patient's Circle-of-Willis vessel graph, guideline
knowledge injected into the decoder by cross-attention, and a consistency
loss. New metric: JDCR (Joint-Decision Consistency Rate). Improvement is
shown as B2 (full KG-XAI) vs B0 (base paper architecture reimplemented, no
graph) on identical data, splits and seeds — never against the base paper's
reported numbers (different labels, extra datasets, a 7-center split the
public data doesn't have).

History: Review 1 used a conformal-threshold approach; faculty said the
threshold wasn't enough novelty and to return to the knowledge graph and use
a GNN. Old Review 1 docs are in git history (commit 011a3ae), not the tree.

## Where things are (Windows machine)

| What | Where |
|---|---|
| Repo | `C:\Users\Admin\Downloads\Stroke-code\Stroke-code` → github.com/dv-ekanath/stroke-triage-risk-control |
| Python env | `.venv\` (gitignored). torch 2.11.0+cu128, MONAI 1.6, nibabel, SimpleITK, TorchIO. torch_geometric 2.8.0.post1 (installed Phase 3). Run as `./.venv/Scripts/python.exe -m src....` |
| GPU | RTX 5060, 8 GB, Blackwell — needs cu128+ wheels |
| Dataset (not in repo) | `D:\ISLES-2024\train` (extracted from `E:\Nosql\train.7z`) |
| Preprocessing cache | `D:\ISLES-2024\cache_1mm` — clear it if `src/model/dataset.py` transforms change |
| Knowledge graph | `kg/cow_topology.json` (Layer 1), `kg/guideline_rules.json` (Layer 2) |
| Every reported number | `outputs/tables/*.json` |
| Baseline checkpoint | `outputs/checkpoints/segresnet_fold0_best.pt` (gitignored) |

Git identity for this repo: `dv-ekanath <ekanath.dv2023@vitstudent.ac.in>`
(set locally). Pushing works through the Windows credential manager.

## Status

- **Phase 1 — knowledge graph: PASSED** (2026-09-13). Both graphs validate;
  7/7 sources verified; laterality gate 89/94 = 94.7% (pass mark 80%).
- **Phase 2 — mostly done** (2026-09-14). Brain mask, acute core, penumbra,
  mismatch, HIR: all built and gated (`src/graph/perfusion_volumes.py`). 5
  more plausibility constraints tested, including a continuous whole-cohort
  re-test of the two collateral constraints (n=143, rho=0.052, p=0.54 —
  decisively negative, not just underpowered). **Net: 1 of 7 plausibility
  constraints supported** (laterality, from Phase 1), 5 rejected, 1
  untestable. See `PHASE2_PROGRESS.md` for the full picture.
- **Decision made (2026-09-14): ~2 weeks available, not 6. Territory/atlas
  rule DROPPED from scope** (deferred to "if time remains after Phase 8," not
  required) — three related location/size tests already failed for the same
  reason (universal successful reperfusion), and it needs the heaviest
  remaining setup (atlas registration) with no cheap way to de-risk it first.
  B0 also simplified: extends the Review 1 trunk instead of building 3 new
  encoders. Evaluation protocol: **3 folds × 3 seeds** (9 runs) for B0 and
  B2, not 5×3. Full revised phase table, gates and day-by-day schedule:
  `PROJECT_PLAN.md` §8. Compute for the final evaluation: ~45–50 GPU-hours,
  start it as soon as B0/B2 exist (day 10 of 14), run unattended.

## Phase 2 — remaining work

Read `PHASE2_PROGRESS.md` first — it has the constraint-testing results and
why the plan below changed from the original Phase 2 list.

1. ~~Re-test the two collateral constraints properly~~ **Done — settled
   negative.** Continuous, whole-cohort (n=143): rho=0.052, p=0.54. Both
   constraints now `not_supported` in `kg/guideline_rules.json`, not
   `inconclusive`. HIR is not usable as a collateral signal in this cohort.
2. **Territory rule DROPPED from the 2-week scope** (see §"Decision made"
   above) — deferred to after Phase 8, not attempted now.
3. ~~LVO label scheme~~ **Done.** Renamed "distal_unlocalized" → "unlocalized"
   throughout (`src/graph/constraint_support.py`, `kg/guideline_rules.json`),
   values unchanged: proximal_anterior 87 / other_confident 15 /
   unlocalized 42 / none 4.
4. Skull stripping is DONE as a side effect of the brain mask
   (`src/graph/perfusion_volumes.py:brain_mask`, gated at 98.6% plausible
   ICV). CTA vessel enhancement (Frangi) — lowest priority, skip unless
   Phase 8 finishes early.
5. ~~Check the MCA-absent subjects~~ **Done.** Of 23 (not 17), 13 have their
   clot within 7mm of the same-side ICA — a real ICA occlusion explains the
   invisible MCA, not a segmentation failure; already correctly classified.
   4 more have both ICA and MCA missing (likely a more complete occlusion).
   Only 3 remain unexplained. See `PHASE2_PROGRESS.md`.
6. ~~Manually look at sub-stroke0049 and sub-stroke0079~~ **Done.** Rendered
   and visually reviewed. Both confirm the known side mismatch; neither shows
   a clear, fixable label error. No relabeling — both stay as known
   exceptions within the laterality gate's 94.7%. See `PHASE1_REPORT.md`.

**Phase 2 is now fully closed.** Only the deferred territory rule and the
skippable vessel-sharpening step remain, neither blocking Phase 3.

## Phase 3 — done, gate FAILED for the two new heads (2026-09-14)

Real 100-epoch run, `outputs/tables/b0_multitask_fold0.json`: segmentation
fine (dice 0.247–0.281, ≥ the 0.215 pass mark), but **LVO localization
(56.0%) and collateral proxy (60.0%) both finished BELOW their trivial
majority-class baseline** (58.8% / 66.7%) — worse than just always guessing
the common class. Recorded as a real gate failure, not rescued.

Root cause (two compounding issues, both fixed in Phase 4/5's B1 build):
1. `src/model/multitask.py`'s heads global-average-pool the WHOLE bottleneck
   into one vector before classifying — throws away location, which is
   fatal for a task that IS a location question.
2. Training crops are lesion-centered (`RandCropByPosNegLabeld`, 128x128x64)
   — the Circle of Willis (base of brain) can simply be outside that crop,
   while validation fed the whole resized brain instead — a train/val
   mismatch on top of the pooling problem.

## Phase 4/5 — B1 (GNN-based heads), built and smoke-tested, not yet run for real (2026-09-14)

Merged what was separately planned as "fix the Phase 3 heads" and "Phase 4
GNN" into one build, per-vessel features instead of one pooled vector:

- `src/model/dataset.py`: `build_transforms_gnn()` forks a second,
  **unaugmented, whole-brain** view (`image_wb`, fixed-size resize, so the
  CoW is always in-frame) alongside the existing lesion-centered patch;
  `ExtractVesselCoordsd` reads real per-node voxel coordinates straight off
  the (resized-in-lockstep) cow-msk, `attach_cow_msk()` locates the file.
  148/148 readable subjects have one — nobody lost.
- `src/model/gnn_head.py`: `sample_node_features()` trilinearly samples the
  encoder's stage-1 feature map (half-res, 32ch) at each of the 12 real
  vessel locations (`grid_sample`); `VesselGNNHead` runs 2 GATConv layers
  over the fixed anatomical edges (`kg/cow_topology.json`), then a masked
  mean+max readout feeds the LVO/collateral heads. Absent vessels: sampled
  feature zeroed, `node_present=0` flag kept as a real GNN input (per the
  graph file's own note: absence is clinically meaningful, not noise).
- `src/model/multitask_gnn.py`: `MultiTaskGNN`, same trunk/warm-start as B0,
  `forward_seg` unchanged, `forward_gnn` is the new path.
- `src/model/train_multitask_gnn.py`: B1 training script, writes
  `outputs/tables/b1_multitask_gnn_fold{N}.json`.
- **Run 1 (real, 100 epochs, no augmentation): FAILED.**
  `outputs/tables/b1_multitask_gnn_fold0.json`. Dice fine (0.232, ≥ 0.215).
  LVO 52.0% (< 58.8% baseline, WORSE than B0's naive pooling), collateral
  62.0% (< 66.7% baseline). Collateral accuracy was identical at all 10
  validation checkpoints despite falling loss — first sign of collapse.
- **Run 2 (real, 100 epochs, + intensity-only wb-branch augmentation +
  higher aux-loss weight, lambda_lvo 0.3→0.6 / lambda_col 0.2→0.4): FAILED,
  more decisively.** `outputs/tables/b1_multitask_gnn_fold0.json` (run 1
  preserved as `..._noaug.json`). Same Dice/accuracy numbers. Direct
  inspection of the checkpoint's actual per-subject predictions (not just
  aggregate accuracy) confirms **total collapse**: predicts the exact same
  LVO class (2/proximal_anterior) and the exact same collateral bin (1) for
  **all 50 validation subjects**, zero exceptions. It has learned only the
  training-fold class prior, nothing image-dependent, on both heads,
  regardless of architecture (pooling vs GNN) or training recipe (plain vs
  augmented+reweighted).
- **Run 3 (frozen-trunk diagnostic, `--freeze-trunk` in
  `train_multitask_gnn.py`): FAILED identically, and this one is decisive.**
  `outputs/tables/b1_multitask_gnn_frozen_trunk_fold0.json`. Locked the
  entire warm-started trunk (4.7M params), trained ONLY the GNN head's own
  9,928 params against it, to rule out segmentation's loss dominating/
  starving the auxiliary heads of gradient. Segmentation confirmed
  unchanged (dice 0.288, matching the trunk's original quality, computed
  once at the end since it's frozen). **LVO 52.0%, collateral 62.0% —
  exactly the same numbers as the un-frozen run, identical at every
  checkpoint again.** Giving the heads sole, undivided access to the
  trunk's features changed nothing.
- **Conclusion: this is a real data-scale/task-difficulty ceiling, not an
  architecture bug and not a gradient-competition problem** — three
  different training regimes (naive pooling, GNN, GNN+augmentation+
  reweighting, GNN+frozen trunk) all land on the identical trivial
  solution (verified by inspecting actual per-subject predictions, not
  just aggregate accuracy: the model predicts ONE constant class for all
  50 validation subjects on both heads). Recommended: stop iterating on
  these two heads — report as a finding (see PHASE2_PROGRESS.md's finding
  for the matching honest-negative-result precedent), put the project's
  main "does KG-XAI help" claim on segmentation quality via Phase 5's
  cross-attention graph injection instead, where B0 has a real working
  baseline to beat.
- **Candidate extra dataset: CODEC-IV — access request SENT 2026-09-15,
  follow-up email sent ~2026-09-19, NO REPLY as of 2026-10-03 (18 days,
  2+ weeks since the follow-up).** Decision: stop waiting on it. Treat as
  non-responsive, not as a pending track — it is not in the plan, the
  report, or the deck as a thing that will arrive. If a reply ever comes,
  evaluate integrating it then; nothing is scheduled around it. — Werdiger
  F et al.,
  NeuroImage 2023;271:119985, PMID 36933627 (verified via E-utilities). 159
  patients, multiphase CTA derived from CTP, expert-annotated clot location
  + hemisphere + **degree of collateral flow** (the real collateral grade
  ISLES'24 lacks). Access by request form (not open download, no stated
  turnaround): https://github.com/MBC-Neuroimaging/ClotDetectEval — the user
  has to submit it. Only CTA-type imaging (no NCCT/perfusion maps, likely no
  CoW masks), so it would feed a CTA-only LVO/collateral branch, not the
  6-channel trunk. Ruled out: UniToBrain (CTP + core/penumbra only), Zenodo
  19350665 (spreadsheet, no images), IACTA-EST 2023 (challenge closed, data
  access unclear).

## Phase 5 — B1(seg): graph-injected segmentation, built and smoke-tested,
real run in progress (2026-09-14)

Redirected the "does the knowledge graph help" claim onto segmentation
(Dice), where there's enough signal to test it fairly — the auxiliary-head
attempt is shelved (see Phase 4/5 above).

- `src/model/gnn_head.py`: refactored — `VesselGNNEncoder` now returns raw
  per-node embeddings (no pooling/heads); `VesselGNNHead` (shelved) wraps it
  with the old pooled-classification path for reference.
- `src/model/segresnet_graph.py` (NEW): `GraphCrossAttention` — decoder
  feature map (voxels as queries) attends over the 12 graph node embeddings
  (keys/values); cheap regardless of voxel count since there are only 12
  keys. **Zero-init residual gate**, verified: at init the model's
  segmentation output is byte-identical (max diff 0.0) to the plain warm-
  started trunk — the graph pathway can only ever improve on a known-good
  baseline, never silently break it. `GraphInjectedSegResNet` reimplements
  SegResNet.decode()'s loop by hand to run cross-attention after each
  up-sampling stage; same trunk/warm-start as B0.
- `src/model/train_graph_seg.py` (NEW): trains it against B0's dice=0.247
  reference. Reuses Phase 4/5's dataset pipeline unchanged. Validation
  computes the graph embedding once per subject (whole-brain quantity) and
  reuses it across sliding-window inference's many overlapping patches.
- Smoke-tested: gates move under real gradient (not just harmless at init),
  6-subject/2-epoch real run finite throughout, sliding-window val with the
  once-per-subject node-embedding closure worked.
- **Real 100-epoch run launched, in progress** — writes
  `outputs/tables/b1_graph_seg_fold0.json`. Gate: Dice should beat B0's
  0.247 to count as the graph actually helping (no formal pass mark fixed
  beyond "beats B0" since this is the direct head-to-head comparison the
  whole B0/B1/B2 design exists for).

## Phase 6 — B2: + laterality consistency loss, real run in progress (2026-09-14)

Adds the ONE plausibility constraint real data supports
(`infarct_side_matches_occlusion_side`, soft, 5.3% real violation rate) as a
differentiable loss on top of B1(seg).

- `src/model/laterality_targets.py` (NEW): per-subject expected infarct
  side, from admission-only data (Phase 1's `nearest_vessel`/
  `unlocalized_side`, never lesion-msk). 124/148 subjects covered
  (70 left / 54 right).
- `src/model/dataset.py`: `ExtractVesselCoordsd` now also emits
  `midline_voxel_x` — computed directly in the wb-grid's own voxel-index
  space (median of L/R vessel-pair midpoints), not world mm, so no
  affine/world-coordinate tracking through the random patch crop is needed.
- `src/model/consistency_loss.py` (NEW): `laterality_consistency_loss` —
  penalizes the FRACTION of predicted foreground probability mass on the
  wrong side of the midline, over the whole-brain prediction. Unit-tested
  against synthetic known-correct/known-wrong predictions (correct-side
  loss ≈0.007, wrong-side ≈0.993, unknown-side excluded exactly 0) — the
  test caught a real bug in the test's own synthetic setup (ambiguous
  50/50 background) before it could hide anything real.
- `src/model/segresnet_graph.py`: added `forward_wb_seg_and_graph` — one
  `encode(x_wb)` pass reused for both the node embeddings (B1's need) and a
  whole-brain segmentation prediction (B2's need), rather than encoding
  x_wb twice.
- `src/model/train_graph_seg_consistency.py` (NEW): `loss = seg_loss(patch)
  + lambda_consistency * laterality_consistency_loss(wb)`, lambda=0.1
  (deliberately small — soft constraint).
- **Hit an OOM on the first real attempt**: decoding the whole-brain volume
  in full (needed for the consistency loss) roughly doubles decode memory
  vs B1(seg), which only ever encoded x_wb. Fixed by shrinking the
  whole-brain grid to `--wb-size 64 64 48` (patch size unchanged at
  128×128×64) — the consistency loss only needs coarse left/right mass, not
  full resolution. Separate cache dir (`cache_1mm_wb64`) since wb_size
  changed. 2-epoch smoke test then ran clean, consistency loss visibly
  dropping (0.095→0.062 train, 0.028→0.019 val).
- **Real 100-epoch run: done.** `outputs/tables/b2_graph_seg_consistency_fold0.json`.
  Compared across every checkpoint, not just the endpoint:

  | | best | final | mean (10 ckpts) | spread |
  |---|---|---|---|---|
  | B0 (plain) | 0.281 | 0.248 | 0.257 | 0.236–0.281 |
  | B1 (graph only) | 0.271 | 0.240 | 0.244 | 0.216–0.271 |
  | B2 (graph+consistency) | 0.271 | 0.236 | **0.251** | 0.236–0.271 |

  B2 > B1 on mean dice, but neither beats B0 in this single run. Real
  secondary finding: B2's checkpoint-to-checkpoint spread (0.035) is
  noticeably tighter than B1's (0.055) — the consistency loss seems to
  stabilize training even without raising the ceiling. Still n=1
  fold/seed — Phase 8 is what actually answers "does the graph help."

## Phase 7 — Rule Evaluator, JDCR, rationale: done (2026-09-14)

Scope, deliberately narrowed to what's real: JDCR is built on the ONE
supported plausibility constraint (laterality) — the other 6 are
`not_supported`/untestable (Phase 2) and correctly excluded, same as the
consistency loss. Eligibility rules stay unevaluable (no timing data).

- `src/eval/rule_evaluator.py` (NEW): `evaluate_laterality` (per-patient
  hard pass/fail using the actual predicted mask, not the soft loss),
  `compute_jdcr`, `generate_rationale` (deterministic template, no LLM).
- `src/eval/run_rule_evaluator.py` (NEW): runs any of the 3 real checkpoints
  (B0/B1/B2) on the val split's whole-brain prediction, computes JDCR,
  prints/saves 10 sample rationales.
- **Gate hit on the first real check**: manually reviewing B0's 10 sample
  rationales caught a real bug — the rationale showed `right_fraction`
  verbatim even when `predicted_side=="left"`, so a left-predicting
  subject's sentence quoted the RIGHT-side percentage while describing the
  left side (e.g. "0.0% ... sit on the left side" when it was actually
  100%). Fixed (report the share ON the named side); re-verified clean
  across all 10 samples for all three models.
- **JDCR results** (`outputs/tables/jdcr_{b0,b1,b2}_fold0.json`):

  | | JDCR | evaluable/total | coverage |
  |---|---|---|---|
  | B0 | 1.0 | 27/50 | 54% |
  | B1 | 1.0 | 23/50 | 46% |
  | B2 | 0.952 | 42/50 | **84%** |

  **Important caveat, not hidden**: "not evaluable" includes subjects where
  the model predicted ZERO lesion voxels — excluded from JDCR's denominator
  rather than penalized. B0/B1 hit a trivial JDCR=1.0 partly BECAUSE they
  abstain (predict nothing) on roughly half the cohort. B2 commits to a
  real prediction for 84% of subjects (vs ~50%) and is still right 95.2% of
  the time on laterality — read JDCR together with coverage, not alone, or
  the real B2 story (far more patients get a usable, still highly
  consistent call) is hidden behind a shrunk denominator. The 2 B2 misses
  (sub-stroke0033, sub-stroke0185) are logged with full rationale text in
  the JSON.

## Phase 8 — COMPLETE (2026-09-18). Real result: PROGRESS_REPORT.md §6a, PHASE8_PLAN.md.

All 21 runs (18 trained + 3 reused per-fold trunks) scored and aggregated.
**Headline: no significant Dice difference B2 vs B0** (paired per-patient
Wilcoxon, p=0.24, n=345, using only B0's 7 non-collapsed runs). **Real
finding: B0 (trained from scratch) collapsed completely — predicted zero
lesion for every patient — in 2/9 runs; B1/B2 (warm-started) never
collapsed in 12/12 runs.** Disclosed confound: can't yet separate "the
graph stabilizes training" from "warm-starting stabilizes training" since
there's no warm-started graph-free B0 variant. Both results reported
plainly in PROGRESS_REPORT.md §6a, not adjusted to favor the project.

Survived a mid-run power loss (2026-09-17, ~1h20min lost, zero data lost —
every job writes its own file, backed up the progress log, fixed a resume
bug for next time). Also fixed two more real bugs while wiring up the
launch: `--seed` was silently controlling fold splits too (fixed: pinned
to a separate constant); none of the training scripts put seed in their
output filenames (fixed: seed 0 keeps old names, seeds 1+ get `_seedN`).

**Phase 9: deck corrections done** — `DECK_CORRECTIONS.md` (slide-by-slide,
including a full replacement "expected results" slide using Phase 8's real
numbers, a suggested new negative-findings slide, and one-line answers to
likely faculty questions). The original deck .pptx isn't in this repo
(only ever shared in conversation), so this is a corrections guide to
apply by hand, not an edited file.

**Interactive demo: built and running (2026-09-19)** — `demo/server.py` +
`demo/index.html` (+ `demo/README.md`). Local stdlib web app, live GPU
inference on the 50 fold-0 held-out patients, 4 real checkpoints (B2, B1,
B0 seed0, and the COLLAPSED B0 seed1 on purpose), rule check + template
rationale, compare-all-models view. Run: `python -m demo.server --root
D:/ISLES-2024/train` then open http://127.0.0.1:8765. Verified: API
endpoints exercised with real inference; page rendered and screenshotted
in headless Edge (default view and #compare view). NOT click-tested in an
interactive browser: slider, channel/overlay toggles, patient and model
switching share the same code paths but were not individually driven.
No uploads by design. All original-plan items are now done.

**CODEC-IV (2026-10-11 update): DEAD, final.** Request sent 2026-09-15,
follow-up ~2026-09-19, still no reply 26 days after the original request /
22 days after the follow-up. Decision (2026-10-11): stop treating this as
a live channel at all -- do not plan around it, do not send a third email,
remove it from the project's critical path entirely. The project proceeds
on open data only (AISD, TopCoW/TopBrain -- see above) for the remaining
time.

**"Strong novelty" push (2026-10-04 on): faculty wants more than the single-
dataset, graph-doesn't-beat-baseline story.** Found two genuinely open (no
request form) datasets: **AISD** (github.com/GriffinLiang/AISD, 397 NCCT +
lesion masks, different hospital/scanner) and **TopBrain**
(zenodo.org/records/15692630's successor, 90 CTA/MRA volumes, 48 labeled
vessels, extends TopCoW to the whole brain) -- scoped but not built: an
NCCT-only ablation of B0 trained on ISLES'24 and zero-shot-tested on AISD
(answers "do you have another dataset," cannot test B1/B2 since AISD has
no CTA/vessel masks); grounding the GNN's node features in a TopCoW/
TopBrain-pretrained vessel-anatomy model instead of our own 148-patient
encoder (the bigger, not-yet-started swing at the actual novelty gap).

**Structural collateral reserve -- tested and REJECTED (2026-10-04).**
`src/graph/constraint_support.py` checks 6-8: a pure CoW-graph-topology
collateral measure (intact Acom/Pcom route, no occlusion label, no learned
component -- can't collapse like the GNN heads did) vs final infarct
volume (n=144, 2 formulations) and vs admission NIHSS (n=141, chosen
specifically to dodge the universal-reperfusion confound that sank the
volume-based tests). All three pre-registered, all three null, one in the
wrong direction. Likely reason: Acom/Pcom are only segmented in 48%/36%/
30% of subjects even when physiologically present, so "absent" is mostly a
segmentation-sensitivity artifact, not a clean anatomical signal. Recorded
in `kg/guideline_rules.json` as `structural_collateral_reserve`,
`not_supported`. This is the second independent collateral operationalization
to fail (after HIR) -- treat "does collateral status predict outcome here"
as settled, not underpowered. Remaining untried levers for the graph's
actual contribution: random-edge-GNN ablation (does real CoW anatomy beat
an arbitrary graph of the same size?), the TopCoW/TopBrain feature-grounding
idea above, and the AISD cross-dataset generalization test.

**Random-edge-GNN ablation -- run and REJECTED (2026-10-04).** Real CoW
anatomy vs. a random same-size graph (12 nodes, 12 edges), B1 only (no
consistency loss), fold 0, 3 seeds each. Code: `random_edge_index()` in
`src/model/gnn_head.py`, threaded through `VesselGNNEncoder`/
`GraphInjectedSegResNet`/`train_graph_seg.py --edge-mode random`/
`run_rule_evaluator.py --edge-mode random` (default stays real anatomy,
zero risk to existing B1/B2 checkpoints). Real-anatomy mean final Dice
0.2247 (sd 0.0024), JDCR strict mean 0.969; random-edge mean Dice 0.2263
(sd 0.0008), JDCR strict mean 0.977 -- random is marginally HIGHER on both,
well within noise, not testable at n=3 (disclosed before running). No
measurable benefit from the specific anatomy over an arbitrary graph of
the same size. Supporting diagnostic: cross-attention gate magnitudes
(saved in training history) are tiny in BOTH arms, same order of
magnitude either way -- consistent with very little information passing
through this pathway regardless of its wiring. Full numbers:
`outputs/tables/random_edge_ablation.json`. Implication: further
graph-STRUCTURE tweaks are unlikely to help; the TopCoW/TopBrain
feature-grounding idea (give the pathway genuinely new information, not
just a different wiring of the same under-informative node features) is
the more promising remaining lever.

**Two off-graph-axis ideas, run 2026-10-11 -- BOTH PASS, both genuinely
new positive findings.**

1. **Triage/abstention check** (`src/eval/triage_check.py`) -- does the
   Rule Evaluator's own ground-truth-free output predict which
   predictions are actually wrong? **CORRECTED 2026-10-11: the original
   3-criterion version (laterality inconsistency, near-zero volume, high
   ECE) wrongly called ECE "ground-truth-free" -- `expected_calibration_error`
   takes the true label directly, so it's computable here only because
   this is a validation set with known outcomes, and it's partly circular
   with Dice. The 2-criterion version below (laterality + near-zero
   volume only) is the one actually used anywhere else in this project.**
   Joined Dice against these signals for all 1038 (subject, run) pairs
   across the 21 real Phase 8 checkpoints, thresholds fixed from the
   signal distributions alone before touching Dice. Corrected result:
   flagged median Dice 0.0000 vs unflagged 0.1995, p=1.4e-50, AUC=0.89 --
   PASS, and STRONGER than the flawed 3-criterion version (p=1.3e-32,
   AUC=0.73), not weaker -- dropping the circular criterion sharpened it.
   Holds within EVERY model type separately (b0 p=6.9e-42, b1 p=0.0055,
   b2 p=1.5e-05), including b1/b2 which never had a single collapsed run
   -- so this is not just rediscovering the known B0 collapses, it's a
   real graded signal. Turns the Rule Evaluator from a pure evaluation
   metric into a usable "flag this
   prediction for radiologist review" triage layer. `outputs/tables/triage_check.json`.
2. **Ensemble fix for the B0 training-collapse finding** (`src/eval/ensemble_check.py`)
   -- unweighted average of the 3 existing seeds' softmax probabilities
   (no retraining), on fold0 and fold1 (the two folds with a real
   collapse, dice=0.02 each). Pre-registered pass mark: does the blind
   3-seed ensemble (including the broken member) come within 0.02 Dice of
   what you'd get from the 2 healthy seeds alone (the fair question --
   not "does it beat the collapsed seed," trivially yes). fold0: gap=0.0074
   PASS. fold1: gap=0.0110 PASS. The ensemble almost completely absorbs a
   catastrophic training failure without knowing in advance which seed
   broke. A complete, actionable result: found the instability (Phase 8),
   explained it (warm-start confound), now fixed it (ensembling), all
   disclosed honestly. `outputs/tables/ensemble_check.json`.

**Idea 3 -- NIHSS auxiliary regression head, run 2026-10-11 -- PASS, a
genuinely new positive multi-task result.** `src/model/segresnet_aux.py`
(`SegResNetWithAux` -- plain global-average-pooled head on the trunk
bottleneck, deliberately NOT the Phase 4/5 per-vessel GNN design: NIHSS is
a whole-brain severity score, not a location question, so pooling is the
architecturally right tool here, not a repeat of Phase 3's pooling bug)
and `src/model/train_seg_nihss.py`. Fold 0, seeds 0-2, warm-started from
the same B0 trunk. All 3 pre-registered marks:
- PRIMARY (Spearman rho(pred, true NIHSS) > 0, p < 0.05): PASS all 3
  seeds, rho 0.56-0.60, p < 3e-5 throughout -- not a one-epoch fluke.
- ANTI-COLLAPSE (pred_std > 0.3*true_std, guards against Phase 3's "one
  constant answer for everyone" failure mode): PASS all 3, pred_std
  3.1-3.3 vs true_std 6.05.
- SEGMENTATION GUARD (dice shouldn't drop > 0.02 vs plain B0 same seed):
  PASS seeds 0/1, borderline at seed 2 (-0.0197, just inside the line) --
  disclosed, not hidden.
Conclusion: Phase 3's heads failed because of label quality (circular/
sparse), not because multi-task learning is broken on this trunk -- given
a real, 98%-populated, non-circular label, the same architecture learns a
genuine signal with no collapse. Notable, NOT rigorously proven (n=1,
hypothesis only): seed1 is the exact fold/seed where plain B0 collapsed to
Dice=0.02; with the NIHSS head added it did NOT collapse (0.2288) -- a
third independent instance (after ensembling and the graph branch) where
something beyond pure single-task training coincided with avoiding that
failure mode. `outputs/tables/seg_nihss_fold0[_seed1,_seed2].json`.

A real risk encountered and fixed during the build: `SegResNet.decode()`
expects `down_x` reversed from what `.encode()` returns (deepest-first
vs shallowest-first) -- calling `decode()` directly without reversing
crashes on a channel-size mismatch. `segresnet_graph.py`'s manual decode
loop already reverses it for exactly this reason; `segresnet_aux.py` now
does too. Caught by a smoke test before committing GPU time, per this
project's standing practice.

**Preprocessing recipe (skull-strip + tight perfusion clip), confirmed
across 3 seeds 2026-10-12 -- the biggest Dice effect found anywhere in
this project, NOT a novelty claim.** Motivated by reading the actual
ISLES'24 challenge winner's own account of what won it (arXiv:2505.18424,
Kurtlab, dice=0.285 on the hidden test set vs. our plain-recipe ~0.24) --
they credit ~10 validation-Dice points to skull-stripping + clinically-
motivated perfusion windowing, which our pipeline had neither of
(`PERFUSION_CLIP` only killed extreme artifact outliers, never a
clinical core/penumbra window; no skull-strip step in the segmentation
pipeline at all). Added `_SkullStripd` (reuses the already-validated
brain_mask from `src/graph/perfusion_volumes.py`, simpler than their
SynthStrip) and `TIGHT_PERFUSION_CLIP` (CBV 0-15->0-3 mL/100g, MTT
0-30->0-15s, toward their literature anchors; CBF/Tmax left alone --
their thresholds for those are relative, not a simple clip) to
`src/model/dataset.py`, gated behind `--skull-strip`/`--tight-perfusion-clip`
on `train_baseline.py`, off by default (zero risk to existing B0/B1/B2).
Fold 0, 3 seeds: **Dice 0.2739 / 0.2667 / 0.2791, mean 0.2732 (sd 0.0051)**
-- vs. the plain recipe's clean-seed mean 0.2403, a real +0.033 gain,
now within 0.012 of Kurtlab's actual winning score. Volume-diff (21.8mL)
and lesion-F1 (0.287) mean now BEAT Kurtlab's own numbers; lesion-count-
diff (9.97) still somewhat worse. One flagged-not-proven observation:
seed 1 (the exact seed that collapsed and never recovered under the
plain recipe) got stuck at the same 0.02 plateau for ~40 epochs here,
then broke out to 0.267 -- interesting, not controlled evidence this
recipe generally prevents collapse (that's the ensemble fix's job, a
separate, already-evidenced finding). Scope: fold 0 only (same disclosed
cut as B1), not yet the full 3-fold grid (~12 more GPU-hours, not run).
**Explicitly not presented as this project's novelty** -- directly
adapted from a published source; see PROGRESS_REPORT.md §6d and its
updated §9 novelty section for the honest separation between "what
makes the numbers better" (this) and "what is actually new here" (JDCR
as a proposed metric + the mechanism-level graph diagnosis).
Checkpoints/results: `*_skullstrip_tightclip*` in outputs/checkpoints,
outputs/tables. Mid-run survived a session restart that killed seed 2 at
epoch 74/100 (exit code 4, external termination, no data loss) --
restarted clean from the still-warm cache, re-ran in full.

**Three more tests, 2026-10-13 -- the most definitive evidence yet that the
graph pathway carries no useful information, from a genuinely new angle
(not more of the same check).**

1. **TopCoW expert-vessel-feature validation (free, no GPU).** Downloaded
   TopCoW's `CTA_ISLES2024_TUM.zip` (Zenodo 15692630) -- their team
   already ran expert vessel annotation directly on 26 of OUR OWN
   ISLES'24 subjects. Compared their expert presence/absence calls
   against our own auto-derived `segment_present`: **91.8% overall
   agreement** (182 comparisons), strong on every vessel (88-100%) except
   Acom specifically (76.9%, we under-detect it). This directly weakens
   the premise for the full TopCoW feature-grounding build (training a
   vessel segmenter on the 10.5GB main release, 5-7 days) -- our features
   were never badly wrong across the board, so there's less to fix than
   assumed. `outputs/tables/topcow_expert_vessel_validation.json`.
2. **Does feature disagreement with experts predict worse Dice?** (free,
   no GPU, n=25 of the 26 subjects, matched to real Dice from existing
   Phase 8 checkpoints). Pre-registered: rho<0, p<0.05. Result: FAIL for
   both B0 (rho=-0.13, p=0.53) and B2 (rho=-0.13, p=0.55) -- no evidence
   our feature quality (such as it varies) actually affects segmentation
   outcome. Confirms the TopCoW de-prioritization from check 1.
   `outputs/tables/topcow_feature_quality_vs_dice.json`.
3. **Graph completeness (0-12 vessels present) vs Dice** (free, no GPU,
   n=1038 subject-run pairs across all 21 Phase 8 checkpoints).
   Pre-registered: rho>0, p<0.05. Result: FAIL -- significant but in the
   WRONG direction (rho=-0.09, p=0.003): more complete graphs correlate
   with slightly LOWER dice. Small effect, likely a lesion-severity
   confound or noise from non-independent repeated-subject observations,
   not chased further. `outputs/tables/graph_completeness_vs_dice.json`.
4. **Force the cross-attention gate open** (`gate_init` param on
   `GraphCrossAttention`/`GraphInjectedSegResNet`, `--gate-init` flag on
   `train_graph_seg.py`, default 0.0 = unchanged behavior everywhere
   else). Every prior run only showed the gate staying near zero
   *on its own* -- never tested whether an open gate would actually help
   if training started there. B1, fold 0, 3 seeds, gate_init=1.0:

   | seed | gate-open dice | zero-init dice | diff |
   |---|---|---|---|
   | 0 | 0.2272 | 0.2270 | +0.0002 |
   | 1 | 0.2244 | 0.2257 | -0.0014 |
   | 2 | 0.2204 | 0.2213 | -0.0009 |

   Mean diff -0.0007. **The gates stayed open the entire 100 epochs in
   all 3 seeds** (final values ~1.0-1.03, never drifted back toward
   zero) -- so the zero-init gate never opening on its own is NOT the
   model actively suppressing the pathway; it simply never had a reason
   to move. But an open, stable gate produces essentially identical Dice
   to a closed one. Combined with the random-edge ablation (doesn't
   matter what it's wired to) and this (doesn't matter how open it is),
   this is about as complete a mechanism-level case as this project can
   build: the graph pathway, as designed, has nothing useful to
   contribute to segmentation accuracy, under every architectural
   condition tried. `outputs/tables/gate_open_experiment.json`.

## Candidate paper direction, started 2026-10-13: knowledge-grounded silent-failure detection

With the graph's accuracy claim fully closed (random-edge + forced-gate-open
+ feature-quality checks above), the strongest remaining candidate for an
actual new contribution is the triage finding (CLAUDE.md "Five further
tests", triage_check.py) reframed as a proper method, benchmarked against
standard uncertainty baselines rather than just validated on its own. User
asked to pursue this (while away 6h); done autonomously, documented here
for review on return.

**Core question:** does the knowledge-grounded flag (laterality
inconsistent OR predicted volume < 1mL, p=1.4e-50 on its own) actually add
something over a standard statistical uncertainty baseline, or is it
redundant with information an ensemble already gives you for free?

**New code:** `src/eval/failure_detection_baselines.py` (collects, per
(subject, seed), for B0's 3 existing seeds x fold0+fold1: dice, the
knowledge flag, a standard ensemble-disagreement signal [1 - mean
pairwise Dice between the 3 seeds' own predictions -- the most standard,
most-cited uncertainty baseline, "deep ensembles"], and predicted-volume
coefficient of variation). `src/eval/mc_dropout_baseline.py` (a second
baseline, B0 fold0 seed0 only -- MC-dropout multiplies cost by N passes,
didn't fit the same scope as the ensemble baseline in one GPU session;
also disclosed: this SegResNet has exactly ONE Dropout3d layer total, not
dropout at every layer, checked directly -- a real architecture
constraint on how strong this baseline can be, not a choice made to
weaken it). `src/eval/failure_detection_compare.py` (the actual
comparison: AUROC for detecting severe failure, dice<0.05 fixed before
running, plus a complementarity analysis of which method catches which
failures).

**Result (n=297 subject-seed pairs, fold0+fold1, 156 severe failures):**

| signal | Spearman rho vs dice | AUROC (severe-failure detection) |
|---|---|---|
| knowledge flag | -0.754, p=9e-56 | **0.836**, p=5e-33 |
| ensemble disagreement | -0.214, p=0.0002 | 0.583, p=0.007 |
| volume CV (ensemble) | -0.186, p=0.001 | 0.575, p=0.013 |

The standard ensemble-disagreement baseline IS a real signal here (passes
its own sanity check, rho<0 and p<0.05) -- not a strawman -- but the
knowledge flag beats it by a wide margin, 0.836 vs 0.583 AUC. **Most
striking finding:** of the 156 real failures, the knowledge flag alone
catches 69.2% (108), ensemble disagreement alone catches only 32.7% (51).
Of the 76 cases the knowledge flag catches that disagreement MISSES, 76/76
(100%) are real failures -- essentially zero false positives in that
unique catch. Disagreement's own unique catches (40 cases) are only 19/40
(47.5%) real failures -- much noisier. Together they catch 81.4% of
failures; 18.6% are missed by both (a disclosed, honest residual, not
hidden).

**MC-dropout baseline -- a more nuanced, important result, NOT just a
repeat win.** Scoped to fold0/seed0 only (50 subjects, N=6 passes, one
Dropout3d layer -- see disclosed scope/architecture limits above). This
is a DIFFERENT, smaller, single-model subset than the 297-row ensemble
comparison (no collapsed-seed cases in it at all), so its numbers are
not directly comparable to the ensemble ones above -- only each baseline
to the knowledge flag within its own scope:

| signal (this 50-subject subset only) | AUROC |
|---|---|
| knowledge flag | 0.652 |
| mc_dropout_disagreement | 0.756 |
| **mc_dropout_volume_cv** | **0.819** |

On THIS subset, MC-dropout's volume coefficient-of-variation BEATS the
knowledge flag outright. Honest, disclosed interpretation (reasoned, not
separately tested): the ensemble comparison's dataset includes the
catastrophic collapsed-seed1 cases, which the near-zero-volume half of
the knowledge flag catches almost by construction -- that's a big, easy
chunk of its 0.836 AUC there. This 50-subject subset has NO collapse
cases at all, only a normally-behaving model's ordinary failures, and on
that harder, more realistic case the statistical signal is competitive
or better. **The honest conclusion is not "knowledge-grounded checks
always win" -- it's narrower and more interesting: they excel
specifically at catching catastrophic, structurally-obvious failures
(total collapse, clear laterality violations) that have an unambiguous
domain signature; for subtler within-normal-range uncertainty, a
well-chosen statistical signal can match or beat them.** That's a more
defensible, more publication-credible claim than an unqualified "ours is
better," and it's what a fair, broader validation (more folds, more
models, no cherry-picking which subset to report) would need to confirm
or refine.

`outputs/tables/failure_detection_raw.json` (the 297 raw rows),
`failure_detection_comparison.json` (the full comparison incl. MC-dropout),
`mc_dropout_raw.json` (the 50-row MC-dropout subset).

**What this is worth, honestly:** this is the strongest, most complete
candidate for an actual new contribution this project has produced --
not just "our check works" (already shown) but a real, specific,
falsifiable claim: knowledge-grounded checks dominate at catching
catastrophic, structurally-obvious failures (where they beat a standard
ensemble baseline by a wide margin, 0.836 vs 0.583 AUC, and have a
100%-precision unique catch), while a well-chosen statistical signal
(MC-dropout volume CV) is competitive or better at subtler,
non-catastrophic failures. That more nuanced version is MORE
publication-credible than an unqualified "ours wins," not less -- it
survived the temptation to only report the comparison that looked best.
Still a course-project-scale result (297 rows for the main comparison,
50 for MC-dropout, 2 of 3 folds, one architecture), not a
submission-ready paper on its own -- would need broader validation (more
folds/architectures, matched-scope MC-dropout, maybe another dataset) to
be reviewer-proof, but the core comparison is real and the direction is
sound.

## Phase 8 prep — COMPLETE (2026-09-17, superseded by the above). Plan and pass marks: `PHASE8_PLAN.md`.

All 3 per-fold baseline trunks trained (`segresnet_3fold_fold{0,1,2}_best.pt`
-- these double as B0/seed0 for each fold, so Phase 8 only needs 6 more B0
runs, not 9 fresh). Hit and fixed a real bug along the way: `RandCropByPosNegLabeld`
attaches `foreground_start_coord`/`foreground_end_coord` metadata that's a
`Tensor` for a normal crop but a plain `numpy.ndarray` for an all-background
one -- crashed collation whenever a batch paired both types, which only
started happening once fold 2 introduced a new subject mix. Fixed for real
(`DeleteItemsd` drops both unused keys right after the crop, in both
`build_transforms` and `build_transforms_gnn`) -- not a per-run workaround,
every future fold/seed benefits. ECE was also actually wired into
`src/eval/run_rule_evaluator.py` (existed since fix 8.2 but was never
called from a real path until now). B1, B2, and the Rule Evaluator all
smoke-tested clean on fold 0's fresh trunk. **Full detail, and the exact
remaining run count/schedule: `PHASE8_PLAN.md`.**

**Launched 2026-09-17** — `python -m src.model.run_phase8 --root D:/ISLES-2024/train`,
running as one background task that sequentially runs all 18 remaining jobs
(orchestrator: `src/model/run_phase8.py`; progress:
`outputs/tables/phase8_run_log.json`, updated after every job). Found and
fixed two more real bugs in the last mile: (1) `--seed` was being passed
straight into the fold-splitting function, so varying it for seed-variance
runs would have silently changed WHICH SUBJECTS are in each fold too --
fixed by hardcoding the fold-split seed to 0 always, `--seed` now only
drives training randomness; (2) none of the three training scripts
included seed in their checkpoint/result filenames, so every B2 seed would
have overwritten the same file -- fixed. Also fixed `run_rule_evaluator.py`'s
B0 loader, which was pointed at the wrong (shelved, Phase 3) architecture.
**When picking this up in a new session: check `outputs/tables/phase8_run_log.json`
for progress before doing anything else** -- do not relaunch
`run_phase8.py` without checking how far it got (it does not auto-resume;
use `--start-at N` to skip already-completed jobs by index if restarting
after an interruption).

- **Fix 8.2 (JDCR scale mismatch): DONE, re-run, changed the conclusion.**
  All three models now score identically (JDCR 0.977, coverage 100%) once
  scored on their real full-resolution prediction instead of a downscaled
  view. The earlier "B2 commits to more patients" story was the scoring
  bug, not a real B2 property. Full detail: `PHASE8_PLAN.md`.
- **Fix 8.1 (validation overlap): code done, retraining in progress.**
  `train_baseline.py` now names per-n_folds checkpoints separately
  (`segresnet_{n_folds}fold_fold{N}_best.pt`) so it stops colliding with
  the original 5-fold checkpoint. Fold-0 3-fold retrain (100 epochs, half
  the original 200 -- this is a warm-start, not a final result) launched
  2026-09-16, background task running.
- Also added: `expected_calibration_error` (ECE) in `src/eval/metrics.py`,
  voxel-level, explicitly NOT the same quantity as the base paper's 0.130
  (segmentation-only here, their heads are shelved) -- unit-tested on
  synthetic well-calibrated/overconfident cases before trusting it.
- `PROGRESS_REPORT.md` updated in place with the corrected JDCR numbers
  and fix status -- re-send/re-share if already given to faculty.

## KNOWN ISSUES — fix BEFORE launching Phase 8 (found 2026-09-14)

1. **Validation leakage in every Phase 3–7 run.** Every model warm-starts
   from `outputs/checkpoints/segresnet_fold0_best.pt`, the Review 1 trunk,
   trained on the **5-fold** split's fold-0 training set (118 subjects). The
   Phase 3–7 runs validate on the **3-fold** split's fold-0 val set (50
   subjects) — **20 of those 50 were in the checkpoint's training set.**
   Same `make_folds(seed=0)`, different `n_folds` → different partitions.
   Absolute Dice/JDCR from Phases 3–7 are therefore contaminated (likely
   optimistic); the frozen-trunk run's dice 0.288 (higher than every model
   trained further) is consistent with this. Relative B0/B1/B2 comparisons
   are less affected (all share the same warm start) but not clean either.
   The auxiliary-head failure conclusion is unaffected in direction (leakage
   could only have helped them, and they still failed).
   **Fix:** per fold, train a fresh baseline trunk on THAT fold's 3-fold
   training subjects only, and warm-start that fold's B0/B1/B2 from it (or
   train from scratch). Never reuse `segresnet_fold0_best.pt` in Phase 8.
2. **JDCR comparison confounded by evaluation scale.** `run_rule_evaluator.py`
   scores all three models on a 64×64×48 downscaled whole-brain prediction.
   Only B2 was trained to segment that view (wb 64×64×48 + consistency
   loss); B0 never saw a whole-brain view; B1 was trained with wb
   128×128×64 and its seg decoder only ever saw patches. B0/B1's high
   "predicted nothing" rate (46–54% coverage vs B2's 84%) may be this
   mismatch, not the consistency loss. **Fix:** in Phase 8, evaluate
   laterality on each model's full-resolution sliding-window prediction
   (the same one Dice uses), with the midline computed on that native grid
   (keep a native-resolution copy of cow_msk in `build_transforms_gnn`
   before the Resized step). Always report coverage next to JDCR.

## Rules this project runs on — keep them

These are what made the work defensible. Don't relax them.

- **Every phase ends with a gate. Fix the pass mark BEFORE running.** A failed
  gate blocks the next phase. Don't rescue a failed test by changing groups or
  thresholds afterwards — record it as failed (see
  `proximal_occlusion_larger_infarct` in `kg/guideline_rules.json`).
- **Leakage:** model inputs and graph features use admission (ses-01) data
  only. Never derive a feature from `lesion-msk` or any ses-02 file — that is
  the target. `lesion-msk` may only *validate*.
- **Citations:** verify against primary records — PubMed via NCBI E-utilities
  (`eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=...`),
  ClinicalTrials.gov API v2, arXiv. Never cite a threshold from memory. The
  AHA/ASA 2026 guideline's recommendation *wording* is still unverified (full
  text blocked).
- **No invented thresholds.** Anything data-driven is fitted on training folds
  only.
- **Occlusion confidence gating (≤ 15 mm).** BA is a default sink — 19/20
  "basilar" matches are low-confidence artifacts.
- **Every number in a document traces to an `outputs/tables/*.json` file.**
- Validate on a small/controlled case before trusting the full pipeline.

## Dataset facts and gotchas (all measured)

- 149 subjects, **2 centers** (99 / 49; sub-stroke0075 blank). Not 7.
- **No timing data** (all 7 timing columns empty) and **no ASPECTS** → no
  eligibility rule is fully evaluable; report per-criterion, "time window not
  recorded".
- **No collateral grade** in the release → HIR proxy, always called a proxy.
- Only **4** LVO-negative subjects → binary LVO head is not viable.
- Target is **final infarct** (day 2–9 DWI), not admission core/penumbra.
- sub-stroke0043: truncated CBF in the source release (excluded by integrity
  check). sub-stroke0016: cow/lvo mask shape mismatch. sub-stroke0142's
  lesion mask is `.nii`, not `.nii.gz`.
- Phenotype CSVs use the literal string `"nan"` for some missing values.
- MTT/CBF/CBV have extreme artifact values — clipped before normalization
  (they caused 29% NaN batches).
- TopCoW label 15 (3rd-A2 variant): 5/149 subjects, never the clot site —
  excluded from the graph.
- Raw 4D CTP has 44–60 frames per subject → use the derived perfusion maps.

## Working with the user

- Explain in simple, plain terms; they often ask for simpler wording.
- When they ask for a plan or an explanation, don't start building.
- Confirm before long GPU runs, big installs, or deleting files.
- Commit only when asked. Ask whether commits should carry a Claude
  co-author line (the first commit was re-titled "your clean commit message",
  which suggests it was removed — unconfirmed).
