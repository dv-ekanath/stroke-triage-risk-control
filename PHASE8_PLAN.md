# Phase 8 — full evaluation: plan and pass marks

Written 2026-09-16, **before** any Phase 8 run is launched. Per this
project's own rule (CLAUDE.md, "Rules this project runs on"): pass marks are
fixed before running, not adjusted after seeing results.

## Why this file exists

While preparing PROGRESS_REPORT.md, two problems were found in every
Phase 3–7 result:

1. **Validation overlap.** Every model warm-started from
   `outputs/checkpoints/segresnet_fold0_best.pt`, trained on the 5-fold
   split's fold-0 training set. Phases 3–7 validated on the 3-fold split's
   fold-0 val set. 20 of those 50 validation subjects were in the
   checkpoint's own training data.
2. **JDCR scale mismatch.** The Rule Evaluator scored laterality on a
   64×64×48 whole-brain view only B2 was ever trained to segment; B0/B1's
   high "predicted nothing" rate there likely reflected that, not the
   consistency loss.

Both are now fixed in code (see CLAUDE.md "KNOWN ISSUES" for the technical
detail). This file locks in how Phase 8 will be run and judged, before
launching it.

## Fixes applied (done, 2026-09-16)

- `src/model/train_baseline.py`: per-fold checkpoint/results naming
  (`segresnet_{n_folds}fold_fold{N}_best.pt`) so a 3-fold baseline run no
  longer overwrites the original 5-fold one.
- `src/model/dataset.py`: `build_transforms_gnn(..., native_midline=True)`
  computes `native_midline_voxel_x` on the same native-resolution grid
  sliding-window inference uses, not the downscaled wb view. Verified
  against the existing wb-grid midline on 4 real subjects — fractional
  position agrees to within 1%.
- `src/eval/rule_evaluator.py`: `evaluate_laterality` now distinguishes
  *why* a subject isn't evaluable (`unknown_side` vs `empty_prediction`);
  `compute_jdcr(strict=True)` (new default) counts an empty prediction
  against the model instead of excluding it, and reports `coverage`
  alongside `jdcr` so neither number is read alone.
- `src/eval/run_rule_evaluator.py`: scores every model's REAL sliding-window
  prediction (same one Dice uses) instead of a downscaled view; reports
  both `jdcr_strict` (headline) and `jdcr_lenient` (for continuity/
  comparison with the Phase 7 numbers already in the report).
- `src/eval/metrics.py`: `expected_calibration_error` (ECE) added,
  voxel-level, scoped to segmentation only — explicitly NOT the same
  quantity as the base paper's reported 0.130 (see its docstring).

## Fixed JDCR, re-run on the existing (fold-0, contaminated-split) checkpoints

Before touching per-fold retraining, re-ran all three existing checkpoints
through the fixed evaluator, to see what the fix itself changes:

| | JDCR (old, wb-grid) | JDCR (fixed, native-res) | coverage (old → fixed) |
|---|---|---|---|
| B0 | 1.00 (27/27) | 0.977 (42/43) | 54% → **100%** |
| B1 | 1.00 (23/23) | 0.977 (42/43) | 46% → **100%** |
| B2 | 0.95 (40/42) | 0.977 (42/43) | 84% → **100%** |

**The fix confirms the suspicion exactly, and changes the conclusion.** At
full resolution, all three models predict *something* for every evaluable
patient (coverage 100%, all three, identical), and all three are laterality-
consistent 42/43 times. The earlier "B2 commits to way more patients" story
was entirely the downscaled-view artifact — B0 and B1 were never really
abstaining, they were just being scored on a resolution they'd never
learned to segment. **On this single fold, JDCR shows no measurable
difference between B0/B1/B2.** Whether one emerges with real statistical
power is exactly what the 3×3 run below will show.

## Still to do before launch

- [x] Fold 0 baseline trunk (`segresnet_3fold_fold0_best.pt`) — done
      2026-09-16, best val dice 0.241, final 0.236 over 50 subjects.
      Confirmed it saved to its own file, original 5-fold checkpoint
      untouched.
- [ ] Fold 1 baseline trunk — hit a real bug, fixed operationally,
      retrying. **DataLoader cache-write race**: PersistentDataset's cache
      is subject-identity-keyed and shared across folds/n_folds (by
      design, safe on its own), but the first time a `num_workers>1`
      DataLoader encounters a subject NOT YET cached under a given
      cache subdirectory, two workers can race to write that subject's
      cache file concurrently and corrupt it (surfaced as
      `AttributeError: 'numpy.ndarray' object has no attribute 'numel'`
      deep in collation). Fold 0 didn't hit this (subjects already cached
      from earlier runs); fold 1 pulled in subjects new to the "train"
      cache folder. **Operational rule for the rest of Phase 8**: the
      FIRST run touching a new (fold, cache-subdir) combination uses
      `--num-workers 0`; once that run completes (cache fully populated,
      read-only from then on), later runs reusing the same fold/cache can
      safely go back to `--num-workers 4`.
- [x] Fold 1 baseline trunk (`segresnet_3fold_fold1_best.pt`) — done
      2026-09-17 after the cache-corruption retry (cleared and rebuilt the
      "train"/"val" cache dirs, `--num-workers 0`). Final dice 0.231 over
      49 subjects.
- [x] Fold 2 baseline trunk (`segresnet_3fold_fold2_best.pt`) — done
      2026-09-17, final dice 0.254 over 49 subjects. **All 3 per-fold
      baseline trunks now exist** (fold0 0.236, fold1 0.231, fold2 0.254).
      Note: these three runs ARE B0's seed-0 entries for each fold (not
      just warm-start prep for B1/B2) — `baseline_3fold_fold{N}.json`
      already covers B0/fold{N}/seed0. Phase 8 only needs 2 MORE B0 seeds
      per fold (6 more runs, not 9 fresh) to complete B0's 3×3 set.
- [x] (superseded description below, kept for the debugging trail) Fold 2
      baseline trunk — hit a SECOND, unrelated bug, real root cause
      found and fixed in code (not worked around). `RandCropByPosNegLabeld`
      attaches `foreground_start_coord`/`foreground_end_coord` metadata
      that's a `Tensor` for a normal crop but a plain `numpy.ndarray` when
      it falls back to an all-background crop (a subject with zero lesion
      voxels) — `EnsureTyped` was never applied to these two keys, so a
      batch pairing one of each type crashed collation. Confirmed via an
      isolated single-item check (297 calls, 3 passes, zero bad outputs —
      ruled out a per-item transform bug) and by reading `pad_list_data_
      collate`'s own failure, which named the exact two mismatched-type
      keys. **Fix**: `DeleteItemsd` drops both keys right after the crop
      (unused downstream) in both `build_transforms` and
      `build_transforms_gnn` — not just this one run; every Phase 8 script
      shares this crop transform and would have hit it on some other
      fold/seed eventually. `pad_list_data_collate` (tried first, didn't
      fix this specific error but is a reasonable general upgrade) kept in
      all 5 training scripts alongside the real fix. Retrying now.
- [x] B1 smoke test (fold 0, 3 epochs, 4 val subjects, warm-started from
      the new `segresnet_3fold_fold0_best.pt`) — clean, no crash, gates
      moved under real gradient. **Note**: this overwrote
      `outputs/tables/b1_graph_seg_fold0.json` (the file behind the real
      numbers already quoted in PROGRESS_REPORT.md) with meaningless
      smoke-test values. Not a data-loss concern (the real numbers are
      already text in the report, and Phase 8 regenerates a proper
      B1/fold0 result anyway) but noted for traceability honesty.
- [x] B2 smoke test (fold 0, 3 epochs, 4 val subjects) — clean, consistency
      loss visibly dropping (0.206→0.107 train). Same overwrite caveat
      applies to `b2_graph_seg_consistency_fold0.json` as B1's.
- [x] ECE wired into `src/eval/run_rule_evaluator.py` for real (it existed
      and was unit-tested but was never called from an actual evaluation
      path until now) — per-subject voxel-level ECE, mean+SD across the
      val set, written to the JSON payload alongside JDCR.
- [x] Rule Evaluator (native JDCR + ECE) smoke test against the smoke-
      tested B2 checkpoint — clean: JDCR 0.977 (42/43, coverage 1.0,
      consistent with the earlier real-checkpoint numbers), ECE
      0.0187 ± 0.0060 over 50 subjects, ran without error.

## One more real bug, found while wiring up the launch (2026-09-17)

`--seed` was being passed straight into `make_datasets`/`make_gnn_datasets`,
which uses it as `StratifiedKFold`'s `random_state` -- meaning varying
`--seed` for the planned seed-variance runs would have silently reshuffled
WHICH SUBJECTS are in which fold too, confounding fold-split variance with
training-seed variance and making "fold 0, seed 1" not actually the same
fold 0 at all. **Fixed** in `train_baseline.py`, `train_graph_seg.py`,
`train_graph_seg_consistency.py`: the fold-split seed is now hardcoded to
0 always; `--seed` only drives `torch.manual_seed` (weight init, batch
order) from here on. The 3 baseline trunks already built are unaffected
(they used seed=0 throughout already, for both purposes at once).
Also fixed: `train_baseline.py`/`train_graph_seg.py`/
`train_graph_seg_consistency.py` never included seed in their checkpoint/
result filenames at all -- every one of Phase 8's 9 B2 runs would have
overwritten the same two files. Seed 0 keeps the original name (matches
existing references); seeds 1+ get a `_seed{N}` suffix.
Not yet fixed (not used by Phase 8, so not blocking, but latent):
`train_multitask.py`/`train_multitask_gnn.py` have the same seed/fold-split
conflation -- fine for now since neither is part of the Phase 8 launch.

## PHASE 8: COMPLETE (2026-09-18) — real result in PROGRESS_REPORT.md §6a

All 18 training runs + 3 pre-existing per-fold trunks (21 total) scored via
`src/eval/run_phase8_eval.py`, aggregated via `src/eval/phase8_aggregate.py`
→ `outputs/tables/phase8_summary.json`. Headline: no significant Dice
difference B2 vs B0 (paired Wilcoxon, p=0.24, n=345); B0 collapsed
completely in 2/9 runs, B1/B2 never did in 12/12 — real finding, disclosed
confound (B0 trains from scratch, B1/B2 warm-start). Full writeup:
PROGRESS_REPORT.md §6a. Next: Phase 9 (write-up already updated; deck
corrections still open) and the interactive demo discussed with the user.

## Phase 8 prep: COMPLETE (2026-09-17)

All three baseline trunks exist (and double as B0/seed0 for each fold).
Both real bugs found during prep (cache-write race red herring, then the
actual root cause -- RandCropByPosNegLabeld's inconsistently-typed
foreground_start_coord/foreground_end_coord metadata) are fixed in code,
not worked around, in both `build_transforms` and `build_transforms_gnn` --
every future Phase 8 run across every fold benefits, not just fold 2.
B1, B2, and the fixed Rule Evaluator (native-resolution JDCR + newly-wired
ECE) all smoke-tested clean end-to-end on fold 0's fresh trunk.

**Interrupted by a power loss ~2026-09-17 18:30-19:47** (machine turned
off), mid-way through job 13/18 (B2_fold1_seed0, no partial checkpoint --
nothing corrupted, just that one job's progress lost). Jobs 0-11 all
completed and saved successfully beforehand. Resumed via `--start-at 12`.
Found and fixed a real bug surfaced by this: `run_phase8.py` started its
progress log fresh on every invocation, so resuming would have silently
overwritten the record of jobs 0-11 (the actual checkpoints/results were
never at risk, only this summary log) -- fixed to load and extend the
existing log on a `--start-at` resume. The pre-overwrite log was also
backed up by hand: `outputs/tables/phase8_run_log_jobs0-11_backup.json`.

**Launched 2026-09-17** via `src/model/run_phase8.py`, an orchestrator that
runs all 18 remaining jobs sequentially (one GPU, no parallelism) and keeps
going past a failed job rather than losing the whole unattended queue to
one bad run. Progress log: `outputs/tables/phase8_run_log.json`, updated
after every job.

Two more real bugs found and fixed in the last mile before launch (see
"One more real bug" section below for the fold-seed one; the other:
`train_baseline.py`/`train_graph_seg.py`/`train_graph_seg_consistency.py`
never included seed in their checkpoint/result filenames, so all 9 B2 runs
would have overwritten the same two files). `run_rule_evaluator.py`'s B0
loader was also pointed at the wrong architecture (`multitask_fold0_best.pt`'s
`MultiTaskSegResNet`, Phase 3's shelved heads) instead of the plain
`SegResNet` `train_baseline.py` actually produces -- fixed to load the
correct architecture for Phase 8's real B0 checkpoints.
- [ ] Launch the full run (see schedule below).

## Pass marks (fixed now, not after)

**Primary claim:** B2's strict JDCR, averaged across all 3 folds × 3 seeds,
is higher than B0's strict JDCR under the same averaging. This is the
project's central novelty claim (PROJECT_PLAN.md §2: "JDCR is where the
knowledge graph should show a clear gain — that is what it is designed to
do"), so it is the one number this phase is really testing.

**Guard rail (non-inferiority):** B2's mean Dice, averaged the same way,
is no more than 0.02 below B0's mean Dice. If B2 buys JDCR at a bigger
segmentation cost than that, it gets reported as a real trade-off, not
absorbed into the primary claim.

**Everything else is descriptive, not gating:** B1 vs B0 (isolates the
graph-injection effect from the consistency loss), per-patient paired
comparison (Wilcoxon signed-rank, B2 vs B0 on the fold/seed combinations
where both have a prediction for that patient), ECE, coverage, and the
standard deviation across seeds/folds for every number above.

**If the primary claim fails:** report it as failed, the same way Phase 2's
6/7 rules and Phase 3–5's auxiliary heads were reported as failed. Do not
adjust `strict`/`lenient` after seeing which one passes — `strict` is
already fixed as the headline definition, above, before the run starts.

## Schedule

| Runs | What | Epochs | Approx. time |
|---|---|---|---|
| 3 | Per-fold baseline trunks (3-fold split) — **done**, these ARE B0/seed0 | 100 | ~1h each → ~3h ✅ |
| 6 | B0, remaining seeds 1-2 × 3 folds (seed 0 already covered above) | 100 | ~1.2h each → ~7h |
| 9 | B2, 3 folds × 3 seeds | 100 | ~1.3h each → ~12h |
| 3 | B1, fold 0 × 3 seeds (isolates injection from consistency loss) | 100 | ~1.2h each → ~4h |
| — | Rule Evaluator + ECE pass over all 21 checkpoints | — | ~1-2h |

**Total remaining: roughly 24-26 GPU-hours** (down from ~30 since baseline
prep is done and doubles as B0/seed0), run as a background queue, mostly
unattended. Seeds: 0, 1, 2 (matching the seed already used throughout
Phases 3–7, so seed 0's numbers are comparable to what's already reported).

## Naming convention for Phase 8 outputs

To avoid any more of the collisions this file exists because of:

- Checkpoints: `outputs/checkpoints/{model}_3fold_fold{F}_seed{S}_best.pt`
- Baseline trunks: `outputs/checkpoints/segresnet_3fold_fold{F}_best.pt`
  (no seed — baseline is warm-start prep, not part of the seeded comparison)
- Result tables: `outputs/tables/{model}_3fold_fold{F}_seed{S}.json`
- JDCR tables: `outputs/tables/jdcr_{model}_fold{F}_seed{S}.json`
- Aggregate summary (written after all runs complete):
  `outputs/tables/phase8_summary.json`
