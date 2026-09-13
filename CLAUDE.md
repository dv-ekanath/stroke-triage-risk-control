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
| Python env | `.venv\` (gitignored). torch 2.11.0+cu128, MONAI 1.6, nibabel, SimpleITK, TorchIO. PyTorch Geometric NOT yet installed (Phase 3). Run as `./.venv/Scripts/python.exe -m src....` |
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
