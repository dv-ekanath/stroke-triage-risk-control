# KG-XAI Stroke Triage Explorer (interactive demo)

A local web app that runs **live inference** with the real Phase 8
checkpoints on held-out ISLES'24 patients, then scores each prediction with
the same Rule Evaluator used for the reported results.

## Run it

From the repo root, with the GPU free (not mid-training):

    ./.venv/Scripts/python.exe -m demo.server --root D:/ISLES-2024/train

Wait for `ready -> http://127.0.0.1:8765`, then open that address. Startup
takes ~10-20 s (loads the validation set and 4 checkpoints); each new
patient runs in ~2 s, cached after that. No new dependencies (stdlib HTTP
server; the page is one static HTML file).

## What you can do

- Pick any of the **50 held-out patients** (fold 0 validation set — none of
  the four models trained on them).
- Switch between four real models: **B2** (full system), **B1** (graph
  injection only), **B0 seed 0** (plain baseline, trained normally), and
  **B0 seed 1** (the plain baseline run that *collapsed* — predicts no
  lesion for anyone; it is kept on purpose so the failure is visible).
- Scroll through axial slices; view NCCT, CTA, CBF, CBV, MTT or Tmax.
- Toggle the model's predicted infarct (red), the actual infarct outline
  from follow-up MRI (green), and the brain midline (yellow, derived from
  the patient's own vessel anatomy).
- Read the **rule check**: does the predicted infarct sit on the same side
  as the occlusion (the one plausibility rule that held in real data), with
  the deterministic, template-built rationale — no language model.
- **Compare all four models** on the same patient and slice.

## Honest limits

- Only the 50 fold-0 validation patients. It does not accept new uploads —
  that would mean re-exposing the whole raw-file → 6-channel preprocessing
  pipeline as a live service, a much larger build.
- Dice on most patients is modest (~0.24 on average across the study);
  large infarcts are typically under-predicted. That is the real model, not
  a demo artifact. The graph did not significantly improve Dice (paired
  test, p = 0.24) — see `PROGRESS_REPORT.md` §6a.
- Research prototype, not a medical device.
- The four checkpoints are hardcoded in `demo/server.py` (`MODELS`); the
  collapsed-run entry needs `segresnet_3fold_fold0_seed1_best.pt` on disk.
