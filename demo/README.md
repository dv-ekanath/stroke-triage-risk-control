# KG-XAI compare demo

A local web app that runs **live inference** with the real trained checkpoints
on held-out ISLES'24 patients, puts two models side by side, scores each
prediction with the same Rule Evaluator used for the reported results, and
explains it.

## Run it

From the repo root, with the GPU free:

    ./.venv/Scripts/python.exe -m demo.server --root D:/ISLES-2024/train --cache-dir D:/cache

Wait for `ready -> http://127.0.0.1:8765`, then open that address (`/`,
`/index.html` and `/compare` all serve the same page). Startup takes 10–40 s;
each new patient runs in about 2 s per model, cached after that. The server is
the Python standard library only.

Optional flags: `--warm-explain` precomputes the "Why this prediction?" data for
all 50 patients (B2 and B0) in the background; `--run-faithfulness` additionally
runs the deletion study in `faithfulness.py` afterwards (about an hour).

For the assistant, run `ollama serve` first and make sure `gemma3:4b` is pulled.
Without Ollama the assistant falls back to a built-in summary.

## What you can do

- Pick any of the **50 held-out patients** (fold 0 validation set; none of the
  models trained on them). The upload box just picks a patient by the
  `sub-strokeNNNN` pattern in the file name.
- Choose the baseline on the left (standard run, the **failed** run that
  predicts nothing for anyone, or the 3-run ensemble) and the graph-guided model
  on the right (full system, or graph only).
- Scroll axial slices and switch between NCCT, CTA, CBF, CBV, MTT and Tmax.
  Red is the prediction, green the true lesion outline, yellow the midline.
- Read each side's **trust check** (laterality vs. the occlusion side, and
  predicted volume; no ground truth used), its Dice and the deterministic
  rationale.
- **Why this prediction?** — the model is re-run with each scan channel blanked
  in turn. The bars show the share of the model's lesion confidence that
  disappears without that channel; clicking a bar overlays where it mattered.
  The first time takes about 16 s per patient and model, then it is cached to
  `<cache-dir>/explain/`.
- **Triage assistant** — five preset questions (summary, why flagged, which model
  to trust, what they relied on, severity) plus free text. A local model phrases
  answers from numbered facts and must cite them; answers that add numbers,
  advice, evaluative claims or misattributed facts are rejected and replaced by
  a built-in summary. Expand "Facts this answer is built from" to see the source.

## Tests

    python -m demo.test_xai            # 25 unit checks, no GPU or server
    python -m demo.test_xai --live     # adds 9 checks against the running server

## Files

| File | Role |
|---|---|
| `server.py` | HTTP server, inference, trust flag, API |
| `compare.html` | The page |
| `explain.py` | Channel-occlusion explanations and heatmaps |
| `assistant.py` | Local-model assistant and its answer checks |
| `faithfulness.py` | Deletion study for the explanations (written, not run to completion) |
| `test_xai.py` | Tests for the explainability addition |
| `architecture.html` | Static architecture page |

## Honest limits

- Only the 50 fold-0 validation patients; new scans are not accepted, since that
  would mean serving the whole raw-file to 6-channel preprocessing pipeline.
- Dice is modest (about 0.24 on average across the study) and large infarcts are
  usually under-predicted. The graph did not significantly improve Dice
  (p = 0.59 on the per-patient seed-0 test, p = 0.24 pooled); see
  `PROJECT_REPORT.md` Part 1 §6a.
- The explanations describe what the model relied on, not the patient's
  physiology, and have not passed a formal faithfulness test.
- The assistant's checks cover numbers, citations, attribution and wording, not
  whether a sentence combines the facts sensibly.
- Research prototype, not a medical device.
- The checkpoints are hardcoded in `server.py` (`MODELS`); the collapsed-run
  entry needs `segresnet_3fold_fold0_seed1_best.pt` on disk.
