"""
Phase 8 orchestrator -- runs every remaining B0/B1/B2 fold x seed
combination sequentially (one GPU, can't parallelize), then a final Rule
Evaluator + ECE pass over every checkpoint. See PROJECT_REPORT.md (Appendix C) for the pass
marks (fixed before this was launched) and the full prep history.

Already done before this script runs (not repeated here):
  - 3 per-fold baseline trunks (segresnet_3fold_fold{0,1,2}_best.pt) --
    these ARE B0/seed0 for each fold already.
  - Smoke tests of B1, B2, and the Rule Evaluator on fold 0.

Continues past a failed job (logs it, moves on) rather than losing an
unattended ~24h queue to one bad run -- appends to `phase8_run_log.json`
after every job so progress is inspectable at any time, not just at the
end.

Usage
-----
    python -m src.model.run_phase8 --root D:/ISLES-2024/train
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

OUT_TAB = pathlib.Path("outputs/tables")
LOG_PATH = OUT_TAB / "phase8_run_log.json"
PY = sys.executable


def build_jobs(root: str, cache_base: str, cache_wb: str) -> list[dict]:
    common_base = ["--root", root, "--n-folds", "3", "--epochs", "100",
                   "--val-interval", "10", "--patch-size", "128", "128", "64",
                   "--batch-size", "2"]
    common_graph = ["--wb-size", "64", "64", "48"]

    jobs = []

    # B0: 6 more runs (seeds 1-2 x 3 folds; seed 0 per fold already done,
    # see PROJECT_REPORT.md (Appendix C))
    for fold in (0, 1, 2):
        for seed in (1, 2):
            jobs.append({
                "name": f"B0_fold{fold}_seed{seed}",
                "cmd": [PY, "-u", "-m", "src.model.train_baseline",
                        *common_base, "--fold", str(fold), "--seed", str(seed),
                        "--cache-dir", cache_base, "--num-workers", "0"],
            })

    # B1: fold 0 only x 3 seeds (isolates graph injection from the
    # consistency loss) -- seed 0 re-run for real (smoke test overwrote it)
    for seed in (0, 1, 2):
        jobs.append({
            "name": f"B1_fold0_seed{seed}",
            "cmd": [PY, "-u", "-m", "src.model.train_graph_seg",
                    *common_base, *common_graph, "--fold", "0", "--seed", str(seed),
                    "--cache-dir", cache_wb, "--num-workers", "0",
                    "--warm-start", "outputs/checkpoints/segresnet_3fold_fold0_best.pt"],
        })

    # B2: full 3 folds x 3 seeds
    for fold in (0, 1, 2):
        for seed in (0, 1, 2):
            jobs.append({
                "name": f"B2_fold{fold}_seed{seed}",
                "cmd": [PY, "-u", "-m", "src.model.train_graph_seg_consistency",
                        *common_base, *common_graph, "--fold", str(fold), "--seed", str(seed),
                        "--cache-dir", cache_wb, "--num-workers", "0",
                        "--warm-start", f"outputs/checkpoints/segresnet_3fold_fold{fold}_best.pt"],
            })

    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache-dir-base", default="D:/ISLES-2024/cache_1mm")
    ap.add_argument("--cache-dir-wb", default="D:/ISLES-2024/cache_1mm_wb64")
    ap.add_argument("--start-at", type=int, default=0, help="skip already-done jobs by index (resume)")
    args = ap.parse_args()

    jobs = build_jobs(args.root, args.cache_dir_base, args.cache_dir_wb)
    OUT_TAB.mkdir(parents=True, exist_ok=True)

    # resume-safe: load and extend the existing log instead of starting
    # fresh, or a --start-at resume after an interruption (e.g. a power
    # loss, 2026-09-17) silently overwrites the earlier jobs' history --
    # the checkpoints/result JSONs themselves were never at risk (each job
    # writes its own file), only this progress-tracking summary was
    if args.start_at > 0 and LOG_PATH.exists():
        log = json.loads(LOG_PATH.read_text())
        log["resumed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        log = {"total_jobs": len(jobs), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "jobs": []}
    print(f"Phase 8: {len(jobs)} training jobs queued, starting at index {args.start_at}", flush=True)

    for i, job in enumerate(jobs):
        if i < args.start_at:
            continue
        print(f"\n{'='*70}\n[{i+1}/{len(jobs)}] STARTING {job['name']}\n{'='*70}", flush=True)
        t0 = time.time()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        result = subprocess.run(job["cmd"], env=env)
        dt = time.time() - t0
        status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f"[{i+1}/{len(jobs)}] {job['name']}: {status}  ({dt/60:.1f} min)", flush=True)

        log["jobs"].append({"index": i, "name": job["name"], "status": status,
                             "minutes": round(dt / 60, 1),
                             "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        LOG_PATH.write_text(json.dumps(log, indent=2))

    print(f"\nAll {len(jobs)} Phase 8 training jobs done. See {LOG_PATH} for the run log.", flush=True)
    print("Next: run the Rule Evaluator + ECE pass over every checkpoint "
          "(not automated here -- see PROJECT_REPORT.md (Appendix C)).", flush=True)


if __name__ == "__main__":
    main()
