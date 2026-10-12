"""
Phase 8 scoring pass -- runs the fixed Rule Evaluator (native-resolution
JDCR + ECE, src/eval/run_rule_evaluator.py) over all 21 real checkpoints
from the Phase 8 training queue (src/model/run_phase8.py), sequentially
(one GPU). Writes outputs/tables/jdcr_{model}_fold{F}_seed{S}.json for each,
same as manual single runs already do -- this just avoids invoking it 21
times by hand.

Usage
-----
    python -m src.eval.run_phase8_eval --root D:/ISLES-2024/train
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

CKPT = pathlib.Path("outputs/checkpoints")
PY = sys.executable


def ckpt_name(model: str, fold: int, seed: int) -> str:
    suffix = "" if seed == 0 else f"_seed{seed}"
    if model == "b0":
        return f"segresnet_3fold_fold{fold}{suffix}_best.pt"
    if model == "b1":
        return f"graph_seg_fold{fold}{suffix}_best.pt"
    return f"graph_seg_consistency_fold{fold}{suffix}_best.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache-dir-wb", default="D:/ISLES-2024/cache_1mm_wb64")
    args = ap.parse_args()

    combos = ([("b0", f, s) for f in (0, 1, 2) for s in (0, 1, 2)]
              + [("b1", 0, s) for s in (0, 1, 2)]
              + [("b2", f, s) for f in (0, 1, 2) for s in (0, 1, 2)])

    print(f"Phase 8 scoring: {len(combos)} checkpoints to evaluate", flush=True)
    failures = []
    for i, (model, fold, seed) in enumerate(combos):
        ckpt = CKPT / ckpt_name(model, fold, seed)
        tag = f"fold{fold}_seed{seed}"
        print(f"\n[{i+1}/{len(combos)}] {model} {tag}  ({ckpt.name})", flush=True)
        if not ckpt.exists():
            print(f"  !! checkpoint missing, skipping: {ckpt}", flush=True)
            failures.append((model, tag, "checkpoint missing"))
            continue
        cmd = [PY, "-u", "-m", "src.eval.run_rule_evaluator",
               "--root", args.root, "--fold", str(fold), "--n-folds", "3",
               "--model-type", model, "--checkpoint", str(ckpt), "--tag", tag,
               "--cache-dir", args.cache_dir_wb, "--n-rationale-samples", "0"]
        t0 = time.time()
        result = subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        dt = time.time() - t0
        status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f"  {status}  ({dt/60:.1f} min)", flush=True)
        if result.returncode != 0:
            failures.append((model, tag, status))

    print(f"\nDone: {len(combos) - len(failures)}/{len(combos)} scored successfully.", flush=True)
    if failures:
        print("Failures:", failures, flush=True)


if __name__ == "__main__":
    main()
