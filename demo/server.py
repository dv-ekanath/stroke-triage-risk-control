"""
KG-XAI interactive demo -- a small local web app that runs LIVE inference
with the real trained Phase 8 checkpoints on held-out ISLES'24 patients
(fold 0's validation set: patients none of these models trained on).

Not a mock-up: clicking a patient runs the actual model on your GPU
(sliding-window over the full 3D volume), then scores the prediction with
the same Rule Evaluator used for the Phase 8 results (src/eval/rule_evaluator.py)
and shows its deterministic rationale.

Five models are exposed, on purpose: B0 seed 0, B0 seed 1 (the run that
COLLAPSED -- predicts no lesion for anyone; seeing it live is the point),
B1 seed 0, B2 seed 0, and a 5th "B0 ensemble" option that averages all 3
of B0's fold-0 seeds' softmax probabilities live, with NO retraining --
the fix for the collapse from the 2026-10-11 ensemble_check.json finding.
Every prediction also carries a live "flagged for review" verdict, using
only the two signals that are genuinely computable with no ground truth
(laterality vs. admission occlusion side; near-zero predicted volume) --
the corrected version of the 2026-10-11 triage_check.json finding (a
third criterion, calibration error, was tested and dropped: it needs the
true label to compute, so it isn't actually usable at inference time).

Standard library HTTP server only -- no new dependencies.

Usage (from the repo root)
--------------------------
    python -m demo.server --root D:/ISLES-2024/train
    then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import struct
import sys
import threading
import time
import zlib
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
os.chdir(REPO)  # src/ uses repo-relative paths (kg/, outputs/)
sys.path.insert(0, str(REPO))

OUT_TAB = pathlib.Path("outputs/tables")
CKPT = pathlib.Path("outputs/checkpoints")
CHANNELS = ["NCCT", "CTA", "CBF", "CBV", "MTT", "Tmax"]
PATCH, WB = (128, 128, 64), (64, 64, 48)

MODELS = [
    {"id": "b2_s0", "type": "b2", "ckpt": CKPT / "graph_seg_consistency_fold0_best.pt",
     "label": "B2 · graph + consistency loss", "tag": "full system",
     "blurb": "Vessel-graph cross-attention in the decoder, plus the same-side consistency loss. Warm-started."},
    {"id": "b1_s0", "type": "b1", "ckpt": CKPT / "graph_seg_fold0_best.pt",
     "label": "B1 · graph injection only", "tag": "ablation",
     "blurb": "Vessel-graph cross-attention, no consistency loss. Warm-started."},
    {"id": "b0_s0", "type": "b0", "ckpt": CKPT / "segresnet_3fold_fold0_best.pt",
     "label": "B0 · plain baseline (seed 0)", "tag": "baseline",
     "blurb": "No graph. Same trunk, trained from scratch. This seed trained normally."},
    {"id": "b0_s1", "type": "b0", "ckpt": CKPT / "segresnet_3fold_fold0_seed1_best.pt",
     "label": "B0 · plain baseline (seed 1)", "tag": "collapsed run",
     "blurb": "No graph. This seed COLLAPSED in training: it predicts no lesion for every patient."},
    {"id": "b0_ensemble", "type": "ensemble",
     "ckpt": [CKPT / "segresnet_3fold_fold0_best.pt", CKPT / "segresnet_3fold_fold0_seed1_best.pt",
              CKPT / "segresnet_3fold_fold0_seed2_best.pt"],
     "label": "B0 · ensemble (3 seeds)", "tag": "fix, no graph",
     "blurb": "Average of all 3 B0 seeds' predicted probabilities, live, no retraining -- "
              "includes the collapsed seed above, and still isn't dragged down by it."},
]

NEAR_ZERO_VOXELS = 1000  # < 1 mL on this project's 1mm^3 grid -- matches src/eval/triage_check.py


# ------------------------------------------------------------------ PNG (stdlib)

def png_bytes(rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rgb.reshape(h, w * 3)], axis=1).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b""))


def outline(mask2d: np.ndarray) -> np.ndarray:
    m = mask2d.astype(bool)
    e = m.copy()
    e[1:, :] &= m[:-1, :]
    e[:-1, :] &= m[1:, :]
    e[:, 1:] &= m[:, :-1]
    e[:, :-1] &= m[:, 1:]
    return m & ~e


def orient(a2d: np.ndarray) -> np.ndarray:
    """(R, A) array slice -> screen: anterior up, patient's right on screen-left
    (radiological convention). Arrays are RAS+ after Orientationd."""
    return a2d.T[::-1, ::-1]


# ------------------------------------------------------------------ the app state

class Demo:
    def __init__(self, root: str, cache_dir: str):
        from src.eval.metrics import dice
        from src.eval.rule_evaluator import evaluate_laterality, generate_rationale
        from src.eval.run_rule_evaluator import build_model
        from src.model.dataset import load_volumes
        from src.model.gnn_head import topcow_label_order
        from src.model.labels import load_auxiliary_labels
        from src.model.train_multitask_gnn import make_gnn_datasets

        self._dice, self._eval_lat, self._rationale = dice, evaluate_laterality, generate_rationale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()

        print("loading validation set (fold 0)...", flush=True)
        _, self.val_ds, _, _ = make_gnn_datasets(
            root, OUT_TAB / "data_audit.json", 0, 3, PATCH, WB, 0, pathlib.Path(cache_dir),
            load_auxiliary_labels(), topcow_label_order(), native_midline=True)
        self.index = {it["subject"]: i for i, it in enumerate(self.val_ds.data)}
        self.volumes_ml = load_volumes(OUT_TAB / "data_audit.json")
        kg = json.loads((OUT_TAB / "kg_node_features.json").read_text(encoding="utf-8"))
        self.occ = {r["subject"]: r for r in kg["subjects"]}

        print("loading models...", flush=True)
        self.models = {}
        for m in MODELS:
            if m["type"] == "ensemble":
                self.models[m["id"]] = [build_model("b0", ckpt, self.device) for ckpt in m["ckpt"]]
                print(f"  {m['id']}: {len(m['ckpt'])} members -- {', '.join(c.name for c in m['ckpt'])}", flush=True)
            else:
                self.models[m["id"]] = build_model(m["type"], m["ckpt"], self.device)
                print(f"  {m['id']}: {m['ckpt'].name}", flush=True)

        self.model_type = {m["id"]: m["type"] for m in MODELS}
        self.vol_cache: OrderedDict = OrderedDict()   # subject -> loaded volume dict
        self.case_cache: dict = {}                    # (subject, model) -> result
        self.win_cache: dict = {}                     # (subject, channel) -> uint8 volume

    # -- data
    def subjects(self) -> list[dict]:
        out = []
        for sub, i in sorted(self.index.items()):
            side = {0: "left", 1: "right"}.get(self.val_ds.data[i].get("expected_side_code", -1))
            out.append({"id": sub, "expected_side": side,
                        "actual_ml": round(self.volumes_ml.get(sub, float("nan")), 1)})
        return out

    def volume(self, sub: str) -> dict:
        if sub in self.vol_cache:
            self.vol_cache.move_to_end(sub)
            return self.vol_cache[sub]
        item = self.val_ds[self.index[sub]]
        v = {
            "image": torch.as_tensor(np.asarray(item["image"]), dtype=torch.float32),
            "image_wb": torch.as_tensor(np.asarray(item["image_wb"]), dtype=torch.float32),
            "label": (np.asarray(item["label"])[0] > 0).astype(np.uint8),
            "node_coords": torch.as_tensor(np.asarray(item["node_coords"]), dtype=torch.float32),
            "node_present": torch.as_tensor(np.asarray(item["node_present"]), dtype=torch.float32),
            "midline_x": float(item["native_midline_voxel_x"]),
            "side": {0: "left", 1: "right"}.get(int(item["expected_side_code"])),
        }
        self.vol_cache[sub] = v
        while len(self.vol_cache) > 6:
            old, _ = self.vol_cache.popitem(last=False)
            self.win_cache = {k: a for k, a in self.win_cache.items() if k[0] != old}
            self.case_cache = {k: a for k, a in self.case_cache.items() if k[0] != old}
        return v

    def windowed(self, sub: str, ch: int) -> np.ndarray:
        key = (sub, ch)
        if key not in self.win_cache:
            a = self.volume(sub)["image"][ch].numpy()
            nz = a[a != 0]
            lo, hi = (np.percentile(nz, 1), np.percentile(nz, 99.5)) if nz.size else (0.0, 1.0)
            self.win_cache[key] = (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        return self.win_cache[key]

    # -- inference
    def case(self, sub: str, model_id: str) -> dict:
        key = (sub, model_id)
        with self.lock:
            if key in self.case_cache:
                return self.case_cache[key]
            vol = self.volume(sub)
            t0 = time.time()
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                if self.model_type[model_id] == "ensemble":
                    # average all 3 members' softmax probabilities, live, no
                    # retraining -- src/eval/ensemble_check.py's finding
                    probs = None
                    for model, compute_emb, predict_native in self.models[model_id]:
                        logits_m = predict_native(vol["image"].to(self.device)[None], None, None, PATCH)
                        p = torch.softmax(logits_m.float(), dim=1)
                        probs = p if probs is None else probs + p
                    logits = probs  # already averaged below; argmax is all that matters from here
                else:
                    model, compute_emb, predict_native = self.models[model_id]
                    emb = present = None
                    if compute_emb is not None:
                        present = vol["node_present"].to(self.device)[None]
                        emb = compute_emb(vol["image_wb"].to(self.device)[None],
                                          vol["node_coords"].to(self.device)[None], present, WB)
                    logits = predict_native(vol["image"].to(self.device)[None], emb, present, PATCH)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
            seconds = time.time() - t0

            gt = vol["label"]
            lat = self._eval_lat(pred, vol["side"], vol["midline_x"])
            rationale = self._rationale(sub, self.occ.get(sub, {}), vol["side"], lat)
            zsum = (gt if gt.any() else pred).sum(axis=(0, 1))

            # live triage flag -- src/eval/triage_check.py's CORRECTED rule:
            # only the two signals that are actually computable with no
            # ground truth (a third, calibration error, needs the true
            # label and was dropped there for exactly that reason)
            near_zero = bool(pred.sum() < NEAR_ZERO_VOXELS)
            inconsistent = bool(lat.get("evaluable") and lat.get("consistent") is False)
            flagged = bool(near_zero or inconsistent)
            if flagged:
                flag_reason = " and ".join(r for r, v in
                    (("predicted volume is near zero", near_zero),
                     ("laterality is inconsistent with the occlusion side", inconsistent)) if v)
            else:
                flag_reason = None

            res = {
                "pred": pred,
                "meta": {
                    "subject": sub, "model": model_id, "shape": list(pred.shape),
                    "default_z": int(np.argmax(zsum)) if zsum.max() > 0 else pred.shape[2] // 2,
                    "dice": round(float(self._dice(pred, gt)), 4),
                    "pred_ml": round(float(pred.sum()) * 1e-3, 1),
                    "actual_ml": round(float(gt.sum()) * 1e-3, 1),
                    "midline_x": vol["midline_x"], "expected_side": vol["side"],
                    "laterality": lat, "rationale": rationale,
                    "flagged": flagged, "flag_reason": flag_reason,
                    "inference_seconds": round(seconds, 1),
                },
            }
            self.case_cache[key] = res
            return res

    # -- rendering
    def slice_png(self, sub, model_id, z, ch, show_pred, show_gt, show_mid) -> bytes:
        res = self.case(sub, model_id)
        with self.lock:
            vol = self.volume(sub)
            base = self.windowed(sub, ch)
            z = int(np.clip(z, 0, base.shape[2] - 1))
            img = orient(base[:, :, z]).astype(np.float32)
            rgb = np.stack([img] * 3, axis=-1)
            if show_pred:
                pm = orient(res["pred"][:, :, z]).astype(bool)
                rgb[pm] = rgb[pm] * 0.55 + np.array([235, 60, 60]) * 0.45
            if show_gt:
                rgb[orient(outline(vol["label"][:, :, z]))] = [70, 230, 110]
            if show_mid and np.isfinite(vol["midline_x"]):
                col = int(round(base.shape[0] - 1 - vol["midline_x"]))
                if 0 <= col < rgb.shape[1]:
                    rgb[::4, col] = [255, 214, 10]
                    rgb[1::4, col] = [255, 214, 10]
            return png_bytes(np.clip(rgb, 0, 255).astype(np.uint8))


# ------------------------------------------------------------------ HTTP

class Handler(BaseHTTPRequestHandler):
    demo: Demo = None  # set in main()

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                self._send((REPO / "demo" / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/subjects":
                self._json({"subjects": self.demo.subjects(), "channels": CHANNELS,
                            "models": [{k: m[k] for k in ("id", "label", "tag", "blurb")} for m in MODELS]})
            elif u.path == "/api/case":
                self._json(self.demo.case(q["subject"], q["model"])["meta"])
            elif u.path == "/slice.png":
                png = self.demo.slice_png(q["subject"], q["model"], int(q.get("z", 0)),
                                          int(q.get("ch", 0)), q.get("pred", "1") == "1",
                                          q.get("gt", "1") == "1", q.get("mid", "1") == "1")
                self._send(png, "image/png")
            else:
                self._send(b"not found", "text/plain", 404)
        except (KeyError, ValueError) as e:
            self._json({"error": f"bad request: {e}"}, 400)
        except Exception as e:  # keep the server alive, show the error in the UI
            import traceback
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="D:/ISLES-2024/train", help="ISLES'24 training-set root")
    ap.add_argument("--cache-dir", default="D:/ISLES-2024/cache_1mm_wb64")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    Handler.demo = Demo(args.root, args.cache_dir)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\nready -> http://127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
