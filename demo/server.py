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

# The actual fold-0 validation subjects for the Phase 8 checkpoints this demo
# loads -- fixed here rather than re-derived, because make_gnn_datasets's
# 3-way fold split is computed over whichever subjects are physically found
# on disk. Point --root at a smaller copy (e.g. a laptop with only these 50
# subjects' files, not the full 148) and the split recomputes differently --
# only a fraction of these 50 would land back in "fold 0" by chance. Pinning
# the list here guarantees the same 50 held-out patients load regardless of
# how many other subjects' files happen to be present.
FOLD0_VAL_SUBJECTS = [
    "sub-stroke0002", "sub-stroke0003", "sub-stroke0008", "sub-stroke0011", "sub-stroke0026",
    "sub-stroke0030", "sub-stroke0033", "sub-stroke0037", "sub-stroke0040", "sub-stroke0047",
    "sub-stroke0048", "sub-stroke0049", "sub-stroke0052", "sub-stroke0066", "sub-stroke0071",
    "sub-stroke0076", "sub-stroke0077", "sub-stroke0084", "sub-stroke0085", "sub-stroke0086",
    "sub-stroke0093", "sub-stroke0095", "sub-stroke0102", "sub-stroke0107", "sub-stroke0108",
    "sub-stroke0110", "sub-stroke0114", "sub-stroke0117", "sub-stroke0135", "sub-stroke0136",
    "sub-stroke0138", "sub-stroke0142", "sub-stroke0143", "sub-stroke0144", "sub-stroke0147",
    "sub-stroke0149", "sub-stroke0155", "sub-stroke0157", "sub-stroke0158", "sub-stroke0162",
    "sub-stroke0165", "sub-stroke0168", "sub-stroke0169", "sub-stroke0172", "sub-stroke0178",
    "sub-stroke0180", "sub-stroke0184", "sub-stroke0185", "sub-stroke0186", "sub-stroke0187",
]

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
        train_ds, val_ds, _, _ = make_gnn_datasets(
            root, OUT_TAB / "data_audit.json", 0, 3, PATCH, WB, 0, pathlib.Path(cache_dir),
            load_auxiliary_labels(), topcow_label_order(), native_midline=True)
        # Re-pin to the known-correct 50 regardless of which bucket this run's
        # fold recomputation put them in (see FOLD0_VAL_SUBJECTS above) --
        # a no-op when all 148 subjects' files are present, a necessary fix
        # when running from a smaller, trimmed copy of the dataset.
        from monai.data import PersistentDataset
        pool = {it["subject"]: it for it in list(train_ds.data) + list(val_ds.data)}
        found = [pool[s] for s in FOLD0_VAL_SUBJECTS if s in pool]
        missing = [s for s in FOLD0_VAL_SUBJECTS if s not in pool]
        if missing:
            print(f"  [demo] {len(missing)}/{len(FOLD0_VAL_SUBJECTS)} expected subject(s) not found "
                  f"on this --root (skipped): {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}",
                  flush=True)
        self.val_ds = PersistentDataset(found, transform=val_ds.transform,
                                         cache_dir=str(pathlib.Path(cache_dir) / "val"))
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
        self.cache_dir = pathlib.Path(cache_dir)
        try:  # recorded admission NIHSS per patient (clinical data), used only by the assistant's facts
            from src.model.labels import load_nihss_labels
            self.nihss = load_nihss_labels(pathlib.Path(root))
        except Exception as e:
            print(f"  [demo] NIHSS not loaded ({type(e).__name__}: {e}); assistant will say none is recorded", flush=True)
            self.nihss = {}
        self.explain_cache: OrderedDict = OrderedDict()  # (subject, model) -> channel-occlusion result
        self.explain_lock = threading.Lock()          # serialises the ~7 inference passes per explanation

    # -- explanations (demo/explain.py)
    def explain(self, sub: str, model_id: str) -> dict:
        from demo import explain as ex
        key = (sub, model_id)
        with self.explain_lock:
            if key in self.explain_cache:
                self.explain_cache.move_to_end(key)
                return self.explain_cache[key]
            if sub not in self.index:
                raise KeyError(sub)
            path = ex.cache_path(self.cache_dir, sub, model_id)
            res = ex.load(path)
            if res is None:
                res = ex.compute(self, sub, model_id, PATCH, WB)
                ex.save(path, res)
            self.explain_cache[key] = res
            while len(self.explain_cache) > 4:  # heat volumes are big; the disk cache keeps the rest
                self.explain_cache.popitem(last=False)
            return res

    def explain_summary(self, sub: str, model_id: str) -> dict:
        from demo import explain as ex
        if self.model_type.get(model_id) == "ensemble":
            return {"available": False, "note": "Explanations are shown for single models, not the ensemble."}
        return ex.summarize(self.explain(sub, model_id))

    def explain_png(self, sub, model_id, z, bg_ch, src) -> bytes:
        """Slice with the occlusion heatmap. src = 'all' (max over channels) or a channel index."""
        from demo import explain as ex
        res = self.explain(sub, model_id)
        case = self.case(sub, model_id)
        with self.lock:
            vol = self.volume(sub)
            base = self.windowed(sub, bg_ch)
            z = int(np.clip(z, 0, base.shape[2] - 1))
            heat = res["heat"].max(axis=0) if src == "all" else res["heat"][int(src)]
            img = orient(base[:, :, z]).astype(np.float32)
            rgb = ex.heat_rgb_overlay(np.stack([img] * 3, axis=-1), orient(heat[:, :, z].astype(np.float32)))
            rgb[orient(outline(case["pred"][:, :, z]))] = [235, 60, 60]
            rgb[orient(outline(vol["label"][:, :, z]))] = [70, 230, 110]
            if np.isfinite(vol["midline_x"]):
                col = int(round(base.shape[0] - 1 - vol["midline_x"]))
                if 0 <= col < rgb.shape[1]:
                    rgb[::4, col] = [255, 214, 10]
                    rgb[1::4, col] = [255, 214, 10]
            return png_bytes(np.clip(rgb, 0, 255).astype(np.uint8))

    def warm_explanations(self, model_ids) -> None:
        for sub in sorted(self.index):
            for mid in model_ids:
                try:
                    self.explain(sub, mid)
                    print(f"  [explain] {sub} {mid} ready", flush=True)
                except Exception as e:  # keep warming the rest
                    print(f"  [explain] {sub} {mid} failed: {type(e).__name__}: {e}", flush=True)
        print("  [explain] warm-up finished", flush=True)

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
            elif u.path in ("/compare", "/compare.html"):
                self._send((REPO / "demo" / "compare.html").read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/subjects":
                self._json({"subjects": self.demo.subjects(), "channels": CHANNELS,
                            "models": [{k: m[k] for k in ("id", "label", "tag", "blurb")} for m in MODELS]})
            elif u.path == "/api/case":
                self._json(self.demo.case(q["subject"], q["model"])["meta"])
            elif u.path == "/api/explain":
                self._json(self.demo.explain_summary(q["subject"], q["model"]))
            elif u.path == "/api/assistant":
                from demo import assistant
                labels = {m["id"]: m["label"] for m in MODELS}
                self._json(assistant.answer(self.demo, q["subject"], q["left"], q["right"],
                                            q.get("q", "summary"), labels))
            elif u.path == "/explain.png":
                png = self.demo.explain_png(q["subject"], q["model"], int(q.get("z", 0)),
                                            int(q.get("ch", 0)), q.get("src", "all"))
                self._send(png, "image/png")
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
    ap.add_argument("--warm-explain", action="store_true",
                    help="precompute channel-occlusion explanations for all 50 patients (B2 and B0 seed 0) in the background")
    ap.add_argument("--run-faithfulness", action="store_true",
                    help="after warm-up, run the deletion/faithfulness study (demo/faithfulness.py) -> outputs/tables/xai_faithfulness.json")
    args = ap.parse_args()

    Handler.demo = Demo(args.root, args.cache_dir)
    if args.warm_explain or args.run_faithfulness:
        def _background():
            Handler.demo.warm_explanations(["b2_s0", "b0_s0"])
            if args.run_faithfulness:
                from demo import faithfulness
                faithfulness.run_all(Handler.demo, ["b2_s0", "b0_s0"], PATCH, WB,
                                     OUT_TAB / "xai_faithfulness.json")
        threading.Thread(target=_background, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\nready -> http://127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
