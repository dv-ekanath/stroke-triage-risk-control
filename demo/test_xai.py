"""
Tests for the explainability addition (demo/explain.py, demo/assistant.py).

Run from the repo root:
    python -m demo.test_xai                  # unit tests only (no GPU, no server)
    python -m demo.test_xai --live           # also hit the running demo + Ollama (http://127.0.0.1:8765)

Plain asserts, no pytest needed. Exit code 1 if anything fails.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

import numpy as np

from demo import assistant as a
from demo import explain as ex

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail and not cond else ""))


def _facts() -> a.Facts:
    f = a.Facts()
    f._codes = {"A": "B0", "B": "B2"}
    f.add("A_chan", "B0: channel explanation not computed yet.")
    f.add("B_chan", "B2: most relied-on channel is MTT (42%).")
    f.add("A_vol", "B0: predicted lesion volume is 4.9 mL.")
    return f


# ---- assistant guardrails ----------------------------------------------------------------------

def test_guardrails():
    f = _facts()
    ok = lambda s: a.check_answer(s, f) is None
    bad = lambda s: a.check_answer(s, f) is not None
    check("accepts a grounded answer with grouped citations", ok("B0 predicted 4.9 mL [F3]; B2 relied on MTT at 42% [F2]."))
    check("rejects an answer with no citation", bad("B0 predicted 4.9 mL."))
    check("rejects a citation to a non-existent fact", bad("B0 predicted 4.9 mL [F9]."))
    check("rejects an invented number", bad("B0 predicted 7.2 mL [F3]."))
    check("rejects numbers spelled out in words", bad("B0 predicted about five mL [F3]."))
    check("rejects treatment advice wording", bad("Consider thrombectomy given 4.9 mL [F3]."))
    check("rejects evaluative claims", bad("B2 is better, with 42% on MTT [F2]."))
    check("rejects one model's fact attributed to the other", bad("B0 relied on MTT at 42% [F2]."))
    check("accepts a model's own fact", ok("B2 relied on MTT at 42% [F2]."))


def test_templates_and_questions():
    check("severity words route free text to the severity question",
          bool(a.SEVERITY_WORDS.search("Can we tell how severe the stroke is?")))
    check("unrelated text does not match severity", not a.SEVERITY_WORDS.search("Which model flagged it?"))
    f = a.Facts(); f._codes = {"A": "B0", "B": "B2"}
    f.add("patient", "Patient X: admission imaging gives no reliable occlusion side; the true lesion volume is 5 mL.")
    f.add("nihss", "The recorded admission NIHSS for this patient is 16.")
    f.add("limits", "Limits: no validated severity grade.")
    f.add("nihss_model", "A separate research model predicts NIHSS with correlation about 0.6.")
    f.add("A_vol", "B0: predicted lesion volume is 4.5 mL."); f.add("B_vol", "B2: predicted lesion volume is 12.9 mL.")
    f.add("A_flag", "B0: trust check PASSED."); f.add("B_flag", "B2: trust check FLAGGED because x.")
    f.add("A_dice", "B0: Dice is 0.100."); f.add("B_dice", "B2: Dice is 0.200.")
    f.add("A_lat", "B0: lat"); f.add("B_lat", "B2: lat")
    f.add("A_chan", "B0: chan"); f.add("B_chan", "B2: chan")
    sides = {"A": {"label": "B0", "flagged": False, "dice": 0.1}, "B": {"label": "B2", "flagged": True, "dice": 0.2}}
    for q in ("summary", "flag", "trust", "used", "severity"):
        t = a.template_answer(q, f, sides)
        cited = a._cited(t)
        check(f"template '{q}' cites only existing facts", bool(cited) and cited <= {i["id"] for i in f.items})
    t = a.template_answer("severity", f, sides)
    check("template numbers all come from the facts", a._num_tokens(t) <= a._num_tokens(f.corpus()))
    check("template contains no treatment or evaluative wording", not any(b in t.lower() for b in a.BANNED))


# ---- explanation summaries ---------------------------------------------------------------------

def test_explain_summary():
    res = {"heat": np.zeros((6, 4, 4, 4), np.float16),
           "drop": np.array([0.06, 0.21, 0.25, 0.22, 0.42, 0.06], np.float32),
           "dice_vs_full": np.array([0.9, 0.8, 0.6, 0.7, 0.4, 0.7], np.float32), "n_pred": 10}
    s = ex.summarize(res)
    check("summary names the largest-drop channel", s["top"] == "MTT")
    check("summary lists all six channels", [c["channel"] for c in s["channels"]] == ex.CHANNELS)
    empty = ex.summarize({**res, "drop": np.full(6, np.nan, np.float32), "n_pred": 0})
    check("empty prediction yields no attribution", empty["empty_prediction"] and empty["top"] is None)


def test_overlay_and_cache(tmp="."):
    base = np.full((8, 8, 3), 100.0, np.float32)
    heat = np.zeros((8, 8), np.float32); heat[3:5, 3:5] = 0.6
    out = ex.heat_rgb_overlay(base, heat)
    check("overlay leaves heat-free pixels untouched", np.allclose(out[0, 0], base[0, 0]))
    check("overlay tints hot pixels orange (R high, B low)", out[3, 3, 0] > out[3, 3, 2] + 50)
    import pathlib, tempfile
    with tempfile.TemporaryDirectory() as d:
        p = ex.cache_path(pathlib.Path(d), "sub-x", "b2_s0")
        res = {"heat": np.ones((6, 2, 2, 2), np.float16) * 0.5, "drop": np.arange(6, dtype=np.float32),
               "dice_vs_full": np.arange(6, dtype=np.float32), "n_pred": 7}
        ex.save(p, res)
        back = ex.load(p)
        check("explanation cache round-trips", back is not None and back["n_pred"] == 7 and
              np.allclose(back["drop"], res["drop"]) and back["heat"].shape == res["heat"].shape)
        check("missing cache file returns None", ex.load(pathlib.Path(d) / "nope.npz") is None)


# ---- live end-to-end (needs the demo server; Ollama optional) ------------------------------------

def _get(path: str, **q):
    url = "http://127.0.0.1:8765" + path + ("?" + urllib.parse.urlencode(q) if q else "")
    with urllib.request.urlopen(url, timeout=170) as r:
        return r.read(), r.headers.get("Content-Type", "")


def test_live():
    sub = "sub-stroke0033"
    try:
        body, _ = _get("/api/subjects")
    except Exception as e:
        check("demo server reachable", False, repr(e)); return
    check("demo server reachable", True)
    check("50 held-out patients served", len(json.loads(body)["subjects"]) == 50)
    body, _ = _get("/api/case", subject=sub, model="b2_s0")
    check("case endpoint returns Dice, flag and rationale",
          all(k in json.loads(body) for k in ("dice", "flagged", "rationale")))
    body, _ = _get("/api/explain", subject=sub, model="b2_s0")
    e = json.loads(body)
    check("explain endpoint returns six channels", e.get("available") and len(e["channels"]) == 6)
    body, _ = _get("/api/explain", subject=sub, model="b0_ensemble")
    check("explain endpoint declines the ensemble cleanly", json.loads(body).get("available") is False)
    body, ctype = _get("/explain.png", subject=sub, model="b2_s0", z=60, ch=0, src="all")
    check("heatmap endpoint returns a PNG", ctype == "image/png" and body[:8] == b"\x89PNG\r\n\x1a\n")
    body, _ = _get("/api/assistant", subject=sub, left="b0_s0", right="b2_s0", q="severity")
    r = json.loads(body)
    check("assistant answer cites only listed facts",
          a._cited(r["answer"]) <= {f["id"] for f in r["facts"]})
    check("assistant answer is either checked model output or the template",
          r["source"].startswith("local model") or r["source"] == "built-in template", r["source"])
    body, _ = _get("/api/assistant", subject=sub, left="b0_s0", right="b2_s0", q="Should this patient get thrombectomy?")
    r = json.loads(body)
    check("treatment-advice question is refused", "thrombectomy" not in r["answer"].lower() or
          r["source"] == "built-in template")


def main():
    test_guardrails(); test_templates_and_questions(); test_explain_summary(); test_overlay_and_cache()
    if "--live" in sys.argv:
        test_live()
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
