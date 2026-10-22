"""
Triage assistant for the compare page.

Design rule: the assistant never supplies facts, it only phrases them.

  1. The server assembles a numbered FACT list from the project's own tools
     (the two models' predictions, the ground-truth-free trust flag, the
     laterality rule check, and -- if already computed -- the channel-occlusion
     explanation).
  2. A small LOCAL model (Ollama, default gemma3:4b) is asked to answer one
     question using only those facts and to cite them as [F3].
  3. The answer is checked: every number must occur in the facts, at least one
     valid citation must be present, and medical-advice wording is rejected.
     Anything that fails is replaced by a deterministic template built from
     the same facts, and the response says so.

If Ollama isn't running the template is used directly. Nothing here is part
of the evaluated pipeline; it is a convenience layer over deterministic tools.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
LOCAL_MODEL = "gemma3:4b"

QUESTIONS = {
    "summary": "Summarise this patient's result for both models.",
    "flag": "Why was each prediction flagged for review, or why did it pass?",
    "trust": "Which model should be trusted for this patient, and why?",
    "used": "What scan channels did each model rely on?",
    "severity": "Can we tell how severe this patient's stroke is, and how plausible is that?",
}

SEVERITY_WORDS = re.compile(r"\b(severe|severity|serious|how bad|nihss|disabil\w*|prognos\w*|outcome)\b", re.I)

SYSTEM = (
    "You are an assistant inside a research demo that compares two stroke-lesion segmentation models on CT. "
    "Answer ONLY from the FACTS list. Never add medical knowledge, numbers, diagnoses or treatment advice that "
    "is not in the FACTS. Cite the facts you use as [F1], [F2]. Use at most four short sentences. "
    "If the FACTS do not answer the question, say that they do not. "
    "Do not judge quality, do not say one model is better, worse or closer, and do not interpret numbers: "
    "quote the facts and their numbers only. "
    "If asked about severity, report the recorded NIHSS and the lesion volumes, and state the limits exactly as "
    "the facts give them; do not grade the stroke yourself. "
    "The trust check (flag) uses no ground truth. Dice uses ground truth and is only known retrospectively, "
    "so never present Dice as something available at prediction time. This is a research prototype."
)

BANNED = ("diagnos", "thrombectomy", "should receive", "prescrib", "treat the patient", "recommend treatment",
          # evaluative / comparative claims the facts do not make
          "poor", "excellent", "good", "bad", "accurate", "closer", "better", "worse", "indicating", "suggesting")


# ------------------------------------------------------------------ facts

class Facts:
    def __init__(self):
        self.items: list[dict] = []
        self.by_key: dict[str, str] = {}

    def add(self, key: str, text: str) -> str:
        fid = f"F{len(self.items) + 1}"
        model = re.match(r"[AB]_", key) and getattr(self, "_codes", {}).get(key[0])
        self.items.append({"id": fid, "key": key, "text": text, "model": model or None})
        self.by_key[key] = fid
        return fid

    def text_of(self, key: str) -> str:
        return next(i["text"] for i in self.items if i["key"] == key)

    def corpus(self) -> str:
        return " ".join(i["text"] for i in self.items)

    def prompt_block(self) -> str:
        return "\n".join(f"{i['id']}: {i['text']}" for i in self.items)


def _channel_summary(summary: dict | None) -> tuple[str, dict]:
    """(sentence, details) from demo.explain.summarize output, or a 'not computed' note."""
    if not summary:
        return "channel explanation not computed yet for this patient", {}
    if not summary.get("available", True) or summary.get("empty_prediction"):
        return "no channel explanation is available (no lesion predicted, or ensemble model)", {}
    ranked = sorted((c for c in summary["channels"] if c["drop"] is not None), key=lambda c: -c["drop"])
    a, b = ranked[0], ranked[1]
    return (f"most relied-on scan channel is {a['channel']} (blanking it removes {round(a['drop'] * 100)}% of "
            f"the model's lesion confidence), then {b['channel']} ({round(b['drop'] * 100)}%)"), {"top": a["channel"]}


def build_facts(demo, sub: str, left_id: str, right_id: str, labels: dict) -> tuple[Facts, dict]:
    from demo import explain as ex
    f = Facts()
    sides = {}
    first = demo.case(sub, left_id)["meta"]
    exp = first.get("expected_side") or "unknown"
    side_txt = (f"admission imaging puts the blocked vessel on the {exp} side" if exp != "unknown"
                else "admission imaging gives no reliable occlusion side")
    f.add("patient", f"Patient {sub}: {side_txt}; "
                     f"the true lesion volume is {first['actual_ml']} mL (ground truth, known only retrospectively).")
    f._codes = {t: (re.match(r"B\d", labels.get(i, i)) or [None])[0] for t, i in (("A", left_id), ("B", right_id))}
    nihss = getattr(demo, "nihss", {}).get(sub)
    if nihss is not None:
        f.add("nihss", f"The recorded admission NIHSS for this patient is {nihss:g} (scale 0 to 42, higher means more "
                       f"severe deficits). It is clinical data from the dataset, not a model output.")
    else:
        f.add("nihss", "No admission NIHSS is recorded for this patient.")
    f.add("limits", "Limits: this demo does not compute a validated severity grade. Predicted lesion volume is only a "
                    "rough indicator of stroke size, and across the study the models reach a Dice of about 0.24 on "
                    "average, with large lesions usually under-predicted.")
    f.add("nihss_model", "A separate research model, not shown in this demo, predicts admission NIHSS with a correlation "
                         "of about 0.6 with the true score on held-out patients. That ranks patients roughly but is not "
                         "accurate enough to give an individual's score.")
    for tag, mid in (("A", left_id), ("B", right_id)):
        m = demo.case(sub, mid)["meta"]
        lab = labels.get(mid, mid)
        lat = m.get("laterality") or {}
        f.add(f"{tag}_vol", f"{lab}: predicted lesion volume is {m['pred_ml']} mL.")
        f.add(f"{tag}_dice", f"{lab}: Dice against the true lesion is {m['dice']:.3f} (retrospective, needs ground truth).")
        if lat.get("evaluable"):
            rf = lat["right_fraction"]
            on_pred = rf if lat["predicted_side"] == "right" else 1 - rf
            verdict = "consistent with" if lat["consistent"] else "INCONSISTENT with"
            f.add(f"{tag}_lat", f"{lab}: {round(on_pred * 100)}% of the predicted lesion lies on the {lat['predicted_side']} "
                                f"side, which is {verdict} the expected {m['expected_side']} side.")
        elif lat.get("reason") == "empty_prediction":
            f.add(f"{tag}_lat", f"{lab}: no lesion was predicted, so laterality cannot be checked.")
        else:
            f.add(f"{tag}_lat", f"{lab}: laterality is not evaluable because the expected occlusion side is unknown.")
        if m["flagged"]:
            f.add(f"{tag}_flag", f"{lab}: trust check says FLAGGED for review because {m['flag_reason']} "
                                 f"(computed without ground truth).")
        else:
            f.add(f"{tag}_flag", f"{lab}: trust check PASSED, nothing implausible found (computed without ground truth).")
        summ = _peek_explain(demo, sub, mid, ex)
        sentence, det = _channel_summary(summ)
        f.add(f"{tag}_chan", f"{lab}: {sentence}.")
        sides[tag] = {"id": mid, "label": lab, "flagged": bool(m["flagged"]), "dice": m["dice"], "top": det.get("top")}
    return f, sides


def _peek_explain(demo, sub, mid, ex):
    """Channel explanation only if it already exists (memory or disk); never triggers the ~16 s compute."""
    if demo.model_type.get(mid) == "ensemble":
        return {"available": False}
    res = demo.explain_cache.get((sub, mid)) or ex.load(ex.cache_path(demo.cache_dir, sub, mid))
    return ex.summarize(res) if res else None


# ------------------------------------------------------------------ deterministic answers

def template_answer(q: str, f: Facts, sides: dict) -> str:
    ref = f.by_key
    A, B = sides["A"], sides["B"]

    def cite(*keys):
        return " ".join(f"[{ref[k]}]" for k in keys)

    if q == "flag":
        return " ".join(f"{f.text_of(t + '_flag')} {cite(t + '_flag')}" +
                        (f" {f.text_of(t + '_lat')} {cite(t + '_lat')}" if sides[t]["flagged"] else "")
                        for t in ("A", "B"))
    if q == "used":
        return " ".join(f"{f.text_of(t + '_chan')} {cite(t + '_chan')}" for t in ("A", "B"))
    if q == "severity":
        return (f"{f.text_of('nihss')} {cite('nihss')} "
                f"{f.text_of('A_vol')} {cite('A_vol')} {f.text_of('B_vol')} {cite('B_vol')} "
                f"{f.text_of('limits')} {cite('limits')} {f.text_of('nihss_model')} {cite('nihss_model')}")
    if q == "trust":
        if A["flagged"] != B["flagged"]:
            ok, bad = (A, B) if B["flagged"] else (B, A)
            tk, bk = ("A", "B") if ok is A else ("B", "A")
            return (f"{ok['label']} passed the trust check while {bad['label']} was flagged, so {ok['label']} is the better "
                    f"supported prediction for this patient {cite(tk + '_flag', bk + '_flag')}. "
                    f"Dice is retrospective and not used for this judgement.")
        both = "both were flagged" if A["flagged"] else "both passed"
        return (f"The trust check does not separate the models here: {both} {cite('A_flag', 'B_flag')}. "
                f"Dice is {A['dice']:.3f} vs {B['dice']:.3f} but that needs ground truth {cite('A_dice', 'B_dice')}.")
    # summary
    return (f"{f.text_of('patient')} {cite('patient')} "
            f"{f.text_of('A_flag')} {cite('A_flag')} {f.text_of('B_flag')} {cite('B_flag')} "
            f"{f.text_of('A_dice')} {cite('A_dice')} {f.text_of('B_dice')} {cite('B_dice')}")


# ------------------------------------------------------------------ local model + checking

def _cited(text: str) -> set[str]:
    """Fact ids cited in any bracket, e.g. [F3], [F5, F10], [F1][F2]."""
    return {i for grp in re.findall(r"\[([^\]]*)\]", text) for i in re.findall(r"F\d+", grp)}


def _num_tokens(text: str) -> set[str]:
    text = re.sub(r"\[[^\]]*\]", " ", text)           # citations, however formatted
    text = re.sub(r"\bB\d\b", " ", text)          # model names (B0, B2)
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def check_answer(ans: str, f: Facts) -> str | None:
    """None if the answer is acceptable, else the reason it was rejected."""
    ids = {i["id"] for i in f.items}
    cited = _cited(ans)
    if not cited:
        return "no fact citations"
    if not cited <= ids:
        return "cites a fact that does not exist"
    extra = _num_tokens(ans) - _num_tokens(f.corpus())
    if extra:
        return f"numbers not found in the facts: {', '.join(sorted(extra))}"
    if re.search(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|(thir|four|fif|six|seven|eigh|nine)teen|"
                 r"(twen|thir|for|fif|six|seven|eigh|nine)ty|hundred|half|third|quarter)\b", ans, re.I):
        return "spells numbers out in words, which cannot be checked against the facts"
    by_id = {i["id"]: i.get("model") for i in f.items}
    for sent in re.split(r"(?<=[.!?])\s+", ans):
        named = set(re.findall(r"\bB[012]\b", sent))
        owners = {by_id[i] for i in _cited(sent) if by_id.get(i)}
        if len(named) == 1 and owners and not owners <= named:
            return f"attributes {', '.join(sorted(owners - named))}'s facts to {next(iter(named))}"
    low = ans.lower()
    if any(b in low for b in BANNED):
        return "contains medical-advice wording"
    if len(ans) > 900:
        return "too long"
    return None


def ask_local(question_text: str, f: Facts, timeout: float = 60.0) -> str:
    body = json.dumps({
        "model": LOCAL_MODEL, "stream": False, "keep_alive": "15m",
        "options": {"temperature": 0, "num_predict": 220},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"FACTS:\n{f.prompt_block()}\n\nQUESTION: {question_text}"}],
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"].strip()


def answer(demo, sub: str, left_id: str, right_id: str, q: str, labels: dict) -> dict:
    f, sides = build_facts(demo, sub, left_id, right_id, labels)
    key = q if q in QUESTIONS else ("severity" if SEVERITY_WORDS.search(q) else None)
    question_text = QUESTIONS[q] if q in QUESTIONS else q.strip()[:200]
    fallback = template_answer(key or "summary", f, sides)
    out = {"question": question_text, "facts": f.items, "answer": fallback,
           "source": "built-in template", "note": None, "model": LOCAL_MODEL}
    try:
        raw = ask_local(question_text, f)
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as e:
        out["note"] = f"local model unavailable ({type(e).__name__}); showing the built-in summary"
        return out
    bad = check_answer(raw, f)
    if bad:
        out["note"] = f"local model answer rejected ({bad}); showing the built-in summary"
        out["rejected_answer"] = raw
        if key is None:
            out["answer"] = ("I can only report what the models and checks computed for this patient, not give clinical "
                             "or treatment advice or judgements beyond it. Here is the built-in summary instead. ") + fallback
        return out
    out["answer"], out["source"] = raw, f"local model ({LOCAL_MODEL}), checked against the facts"
    return out
