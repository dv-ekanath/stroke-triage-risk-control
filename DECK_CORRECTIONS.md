# Deck corrections — Review 2

What to change in the existing presentation deck, and why. Every correction
here traces to a real, measured fact in `PROGRESS_REPORT.md` — nothing is a
guess. Organized by *what kind of slide* it affects, since the original
deck's exact slide numbers aren't available in this repo.

---

## 1. Dataset facts (any "Dataset" or "Data" slide)

| Deck currently says | Correct to | Why |
|---|---|---|
| 7 centers | **2 centers** (99 / 49 subjects) | Measured directly from the phenotype CSVs' own Center column — see PROGRESS_REPORT.md §3. |
| GE, Siemens, Philips scanners | Siemens and Philips only | No GE scanner appears anywhere in this release. |
| Core/penumbra segmentation target | **Final infarct** (day 2–9 follow-up DWI) | This is what ISLES'24's lesion mask actually is — a different, later-timepoint target than the base paper's. |
| Binary LVO detection | **LVO localization** (4-class: none / proximal anterior / other confident / unlocalized) | Only 4 of 149 subjects are LVO-negative — a binary detector has no negatives to learn from. Reframed as "where," not "whether." |
| Real collateral grade | **HIR-derived proxy** — and note it failed validation as a real collateral signal (Phase 2) | ISLES'24 provides no expert collateral grade at all. Never call the HIR bin "collateral grade" out loud in the defense. |
| Full guideline-eligibility checking (DEFUSE-3/DAWN) | **Per-criterion only**, "time window not recorded" | All 7 timing fields are empty for 149/149 subjects; no ASPECTS either. |

**If asked live "why not 7 centers like the paper":** *"We checked the release's own metadata directly rather than trust the paper's description, and found 2, not 7 — we're reporting what the data actually contains."* That is a strength to state plainly, not something to hedge around.

---

## 2. System architecture slide

| Deck currently shows | Correct to | Why |
|---|---|---|
| 3 separate per-modality encoders → fusion | **One shared encoder** (the Review 1 SegResNet trunk), reused by B0/B1/B2 | Disclosed simplification for an 8 GB GPU and a 2-week window — stated as a scope decision, not hidden. |
| LVO + collateral + segmentation, 3 independent heads | **Segmentation only** is the live head; LVO/collateral heads were built, tested four separate ways, and shelved as unlearnable at this data scale | Say this as a tested finding, not an omission — see §4. |
| Knowledge graph feeding a decision layer after prediction | Graph feeds in **during** decoding, via cross-attention at every decoder stage | This is real and unchanged from the original proposal — keep this slide's actual GNN diagram. |

**New sub-panel worth adding:** the zero-init cross-attention gate. One line: *"the graph pathway is mathematically proven to start as a no-op — it can only ever add to the baseline, never break it at initialization."* This is a genuinely good engineering detail to show, since it answers "how do you know the graph part didn't wreck the working baseline" before anyone asks it.

---

## 3. "Expected results" / hypothesis slide — replace entirely

Whatever this slide currently claims (e.g. "the knowledge graph will improve Dice / LVO AUC / collateral grading"), **replace it with the actual, measured Phase 8 result:**

> **The graph-injected model does not show a statistically significant
> segmentation-accuracy improvement over the plain baseline** (paired
> per-patient test across 345 matched observations: p = 0.24).
>
> **What the graph-injected model does show: it never collapsed.**
> Trained from scratch, the plain baseline completely failed — predicted
> no lesion at all, for every patient — in 2 of 9 runs. The graph-injected
> version never failed this way, in any of its 12 runs.

Follow immediately with the one honest caveat, said out loud rather than
left for a question: *"We can't yet say for certain whether that stability
comes from the graph itself or simply from warm-starting the model from a
pretrained checkpoint — B0 always trains from scratch, B1/B2 never do, and
this design doesn't isolate the two. We're reporting the pattern, not
overclaiming the cause."* **Two follow-up tests since this slide was
first written make that caveat sharper, not weaker: a same-size random
graph trains exactly as stably and as accurately as the real anatomy
(§4, new item 3), and a plain unweighted ensemble of B0's own 3 seeds —
no graph at all — closes the same stability gap on its own (§4a). The
honest claim is now: the graph correlates with stability here; it has
not been shown to be its cause, and a much cheaper fix achieves the same
practical outcome.*

**Why present a negative accuracy result instead of hiding it:** it already
survived one faculty review pass on the strength of honesty about what
worked and what didn't (Review 1's threshold approach was rejected, and
the graph pivot was itself a response to being asked for more rigor, not
less). A disclosed limitation, backed by a real statistical test, reads as
more credible than an undisclosed claim that later falls apart under a
follow-up question.

---

## 4. Add a slide: "What we tried and ruled out" (new, if there's room)

This project generated two genuine negative findings worth a slide of
their own, because they show the testing discipline was real:

1. **6 of 7 clinical plausibility rules do not hold in this cohort.**
   Tested against real data, not assumed. Likely explanation: universal
   successful reperfusion scrambles the size/location signal the rules
   depend on. Confirmed it wasn't a single-center artifact by re-testing
   separately in both centers.
2. **LVO localization and collateral grading could not be learned from
   this data**, in four separate architecture attempts (global pooling →
   per-vessel GNN sampling → same + augmentation → same + a frozen,
   undivided encoder). All four converged on predicting one constant
   answer for every patient. Diagnosed with real evidence (inspected the
   actual per-patient predictions, not just the accuracy number) before
   concluding it was a data-scale limit, not a bug.
3. **A same-size random graph performs identically to the real
   Circle-of-Willis anatomy.** Tested directly: B1, fold 0, 3 seeds, real
   anatomical edges vs. 12 random edges over the same 12 nodes. Dice and
   JDCR came out statistically indistinguishable (random was marginally
   *higher* on both). The cross-attention gate controlling how much the
   graph can influence the decoder stayed near zero in every run,
   regardless of which graph it was wired to — which is the likely reason
   the wiring didn't matter.
4. **A second, independent collateral measure also failed — on two
   different outcomes.** Not the same test repeated: a pure graph-topology
   measure (an intact communicating-artery route, no learned component,
   no occlusion label) tested against both final infarct volume and
   admission NIHSS, chosen specifically because NIHSS is pre-treatment and
   so isn't confounded by this cohort's universal successful reperfusion.
   Null on both. Two independent collateral measures, two independent
   outcomes, both null — this is now reported as settled, not
   underpowered.

**Why this slide helps rather than hurts:** it directly answers "is this
just a hand-wave" with "no, we tested it four different ways and inspected
the failure mode each time before giving up on it."

---

## 4a. Add a slide: "Three findings beyond the graph" (new)

With the graph's accuracy claim settled negative, three more tests —
run after the above, same discipline, pre-registered before running —
produced genuinely new, positive results that don't depend on the graph
winning at all:

1. **The Rule Evaluator's own output predicts which of its predictions
   are wrong.** Flag a prediction using only what's known at inference
   time (does the predicted side disagree with the admission occlusion
   side, or is the predicted volume near zero) — thresholds fixed from
   those signals alone, never from Dice. (A third criterion, calibration
   error, was dropped after catching that it isn't actually
   ground-truth-free — see the source note — and removing it made the
   result *stronger*, not weaker.) Flagged predictions scored a median
   Dice of 0.0000; unflagged, 0.1995 (p = 1.4×10⁻⁵⁰, n = 1038). Holds
   inside B1 and B2 alone, which never collapsed — not just a
   re-detection of B0's known failures. This turns the Rule Evaluator
   from a reporting metric into a usable "send this one to a
   radiologist" mechanism.
2. **A cheap, no-retraining ensemble nearly erases the training-collapse
   problem.** Averaging the 3 existing baseline seeds' predictions, with
   no knowledge of which seed (if any) collapsed, lands within 0.01 Dice
   of what you'd get by correctly guessing which seed to drop in advance,
   on both folds that had a real collapse.
3. **Multi-task learning works on this architecture, given a real
   label.** Swapped the two failed classification heads (item 2, previous
   slide) for a single regression head predicting NIHSS at admission —
   real, non-circular, 98% populated, never used anywhere else in this
   project. All 3 seeds: a significant correlation (rho≈0.56-0.60,
   p<0.0001), a real spread of predictions (not the constant-answer
   collapse that killed the earlier heads), at essentially no cost to
   segmentation. **Conclusion: the earlier heads failed because of label
   quality, not because multi-task learning is broken here.**

**Why this slide matters most:** it's the answer to "so none of this
worked" — three of it did, and none of the three needed the graph to be
right about accuracy.

---

## 5. Novelty statement — tighten, don't retract

The original one-sentence novelty claim (multi-task consistency via a GNN +
guideline cross-attention + a new JDCR metric) is still accurate as a
description of **what was built**. What changes is the claim about
**what it does**:

- Before: *"this improves stroke triage accuracy and gives it clinical
  coherence."*
- Now: *"this proposes JDCR, a new way to quantify cross-task anatomical
  consistency for a multi-task stroke model, grounded in each patient's
  real vessel anatomy rather than an arbitrary post-hoc rule, and shows
  on real data that its components predict which predictions are
  actually wrong (p=1.4×10⁻⁵⁰). As important a result as the positive
  one would have been, this project also rigorously establishes where
  the consistency framework does and doesn't pay off: it doesn't move
  raw segmentation accuracy (paired test, p=0.24), and we traced that
  down to a mechanism — the cross-attention gate stays near zero
  regardless of training outcome, and a random graph of the same size
  performs identically to the real anatomy, so the specific anatomical
  structure isn't what's doing any work that does happen. A separate
  finding: this architecture's multi-task design was never broken — it
  was specifically the two labels we first tried (LVO localization,
  collateral grade) that were unusable; given a real label (NIHSS) the
  same design learns a genuine signal."*

**A deliberate framing choice, worth saying out loud if asked:** we are
leading with JDCR as a *metric* contribution, not primarily with the
triage/flagging *application* of it. "Flag predictions you shouldn't
trust" is conceptually close enough to Review 1's rejected certified-
threshold idea that presenting it as the headline novelty risks the same
pushback. Proposing a new, validated way to *measure* consistency is a
different and better-defended claim; the triage result is presented as
evidence the metric has real discriminative power, not as the novelty
itself.

This is a smaller, truer claim, and a defensible one under direct
questioning — which is what the previous claim (thresholding "not enough
novelty") was not.

---

## 6a. Optional slide: segmentation accuracy, benchmarked and improved (NOT the novelty slide)

If there's room for one more slide, this one is worth having — but label
it clearly as engineering, separate from the novelty slides above, so it
can't be mistaken for the project's contribution.

> We benchmarked our baseline against the actual ISLES'24 challenge
> leaderboard (453 participants, 12 teams submitted a final solution).
> The winning team scored 0.285 Dice on the hidden test set — we were at
> 0.24. Their own published write-up credited two preprocessing steps
> (skull-stripping, clinically-motivated perfusion windowing) that our
> pipeline didn't have. We added both, confirmed the result across 3
> seeds (0.274 / 0.267 / 0.279, sd 0.005), and closed most of the gap:
> mean Dice 0.273, within 0.012 of the winning score, and we now beat
> their volume-difference and lesion-F1 numbers outright.

Say explicitly, before anyone asks: *"This is applied engineering — we
benchmarked against the real state of the art and closed a known gap.
It is not our novelty claim."*

---

## 6. One-line answers for likely faculty questions

- **"So did the knowledge graph work or not?"** — *"The learned GNN
  pathway didn't improve accuracy, tested properly with a paired
  statistical test, and a random graph of the same size does just as
  well — so we can't credit the specific anatomy. The knowledge graph's
  rule-checking layer did work: it catches real training failures and
  flags unreliable predictions with a large, significant effect. Those
  are two different mechanisms, and we're not claiming success for one
  using evidence from the other."*
- **"Couldn't the stability result just be the warm start?"** — *"Also
  tested: a plain ensemble of B0's own 3 seeds, no graph at all, closes
  the same gap. So yes, warm-starting (or something not specific to the
  graph) likely explains most of it — which is also why we're presenting
  the ensemble fix as the practical solution, not the graph."*
- **"Why keep the GNN in the architecture if it isn't doing anything?"**
  — *"It's reported as a tested, disclosed negative result, not removed
  retroactively to hide it. The two things in this project that actually
  work — the laterality rule and the triage signal — don't require the
  learned GNN at all; they use the knowledge graph's rule layer directly
  on any model's output, B0 included."*
- **"Why only 2 centers, the paper says 7?"** — *"We checked the release's
  own data directly and found 2. We're reporting what's actually there."*
- **"Why no live collateral grading?"** — *"The public release doesn't
  include an expert collateral grade at all. We built a proxy, tested it
  four independent ways, and it failed every one — so we say that plainly
  instead of quietly using it as if it were validated."*
- **"Is 345 patients enough for that p-value to mean anything?"** —
  *"That's 345 paired observations across 7 independent training runs on
  non-collapsed models, using the same validation patients — small in the
  clinical-trial sense, appropriately powered for what a course-scale GPU
  budget can support, and reported with the actual p-value rather than
  just a description of which mean was bigger."*
- **"Your Dice jumped from 0.24 to 0.27 — is THAT the novelty?"** —
  *"No, and we want to be upfront about that: we read the actual ISLES'24
  challenge winner's published account of what won it, and adopted their
  two preprocessing steps — skull-stripping and clinically-motivated
  perfusion windowing — which our pipeline didn't have. That's applied
  engineering borrowed from a published source, confirmed across 3 seeds,
  and it closed most of our gap to their winning score. It is not
  presented as our contribution; it's reported separately, in its own
  section, specifically so it isn't confused with the actual novelty
  (JDCR and the mechanism-level graph diagnosis)."*

---

## Source of every number above

`PROGRESS_REPORT.md`, §§3, 4–5, 6a, 6b, 6c. Every figure in this document
(centers, subject counts, the p-value, the collapse count) is copied
verbatim from there, which in turn traces to files under `outputs/tables/`.
