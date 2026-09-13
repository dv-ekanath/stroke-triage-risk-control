# KG-XAI Build Plan — Knowledge-Graph-Guided Multi-Task Learning for Stroke Triage

Restores the original project direction (per faculty feedback on Review 1:
the certified-threshold work read as insufficiently novel on its own, and the
original knowledge-graph plan should not have been dropped). Nothing built
this session is discarded — the real data pipeline, the trained baseline
segmentation model, and the validated conformal calibration layer all become
inputs to this plan rather than replaced by it.

**Open decision, resolve before Phase 4:** faculty said to use a GNN. Two
readings are live (see conversation record) —
**(A)** GAT over a small hand-authored clinical-guideline graph (the
original deck's literal plan), or
**(B)** a GNN over the *real* Circle-of-Willis vascular topology (13
anatomical segments, fixed connectivity, per-patient node features from
imaging) — the stronger, more defensible reading, since collateral
circulation is literally blood rerouting through the graph's Acom/Pcom
edges, not a metaphor.
This plan is written so Phases 1–3 are useful under either reading, and the
decision only has to be locked before Phase 4. Confirm with faculty which
was meant; default to (B) if not confirmed, with (A)'s rule table folded in
at Phase 8 as the Rule Evaluator's constraint set either way.

**Discipline carried over from Review 1** (this is what made the last cycle
defensible): validate every new component on a small, controlled case before
trusting it on the real pipeline, and treat every gate below as a real
stop — a phase that fails its test is not "good enough for now," it's
blocking, exactly like the calibration layer's synthetic-data gate was.

---

## Phase 1 — Environment for graph learning

**Goal:** the `.venv` can actually run graph neural networks; nothing here
touches the model yet.

**Build:**
- Install PyTorch Geometric into the existing `.venv` (`torch-geometric` +
  its `torch-scatter`/`torch-sparse` companions, matched to the installed
  `torch==2.11.0+cu128` build).
- Update `requirements.txt` with the pinned versions.

**Test before moving on:** a trivial script builds a 5-node toy graph, runs
one `GATConv` (or `GCNConv`) forward pass on the GPU, confirms output shape
and `torch.cuda.is_available()` still true inside the same process. If this
doesn't import and run cleanly, nothing downstream can be trusted — same
principle as the calibration-layer gate in Review 1.

---

## Phase 2 — Collateral proxy from real perfusion data

**Goal:** fill the documented ISLES'24 gap (no collateral grade shipped)
with a derived, clinically-motivated proxy, rather than skipping the head
the original architecture calls for.

**Build:** `src/data/hir_proxy.py` — computes the Hypoperfusion Intensity
Ratio (volume of Tmax>10s ÷ volume of Tmax>6s) per subject from the CBF/Tmax
maps already loading correctly in `src/model/dataset.py`. Bin into a 4-level
proxy (0–L3) matching the original architecture's output shape.

**Test before moving on:** HIR values fall in a plausible range (published
HIR is typically 0–1); subjects already known to be LVO-positive (from
`src/data/occlusion_site.py`, real, already validated) show worse (higher)
HIR on average than LVO-negative subjects, checked as a simple two-group
comparison — if this direction is backwards, the proxy computation has a
bug, full stop, before it goes anywhere near a model.

---

## Phase 3 — Multi-task heads on the existing trunk

**Goal:** LVO detection and occlusion-site classification as trained heads,
not just derived labels sitting in a JSON file.

**Build:** extend `src/model/` with H2 (LVO detection, binary) and H3
(occlusion-site classification, 4-class: ICA/M1/M2-or-distal/other) heads
branching off the same shared trunk that already produces the segmentation
result. Labels: LVO from non-empty `lvo-msk` (already computable), site from
the real nearest-distance derivation (`occlusion_site.json`, already run).
Loss: add classification terms to the existing Dice+CE segmentation loss.

**Test before moving on:** train on one fold, a handful of epochs first
(mirror the smoke-test discipline from the baseline run) to confirm no
crash / no NaN before committing to a full run; then a full run, checked
against two floors: LVO-detection AUC clearly above 0.5, site-classification
accuracy clearly above the majority-class baseline (from the real class
distribution: M1 is 49.3% of confident cases, so majority-class alone scores
~49%). No task should look catastrophically worse than training it alone
would — that's the standard multi-task negative-transfer check.

---

## Phase 4 — Knowledge graph construction — **decision gate**

**Goal:** the actual graph object, built from real anatomy and/or real
guideline documents, not a placeholder.

**Build**, per whichever reading was confirmed:
- **(A) Guideline-rule graph:** ~15–25 nodes (`OcclusionSite`,
  `CoreVolumeRange`, `CollateralProxyRange`, `TimeWindow`,
  `EligibilityClass`), edges carrying threshold values, sourced from the
  2026 AHA/ASA guideline (DOI 10.1161/STR.0000000000000513), DAWN
  (Nogueira et al. 2018), DEFUSE-3 (Albers et al. 2018), and the base
  paper's own decision chain. Stored as a small JSON edge list.
- **(B) Vascular topology graph:** 13 nodes (BA, L/R-ICA, L/R-MCA, L/R-ACA,
  L/R-PCA, Acom, L/R-Pcom) with fixed anatomical adjacency (the same
  TopCoW convention already verified geometrically against real `cow-msk`
  data this session). Per-patient node features: occlusion present at that
  segment (from Phase 3's site head), local hypoperfusion severity (from
  Phase 2's HIR-contributing maps).

Both are cheap to build in parallel if the decision is still unconfirmed —
(A) is a day of literature extraction, (B) is mostly already-verified
geometry plus a feature-population script.

**Test before moving on:** manually inspect the instantiated graph for 5–10
real subjects with known, unambiguous occlusion sites (from the confident
bucket in `occlusion_site.json`, distance ≤15mm) and confirm the graph marks
the occlusion at the anatomically correct node. A graph that's wrong on the
easy, confident cases is not ready to guide anything.

---

## Phase 5 — GNN encoder, validated standalone before fusion

**Goal:** prove the GNN learns something real from the graph *before*
wiring it into the bigger model, where a bug would be much harder to isolate.

**Build:** `src/graph/encoder.py` — a GAT (or GCN, per Phase 1's toy test)
encoder over the Phase 4 graph, trained on its own to predict something
checkable directly from graph structure — e.g., predict LVO status or the
HIR-proxy bin from the graph's node features alone, no image encoder
involved yet.

**Test before moving on:** the standalone GNN beats a trivial baseline
(majority class, or logistic regression on the same node features flattened)
on held-out subjects. If a GNN can't beat flattening the same features into
a plain classifier, the graph structure isn't adding information yet, and
cross-attention fusion (Phase 6) would just be adding complexity to
something that doesn't work — fix the graph or the features first, don't
proceed on hope.

---

## Phase 6 — Cross-attention injection, tested as an ablation

**Goal:** the graph actually changes the image model's predictions for the
better — measured, not assumed.

**Build:** inject Phase 5's GNN node embeddings into the shared decoder's
features via cross-attention (graph embeddings as keys/values, decoder
features as queries), matching the original architecture's "Guideline
Injection" box.

**Test before moving on:** train the *same* multi-task model from Phase 3,
same fold, same seed, twice — once with the cross-attention injection, once
without (the Phase 3 model *is* the without-condition, already trained).
Compare LVO-detection AUC, site accuracy, and Dice. The graph module has to
clear a real bar here: **it should not make any task-head metric worse**,
and should measurably improve at least one (most plausibly LVO detection or
site classification, since those are what the vascular graph most directly
informs). Report this ablation honestly even if the improvement is small —
a null result here, reported, is more defensible than an unmeasured claim.

---

## Phase 7 — Differentiable consistency loss + joint training

**Goal:** train the full model with the graph in the loop, not just as a
frozen feature injector.

**Build:** add the consistency loss term — penalize joint outputs (LVO
status, site, core volume, collateral proxy) that violate the graph's
encoded constraints, backpropagated jointly with the task losses.

**Test before moving on:** compute the Joint-Decision Consistency Rate
(JDCR) — fraction of held-out cases whose joint output is graph-consistent —
with vs. without the consistency loss term, same fold/seed. The loss term
needs to move JDCR up; if it doesn't, or if it tanks the task-head metrics
to buy consistency, that trade needs to be reported and tuned (loss
weighting), not hidden.

---

## Phase 8 — Rule Evaluator + JDCR + certified volume decision

**Goal:** the clinician-facing consistency check, using the already-built
statistical rigor rather than a bare cutoff.

**Build:** the post-hoc Rule Evaluator checks the trained model's joint
output against the formalized decision chain (Phase 4's rule graph, if (A),
or a small explicit rule set derived from the guidelines if the graph itself
was vascular topology (B)). **The volume-eligibility check inside it uses
the already-validated conformal predictive system + Learn-then-Test
certification from Review 1 (`src/conformal/`)** — a statistically certified
decision with a finite-sample risk guarantee, not a hand-set threshold. This
is the direct answer to "thresholding isn't the novelty": it isn't presented
as one here, it's a component inside a larger consistency check.

**Test before moving on:** JDCR computed and logged at train and test time,
per the original methodology slide. Confirm the certified volume decision
still clears the real-data feasibility check from Review 1 (τ=110mL clears
the LTT sample-size floor by a margin of 1 case — re-verify this hasn't
changed if the fold composition changes).

---

## Phase 9 — Rationale generation

**Goal:** a clinician-readable explanation, deterministic, no LLM.

**Build:** template filling over the joint output — predicted values, which
graph nodes/rules were satisfied or violated, the certified decision and its
position relative to τ, in the style already specified in the v4 proposal's
§5.4 (that section carries over unchanged in spirit, now with graph-node
activations to report instead of just the volume interval).

**Test before moving on:** manually review generated rationales for 10 real
subjects spanning confident and low-confidence occlusion-site cases; confirm
each rationale is factually consistent with that subject's actual head
outputs and graph state — a generated rationale that misstates its own
model's output is worse than no explanation.

---

## Phase 10 — Full evaluation, ablations, write-up

**Goal:** the numbers that go in front of faculty next.

**Build/run:**
- All metrics: AUC (LVO), κ (site, if ordinal framing used) or accuracy,
  Dice/AVD/lesion-F1/ALD (segmentation, vs. the leaderboard as before), ECE
  (not yet implemented anywhere in this codebase — add it), JDCR,
  concept-attention alignment (do the GAT attention weights concentrate on
  clinically sensible graph nodes, checked qualitatively on a handful of
  cases).
- Ablations, each reported whether or not it flatters the model: with/without
  graph injection (Phase 6's comparison), with/without consistency loss
  (Phase 7's comparison), GAT vs. plain GCN if Phase 1 tried both, reading
  (A) vs. (B) graph if both were built.
- Update `stroke_triage_proposal_v4.md` (or a new v5) and the prior-art
  table to reflect what was actually built and measured, not the plan.

**Test before calling it done:** every number quoted in the write-up traces
to a real `outputs/tables/*.json` file, exactly like `REVIEW1_RESULTS.md`
did for Review 1 — no recalled-from-memory figures in the next presentation.

---

## Summary table

| Phase | Deliverable | Gate before proceeding |
|---|---|---|
| 1 | PyTorch Geometric installed | Toy graph forward pass runs on GPU |
| 2 | HIR collateral proxy | Correct direction vs. known LVO status |
| 3 | LVO + site heads trained | Both beat their majority-class floor |
| 4 | Knowledge graph built | Correct on 5–10 confident real cases |
| 5 | Standalone GNN | Beats a flat-feature baseline |
| 6 | Cross-attention fusion | Ablation: no task-head metric gets worse |
| 7 | Consistency loss + joint training | Ablation: JDCR improves |
| 8 | Rule Evaluator + certified decision | JDCR logged; τ=110 feasibility re-checked |
| 9 | Rationale generation | Manually correct on 10 real subjects |
| 10 | Full evaluation + write-up | Every number traces to a real output file |
