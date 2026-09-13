# Where the knowledge graph comes from

**There is nothing to download.** No pre-built knowledge graph exists for
ISLES'24 or for stroke triage. The original deck's graph box names
*documents* ("AHA/ASA, DAWN, DEFUSE-3"), not a dataset. The graph is authored
from the sources below, and **every citation was checked against a real
record on 2026-09-13** — nothing here is cited from memory.

| File | What it is |
|---|---|
| [`kg/cow_topology.json`](kg/cow_topology.json) | Layer 1 — the Circle of Willis: 12 vessels, 12 connections. The GNN runs on this. |
| [`kg/guideline_rules.json`](kg/guideline_rules.json) | Layer 2 — 16 concepts, 7 eligibility rules, 4 plausibility constraints. Injected into the model and enforced by the consistency loss. |

Check both with `python -m src.graph.build`; build per-patient vessel
features with `python -m src.graph.features --root D:/ISLES-2024/train`.

---

## Verified sources

| Source | Used for | How it was verified |
|---|---|---|
| **DAWN** — Nogueira RG et al., *NEJM* 2018;378(1):11–21, PMID 29129157 | Eligibility rules, groups A/B/C | PubMed record + trial registry NCT02142283 (eligibility text quoted in the rules file) |
| **DEFUSE 3** — Albers GW et al., *NEJM* 2018;378(8):708–718, PMID 29364767 | Eligibility rule | PubMed record + trial registry NCT02586415 |
| **AHA/ASA 2026 guideline** — Prabhakaran S et al., *Stroke* 2026;57(8):e316–e436, PMID 41582814 | Eligibility rules (0–6 h, 6–24 h large core) | PubMed record. **Recommendation wording NOT verified** — full text blocked (HTTP 403); wording came from secondary summaries |
| **Kim H et al.** *Defining the Therapeutic Ceiling of EVT in Large-Core Stroke*, *Stroke* 2026;57(9):2677–2686, PMID 42403349 | ≥110 mL "negligible benefit" rule | PubMed record and abstract. Observational (n=552), diffusion-MRI core |
| **Olivot JM et al.** *HIR predicts infarct progression…*, *Stroke* 2014;45(4):1018–23, PMID 24595591 | Collateral proxy (HIR) and the collateral plausibility constraints | PubMed abstract: "A high HIR predicted poor collaterals with an area under the curve of 0.73" |
| **TopCoW** — Yang K et al., arXiv:2312.17670 (2023) | Vessel definitions and labels | arXiv abstract; defines 13 vessel components |
| **Alastruey J et al.** *J Biomech* 2007;40(8):1794–805, PMID 17045276 | Published adjacency of the Circle of Willis | PubMed record |
| **Liu CF et al.** *Digital 3D Brain MRI Arterial Territories Atlas*, *Sci Data* 2023;10(1):74, PMID 36739282 | Which brain region each vessel supplies (Phase 2) | PubMed record |
| **Luan T et al.** (base paper), *npj Digit Med* 2026;9:441 | The decision chain the graph formalizes (deck Objective 1) | DOI on the Review 1 deck |

---

## What the data allows — measured, not assumed

**Layer 1 lines up with the real data.**
- **Laterality check:** confident clot locations match the side of the follow-up infarct in 89 of 94 front-of-brain cases (94.7%). The pass mark, 80%, was fixed before running.
- **Default-sink filter:** only confident matches set a vessel as occluded. Without that filter, 19 low-confidence "basilar" cases, at a median 39 mm from the vessel, would have entered as fake basilar strokes.
- **Label 15 left out:** TopCoW's 13th vessel (a rare variant) appears in 5 of 149 patients and is never the clot site.

**Layer 2 runs into three data limits.**

| Limit | Measured | Consequence |
|---|---|---|
| **No timing data** | All 7 timing columns (onset-to-door, door-to-imaging, door-to-groin, …) are empty for 149/149 | No time-window condition can be evaluated. **No eligibility rule is fully evaluable on ISLES'24** — the Rule Evaluator can only report "imaging and clinical criteria met; time window not recorded." |
| **No ASPECTS** | Not in the release | The two AHA/ASA 2026 rules can't be evaluated at all. The trial rules, written in mL, can (after Phase 2). |
| **Few LVO-negative patients** | 4/149 | The "large infarct needs an occlusion" constraint can't be checked; excluded from JDCR by default. |

**Where the plausibility constraints stand** (these drive JDCR and the consistency loss):

| Constraint | Status |
|---|---|
| Infarct side matches occlusion side | **Supported** — 94.7% in the ground truth, so it's a *soft* constraint (a hard one would punish the model for matching real data) |
| Good collaterals limit infarct size | Pending Phase 2 (needs the HIR proxy) |
| Poor collaterals expect an infarct | Pending Phase 2 — may be weak, since every ISLES'24 patient was successfully reperfused |
| Large infarct needs an occlusion | Unverifiable (4 negatives) |

---

## Not used, and why

- **UMLS / SNOMED CT / BioPortal** give standard concept IDs, not triage
  rules. Add them only if a reviewer asks for ontology grounding.
- **Generic "medical knowledge graph" datasets** (Kaggle, HuggingFace) don't
  contain the stroke triage decision chain, and their provenance usually can't
  be traced, which contradicts deck Objective 4.
