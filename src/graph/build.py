"""
KG-XAI Phase 1 -- load, validate, and cross-check the knowledge graphs.

Two graphs, both in kg/ (see KNOWLEDGE_GRAPH_SOURCES.md for where every node
and edge came from):

  kg/cow_topology.json    the Circle of Willis -- fixed anatomy, 12 nodes
  kg/guideline_rules.json the formalized decision chain -- eligibility rules
                          plus cross-task plausibility constraints

Deliberately depends on numpy ONLY, not torch_geometric. The graph structure
can and should be validated before the GNN stack is installed -- a graph that
is wrong about anatomy is wrong whether or not PyTorch Geometric can load it,
and finding that out during GNN training (Phases 4-6) would cost far more.
`edge_index()` emits the array PyG wants, without importing it.

The cross-check against real data is the part that matters: the graph's node
labels have to line up with what src/data/occlusion_site.py actually found in
the real cow-msk files, or the two halves of the pipeline are talking about
different vessels.

Usage
-----
    python -m src.graph.build
    python -m src.graph.build --occlusion-json outputs/tables/occlusion_site.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, deque

import numpy as np

KG_DIR = pathlib.Path("kg")
OUT_TAB = pathlib.Path("outputs/tables")


# ------------------------------------------------------------------ loading

def load_graph(path: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def node_index(graph: dict) -> dict[str, int]:
    """Stable id -> row mapping. Order follows the file, so the index is
    reproducible across runs rather than depending on dict iteration."""
    return {n["id"]: i for i, n in enumerate(graph["nodes"])}


def edge_index(graph: dict) -> np.ndarray:
    """(2, n_edges*2) COO array, both directions for an undirected graph --
    the exact layout torch_geometric.data.Data expects, built without
    importing it."""
    idx = node_index(graph)
    pairs = [(idx[e["source"]], idx[e["target"]]) for e in graph["edges"]]
    if not graph.get("directed", False):
        pairs += [(b, a) for a, b in pairs]
    return np.asarray(pairs, dtype=np.int64).T


# --------------------------------------------------------------- validation

def validate_cow(graph: dict) -> list[str]:
    """Returns a list of problems; empty list means the graph is sound."""
    problems = []
    ids = [n["id"] for n in graph["nodes"]]
    labels = [n["topcow_label"] for n in graph["nodes"]]

    if len(set(ids)) != len(ids):
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        problems.append(f"duplicate node ids: {dupes}")
    if len(set(labels)) != len(labels):
        dupes = [k for k, v in Counter(labels).items() if v > 1]
        problems.append(f"duplicate topcow_labels: {dupes}")

    known = set(ids)
    for e in graph["edges"]:
        for end in ("source", "target"):
            if e[end] not in known:
                problems.append(f"edge references unknown node: {e[end]}")

    # Connectivity: an anatomical circle with a detached component means an
    # edge was missed. Checked explicitly rather than assumed.
    adj: dict[str, list[str]] = {i: [] for i in ids}
    for e in graph["edges"]:
        if e["source"] in adj and e["target"] in adj:
            adj[e["source"]].append(e["target"])
            adj[e["target"]].append(e["source"])
    seen, queue = {ids[0]}, deque([ids[0]])
    while queue:
        for nxt in adj[queue.popleft()]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    if len(seen) != len(ids):
        problems.append(f"graph is not connected -- unreachable: {sorted(set(ids) - seen)}")

    # Each side's carotid must reach its own MCA and ACA, and the
    # communicating arteries must bridge the two sides. If this fails, the
    # collateral story the whole GNN rests on is not actually in the graph.
    for side in ("L", "R"):
        for downstream in ("MCA", "ACA"):
            if f"{side}-{downstream}" not in adj.get(f"{side}-ICA", []):
                problems.append(f"{side}-ICA does not connect to {side}-{downstream}")
    for bridge, ends in (("Acom", ("L-ACA", "R-ACA")),):
        for end in ends:
            if end not in adj.get(bridge, []):
                problems.append(f"{bridge} does not bridge {end}")
    return problems


SUPPORT_STATUSES = {"supported", "pending_phase2", "unverifiable_on_isles24",
                    "not_supported", "contradicted"}


def validate_rules(graph: dict) -> list[str]:
    """Structure AND provenance: every rule must cite a source that exists in
    the file and is marked verified. An unverified citation is a failure, not
    a warning -- that is the Phase 1 gate."""
    problems = []
    concepts = {c["id"] for c in graph["concept_nodes"]}
    sources = graph.get("sources", {})

    for sid, s in sources.items():
        if not s.get("verified"):
            problems.append(f"source {sid}: citation not verified")

    for rule in graph["eligibility_rules"]:
        if not rule.get("source_ids"):
            problems.append(f"rule {rule['id']}: no source_ids")
        for sid in rule.get("source_ids", []):
            if sid not in sources:
                problems.append(f"rule {rule['id']}: unknown source {sid}")
        for cond in rule["conditions"]:
            if cond["concept"] not in concepts:
                problems.append(f"rule {rule['id']}: unknown concept {cond['concept']}")
        if rule["implies"]["concept"] not in concepts:
            problems.append(f"rule {rule['id']}: unknown implied concept")

    for con in graph["plausibility_constraints"]:
        for c in con["concepts"]:
            if c not in concepts:
                problems.append(f"constraint {con['id']}: unknown concept {c}")
        if con["strength"] not in ("hard", "soft"):
            problems.append(f"constraint {con['id']}: strength must be hard or soft")
        status = con.get("support", {}).get("status")
        if status not in SUPPORT_STATUSES:
            problems.append(f"constraint {con['id']}: support status {status!r} not in {sorted(SUPPORT_STATUSES)}")
        for sid in con.get("support", {}).get("source_ids", []) + con.get("clinical_basis_source_ids", []):
            if sid not in sources:
                problems.append(f"constraint {con['id']}: unknown source {sid}")
    return problems


def rule_evaluability(graph: dict) -> dict[str, dict]:
    """Which conditions of each rule can be computed on ISLES'24 -- a property
    of the data (see src/data/clinical_audit.py), recorded per rule so the
    Rule Evaluator knows what it may and may not claim."""
    avail = {c["id"]: c.get("available_on_isles24", True) for c in graph["concept_nodes"]}
    out = {}
    for rule in graph["eligibility_rules"]:
        blocked = sorted({c["concept"] for c in rule["conditions"] if avail.get(c["concept"]) is False})
        pending = sorted({c["concept"] for c in rule["conditions"] if avail.get(c["concept"]) == "phase2"})
        out[rule["id"]] = {
            "fully_evaluable": not blocked and not pending,
            "not_evaluable_concepts": blocked,
            "pending_phase2_concepts": pending,
        }
    return out


# ----------------------------------------------------- real-data cross-check

def cross_check_against_real_data(graph: dict, occlusion_json: pathlib.Path) -> dict:
    """Do the graph's nodes line up with the vessels actually found in the
    real cow-msk files?

    Splits confident from low-confidence matches, because they mean different
    things and only the confident ones may set an `occluded` node feature.
    A low-confidence match is not a localisation -- it is the nearest-vessel
    search defaulting to whatever sits closest to the middle when the
    occlusion is beyond the segmented circle entirely (see
    occluded_gating_warning in kg/cow_topology.json)."""
    if not occlusion_json.exists():
        return {"status": "skipped", "reason": f"{occlusion_json} not found"}

    payload = json.loads(occlusion_json.read_text(encoding="utf-8"))
    by_label = {n["topcow_label"]: n["id"] for n in graph["nodes"]}

    confident, low_conf, unmapped = Counter(), Counter(), Counter()
    for row in payload.get("rows", []):
        if row.get("status") != "ok":
            continue
        label = row.get("nearest_cow_label")
        if label not in by_label:
            unmapped[label] += 1
            continue
        (confident if row.get("confident") else low_conf)[by_label[label]] += 1

    # A node whose matches are overwhelmingly low-confidence is a default
    # sink, not a finding. Surfaced rather than left for someone to notice.
    sinks = {
        node: {"confident": confident[node], "low_confidence": low_conf[node]}
        for node in set(confident) | set(low_conf)
        if low_conf[node] >= 3 and low_conf[node] > 2 * confident[node]
    }

    return {
        "status": "ok",
        "subjects_total": int(sum(confident.values()) + sum(low_conf.values())),
        "confident_per_node": dict(confident.most_common()),
        "low_confidence_per_node": dict(low_conf.most_common()),
        "default_sink_nodes": sinks,
        "labels_not_in_graph": dict(unmapped),
        "nodes_never_hit_confidently": sorted(set(by_label.values()) - set(confident)),
    }


# ---------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kg-dir", type=pathlib.Path, default=KG_DIR)
    ap.add_argument("--occlusion-json", type=pathlib.Path,
                    default=OUT_TAB / "occlusion_site.json")
    args = ap.parse_args()

    cow = load_graph(args.kg_dir / "cow_topology.json")
    rules = load_graph(args.kg_dir / "guideline_rules.json")

    print(f"\nCircle of Willis graph: {len(cow['nodes'])} nodes, "
          f"{len(cow['edges'])} edges")
    ei = edge_index(cow)
    print(f"  edge_index shape {ei.shape} (both directions, PyG-ready)")

    cow_problems = validate_cow(cow)
    print("  structure: " + ("OK" if not cow_problems else "FAILED"))
    for p in cow_problems:
        print(f"    !! {p}")

    print(f"\nGuideline rule graph: {len(rules['concept_nodes'])} concepts, "
          f"{len(rules['eligibility_rules'])} eligibility rules, "
          f"{len(rules['plausibility_constraints'])} plausibility constraints")
    rule_problems = validate_rules(rules)
    print("  structure: " + ("OK" if not rule_problems else "FAILED"))
    for p in rule_problems:
        print(f"    !! {p}")

    sources = rules.get("sources", {})
    print(f"  sources: {sum(s.get('verified', False) for s in sources.values())}/"
          f"{len(sources)} citations verified")
    unverified = sorted({sid for r in rules["eligibility_rules"] for sid in r["source_ids"]
                         if not sources.get(sid, {}).get("wording_verified", False)})
    if unverified:
        print(f"  NOTE: recommendation wording not yet checked against the full text "
              f"for: {', '.join(unverified)} -- do not quote these rules until it is")

    evaluability = rule_evaluability(rules)
    print("\n  evaluable on ISLES'24:")
    for rid, e in evaluability.items():
        tag = "yes" if e["fully_evaluable"] else "no "
        why = []
        if e["not_evaluable_concepts"]:
            why.append("no data for " + ", ".join(e["not_evaluable_concepts"]))
        if e["pending_phase2_concepts"]:
            why.append("Phase 2 builds " + ", ".join(e["pending_phase2_concepts"]))
        print(f"    {tag}  {rid:<38} {'; '.join(why)}")

    print("\n  plausibility constraints (drive JDCR and the consistency loss):")
    for con in rules["plausibility_constraints"]:
        print(f"    {con['support']['status']:<24} {con['id']}")

    print("\nCross-check against real data "
          f"({args.occlusion_json}):")
    xc = cross_check_against_real_data(cow, args.occlusion_json)
    if xc["status"] != "ok":
        print(f"  skipped -- {xc['reason']}")
    else:
        conf, low = xc["confident_per_node"], xc["low_confidence_per_node"]
        print(f"  {xc['subjects_total']} real subjects, "
              f"{sum(conf.values())} confident / {sum(low.values())} low-confidence")
        print(f"    {'node':>7} {'confident':>10} {'low-conf':>9}")
        for node in sorted(set(conf) | set(low), key=lambda k: -(conf.get(k, 0) + low.get(k, 0))):
            print(f"    {node:>7} {conf.get(node, 0):>10} {low.get(node, 0):>9}")
        for node, c in xc["default_sink_nodes"].items():
            print(f"  !! {node} is a default sink: {c['low_confidence']} low-confidence vs "
                  f"{c['confident']} confident matches -- only confident matches may "
                  f"set an `occluded` node feature")
        if xc["labels_not_in_graph"]:
            print(f"  !! labels found in real data but absent from the graph: "
                  f"{xc['labels_not_in_graph']}")
        if xc["nodes_never_hit_confidently"]:
            print(f"  never a confident occlusion site in this cohort "
                  f"(expected for communicating arteries): {xc['nodes_never_hit_confidently']}")

    OUT_TAB.mkdir(parents=True, exist_ok=True)
    out = OUT_TAB / "kg_validation.json"
    out.write_text(json.dumps({
        "cow": {"n_nodes": len(cow["nodes"]), "n_edges": len(cow["edges"]),
                "problems": cow_problems},
        "rules": {"n_concepts": len(rules["concept_nodes"]),
                  "n_eligibility_rules": len(rules["eligibility_rules"]),
                  "n_plausibility_constraints": len(rules["plausibility_constraints"]),
                  "problems": rule_problems,
                  "wording_unverified_sources": unverified,
                  "evaluability": evaluability,
                  "constraint_support": {c["id"]: c["support"]["status"]
                                         for c in rules["plausibility_constraints"]}},
        "real_data_cross_check": xc,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    if cow_problems or rule_problems:
        raise SystemExit("graph validation FAILED -- fix before moving past Phase 1")


if __name__ == "__main__":
    main()
