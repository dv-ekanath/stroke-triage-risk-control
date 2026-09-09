"""
Review 1, item 1.4 -- recover the center label for each case.

The ISLES'24 public training set is TWO centers, not one and not seven:
Center 1 = 100 cases (67.1%), Center 2 = 49 (32.9%), on different scanner
fleets (Siemens Somatom Force / Xcite / AS+ versus Philips Brilliance 64 /
Ingenuity).  The base paper's claim of leave-one-center-out across 7 centers
with GE, Siemens and Philips scanners is not compatible with this release --
verifying that here is worth doing before repeating the claim.

Recovery order, most to least trustworthy:
  1. per-subject phenotype CSV carrying an explicit site or center column
     (the 2026 release ships this as train/phenotype/<sub>/ses-01/
     <sub>_ses-01_demographic_baseline.csv -- no participants.tsv at all)
  2. participants.tsv / any *.tsv carrying an explicit site or center column
  3. BIDS JSON sidecars: Manufacturer / ManufacturersModelName / StationName
  4. unsupervised clustering on acquisition parameters -- reported as INFERRED,
     never as known

If none succeeds, the cross-center robustness check is dropped and that is
stated.  Do not fabricate a proxy and call it a center.

Usage
-----
    python3 -m src.data.center_labels --root /path/to/ISLES-2024
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import Counter

OUT_TAB = pathlib.Path("outputs/tables")
SUBJECT_RE = re.compile(r"(sub-[A-Za-z0-9]+)")

SITE_COLS = ("center", "centre", "site", "institution", "hospital", "scanner")
VENDOR_KEYS = ("Manufacturer", "ManufacturersModelName", "ManufacturerModelName",
               "StationName", "InstitutionName", "DeviceSerialNumber")


def subject_of(path: pathlib.Path) -> str:
    m = SUBJECT_RE.search(str(path))
    return m.group(1) if m else path.stem


def from_phenotype_csv(root: pathlib.Path):
    """Strategy 1 -- an explicit site column in a per-subject phenotype CSV.

    The 2026 ISLES'24 release has no participants.tsv; instead each subject
    carries its own train/phenotype/<sub>/ses-01/<sub>_..._baseline.csv with a
    'Center' column as its first field.  One row of data, subject taken from
    the path rather than from an id column inside the file.
    """
    out: dict[str, str] = {}
    used = None
    for csv in sorted(root.glob("**/phenotype/**/*.csv")):
        try:
            lines = csv.read_text(errors="replace").splitlines()
        except OSError:
            continue
        if len(lines) < 2:
            continue
        header = [h.strip().lower() for h in lines[0].split(",")]
        hit = next((i for i, h in enumerate(header)
                    if any(k in h for k in SITE_COLS)), None)
        if hit is None:
            continue
        sub = subject_of(csv)
        val = lines[1].split(",")[hit].strip() if len(lines[1].split(",")) > hit else ""
        if val:
            out[sub] = val
            used = used or f"phenotype-csv:{header[hit]}"
    return out, used


def from_tsv(root: pathlib.Path):
    """Strategy 2 -- an explicit site column in any TSV."""
    for tsv in sorted(root.glob("**/*.tsv")):
        try:
            lines = tsv.read_text(errors="replace").splitlines()
        except OSError:
            continue
        if len(lines) < 2:
            continue
        header = [h.strip().lower() for h in lines[0].split("\t")]
        hit = next((i for i, h in enumerate(header)
                    if any(k in h for k in SITE_COLS)), None)
        if hit is None:
            continue
        sid = next((i for i, h in enumerate(header)
                    if "participant" in h or h in ("subject", "sub", "id")), 0)
        out = {}
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) > max(hit, sid):
                out[parts[sid].strip()] = parts[hit].strip()
        if out:
            return out, f"tsv:{tsv.relative_to(root)}:{header[hit]}"
    return {}, None


def from_sidecars(root: pathlib.Path):
    """Strategy 2 -- scanner vendor/model from BIDS JSON sidecars."""
    per_subject: dict[str, Counter] = {}
    for js in sorted(root.glob("**/*.json")):
        if js.name == "dataset_description.json":
            continue
        try:
            meta = json.loads(js.read_text(errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        vals = [str(meta[k]).strip() for k in VENDOR_KEYS
                if k in meta and str(meta[k]).strip()]
        if vals:
            per_subject.setdefault(subject_of(js), Counter()).update(["|".join(vals)])

    if not per_subject:
        return {}, None, {}
    resolved = {s: c.most_common(1)[0][0] for s, c in per_subject.items()}
    return resolved, "json-sidecar", Counter(resolved.values())


def normalise_vendor(raw: str) -> str:
    low = raw.lower()
    if "siemens" in low:
        return "Center-1 (Siemens)"
    if "philips" in low:
        return "Center-2 (Philips)"
    if "ge " in low or low.startswith("ge") or "general electric" in low:
        return "UNEXPECTED (GE)"
    return f"UNKNOWN ({raw[:40]})"


def normalise_phenotype(raw: str) -> str:
    """The phenotype CSV's 'Center' column is a bare integer (1, 2, ...)."""
    r = raw.strip()
    return f"Center-{r}" if r else "UNKNOWN (blank)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path)
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"{root} does not exist")
    OUT_TAB.mkdir(parents=True, exist_ok=True)

    print(f"\nCenter-label recovery  root={root}")

    all_subjects = sorted({subject_of(p) for p in root.glob("**/sub-*")
                            if p.is_dir() and SUBJECT_RE.fullmatch(p.name)})

    labels, source = from_phenotype_csv(root)
    if labels:
        print(f"  [1] explicit site column found in phenotype CSVs -- {source} "
              f"({len(labels)} subjects)")
        missing = [s for s in all_subjects if s not in labels or not labels[s]]
        if missing:
            print(f"      NOTE: {len(missing)} subject(s) have a blank Center "
                  f"field in their own CSV: {', '.join(missing)}")
        labels = {s: normalise_phenotype(v) for s, v in labels.items()}
    else:
        print("  [1] no site/center column in any phenotype CSV")
        labels, source = from_tsv(root)
        if labels:
            print(f"  [2] explicit site column found -- {source}")
        else:
            print("  [2] no explicit site/center column in any TSV")
            labels, source, vendors = from_sidecars(root)
            if labels:
                print(f"  [3] recovered from JSON sidecars ({len(labels)} subjects)")
                print("      raw vendor strings:")
                for v, n in vendors.most_common(10):
                    print(f"        {n:>4}  {v[:70]}")
                labels = {s: normalise_vendor(v) for s, v in labels.items()}
            else:
                print("  [3] no vendor fields in sidecars either")
                print("  [4] FALLBACK: cluster on acquisition parameters, and report")
                print("      the result as INFERRED, not known.  Not run automatically")
                print("      -- inspect the tree first, then decide the features.")
                (OUT_TAB / "center_labels.json").write_text(json.dumps(
                    {"root": str(root), "status": "unrecovered",
                     "action": "drop the cross-center robustness check and say so"},
                    indent=2))
                print(f"\nwrote {OUT_TAB/'center_labels.json'} (status: unrecovered)")
                return

    counts = Counter(labels.values())
    print(f"\n  {len(labels)} subjects labelled")
    for name, n in counts.most_common():
        print(f"    {n:>4}  ({n/len(labels)*100:>5.1f}%)  {name}")
    print("\n  descriptor reference: Center 1 = 100 (67.1%), Center 2 = 49 (32.9%)")
    if any("GE" in k for k in counts):
        print("  NOTE: a GE scanner appeared. The descriptor lists Siemens and")
        print("        Philips only -- check this before citing the vendor claim.")
    if len(counts) > 2:
        print(f"  NOTE: {len(counts)} distinct labels, descriptor says 2.")

    (OUT_TAB / "center_labels.json").write_text(json.dumps(
        {"root": str(root), "source": source, "status": "recovered",
         "counts": dict(counts), "labels": labels}, indent=2))
    print(f"\nwrote {OUT_TAB/'center_labels.json'}")


if __name__ == "__main__":
    main()
