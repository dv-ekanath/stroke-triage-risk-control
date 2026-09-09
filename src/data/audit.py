"""
Review 1, item 1.3 -- the ISLES'24 data audit.

Answers, in order of how badly you need the answer:

  1. What is the real volume distribution?
  2. What is the class balance at each candidate tau, and what does the
     majority-class baseline score?  (proposal v3 section 2.1: at tau = 70 mL a
     constant "below cutoff" predictor scores in the low-to-mid 80s, so plain
     accuracy is not reportable)
  3. How many cases sit ABOVE each candidate tau -- and does that clear the
     Learn-then-Test certification floor?

Point 3 is the gate.  R_futile = P(say eligible | V >= tau) can only be
certified from cases with V >= tau.  If there are fewer of those than the floor
in src/conformal/ltt.py, no amount of modelling will produce a guarantee at
that threshold, and the primary tau must be chosen elsewhere.

Usage
-----
    python3 -m src.data.audit --root /path/to/ISLES-2024
    python3 -m src.data.audit --root /path/to/ISLES-2024 --mask-pattern '*lesion*msk*.nii.gz'
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np

from src.conformal.ltt import min_n_to_certify

TAUS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 130, 150]
OUT_TAB = pathlib.Path("outputs/tables")
OUT_FIG = pathlib.Path("outputs/figures")

# Default globs, ordered most- to least-specific.  ISLES'24 ships the final
# infarct mask under derivatives/; the exact token has varied between the
# Zenodo release and the challenge tarball, so try several.
MASK_PATTERNS = [
    "**/derivatives/**/*lesion*msk*.nii*",
    "**/derivatives/**/*infarct*.nii*",
    "**/*lesion*msk*.nii*",
    "**/*_msk.nii*",
    "**/*mask*.nii*",
]
SUBJECT_RE = re.compile(r"(sub-[A-Za-z0-9]+)")


def find_masks(root: pathlib.Path, pattern: str | None):
    pats = [pattern] if pattern else MASK_PATTERNS
    for pat in pats:
        hits = sorted(root.glob(pat))
        if hits:
            return hits, pat
    return [], None


def lesion_volume_ml(path: pathlib.Path) -> float:
    """Volume of the binary mask in mL, from voxel spacing in the NIfTI header."""
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    vox_mm3 = float(np.prod(img.header.get_zooms()[:3]))
    return float((data > 0).sum() * vox_mm3 / 1000.0)


def lesion_count(path: pathlib.Path) -> int:
    """Number of connected components -- ISLES'24's fourth metric needs this,
    and a third of the cohort is scattered multifocal disease."""
    import nibabel as nib
    from scipy import ndimage

    data = np.asanyarray(nib.load(str(path)).dataobj) > 0
    return int(ndimage.label(data)[1])


def subject_of(path: pathlib.Path) -> str:
    m = SUBJECT_RE.search(str(path))
    return m.group(1) if m else path.stem


def collect(root: pathlib.Path, pattern: str | None, with_counts: bool):
    masks, used = find_masks(root, pattern)
    if not masks:
        raise SystemExit(
            f"No lesion masks found under {root}.\n"
            f"Tried: {pattern or MASK_PATTERNS}\n"
            "Pass the right glob with --mask-pattern after inspecting the tree."
        )
    print(f"  matched {len(masks)} masks with pattern {used!r}")

    rows = []
    for i, m in enumerate(masks, 1):
        rec = {"subject": subject_of(m), "path": str(m),
               "volume_ml": lesion_volume_ml(m)}
        if with_counts:
            rec["n_lesions"] = lesion_count(m)
        rows.append(rec)
        if i % 25 == 0 or i == len(masks):
            print(f"    {i}/{len(masks)}")
    return rows


def describe(vols: np.ndarray) -> dict:
    q = np.percentile(vols, [5, 25, 50, 75, 95])
    return {
        "n": int(vols.size),
        "mean": float(vols.mean()), "sd": float(vols.std(ddof=1)),
        "min": float(vols.min()), "max": float(vols.max()),
        "p5": float(q[0]), "q1": float(q[1]), "median": float(q[2]),
        "q3": float(q[3]), "p95": float(q[4]),
        "n_zero": int((vols <= 0.01).sum()),
    }


def tau_table(vols: np.ndarray, alpha_missed: float, alpha_futile: float,
              delta: float, band_ml: float) -> list[dict]:
    """Class balance, near-boundary mass, and certifiability at each tau."""
    floor_m = min_n_to_certify(alpha_missed, delta)
    floor_f = min_n_to_certify(alpha_futile, delta)
    n = vols.size

    rows = []
    for tau in TAUS:
        above = int((vols >= tau).sum())
        below = n - above
        near = int((np.abs(vols - tau) <= band_ml).sum())
        maj = max(above, below) / n
        rows.append({
            "tau_ml": float(tau),
            "n_below": below, "n_above": above,
            "prevalence_above": above / n,
            "majority_baseline_acc": float(maj),
            f"n_within_{band_ml:g}ml": near,
            "floor_missed": floor_m, "floor_futile": floor_f,
            "can_certify_missed": bool(below >= floor_m),
            "can_certify_futile": bool(above >= floor_f),
            "certifiable": bool(below >= floor_m and above >= floor_f),
        })
    return rows


def make_figure(vols, rows, path, band_ml):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].hist(vols, bins=40, color="#4C72B0", edgecolor="white")
    for tau, c in ((70, "crimson"), (110, "darkorange")):
        ax[0].axvline(tau, color=c, ls="--", lw=1.5, label=f"$\\tau$={tau} mL")
    ax[0].set_xlabel("final infarct volume (mL)"); ax[0].set_ylabel("cases")
    ax[0].set_title("Volume distribution"); ax[0].legend(fontsize=8)

    taus = [r["tau_ml"] for r in rows]
    ax[1].plot(taus, [r["n_above"] for r in rows], "o-",
               color="#4C72B0", label="cases $\\geq \\tau$")
    ax[1].axhline(rows[0]["floor_futile"], color="crimson", ls="--", lw=1.5,
                  label=f"LTT floor ({rows[0]['floor_futile']})")
    ax[1].set_xlabel("$\\tau$ (mL)"); ax[1].set_ylabel("cases above $\\tau$")
    ax[1].set_title("Certifiability of $R_{futile}$"); ax[1].legend(fontsize=8)

    ax[2].plot(taus, [r["majority_baseline_acc"] for r in rows], "o-",
               color="#C44E52")
    ax[2].axhline(0.5, color="grey", ls=":", lw=1)
    ax[2].set_xlabel("$\\tau$ (mL)"); ax[2].set_ylabel("majority-class accuracy")
    ax[2].set_title("Trivial baseline\n(why plain accuracy is unreportable)")
    ax[2].set_ylim(0.4, 1.02)

    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--mask-pattern", default=None,
                    help="glob for the final infarct masks, relative to --root")
    ap.add_argument("--band-ml", type=float, default=20.0,
                    help="near-boundary band half-width")
    ap.add_argument("--alpha-missed", type=float, default=0.05)
    ap.add_argument("--alpha-futile", type=float, default=0.20)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--lesion-counts", action="store_true",
                    help="also compute connected-component counts (slower)")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"{root} does not exist")

    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    print(f"\nISLES'24 audit  root={root}")
    rows = collect(root, args.mask_pattern, args.lesion_counts)
    vols = np.array([r["volume_ml"] for r in rows])

    d = describe(vols)
    print(f"\nVolume distribution (n={d['n']})")
    print(f"  mean {d['mean']:.2f} +/- {d['sd']:.2f} mL   "
          f"median {d['median']:.2f}   range {d['min']:.2f}-{d['max']:.2f}")
    print(f"  IQR {d['q1']:.2f}-{d['q3']:.2f}   "
          f"cases with ~zero infarct: {d['n_zero']}")
    print("  descriptor reference: 33.16 +/- 48.67 mL, range <0.01-318.25")

    if args.lesion_counts:
        lc = np.array([r["n_lesions"] for r in rows])
        print(f"  lesion count  mean {lc.mean():.2f} +/- {lc.std(ddof=1):.2f}  "
              f"max {lc.max()}   (descriptor: 10.55 +/- 11.10, max 84)")

    trows = tau_table(vols, args.alpha_missed, args.alpha_futile,
                      args.delta, args.band_ml)
    print(f"\nThreshold table  (LTT floors: R_missed needs "
          f"{trows[0]['floor_missed']} below-tau cases, R_futile needs "
          f"{trows[0]['floor_futile']} above-tau cases)")
    print(f"  {'tau':>5} {'below':>6} {'above':>6} {'prev':>6} "
          f"{'majAcc':>7} {'near':>5}  certifiable")
    print("  " + "-" * 58)
    for r in trows:
        mark = "YES" if r["certifiable"] else (
            "no (futile)" if not r["can_certify_futile"] else "no (missed)")
        print(f"  {r['tau_ml']:>5.0f} {r['n_below']:>6d} {r['n_above']:>6d} "
              f"{r['prevalence_above']:>6.3f} {r['majority_baseline_acc']:>7.3f} "
              f"{r[f'n_within_{args.band_ml:g}ml']:>5d}  {mark}")

    ok = [r["tau_ml"] for r in trows if r["certifiable"]]
    print("\n  CERTIFIABLE tau values: " + (
        ", ".join(f"{t:.0f}" for t in ok) + " mL" if ok else
        "NONE -- report the diagnostic and the certification-limits table instead"))

    make_figure(vols, trows, OUT_FIG / "data_audit.png", args.band_ml)
    payload = {"root": str(root), "config": vars(args) | {"root": str(root)},
               "distribution": d, "tau_table": trows, "cases": rows}
    (OUT_TAB / "data_audit.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {OUT_FIG/'data_audit.png'}")
    print(f"wrote {OUT_TAB/'data_audit.json'}")


if __name__ == "__main__":
    main()
