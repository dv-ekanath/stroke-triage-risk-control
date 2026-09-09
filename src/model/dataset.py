"""
Review 1, item 1.6 -- data pipeline for the baseline segmentation model.

Input channels, all verified to share one per-subject grid already (checked
on real data: raw_data/<sub>/ses-01/*_ncct.nii.gz defines the grid; every
"space-ncct" derivatives file for that subject has the identical shape and
zooms -- no extra registration step needed):

    NCCT, CTA, CBF, CBV, MTT, Tmax   (6 channels)

Target: derivatives/<sub>/ses-02/*_space-ncct_lesion-msk.nii.gz (binary).

Raw 4D CTP is deliberately NOT used as a channel -- its 4th-dimension frame
count varies across subjects (44-60 in a 6-subject spot check), which would
make it awkward as a fixed-channel model input, and v3 section 5.1 already
specifies the derived perfusion maps in its place.

Preprocessing follows v3's Data Preparation step: isotropic 1mm^3 resample,
z-score intensity normalisation. Splits are 5-fold, stratified by infarct-
volume tercile using the real per-subject volumes already computed by
src/data/audit.py (outputs/tables/data_audit.json) -- no need to recompute
them from the masks a second time.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

MODALITIES = ("ncct", "cta", "cbf", "cbv", "mtt", "tmax")

# Physiologically-plausible clip ranges for the perfusion maps, applied
# BEFORE z-score normalisation. Real-data check found catastrophic outliers
# in MTT specifically -- e.g. sub-stroke0001 ranges -54112 to 48.88 (plus 2
# literal non-finite voxels), sub-stroke0005 down to -20880 -- a known CT
# perfusion artifact (MTT = CBV/CBF blows up wherever CBF is near zero, in
# background/noise voxels). CBF and CBV have milder versions of the same
# thing. Left unclipped, z-score normalisation turns these into enormous
# |z| values that overflow float16 under AMP on the very first conv layer --
# this was the direct cause of ~29% of training batches producing a NaN loss
# in the first real training run. NCCT/CTA are already in the standard
# Hounsfield range and were not implicated, so they're left unclipped.
PERFUSION_CLIP = {
    "cbf": (0.0, 200.0),    # mL/100g/min
    "cbv": (0.0, 15.0),     # mL/100g
    "mtt": (0.0, 30.0),     # seconds
    "tmax": (0.0, 30.0),    # seconds -- also floors the -30 background-fill
                            # value seen in every spot-checked subject to 0
}

_INTEGRITY_CACHE = pathlib.Path("outputs/tables/dataset_integrity_cache.json")


def _find(root: pathlib.Path, sub: str, pattern: str) -> pathlib.Path | None:
    hits = sorted(root.glob(pattern.format(sub=sub)))
    return hits[0] if hits else None


def _file_fingerprint(path: pathlib.Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def _is_readable(path: pathlib.Path) -> bool:
    """True iff nibabel can actually decode the file's voxel data, not just
    open it. Real-data check found a file whose bytes exist and whose header
    parses, but whose gzip stream is truncated mid-file (the archive's own
    manifest confirms the defect is in the source release, not extraction:
    sub-stroke0043's cbf map is 18.5KB on disk where every neighbouring
    subject's is ~7.5MB). Existence alone is not enough to trust a file."""
    import nibabel as nib
    try:
        img = nib.load(str(path))
        np.asanyarray(img.dataobj)  # forces full decode, not just header parse
        return True
    except Exception:
        return False


def _verify_cached(paths: dict[str, pathlib.Path]) -> list[str]:
    """Returns the list of dict keys whose file failed integrity verification.
    Verified-good files are cached by (size, mtime) so repeated pipeline
    startups (many folds, many training runs) don't re-decode every file
    every time -- the initial full scan costs real time (~1-2 min over the
    whole 149-subject cohort)."""
    cache = {}
    if _INTEGRITY_CACHE.exists():
        try:
            cache = json.loads(_INTEGRITY_CACHE.read_text())
        except (OSError, json.JSONDecodeError):
            cache = {}

    bad_keys = []
    dirty = False
    for key, p in paths.items():
        fp = _file_fingerprint(p)
        cache_key = str(p)
        cached = cache.get(cache_key)
        if cached is not None and cached.get("fingerprint") == fp:
            ok = cached["ok"]
        else:
            ok = _is_readable(p)
            cache[cache_key] = {"fingerprint": fp, "ok": ok}
            dirty = True
        if not ok:
            bad_keys.append(key)

    if dirty:
        _INTEGRITY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _INTEGRITY_CACHE.write_text(json.dumps(cache, indent=2))
    return bad_keys


def list_cases(root: pathlib.Path, verify: bool = True) -> dict[str, dict]:
    """{subject: {"ncct": path, "cta": path, ..., "tmax": path, "label": path,
    "subject": id}} for every subject with all 7 files present AND readable.
    Skips (and does not raise on) incomplete or corrupt subjects -- report
    the count, don't crash a training run partway through over it.

    Each modality kept under its own dict key rather than pre-stacked into a
    single "image" list: real-data check found some subjects' per-modality
    NIfTI affines differ by ~1e-8 even though they're all nominally in
    "space-ncct" (independent resampling/writing per file, not a real
    registration difference). MONAI's list-based multi-file LoadImaged
    requires byte-identical affines across the list and throws on that tiny
    a mismatch. Loading each modality separately and resampling to a common
    grid per-key (build_transforms below) sidesteps it entirely -- and is
    the standard MONAI pattern for multi-modal data that isn't guaranteed
    to share a header exactly.

    verify=False skips the integrity decode (existence-only) for fast dev
    iteration; leave it True for anything that will actually train.
    """
    root = pathlib.Path(root)
    deriv, raw = root / "derivatives", root / "raw_data"

    cases = {}
    skipped_missing = []
    skipped_corrupt = []
    for sdir in sorted(deriv.glob("sub-stroke*")):
        sub = sdir.name
        paths = {
            "ncct": _find(raw, sub, "{sub}/ses-01/*_ncct.nii*"),
            "cta": _find(deriv, sub, "{sub}/ses-01/*_space-ncct_cta.nii*"),
            "cbf": _find(deriv, sub, "{sub}/ses-01/perfusion-maps/*_cbf.nii*"),
            "cbv": _find(deriv, sub, "{sub}/ses-01/perfusion-maps/*_cbv.nii*"),
            "mtt": _find(deriv, sub, "{sub}/ses-01/perfusion-maps/*_mtt.nii*"),
            "tmax": _find(deriv, sub, "{sub}/ses-01/perfusion-maps/*_tmax.nii*"),
            "label": _find(deriv, sub, "{sub}/ses-02/*_lesion-msk.nii*"),
        }
        if any(v is None for v in paths.values()):
            skipped_missing.append(sub)
            continue

        if verify:
            bad = _verify_cached(paths)
            if bad:
                skipped_corrupt.append((sub, bad))
                continue

        cases[sub] = {k: str(v) for k, v in paths.items()} | {"subject": sub}

    if skipped_missing:
        print(f"  [dataset] skipped {len(skipped_missing)} incomplete "
              f"subject(s): {skipped_missing}")
    if skipped_corrupt:
        print(f"  [dataset] skipped {len(skipped_corrupt)} subject(s) with an "
              f"unreadable file:")
        for sub, keys in skipped_corrupt:
            print(f"    {sub}: {keys}")
    return cases


def load_volumes(audit_json: pathlib.Path) -> dict[str, float]:
    """Per-subject infarct volume (mL), from the already-computed data audit
    (src/data/audit.py), keyed by subject id."""
    payload = json.loads(pathlib.Path(audit_json).read_text())
    out = {}
    for row in payload["cases"]:
        out[row["subject"]] = float(row["volume_ml"])
    return out


def make_folds(subjects: list[str], volumes: dict[str, float],
                n_folds: int = 5, seed: int = 0) -> list[tuple[list[str], list[str]]]:
    """5-fold split, stratified by infarct-volume tercile (v3 section 7)."""
    from sklearn.model_selection import StratifiedKFold

    subs = [s for s in subjects if s in volumes]
    missing = [s for s in subjects if s not in volumes]
    if missing:
        print(f"  [dataset] {len(missing)} subject(s) have no audited volume, "
              f"excluded from stratified folds: {missing}")

    vols = np.array([volumes[s] for s in subs])
    tercile_edges = np.quantile(vols, [1 / 3, 2 / 3])
    tercile = np.digitize(vols, tercile_edges)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    subs_arr = np.array(subs)
    for train_idx, val_idx in skf.split(subs_arr, tercile):
        folds.append((subs_arr[train_idx].tolist(), subs_arr[val_idx].tolist()))
    return folds


def _sanitize_nonfinite(x):
    """A literal Inf/NaN voxel was observed in one perfusion map (see
    PERFUSION_CLIP comment). Cleaned up before Spacingd's bilinear
    interpolation can spread it into neighbouring voxels.

    Module-level, not a closure: num_workers>0 on Windows spawns worker
    processes that pickle the whole transform pipeline, and Python cannot
    pickle a function defined inside another function."""
    import torch
    if isinstance(x, torch.Tensor):
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def build_transforms(patch_size: tuple[int, int, int], train: bool):
    from monai.transforms import (
        Compose, LoadImaged, Orientationd, Spacingd, NormalizeIntensityd,
        CropForegroundd, SpatialPadd, RandCropByPosNegLabeld,
        RandFlipd, RandRotate90d, EnsureTyped, ConcatItemsd, DeleteItemsd,
        Lambdad, ScaleIntensityRanged,
    )

    mod_keys = list(MODALITIES)
    all_keys = mod_keys + ["label"]

    common = [
        # each modality loaded and resampled under its OWN key first -- see
        # list_cases' docstring for why (affines aren't guaranteed identical
        # across a subject's files, so they can't be stacked before this).
        LoadImaged(keys=all_keys, ensure_channel_first=True, image_only=True),
        Lambdad(keys=mod_keys, func=_sanitize_nonfinite),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0),
                  mode=(*["bilinear"] * len(mod_keys), "nearest")),
        *[ScaleIntensityRanged(keys=[k], a_min=lo, a_max=hi, b_min=lo, b_max=hi,
                                clip=True)
          for k, (lo, hi) in PERFUSION_CLIP.items()],
        NormalizeIntensityd(keys=mod_keys, nonzero=True, channel_wise=True),
        CropForegroundd(keys=all_keys, source_key=mod_keys[0], allow_smaller=True),
        SpatialPadd(keys=all_keys, spatial_size=patch_size),
        ConcatItemsd(keys=mod_keys, name="image", dim=0),
        DeleteItemsd(keys=mod_keys),
    ]
    if train:
        common += [
            RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label",
                spatial_size=patch_size, pos=2, neg=1, num_samples=1,
                image_key="image", image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
        ]
    common.append(EnsureTyped(keys=["image", "label"]))
    return Compose(common)


def make_datasets(root, audit_json, fold: int, n_folds: int, patch_size, seed: int = 0,
                   cache_dir: pathlib.Path | None = None):
    """Returns (train_ds, val_ds, val_subjects) for one fold.

    With cache_dir set, uses MONAI's PersistentDataset: it auto-detects the
    random transforms in build_transforms' pipeline (RandCropByPosNegLabeld
    etc.) and caches everything BEFORE them to disk -- the load / sanitize /
    resample / clip / normalize chain, which is what made the first real
    training run CPU-bound (506s for one epoch at num_workers=0, all of it
    spent re-decompressing and re-resampling the same 118 subjects). Only
    the first epoch pays that cost; every epoch after reads the cached
    tensors and reapplies just the cheap random crop/flip/rotate.

    Caution: the cache is keyed by input file identity, not by the transform
    definitions -- changing build_transforms (e.g. PERFUSION_CLIP ranges) and
    reusing an old cache_dir will silently serve stale preprocessing. Clear
    the cache directory after any change to build_transforms.
    """
    from monai.data import Dataset, PersistentDataset

    cases = list_cases(pathlib.Path(root))
    volumes = load_volumes(audit_json)
    folds = make_folds(sorted(cases.keys()), volumes, n_folds=n_folds, seed=seed)
    if not (0 <= fold < len(folds)):
        raise ValueError(f"fold {fold} out of range [0, {len(folds)})")
    train_ids, val_ids = folds[fold]

    train_items = [cases[s] for s in train_ids if s in cases]
    val_items = [cases[s] for s in val_ids if s in cases]

    if cache_dir is not None:
        cache_dir = pathlib.Path(cache_dir)
        (cache_dir / "train").mkdir(parents=True, exist_ok=True)
        (cache_dir / "val").mkdir(parents=True, exist_ok=True)
        train_ds = PersistentDataset(train_items, transform=build_transforms(patch_size, train=True),
                                      cache_dir=str(cache_dir / "train"))
        val_ds = PersistentDataset(val_items, transform=build_transforms(patch_size, train=False),
                                    cache_dir=str(cache_dir / "val"))
    else:
        train_ds = Dataset(train_items, transform=build_transforms(patch_size, train=True))
        val_ds = Dataset(val_items, transform=build_transforms(patch_size, train=False))
    return train_ds, val_ds, val_ids
