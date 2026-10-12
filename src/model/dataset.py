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

# Tighter, clinically-motivated clips (2026-10-11), tried after reading the
# ISLES'24 challenge winner's own account of what won it (arXiv:2505.18424,
# "How We Won the ISLES'24 Challenge by Preprocessing"): they report ~10
# validation-Dice points from custom, clinically-meaningful windowing
# starting from literature anchors (CBF<30%, CBV<2.0 mL/100g, MTT~12s,
# Tmax>6s), then empirically refined -- exact final numbers not published.
# PERFUSION_CLIP above was set only to kill extreme artifact outliers (a
# real NaN-loss bug, see the comment below), not to express a clinically
# meaningful core/penumbra window, which is the gap this closes partway.
# CBF and Tmax are left unchanged here: their literature anchors are
# RELATIVE/threshold quantities (percent of normal tissue; a hypoperfusion
# onset marker), not a simple absolute clip, and guessing wrong there could
# hurt rather than help. CBV and MTT are tightened toward the literature
# anchors above (not copied exactly -- the paper doesn't publish its final
# values either).
TIGHT_PERFUSION_CLIP = {
    "cbf": (0.0, 200.0),
    "cbv": (0.0, 3.0),
    "mtt": (0.0, 15.0),
    "tmax": (0.0, 30.0),
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


def _brain_mask(ncct: np.ndarray):
    """Simple HU-threshold + largest-connected-component brain mask.
    Duplicated from src/graph/perfusion_volumes.py:brain_mask (not
    imported -- src/model stays independent of src/graph for one small
    function, same reason src/model/labels.py duplicates its LVO logic).
    Already validated there at 98.6% plausible intracranial volume.
    Simpler than the ISLES'24 challenge winner's SynthStrip (a deep
    model, arXiv:2505.18424) -- real, but a weaker skull-strip than
    theirs, disclosed as such."""
    from scipy import ndimage
    soft = (ncct > 0) & (ncct < 100)
    soft = ndimage.binary_opening(soft, structure=np.ones((3, 3, 1)))
    lbl, n = ndimage.label(soft)
    if n == 0:
        return soft
    sizes = ndimage.sum(soft, lbl, range(1, n + 1))
    mask = lbl == (int(np.argmax(sizes)) + 1)
    for z in range(mask.shape[2]):
        mask[:, :, z] = ndimage.binary_fill_holes(mask[:, :, z])
    return mask


class _SkullStripd:
    """Zero out non-brain voxels across every modality channel, using a
    mask computed from the (already 1mm-resampled) NCCT channel. Mirrors
    the ISLES'24 challenge winner's step 1: skull-strip on NCCT, apply
    that mask to every co-registered scan (arXiv:2505.18424). Runs AFTER
    Spacingd (so the mask is computed at the same 1mm grid every channel
    is on) and BEFORE the perfusion clip / normalization steps."""

    def __init__(self, mod_keys, ncct_key: str = "ncct"):
        self.mod_keys = mod_keys
        self.ncct_key = ncct_key

    def __call__(self, data):
        import torch
        d = dict(data)
        ncct = np.asarray(d[self.ncct_key][0])
        mask = torch.as_tensor(_brain_mask(ncct))
        for k in self.mod_keys:
            d[k] = d[k] * mask[None]
        return d


def build_transforms(patch_size: tuple[int, int, int], train: bool,
                      skull_strip: bool = False, perfusion_clip: dict | None = None):
    from monai.transforms import (
        Compose, LoadImaged, Orientationd, Spacingd, NormalizeIntensityd,
        CropForegroundd, SpatialPadd, RandCropByPosNegLabeld,
        RandFlipd, RandRotate90d, EnsureTyped, ConcatItemsd, DeleteItemsd,
        Lambdad, ScaleIntensityRanged,
    )

    mod_keys = list(MODALITIES)
    all_keys = mod_keys + ["label"]
    clip = perfusion_clip if perfusion_clip is not None else PERFUSION_CLIP

    common = [
        # each modality loaded and resampled under its OWN key first -- see
        # list_cases' docstring for why (affines aren't guaranteed identical
        # across a subject's files, so they can't be stacked before this).
        LoadImaged(keys=all_keys, ensure_channel_first=True, image_only=True),
        Lambdad(keys=mod_keys, func=_sanitize_nonfinite),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0),
                  mode=(*["bilinear"] * len(mod_keys), "nearest")),
        *([_SkullStripd(mod_keys)] if skull_strip else []),
        *[ScaleIntensityRanged(keys=[k], a_min=lo, a_max=hi, b_min=lo, b_max=hi,
                                clip=True)
          for k, (lo, hi) in clip.items()],
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
            # RandCropByPosNegLabeld attaches foreground_start_coord/
            # foreground_end_coord metadata that's a Tensor for a normal
            # (foreground) crop but a plain numpy.ndarray when it falls back
            # to an all-background crop (a subject with zero lesion voxels,
            # the "Num foregrounds 0" warning) -- EnsureTyped below never
            # covered these two keys, so a batch pairing one of each type
            # crashes collation. Found 2026-09-17 (Phase 8 fold-2 prep):
            # reproducible, not a worker race or a per-item transform bug
            # (confirmed via an isolated single-item check across 3 passes).
            # Unused downstream -- just drop them instead of typing them.
            DeleteItemsd(keys=["foreground_start_coord", "foreground_end_coord"]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
        ]
    common.append(EnsureTyped(keys=["image", "label"]))
    return Compose(common)


def attach_cow_msk(root: pathlib.Path, cases: dict[str, dict]) -> dict[str, dict]:
    """Adds a "cow_msk" path to each case dict that has one (ses-01 CoW
    segmentation, same file src/graph/features.py already reads for Phase 1).
    NOT required for a subject to appear in `cases` -- only Phase 4+'s GNN
    branch needs it; segmentation and the H2/H3 heads' loss are already
    masked out per-sample for subjects without a usable graph (see
    src/model/labels.py). Subjects missing it are simply left without the
    key -- callers that require it (build_transforms_gnn's dataset builder)
    filter those out explicitly, so the drop is visible and counted rather
    than silently defaulting to a zero mask."""
    root = pathlib.Path(root)
    deriv = root / "derivatives"
    out = {}
    n_missing = 0
    for sub, item in cases.items():
        hits = sorted((deriv / sub).glob("ses-01/*cow-msk.nii*"))
        item = dict(item)
        if hits:
            item["cow_msk"] = str(hits[0])
        else:
            n_missing += 1
        out[sub] = item
    if n_missing:
        print(f"  [dataset] {n_missing} subject(s) have no cow-msk file -- "
              f"excluded from the GNN branch only, still usable elsewhere")
    return out


class ExtractVesselCoordsd:
    """Turns a resized CoW label map into per-node (voxel-grid) coordinates
    for the GNN branch: `node_coords` [n_nodes, 3] (row/col/depth index into
    the SAME grid the whole-brain image tensor uses -- so
    src/model/gnn_head.py's grid_sample call can use it directly with no
    further rescaling) and `node_present` [n_nodes] (0/1).

    A plain callable, not a MapTransform subclass -- Compose only needs
    __call__, and this keeps it trivially picklable for Windows'
    spawn-based DataLoader workers (see _sanitize_nonfinite's docstring for
    why module-level/simple-attribute objects matter here).

    Coordinates are computed fresh from the already-transformed tensor at
    every call rather than precomputed once and cached -- cheap (12 argwhere
    + mean calls over a small label volume) and guarantees they can never
    drift out of sync with whatever Resized/CropForegroundd actually did for
    that subject.

    Also emits `midline_voxel_x` (scalar, or NaN if no L/R pair is present):
    Phase 6's laterality consistency loss needs the midline's location in
    THIS SAME wb-grid voxel-index space (axis 0, the R/L axis after
    Orientationd's RAS canonicalisation) -- computed the same way as
    src/graph/features.py:midline_x (median of L/R vessel-pair midpoints),
    but directly in voxel indices here instead of world mm, since that's
    the space the whole-brain segmentation prediction lives in too. No
    affine/world-coordinate round-trip needed -- see train_graph_seg's
    module docstring for why that matters (the patch branch's random crop
    makes world<->voxel tracking fragile; the wb branch's fixed, unaugmented
    grid does not have that problem).
    """

    PAIRS = ("ICA", "MCA", "ACA", "PCA", "Pcom")  # same as src/graph/features.py

    def __init__(self, key: str, node_ids: list[str], topcow_labels: list[int],
                 out_prefix: str = ""):
        """out_prefix: instantiate this TWICE in one pipeline (once on the
        wb-grid cow_msk, once on a native-resolution copy -- see Phase 8's
        fix in build_transforms_gnn / CLAUDE.md KNOWN ISSUES #2) by giving
        the second instance a prefix, e.g. "native_", so its outputs land
        in `native_node_coords` / `native_midline_voxel_x` instead of
        colliding with the wb-grid ones."""
        self.key = key
        self.node_ids = node_ids
        self.topcow_labels = topcow_labels
        self.out_prefix = out_prefix

    def __call__(self, data):
        import numpy as np
        d = dict(data)
        cow = d[self.key]
        arr = cow[0]  # drop channel dim -> [S0, S1, S2]
        arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
        arr = np.rint(arr)  # Resized(mode="nearest") shouldn't blur labels, but guard anyway

        n = len(self.topcow_labels)
        coords = np.zeros((n, 3), dtype=np.float32)
        present = np.zeros((n,), dtype=np.float32)
        for i, label in enumerate(self.topcow_labels):
            idx = np.argwhere(arr == label)
            if idx.size:
                coords[i] = idx.mean(axis=0)
                present[i] = 1.0
        d[f"{self.out_prefix}node_coords"] = coords
        d[f"{self.out_prefix}node_present"] = present

        node_idx = {v: i for i, v in enumerate(self.node_ids)}
        mids = []
        for vessel in self.PAIRS:
            li, ri = node_idx.get(f"L-{vessel}"), node_idx.get(f"R-{vessel}")
            if li is None or ri is None or not (present[li] and present[ri]):
                continue
            mids.append((coords[li, 0] + coords[ri, 0]) / 2.0)
        d[f"{self.out_prefix}midline_voxel_x"] = np.float32(np.median(mids)) if mids else np.float32(np.nan)
        del d[self.key]
        return d


def build_transforms_gnn(patch_size: tuple[int, int, int], wb_size: tuple[int, int, int],
                          train: bool, topcow_labels: list[int], node_ids: list[str] | None = None,
                          native_midline: bool = False):
    """Same segmentation patch pipeline as build_transforms, PLUS a second,
    unaugmented "whole-brain" view for the GNN branch (Phase 4/5):

      image_wb   the full head (post CropForegroundd, i.e. background
                 already stripped, same crop box as the patch branch since
                 it's forked from the identical tensor) squashed to a fixed
                 size -- so the Circle of Willis is always in-frame, unlike
                 the patch branch's lesion-centered random crop, which the
                 real Phase 3 run showed can miss it entirely.
      node_coords / node_present   from `cow_msk`, resized in lockstep with
                 image_wb (same Resized call, same interpolation grid) so a
                 coordinate sampled from image_wb's feature map lands on the
                 right vessel.

    image_wb/node_coords deliberately see NO random crop/flip/rotate/any
    SPATIAL augmentation -- see ExtractVesselCoordsd's docstring; moving
    voxels would require transforming the coordinates too, which isn't
    implemented, so it's left out rather than silently wrong.

    INTENSITY-only augmentation (train only) is applied to image_wb, since
    it doesn't move anything and so can't desync node_coords: gaussian
    noise, contrast, and small scale/shift of intensity. Motivation: the
    real B1 run (2026-09-15) found the LVO/collateral heads converging to a
    near-constant prediction (collateral_acc identical at all 10 validation
    checkpoints despite falling training loss) -- classic small-n
    overfitting-to-a-shortcut behaviour with 98 training subjects across a
    4-way, badly-imbalanced label. This doesn't add subjects, but it forces
    the model to not key off exact noise/contrast patterns specific to
    individual scans, which is the cheapest available lever without new
    data (see PROJECT_PLAN.md's discussion of why a compatible new dataset
    is not reachable in this project's timeline).

    native_midline=True adds `native_midline_voxel_x` (+ native_node_coords/
    native_node_present, unused by Phase 8's fix but computed for free):
    the SAME midline computation as `midline_voxel_x`, but on the "image"/
    "label" grid's OWN native resolution (post CropForegroundd+SpatialPadd,
    no Resized) instead of the wb branch's downscaled grid. Phase 8 fix
    (CLAUDE.md KNOWN ISSUES #2): the Rule Evaluator was scoring laterality
    on a 64x64x48 view only B2 was ever trained to segment, which likely
    explains B0/B1's high "predicted nothing" rate more than the
    consistency loss did. Off by default (extra cow_msk copy carried
    through SpatialPadd) -- only the evaluation scripts need it, not
    B0/B1/B2 training itself.
    """
    from monai.transforms import (
        Compose, LoadImaged, Orientationd, Spacingd, NormalizeIntensityd,
        CropForegroundd, SpatialPadd, RandCropByPosNegLabeld,
        RandFlipd, RandRotate90d, EnsureTyped, ConcatItemsd, DeleteItemsd,
        CopyItemsd, Resized, Lambdad, ScaleIntensityRanged,
        RandGaussianNoised, RandAdjustContrastd, RandScaleIntensityd, RandShiftIntensityd,
    )

    if node_ids is None:
        from src.model.gnn_head import node_order
        node_ids = node_order()

    mod_keys = list(MODALITIES)
    wb_keys = [f"{k}_wb" for k in mod_keys]
    all_keys = mod_keys + ["label", "cow_msk"]

    common = [
        LoadImaged(keys=all_keys, ensure_channel_first=True, image_only=True),
        Lambdad(keys=mod_keys, func=_sanitize_nonfinite),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0),
                  mode=(*["bilinear"] * len(mod_keys), "nearest", "nearest")),
        *[ScaleIntensityRanged(keys=[k], a_min=lo, a_max=hi, b_min=lo, b_max=hi,
                                clip=True)
          for k, (lo, hi) in PERFUSION_CLIP.items()],
        NormalizeIntensityd(keys=mod_keys, nonzero=True, channel_wise=True),
        CropForegroundd(keys=all_keys, source_key=mod_keys[0], allow_smaller=True),

        # Phase 8 fix: a SEPARATE cow_msk copy that stays at native
        # resolution all the way through SpatialPadd, so its midline lines
        # up with sliding-window inference's own output grid -- see this
        # function's docstring ("native_midline").
        *([CopyItemsd(keys=["cow_msk"], names=["cow_msk_native"])] if native_midline else []),

        # fork the whole-brain branch BEFORE the patch branch pads/crops
        CopyItemsd(keys=mod_keys, names=wb_keys),
        ConcatItemsd(keys=wb_keys, name="image_wb"),
        DeleteItemsd(keys=wb_keys),
        Resized(keys=["image_wb", "cow_msk"], spatial_size=wb_size,
                mode=("trilinear", "nearest"), align_corners=(False, None)),
        ExtractVesselCoordsd(key="cow_msk", node_ids=node_ids, topcow_labels=topcow_labels),
        # intensity-only, train-only augmentation for the wb branch -- see
        # this function's docstring for why (cheap lever against the small-n
        # overfitting-to-a-shortcut pattern seen in the real B1 run, without
        # touching geometry/node_coords).
        *([
            RandGaussianNoised(keys=["image_wb"], prob=0.3, mean=0.0, std=0.05),
            RandAdjustContrastd(keys=["image_wb"], prob=0.3, gamma=(0.8, 1.25)),
            RandScaleIntensityd(keys=["image_wb"], prob=0.3, factors=0.1),
            RandShiftIntensityd(keys=["image_wb"], prob=0.3, offsets=0.1),
        ] if train else []),

        # patch branch continues as build_transforms
        SpatialPadd(keys=mod_keys + ["label"] + (["cow_msk_native"] if native_midline else []),
                    spatial_size=patch_size),
        *([ExtractVesselCoordsd(key="cow_msk_native", node_ids=node_ids,
                                 topcow_labels=topcow_labels, out_prefix="native_")]
          if native_midline else []),
        ConcatItemsd(keys=mod_keys, name="image"),
        DeleteItemsd(keys=mod_keys),
    ]
    if train:
        common += [
            RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label",
                spatial_size=patch_size, pos=2, neg=1, num_samples=1,
                image_key="image", image_threshold=0,
            ),
            # see the matching comment in build_transforms -- same fix,
            # same root cause (Phase 8 fold-2 prep, 2026-09-17).
            DeleteItemsd(keys=["foreground_start_coord", "foreground_end_coord"]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
        ]
    common.append(EnsureTyped(keys=["image", "label", "image_wb"]))
    return Compose(common)


def make_datasets(root, audit_json, fold: int, n_folds: int, patch_size, seed: int = 0,
                   cache_dir: pathlib.Path | None = None,
                   skull_strip: bool = False, perfusion_clip: dict | None = None):
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
        train_ds = PersistentDataset(train_items,
                                      transform=build_transforms(patch_size, train=True,
                                                                  skull_strip=skull_strip, perfusion_clip=perfusion_clip),
                                      cache_dir=str(cache_dir / "train"))
        val_ds = PersistentDataset(val_items,
                                    transform=build_transforms(patch_size, train=False,
                                                                skull_strip=skull_strip, perfusion_clip=perfusion_clip),
                                    cache_dir=str(cache_dir / "val"))
    else:
        train_ds = Dataset(train_items, transform=build_transforms(patch_size, train=True,
                                                                     skull_strip=skull_strip, perfusion_clip=perfusion_clip))
        val_ds = Dataset(val_items, transform=build_transforms(patch_size, train=False,
                                                                 skull_strip=skull_strip, perfusion_clip=perfusion_clip))
    return train_ds, val_ds, val_ids
