#!/usr/bin/env python3
"""Shared loading / aggregation / styling utilities for uncertainty figures.

All paper figures and uncertainty tables import from this module so that data
selection, seed pooling, NaN handling, model naming and plot styling are
identical everywhere.

Data layout (the tree the user pointed at, which actually contains the
``uncertainty`` block -- NOTE: ``tests_uncertainty_runs`` does NOT):

    tests_uncertainty/<group>/<dataset>/<family_dir>/<config>/*.json

where
    group       in {BTN, ALS, baseline}      (ALS has NO uncertainty block)
    dataset     in DATASET_ORDER
    family_dir  e.g. "LMPO2_L3_d18" (BTN) or "SparseGP/<config>" (baseline)

Each run json has an ``uncertainty`` dict with the metrics described in
UNCERTAINTY_METRICS.md plus two stored curves:
    reliability_curve  : {expected:[...], observed:[...]}
    sparsification_curve : {fractions:[...], rmse_by_unc:[...], rmse_by_oracle:[...]}
"""

import os
import re
import glob
import json
import math
import statistics


# ---------------------------------------------------------------------------
# Robust JSON loading
# ---------------------------------------------------------------------------
# Some run JSONs on disk contain a corruption where a stray "--" was spliced
# into the middle of a numeric literal, e.g.
#     "test_loss": 355.657--501,     "elbo_relative": 806423441.--7,
#     "val_quality": 0.4--3385691534, "h_nodes": 2691.--826591,
# This makes the file unparseable. We DO NOT touch the files on disk (the
# experiment records must stay pristine); instead we repair the text in memory
# and parse that. The affected fields are auxiliary and unused by the tables,
# but the repair also rescues any run whose primary metrics sit in the same
# file, so those seeds are no longer silently dropped.

# Matches a number where a run of '-' was injected after the decimal point or
# between digits, e.g. 355.657--501 / 806423441.--7 / 0.4--3385691534.
_CORRUPT_NUM_RE = re.compile(r"(?<=\d)-{2,}(?=\d)|(?<=\.)-{2,}(?=\d)")


def load_run_json(filepath):
    """Parse a run JSON, transparently repairing the known '--' number
    corruption in memory. Returns the parsed dict, or None if it still can't
    be parsed. The file on disk is never modified."""
    try:
        with open(filepath) as fh:
            text = fh.read()
    except OSError:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip stray '--' splices from inside numeric literals and retry.
    repaired = _CORRUPT_NUM_RE.sub("", text)
    if repaired != text:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None
    return None

# ============================================================================
# CONFIG  (edit everything here)
# ============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)

# The tree that contains the `uncertainty` block.
ROOT = os.path.join(_REPO_ROOT, "tests_uncertainty")

IMAGES_DIR = os.path.join(_SCRIPT_DIR, "images", "uncertainty")
TABLES_DIR = os.path.join(_SCRIPT_DIR, "tables")
CAPTIONS_DIR = os.path.join(_SCRIPT_DIR, "captions")

# Per-dataset std(y), precomputed by compute_target_std.py. The pipeline
# standardizes X but NOT y, so predictive std / sharpness live in raw target
# units and are not comparable across datasets; dividing by std(y) makes them
# a data-intrinsic, comparable scale (1.0 == as wide as the marginal spread).
_TARGET_STD_CSV = os.path.join(_SCRIPT_DIR, "target_std.csv")


def _load_target_std():
    """dataset -> std(y), read from target_std.csv. {} if the file is absent."""
    import csv
    if not os.path.isfile(_TARGET_STD_CSV):
        return {}
    out = {}
    with open(_TARGET_STD_CSV, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["dataset"]] = float(row["y_std"])
            except (KeyError, ValueError, TypeError):
                continue
    return out


_TARGET_STD = _load_target_std()


def target_std(dataset):
    """std(y) for a dataset, or None if unavailable/non-positive."""
    s = _TARGET_STD.get(dataset)
    return s if (s and s > 0 and math.isfinite(s)) else None


def caption(name, trailing_percent=False):
    r"""Return a ``\caption{...}`` block whose body is read verbatim from
    ``captions/<name>_caption.tex``.

    Captions live in external files so the wording can be edited without
    touching the generation code. ``trailing_percent`` appends ``%`` right
    after the closing brace (used by some tables to suppress the space that a
    line break would otherwise introduce)."""
    path = os.path.join(CAPTIONS_DIR, name + "_caption.tex")
    with open(path) as fh:
        body = fh.read()
    return r"\caption{" + body + "}" + ("%" if trailing_percent else "")

# --- Datasets ----------------------------------------------------------------
DATASET_ORDER = [
    "abalone", "ai4i", "appliances", "bike", "concrete",
    "energy_efficiency", "obesity", "realstate", "seoulBike", "student_perf",
]
DATASET_DISPLAY = {
    "abalone": "AB", "ai4i": "AI", "appliances": "AP",
    "bike": "BK", "concrete": "CO", "energy_efficiency": "EE",
    "obesity": "OB", "realstate": "RS", "seoulBike": "SB",
    "student_perf": "SP",
}

# --- BTN families ------------------------------------------------------------
TN_FAMILY_ORDER = ["CPD", "LMPO2", "MPO2", "BTT", "TR"]

# --- Baselines (row key -> source directory names pooled into that row) ------
# BayesianWideDeep ("BWD") excluded (unstable; matches make_r2_table.py).
BASELINE_ORDER = ["GP", "BDE", "HSBNN", "BASS"]
BASELINE_DISPLAY = {
    "GP": "GP", "BDE": "BDE", "HSBNN": "HSBNN", "BASS": "BASS",
}
BASELINE_SOURCES = {
    "GP": ["ExactGP", "SparseGP"],
    "BDE": ["BdeMile"],
    "HSBNN": ["HorseshoeBNN"],
    "BASS": ["MvBayes"],
}

# The BTN entry is a single curated row per dataset: the BTN family that
# predicts best on the validation set (`val_quality`) is chosen automatically
# (see valbest_btn_family), and THAT family's uncertainty metrics are reported.
BTN_DISPLAY = "BTNR"

# --- Metric semantics --------------------------------------------------------
# direction: "min" = lower is better, "max" = higher is better,
#            "target" = closest to TARGET is best.
METRIC_INFO = {
    "unc_nll":              dict(label="NLL",                    direction="min"),
    "unc_crps":             dict(label="CRPS",                   direction="min"),
    "unc_ece":              dict(label="ECE",                    direction="min"),
    "unc_max_ce":           dict(label="Max CE",                 direction="min"),
    "unc_picp":             dict(label="PICP (95%)",             direction="target", target=0.95),
    "unc_mpiw":             dict(label="MPIW",                   direction="min"),
    "unc_coverage_error":   dict(label="Coverage error",        direction="min"),
    "unc_sharpness":        dict(label="Sharpness",             direction="min"),
    "unc_ause":             dict(label="AUSE",                   direction="min"),
    "unc_spearman_err":     dict(label="Spearman (unc vs err)", direction="max"),
    "unc_outlier_auroc_std":dict(label="Outlier AUROC (std)",   direction="max"),
    "unc_outlier_aupr_std": dict(label="Outlier AUPR (std)",    direction="max"),
    "unc_outlier_auroc_nll":dict(label="Outlier AUROC (NLL)",   direction="max"),
    "unc_outlier_aupr_nll": dict(label="Outlier AUPR (NLL)",    direction="max"),
}

# --- Styling -----------------------------------------------------------------
# One consistent color per model across every figure. BTN is the highlight.
# "BTN" is the single per-dataset val-quality-best BTN family row.
MODEL_ORDER = ["BTN"] + BASELINE_ORDER
COLORS = {
    "BTN":   "#d62728",   # red, the highlight (BTNR)
    "GP":    "#1f77b4",   # blue
    "BDE":   "#2ca02c",   # green
    "HSBNN": "#9467bd",   # purple
    "BASS":   "#ff7f0e",   # orange
}
MARKERS = {
    "BTN": "x",
    "GP": "s", "BDE": "^", "HSBNN": "o", "BASS": "v",
}
# z-order so BTN draws on top.
ZORDER = {"BTN": 5, "GP": 3, "BDE": 3, "HSBNN": 3, "BASS": 3}

# The single model we visually emphasise everywhere (drawn bigger/bolder/on top).
HIGHLIGHT = "BTN"


def is_btn_key(key):
    """True for the BTN row key."""
    return key == "BTN"


def model_order():
    """Full ordered model keys (BTN row then baselines)."""
    return ["BTN"] + BASELINE_ORDER

# --- Centralised plot dimensions ---------------------------------------------
# Every figure pulls its line widths / marker sizes from here, distinguishing
# the highlighted model from the rest. Edit once -> applies to all figures.
STYLE = {
    # line plots (reliability / sparsification curves)
    "line_lw":        {"hi": 0.7, "lo": 0.5},   # main curves
    "line_ms":        {"hi": 3.0, "lo": 2.5},   # markers on curves
    "oracle_lw":      0.9,                        # BTN oracle (sparsification)
    "ref_lw":         1.0,                        # reference diagonal / target line
    "band_alpha":     0.2,                       # +-std shaded band (reliability)
    "gap_alpha":      0.25,                       # AUSE gap shading (sparsification)
    # marker / scatter plots (scatter / rank dotplot / picp)
    "marker_ms":      {"hi": 9, "lo": 7},
    "edge_lw":        {"hi": 1.5, "lo": 1.2},    # marker edge (black ring on highlight)
    "errorbar_lw":    1.0,
    "capsize":        2.5,
    # rank heatmap highlight box
    "box_lw":         2.5,
}


def style_for(key, kind="line"):
    """Return per-model plot kwargs from STYLE, branching on highlight vs rest.

    kind="line"   -> dict(lw, ms, zorder)
    kind="marker" -> dict(ms, markeredgewidth, markeredgecolor, zorder)
    Always merge with COLORS[key] / MARKERS[key] at the call site.
    """
    hi = is_btn_key(key)
    sel = "hi" if hi else "lo"
    if kind == "line":
        return dict(lw=STYLE["line_lw"][sel], ms=STYLE["line_ms"][sel],
                    zorder=ZORDER[key])
    if kind == "marker":
        return dict(ms=STYLE["marker_ms"][sel],
                    markeredgewidth=STYLE["edge_lw"][sel],
                    markeredgecolor= COLORS[key],
                    zorder=ZORDER[key])
    raise ValueError(kind)


def display_name(key):
    if key == "BTN":
        return BTN_DISPLAY
    return BASELINE_DISPLAY.get(key, key)


# Prefix for BTN family display names (matches make_r2_table.py: "B-").
BTN_FAMILY_PREFIX = "B-"

# Data keys that should render under a different display name (keys stay as-is
# for result-directory lookup; only the paper label changes). "MPO2" -> "MPS".
FAMILY_DISPLAY_NAME = {
    "MPO2": "MPS",
}


def family_display(family):
    """Display name for a BTN family, e.g. 'LMPO2' -> 'B-LMPO2', 'MPO2' -> 'B-MPS'."""
    if not family:
        return family
    return BTN_FAMILY_PREFIX + FAMILY_DISPLAY_NAME.get(family, family)


# ============================================================================
# Matplotlib style
# ============================================================================

def apply_style():
    """Apply a clean, paper-ready matplotlib style. Call once per script."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,            # editable text in PDF
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
    })
    return plt


def finalize_grid(fig, *, supxlabel, supylabel, suptitle, handles, labels,
                  ncol=None, legend_pad=0.0):
    """Lay out a small-multiples grid with no trapped whitespace or overlap.

    Requires the figure to have been created with ``layout="constrained"``.
    Elements are stacked cleanly from top to bottom as
    ``suptitle -> panels -> supxlabel -> legend`` by putting the shared
    x-axis description on the legend via its title, which avoids the classic
    ``supxlabel``/figure-legend collision.

    Knobs:
      * ``ncol``       -- legend columns (default: one per entry, single row)
      * ``legend_pad`` -- extra vertical gap (figure fraction) below the panels
                          before the legend; increase to push the legend down.
    """
    fig.suptitle(suptitle, fontsize=12)
    fig.supylabel(supylabel)
    ncol = ncol or len(labels)
    leg = fig.legend(handles, labels, loc="outside lower center", ncol=ncol,
                     title=supxlabel, title_fontsize=10)
    # Render the legend title (the x-axis description) like a normal supxlabel.
    if leg.get_title() is not None:
        leg.get_title().set_fontsize(plt_labelsize())
    if legend_pad:
        eng = fig.get_layout_engine()
        try:
            eng.set(h_pad=eng.get()["h_pad"] + legend_pad)
        except Exception:
            pass
    return leg


def plt_labelsize():
    import matplotlib
    return matplotlib.rcParams.get("axes.labelsize", 10)


def savefig(fig, name, pad_inches=0.08):
    """Save a figure as PDF into IMAGES_DIR and report the path.

    A tiny padding (``pad_inches``) is added around the tight bounding box so
    that markers, tick labels and annotations near the right/top edges are not
    clipped. Lower ``pad_inches`` for a tighter crop.
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)
    path = os.path.join(IMAGES_DIR, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=pad_inches)
    print(f"  wrote {os.path.relpath(path, _REPO_ROOT)}")
    return path


# ============================================================================
# Output-naming helper
# ============================================================================
# There is a single BTN aggregation mode ("valbest": the per-dataset
# validation-quality-best BTN family), so figure/table files are always
# canonical and UNSUFFIXED. `variant_suffix` is retained (returning "") only so
# existing f-strings in the generators keep working unchanged.

def variant_suffix(variant=None):
    """Filename suffix (always empty: only one BTN aggregation mode exists)."""
    return ""


# ============================================================================
# Data loading
# ============================================================================

def _is_finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def _model_dirs(group, dataset, key):
    """Return the directories whose seed jsons belong to (group, dataset, key)."""
    gdir = os.path.join(ROOT, group, dataset)
    if not os.path.isdir(gdir):
        return []
    if group == "baseline":
        out = []
        for src in BASELINE_SOURCES[key]:
            d = os.path.join(gdir, src)
            if os.path.isdir(d):
                out.append(d)
        return out
    # BTN/ALS: key is a family prefix, match "<FAM>_*".
    return [
        os.path.join(gdir, m) for m in os.listdir(gdir)
        if m.split("_")[0] == key and os.path.isdir(os.path.join(gdir, m))
    ]


def _seed_jsons(model_dir):
    # configs are one level deeper: <model_dir>/<config>/*.json
    files = sorted(glob.glob(os.path.join(model_dir, "*", "*.json")))
    if not files:  # tolerate flat layout
        files = sorted(glob.glob(os.path.join(model_dir, "*.json")))
    return files


def _load_runs(group, dataset, key):
    """Yield the parsed `uncertainty` dict for every available seed run."""
    for md in _model_dirs(group, dataset, key):
        for f in _seed_jsons(md):
            d = load_run_json(f)
            if d is None:
                continue
            u = d.get("uncertainty")
            if isinstance(u, dict):
                yield u


def scalar_values(group, dataset, key, metric):
    """List of finite per-seed values of `metric` for one model on one dataset."""
    out = []
    for u in _load_runs(group, dataset, key):
        v = u.get(metric)
        if _is_finite(v):
            out.append(float(v))
    return out


# Top-level predictive-quality key (R^2-like; HIGHER is better). It lives at the
# TOP of each run json (NOT inside the `uncertainty` block), so it needs its own
# loader below.
VAL_QUALITY_KEY = "val_quality"


def val_quality_values(group, dataset, key):
    """List of finite per-seed `val_quality` values for one model on one dataset.

    `val_quality` is read from the top level of each run json (it is the
    validation predictive quality used to choose the winning BTN family; higher
    is better).
    """
    out = []
    for md in _model_dirs(group, dataset, key):
        for f in _seed_jsons(md):
            d = load_run_json(f)
            if d is None:
                continue
            v = d.get(VAL_QUALITY_KEY)
            if _is_finite(v):
                out.append(float(v))
    return out


def agg(values):
    """(mean, std, n) for a list, or None if empty."""
    if not values:
        return None
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s, len(values)


# --- BTN family selection (validation-quality-best, per dataset) ------------

def _score_for_direction(mean, info):
    """Lower-is-better score, used to BOLD the best cell per column in tables."""
    d = info["direction"]
    if d == "min":
        return mean
    if d == "max":
        return -mean
    if d == "target":
        return abs(mean - info["target"])
    raise ValueError(d)


def valbest_btn_family(dataset):
    """Return the BTN family with the highest MEAN `val_quality` on `dataset`.

    This is a *metric-independent*, per-dataset choice: for each dataset we pick
    the single BTN family that predicts best on the validation set, then present
    THAT family's uncertainty metrics everywhere. Returns the family name, or
    None if no BTN family has any `val_quality` here.
    """
    best = None  # (mean_val_quality, family)
    for fam in TN_FAMILY_ORDER:
        vals = val_quality_values("BTN", dataset, fam)
        if not vals:
            continue
        m = statistics.mean(vals)
        if best is None or m > best[0]:
            best = (m, fam)
    return None if best is None else best[1]


def valbest_btn_metric(dataset, metric):
    """(family, mean, std, n) of the val-quality-best BTN family for `metric`.

    The family is fixed per dataset by `valbest_btn_family` (chosen on
    validation predictive quality, NOT on the uncertainty metric itself); this
    returns that family's aggregated uncertainty `metric`. Returns None if the
    chosen family has no data for this metric here.
    """
    fam = valbest_btn_family(dataset)
    if fam is None:
        return None
    a = agg(scalar_values("BTN", dataset, fam, metric))
    if a is None:
        return None
    mean, std, n = a
    return fam, mean, std, n


def btn_rows(dataset, metric):
    """The single BTN row for a dataset/metric.

    Returns a list with 0 or 1 tuple (display_label, key, mean, std, n, family):
    the BTN family that predicts best on validation (`val_quality`) for this
    dataset (the family is fixed per dataset, independent of `metric`), reporting
    that family's uncertainty `metric` as mean +/- std over its seeds. `key` is
    always "BTN" (drives colour/highlight styling). Missing data -> empty list.
    """
    vb = valbest_btn_metric(dataset, metric)
    if vb is None:
        return []
    fam, mean, std, n = vb
    return [(BTN_DISPLAY, "BTN", mean, std, n, fam)]


# --- Curve loading -----------------------------------------------------------

def _avg_curve(curves, fields):
    """Average a list of curve dicts elementwise over the requested fields.

    Curves with differing lengths are skipped to keep arrays rectangular.
    Returns dict field -> (mean_list, std_list), or None.
    """
    if not curves:
        return None
    import numpy as np
    ref_len = {f: len(curves[0].get(f, [])) for f in fields}
    stacks = {f: [] for f in fields}
    for c in curves:
        ok = all(len(c.get(f, [])) == ref_len[f] and ref_len[f] > 0 for f in fields)
        if not ok:
            continue
        if any(not all(_is_finite(x) for x in c[f]) for f in fields):
            continue
        for f in fields:
            stacks[f].append(c[f])
    out = {}
    for f in fields:
        if not stacks[f]:
            return None
        arr = np.asarray(stacks[f], dtype=float)
        out[f] = (arr.mean(axis=0), arr.std(axis=0))
    return out


def reliability_curve(group, dataset, key):
    """Averaged reliability curve: dict with expected/observed -> (mean,std)."""
    curves = [u["reliability_curve"] for u in _load_runs(group, dataset, key)
              if isinstance(u.get("reliability_curve"), dict)]
    return _avg_curve(curves, ["expected", "observed"])


def sparsification_curve(group, dataset, key):
    """Averaged sparsification curve: fractions/rmse_by_unc/rmse_by_oracle."""
    curves = [u["sparsification_curve"] for u in _load_runs(group, dataset, key)
              if isinstance(u.get("sparsification_curve"), dict)]
    return _avg_curve(curves, ["fractions", "rmse_by_unc", "rmse_by_oracle"])


def btn_curve_series(dataset, metric, curve_field, fields):
    """BTN curve series: a single-element list [(key, curve, family_tag)].

    The curve is the seed-average of `curve_field` for the val-quality-best BTN
    family on this dataset (key "BTN", tag = that family name). Empty list if the
    chosen family has no usable curve here.
    """
    fam = valbest_btn_family(dataset)
    if fam is None:
        return []
    c = _avg_curve(
        [u[curve_field] for u in _load_runs("BTN", dataset, fam)
         if isinstance(u.get(curve_field), dict)],
        fields,
    )
    return [("BTN", c, fam)] if c is not None else []


# --- Convenience: iterate "BTN(val-best) + each baseline" for a metric -------

def models_for_metric(dataset, metric):
    """Ordered list of (model_key, mean, std, n, family, label) for one dataset.

    The first entry is the val-quality-best BTN family (model_key 'BTN', family
    = its name); the baselines follow (family = None). `label` is the display
    name for the row.
    """
    out = []
    for label, key, mean, std, n, fam in btn_rows(dataset, metric):
        out.append((key, mean, std, n, fam, label))
    for b in BASELINE_ORDER:
        a = agg(scalar_values("baseline", dataset, b, metric))
        if a is not None:
            out.append((b, a[0], a[1], a[2], None, display_name(b)))
    return out
