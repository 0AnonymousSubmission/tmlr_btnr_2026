#!/usr/bin/env python3
"""Shared loading / aggregation / styling utilities for uncertainty figures.
"""

import os
import glob
import json
import math
import statistics
from collections import defaultdict

# ============================================================================
# CONFIG  (edit everything here)
# ============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)

# The tree that contains the `uncertainty` block.
ROOT = os.path.join(_REPO_ROOT, "tests_uncertainty")

IMAGES_DIR = os.path.join(_SCRIPT_DIR, "images", "uncertainty")
TABLES_DIR = os.path.join(_SCRIPT_DIR, "tables")

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
TN_FAMILY_ORDER = ["CPD", "LMPO2", "MPO2", "BTT"]

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

# The BTN entry is a single curated row: the best-performing family per metric
# is chosen automatically (see best_btn_family).
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
MODEL_ORDER = ["BTN"] + BASELINE_ORDER
COLORS = {
    "BTN":   "#d62728",   # red, the highlight
    "GP":    "#1f77b4",   # blue
    "BDE":   "#2ca02c",   # green
    "HSBNN": "#9467bd",   # purple
    "BASS":   "#ff7f0e",   # orange
}
MARKERS = {
    "BTN": "o", "GP": "s", "BDE": "^", "HSBNN": "D", "BASS": "v",
}
# z-order so BTN draws on top.
ZORDER = {"BTN": 5, "GP": 3, "BDE": 3, "HSBNN": 3, "BASS": 3}

# The single model we visually emphasise everywhere (drawn bigger/bolder/on top).
HIGHLIGHT = "BTN"

# --- Centralised plot dimensions ---------------------------------------------
# Every figure pulls its line widths / marker sizes from here, distinguishing
# the highlighted model from the rest. Edit once -> applies to all figures.
STYLE = {
    # line plots (reliability / sparsification curves)
    "line_lw":        {"hi": 1.6, "lo": 1.0},   # main curves
    "line_ms":        {"hi": 3.0, "lo": 2.5},   # markers on curves
    "oracle_lw":      0.9,                        # BTN oracle (sparsification)
    "ref_lw":         1.0,                        # reference diagonal / target line
    "band_alpha":     0.12,                       # +-std shaded band (reliability)
    "gap_alpha":      0.15,                       # AUSE gap shading (sparsification)
    # marker / scatter plots (scatter / rank dotplot / picp)
    "marker_ms":      {"hi": 12, "lo": 8},
    "edge_lw":        {"hi": 1.0, "lo": 0.0},    # marker edge (black ring on highlight)
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
    hi = (key == HIGHLIGHT)
    sel = "hi" if hi else "lo"
    if kind == "line":
        return dict(lw=STYLE["line_lw"][sel], ms=STYLE["line_ms"][sel],
                    zorder=ZORDER[key])
    if kind == "marker":
        return dict(ms=STYLE["marker_ms"][sel],
                    markeredgewidth=STYLE["edge_lw"][sel],
                    markeredgecolor="black" if hi else "none",
                    zorder=ZORDER[key])
    raise ValueError(kind)


def display_name(key):
    if key == "BTN":
        return BTN_DISPLAY
    return BASELINE_DISPLAY.get(key, key)


# Prefix for BTN family display names (matches make_r2_table.py: "B-").
BTN_FAMILY_PREFIX = "B-"


def family_display(family):
    """Display name for a BTN family, e.g. 'LMPO2' -> 'B-LMPO2'."""
    return BTN_FAMILY_PREFIX + family if family else family


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


def savefig(fig, name):
    """Save a figure as PDF into IMAGES_DIR and report the path.

    A tiny padding is added around the tight bounding box so that markers,
    tick labels and annotations near the right/top edges are not clipped.
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)
    path = os.path.join(IMAGES_DIR, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    print(f"  wrote {os.path.relpath(path, _REPO_ROOT)}")
    return path


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
            try:
                d = json.load(open(f))
            except Exception:
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


def agg(values):
    """(mean, std, n) for a list, or None if empty."""
    if not values:
        return None
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s, len(values)


# --- Best BTN family selection ----------------------------------------------

def _score_for_direction(mean, info):
    """Lower-is-better score so we can pick the 'best' family uniformly."""
    d = info["direction"]
    if d == "min":
        return mean
    if d == "max":
        return -mean
    if d == "target":
        return abs(mean - info["target"])
    raise ValueError(d)


def best_btn_family(dataset, metric):
    """Return (family, mean, std, n) of the best BTN family for this metric/dataset.

    'Best' is defined by the metric's optimisation direction (METRIC_INFO).
    Returns None if no BTN family has data here.
    """
    info = METRIC_INFO[metric]
    best = None
    for fam in TN_FAMILY_ORDER:
        a = agg(scalar_values("BTN", dataset, fam, metric))
        if a is None:
            continue
        mean, std, n = a
        score = _score_for_direction(mean, info)
        if best is None or score < best[0]:
            best = (score, fam, mean, std, n)
    if best is None:
        return None
    _, fam, mean, std, n = best
    return fam, mean, std, n


def best_btn_family_overall(metric):
    """Family with the best AVERAGE rank for `metric` across all datasets.

    NOTE: All current figures select the best BTN family PER DATASET via
    `best_btn_family`. This helper is kept for the (optional) case where a single
    family must be fixed across panels; it is not used by default.
    """
    info = METRIC_INFO[metric]
    # rank families per dataset, then average ranks.
    rank_sums = defaultdict(float)
    rank_cnts = defaultdict(int)
    for ds in DATASET_ORDER:
        scored = []
        for fam in TN_FAMILY_ORDER:
            a = agg(scalar_values("BTN", ds, fam, metric))
            if a is None:
                continue
            scored.append((_score_for_direction(a[0], info), fam))
        scored.sort()
        for rank, (_, fam) in enumerate(scored):
            rank_sums[fam] += rank
            rank_cnts[fam] += 1
    avg = {f: rank_sums[f] / rank_cnts[f] for f in rank_cnts if rank_cnts[f]}
    if not avg:
        return TN_FAMILY_ORDER[0]
    return min(avg, key=avg.get)


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


# --- Convenience: iterate "BTN(best) + each baseline" for a metric -----------

def models_for_metric(dataset, metric):
    """Ordered list of (model_key, mean, std, n, extra) for one dataset.

    model_key is 'BTN' (best family) or a baseline key. extra holds the chosen
    BTN family name (or None).
    """
    out = []
    bf = best_btn_family(dataset, metric)
    if bf is not None:
        fam, mean, std, n = bf
        out.append(("BTN", mean, std, n, fam))
    for b in BASELINE_ORDER:
        a = agg(scalar_values("baseline", dataset, b, metric))
        if a is not None:
            out.append((b, a[0], a[1], a[2], None))
    return out
