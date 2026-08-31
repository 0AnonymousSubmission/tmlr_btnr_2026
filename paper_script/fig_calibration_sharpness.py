#!/usr/bin/env python3
"""Figure 3 -- Calibration vs Sharpness trade-off scatter.

The headline "one glance" figure. A useful probabilistic model must be BOTH
calibrated (small ECE) AND sharp (small predictive std / interval width).
We plot one marker per model:

    x = ECE            (calibration error, ->0 is better)
    y = sharpness      (mean predictive std, lower = more confident)

To make the two axes comparable across datasets with very different y-scales,
sharpness is normalised per dataset by the median sharpness across models on
that dataset, then averaged over datasets. ECE is already scale-free.

The bottom-left corner = "calibrated AND sharp" = best. An arrow annotation
points there so the reading is immediate.

Outputs (PDF):
  * calib_sharp_scatter.pdf  -- aggregated over datasets (means +- std error bars)
  * calib_sharp_per_dataset.pdf -- 2x5 small multiples, raw per-dataset points
"""

import os
import csv

import numpy as np
import unc_common as C

# ---- CONFIG ----------------------------------------------------------------
X_METRIC = "unc_ece"
Y_METRIC = "unc_sharpness"   # alternative: "unc_mpiw"
GRID_COLS = 5

# Per-dataset grid: show EVERY BTN family (not just the val-best one) so the
# spread across architectures is visible. Each family gets its own marker (all in
# the BTN red) and a short text tag; the val-best family is ringed in black to
# flag which one the tables/curves actually report.
# All FILLED marker shapes so the black "val-best" ring is always visible
# (unfilled markers like 'x'/'1' ignore edgecolor in matplotlib). These are
# chosen to be DISTINCT from the baseline markers (GP=s, BDE=^, HSBNN=o, BASS=v),
# so BTN families never collide with a baseline shape.
BTN_FAMILY_MARKERS = {
    "CPD": "X", "LMPO2": "P", "MPO2": "*", "BTT": "D", "TR": "p",
}
BTN_FAMILY_MS = 7.0          # base marker size for family points

# ---- LAYOUT / TIGHTNESS KNOBS ----------------------------------------------
# Tweak these to control how tight the figures are. Smaller pad = tighter.
PANEL_W = 2.4                # width  (inches) per small-multiple panel
PANEL_H = 2.4                # height (inches) per small-multiple panel
GRID_WPAD = 0.04             # horizontal padding between panels (constrained layout, inches)
GRID_HPAD = 0.06             # vertical padding between panels (constrained layout, inches)
GRID_PAD = 0.02              # outer padding around the whole grid (inches)
SAVE_PAD_INCHES = 0.02       # extra border kept by the final tight-bbox crop
# Scatter (aggregated) figure:
SCATTER_W = 5.0
SCATTER_H = 4.4
LABEL_FONTSIZE = 9           # per-marker text label size on the scatter

# Per-dataset std(y), precomputed by compute_target_std.py. Sharpness is in raw
# y-units (the pipeline normalizes X but NOT y), so we divide by std(y) to get a
# data-intrinsic, composition-independent scale: 1.0 == as wide as the marginal
# (model learned nothing), <1 == genuinely sharp.
_TARGET_STD_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "target_std.csv")


def _load_target_std():
    """dataset -> std(y), read from target_std.csv. {} if the file is absent."""
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


def _per_dataset_points(metric):
    """dict dataset -> {model_key: (mean,std)} for `metric`."""
    out = {}
    for ds in C.DATASET_ORDER:
        d = {}
        for key, mean, std, n, fam, label in C.models_for_metric(ds, metric):
            d[key] = (mean, std)
        out[ds] = d
    return out


def _per_dataset_family_points(metric):
    """dict dataset -> {btn_family: (mean,std)} for every BTN family with data."""
    out = {}
    for ds in C.DATASET_ORDER:
        d = {}
        for fam in C.TN_FAMILY_ORDER:
            a = C.agg(C.scalar_values("BTN", ds, fam, metric))
            if a is not None:
                d[fam] = (a[0], a[1])
        out[ds] = d
    return out


def _sharpness_scale(ds, ypts_ds):
    """Per-dataset normaliser for sharpness.

    Prefer the data-intrinsic std(y) (from target_std.csv); fall back to the
    across-model median only if std(y) is unavailable for this dataset.
    """
    s = _TARGET_STD.get(ds)
    if s and s > 0 and np.isfinite(s):
        return s
    ymeans = [v[0] for v in ypts_ds.values() if np.isfinite(v[0])]
    med = np.median(ymeans) if ymeans else 1.0
    return med if med > 0 else 1.0


def _aggregate_normalised():
    """Return per-model aggregated (x_mean, x_sem, y_mean, y_sem).

    y (sharpness) is normalised per dataset by std(y) so datasets with large
    target ranges don't dominate, and so the value has an absolute meaning
    (relative to the data, not to the other models).
    """
    order = C.model_order()
    xpts = _per_dataset_points(X_METRIC)
    ypts = _per_dataset_points(Y_METRIC)

    xvals = {k: [] for k in order}
    yvals = {k: [] for k in order}
    for ds in C.DATASET_ORDER:
        scale = _sharpness_scale(ds, ypts[ds])
        for k in order:
            if k in xpts[ds]:
                xvals[k].append(xpts[ds][k][0])
            if k in ypts[ds]:
                yvals[k].append(ypts[ds][k][0] / scale)

    res = {}
    for k in order:
        if not xvals[k] or not yvals[k]:
            continue
        xm = np.mean(xvals[k]); xs = np.std(xvals[k]) / np.sqrt(len(xvals[k]))
        ym = np.mean(yvals[k]); ys = np.std(yvals[k]) / np.sqrt(len(yvals[k]))
        res[k] = (xm, xs, ym, ys)
    return res


# ---- LABEL DE-OVERLAP KNOBS -------------------------------------------------
LABEL_BASE_OFFSET = 9.0      # initial label offset from its marker (points)
LABEL_MAX_DIST = 60.0        # max distance a label may drift from its marker (points)
LABEL_ITERATIONS = 600       # repulsion iterations (more = better separation)
LABEL_REPULSE_STEP = 4.0     # push strength per overlapping-pair per iteration (points)
LABEL_MIN_GAP = 3.0          # min pixel gap enforced between two labels

# Manual placement overrides, keyed by the label text. These labels are pinned
# to a fixed corner offset (in points) relative to their marker and are excluded
# from the automatic repulsion (they neither move nor push others). Use for the
# few cases where a specific hand-placed corner reads best, e.g. HSBNN sitting
# below-left of its point and BDE below-right of its point.
#   (dx, dy) in points; +x = right, +y = up, so bottom-left = (-, -).
LABEL_MANUAL_OFFSETS = {
    "HSBNN": (-8.0, -14.0),   # bottom-left of the point
    "BDE":   (8.0, -8.0),     # bottom-right of the point (raised a bit)
}
LABEL_ZORDER = 1000          # draw all labels in front of markers/error bars/arrow


def _place_labels_no_overlap(ax, points, *, fontsize=9):
    """Annotate ``points`` next to their markers, nudging labels apart.

    ``points`` is a list of ``(x, y, text, kwargs)`` where ``kwargs`` is passed
    to ``ax.annotate``. A light iterative repulsion in display space separates
    labels that would otherwise collide (like the BTNR/BDE and HSBNN clashes)
    without needing an external dependency.
    Behaviour is tunable via the LABEL_* constants above. Labels listed in
    LABEL_MANUAL_OFFSETS are pinned to their given corner and left out of the
    repulsion entirely.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi

    anns = []            # only the auto-placed annotations participate in repulsion
    offsets = []
    for x, y, text, kw in points:
        manual = LABEL_MANUAL_OFFSETS.get(text)
        if manual is not None:
            # Pinned label: fixed corner offset, excluded from repulsion.
            ha = "right" if manual[0] < 0 else "left"
            va = "top" if manual[1] < 0 else "bottom"
            ax.annotate(text, (x, y), xycoords="data",
                        textcoords="offset points", xytext=manual,
                        fontsize=fontsize, ha=ha, va=va,
                        zorder=LABEL_ZORDER, **kw)
            continue
        anns.append(ax.annotate(text, (x, y), xycoords="data",
                                textcoords="offset points",
                                xytext=(LABEL_BASE_OFFSET, LABEL_BASE_OFFSET),
                                fontsize=fontsize, zorder=LABEL_ZORDER, **kw))
        offsets.append(np.array([LABEL_BASE_OFFSET / 72.0 * dpi] * 2, dtype=float))

    def _boxes():
        fig.canvas.draw()
        return [a.get_window_extent(renderer=renderer) for a in anns]

    for _ in range(LABEL_ITERATIONS):
        boxes = _boxes()
        moved = False
        for i in range(len(anns)):
            push = np.zeros(2)
            bi = boxes[i]
            ci = np.array([(bi.x0 + bi.x1) / 2, (bi.y0 + bi.y1) / 2])
            for j in range(len(anns)):
                if i == j:
                    continue
                bj = boxes[j]
                ovx = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
                ovy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
                if ovx > -LABEL_MIN_GAP and ovy > -LABEL_MIN_GAP:
                    cj = np.array([(bj.x0 + bj.x1) / 2, (bj.y0 + bj.y1) / 2])
                    d = ci - cj
                    norm = np.hypot(*d) or 1.0
                    push += (d / norm) * LABEL_REPULSE_STEP
            if np.hypot(*push):
                offsets[i] += push
                # keep the label tethered to its marker
                dist = np.hypot(*offsets[i])
                max_px = LABEL_MAX_DIST / 72.0 * dpi
                if dist > max_px:
                    offsets[i] = offsets[i] / dist * max_px
                anns[i].xyann = (offsets[i][0] / dpi * 72.0,
                                 offsets[i][1] / dpi * 72.0)
                moved = True
        if not moved:
            break
    return anns


def make_scatter():
    plt = C._plt
    res = _aggregate_normalised()
    fig, ax = plt.subplots(figsize=(SCATTER_W, SCATTER_H))

    label_points = []  # (x, y, text, annotate-kwargs) for de-overlap pass
    for k in C.model_order():
        if k not in res:
            continue
        xm, xs, ym, ys = res[k]
        big = C.is_btn_key(k)
        st = C.style_for(k, "marker")
        ax.errorbar(xm, ym, xerr=xs, yerr=ys, fmt=C.MARKERS[k],
                    color=C.COLORS[k], elinewidth=C.STYLE["errorbar_lw"],
                    capsize=C.STYLE["capsize"], label=C.display_name(k), **st)
        label_points.append((xm, ym, C.display_name(k),
                             dict(fontweight="bold" if big else "normal",
                                  color=C.COLORS[k])))

    # Place the per-marker text labels so they do not overlap each other.
    _place_labels_no_overlap(ax, label_points, fontsize=LABEL_FONTSIZE)

    ax.set_xlabel(f"{C.METRIC_INFO[X_METRIC]['label']}  (calibration error lower is better)")
    ax.set_ylabel(f"Normalised {C.METRIC_INFO[Y_METRIC]['label'].lower()}  (lower is sharper)")
    ax.set_title("Calibration sharpness trade off\n(bottom left is calibrated and sharp)")

    # "best corner" arrow (half length, tail label on its left)
    # xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()
    # ax.annotate("better", xy=(xlo, ylo), xytext=(xlo + 0.225 * (xhi - xlo),
    #             ylo + 0.225 * (yhi - ylo)),
    #             arrowprops=dict(arrowstyle="->", color="0.4", lw=1.4),
    #             color="0.4", fontsize=9, ha="right", va="center")
    fig.tight_layout()
    C.savefig(fig, "calib_sharp_scatter.pdf")
    plt.close(fig)


def make_per_dataset():
    plt = C._plt
    # BTN: every family (not just val-best). Baselines: their single points.
    fam_x = _per_dataset_family_points(X_METRIC)
    fam_y = _per_dataset_family_points(Y_METRIC)
    xpts = _per_dataset_points(X_METRIC)
    ypts = _per_dataset_points(Y_METRIC)
    n = len(C.DATASET_ORDER)
    rows = int(np.ceil(n / GRID_COLS))
    fig, axes = plt.subplots(rows, GRID_COLS,
                             figsize=(PANEL_W * GRID_COLS, PANEL_H * rows),
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=GRID_WPAD, h_pad=GRID_HPAD,
                                wspace=0, hspace=0)
    axes = np.atleast_1d(axes).ravel()

    from matplotlib.lines import Line2D
    legend_handles = []

    for i, ds in enumerate(C.DATASET_ORDER):
        ax = axes[i]

        # --- baselines ------------------------------------------------------
        for k in C.BASELINE_ORDER:
            if k in xpts[ds] and k in ypts[ds]:
                st = C.style_for(k, "marker")
                ax.scatter(xpts[ds][k][0], ypts[ds][k][0],
                           s=st["ms"] ** 2, color=C.COLORS[k],
                           marker=C.MARKERS[k], zorder=st["zorder"],
                           edgecolor=st["markeredgecolor"],
                           linewidth=st["markeredgewidth"])

        # --- BTN families (all of them) ------------------------------------
        # All BTNR points share a thin black outline so they read as one group,
        # distinct from the baselines. Families are told apart by marker shape
        # (see legend); no per-family highlight.
        for fam in C.TN_FAMILY_ORDER:
            if fam not in fam_x[ds] or fam not in fam_y[ds]:
                continue
            xv = fam_x[ds][fam][0]
            yv = fam_y[ds][fam][0]
            mk = BTN_FAMILY_MARKERS.get(fam, "x")
            ax.scatter(xv, yv, s=BTN_FAMILY_MS ** 2, color=C.COLORS["BTN"],
                       marker=mk, zorder=5, edgecolor="black", linewidth=0.6)
            # families are identified by their legend symbols (no in-plot tags).

        ax.set_title(C.DATASET_DISPLAY[ds])
        # Pad the data limits so markers sitting at the top/right edge of the
        # axes are not clipped by the spines.
        ax.margins(x=0.15, y=0.18)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    # Build a combined legend: one entry per BTN family, then each baseline.
    # BTN families carry the same thin black outline used in the panels.
    for fam in C.TN_FAMILY_ORDER:
        legend_handles.append(Line2D(
            [0], [0], marker=BTN_FAMILY_MARKERS.get(fam, "x"), linestyle="none",
            color=C.COLORS["BTN"], markerfacecolor=C.COLORS["BTN"],
            markeredgecolor="black", markeredgewidth=0.6, markersize=6,
            label=C.family_display(fam)))
    for k in C.BASELINE_ORDER:
        legend_handles.append(Line2D(
            [0], [0], marker=C.MARKERS[k], linestyle="none", color=C.COLORS[k],
            markerfacecolor=C.COLORS[k], markeredgecolor=C.COLORS[k],
            markersize=6, label=C.display_name(k)))
    labels = [h.get_label() for h in legend_handles]

    C.finalize_grid(
        fig,
        supxlabel=f"{C.METRIC_INFO[X_METRIC]['label']} (lower is better)",
        supylabel=f"{C.METRIC_INFO[Y_METRIC]['label']} (lower is better)",
        suptitle="Calibration vs sharpness per dataset (all BTNR families)",
        handles=legend_handles, labels=labels,
        ncol=len(legend_handles))
    C.savefig(fig, "calib_sharp_per_dataset.pdf", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


def main(variant=None):
    C._plt = C.apply_style()
    print("Figure 3: calibration-sharpness scatter [valbest]")
    make_scatter()
    make_per_dataset()


if __name__ == "__main__":
    main()
