#!/usr/bin/env python3
"""Figure 3 -- Calibration vs Sharpness trade-off scatter.
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
        for key, mean, std, n, fam in C.models_for_metric(ds, metric):
            d[key] = (mean, std)
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
    xpts = _per_dataset_points(X_METRIC)
    ypts = _per_dataset_points(Y_METRIC)

    xvals = {k: [] for k in C.MODEL_ORDER}
    yvals = {k: [] for k in C.MODEL_ORDER}
    for ds in C.DATASET_ORDER:
        scale = _sharpness_scale(ds, ypts[ds])
        for k in C.MODEL_ORDER:
            if k in xpts[ds]:
                xvals[k].append(xpts[ds][k][0])
            if k in ypts[ds]:
                yvals[k].append(ypts[ds][k][0] / scale)

    res = {}
    for k in C.MODEL_ORDER:
        if not xvals[k] or not yvals[k]:
            continue
        xm = np.mean(xvals[k]); xs = np.std(xvals[k]) / np.sqrt(len(xvals[k]))
        ym = np.mean(yvals[k]); ys = np.std(yvals[k]) / np.sqrt(len(yvals[k]))
        res[k] = (xm, xs, ym, ys)
    return res


def make_scatter():
    plt = C._plt
    res = _aggregate_normalised()
    fig, ax = plt.subplots(figsize=(5.0, 4.4))

    for k in C.MODEL_ORDER:
        if k not in res:
            continue
        xm, xs, ym, ys = res[k]
        big = (k == C.HIGHLIGHT)
        st = C.style_for(k, "marker")
        ax.errorbar(xm, ym, xerr=xs, yerr=ys, fmt=C.MARKERS[k],
                    color=C.COLORS[k], elinewidth=C.STYLE["errorbar_lw"],
                    capsize=C.STYLE["capsize"], label=C.display_name(k), **st)
        ax.annotate(C.display_name(k), (xm, ym),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=9, fontweight="bold" if big else "normal",
                    color=C.COLORS[k])

    ax.set_xlabel(f"{C.METRIC_INFO[X_METRIC]['label']}  (calibration error lower is better)")
    ax.set_ylabel(f"Normalised {C.METRIC_INFO[Y_METRIC]['label'].lower()}  (lower is sharper)")
    ax.set_title("Calibration sharpness trade off\n(bottom left is calibrated and sharp)")

    # "best corner" arrow
    xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()
    ax.annotate("better", xy=(xlo, ylo), xytext=(xlo + 0.45 * (xhi - xlo),
                ylo + 0.45 * (yhi - ylo)),
                arrowprops=dict(arrowstyle="->", color="0.4", lw=1.4),
                color="0.4", fontsize=9, ha="center")
    fig.tight_layout()
    C.savefig(fig, "calib_sharp_scatter.pdf")
    plt.close(fig)


def make_per_dataset():
    plt = C._plt
    xpts = _per_dataset_points(X_METRIC)
    ypts = _per_dataset_points(Y_METRIC)
    n = len(C.DATASET_ORDER)
    rows = int(np.ceil(n / GRID_COLS))
    fig, axes = plt.subplots(rows, GRID_COLS, figsize=(2.4 * GRID_COLS, 2.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, ds in enumerate(C.DATASET_ORDER):
        ax = axes[i]
        for k in C.MODEL_ORDER:
            if k in xpts[ds] and k in ypts[ds]:
                st = C.style_for(k, "marker")
                ax.scatter(xpts[ds][k][0], ypts[ds][k][0],
                           s=st["ms"] ** 2, color=C.COLORS[k],
                           marker=C.MARKERS[k], zorder=st["zorder"],
                           edgecolor=st["markeredgecolor"],
                           linewidth=st["markeredgewidth"],
                           label=C.display_name(k) if i == 0 else None)
        ax.set_title(C.DATASET_DISPLAY[ds])
        # Pad the data limits so markers sitting at the top/right edge of the
        # axes are not clipped by the spines.
        ax.margins(x=0.12, y=0.15)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.supxlabel(f"{C.METRIC_INFO[X_METRIC]['label']} (lower is better)", y=0.02)
    fig.supylabel(f"{C.METRIC_INFO[Y_METRIC]['label']} (lower is better)", x=0.01)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Calibration vs sharpness per dataset", y=1.0, fontsize=12)
    fig.tight_layout(rect=(0.03, 0.04, 1, 0.98))
    C.savefig(fig, "calib_sharp_per_dataset.pdf")
    plt.close(fig)


def main():
    C._plt = C.apply_style()
    print("Figure 3: calibration-sharpness scatter")
    make_scatter()
    make_per_dataset()


if __name__ == "__main__":
    main()
