#!/usr/bin/env python3
"""Figure 2 -- Sparsification error plots (does uncertainty rank errors?).
Outputs (PDF):
  * sparsification_grid.pdf -- 2x5 small multiples (all datasets).
  * sparsification_main.pdf -- one headline dataset with the BTN gap shaded.
"""

import numpy as np
import unc_common as C

# ---- CONFIG ----------------------------------------------------------------
METRIC = "unc_ause"
MAIN_DATASET = "concrete"
GRID_COLS = 5
NORMALIZE = True   # normalise each curve by its RMSE at fraction 0 (scale-free)


def _norm(arr, ref):
    return arr / ref if (NORMALIZE and ref and np.isfinite(ref) and ref > 0) else arr


def _plot_one(ax, dataset, legend=False, shade_gap=False):
    # Best BTN family chosen PER DATASET for this metric.
    bf = C.best_btn_family(dataset, METRIC)
    btn_family = bf[0] if bf else None
    sc = C.sparsification_curve("BTN", dataset, btn_family) if btn_family else None
    if sc is not None:
        x = sc["fractions"][0]
        unc = sc["rmse_by_unc"][0]
        orc = sc["rmse_by_oracle"][0]
        ref = unc[0]
        unc, orc = _norm(unc, ref), _norm(orc, ref)
        st = C.style_for("BTN", "line")
        ax.plot(x, orc, color=C.COLORS["BTN"], lw=C.STYLE["oracle_lw"], ls="--",
                alpha=0.8, label="oracle" if legend else None,
                zorder=st["zorder"])
        ax.plot(x, unc, color=C.COLORS["BTN"], lw=st["lw"], marker="o",
                ms=st["ms"], label="BTNR" if legend else None,
                zorder=st["zorder"] + 1)
        if shade_gap:
            ax.fill_between(x, orc, unc, color=C.COLORS["BTN"],
                            alpha=C.STYLE["gap_alpha"], lw=0,
                            label="AUSE gap" if legend else None)

    for b in C.BASELINE_ORDER:
        sc = C.sparsification_curve("baseline", dataset, b)
        if sc is None:
            continue
        x = sc["fractions"][0]
        unc = sc["rmse_by_unc"][0]
        unc = _norm(unc, unc[0])
        st = C.style_for(b, "line")
        ax.plot(x, unc, color=C.COLORS[b], marker=C.MARKERS[b],
                label=C.display_name(b) if legend else None, **st)

    fam_tag = f"  ({C.family_display(btn_family)})" if btn_family else ""
    ax.set_title(C.DATASET_DISPLAY[dataset] + fam_tag, fontsize=10)
    ax.set_xlim(0, max(x) if 'x' in dir() else 1)


def make_grid():
    plt = C._plt
    n = len(C.DATASET_ORDER)
    rows = int(np.ceil(n / GRID_COLS))
    fig, axes = plt.subplots(rows, GRID_COLS, figsize=(2.4 * GRID_COLS, 2.3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, ds in enumerate(C.DATASET_ORDER):
        _plot_one(axes[i], ds, legend=(i == 0))
    for j in range(n, len(axes)):
        axes[j].axis("off")
    ylab = "RMSE (normalised)" if NORMALIZE else "RMSE on remaining points"
    fig.supxlabel("Fraction of most uncertain points removed", y=0.02)
    fig.supylabel(ylab, x=0.01)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Sparsification: lower and closer to oracle is better",
                 y=1.0, fontsize=12)
    fig.tight_layout(rect=(0.03, 0.05, 1, 0.98))
    C.savefig(fig, "sparsification_grid.pdf")
    plt.close(fig)


def make_main():
    plt = C._plt
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    _plot_one(ax, MAIN_DATASET, legend=True, shade_gap=True)
    ax.set_xlabel("Fraction of most-uncertain points removed")
    ax.set_ylabel("RMSE (normalised)" if NORMALIZE else "RMSE on remaining points")
    ax.legend(loc="upper right", ncol=1)
    fig.tight_layout()
    C.savefig(fig, "sparsification_main.pdf")
    plt.close(fig)


def main():
    C._plt = C.apply_style()
    print("Figure 2: sparsification plots")
    make_grid()
    make_main()


if __name__ == "__main__":
    main()
