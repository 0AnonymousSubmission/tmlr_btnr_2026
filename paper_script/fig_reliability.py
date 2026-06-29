#!/usr/bin/env python3
"""Figure 1 -- Reliability diagrams (calibration).
"""

import numpy as np
import unc_common as C

# ---- CONFIG ----------------------------------------------------------------
METRIC = "unc_ece"                 # used only to choose the BTN family
MAIN_DATASET = "concrete"          # dataset shown in the single headline panel
SHOW_BAND = True                   # shaded +-std band around each curve
GRID_COLS = 5


def _plot_one(ax, dataset, legend=False):
    plt = C._plt
    ax.plot([0, 1], [0, 1], ls="--", lw=C.STYLE["ref_lw"], color="0.6",
            label="Perfect" if legend else None, zorder=1)

    # Best BTN family is chosen PER DATASET for this metric.
    bf = C.best_btn_family(dataset, METRIC)
    btn_family = bf[0] if bf else None

    series = []  # (key, curve)
    if btn_family is not None:
        rc = C.reliability_curve("BTN", dataset, btn_family)
        if rc is not None:
            series.append(("BTN", rc))
    for b in C.BASELINE_ORDER:
        rc = C.reliability_curve("baseline", dataset, b)
        if rc is not None:
            series.append((b, rc))

    for key, rc in series:
        x = rc["expected"][0]
        y_m, y_s = rc["observed"]
        color = C.COLORS[key]
        st = C.style_for(key, "line")
        ax.plot(x, y_m, color=color, marker=C.MARKERS[key],
                label=C.display_name(key) if legend else None, **st)
        if SHOW_BAND:
            ax.fill_between(x, y_m - y_s, y_m + y_s, color=color,
                            alpha=C.STYLE["band_alpha"], lw=0,
                            zorder=st["zorder"] - 1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    # title names the per-dataset BTN family so the choice is transparent.
    fam_tag = f"  ({C.family_display(btn_family)})" if btn_family else ""
    ax.set_title(C.DATASET_DISPLAY[dataset] + fam_tag, fontsize=10)


def make_grid():
    plt = C._plt
    n = len(C.DATASET_ORDER)
    rows = int(np.ceil(n / GRID_COLS))
    fig, axes = plt.subplots(rows, GRID_COLS, figsize=(2.1 * GRID_COLS, 2.3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, ds in enumerate(C.DATASET_ORDER):
        _plot_one(axes[i], ds, legend=(i == 0))
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # shared axis labels
    fig.supxlabel("Expected confidence level", y=0.02)
    fig.supylabel("Observed coverage", x=0.02)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Calibration reliability diagrams", y=1.00, fontsize=12)
    fig.tight_layout(rect=(0.03, 0.04, 1, 0.98))
    C.savefig(fig, "reliability_grid.pdf")
    plt.close(fig)


def make_main():
    plt = C._plt
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    _plot_one(ax, MAIN_DATASET, legend=True)
    ax.set_xlabel("Expected confidence level")
    ax.set_ylabel("Observed coverage")
    ax.legend(loc="upper left")
    fig.tight_layout()
    C.savefig(fig, "reliability_main.pdf")
    plt.close(fig)


def main():
    C._plt = C.apply_style()
    print("Figure 1: reliability diagrams")
    make_grid()
    make_main()


if __name__ == "__main__":
    main()
