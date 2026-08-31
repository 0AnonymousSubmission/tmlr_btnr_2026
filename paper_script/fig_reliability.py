#!/usr/bin/env python3
"""Figure 1 -- Reliability diagrams (calibration).

Reads the per-run ``reliability_curve`` (expected vs observed quantile
coverage), averages over seeds, and plots observed vs expected. The diagonal
y=x is "perfectly calibrated"; curves below the diagonal are over-confident,
above are under-confident.

Outputs (PDF, into images/uncertainty/):
  * reliability_grid.pdf  -- 2x5 small multiples, one panel per dataset,
                             BTN (best family) vs every baseline.
  * reliability_main.pdf  -- single clean headline panel (one dataset),
                             for the main text.

The "immediate read" design choices:
  * shaded +-std band so seed variability is visible at a glance;
  * BTN drawn thick + red + on top, baselines thin;
  * the y=x reference is a light grey dashed line and labelled once.
"""

import numpy as np
import unc_common as C

# ---- CONFIG ----------------------------------------------------------------
METRIC = "unc_ece"                 # used only to choose the BTN family
MAIN_DATASET = "concrete"          # dataset shown in the single headline panel
SHOW_BAND = True                   # shaded +-std band around each curve
GRID_COLS = 5

# ---- LAYOUT / TIGHTNESS KNOBS ----------------------------------------------
# Tweak these to control how tight the grid figure is. Smaller pad = tighter.
PANEL_W = 2.1                # width  (inches) per small-multiple panel
PANEL_H = 2.3                # height (inches) per small-multiple panel
GRID_WPAD = 0.02             # horizontal padding between panels (inches)
GRID_HPAD = 0.06             # vertical padding between panels (inches)
SAVE_PAD_INCHES = 0.02       # extra border kept by the final tight-bbox crop


def _plot_one(ax, dataset, legend=False):
    plt = C._plt
    ax.plot([0, 1], [0, 1], ls="--", lw=C.STYLE["ref_lw"], color="0.6",
            label="Perfect" if legend else None, zorder=1)

    series = []  # (key, curve)
    fam_tags = []
    for key, rc, fam in C.btn_curve_series(dataset, METRIC,
                                           "reliability_curve",
                                           ["expected", "observed"]):
        series.append((key, rc))
        if fam:
            fam_tags.append(C.family_display(fam))
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
    # title names the per-dataset BTN family/families so the choice is transparent.
    fam_tag = ("  (" + ", ".join(fam_tags) + ")") if fam_tags else ""
    ax.set_title(C.DATASET_DISPLAY[dataset] + fam_tag, fontsize=10)


def make_grid():
    plt = C._plt
    n = len(C.DATASET_ORDER)
    rows = int(np.ceil(n / GRID_COLS))
    fig, axes = plt.subplots(rows, GRID_COLS,
                             figsize=(PANEL_W * GRID_COLS, PANEL_H * rows),
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=GRID_WPAD, h_pad=GRID_HPAD,
                                wspace=0, hspace=0)
    axes = np.atleast_1d(axes).ravel()
    for i, ds in enumerate(C.DATASET_ORDER):
        _plot_one(axes[i], ds, legend=(i == 0))
    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    C.finalize_grid(fig,
                    supxlabel="Expected confidence level",
                    supylabel="Observed coverage",
                    suptitle="Calibration reliability diagrams",
                    handles=handles, labels=labels)
    C.savefig(fig, "reliability_grid.pdf", pad_inches=SAVE_PAD_INCHES)
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


def main(variant=None):
    C._plt = C.apply_style()
    print("Figure 1: reliability diagrams [valbest]")
    make_grid()
    make_main()


if __name__ == "__main__":
    main()
