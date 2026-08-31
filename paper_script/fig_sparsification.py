#!/usr/bin/env python3
"""Figure 2 -- Sparsification error plots (does uncertainty rank errors?).

For each run we stored a ``sparsification_curve``:
    fractions       : fraction of most-uncertain points removed
    rmse_by_unc     : RMSE on the remaining points when removing by predicted std
    rmse_by_oracle  : RMSE when removing by the TRUE error (oracle ordering)

If uncertainty is informative, ``rmse_by_unc`` should drop as uncertain points
are removed and hug the oracle curve. The gap between them is the
sparsification error (its area = AUSE).

Design for "immediate read":
  * BTN's by-uncertainty curve is solid red; its oracle curve is a thin dashed
    line of the same colour -> the smaller the shaded gap, the better.
  * Baselines shown as their by-uncertainty curves only (their oracle curve is
    essentially identical), to avoid clutter.

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

# ---- LAYOUT / TIGHTNESS KNOBS ----------------------------------------------
# Tweak these to control how tight the grid figure is. Smaller pad = tighter.
PANEL_W = 2.4                # width  (inches) per small-multiple panel
PANEL_H = 2.3                # height (inches) per small-multiple panel
GRID_WPAD = 0.04             # horizontal padding between panels (inches)
GRID_HPAD = 0.06             # vertical padding between panels (inches)
SAVE_PAD_INCHES = 0.02       # extra border kept by the final tight-bbox crop


def _norm(arr, ref):
    return arr / ref if (NORMALIZE and ref and np.isfinite(ref) and ref > 0) else arr


def _plot_one(ax, dataset, legend=False, shade_gap=False):
    fam_tags = []
    btn_series = C.btn_curve_series(dataset, METRIC,
                                    "sparsification_curve",
                                    ["fractions", "rmse_by_unc", "rmse_by_oracle"])
    last_x = None
    for si, (key, sc, fam) in enumerate(btn_series):
        if fam:
            fam_tags.append(C.family_display(fam))
        x = sc["fractions"][0]
        unc = sc["rmse_by_unc"][0]
        orc = sc["rmse_by_oracle"][0]
        ref = unc[0]
        unc, orc = _norm(unc, ref), _norm(orc, ref)
        st = C.style_for(key, "line")
        # Draw the oracle only for the first BTN curve (avoids clutter).
        if si == 0:
            ax.plot(x, orc, color=C.COLORS[key], lw=C.STYLE["oracle_lw"], ls="--",
                    alpha=0.8, label="oracle" if legend else None,
                    zorder=st["zorder"])
        ax.plot(x, unc, color=C.COLORS[key], lw=st["lw"], marker=C.MARKERS[key],
                ms=st["ms"], label=C.display_name(key) if legend else None,
                zorder=st["zorder"] + 1)
        if shade_gap and si == 0:
            ax.fill_between(x, orc, unc, color=C.COLORS[key],
                            alpha=C.STYLE["gap_alpha"], lw=0,
                            label="AUSE gap" if legend else None)
        last_x = x

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
        last_x = x

    fam_tag = ("  (" + ", ".join(fam_tags) + ")") if fam_tags else ""
    ax.set_title(C.DATASET_DISPLAY[dataset] + fam_tag, fontsize=10)
    ax.set_xlim(0, max(last_x) if last_x is not None else 1)


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
    ylab = "RMSE (normalised)" if NORMALIZE else "RMSE on remaining points"
    handles, labels = axes[0].get_legend_handles_labels()
    C.finalize_grid(fig,
                    supxlabel="Fraction of most uncertain points removed",
                    supylabel=ylab,
                    suptitle="Sparsification: lower and closer to oracle is better",
                    handles=handles, labels=labels)
    C.savefig(fig, "sparsification_grid.pdf", pad_inches=SAVE_PAD_INCHES)
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


def main(variant=None):
    C._plt = C.apply_style()
    print("Figure 2: sparsification plots [valbest]")
    make_grid()
    make_main()


if __name__ == "__main__":
    main()
