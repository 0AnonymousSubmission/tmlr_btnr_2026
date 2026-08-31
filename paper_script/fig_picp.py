#!/usr/bin/env python3
"""Figure 5 -- PICP coverage vs the 95% target.

Prediction Interval Coverage Probability (PICP) should equal the nominal
confidence level (0.95). Below the line = over-confident (intervals too narrow);
above = under-confident (too wide). This is the honest, one-glance check that
complements the reliability diagram.

Design:
  * grouped dot plot: x = datasets, y = PICP, one marker per model, +-std bars;
  * a bold horizontal line at 0.95 (target) with a shaded +-0.05 tolerance band;
  * BTN markers enlarged + black-edged so they pop.

Output: picp_coverage.pdf
"""

import numpy as np
import unc_common as C

# ---- CONFIG ----------------------------------------------------------------
METRIC = "unc_picp"
TARGET = 0.95
TOL = 0.05


def make():
    plt = C._plt
    datasets = C.DATASET_ORDER
    xbase = np.arange(len(datasets))
    present_models = C.model_order()
    # small horizontal offsets so markers don't overlap
    offs = np.linspace(-0.3, 0.3, len(present_models))

    # Precompute per-dataset model values keyed by model.
    per_ds = {ds: {key: (mean, std)
                   for key, mean, std, n, fam, label
                   in C.models_for_metric(ds, METRIC)}
              for ds in datasets}

    fig, ax = plt.subplots(figsize=(10, 4.2))
    # Alternating vertical background bands to differentiate datasets:
    # light yellow for even-indexed datasets, plain background for odd.
    for di in range(len(datasets)):
        if di % 2 == 0:
            ax.axvspan(di - 0.5, di + 0.5, color="#fffbe6", zorder=-1)
    ax.axhspan(TARGET - TOL, TARGET + TOL, color="0.85", zorder=0,
               label=f"\u00b1{TOL:g} tolerance")
    ax.axhline(TARGET, color="0.2", lw=C.STYLE["ref_lw"], ls="--", zorder=1,
               label=f"target {TARGET:g}")

    from matplotlib.lines import Line2D
    legend_handles, legend_labels = [], []
    for mi, k in enumerate(present_models):
        xs, ys, es = [], [], []
        for di, ds in enumerate(datasets):
            a = per_ds[ds].get(k)
            if a is None:
                continue
            xs.append(di + offs[mi]); ys.append(a[0]); es.append(a[1])
        if not xs:
            continue
        st = C.style_for(k, "marker")
        ax.errorbar(xs, ys, yerr=es, fmt=C.MARKERS[k], color=C.COLORS[k],
                    capsize=C.STYLE["capsize"], elinewidth=C.STYLE["errorbar_lw"],
                    ls="none", **st)
        legend_handles.append(Line2D([0], [0], marker=C.MARKERS[k], color=C.COLORS[k],
                                     ls="none", **st))
        legend_labels.append(C.display_name(k))

    ax.set_xticks(xbase)
    ax.set_xticklabels([C.DATASET_DISPLAY[d] for d in datasets], rotation=30, ha="right")
    ax.set_ylabel("PICP (95% interval coverage)")
    ax.set_title("Interval coverage vs nominal 95% (on the line means well calibrated)")
    ax.set_ylim(min(0.5, ax.get_ylim()[0]), 1.02)
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ref_handles, ref_labels = ax.get_legend_handles_labels()
    handles = ref_handles + legend_handles
    labels = ref_labels + legend_labels
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=len(labels), frameon=False, handletextpad=0.4,
              columnspacing=1.0, borderaxespad=0.0)
    fig.tight_layout()
    C.savefig(fig, "picp_coverage.pdf")
    plt.close(fig)


def main(variant=None):
    C._plt = C.apply_style()
    print("Figure 5: PICP coverage [valbest]")
    make()


if __name__ == "__main__":
    main()
