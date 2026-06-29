#!/usr/bin/env python3
"""Figure 5 -- PICP coverage vs the 95% target.
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
    present_models = [k for k in C.MODEL_ORDER]
    # small horizontal offsets so markers don't overlap
    offs = np.linspace(-0.3, 0.3, len(present_models))

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axhspan(TARGET - TOL, TARGET + TOL, color="0.85", zorder=0,
               label=f"\u00b1{TOL:g} tolerance")
    ax.axhline(TARGET, color="0.2", lw=C.STYLE["ref_lw"], ls="--", zorder=1,
               label=f"target {TARGET:g}")

    drawn = set()
    for mi, k in enumerate(present_models):
        xs, ys, es = [], [], []
        for di, ds in enumerate(datasets):
            if k == "BTN":
                # best BTN family chosen PER DATASET for PICP
                bf = C.best_btn_family(ds, METRIC)
                a = C.agg(C.scalar_values("BTN", ds, bf[0], METRIC)) if bf else None
            else:
                a = C.agg(C.scalar_values("baseline", ds, k, METRIC))
            if a is None:
                continue
            xs.append(di + offs[mi]); ys.append(a[0]); es.append(a[1])
        if not xs:
            continue
        st = C.style_for(k, "marker")
        ax.errorbar(xs, ys, yerr=es, fmt=C.MARKERS[k], color=C.COLORS[k],
                    capsize=C.STYLE["capsize"], elinewidth=C.STYLE["errorbar_lw"],
                    ls="none", label=C.display_name(k), **st)
        drawn.add(k)

    ax.set_xticks(xbase)
    ax.set_xticklabels([C.DATASET_DISPLAY[d] for d in datasets], rotation=30, ha="right")
    ax.set_ylabel("PICP (95% interval coverage)")
    ax.set_title("Interval coverage vs nominal 95% (on the line means well calibrated)")
    ax.set_ylim(min(0.5, ax.get_ylim()[0]), 1.02)
    ax.legend(loc="lower left", ncol=3)
    fig.tight_layout()
    C.savefig(fig, "picp_coverage.pdf")
    plt.close(fig)


def main():
    C._plt = C.apply_style()
    print("Figure 5: PICP coverage")
    make()


if __name__ == "__main__":
    main()
