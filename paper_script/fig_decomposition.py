#!/usr/bin/env python3
"""Figure 6 -- BTN uncertainty decomposition (epistemic vs aleatoric).
Output: uncertainty_decomposition.pdf
"""

import numpy as np
import unc_common as C

# ---- CONFIG ----------------------------------------------------------------
EPI = "unc_epistemic_std"
ALE = "unc_aleatoric_std"
ALE_COLOR = "#4c72b0"
EPI_COLOR = "#dd8452"


def _btn_component(dataset, metric):
    """Mean of `metric` over the best (NLL) BTN family for this dataset."""
    bf = C.best_btn_family(dataset, "unc_nll")
    if bf is None:
        return None
    a = C.agg(C.scalar_values("BTN", dataset, bf[0], metric))
    return a[0] if a else None


def make():
    plt = C._plt
    rows = []
    for ds in C.DATASET_ORDER:
        ale = _btn_component(ds, ALE)
        epi = _btn_component(ds, EPI)
        if ale is None or epi is None:
            continue
        rows.append((ds, ale, epi))
    if not rows:
        print("  no epistemic/aleatoric data found; skipping")
        return
    rows.sort(key=lambda r: r[1] + r[2])
    names = [C.DATASET_DISPLAY[r[0]] for r in rows]
    ale = np.array([r[1] for r in rows])
    epi = np.array([r[2] for r in rows])
    frac = epi / (ale + epi)
    y = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 0.5 * len(rows) + 1.5),
                                   gridspec_kw={"width_ratios": [2, 1]})

    ax1.barh(y, ale, color=ALE_COLOR, label="Aleatoric (irreducible)")
    ax1.barh(y, epi, left=ale, color=EPI_COLOR, label="Epistemic (reducible)")
    ax1.set_yticks(y); ax1.set_yticklabels(names)
    ax1.set_xlabel("Mean predictive std (target units)")
    ax1.set_title("Uncertainty decomposition (BTNR)")
    ax1.legend(loc="lower right")
    ax1.grid(axis="x", alpha=0.25); ax1.grid(axis="y", visible=False)

    ax2.barh(y, frac, color=EPI_COLOR)
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Epistemic fraction")
    ax2.set_title("Reducible share")
    for yi, fv in zip(y, frac):
        ax2.text(min(fv + 0.02, 0.98), yi, f"{fv:.2f}", va="center", fontsize=8)
    ax2.grid(axis="x", alpha=0.25); ax2.grid(axis="y", visible=False)

    fig.tight_layout()
    C.savefig(fig, "uncertainty_decomposition.pdf")
    plt.close(fig)


def main():
    C._plt = C.apply_style()
    print("Figure 6: epistemic/aleatoric decomposition")
    make()


if __name__ == "__main__":
    main()
