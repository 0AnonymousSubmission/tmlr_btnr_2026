#!/usr/bin/env python3
"""Figure 4 -- Average-rank summary across datasets.
"""

import numpy as np
import unc_common as C

# ---- CONFIG ----------------------------------------------------------------
METRICS = [
    "unc_nll", "unc_crps", "unc_ece", "unc_ause",
    "unc_spearman_err", "unc_outlier_auroc_nll", "unc_coverage_error",
]


def _rank_table():
    """Return ranks[model][metric] = list of per-dataset ranks (1=best)."""
    ranks = {k: {m: [] for m in METRICS} for k in C.MODEL_ORDER}
    for m in METRICS:
        info = C.METRIC_INFO[m]
        for ds in C.DATASET_ORDER:
            entries = []  # (score_lower_is_better, model_key)
            for key, mean, std, n, fam in C.models_for_metric(ds, m):
                entries.append((C._score_for_direction(mean, info), key))
            if len(entries) < 2:
                continue
            entries.sort(key=lambda t: t[0])
            # average ranks for ties
            scores = [s for s, _ in entries]
            for pos, (s, key) in enumerate(entries):
                tied = [i for i, (ss, _) in enumerate(entries) if ss == s]
                rank = np.mean([i + 1 for i in tied])
                ranks[key][m].append(rank)
    return ranks


def _mean_rank_matrix(ranks):
    present = [k for k in C.MODEL_ORDER
               if any(ranks[k][m] for m in METRICS)]
    M = np.full((len(present), len(METRICS)), np.nan)
    S = np.full_like(M, np.nan)
    for i, k in enumerate(present):
        for j, m in enumerate(METRICS):
            if ranks[k][m]:
                M[i, j] = np.mean(ranks[k][m])
                S[i, j] = np.std(ranks[k][m])
    return present, M, S


def make_heatmap():
    plt = C._plt
    ranks = _rank_table()
    models, M, S = _mean_rank_matrix(ranks)
    labels = [C.METRIC_INFO[m]["label"] for m in METRICS]

    fig, ax = plt.subplots(figsize=(1.0 * len(METRICS) + 2.5, 0.62 * len(models) + 1.6))
    im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=len(models))

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([C.display_name(k) for k in models])
    ax.grid(False)

    for i in range(len(models)):
        for j in range(len(METRICS)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        fontsize=9, color="black",
                        fontweight="bold" if models[i] == "BTN" else "normal")
    # highlight BTN row
    if "BTN" in models:
        r = models.index("BTN")
        ax.add_patch(plt.Rectangle((-0.5, r - 0.5), len(METRICS), 1, fill=False,
                                   edgecolor=C.COLORS["BTN"], lw=C.STYLE["box_lw"]))
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Mean rank (1 is the best)")
    ax.set_title("Mean rank across 10 datasets  (green is better)")
    fig.tight_layout()
    C.savefig(fig, "rank_heatmap.pdf")
    plt.close(fig)


def make_dotplot():
    plt = C._plt
    ranks = _rank_table()
    models, M, S = _mean_rank_matrix(ranks)
    ncol = len(METRICS)
    fig, axes = plt.subplots(1, ncol, figsize=(2.0 * ncol, 3.0), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    order = list(range(len(models)))
    ypos = np.arange(len(models))[::-1]  # BTN-first on top

    for j, m in enumerate(METRICS):
        ax = axes[j]
        for i, k in enumerate(models):
            if not np.isfinite(M[i, j]):
                continue
            st = C.style_for(k, "marker")
            ax.errorbar(M[i, j], ypos[i], xerr=S[i, j], fmt=C.MARKERS[k],
                        color=C.COLORS[k], capsize=C.STYLE["capsize"],
                        elinewidth=C.STYLE["errorbar_lw"], **st)
        ax.set_title(C.METRIC_INFO[m]["label"], fontsize=9)
        ax.set_xlim(0.5, len(models) + 0.5)
        ax.set_xlabel("rank")
        ax.invert_xaxis()  # left = better (rank 1)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([C.display_name(k) for k in models])
    fig.suptitle("Average rank per metric  (left is better)", y=1.02, fontsize=12)
    fig.tight_layout()
    C.savefig(fig, "rank_dotplot.pdf")
    plt.close(fig)


def main():
    C._plt = C.apply_style()
    print("Figure 4: rank diagrams")
    make_heatmap()
    make_dotplot()


if __name__ == "__main__":
    main()
