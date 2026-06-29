#!/usr/bin/env python3
"""Generate the LaTeX outlier-detection table (averaged across datasets).
Output (into paper_script/tables/):
  unc_outlier_table.tex
"""

import os
import statistics
import unc_common as C

# ---- CONFIG ----------------------------------------------------------------
# (metric key, column header)
METRIC_COLUMNS = [
    ("unc_outlier_auroc_std", r"\shortstack{AUROC\\(std)}"),
    ("unc_outlier_aupr_std",  r"\shortstack{AUPR\\(std)}"),
    ("unc_outlier_auroc_nll", r"\shortstack{AUROC\\(NLL)}"),
    ("unc_outlier_aupr_nll",  r"\shortstack{AUPR\\(NLL)}"),
]
DECIMALS = 3
PM = r"$\pm$"
MISSING = "--"
RESIZE_TO_TEXTWIDTH = True

CAPTION = (r"Outlier detection averaged across the ten datasets. "
           r"Synthetic outliers are flagged using the predictive standard "
           r"deviation (std) or the per-point NLL as the anomaly score. We report "
           r"the area under the ROC curve (AUROC) and the area under the "
           r"precision--recall curve (AUPR), where higher is better. Each cell is the "
           r"mean over datasets of the per-dataset seed mean \(\pm\) the standard "
           r"deviation across datasets. BTN is its best configuration per dataset "
           r"and metric. Best per column is in bold.")
LABEL = "tab:outlier"


def _model_dataset_mean(k, ds, metric):
    """Per-dataset seed mean for model `k` on dataset `ds`, or None."""
    if k == "BTN":
        bf = C.best_btn_family(ds, metric)
        a = C.agg(C.scalar_values("BTN", ds, bf[0], metric)) if bf else None
    else:
        a = C.agg(C.scalar_values("baseline", ds, k, metric))
    return a[0] if a else None


def _avg_over_datasets(k, metric):
    """(mean, std) across datasets of the per-dataset seed means, or None."""
    vals = [v for ds in C.DATASET_ORDER
            if (v := _model_dataset_mean(k, ds, metric)) is not None]
    if not vals:
        return None
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def format_table():
    fmt = f"{{:.{DECIMALS}f}}"
    models = list(C.MODEL_ORDER)

    # cell values: cells[model][metric] = (mean, std) or None
    cells = {k: {} for k in models}
    for k in models:
        for metric, _ in METRIC_COLUMNS:
            cells[k][metric] = _avg_over_datasets(k, metric)

    # best per metric column (by direction)
    best = {}
    for metric, _ in METRIC_COLUMNS:
        info = C.METRIC_INFO[metric]
        cands = [(C._score_for_direction(cells[k][metric][0], info),
                  fmt.format(cells[k][metric][0]))
                 for k in models if cells[k][metric] is not None]
        best[metric] = min(cands)[1] if cands else None

    def cell(k, metric):
        agg = cells[k][metric]
        if agg is None:
            return MISSING
        m, s = agg
        mstr = fmt.format(m)
        sstr = fmt.format(s)
        is_best = best[metric] is not None and float(mstr) == float(best[metric])
        body = (r"\textbf{" + mstr + "}") if is_best else mstr
        return f"{body} {PM} {sstr}"

    ncol = len(METRIC_COLUMNS)
    colspec = "l" + "c" * ncol
    out = [r"\begin{tabular}{" + colspec + "}", r"\toprule"]
    header = ["Model"] + [h for _, h in METRIC_COLUMNS]
    out.append(" & ".join(header) + r" \\")
    out.append(r"\midrule")
    for ki, k in enumerate(models):
        if ki == 1:
            out.append(r"\midrule\midrule")
        elif ki > 1:
            out.append(r"\midrule")
        row = [C.display_name(k)] + [cell(k, m) for m, _ in METRIC_COLUMNS]
        out.append(" & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(out)
    if RESIZE_TO_TEXTWIDTH:
        body = "\\resizebox{\\textwidth}{!}{%\n" + body + "\n}"

    wrapped = [r"\begin{table}[t]", r"\centering",
               r"\caption{" + CAPTION + "}",
               r"\label{" + LABEL + "}", body, r"\end{table}"]
    return "\n".join(wrapped)


def main():
    os.makedirs(C.TABLES_DIR, exist_ok=True)
    print("Outlier table:")
    tex = format_table()
    path = os.path.join(C.TABLES_DIR, "unc_outlier_table.tex")
    with open(path, "w") as f:
        f.write(tex + "\n")
    print(f"  wrote {os.path.relpath(path, C._REPO_ROOT)}")


if __name__ == "__main__":
    main()
