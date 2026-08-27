#!/usr/bin/env python3
"""Generate the LaTeX outlier-detection table (averaged across datasets).

Synthetic outliers are injected into the test set and each model must flag them
using an anomaly score derived from its predictive distribution (the predictive
standard deviation or the per-point NLL). We report the area under the ROC curve
and the area under the precision-recall curve for both scores.

Because the per-dataset table is unreadable (ten dataset columns), this table is
compact: models are rows and metrics are columns, and each cell is the AVERAGE
across the ten datasets, i.e. the mean over datasets of the per-dataset seed mean,
plus or minus the standard deviation across datasets. Best per metric column is
bolded according to the metric direction (higher AUROC/AUPR is better).

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

# Caption read verbatim from captions/unc_outlier_caption.tex.
CAPTION_NAME = "unc_outlier"
CAPTION_TRAILING_PERCENT = False
# Width passed to \resizebox (the target scales this table to 0.7\textwidth).
RESIZE_WIDTH = r"0.7\textwidth"
LABEL = "tab:outlier"


# A "row spec" is (row_key, label) where row_key is either a baseline key or
# the BTN selector tag "BTN" (the val-quality-best family per dataset).
def _row_specs():
    btn = [("BTN", C.BTN_DISPLAY)]
    return btn + [(b, C.display_name(b)) for b in C.BASELINE_ORDER]


def _model_dataset_mean(row_key, ds, metric):
    """Per-dataset seed mean for a row_key on dataset `ds`, or None."""
    if row_key == "BTN":
        vb = C.valbest_btn_metric(ds, metric)
        a = (vb[1], vb[2], vb[3]) if vb else None
    else:
        a = C.agg(C.scalar_values("baseline", ds, row_key, metric))
    return a[0] if a else None


def _avg_over_datasets(row_key, metric):
    """(mean, std) across datasets of the per-dataset seed means, or None."""
    vals = [v for ds in C.DATASET_ORDER
            if (v := _model_dataset_mean(row_key, ds, metric)) is not None]
    if not vals:
        return None
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def format_table():
    fmt = f"{{:.{DECIMALS}f}}"
    specs = _row_specs()
    n_btn = 1
    models = [rk for rk, _ in specs]
    labels = {rk: lbl for rk, lbl in specs}

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
        if ki == n_btn:                     # after the BTN row(s)
            out.append(r"\midrule\midrule")
        elif ki > n_btn or (0 < ki < n_btn):
            out.append(r"\midrule")
        row = [labels[k]] + [cell(k, m) for m, _ in METRIC_COLUMNS]
        out.append(" & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(out)
    if RESIZE_TO_TEXTWIDTH:
        body = "\\resizebox{" + RESIZE_WIDTH + "}{!}{%\n" + body + "\n}"

    label = LABEL
    wrapped = [r"\begin{table}[t]", r"\centering",
               C.caption(CAPTION_NAME, trailing_percent=CAPTION_TRAILING_PERCENT),
               r"\label{" + label + "}", body, r"\end{table}"]
    return "\n".join(wrapped)


def main(variant=None):
    os.makedirs(C.TABLES_DIR, exist_ok=True)
    print("Outlier table: [valbest]")
    tex = format_table()
    path = os.path.join(C.TABLES_DIR, "unc_outlier_table.tex")
    with open(path, "w") as f:
        f.write(tex + "\n")
    print(f"  wrote {os.path.relpath(path, C._REPO_ROOT)}")


if __name__ == "__main__":
    main()
