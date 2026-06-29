#!/usr/bin/env python3
"""Generate the LaTeX discrimination table (AUSE and Spearman).
Output (into paper_script/tables/):
  unc_discrimination_table.tex
"""

import os
import unc_common as C

# ---- CONFIG ----------------------------------------------------------------
# (metric key, decimals, block heading shown as a full-width subheader row)
BLOCK_METRICS = [
    ("unc_ause", 3, r"AUSE"),
    ("unc_spearman_err", 3, r"Spearman"),
]
PM = r"$\pm$"
MISSING = "--"
CELL = r"\shortstack{{{mean} \\ {pm} {std}}}"
CELL_BOLD = r"\shortstack{{\textbf{{{mean}}} \\ {pm} {std}}}"
ROW_LABEL = r"\shortstack[l]{{{label} \\ {{}}}}"
ROW_RULE = r"\midrule"
BLOCK_RULE = r"\midrule\midrule"
RESIZE_TO_TEXTWIDTH = True

CAPTION = (r"Discrimination metrics, mean \(\pm\) std over seeds: the area under the "
           r"sparsification error (AUSE, lower is better) and the Spearman correlation "
           r"between predicted uncertainty and absolute error (higher is better). "
           r"BTN is its best configuration per dataset and metric. Best per dataset "
           r"within each block is in bold.")
LABEL = "tab:discrimination"


def _block_rows(metric):
    """rows: list of (label, {dataset: (mean,std) or None}) for one metric."""
    rows = []
    btn_cells = {}
    for ds in C.DATASET_ORDER:
        bf = C.best_btn_family(ds, metric)
        btn_cells[ds] = (bf[1], bf[2]) if bf else None
    rows.append((C.BTN_DISPLAY, btn_cells))
    for b in C.BASELINE_ORDER:
        cells = {}
        present = False
        for ds in C.DATASET_ORDER:
            a = C.agg(C.scalar_values("baseline", ds, b, metric))
            cells[ds] = (a[0], a[1]) if a else None
            present = present or (a is not None)
        if present:
            rows.append((C.display_name(b), cells))
    return rows


def _best_per_dataset(rows, metric, fmt):
    info = C.METRIC_INFO[metric]
    best = {}
    for ds in C.DATASET_ORDER:
        cands = [(C._score_for_direction(c[ds][0], info), fmt.format(c[ds][0]))
                 for _, c in rows if c[ds] is not None]
        best[ds] = min(cands)[1] if cands else None
    return best


def format_table():
    n = len(C.DATASET_ORDER)
    colspec = "l" + "c" * n
    out = [r"\begin{tabular}{" + colspec + "}", r"\toprule"]
    header = ["Model"] + [C.DATASET_DISPLAY[d] for d in C.DATASET_ORDER]
    out.append(" & ".join(header) + r" \\")

    for bi, (metric, dec, heading) in enumerate(BLOCK_METRICS):
        fmt = f"{{:.{dec}f}}"
        rows = _block_rows(metric)
        best = _best_per_dataset(rows, metric, fmt)

        def cell(ds, agg):
            if agg is None:
                return MISSING
            m, s = agg
            mstr = fmt.format(m)
            sstr = fmt.format(s)
            is_best = best[ds] is not None and float(mstr) == float(best[ds])
            tmpl = CELL_BOLD if is_best else CELL
            return tmpl.format(mean=mstr, std=sstr, pm=PM)

        out.append(BLOCK_RULE if bi else ROW_RULE)
        out.append(r"\multicolumn{" + str(n + 1) + r"}{l}{\textit{" + heading + r"}} \\")
        out.append(ROW_RULE)
        for ri, (label, cells) in enumerate(rows):
            if ri == 1:           # after BTN, before the first baseline
                out.append(BLOCK_RULE)
            elif ri > 1:          # between consecutive baselines
                out.append(ROW_RULE)
            rl = ROW_LABEL.format(label=label)
            out.append(" & ".join([rl] + [cell(ds, cells[ds]) for ds in C.DATASET_ORDER]) + r" \\")

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
    print("Discrimination table:")
    tex = format_table()
    path = os.path.join(C.TABLES_DIR, "unc_discrimination_table.tex")
    with open(path, "w") as f:
        f.write(tex + "\n")
    print(f"  wrote {os.path.relpath(path, C._REPO_ROOT)}")


if __name__ == "__main__":
    main()
