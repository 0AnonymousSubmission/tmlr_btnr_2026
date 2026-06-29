#!/usr/bin/env python3
"""Generate LaTeX tables of headline uncertainty metrics.
Outputs (into paper_script/tables/):
  unc_nll_table.tex, unc_crps_table.tex, unc_ece_table.tex
"""

import os
import unc_common as C

# ---- CONFIG ----------------------------------------------------------------
TABLE_METRICS = ["unc_nll", "unc_crps", "unc_ece"]
DECIMALS = {"unc_nll": 2, "unc_crps": 2, "unc_ece": 3}
PM = r"$\pm$"
MISSING = "--"
CELL = r"\shortstack{{{mean} \\ {pm} {std}}}"
CELL_BOLD = r"\shortstack{{\textbf{{{mean}}} \\ {pm} {std}}}"
ROW_LABEL = r"\shortstack[l]{{{label} \\ {{}}}}"
ROW_RULE = r"\midrule"
BLOCK_RULE = r"\midrule\midrule"
# Wrap tabular in \resizebox{\textwidth}{!}{...}. Off by default to follow the
# paper's table style (no font scaling); enable only if a table overflows.
RESIZE_TO_TEXTWIDTH = True

CAPTIONS = {
    "unc_nll":  r"Test negative log-likelihood (NLL) (lower is better), mean \(\pm\) std. Best per dataset in bold.",
    "unc_crps": r"Test continuous ranked probability score (CRPS) (lower is better), mean \(\pm\) std. Best per dataset in bold.",
    "unc_ece":  r"Expected calibration error (ECE) (lower is better), mean \(\pm\) std. Best per dataset in bold.",
}
LABELS = {"unc_nll": "tab:nll", "unc_crps": "tab:crps", "unc_ece": "tab:ece"}


def build_rows(metric):
    """rows: list of (block, label, {dataset: (mean,std) or None})."""
    rows = []
    # BTN row (best family per dataset)
    btn_cells = {}
    for ds in C.DATASET_ORDER:
        bf = C.best_btn_family(ds, metric)
        btn_cells[ds] = (bf[1], bf[2]) if bf else None
    rows.append(("BTN", C.BTN_DISPLAY, btn_cells))
    # baseline rows
    for b in C.BASELINE_ORDER:
        cells = {}
        present = False
        for ds in C.DATASET_ORDER:
            a = C.agg(C.scalar_values("baseline", ds, b, metric))
            cells[ds] = (a[0], a[1]) if a else None
            present = present or (a is not None)
        if present:
            rows.append(("baseline", C.display_name(b), cells))
    return rows


def _best_per_dataset(rows, metric, fmt):
    info = C.METRIC_INFO[metric]
    best = {}
    for ds in C.DATASET_ORDER:
        cands = [(C._score_for_direction(c[ds][0], info), fmt.format(c[ds][0]))
                 for _, _, c in rows if c[ds] is not None]
        best[ds] = min(cands)[1] if cands else None
    return best


def format_table(metric):
    dec = DECIMALS[metric]
    fmt = f"{{:.{dec}f}}"
    rows = build_rows(metric)
    best = _best_per_dataset(rows, metric, fmt)

    def cell(ds, agg):
        if agg is None:
            return MISSING
        m, s = agg
        mstr = fmt.format(m); sstr = fmt.format(s)
        is_best = best[ds] is not None and float(mstr) == float(best[ds])
        tmpl = CELL_BOLD if is_best else CELL
        return tmpl.format(mean=mstr, std=sstr, pm=PM)

    n = len(C.DATASET_ORDER)
    colspec = "l" + "c" * n
    out = [r"\begin{tabular}{" + colspec + "}", r"\toprule"]
    header = ["Model"] + [C.DATASET_DISPLAY[d] for d in C.DATASET_ORDER]
    out.append(" & ".join(header) + r" \\")
    out.append(ROW_RULE)
    prev = None
    for block, label, cells in rows:
        if prev is not None:
            out.append(BLOCK_RULE if block != prev else ROW_RULE)
        rl = ROW_LABEL.format(label=label)
        out.append(" & ".join([rl] + [cell(ds, cells[ds]) for ds in C.DATASET_ORDER]) + r" \\")
        prev = block
    out += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(out)
    if RESIZE_TO_TEXTWIDTH:
        body = "\\resizebox{\\textwidth}{!}{%\n" + body + "\n}"

    wrapped = [r"\begin{table}[t]", r"\centering",
               r"\caption{" + CAPTIONS[metric] + "}",
               r"\label{" + LABELS[metric] + "}", body, r"\end{table}"]
    return "\n".join(wrapped)


def main():
    os.makedirs(C.TABLES_DIR, exist_ok=True)
    print("Uncertainty tables:")
    for m in TABLE_METRICS:
        tex = format_table(m)
        path = os.path.join(C.TABLES_DIR, m + "_table.tex")
        with open(path, "w") as f:
            f.write(tex + "\n")
        print(f"  wrote {os.path.relpath(path, C._REPO_ROOT)}")


if __name__ == "__main__":
    main()
