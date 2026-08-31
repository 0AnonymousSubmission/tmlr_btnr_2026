#!/usr/bin/env python3
"""Generate a LaTeX table of test R^2 (x100) at minimum validation loss.

Rows are grouped into three blocks (BTN, ALS, baseline), separated by a double
midrule. Columns are datasets. Missing model x dataset combinations show "-".
The best (highest) mean value per dataset column is bolded.

All naming / style / behaviour choices are controlled by the variables in the
CONFIG section at the top of this file.
"""

import os
import glob
import json
import math
import statistics

# Robust JSON loader (repairs the known '--' number corruption in memory,
# without ever modifying the pristine run files on disk) + external caption loader.
from unc_common import load_run_json, caption

# ============================================================================
# CONFIG  (edit everything here)
# ============================================================================

# Paths are resolved relative to this script's location, so it can be run from
# anywhere. The script lives in <repo>/paper_script/ ; runs are in <repo>/.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)

ROOT = os.path.join(_REPO_ROOT, "tests_uncertainty_runs")

# Order of the three blocks and which directory each maps to.
GROUP_DIRS = {
    "BTN": "BTN",
    "ALS": "ALS",
    "baseline": "baseline",
}
GROUP_ORDER = ["BTN", "ALS", "baseline"]

# --- Datasets (columns) ------------------------------------------------------
# Order of dataset columns + their display (header) names.
DATASET_ORDER = [
    "abalone",
    "ai4i",
    "appliances",
    "bike",
    "concrete",
    "energy_efficiency",
    "obesity",
    "realstate",
    "seoulBike",
    "student_perf",
]
DATASET_DISPLAY = {
    "abalone": "AB",
    "ai4i": "AI",
    "appliances": "AP",
    "bike": "BK",
    "concrete": "CO",
    "energy_efficiency": "EE",
    "obesity": "OB",
    "realstate": "RS",
    "seoulBike": "SB",
    "student_perf": "SP",
}

# --- Model families (rows) ---------------------------------------------------
# BTN / ALS rows keyed by the family prefix found in the run directory name
# (e.g. "CPD_L3_d64" -> "CPD"). Display name added as a per-group prefix below.
TN_FAMILY_ORDER = ["CPD", "LMPO2", "MPO2", "BTT", "TR"]
TN_FAMILY_DISPLAY = {
    "CPD": "CPD",
    "LMPO2": "LMPO2",
    "MPO2": "MPS",
    "BTT": "BTT",
    "TR": "TR",
}
# Row-name prefix per group, e.g. BTN-CPD, A-CPD.
GROUP_ROW_PREFIX = {
    "BTN": "B-",
    "ALS": "A-",
}
# Collapsed-table row label per TN group: all families in a group are merged
# into one row, keeping the best (highest-mean) family per dataset.
GROUP_COLLAPSE_LABEL = {
    "BTN": "BTNR",
    "ALS": "ALS",
}

# Baseline rows. Each row has a short acronym (<= 5 letters) and a list of
# source directory names whose seed results are pooled into that row.
# (e.g. ExactGP + SparseGP are collected together into a single "GP" row.)
# NOTE: BayesianWideDeep ("BWD") is currently disabled (unstable: R2Score
# division-by-zero on ai4i, no config for bike). Re-add "BWD" to BASELINE_ORDER
# to bring it back.
BASELINE_ORDER = ["BDE", "HSBNN", "BASS", "GP"]
BASELINE_DISPLAY = {
    "BWD": "BWD",
    "BDE": "BDE",
    "HSBNN": "HSBNN",
    "BASS": "BASS",
    "GP": "GP",
}
BASELINE_SOURCES = {
    "BWD": ["BayesianWideDeep"],
    "BDE": ["BdeMile"],
    "HSBNN": ["HorseshoeBNN"],
    "BASS": ["MvBayes"],
    "GP": ["ExactGP", "SparseGP"],
}

# --- Metric extraction -------------------------------------------------------
VAL_LOSS_KEY = "val_loss"          # key in metrics_log used to pick the best step
TEST_QUALITY_KEY = "test_quality"  # key in metrics_log read at the best step
BASELINE_TEST_KEY = "test_r2"      # baselines: read directly from summary
SCALE = 100.0                      # report R^2 * 100

# --- Formatting / style ------------------------------------------------------
DECIMALS = 1                       # decimals for mean and std
PM = r"$\pm$"                      # plus/minus symbol
MISSING = "--"                      # placeholder for missing combos
BOLD_BEST = True                   # bold the best mean per dataset column
ROW_RULE = r"\midrule"             # single rule drawn between consecutive rows
BLOCK_RULE = r"\midrule\midrule"   # double rule drawn between BTN/ALS/baseline blocks
# Cell template; {mean} and {std} are pre-formatted strings.
# std is stacked below the mean via \shortstack.
CELL_TEMPLATE = r"\shortstack{{{mean} \\ {pm} {std}}}"
BOLD_TEMPLATE = r"\shortstack{{\textbf{{{mean}}} \\ {pm} {std}}}"
# Row label: stacked with an empty second line so the model name aligns with
# the mean (top line) rather than centering across mean+std.
ROW_LABEL_TEMPLATE = r"\shortstack[l]{{{label} \\ {{}}}}"

# Column spec for the data columns (one per dataset).
DATA_COLSPEC = "c"
ROW_LABEL_COLSPEC = "l"

# Overflow control: with 10 dataset columns the raw tabular is wider than
# \textwidth. Wrapping in \resizebox guarantees it fits the page width.
RESIZE_TO_TEXTWIDTH = False  # wrap tabular in \resizebox{\textwidth}{!}{...}
COLUMN_SEP = None            # e.g. "3pt" to shrink \tabcolsep; None = leave default

# Caption: False -> no caption (tabular emitted bare). True -> wrap in a
# `table` environment with \caption (+ optional \label). The caption text lives
# in captions/<CAPTION_NAME>_caption.tex and is loaded verbatim.
CAPTION = True
CAPTION_NAME = "r2"          # -> captions/r2_caption.tex
CAPTION_TRAILING_PERCENT = True   # target emits \caption{...}%
LABEL = "tab:r2"             # e.g. "tab:r2"; ignored when empty
TABLE_POSITION = "t"         # float placement specifier, e.g. "t", "h", "htbp"

# Output
# Main table used by the paper: every TN family as its own row (B-CPD, A-CPD,
# ...). This is what tables/r2_table.tex must contain.
OUTPUT_PATH = os.path.join(_SCRIPT_DIR, "tables", "r2_table.tex")
# Collapsed variant: all B- families merged into one BTNR row, all A- families
# merged into one ALS row (best family per dataset), baselines kept as-is.
OUTPUT_PATH_COLLAPSED = os.path.join(_SCRIPT_DIR, "tables", "r2_table_collapsed.tex")
PRINT_TO_STDOUT = False


# ============================================================================
# Core logic
# ============================================================================

def _seed_json_files(model_dir):
    """All per-seed json result files under a model directory."""
    return sorted(glob.glob(os.path.join(model_dir, "*", "*", "*.json")))


def _r2_from_tn_json(d):
    """R^2 (test_quality) at the step of minimum validation loss."""
    ml = d.get("metrics_log") or []
    valid = [r for r in ml if r.get(VAL_LOSS_KEY) is not None
             and math.isfinite(r[VAL_LOSS_KEY])]
    if not valid:
        return None
    best = min(valid, key=lambda r: r[VAL_LOSS_KEY])
    v = best.get(TEST_QUALITY_KEY)
    if v is None or not math.isfinite(v):
        return None
    return v


def _r2_from_baseline_json(d):
    """Baselines: test R^2 from the summary (model already early-stopped on val)."""
    v = d.get("summary", {}).get(BASELINE_TEST_KEY)
    if v is None or not math.isfinite(v):
        return None
    return v


def collect_values(group, dataset, family):
    """Return list of per-seed R^2*SCALE values, or None if combo missing."""
    group_dir = os.path.join(ROOT, GROUP_DIRS[group], dataset)
    if not os.path.isdir(group_dir):
        return None

    if group == "baseline":
        # `family` is a baseline row key; pool all its source directories.
        model_dirs = [
            os.path.join(group_dir, src)
            for src in BASELINE_SOURCES[family]
            if os.path.isdir(os.path.join(group_dir, src))
        ]
        if not model_dirs:
            return None
        extractor = _r2_from_baseline_json
    else:
        # Match family prefix, e.g. "CPD" -> "CPD_L3_d64".
        model_dirs = [
            os.path.join(group_dir, m)
            for m in os.listdir(group_dir)
            if m.split("_")[0] == family
            and os.path.isdir(os.path.join(group_dir, m))
        ]
        if not model_dirs:
            return None
        extractor = _r2_from_tn_json

    values = []
    for md in model_dirs:
        for f in _seed_json_files(md):
            d = load_run_json(f)
            if d is None:
                continue
            v = extractor(d)
            if v is not None:
                values.append(v * SCALE)
    return values if values else None


def aggregate(values):
    """Return (mean, std) or None."""
    if not values:
        return None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def build_rows():
    """Return list of (block, row_label, {dataset: (mean,std) or None})."""
    rows = []
    for group in GROUP_ORDER:
        if group == "baseline":
            families = BASELINE_ORDER
            display = BASELINE_DISPLAY
            prefix = ""
        else:
            families = TN_FAMILY_ORDER
            display = TN_FAMILY_DISPLAY
            prefix = GROUP_ROW_PREFIX[group]

        for fam in families:
            label = prefix + display[fam]
            cells = {}
            present = False
            for ds in DATASET_ORDER:
                vals = collect_values(group, ds, fam)
                agg = aggregate(vals) if vals is not None else None
                cells[ds] = agg
                if agg is not None:
                    present = True
            if present:  # skip entirely-missing rows
                rows.append((group, label, cells))
    return rows


def build_collapsed_rows():
    """Like build_rows(), but merge all TN families of a group into one row.

    For each TN group (BTN, ALS) and each dataset, keep the (mean, std) of the
    family with the highest mean R^2. Baseline rows are emitted unchanged.
    """
    rows = []
    for group in GROUP_ORDER:
        if group == "baseline":
            for fam in BASELINE_ORDER:
                cells = {}
                present = False
                for ds in DATASET_ORDER:
                    vals = collect_values(group, ds, fam)
                    agg = aggregate(vals) if vals is not None else None
                    cells[ds] = agg
                    if agg is not None:
                        present = True
                if present:
                    rows.append((group, BASELINE_DISPLAY[fam], cells))
            continue

        # TN group: pick, per dataset, the family with the highest mean.
        label = GROUP_COLLAPSE_LABEL[group]
        cells = {}
        present = False
        for ds in DATASET_ORDER:
            best_agg = None
            for fam in TN_FAMILY_ORDER:
                vals = collect_values(group, ds, fam)
                agg = aggregate(vals) if vals is not None else None
                if agg is None:
                    continue
                if best_agg is None or agg[0] > best_agg[0]:
                    best_agg = agg
            cells[ds] = best_agg
            if best_agg is not None:
                present = True
        if present:
            rows.append((group, label, cells))
    return rows


def format_table(rows):
    fmt = f"{{:.{DECIMALS}f}}"

    # Best per dataset column, computed on the DISPLAYED (rounded) means so that
    # all cells sharing the best rounded value get bolded.
    def disp(mean):
        return fmt.format(mean)

    best_disp = {}
    for ds in DATASET_ORDER:
        shown = [disp(cells[ds][0]) for _, _, cells in rows if cells[ds] is not None]
        # Pick the display string with the maximum numeric value.
        best_disp[ds] = max(shown, key=float) if shown else None

    def cell_str(ds, agg):
        if agg is None:
            return MISSING
        mean, std = agg
        mstr = disp(mean)
        sstr = fmt.format(std)
        is_best = (BOLD_BEST and best_disp[ds] is not None
                   and float(mstr) == float(best_disp[ds]))
        tmpl = BOLD_TEMPLATE if is_best else CELL_TEMPLATE
        return tmpl.format(mean=mstr, std=sstr, pm=PM)

    n_data = len(DATASET_ORDER)
    colspec = ROW_LABEL_COLSPEC + DATA_COLSPEC * n_data

    lines = []
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\toprule")

    header = ["Model"] + [DATASET_DISPLAY[ds] for ds in DATASET_ORDER]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    prev_block = None
    for block, label, cells in rows:
        if prev_block is not None:
            # double rule between blocks, single rule between rows of a block
            lines.append(BLOCK_RULE if block != prev_block else ROW_RULE)
        row_label = ROW_LABEL_TEMPLATE.format(label=label)
        row_cells = [row_label] + [cell_str(ds, cells[ds]) for ds in DATASET_ORDER]
        lines.append(" & ".join(row_cells) + r" \\")
        prev_block = block

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    body = "\n".join(lines)
    if RESIZE_TO_TEXTWIDTH:
        body = "\\resizebox{\\textwidth}{!}{%\n" + body + "\n}"
    if COLUMN_SEP is not None:
        body = ("\\setlength{\\tabcolsep}{" + COLUMN_SEP + "}\n") + body

    if CAPTION:
        wrapped = [r"\begin{table}[" + TABLE_POSITION + "]", r"\centering",
                   caption(CAPTION_NAME, trailing_percent=CAPTION_TRAILING_PERCENT)]
        if LABEL:
            wrapped.append(r"\label{" + LABEL + "}")
        wrapped.append(body)
        wrapped.append(r"\end{table}")
        body = "\n".join(wrapped)
    return body


def main(variant=None):
    # variant is accepted for a uniform driver interface but unused: the R2
    # table does not depend on the BTN aggregation variant.
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Main table (used by the paper): every TN family as its own row.
    full_rows = build_rows()
    full_table = format_table(full_rows)
    with open(OUTPUT_PATH, "w") as f:
        f.write(full_table + "\n")

    if PRINT_TO_STDOUT:
        print(full_table)

    print(f"\n% written to {OUTPUT_PATH}", flush=True)
    # print(f"% written to {OUTPUT_PATH_COLLAPSED}", flush=True)


if __name__ == "__main__":
    main()
