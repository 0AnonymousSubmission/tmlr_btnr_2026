#!/usr/bin/env python3
"""Driver: regenerate ALL uncertainty figures and tables in one shot.

Run from anywhere:
    python paper_script/make_all_uncertainty.py

For every dataset, the BTN row ("BTNR") is the single BTN family that achieves
the best validation predictive quality (`val_quality`) on that dataset; that
family's uncertainty metrics are what get reported/plotted. Files are canonical
and UNSUFFIXED (e.g. reliability_grid.pdf, unc_ece_table.tex).

Figures (PDF) -> paper_script/images/uncertainty/
Tables (LaTeX) -> paper_script/tables/
"""

import fig_reliability
import fig_sparsification
import fig_calibration_sharpness
import fig_picp
import fig_decomposition
import make_r2_table
import make_uncertainty_tables
import make_discrimination_table
import make_outlier_table


def main():
    for mod in (fig_reliability, fig_sparsification, fig_calibration_sharpness,
                fig_picp, fig_decomposition,
                make_r2_table, make_uncertainty_tables, make_discrimination_table,
                make_outlier_table):
        mod.main()
    print("\nAll uncertainty figures and tables regenerated [valbest].")


if __name__ == "__main__":
    main()
