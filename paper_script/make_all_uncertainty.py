#!/usr/bin/env python3
"""Driver: regenerate ALL uncertainty figures and tables in one shot.
Figures (PDF) -> paper_script/images/uncertainty/
Tables (LaTeX) -> paper_script/tables/
"""

import fig_reliability
import fig_sparsification
import fig_calibration_sharpness
import fig_rank_diagram
import fig_picp
import fig_decomposition
import make_uncertainty_tables
import make_discrimination_table
import make_outlier_table


def main():
    for mod in (fig_reliability, fig_sparsification, fig_calibration_sharpness,
                fig_rank_diagram, fig_picp, fig_decomposition,
                make_uncertainty_tables, make_discrimination_table,
                make_outlier_table):
        mod.main()
    print("\nAll uncertainty figures and tables regenerated.")


if __name__ == "__main__":
    main()
