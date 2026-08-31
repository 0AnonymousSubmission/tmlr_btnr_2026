#!/usr/bin/env python3
"""Figure 6 -- BTN uncertainty decomposition (epistemic vs aleatoric).

A unique capability of BTN: it splits predictive uncertainty into
  * epistemic (reducible, model/data-limited) -- unc_epistemic_std
  * aleatoric (irreducible data noise)         -- unc_aleatoric_std
Baselines in this study do NOT report this decomposition (they store only a
single total predictive std, ``unc_sharpness``), so the split figure is
BTN-only and illustrative of an additional benefit.

Design: horizontal stacked bars of VARIANCE (aleatoric + epistemic). We plot
variances, not standard deviations, because variances of independent components
ADD -- ``Var_total = Var_ale + Var_epi`` -- whereas standard deviations combine
in quadrature (``std_total = sqrt(std_ale^2 + std_epi^2)``). Stacking stds would
therefore overstate the total (the bar could exceed the true predictive std) and
break the ``= target variance`` reference line; stacking variances is exact, so
the stacked length equals the true predictive variance and stays <= 1.0 whenever
the model is sharper than the marginal. ALL BTN families are shown per dataset as
grouped stacked bars, so the spread across architectures is visible directly.
Because the components live on the target's natural scale and datasets differ
wildly, we normalise by the target variance and additionally show the epistemic
FRACTION of variance as a second panel (scale-free), the genuinely comparable
quantity.

Because the baselines lack the split, a companion figure reports their TOTAL
predictive VARIANCE (``unc_sharpness``^2, normalised by target variance) as
single bars per dataset -- on the SAME axis as the BTN decomposition, so the two
figures can be read directly against each other (a BTN total-variance bar is
comparable, unit-for-unit, to a baseline bar). We do not invent an
epistemic/aleatoric division the baselines never computed.

Both figures use the SAME axis: predictive variance ÷ target variance, with a
1.0 reference at "variance == target variance".

Outputs:
  * uncertainty_decomposition.pdf          -- BTN epistemic/aleatoric variance split
  * uncertainty_sharpness_baselines.pdf    -- baseline total variance
"""

import numpy as np
import unc_common as C

# ---- CONFIG ----------------------------------------------------------------
EPI = "unc_epistemic_std"
ALE = "unc_aleatoric_std"
ALE_COLOR = "#4c72b0"
EPI_COLOR = "#dd8452"
# One hatch per BTN family so families are distinguishable within a dataset
# group even in greyscale; colours still encode aleatoric vs epistemic.
FAMILY_HATCH = {
    "CPD":   "",
    "LMPO2": "///",
    "MPO2":  "...",
    "BTT":   "xxx",
    "TR":    "\\\\\\",
}
GROUP_PAD = 0.12             # fraction of each dataset slot used by the bars gap
# Vertical pitch between consecutive dataset groups (in slot units). >1 leaves a
# small gap between groups so the on-top dataset header has room.
SLOT_PITCH = 1.27
# Alternating background band behind every other dataset group (readability).
BAND_COLOR = "0.92"          # light gray for the shaded groups
BAND_ALPHA = 1.0
HEADER_FONTSIZE = 9
TICK_FONTSIZE = 7
# Dataset label x-position in the LEFT margin (y-axis transform units; negative =
# outside the axes, i.e. in the same gutter as the model/family tick labels).
HEADER_X = -0.02
# The pipeline standardizes X but NOT y, so predictive variance is in raw target
# units^2 and is not comparable across datasets. Divide by var(y) so the x-axis
# reads as a fraction of the target's own variance: 1.0 == as much variance as
# the marginal (a mean-only predictor that learned nothing), <1 == genuinely
# more certain. Variance (not std) is used so the stacked ale+epi bar is exact.
NORMALIZE_BY_TARGET_VAR = True
BASELINE_LINE = 1.0          # "no-skill" reference (predictive var == var(y))


def _var_scale(dataset):
    """Divisor applied to variance components for a dataset (var(y), or 1.0)."""
    if not NORMALIZE_BY_TARGET_VAR:
        return 1.0
    s = C.target_std(dataset)
    return (s * s) if s else 1.0


def _family_var_component(dataset, family, std_metric):
    """Mean VARIANCE (std^2, ÷ var(y) if enabled) for a BTN family, or None.

    `std_metric` is a stored *std* metric (aleatoric/epistemic); we square each
    per-seed std into a variance before aggregating, then normalise by var(y).
    """
    if family is None:
        return None
    stds = C.scalar_values("BTN", dataset, family, std_metric)
    if not stds:
        return None
    a = C.agg([s * s for s in stds])
    return a[0] / _var_scale(dataset) if a else None


def _xlabel():
    return ("Predictive variance (÷ target variance)" if NORMALIZE_BY_TARGET_VAR
            else "Predictive variance (target units$^2$)")


def _slot_geometry(n_items):
    """Shared vertical geometry for a decomposition panel.

    Returns (y_centers, slot, h) where `y_centers[di]` is the centre of dataset
    group `di` (spaced by SLOT_PITCH so headers have room), `slot` is the usable
    height per group, and `h` is the per-bar height (`slot / n_items`).
    """
    def make(n_ds):
        y = np.arange(n_ds) * SLOT_PITCH
        slot = 1.0 - GROUP_PAD
        h = slot / n_items
        return y, slot, h
    return make


def _draw_dataset_bands(fig, ax_left, ax_right, y_centers, span):
    """Shade alternate dataset groups with a CONTINUOUS light-gray band.

    The band is drawn once at the FIGURE level, spanning horizontally from the
    left edge of `ax_left` to the right edge of `ax_right` (so it does not break
    in the gap between the two panels). Vertical extents come from each group's
    data y-range, converted to figure coordinates. Call AFTER the layout is
    finalised (tight_layout) so axes positions are correct. `ax_right` may be the
    same as `ax_left` for single-panel figures.
    """
    from matplotlib.patches import Rectangle
    fig.canvas.draw()                               # ensure positions are final
    inv = fig.transFigure.inverted()
    x0 = ax_left.get_position().x0
    x1 = ax_right.get_position().x1
    for di, yc in enumerate(y_centers):
        if di % 2 == 1:                             # shade alternate groups only
            continue
        # group's vertical extent (data) -> figure y via the left axis
        (_, yb) = inv.transform(ax_left.transData.transform((0, yc - span / 2)))
        (_, yt) = inv.transform(ax_left.transData.transform((0, yc + span / 2)))
        fig.add_artist(Rectangle((x0, yb), x1 - x0, yt - yb,
                                 transform=fig.transFigure, facecolor=BAND_COLOR,
                                 alpha=BAND_ALPHA, edgecolor="none", zorder=0))


def _draw_dataset_header(ax, ds, y_center, slot, h):
    """Dataset label OUTSIDE the plot on the left, in the tick-label gutter.

    Placed above the group's top bar and left-aligned at HEADER_X (which is <0,
    i.e. outside the left spine), so it lines up with the model/family tick
    labels rather than floating over the bars.
    """
    top = y_center + slot / 2                       # top edge of the bar group
    ax.text(HEADER_X, top + h * 0.20, C.DATASET_DISPLAY[ds],
            transform=ax.get_yaxis_transform(), va="bottom", ha="right",
            fontsize=HEADER_FONTSIZE, fontweight="bold", clip_on=False)


# ---------------------------------------------------------------------------
# All BTN families, grouped per dataset
# ---------------------------------------------------------------------------

def _make_all_families():
    plt = C._plt
    families = C.TN_FAMILY_ORDER

    ds_rows = []
    for ds in C.DATASET_ORDER:
        comps = {}
        for fam in families:
            a = _family_var_component(ds, fam, ALE)
            e = _family_var_component(ds, fam, EPI)
            if a is not None and e is not None:
                comps[fam] = (a, e)
        if comps:
            ds_rows.append((ds, comps))
    if not ds_rows:
        print("  no epistemic/aleatoric data found; skipping")
        return

    ds_rows.sort(key=lambda r: np.mean([a + e for a, e in r[1].values()]))
    n_ds = len(ds_rows)
    n_fam = len(families)

    y, slot, h = _slot_geometry(n_fam)(n_ds)
    fam_off = {fam: slot / 2 - (i + 0.5) * h for i, fam in enumerate(families)}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.0, 0.85 * n_ds * SLOT_PITCH + 1.8),
        gridspec_kw={"width_ratios": [2, 1]})

    ax1.patch.set_visible(False)
    ax2.patch.set_visible(False)

    tick_pos, tick_lab = [], []       # one y-tick per bar (family name)
    for di, (ds, comps) in enumerate(ds_rows):
        for fam in families:
            if fam not in comps:
                continue
            a, e = comps[fam]
            yy = y[di] + fam_off[fam]
            hatch = FAMILY_HATCH.get(fam, "")
            ax1.barh(yy, a, height=h * 0.92, color=ALE_COLOR,
                     hatch=hatch, edgecolor="white", linewidth=0.4)
            ax1.barh(yy, e, height=h * 0.92, left=a, color=EPI_COLOR,
                     hatch=hatch, edgecolor="white", linewidth=0.4)
            tick_pos.append(yy)
            tick_lab.append(C.family_display(fam))

            # rho_epi is defined on STANDARD DEVIATIONS, not variances:
            #   rho_epi = sigma_epi / (sigma_epi + sigma_ale).
            # `a` and `e` are variances (needed for the additive stacked bar on
            # ax1), so take sqrt here. Both were divided by the SAME var(y)
            # (=> sqrt(a) and sqrt(e) both carry a common 1/sqrt(var(y)) factor),
            # so that factor pulls out of the sum and cancels the ratio exactly:
            #   sqrt(e)/(sqrt(a)+sqrt(e)) == sqrt(v_e)/(sqrt(v_a)+sqrt(v_e)).
            # Hence no denormalisation is needed. (This ONLY holds because the
            # scale is common to both components; a per-component scale would
            # need explicit denormalisation before the sqrt.)
            s_ale, s_epi = np.sqrt(a), np.sqrt(e)
            frac = s_epi / (s_ale + s_epi)
            ax2.barh(yy, frac, height=h * 0.92, color=EPI_COLOR,
                     hatch=hatch, edgecolor="white", linewidth=0.4)
            # Always print the numeric fraction: near-zero bars (e.g. B-TR on
            # RS/CO, where epistemic collapses) are sub-pixel and would otherwise
            # look like missing data.
            ax2.text(min(frac + 0.02, 0.90), yy, f"{frac:.2f}",
                     va="center", fontsize=6)

        _draw_dataset_header(ax1, ds, y[di], slot, h)

    if NORMALIZE_BY_TARGET_VAR:
        ax1.axvline(BASELINE_LINE, color="0.4", ls="--", lw=1.0, zorder=5)
    # y-ticks = family names (one per bar); dataset names are the bold headers.
    ax1.set_yticks(tick_pos); ax1.set_yticklabels(tick_lab, fontsize=TICK_FONTSIZE)
    ax1.set_xlabel(_xlabel())
    ax1.set_title("Uncertainty decomposition BTNR")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=ALE_COLOR, label="Aleatoric (irreducible)"),
        Patch(facecolor=EPI_COLOR, label="Epistemic (reducible)"),
    ]
    if NORMALIZE_BY_TARGET_VAR:
        handles.append(Line2D([0], [0], color="0.4", ls="--", lw=1.0,
                              label="mean-predictor (var = target var)"))
    ax1.legend(handles=handles, loc="lower right", fontsize=8)
    ax1.grid(axis="x", alpha=0.25); ax1.grid(axis="y", visible=False)

    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks([])
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Scale free epistemic fraction")
    ax2.set_title("Reducible share")
    ax2.grid(axis="x", alpha=0.25); ax2.grid(axis="y", visible=False)

    fig.tight_layout()
    # Continuous alternating bands spanning BOTH panels (drawn post-layout).
    _draw_dataset_bands(fig, ax1, ax2, y, SLOT_PITCH)
    C.savefig(fig, "uncertainty_decomposition.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Baselines: total predictive VARIANCE (no epistemic/aleatoric split available)
# ---------------------------------------------------------------------------
# Reported in the SAME units as the BTN decomposition (predictive variance ÷
# target variance) so the two figures share one axis and one 1.0 reference and
# can be read against each other directly. Baselines store only a single total
# predictive std (`unc_sharpness`); we square it into a variance per seed before
# aggregating.

SHARP = "unc_sharpness"


def _baseline_variance(dataset, model):
    """Mean total predictive VARIANCE (sharpness^2, ÷ var(y)) for a baseline.

    Returns None if the baseline has no sharpness on this dataset.
    """
    stds = C.scalar_values("baseline", dataset, model, SHARP)
    if not stds:
        return None
    a = C.agg([s * s for s in stds])
    return a[0] / _var_scale(dataset) if a else None


def _make_baselines():
    plt = C._plt
    models = C.BASELINE_ORDER

    # Collect per (dataset, model) total predictive variance; keep datasets with
    # >=1 model.
    ds_rows = []
    for ds in C.DATASET_ORDER:
        comps = {}
        for m in models:
            v = _baseline_variance(ds, m)
            if v is not None:
                comps[m] = v
        if comps:
            ds_rows.append((ds, comps))
    if not ds_rows:
        print("  no baseline sharpness data found; skipping")
        return

    # Same sort convention as the BTN figure: by mean total across models.
    ds_rows.sort(key=lambda r: np.mean(list(r[1].values())))
    n_ds = len(ds_rows)
    n_mod = len(models)

    y, slot, h = _slot_geometry(n_mod)(n_ds)   # identical geometry to the BTN fig
    # offsets so models stack top->bottom within the slot, centred on y.
    mod_off = {m: slot / 2 - (i + 0.5) * h for i, m in enumerate(models)}

    # Single panel, but the SAME width as the BTN figure's LEFT panel: that panel
    # gets 2/3 of a width-10 figure (width_ratios=[2,1]), i.e. ~6.67in of axes.
    fig, ax = plt.subplots(figsize=(6.8, 0.85 * n_ds * SLOT_PITCH + 1.8))

    # transparent axes background so the figure-level bands show through
    ax.patch.set_visible(False)

    tick_pos, tick_lab = [], []       # one y-tick per bar (model name)
    for di, (ds, comps) in enumerate(ds_rows):
        for m in models:
            if m not in comps:
                continue
            v = comps[m]
            yy = y[di] + mod_off[m]
            ax.barh(yy, v, height=h * 0.92, color=C.COLORS[m],
                    edgecolor="white", linewidth=0.4)
            tick_pos.append(yy)
            tick_lab.append(C.display_name(m))

        # Dataset name in the LEFT margin (same gutter as the tick labels).
        _draw_dataset_header(ax, ds, y[di], slot, h)

    if NORMALIZE_BY_TARGET_VAR:
        ax.axvline(BASELINE_LINE, color="0.4", ls="--", lw=1.0, zorder=5)
    ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lab, fontsize=TICK_FONTSIZE)
    ax.set_xlabel(_xlabel())          # identical axis to the BTN decomposition
    ax.set_title("Total predictive variance (baselines)")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=C.COLORS[m], label=C.display_name(m))
               for m in models]
    if NORMALIZE_BY_TARGET_VAR:
        handles.append(Line2D([0], [0], color="0.4", ls="--", lw=1.0,
                              label="mean-predictor (var = target var)"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", visible=False)

    fig.tight_layout()
    # Continuous alternating bands (single panel: ax spans itself).
    _draw_dataset_bands(fig, ax, ax, y, SLOT_PITCH)
    C.savefig(fig, "uncertainty_sharpness_baselines.pdf")
    plt.close(fig)


def make():
    _make_all_families()
    _make_baselines()


def main(variant=None):
    C._plt = C.apply_style()
    print("Figure 6: epistemic/aleatoric decomposition (BTN) + sharpness (baselines) [valbest]")
    make()


if __name__ == "__main__":
    main()
