# type: ignore
"""
Uncertainty-quality metrics for probabilistic regression.

All functions operate on 1D torch tensors of equal length:
    y      : ground-truth targets
    mean   : predictive mean (point estimate)
    var    : *total* predictive variance (epistemic + aleatoric), already > 0

They assume a Gaussian predictive distribution  y ~ N(mean, var), which is the
observation model used throughout this codebase (see NLL in baselines/base.py).

These metrics answer three questions that R2/RMSE cannot:
  (A) Calibration  - is the predicted sigma the right *size*?   -> ECE, PICP, MPIW
  (B) Discrimination - does sigma *rise where error is large*?  -> AUSE, err/unc corr
  (C) Outlier detection - can sigma/NLL flag corrupted targets? -> AUROC / AUPR

Everything here is pure-tensor and model-agnostic: feed it BTN predictions or
baseline predictions alike.  ALS (non-Bayesian) has no predictive variance and
is intentionally not evaluated with these.
"""

import math
import torch

from utils.device_utils import DEVICE

_EPS = 1e-12
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _flatten(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(DEVICE).reshape(-1).to(torch.float64)


def _standard_normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / _SQRT2))


def _standard_normal_pdf(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * z * z) / _SQRT2PI


# ---------------------------------------------------------------------------
# (0) Probabilistic accuracy
# ---------------------------------------------------------------------------
def gaussian_nll(y, mean, var) -> float:
    """Mean negative log-likelihood under N(mean, var). Lower is better."""
    y, mean, var = _flatten(y), _flatten(mean), _flatten(var).clamp_min(_EPS)
    nll = 0.5 * (torch.log(2 * math.pi * var) + (y - mean) ** 2 / var)
    return float(nll.mean())


def gaussian_crps(y, mean, var) -> float:
    """
    Continuous Ranked Probability Score for a Gaussian predictive distribution
    (closed form, Gneiting & Raftery 2007). Strictly proper, in units of y.
    Lower is better. More robust to tail outliers than NLL.
    """
    y, mean, std = _flatten(y), _flatten(mean), _flatten(var).clamp_min(_EPS).sqrt()
    z = (y - mean) / std
    crps = std * (z * (2 * _standard_normal_cdf(z) - 1)
                  + 2 * _standard_normal_pdf(z)
                  - 1.0 / math.sqrt(math.pi))
    return float(crps.mean())


# ---------------------------------------------------------------------------
# (A) Calibration
# ---------------------------------------------------------------------------
def calibration_error(y, mean, var, n_bins: int = 10):
    """
    Regression calibration (Kuleshov et al. 2018).

    For a set of target quantile levels p, compute the empirical fraction of
    points whose true value falls at-or-below the predicted p-quantile.
    A perfectly calibrated model lies on the diagonal observed == expected.

    Returns dict with:
        ece        : mean |observed - expected|  (Expected Calibration Error)
        max_ce     : max  |observed - expected|
        expected   : list of target levels
        observed   : list of empirical coverages (for reliability diagram)
    """
    y, mean, std = _flatten(y), _flatten(mean), _flatten(var).clamp_min(_EPS).sqrt()
    z = (y - mean) / std                       # standardized residuals
    cdf = _standard_normal_cdf(z)              # PIT values; ~Uniform(0,1) if calibrated

    expected = torch.linspace(1.0 / n_bins, 1.0, n_bins, dtype=torch.float64, device=DEVICE)
    observed = torch.stack([(cdf <= p).to(torch.float64).mean() for p in expected])

    abs_err = (observed - expected).abs()
    return {
        "ece": float(abs_err.mean()),
        "max_ce": float(abs_err.max()),
        "expected": expected.tolist(),
        "observed": observed.tolist(),
    }


def interval_metrics(y, mean, var, confidence: float = 0.95):
    """
    Prediction-interval coverage & sharpness for a central `confidence` interval.

    Returns dict with:
        picp     : Prediction Interval Coverage Probability (target == confidence)
        mpiw     : Mean Prediction Interval Width (sharpness; smaller is better
                   *given* correct picp)
        coverage_error : |picp - confidence|
        sharpness      : mean predicted std
    """
    y, mean, std = _flatten(y), _flatten(mean), _flatten(var).clamp_min(_EPS).sqrt()
    # two-sided z for the requested central mass
    z = float(_SQRT2 * torch.erfinv(torch.tensor(confidence, dtype=torch.float64, device=DEVICE)))
    half = z * std
    lower, upper = mean - half, mean + half
    inside = ((y >= lower) & (y <= upper)).to(torch.float64)
    return {
        "picp": float(inside.mean()),
        "mpiw": float((2 * half).mean()),
        "coverage_error": float(abs(inside.mean() - confidence)),
        "sharpness": float(std.mean()),
    }


# ---------------------------------------------------------------------------
# (B) Discrimination: does uncertainty track error?
# ---------------------------------------------------------------------------
def sparsification(y, mean, var, n_steps: int = 20):
    """
    Sparsification / error-retention curve and AUSE.

    Rank points by predicted std (desc). Progressively remove the most uncertain
    fraction and recompute RMSE on the remainder. Compare against the *oracle*
    ordering (rank by true error). The gap is AUSE.

    Returns dict with:
        ause           : Area Under the Sparsification Error curve (lower better;
                         0 == uncertainty ranks errors as well as the oracle)
        spearman       : Spearman rho between predicted std and |error|
        fractions      : removed fractions (x-axis)
        rmse_by_unc    : RMSE after removing most-uncertain x  (model curve)
        rmse_by_oracle : RMSE after removing largest-error  x  (oracle curve)
    """
    y, mean, std = _flatten(y), _flatten(mean), _flatten(var).clamp_min(_EPS).sqrt()
    err = (y - mean) ** 2
    n = err.numel()

    order_unc = torch.argsort(std, descending=True)
    order_orc = torch.argsort(err, descending=True)
    err_by_unc = err[order_unc]
    err_by_orc = err[order_orc]

    fracs = torch.linspace(0.0, 1.0 - 1.0 / n_steps, n_steps, dtype=torch.float64, device=DEVICE)

    def _curve(sorted_err):
        out = []
        for f in fracs:
            k = int(f * n)                       # drop k most-uncertain/erroneous
            remain = sorted_err[k:]
            out.append(float(remain.mean().sqrt()) if remain.numel() else 0.0)
        return out

    rmse_unc = _curve(err_by_unc)
    rmse_orc = _curve(err_by_orc)
    ause = float(torch.trapezoid(torch.tensor(rmse_unc, device=DEVICE) - torch.tensor(rmse_orc, device=DEVICE), fracs))

    spearman = _spearman(std, err.sqrt())
    return {
        "ause": ause,
        "spearman_unc_err": spearman,
        "fractions": fracs.tolist(),
        "rmse_by_unc": rmse_unc,
        "rmse_by_oracle": rmse_orc,
    }


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation."""
    if a.numel() < 2:
        return 0.0
    ra = a.argsort().argsort().to(torch.float64)
    rb = b.argsort().argsort().to(torch.float64)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(_EPS)
    return float((ra @ rb) / denom)


# ---------------------------------------------------------------------------
# (C) Outlier / OOD detection via uncertainty
# ---------------------------------------------------------------------------
def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC where higher score => more likely positive (label==1). Rank-based."""
    scores, labels = _flatten(scores), _flatten(labels)
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = scores.argsort().argsort().to(torch.float64) + 1.0  # average-ties ignored
    sum_pos_ranks = ranks[labels == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _auprc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Average-precision style area under precision-recall curve."""
    scores, labels = _flatten(scores), _flatten(labels)
    order = torch.argsort(scores, descending=True)
    lab = labels[order]
    tp = torch.cumsum(lab, dim=0)
    fp = torch.cumsum(1 - lab, dim=0)
    total_pos = float(lab.sum())
    if total_pos == 0:
        return 0.0
    precision = tp / (tp + fp).clamp_min(_EPS)
    recall = tp / total_pos
    recall_prev = torch.cat([torch.zeros(1, dtype=torch.float64, device=DEVICE), recall[:-1]])
    return float((precision * (recall - recall_prev)).sum())


def label_outlier_detection(y, mean, var, corrupt_fraction=0.1,
                            corrupt_scale=5.0, seed=0):
    """
    Inject label outliers and test whether predictive uncertainty (and NLL)
    flags them.

    A random `corrupt_fraction` of test targets is replaced by extreme values
    (mean +/- corrupt_scale * std_of_y). Each point gets an anomaly score and we
    measure how well that score separates corrupted (positive) from clean points.

    Two scores are reported:
        std : the predictive std (does the model become *unsure* on outliers?)
        nll : per-point negative log-likelihood (does the outlier look unlikely?)

    Returns dict with AUROC/AUPR for both scores. AUROC ~0.5 == no signal,
    ~1.0 == perfect detection.
    """
    y = _flatten(y).clone()
    mean = _flatten(mean)
    std = _flatten(var).clamp_min(_EPS).sqrt()
    n = y.numel()
    if n < 2:
        return {}

    g = torch.Generator(device=DEVICE).manual_seed(int(seed))
    n_out = max(1, int(round(corrupt_fraction * n)))
    idx = torch.randperm(n, generator=g, device=DEVICE)[:n_out]

    labels = torch.zeros(n, dtype=torch.float64, device=DEVICE)
    labels[idx] = 1.0

    y_std = y.std().clamp_min(_EPS)
    signs = torch.where(
        torch.rand(n_out, generator=g, device=DEVICE) < 0.5,
        torch.tensor(-1.0, dtype=torch.float64, device=DEVICE),
        torch.tensor(1.0, dtype=torch.float64, device=DEVICE),
    )
    y[idx] = y[idx].mean() + signs * corrupt_scale * y_std

    # Per-point anomaly scores
    var_c = std ** 2
    nll_pp = 0.5 * (torch.log(2 * math.pi * var_c) + (y - mean) ** 2 / var_c)

    return {
        "outlier_fraction": corrupt_fraction,
        "outlier_scale": corrupt_scale,
        "auroc_std": _auroc(std, labels),
        "aupr_std": _auprc(std, labels),
        "auroc_nll": _auroc(nll_pp, labels),
        "aupr_nll": _auprc(nll_pp, labels),
    }


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------
def compute_uncertainty_metrics(y, mean, var, *, confidence=0.95,
                                run_outlier=True, outlier_fraction=0.1,
                                outlier_scale=5.0, seed=0,
                                keep_curves=False):
    """
    Compute the full uncertainty-quality suite for one prediction set.

    Args:
        y, mean, var : 1D tensors (var = total predictive variance > 0)
        confidence   : central interval level for PICP/MPIW
        run_outlier  : also run label-outlier injection AUROC
        keep_curves  : if True, retain reliability + sparsification curves
                       (arrays) for plotting; otherwise only scalars are kept.

    Returns a flat dict of scalar metrics (prefixed) plus, optionally, curve
    arrays under '*_curve' keys.
    """
    out = {}

    out["unc_nll"] = gaussian_nll(y, mean, var)
    out["unc_crps"] = gaussian_crps(y, mean, var)

    cal = calibration_error(y, mean, var)
    out["unc_ece"] = cal["ece"]
    out["unc_max_ce"] = cal["max_ce"]

    iv = interval_metrics(y, mean, var, confidence=confidence)
    out["unc_picp"] = iv["picp"]
    out["unc_mpiw"] = iv["mpiw"]
    out["unc_coverage_error"] = iv["coverage_error"]
    out["unc_sharpness"] = iv["sharpness"]

    sp = sparsification(y, mean, var)
    out["unc_ause"] = sp["ause"]
    out["unc_spearman_err"] = sp["spearman_unc_err"]

    if run_outlier:
        od = label_outlier_detection(
            y, mean, var,
            corrupt_fraction=outlier_fraction,
            corrupt_scale=outlier_scale,
            seed=seed,
        )
        for k, v in od.items():
            out[f"unc_outlier_{k}"] = v

    if keep_curves:
        out["reliability_curve"] = {
            "expected": cal["expected"], "observed": cal["observed"],
        }
        out["sparsification_curve"] = {
            "fractions": sp["fractions"],
            "rmse_by_unc": sp["rmse_by_unc"],
            "rmse_by_oracle": sp["rmse_by_oracle"],
        }

    return out
