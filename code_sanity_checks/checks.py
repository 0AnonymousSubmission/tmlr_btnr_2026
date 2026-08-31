# type: ignore
"""
Sanity checks exercised by ``python run.py --sanity-check``.

Each check returns (passed: bool, detail: str). ``run_sanity_checks`` builds a
model via the normal path and runs them all on small synthetic regression data,
printing a summary and returning True iff everything passed.
"""

import numpy as np
import torch
import quimb.tensor as qt

from tensor.btn import BTN, NOT_TRAINABLE_TAG
from tensor.builder import Inputs
from model.utils import REGRESSION_METRICS

torch.set_default_dtype(torch.float64)

TOL = 1e-12


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_stream(model, X, y, batch_size):
    return Inputs(
        inputs=[X],
        outputs=[y.squeeze(-1)] if not model.output_dims else [y],
        outputs_labels=list(model.output_dims),
        input_labels=list(model.input_dims),
        batch_dim="s",
        batch_size=batch_size,
    )


def _synthetic_data(n, n_features, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, n_features, generator=g)
    w = torch.randn(n_features, generator=g)
    y = (X @ w).unsqueeze(-1) + 0.05 * torch.randn(n, 1, generator=g)
    return X, y


def _fresh_btn(model_factory, n_features, seed, **btn_kwargs):
    """Build a fresh model + BTN + data streams."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_factory(n_features)
    n = 160
    X, y = _synthetic_data(n, n_features, seed)
    Xv, yv = _synthetic_data(60, n_features, seed + 1)
    train = _make_stream(model, X, y, n)
    val = _make_stream(model, Xv, yv, 60)
    tn = BTN(
        mu=model.tn,
        data_stream=train,
        batch_dim="s",
        method="cholesky",
        device=torch.device("cpu"),
        bond_prior_alpha=getattr(model, "bond_prior_alpha", 5.0),
        **btn_kwargs,
    )
    tn.register_data_streams(val)
    return tn, train, val


def _cavi_sweep(tn, n_epochs=10, update_bonds=True):
    nodes = list(tn.mu.tag_map.keys())
    for _ in range(n_epochs):
        for nd in nodes:
            tn.update_sigma_node(nd)
            tn.update_mu_node(nd)
        if update_bonds:
            for b in [i for i in tn.mu.ind_map if i not in tn.output_dimensions]:
                tn.update_bond(b)
        tn.update_tau()


def _internal_bonds(tn):
    excl = set(tn.output_dimensions) | set(tn.input_indices)
    return [b for b in tn.mu.ind_map if b not in excl]


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
def check_builds_and_predicts(model_factory, n_features, seed):
    tn, train, _ = _fresh_btn(model_factory, n_features, seed)
    _cavi_sweep(tn, n_epochs=5)
    scores = tn.evaluate(REGRESSION_METRICS, data_stream=train)
    loss = float(scores["loss"])
    ok = np.isfinite(loss)
    return ok, f"train loss={loss:.4f}"


def check_elbo_monotone(model_factory, n_features, seed):
    tn, _, _ = _fresh_btn(model_factory, n_features, seed)
    nodes = list(tn.mu.tag_map.keys())
    bonds = [i for i in tn.mu.ind_map if i not in tn.output_dimensions]
    elbos = [tn.compute_elbo(verbose=False, relative=False)]
    for epoch in range(12):
        for nd in nodes:
            tn.update_sigma_node(nd)
            tn.update_mu_node(nd)
        if epoch >= 2:
            for b in bonds:
                tn.update_bond(b)
        tn.update_tau()
        elbos.append(tn.compute_elbo(verbose=False, relative=False))
    diffs = np.diff(np.array(elbos))
    worst = float(diffs.min())
    ok = worst >= -1e-4
    return ok, f"worst ELBO step={worst:+.2e} (>= -1e-4 required)"


def check_gamma_nonnegative(model_factory, n_features, seed):
    tn, _, _ = _fresh_btn(model_factory, n_features, seed)
    _cavi_sweep(tn, n_epochs=12)
    # trim() refreshes sigma before scoring; replicate that here.
    for nd in list(tn.mu.tags):
        tn.update_sigma_node(nd)
    mins = []
    for b in _internal_bonds(tn):
        g = tn._get_effective_parameters(b)
        if g is not None:
            mins.append(float(g.min()))
    if not mins:
        return True, "no internal bonds to score"
    gmin = min(mins)
    ok = gmin >= -TOL
    return ok, f"min gamma={gmin:.2e} (>= -1e-6 required)"


def _check_trim_method(model_factory, n_features, seed, method, threshold):
    tn, train, _ = _fresh_btn(model_factory, n_features, seed)
    _cavi_sweep(tn, n_epochs=12)
    n_before = sum(t.data.numel() for t in tn.mu.tensors)
    trimmed = tn.trim(method=method, threshold=threshold, verbose=False)
    # network must still contract + predict
    scores = tn.evaluate(REGRESSION_METRICS, data_stream=train)
    loss = float(scores["loss"])
    n_after = sum(t.data.numel() for t in tn.mu.tensors)
    ok = np.isfinite(loss) and n_after <= n_before
    return ok, f"trimmed={trimmed} params {n_before}->{n_after} loss={loss:.4f}"


def check_trim_relevance(model_factory, n_features, seed):
    return _check_trim_method(model_factory, n_features, seed, "relevance", 2.0)


def check_trim_variance(model_factory, n_features, seed):
    return _check_trim_method(model_factory, n_features, seed, "variance", 0.3)


def check_trim_gamma(model_factory, n_features, seed):
    return _check_trim_method(model_factory, n_features, seed, "gamma", 0.3)


def check_trivial_bond_removal(model_factory, n_features, seed):
    """A bond forced to size 1 must be squeezed out losslessly."""
    tn, train, _ = _fresh_btn(
        model_factory, n_features, seed, remove_trivial_bonds=True
    )
    _cavi_sweep(tn, n_epochs=8)
    internal = _internal_bonds(tn)
    if not internal:
        return True, "no internal bonds (nothing to remove)"
    bond = internal[0]
    pred_before, _, _ = tn.predict_mean_var(data_stream=train)
    # twin without removal, sliced to size 1 for the reference prediction
    tn2, train2, _ = _fresh_btn(
        model_factory, n_features, seed, remove_trivial_bonds=False
    )
    _cavi_sweep(tn2, n_epochs=8)
    tn2._apply_bond_trim(bond, [0])
    ref, _, _ = tn2.predict_mean_var(data_stream=train2)
    # removal path
    tn._apply_bond_trim(bond, [0])
    got, _, _ = tn.predict_mean_var(data_stream=train)
    removed = bond not in tn.mu.ind_map
    lossless = torch.allclose(ref, got, atol=1e-9)
    ok = removed and lossless
    return ok, f"bond {bond} removed={removed}, lossless={lossless}"


def check_nt_block_preserved(model_factory, n_features, seed):
    """
    Insert a fixed Identity NT node on a bond and confirm training never
    changes it. Uses a plain MPS so the check is model-agnostic.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    L, bd, pd, n = 3, 4, n_features, 160
    ts = [
        qt.Tensor(torch.randn(pd, bd) * 0.1, inds=("x0", "b0"), tags={"Node0"}),
        qt.Tensor(torch.eye(bd), inds=("b0", "b0b"), tags={"Identity", NOT_TRAINABLE_TAG}),
        qt.Tensor(torch.randn(bd, pd, bd) * 0.1, inds=("b0b", "x1", "b1"), tags={"Node1"}),
        qt.Tensor(torch.randn(bd, pd) * 0.1, inds=("b1", "x2"), tags={"Node2"}),
    ]
    mu = qt.TensorNetwork(ts)
    X, y = _synthetic_data(n, pd, seed)
    ds = Inputs(inputs=[X], outputs=[y.squeeze(-1)], outputs_labels=[],
                input_labels=[f"x{i}" for i in range(L)], batch_dim="s", batch_size=n)
    tn = BTN(mu=mu, data_stream=ds, batch_dim="s", device=torch.device("cpu"),
             bond_prior_alpha=5.0)
    before = tn.mu["Identity"].data.clone()
    _cavi_sweep(tn, n_epochs=10)
    after = tn.mu["Identity"].data
    ok = torch.allclose(before, after, atol=1e-12)
    max_diff = (after - before).abs().max().item()
    return ok, f"max|dI|={max_diff:.2e}"


def check_input_trim_and_detach(model_factory, n_features, seed):
    """
    With trim_input=True, input legs shrink and (non-last) can detach fully,
    with data streams kept consistent and the network still contractible.
    """
    tn, train, val = _fresh_btn(
        model_factory, n_features, seed, trim_input=True
    )
    _cavi_sweep(tn, n_epochs=10)
    tn.threshold = 1e9  # aggressive: trim inputs hard
    tn.trim(method="relevance", verbose=False)
    # at least one input must survive (allow_empty_input defaults False)
    survived = len(tn.input_indices) >= 1
    # model / data consistency for surviving inputs
    mu_t, _, _ = train.batches[0]
    data_dims = {}
    for t in mu_t:
        for ix in t.inds:
            if ix != train.batch_dim:
                data_dims[ix] = t.shape[t.inds.index(ix)]
    consistent = all(
        tn.mu.ind_size(x) == data_dims.get(x) for x in tn.input_indices
    )
    scores = tn.evaluate(REGRESSION_METRICS, data_stream=val)
    ok = survived and consistent and np.isfinite(float(scores["loss"]))
    return ok, f"inputs_left={len(tn.input_indices)} consistent={consistent}"


CHECKS = [
    ("builds & predicts", check_builds_and_predicts),
    ("ELBO monotone", check_elbo_monotone),
    ("gamma >= 0", check_gamma_nonnegative),
    ("trim: relevance", check_trim_relevance),
    ("trim: variance", check_trim_variance),
    ("trim: gamma", check_trim_gamma),
    ("trivial bond removal", check_trivial_bond_removal),
    ("NT block preserved", check_nt_block_preserved),
    ("input trim & detach", check_input_trim_and_detach),
]


def run_sanity_checks(model_factory, n_features, model_name="model", seed=0):
    """
    Run all sanity checks against ``model_factory`` (n_features -> model object
    exposing .tn, .input_dims, .output_dims).

    Returns True iff every check passed.
    """
    print(f"\n{'=' * 64}")
    print(f" SANITY CHECKS | model={model_name} | n_features={n_features} | seed={seed}")
    print(f"{'=' * 64}")

    results = []
    for name, fn in CHECKS:
        try:
            passed, detail = fn(model_factory, n_features, seed)
        except Exception as e:  # a crash is a failed check
            import traceback
            passed, detail = False, f"EXCEPTION: {type(e).__name__}: {e}"
            if _VERBOSE:
                traceback.print_exc()
        results.append((name, passed, detail))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name:<24} {detail}")

    n_pass = sum(1 for _, p, _ in results if p)
    all_ok = n_pass == len(results)
    print(f"{'-' * 64}")
    print(f" {n_pass}/{len(results)} checks passed  ->  "
          f"{'ALL GOOD' if all_ok else 'FAILURES PRESENT'}")
    print(f"{'=' * 64}\n")
    return all_ok


_VERBOSE = False


def set_verbose(v):
    global _VERBOSE
    _VERBOSE = v
