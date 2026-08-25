"""Counterfactual absolute-magnitude neutrality (Delta-H) for ActivitySCOPE.

Motivation
----------
The anomaly scores used elsewhere in ActivitySCOPE (``P(N_opp >= 4)``, ``DeltaQ``,
``S_EV``) all express an under-observation deficit in units of *oppositions*.  A
referee's natural question is instead: how much of the deficit could be explained
away by an error in the catalog absolute magnitude?  Lightcurve amplitude,
phase-function error under the fixed ``G = 0.15`` assumption, and detection-limit
bias all inflate a fitted ``H_V``, and an inflated ``H_V`` is exactly what makes an
inert object look under-observed.

This module answers that question directly by inverting the models along the
``H`` axis.  Holding the orbit and the observational record fixed, we ask for the
smallest brightness correction ``dH >= 0`` (i.e. the true object being *fainter*
than the catalog says by ``dH`` mag) at which the object stops being anomalous:

    dH_E  =  min{ d >= 0 : E[N_opp](H + d)   <= N_obs }      "full reconciliation"
    dH_Q  =  min{ d >= 0 : Q_0.006(H + d)    <= N_obs }      "leaves the 0.006 tail"
    dH_P  =  min{ d >= 0 : P(N_opp >= 4)(H+d) <= p_star }    "no longer more likely
                                                              than not to be seen"

``dH_Q`` is precisely the point at which ``DeltaQ`` crosses zero, and ``dH_E`` is
precisely the point at which the observed count equals the model expectation.
Neither definition introduces a new free parameter: both re-use operating points
that already exist in the method.  ``dH_Q <= dH_E`` always.

Why this is a better severity metric than ``DeltaQ`` alone: to first order

    dH_Q  ~  DeltaQ / |d Q_0.006 / d H|

so ``dH`` is the opposition deficit divided by its own sensitivity to the dominant
systematic.  It is the "significance" version of ``DeltaQ`` -- a three-opposition
deficit that 0.3 mag of lightcurve scatter can erase is not interesting, while a
three-opposition deficit that requires 4 mag is.

Implementation notes
--------------------
*   The scores are monotone (non-increasing) in ``H`` in practice, so the crossing
    is found by vectorised bisection rather than a dense grid: ~12 model calls
    total regardless of how many objects are being processed, versus one call per
    grid point per object.  This makes whole-catalog application feasible.
*   Gradient-boosted trees are piecewise constant in ``H``, so the crossing is an
    interval rather than a point.  Do not report ``dH`` to better than ~0.1 mag.
*   Objects that never reconcile within ``delta_max`` are returned right-censored
    (``censored=True``, ``dH = delta_max``); treat those as lower bounds.
*   All predictions here come from the refit full-data predictors, not the
    out-of-fold predictors used for ``exp_Num_opps`` / ``prob`` elsewhere.  The
    baseline column ``score_at_H0`` is reported so the offset is visible.
"""

import numpy as np
import pandas as pd

# Adopted 1-sigma uncertainty on a catalog H_V for a sparsely observed object,
# combining lightcurve half-amplitude (~0.2 mag), phase-function error from the
# fixed G = 0.15 assumption (~0.3 mag), and detection-limit bias (~0.2 mag) in
# quadrature.  Override per-object where multi-apparition photometry allows an
# empirical estimate.
SIGMA_H_DEFAULT = 0.45

# Features recomputed by utils.feature_engineering that depend on H.  Listed for
# documentation only; feature_engineering recomputes everything.
H_DEPENDENT_FEATURES = (
    "vis_timeavg", "vis_typ", "vis_flux", "vis_mid", "vis_q",
    "vis_opp_mean", "vis_orbit_mag_multi", "spatial_discoverability_fraction",
)


def _replicate(df, index_values):
    """Row-replicate ``df`` by label without the dtype coercion of ``iterrows``."""
    return df.loc[index_values].copy()


def _score_at(df_base, delta, utils, score_fn):
    """Evaluate ``score_fn`` with every object's H shifted by its entry in ``delta``."""
    scen = df_base.copy()
    scen["H"] = np.asarray(df_base["H"], dtype=float) + np.asarray(delta, dtype=float)
    scen = utils.feature_engineering(scen)
    return np.asarray(score_fn(scen), dtype=float)


def solve_delta_h(
    df,
    utils,
    score_fn,
    target,
    delta_max=15.0,
    n_iter=12,
    delta_min=0.0,
):
    """Smallest ``d`` in ``[delta_min, delta_max]`` with ``score_fn(H + d) <= target``.

    Assumes ``score_fn`` is non-increasing in ``H``.  Vectorised over all rows of
    ``df``: cost is ``n_iter + 2`` calls to ``feature_engineering`` + ``score_fn``,
    independent of the number of objects.

    Parameters
    ----------
    df : pd.DataFrame
        Candidate rows.  Must carry every column ``utils.feature_engineering``
        needs (pass slices of ``orb_pred`` / ``final``, not reconstructed rows).
    utils : module
        ``activityscope_utils``.
    score_fn : callable
        Maps a feature-engineered frame to an array of scores, e.g.
        ``lambda d: predictor_reg.predict(d) + 1``.
    target : array-like
        Per-object value the score must fall to, e.g. ``Num_opps``.

    Returns
    -------
    dict with keys ``delta``, ``censored``, ``already_neutral``, ``score_at_H0``,
    ``score_at_delta``.
    """
    idx = df.index
    n = len(df)
    target = pd.to_numeric(pd.Series(np.asarray(target, dtype=object)), errors="coerce").to_numpy(dtype=float)

    s0 = _score_at(df, np.zeros(n), utils, score_fn)
    # A NaN score or NaN target means the object cannot be solved at all; flag it
    # rather than silently reporting dH = 0.
    invalid = ~np.isfinite(s0) | ~np.isfinite(target)
    already_neutral = ~(s0 > target)          # NaN-safe: NaN -> treated as neutral

    s_max = _score_at(df, np.full(n, delta_max), utils, score_fn)
    censored = (s_max > target) & ~already_neutral

    lo = np.full(n, float(delta_min))
    hi = np.full(n, float(delta_max))
    active = ~already_neutral & ~censored

    for _ in range(n_iter):
        if not active.any():
            break
        mid = 0.5 * (lo + hi)
        s_mid = _score_at(df, mid, utils, score_fn)
        # score still too high at mid -> need to go fainter -> raise the floor
        too_high = (s_mid > target) & active
        lo = np.where(too_high, mid, lo)
        hi = np.where(~too_high & active, mid, hi)

    delta = np.where(already_neutral, 0.0, np.where(censored, delta_max, hi))
    delta = np.where(invalid, np.nan, delta)
    s_final = _score_at(df, np.nan_to_num(delta), utils, score_fn)

    return {
        "delta": pd.Series(delta, index=idx),
        "censored": pd.Series(censored & ~invalid, index=idx),
        "already_neutral": pd.Series(already_neutral & ~invalid, index=idx),
        "invalid": pd.Series(invalid, index=idx),
        "score_at_H0": pd.Series(s0, index=idx),
        "score_at_delta": pd.Series(s_final, index=idx),
    }


def h_neutrality_table(
    candidates,
    utils,
    predictor_reg,
    predictor_quant=None,
    predictor_bin=None,
    quantile_level=0.006,
    p_star=0.5,
    sigma_h=SIGMA_H_DEFAULT,
    delta_max=15.0,
    n_iter=12,
    extra_columns=(),
):
    """Delta-H table for a set of candidates.

    ``candidates`` must be indexed by designation and carry ``H``, ``Num_opps``
    and everything ``utils.feature_engineering`` consumes.  ``sigma_h`` may be a
    scalar or a per-object Series aligned to ``candidates.index``.
    """
    cand = candidates.copy()
    n_obs = pd.to_numeric(cand["Num_opps"], errors="coerce").to_numpy(dtype=float)

    out = pd.DataFrame(index=cand.index)
    out["H_current"] = np.asarray(cand["H"], dtype=float)
    out["Num_opps"] = n_obs

    # ---------------------------------------------------------------------
    # CASCADE.  The ActivitySCOPE notebooks feed out-of-fold regression output
    # into the quantile model as the ``exp_Num_opps`` feature, and feed both
    # ``exp_Num_opps`` and ``quantile_Opps`` into the binary classifier.  Those
    # columns are therefore *H-dependent inputs*, and a counterfactual H sweep
    # that leaves them at their catalog values silently holds part of the model
    # fixed -- producing a dH_Q and dH_P that are too small.  We recompute the
    # upstream predictions at every trial H before calling the downstream model.
    # ---------------------------------------------------------------------
    def _score_e(d):
        return np.asarray(predictor_reg.predict(d), dtype=float) + 1.0

    def _score_q(d):
        d["exp_Num_opps"] = _score_e(d)
        return np.asarray(predictor_quant.predict(d)[quantile_level], dtype=float) + 1.0

    def _score_p(d):
        d["exp_Num_opps"] = _score_e(d)
        if predictor_quant is not None:
            d["quantile_Opps"] = np.asarray(
                predictor_quant.predict(d)[quantile_level], dtype=float) + 1.0
        return np.asarray(predictor_bin.predict_proba(d)[1], dtype=float)

    # --- dH_E: brightening that makes the expectation match the observed count --
    res_e = solve_delta_h(
        cand, utils, _score_e,
        target=n_obs, delta_max=delta_max, n_iter=n_iter,
    )
    out["exp_Num_opps_full"] = res_e["score_at_H0"]
    out["dH_E"] = res_e["delta"]
    out["dH_E_censored"] = res_e["censored"]
    out["dH_E_invalid"] = res_e["invalid"]

    # --- dH_Q: brightening that takes the object out of the 0.006 lower tail -----
    if predictor_quant is not None:
        res_q = solve_delta_h(
            cand, utils, _score_q,
            target=n_obs, delta_max=delta_max, n_iter=n_iter,
        )
        out["quantile_Opps_full"] = res_q["score_at_H0"]
        out["dH_Q"] = res_q["delta"]
        out["dH_Q_censored"] = res_q["censored"]

    # --- dH_P: brightening after which the classifier no longer favours 4+ opps --
    if predictor_bin is not None:
        res_p = solve_delta_h(
            cand, utils, _score_p,
            target=np.full(len(cand), float(p_star)), delta_max=delta_max, n_iter=n_iter,
        )
        out["prob_full"] = res_p["score_at_H0"]
        out["dH_P"] = res_p["delta"]
        out["dH_P_censored"] = res_p["censored"]

    primary = "dH_Q" if "dH_Q" in out.columns else "dH_E"
    out["H_neutral"] = out["H_current"] + out[primary]

    sig = pd.Series(sigma_h, index=out.index) if np.isscalar(sigma_h) else sigma_h.reindex(out.index)
    out["sigma_H"] = sig
    out["Z_H"] = out[primary] / sig

    for col in extra_columns:
        if col in cand.columns:
            out[col] = cand[col]

    return out.sort_values(primary, ascending=False)


def archival_consistency(table, archival_limits):
    """Compare the required brightening against measured archival non-detections.

    ``archival_limits`` maps designation -> faintest well-established quiescent
    ``H_V`` limit (from a negative archival recovery at a known limiting
    magnitude).  Where ``H_limit >= H_neutral`` the brightening the model demands
    is independently corroborated: the object really was at least that faint at
    another epoch, so the deficit needs no photometric-scatter explanation.
    """
    lim = pd.Series(archival_limits, dtype=float).reindex(table.index)
    res = table.copy()
    res["H_archival_limit"] = lim
    res["archival_margin"] = lim - res["H_neutral"]
    res["archival_supports_dH"] = res["archival_margin"] >= 0
    return res
