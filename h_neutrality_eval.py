"""Head-to-head evaluation of ActivitySCOPE anomaly rankings.

Answers the question "does Delta-H separate active from background as well as
DeltaQ?" with two complementary label sets:

1.  ``known_active`` -- the curated list of confirmed / strongly suspected active
    objects.  Convenient but *selection-contaminated*: most of these objects were
    found by DeltaQ ranking in the first place, so DeltaQ enjoys a home-field
    advantage and any DeltaQ-vs-DeltaH comparison on this set is biased toward
    DeltaQ.  Report it, but do not rest the argument on it.

2.  ``comet_holdout`` -- comets from the MPC comet file, scored through the same
    models using the asteroidal H_V they would have had (this is already built in
    the notebooks as ``cometmerge``).  These are genuinely active objects that
    were *never* selected by any ActivitySCOPE score, so they give an unbiased
    benchmark.  This is the evaluation that should go in the paper.

The operational figure of merit is not AUC but the paper's own separability
claim: how many non-active objects intervene above the lowest-ranked known
active object.
"""

import numpy as np
import pandas as pd


def roc_auc(scores, labels, higher_is_more_anomalous=True):
    """Rank-based AUC with tie handling; no sklearn dependency."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(bool)
    keep = np.isfinite(s)
    s, y = s[keep], y[keep]
    if not (y.any() and (~y).any()):
        return np.nan
    if not higher_is_more_anomalous:
        s = -s
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties
    df = pd.DataFrame({"s": s, "r": ranks})
    ranks = df.groupby("s")["r"].transform("mean").to_numpy()
    n_pos, n_neg = y.sum(), (~y).sum()
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def separability(scores, labels, higher_is_more_anomalous=True):
    """Paper-style separability of a ranking.

    Returns the rank of the worst-ranked positive, how many negatives sit above
    it (the "interlopers"), and the implied precision of the head of the list.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(bool)
    keep = np.isfinite(s)
    s, y = s[keep], y[keep]
    if not y.any():
        return {}
    order = np.argsort(-s if higher_is_more_anomalous else s, kind="mergesort")
    y_sorted = y[order]
    last_pos = int(np.flatnonzero(y_sorted)[-1])           # 0-based
    n_above = last_pos + 1
    return {
        "n_positives": int(y.sum()),
        "rank_of_worst_positive": n_above,
        "interlopers_above": int(n_above - y_sorted[: last_pos + 1].sum()),
        "precision_at_full_recall": float(y_sorted[: last_pos + 1].mean()),
        "precision_at_n_pos": float(y_sorted[: int(y.sum())].mean()),
    }


def compare_rankings(df, label_col, metrics):
    """Compare several anomaly metrics on one labelled frame.

    ``metrics`` maps a display name to ``(column, higher_is_more_anomalous)``.
    """
    rows = []
    labels = df[label_col].astype(bool)
    for name, (col, higher) in metrics.items():
        if col not in df.columns:
            continue
        rec = {"metric": name, "auc": roc_auc(df[col], labels, higher)}
        rec.update(separability(df[col], labels, higher))
        rows.append(rec)
    return pd.DataFrame(rows).set_index("metric").sort_values("auc", ascending=False)


def marginal_discoveries(df, new_col, old_col, old_threshold, top_n=50):
    """Objects the new ranking promotes that the old ranking's threshold excludes.

    This is the direct test of "can Delta-H find objects DeltaQ misses": DeltaQ
    has a hard floor because a deficit of 3 oppositions is unreachable for
    objects whose predicted count is small, whereas Delta-H is scale-free in
    ``N_opp``.  The 2009 DP2 case discussed in the paper is the archetype.
    """
    top = df.sort_values(new_col, ascending=False).head(top_n)
    return top[top[old_col] < old_threshold]
