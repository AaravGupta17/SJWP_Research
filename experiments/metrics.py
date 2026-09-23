"""
metrics.py — detection metrics with honest uncertainty
=======================================================
Why not just AUROC/F1/accuracy?
  - With 80% leak windows, a model that says "leak" for everything gets
    F1 ~0.89 and accuracy ~0.80. Per-class rates (detection rate and
    false-alarm rate) and balanced accuracy expose that immediately.
  - Windows cut from the same recording are not independent samples.
    Bootstrapping individual windows makes confidence intervals far too
    narrow. We resample whole recordings (a cluster bootstrap), stratified
    by class, so the CI reflects how many *recordings* we actually have.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def _auroc(y: np.ndarray, p: np.ndarray):
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def point_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    pred = p >= threshold
    pos, neg = y == 1, y == 0
    tpr = float(pred[pos].mean()) if pos.any() else None      # detection rate
    fpr = float(pred[neg].mean()) if neg.any() else None      # false-alarm rate
    bal = (tpr + (1 - fpr)) / 2 if tpr is not None and fpr is not None else None
    return {
        "auroc": _auroc(y, p),
        "detection_rate": tpr,
        "false_alarm_rate": fpr,
        "balanced_accuracy": bal,
        "threshold": threshold,
        "n_leak": int(pos.sum()),
        "n_no_leak": int(neg.sum()),
    }


def cluster_bootstrap(y, p, groups, threshold: float = 0.5, n_boot: int = 2000,
                      seed: int = 0, alpha: float = 0.05) -> dict:
    """95% CIs by resampling whole groups (recordings or sites).

    If every group has a single class (a recording is either leak or
    no-leak), groups are resampled separately within the leak and no-leak
    strata, so each replicate keeps the original number of groups per class.
    If some groups contain both classes (e.g. a Hong Kong site recorded
    before and after repair), stratification is impossible without splitting
    a group, so all groups are resampled together; replicates that happen to
    contain one class only are skipped.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uniq}
    cls_of = {}
    for g, idx in idx_of.items():
        c = np.unique(y[idx])
        cls_of[g] = int(c[0]) if len(c) == 1 else -1        # -1 = mixed group
    mixed = any(c == -1 for c in cls_of.values())
    strata = ({"all": list(uniq)} if mixed else
              {c: [g for g in uniq if cls_of[g] == c] for c in (0, 1)})

    rng = np.random.default_rng(seed)
    keys = ("auroc", "detection_rate", "false_alarm_rate", "balanced_accuracy")
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        chosen = []
        for gs in strata.values():
            if gs:
                picks = rng.choice(len(gs), size=len(gs), replace=True)
                chosen.extend(gs[i] for i in picks)
        sel = np.concatenate([idx_of[g] for g in chosen]) if chosen else np.array([], int)
        m = point_metrics(y[sel], p[sel], threshold)
        for k in keys:
            if m[k] is not None:
                draws[k].append(m[k])

    ci = {}
    for k in keys:
        d = np.asarray(draws[k])
        ci[k] = ([float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))]
                 if len(d) >= 20 else None)
    return {"ci95": ci, "n_boot": n_boot, "mixed_groups": mixed,
            "n_groups_leak": sum(1 for g in uniq if (y[idx_of[g]] == 1).any()),
            "n_groups_no_leak": sum(1 for g in uniq if (y[idx_of[g]] == 0).any())}


def detection_report(y, p, groups=None, threshold: float = 0.5,
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """Point metrics + (if groups given) recording-level bootstrap CIs."""
    rep = point_metrics(y, p, threshold)
    if groups is not None and rep["auroc"] is not None:
        rep.update(cluster_bootstrap(y, p, groups, threshold, n_boot, seed))
    return rep


def fmt(rep: dict, key: str = "auroc") -> str:
    """'0.873 [0.801, 0.930]' style string for printing."""
    v = rep.get(key)
    if v is None:
        return "n/a"
    ci = (rep.get("ci95") or {}).get(key)
    return f"{v:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
