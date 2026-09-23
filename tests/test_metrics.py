import numpy as np
import pytest

from metrics import cluster_bootstrap, detection_report, point_metrics


def test_always_leak_is_exposed_by_per_class_rates():
    # 80% leak prevalence, model says "leak" for everything
    y = np.array([1] * 80 + [0] * 20)
    p = np.full(100, 0.99)
    m = point_metrics(y, p)
    assert m["detection_rate"] == 1.0
    assert m["false_alarm_rate"] == 1.0
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["auroc"] == pytest.approx(0.5)


def test_perfect_separation():
    y = np.array([0, 0, 1, 1])
    m = point_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["auroc"] == 1.0 and m["balanced_accuracy"] == 1.0


def test_single_class_gives_none_not_crash():
    m = point_metrics(np.ones(5), np.linspace(0, 1, 5))
    assert m["auroc"] is None and m["false_alarm_rate"] is None


def test_cluster_bootstrap_ci_contains_point_and_counts_groups():
    rng = np.random.default_rng(0)
    groups, y, p = [], [], []
    for g in range(10):
        label = g % 2
        for _ in range(30):
            groups.append(f"rec{g}")
            y.append(label)
            p.append(rng.normal(0.6 if label else 0.4, 0.15))
    rep = detection_report(np.array(y), np.array(p), np.array(groups), n_boot=500)
    lo, hi = rep["ci95"]["auroc"]
    assert lo <= rep["auroc"] <= hi
    assert rep["n_groups_leak"] == 5 and rep["n_groups_no_leak"] == 5


def test_cluster_ci_wider_than_window_level_ci_when_windows_correlated():
    # each recording has its own offset: windows inside a recording are
    # correlated, so the recording-level CI must be wider than a naive one
    rng = np.random.default_rng(1)
    groups, y, p = [], [], []
    for g in range(8):
        label = g % 2
        offset = rng.normal(0, 0.3)
        for _ in range(100):
            groups.append(g); y.append(label)
            p.append(offset + 0.1 * label + rng.normal(0, 0.05))
    y, p, groups = map(np.array, (y, p, groups))
    clustered = cluster_bootstrap(y, p, groups, n_boot=400)["ci95"]["auroc"]
    naive = cluster_bootstrap(y, p, np.arange(len(y)), n_boot=400)["ci95"]["auroc"]
    assert (clustered[1] - clustered[0]) > 2 * (naive[1] - naive[0])


def test_mixed_label_groups_use_unstratified_site_bootstrap():
    # e.g. a Hong Kong site recorded leaking and again after repair
    rng = np.random.default_rng(0)
    y, p, g = [], [], []
    for site in range(12):
        for label in (0, 1):
            for _ in range(5):
                y.append(label); g.append(f"site{site}")
                p.append(rng.normal(0.7 if label else 0.3, 0.2))
    rep = detection_report(np.array(y), np.array(p), np.array(g), n_boot=300)
    assert rep["mixed_groups"] is True
    assert rep["n_groups_leak"] == rep["n_groups_no_leak"] == 12
    lo, hi = rep["ci95"]["auroc"]
    assert lo <= rep["auroc"] <= hi


def test_saturated_sigmoid_hides_ranking_but_logits_do_not():
    """Why experiments score AUROC on logits: float32 sigmoid of large logits
    is exactly 1.0, so correctly ranked windows become ties (AUROC 0.5)."""
    import torch
    logits = torch.tensor([20.0, 25.0, 30.0, 35.0])      # no-leak, no-leak, leak, leak
    y = np.array([0, 0, 1, 1])
    probs = torch.sigmoid(logits).numpy()
    assert (probs == 1.0).all()
    assert point_metrics(y, probs)["auroc"] == 0.5
    assert point_metrics(y, logits.numpy(), threshold=0.0)["auroc"] == 1.0
