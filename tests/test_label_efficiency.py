import numpy as np
import torch

from label_efficiency import balanced_batches, stratified_subset, train_network
from _common import build_model


def test_subset_is_stratified_deterministic_and_has_minimum():
    y = np.array([1] * 800 + [0] * 200)
    a = stratified_subset(y, 0.1, seed=3)
    b = stratified_subset(y, 0.1, seed=3)
    assert np.array_equal(a, b)
    assert (y[a] == 1).sum() == 80 and (y[a] == 0).sum() == 20
    tiny = stratified_subset(y, 0.001, seed=0, min_per_class=4)
    assert (y[tiny] == 0).sum() == 4 and (y[tiny] == 1).sum() == 4
    assert not np.array_equal(stratified_subset(y, 0.1, 1), a)


def test_batches_are_class_balanced():
    y = np.array([1] * 95 + [0] * 5)
    for b in balanced_batches(y, 32, 5, np.random.default_rng(0)):
        assert (y[b] == 1).sum() == 16 and (y[b] == 0).sum() == 16


def test_probe_only_changes_detection_head():
    torch.manual_seed(0)
    m = build_model({"base_channels": 8})
    before = {k: v.clone() for k, v in m.state_dict().items()}
    x = np.random.default_rng(0).normal(0, 0.1, (16, 2, 2000)).astype(np.float32)
    y = np.array([0, 1] * 8)
    train_network(m, x, y, steps=3, batch=8, lr=1e-2, seed=0, device="cpu", probe=True)
    after = m.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed and all(k.startswith("detection_head") for k in changed)
