"""
model_f_eval.py — E11: Model F on data it never heard
=======================================================
For each Model F checkpoint (Model_F/train_f.py), tests on

  mendeley_looped_acc / _hyd   Mendeley LOOPED, both real sensors (the clean
                               test: Model F never saw any Looped recording
                               and no Mendeley label)
  <held-out source>            every public source the checkpoint was trained
                               WITHOUT (--exclude-source), one-sensor windows
                               paired into two channels exactly as in training
  noise sanity                 band-limited white and coloured noise: share
                               flagged as a leak (E7 found 100% for Models C/D)

Input processing matches training: 2 kHz band limit, joint z-score.
Baseline on the SAME test windows: logistic regression on the loudness-free
features of E9, trained on the same real sources the checkpoint used.
AUROC on logits; detection / false-alarm rates at logit 0; 95% CIs from a
bootstrap over recordings (Mendeley) or sites/recordings (public data).

    python experiments/model_f_eval.py --ckpt best_model_f_seed0.pt best_model_f_nohk_seed0.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import mendeley as M
import public_data as P
from _common import DATASETS, load_model, predict_proba, record_run
from cross_dataset import features_1ch, standardise
from metrics import detection_report, fmt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Model_F"))
import augment_f as A                                      # noqa: E402

PUBLIC_SOURCES = {"hongkong": ["hk_noiselogger", "hk_hydrophone"], "dongguan": ["dongguan"]}
MENDELEY_ROOTS = {"acc": ("accelerometer", DATASETS / "Accelerometer" / "Accelerometer"),
                  "hyd": ("hydrophone", DATASETS / "Hydrophone" / "Hydrophone")}


def prepare(x: np.ndarray) -> np.ndarray:
    """(N, 2, T) raw -> band-limited, joint z-scored, as in training."""
    return np.stack([A.joint_zscore(A.band_limit(w.astype(np.float64))) for w in x])


def pair_public(W: P.Windows, seed: int = 0) -> np.ndarray:
    """Two-channel windows from one-sensor windows: the second channel mixes a
    delayed copy with another window of the same recording (train_f.py [F5])."""
    rng = np.random.default_rng(seed)
    x = W.x / (np.sqrt(np.mean(W.x.astype(np.float64) ** 2, axis=1, keepdims=True)) + 1e-12)
    out = np.empty((len(x), 2, x.shape[1]))
    for g in np.unique(W.group):
        idx = np.flatnonzero(W.group == g)
        for i in idx:
            out[i] = A.pair_channels(x[i], x[rng.choice(idx)], rng)
    return out


def load_mendeley_looped():
    sets = {}
    for tag, (sensor, root) in MENDELEY_ROOTS.items():
        try:
            ws = M.load_windowset(root, sensor, verbose=False)
        except FileNotFoundError as e:
            print(f"  skip mendeley {tag}: {e}")
            continue
        ws = ws.subset(ws.topology == "Looped")
        sets[f"mendeley_looped_{tag}"] = (ws.x, ws.y, ws.group)
    return sets


def logreg_baseline(train_W: P.Windows, test_ch1: np.ndarray) -> np.ndarray:
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
    clf.fit(features_1ch(standardise(train_W.x)), train_W.y)
    return clf.decision_function(features_1ch(standardise(test_ch1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=["best_model_f_seed0.pt"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Loading test data...")
    public = {s: P.load(names) for s, names in PUBLIC_SOURCES.items()}
    mendeley = {k: (prepare(x), y, g, A.band_limit(x[:, 0].astype(np.float64)))
                for k, (x, y, g) in load_mendeley_looped().items()}
    rng = np.random.default_rng(args.seed)
    noise = {"white": np.stack([A.band_limit(rng.standard_normal((2, A.T))) for _ in range(500)]),
             "coloured": np.stack([A.band_limit(A.pair_channels(A.coloured_noise(rng), A.coloured_noise(rng),
                                                                rng)) for _ in range(500)])}
    noise = {k: prepare(v) for k, v in noise.items()}

    results = {}
    for ck in args.ckpt:
        model, c = load_model(ck)
        cfg = c.get("cfg", {})
        if cfg.get("input_norm") != "zscore":
            print(f"{ck}: not a Model F checkpoint (input_norm={cfg.get('input_norm')}), skipped")
            continue
        excluded = cfg.get("exclude_source") or []
        train_src = [s for s in PUBLIC_SOURCES if s not in excluded and len(public[s])]
        train_W = P.Windows.concat([public[s] for s in train_src]) if train_src else None
        print(f"\n{ck} (epoch {c.get('epoch')}) | trained on real: {train_src or 'none'} | "
              f"held out: {excluded or 'none'}")
        res = results[ck] = {"epoch": c.get("epoch"), "exclude_source": excluded, "tests": {}}

        tests = dict(mendeley)
        for s in excluded:
            for d in PUBLIC_SOURCES[s]:
                Wd = public[s].subset(public[s].dataset == d)
                if len(Wd) and len(np.unique(Wd.y)) == 2:
                    tests[d] = (prepare(pair_public(Wd, args.seed)), Wd.y, Wd.group, Wd.x)
        for name, (x, y, g, ch1) in tests.items():
            logit = predict_proba(model, x, logits=True)
            r = detection_report(y, logit, g, threshold=0.0, n_boot=args.n_boot, seed=args.seed)
            row = {"model": r}
            line = (f"  {name:22s} Model F  AUROC {fmt(r)} | detect {r['detection_rate']:.2f} "
                    f"false alarm {r['false_alarm_rate']:.2f}")
            if train_W is not None:
                b = detection_report(y, logreg_baseline(train_W, ch1), g, threshold=0.0,
                                     n_boot=args.n_boot, seed=args.seed)
                row["logreg"] = b
                line += f"\n  {'':22s} logreg   AUROC {fmt(b)}"
            res["tests"][name] = row
            print(line)
        res["noise_flagged"] = {k: float((predict_proba(model, v, logits=True) >= 0).mean())
                                for k, v in noise.items()}
        print("  noise flagged as leak: " + ", ".join(f"{k} {v:.0%}" for k, v in res["noise_flagged"].items()))

    summary = "; ".join(f"{ck}: " + ", ".join(f"{t} {r['tests'][t]['model']['auroc']:.2f}"
                                               for t in r["tests"]) for ck, r in results.items())
    record_run("e11_model_f_eval", vars(args), results, summary[:900])


if __name__ == "__main__":
    main()
