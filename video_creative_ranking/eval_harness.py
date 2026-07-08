"""Shared evaluation / reporting for the creative_ranking_videos experiments."""

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """95% CI for accuracy over pairs."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    if n == 0:
        return (0.0, 0.0)
    accs = rng.choice(correct, size=(n_boot, n), replace=True).mean(axis=1)
    return (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))


def accuracy_report(name: str, sa, sb, y) -> dict:
    sa, sb, y = np.asarray(sa), np.asarray(sb), np.asarray(y)
    pred = (sb > sa).astype(int)
    correct = (pred == y).astype(float)
    lo, hi = bootstrap_ci(correct)
    return {
        "method": name,
        "accuracy": float(correct.mean()),
        "n": int(len(y)),
        "ci95": [lo, hi],
    }


def split_train_val(pairs, val_frac: float, seed: int):
    """Split at the creative level so no video is in both fit and val."""
    rng = random.Random(seed)
    ids = sorted({c.id for p in pairs for c in (p.creative_a, p.creative_b)})
    rng.shuffle(ids)
    val_ids = set(ids[: max(2, int(len(ids) * val_frac))])
    val = [p for p in pairs
           if p.creative_a.id in val_ids and p.creative_b.id in val_ids]
    fit = [p for p in pairs
           if p.creative_a.id not in val_ids and p.creative_b.id not in val_ids]
    if len(val) < 5:  # tiny datasets: fall back to a pair-level split
        pairs = list(pairs)
        rng.shuffle(pairs)
        k = max(2, int(len(pairs) * val_frac))
        val, fit = pairs[:k], pairs[k:]
    return fit, val


def print_summary(title: str, reports: list[dict]):
    print("\n" + "=" * 70)
    print(f"SUMMARY — {title}")
    print("=" * 70)
    for r in sorted(reports, key=lambda r: -r["accuracy"]):
        lo, hi = r["ci95"]
        print(f"  {r['method']:34s}: {r['accuracy']*100:5.1f}%  "
              f"[{lo*100:.1f}, {hi*100:.1f}]  (n={r['n']})")
    print("\n  Random baseline: 50.0%")


def save_results(out_dir: Path, results: dict):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {path}")


def save_scores(out_dir: Path, name: str, scores_by_creative: dict[str, float]):
    """Persist per-creative scores so later experiments can fuse methods."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scores.json"
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    for cid, s in scores_by_creative.items():
        existing.setdefault(cid, {})[name] = float(s)
    with open(path, "w") as f:
        json.dump(existing, f, indent=1)
