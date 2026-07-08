"""Contrastive (persona-style) vector over LLM activations.

vector = mean(activations | winning creatives) - mean(activations | losing
creatives), optionally in per-dimension standardized space (a diagonal-LDA
flavor that usually separates better when feature scales vary).

score(creative) = <pooled_hidden, vector>; a pair is predicted by comparing
the two scores. Layer / pooling / standardization are selected on a
validation slice of train, never on test.
"""

import numpy as np
import torch

POOLING_NAMES = ["mean_all", "mean_last32", "last_token"]


def stack_features(pairs, acts, layer: int, pooling: int):
    """Xa, Xb [n, hidden] float32 numpy + labels [n]."""
    xa = np.stack([acts[p.creative_a.id][layer, pooling].float().numpy() for p in pairs])
    xb = np.stack([acts[p.creative_b.id][layer, pooling].float().numpy() for p in pairs])
    y = np.array([p.label for p in pairs])
    return xa, xb, y


def _standardizer(pairs, acts, layer, pooling):
    """Mean/std over the unique train creatives at this (layer, pooling)."""
    seen, rows = set(), []
    for p in pairs:
        for c in (p.creative_a, p.creative_b):
            if c.id not in seen:
                seen.add(c.id)
                rows.append(acts[c.id][layer, pooling].float().numpy())
    x = np.stack(rows)
    return x.mean(axis=0), x.std(axis=0) + 1e-6


def fit_contrastive_vector(pairs, acts, layer, pooling, standardize=True):
    """Returns (vector, mu, sd). Score with score_creatives()."""
    mu, sd = _standardizer(pairs, acts, layer, pooling)
    if not standardize:
        mu, sd = np.zeros_like(mu), np.ones_like(sd)
    winners, losers = [], []
    for p in pairs:
        w = acts[p.winner.id][layer, pooling].float().numpy()
        l = acts[p.loser.id][layer, pooling].float().numpy()
        winners.append((w - mu) / sd)
        losers.append((l - mu) / sd)
    vec = np.stack(winners).mean(axis=0) - np.stack(losers).mean(axis=0)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec, mu, sd


def score_pairs(pairs, acts, layer, pooling, vec, mu, sd):
    xa, xb, y = stack_features(pairs, acts, layer, pooling)
    sa = ((xa - mu) / sd) @ vec
    sb = ((xb - mu) / sd) @ vec
    return sa, sb, y


def pairwise_accuracy(sa, sb, y) -> float:
    pred = (sb > sa).astype(int)  # label 1 means B wins
    return float((pred == y).mean())


def sweep(train_pairs, val_pairs, acts, layers=None):
    """Grid over layer x pooling x standardization, selected on val.

    Returns (best cfg dict, table rows sorted by val accuracy).
    """
    n_layers = next(iter(acts.values())).shape[0]
    layers = layers if layers is not None else range(n_layers)
    rows = []
    for layer in layers:
        for pooling in range(len(POOLING_NAMES)):
            for standardize in (True, False):
                vec, mu, sd = fit_contrastive_vector(
                    train_pairs, acts, layer, pooling, standardize)
                sa, sb, y = score_pairs(val_pairs, acts, layer, pooling, vec, mu, sd)
                rows.append({
                    "layer": int(layer),
                    "pooling": POOLING_NAMES[pooling],
                    "pooling_idx": pooling,
                    "standardize": standardize,
                    "val_acc": pairwise_accuracy(sa, sb, y),
                })
    rows.sort(key=lambda r: -r["val_acc"])
    return rows[0], rows


def separation(pairs, acts, layer, pooling, vec, mu, sd) -> float:
    """Winner-vs-loser score gap in pooled-std units (diagnostic)."""
    sa, sb, y = score_pairs(pairs, acts, layer, pooling, vec, mu, sd)
    win = np.where(y == 0, sa, sb)
    lose = np.where(y == 0, sb, sa)
    denom = np.concatenate([win, lose]).std() + 1e-8
    return float((win.mean() - lose.mean()) / denom)
