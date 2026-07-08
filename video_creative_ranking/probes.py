"""Trained linear probes over the same cached activations (v3).

Two probe families, both selected on the validation slice:

  delta probe        - logistic regression on (z_a - z_b) -> P(B wins),
                       trained symmetrized so A/B position carries no signal.
  independent scorer - logistic regression winner=1 / loser=0 per creative;
                       scores creatives one at a time, so it ranks arbitrary
                       candidate sets in production (not just pairs).

A trained probe uses the same features as the contrastive vector but fits
direction AND per-dimension weighting discriminatively, which in the probing
literature reliably beats the raw difference-of-means when a few hundred
labeled examples are available.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from .contrastive import POOLING_NAMES, _standardizer, pairwise_accuracy, stack_features

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3]
BT_L2_GRID = [1e-3, 1e-2, 1e-1, 1.0]


def _fit_delta(train_pairs, acts, layer, pooling, C):
    mu, sd = _standardizer(train_pairs, acts, layer, pooling)
    xa, xb, y = stack_features(train_pairs, acts, layer, pooling)
    d = ((xa - mu) / sd) - ((xb - mu) / sd)
    # symmetrize: swapping A and B must flip the prediction
    X = np.concatenate([d, -d])
    yy = np.concatenate([y, 1 - y])
    clf = LogisticRegression(C=C, max_iter=2000, fit_intercept=False)
    clf.fit(X, yy)
    return clf, mu, sd


def _eval_delta(clf, mu, sd, pairs, acts, layer, pooling):
    xa, xb, y = stack_features(pairs, acts, layer, pooling)
    d = ((xa - mu) / sd) - ((xb - mu) / sd)
    prob_b = clf.predict_proba(d)[:, 1]
    # pred = B wins iff P(B) > P(A); pairwise_accuracy compares "scores"
    return pairwise_accuracy(1.0 - prob_b, prob_b, y), prob_b


def _fit_independent(train_pairs, acts, layer, pooling, C):
    mu, sd = _standardizer(train_pairs, acts, layer, pooling)
    X, y = [], []
    for p in train_pairs:
        X.append((acts[p.winner.id][layer, pooling].float().numpy() - mu) / sd)
        y.append(1)
        X.append((acts[p.loser.id][layer, pooling].float().numpy() - mu) / sd)
        y.append(0)
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(np.stack(X), np.array(y))
    return clf, mu, sd


def _eval_independent(clf, mu, sd, pairs, acts, layer, pooling):
    xa, xb, y = stack_features(pairs, acts, layer, pooling)
    sa = clf.decision_function((xa - mu) / sd)
    sb = clf.decision_function((xb - mu) / sd)
    return pairwise_accuracy(sa, sb, y), sa, sb


def _fit_bt(train_pairs, acts, layer, pooling, l2):
    """Linear scorer trained with the Bradley-Terry pairwise loss
    -log sigmoid(s_winner - s_loser)  (ELHSR-style reward probing).

    Uses the pairwise labels exactly as pairwise information instead of
    flattening to winner=1/loser=0, and is symmetric by construction.
    """
    mu, sd = _standardizer(train_pairs, acts, layer, pooling)
    w_rows = np.stack([(acts[p.winner.id][layer, pooling].float().numpy() - mu) / sd
                       for p in train_pairs])
    l_rows = np.stack([(acts[p.loser.id][layer, pooling].float().numpy() - mu) / sd
                       for p in train_pairs])
    D = torch.tensor(w_rows - l_rows, dtype=torch.float32)
    w = torch.zeros(D.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=0.5, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = -F.logsigmoid(D @ w).mean() + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), mu, sd


def _eval_bt(w, mu, sd, pairs, acts, layer, pooling):
    xa, xb, y = stack_features(pairs, acts, layer, pooling)
    sa = ((xa - mu) / sd) @ w
    sb = ((xb - mu) / sd) @ w
    return pairwise_accuracy(sa, sb, y), sa, sb


def sweep_probe(train_pairs, val_pairs, acts, mode="delta", layers=None):
    """Grid over layer x pooling x C, selected on val accuracy."""
    n_layers = next(iter(acts.values())).shape[0]
    layers = layers if layers is not None else range(n_layers)
    rows = []
    grid = BT_L2_GRID if mode == "bt" else C_GRID
    for layer in layers:
        for pooling in range(len(POOLING_NAMES)):
            for C in grid:
                if mode == "delta":
                    clf, mu, sd = _fit_delta(train_pairs, acts, layer, pooling, C)
                    acc, _ = _eval_delta(clf, mu, sd, val_pairs, acts, layer, pooling)
                elif mode == "bt":
                    w, mu, sd = _fit_bt(train_pairs, acts, layer, pooling, C)
                    acc, _, _ = _eval_bt(w, mu, sd, val_pairs, acts, layer, pooling)
                else:
                    clf, mu, sd = _fit_independent(train_pairs, acts, layer, pooling, C)
                    acc, _, _ = _eval_independent(clf, mu, sd, val_pairs, acts, layer, pooling)
                rows.append({"layer": int(layer), "pooling": POOLING_NAMES[pooling],
                             "pooling_idx": pooling, "C": C, "val_acc": acc})
    rows.sort(key=lambda r: -r["val_acc"])
    return rows[0], rows


def fit_final(pairs, acts, best, mode="delta"):
    """Refit the selected probe on the full training set."""
    layer, pooling, C = best["layer"], best["pooling_idx"], best["C"]
    if mode == "delta":
        clf, mu, sd = _fit_delta(pairs, acts, layer, pooling, C)
    elif mode == "bt":
        clf, mu, sd = _fit_bt(pairs, acts, layer, pooling, C)
    else:
        clf, mu, sd = _fit_independent(pairs, acts, layer, pooling, C)
    return clf, mu, sd


def eval_final(clf, mu, sd, pairs, acts, best, mode="delta"):
    layer, pooling = best["layer"], best["pooling_idx"]
    if mode == "delta":
        acc, prob_b = _eval_delta(clf, mu, sd, pairs, acts, layer, pooling)
        return acc, {"prob_b": prob_b.tolist()}
    if mode == "bt":
        acc, sa, sb = _eval_bt(clf, mu, sd, pairs, acts, layer, pooling)
        return acc, {"score_a": sa.tolist(), "score_b": sb.tolist()}
    acc, sa, sb = _eval_independent(clf, mu, sd, pairs, acts, layer, pooling)
    return acc, {"score_a": sa.tolist(), "score_b": sb.tolist()}
