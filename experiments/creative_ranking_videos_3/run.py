#!/usr/bin/env python3
"""
creative-ranking-videos-3: trained linear probes over the same
VLM-tokens-in-LLM-context activations used by creative-ranking-videos-2.

Motivation: the v2 contrastive vector is the difference of class means —
it ignores feature covariance and cannot downweight noisy dimensions. With
a few hundred labeled pairs, a regularized logistic probe on the identical
activations is strictly more expressive and, in the probing literature,
reliably stronger. If v2 underperforms on padsplit, this is the first
thing to try because it costs nothing new: all captions, transcripts, and
activations are reused from the shared cache.

Methods evaluated on the same test pairs:
  1. delta probe        - logistic regression on (z_a - z_b), symmetrized
  2. independent scorer - logistic winner-vs-loser probe; scores a single
                          creative, so it also ranks new candidate sets
  3. Bradley-Terry probe- linear scorer trained with -log sigmoid(s_w - s_l),
                          the reward-model loss (ELHSR, arXiv:2505.12225,
                          shows linear heads on hidden states rival full RMs)
  4. v2 contrastive     - re-run here on the identical split for a fair
                          side-by-side
  5. zero-shot judge    - logP("Yes") - logP("No") baseline

Usage (same flags as v2):
  python experiments/creative_ranking_videos_3/run.py \
      --manifest data/padsplit_videos.json --preset padsplit_7b --n-test 500
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video_creative_ranking.config import make_config
from video_creative_ranking.data import load_manifest, unique_creatives
from video_creative_ranking.captioner import caption_creatives
from video_creative_ranking.transcriber import transcribe_creatives
from video_creative_ranking.activations import extract_activations
from video_creative_ranking import contrastive, probes
from video_creative_ranking.eval_harness import (
    accuracy_report, print_summary, save_results, save_scores, split_train_val)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--preset", default="padsplit_7b")
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--vlm-backend", default=None, choices=["hf", "openai_compat", "mock"])
    p.add_argument("--vlm-model", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--whisper-model", default=None)
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = make_config(
        preset=args.preset, manifest=args.manifest, n_test=args.n_test,
        n_train=args.n_train, vlm_backend=args.vlm_backend,
        vlm_model=args.vlm_model, llm_model=args.llm_model,
        whisper_model=args.whisper_model, device=args.device,
        cache_dir=args.cache_dir, seed=args.seed,
    )
    if args.no_audio:
        cfg.use_audio = False

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    out_dir = args.out_dir or Path(__file__).parent / "results" / cfg.manifest.stem

    print("=" * 70)
    print("CREATIVE-RANKING-VIDEOS-3  (linear probes on v2 activations)")
    print("=" * 70)

    # Identical featurization to v2 — served from the shared cache when
    # v2 already ran on this manifest.
    train_pairs, test_pairs = load_manifest(cfg)
    creatives = unique_creatives([train_pairs, test_pairs])
    captions = caption_creatives(creatives, cfg)
    transcripts = transcribe_creatives(creatives, cfg)
    acts, judges = extract_activations(creatives, captions, transcripts, cfg)

    fit_pairs, val_pairs = split_train_val(train_pairs, cfg.val_frac, cfg.seed)
    print(f"\nFit {len(fit_pairs)} / val {len(val_pairs)} / test {len(test_pairs)} pairs")

    reports, selected = [], {}

    # ---- 1. delta probe ----
    best_d, table_d = probes.sweep_probe(fit_pairs, val_pairs, acts, mode="delta")
    print(f"\nDelta probe:       L{best_d['layer']} {best_d['pooling']} C={best_d['C']} "
          f"(val {best_d['val_acc']*100:.1f}%)")
    clf, mu, sd = probes.fit_final(train_pairs, acts, best_d, mode="delta")
    acc_d, _ = probes.eval_final(clf, mu, sd, test_pairs, acts, best_d, mode="delta")
    xa, xb, y = contrastive.stack_features(test_pairs, acts,
                                           best_d["layer"], best_d["pooling_idx"])
    d = ((xa - mu) / sd) - ((xb - mu) / sd)
    prob_b = clf.predict_proba(d)[:, 1]
    reports.append(accuracy_report("delta_probe (v3)", 1.0 - prob_b, prob_b, y))
    selected["delta_probe"] = best_d

    # ---- 2. independent scorer ----
    best_i, table_i = probes.sweep_probe(fit_pairs, val_pairs, acts, mode="independent")
    print(f"Independent probe: L{best_i['layer']} {best_i['pooling']} C={best_i['C']} "
          f"(val {best_i['val_acc']*100:.1f}%)")
    clf_i, mu_i, sd_i = probes.fit_final(train_pairs, acts, best_i, mode="independent")
    li, pi = best_i["layer"], best_i["pooling_idx"]
    xa, xb, y = contrastive.stack_features(test_pairs, acts, li, pi)
    sa_i = clf_i.decision_function((xa - mu_i) / sd_i)
    sb_i = clf_i.decision_function((xb - mu_i) / sd_i)
    reports.append(accuracy_report("independent_probe (v3)", sa_i, sb_i, y))
    selected["independent_probe"] = best_i

    # ---- 3. Bradley-Terry probe ----
    best_bt, table_bt = probes.sweep_probe(fit_pairs, val_pairs, acts, mode="bt")
    print(f"BT probe:          L{best_bt['layer']} {best_bt['pooling']} l2={best_bt['C']} "
          f"(val {best_bt['val_acc']*100:.1f}%)")
    w_bt, mu_bt, sd_bt = probes.fit_final(train_pairs, acts, best_bt, mode="bt")
    lb, pb = best_bt["layer"], best_bt["pooling_idx"]
    xa, xb, y = contrastive.stack_features(test_pairs, acts, lb, pb)
    sa_bt = ((xa - mu_bt) / sd_bt) @ w_bt
    sb_bt = ((xb - mu_bt) / sd_bt) @ w_bt
    reports.append(accuracy_report("bradley_terry_probe (v3)", sa_bt, sb_bt, y))
    selected["bt_probe"] = best_bt

    # ---- 4. v2 contrastive on the same split, for reference ----
    best_c, _ = contrastive.sweep(fit_pairs, val_pairs, acts)
    vec, mu_c, sd_c = contrastive.fit_contrastive_vector(
        train_pairs, acts, best_c["layer"], best_c["pooling_idx"], best_c["standardize"])
    sa, sb, y = contrastive.score_pairs(test_pairs, acts, best_c["layer"],
                                        best_c["pooling_idx"], vec, mu_c, sd_c)
    reports.append(accuracy_report("contrastive_vector (v2 ref)", sa, sb, y))
    selected["contrastive"] = best_c

    # ---- 5. judge baseline ----
    ja = np.array([judges[p.creative_a.id] for p in test_pairs])
    jb = np.array([judges[p.creative_b.id] for p in test_pairs])
    reports.append(accuracy_report("zero-shot judge logP(Yes/No)", ja, jb, y))

    print_summary(f"test = {len(test_pairs)} pairs, manifest = {cfg.manifest.name}", reports)

    # per-creative scores from the two deployable scorers
    all_scores, bt_scores = {}, {}
    for c in creatives:
        x = acts[c.id][li, pi].float().numpy()
        all_scores[c.id] = float(clf_i.decision_function(((x - mu_i) / sd_i)[None])[0])
        xb_ = acts[c.id][lb, pb].float().numpy()
        bt_scores[c.id] = float(((xb_ - mu_bt) / sd_bt) @ w_bt)
    save_scores(out_dir, "v3_independent_probe", all_scores)
    save_scores(out_dir, "v3_bt_probe", bt_scores)
    save_results(out_dir, {
        "experiment": "creative_ranking_videos_3",
        "config": {
            "preset": args.preset, "vlm_model": cfg.vlm_model,
            "llm_model": cfg.llm_model, "use_audio": cfg.use_audio,
            "manifest": str(cfg.manifest), "seed": cfg.seed,
        },
        "selected": selected,
        "reports": reports,
        "delta_val_table_top10": table_d[:10],
        "independent_val_table_top10": table_i[:10],
        "bt_val_table_top10": table_bt[:10],
    })


if __name__ == "__main__":
    main()
