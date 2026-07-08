#!/usr/bin/env python3
"""
creative-ranking-videos-4: beyond the caption bottleneck.

v2/v3 compress the video into VLM output TOKENS before the LLM ever sees
it. Anything the description doesn't verbalize — pacing, color energy,
production quality, "scroll-stopping-ness" — is lost. The evidence says
that loss is real: on SnapUGC engagement prediction, heads on pooled VLM
hidden states beat token-based scoring on the same backbone
(arXiv:2508.02516), and LinkedOut (arXiv:2512.16891) shows video-LLM
inner representations beat text summaries for recommendation.

Methods:
  A. VLM-native activations - pooled hidden states from the VLM's language
     tower while it watches the actual frames (captured during the same
     captioning pass v2 uses, so the marginal cost is one forward per
     creative). Scored two ways: contrastive vector + Bradley-Terry probe.
  B. Order-debiased pairwise judge - both creatives' contexts in one
     prompt, margin read from logits as logP("A")-logP("B"), averaged over
     both presentation orders (position bias flips ~1/3 of LLM verdicts).
     Zero-training floor.
  C. Fusion - logistic combination of every per-creative score available
     (v4 VLM scores + v2/v3 scores.json when present) + the judge margin.
     MindMem (arXiv:2502.18371) and VAST report fusion > any single branch.

Requires vlm_backend='hf' (or 'mock') — hidden states aren't visible
through an OpenAI-compatible API.

Usage:
  python experiments/creative_ranking_videos_4/run.py \
      --manifest data/padsplit_videos.json --preset padsplit_7b --n-test 500
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video_creative_ranking.config import make_config
from video_creative_ranking.data import load_manifest, unique_creatives
from video_creative_ranking.captioner import caption_creatives
from video_creative_ranking.transcriber import transcribe_creatives
from video_creative_ranking.pairwise_judge import judge_pairs
from video_creative_ranking import contrastive, probes
from video_creative_ranking.eval_harness import (
    accuracy_report, print_summary, save_results, save_scores, split_train_val)

EXP_DIR = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--preset", default="padsplit_7b")
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--vlm-backend", default=None, choices=["hf", "mock"])
    p.add_argument("--vlm-model", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--whisper-model", default=None)
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--skip-judge", action="store_true",
                   help="skip method B (2 LLM forwards per pair)")
    p.add_argument("--extra-scores", type=Path, nargs="*", default=None,
                   help="scores.json files from other experiments to fuse "
                        "(default: sibling v2/v3 results for this manifest)")
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_extra_scores(paths, manifest_stem):
    if paths is None:
        paths = [
            EXP_DIR.parent / "creative_ranking_videos_2" / "results" / manifest_stem / "scores.json",
            EXP_DIR.parent / "creative_ranking_videos_3" / "results" / manifest_stem / "scores.json",
        ]
    merged: dict[str, dict[str, float]] = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"  (no extra scores at {path})")
            continue
        with open(path) as f:
            data = json.load(f)
        for cid, methods in data.items():
            merged.setdefault(cid, {}).update(methods)
        print(f"  fused scores from {path}")
    return merged


def fusion_features(pairs, score_dicts, methods, judge_margins):
    """Per-pair feature = per-method score diff (a-b) [+ judge margin]."""
    feats = []
    for i, p in enumerate(pairs):
        row = [score_dicts[p.creative_a.id][m] - score_dicts[p.creative_b.id][m]
               for m in methods]
        if judge_margins is not None:
            row.append(judge_margins[i])
        feats.append(row)
    return np.array(feats)


def main():
    args = parse_args()
    cfg = make_config(
        preset=args.preset, manifest=args.manifest, n_test=args.n_test,
        n_train=args.n_train, vlm_backend=args.vlm_backend,
        vlm_model=args.vlm_model, llm_model=args.llm_model,
        whisper_model=args.whisper_model, device=args.device,
        cache_dir=args.cache_dir, seed=args.seed,
    )
    cfg.capture_vlm_hidden = True
    if args.no_audio:
        cfg.use_audio = False
    if cfg.vlm_backend == "openai_compat":
        raise SystemExit("v4 needs local VLM hidden states; use --vlm-backend hf")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    out_dir = args.out_dir or EXP_DIR / "results" / cfg.manifest.stem

    print("=" * 70)
    print("CREATIVE-RANKING-VIDEOS-4  (VLM-native activations + judge + fusion)")
    print("=" * 70)

    train_pairs, test_pairs = load_manifest(cfg)
    creatives = unique_creatives([train_pairs, test_pairs])
    captions = caption_creatives(creatives, cfg, capture_hidden=True)
    transcripts = transcribe_creatives(creatives, cfg)

    # ---- Method A: VLM-native activations ----
    acts_vlm = {c.id: torch.load(captions[c.id]["vlm_hidden_path"], weights_only=True)
                for c in creatives}
    n_layers, _, hidden = next(iter(acts_vlm.values())).shape
    print(f"\nVLM activations: {n_layers} layers x 3 poolings x {hidden} dims")

    fit_pairs, val_pairs = split_train_val(train_pairs, cfg.val_frac, cfg.seed)
    reports, selected = [], {}

    best_c, table_c = contrastive.sweep(fit_pairs, val_pairs, acts_vlm)
    print(f"VLM contrastive: L{best_c['layer']} {best_c['pooling']} "
          f"std={best_c['standardize']} (val {best_c['val_acc']*100:.1f}%)")
    vec, mu, sd = contrastive.fit_contrastive_vector(
        train_pairs, acts_vlm, best_c["layer"], best_c["pooling_idx"], best_c["standardize"])
    sa, sb, y = contrastive.score_pairs(test_pairs, acts_vlm, best_c["layer"],
                                        best_c["pooling_idx"], vec, mu, sd)
    reports.append(accuracy_report("vlm_native_contrastive (v4)", sa, sb, y))
    selected["vlm_contrastive"] = best_c

    best_bt, _ = probes.sweep_probe(fit_pairs, val_pairs, acts_vlm, mode="bt")
    print(f"VLM BT probe:    L{best_bt['layer']} {best_bt['pooling']} "
          f"l2={best_bt['C']} (val {best_bt['val_acc']*100:.1f}%)")
    w_bt, mu_bt, sd_bt = probes.fit_final(train_pairs, acts_vlm, best_bt, mode="bt")
    lb, pb = best_bt["layer"], best_bt["pooling_idx"]
    xa, xb, _ = contrastive.stack_features(test_pairs, acts_vlm, lb, pb)
    reports.append(accuracy_report("vlm_native_bt_probe (v4)",
                                   ((xa - mu_bt) / sd_bt) @ w_bt,
                                   ((xb - mu_bt) / sd_bt) @ w_bt, y))
    selected["vlm_bt_probe"] = best_bt

    # per-creative v4 scores (used below for fusion, saved for later runs)
    v4_scores = {"v4_vlm_contrastive": {}, "v4_vlm_bt": {}}
    for c in creatives:
        xc = acts_vlm[c.id][best_c["layer"], best_c["pooling_idx"]].float().numpy()
        v4_scores["v4_vlm_contrastive"][c.id] = float(((xc - mu) / sd) @ vec)
        xb_ = acts_vlm[c.id][lb, pb].float().numpy()
        v4_scores["v4_vlm_bt"][c.id] = float(((xb_ - mu_bt) / sd_bt) @ w_bt)

    # ---- Method B: order-debiased pairwise judge ----
    judge_test = judge_train = None
    if not args.skip_judge:
        judge_test = judge_pairs(test_pairs, captions, transcripts, cfg)
        m = np.array(judge_test)
        reports.append(accuracy_report("pairwise_judge_debiased (v4)", m, np.zeros_like(m), y))

    # ---- Method C: fusion ----
    print("\nCollecting scores for fusion...")
    score_dicts = load_extra_scores(args.extra_scores, cfg.manifest.stem)
    for method, per_creative in v4_scores.items():
        for cid, s in per_creative.items():
            score_dicts.setdefault(cid, {})[method] = s

    all_ids = {c.id for c in creatives}
    methods = sorted({m for cid in all_ids for m in score_dicts.get(cid, {})
                      if all(m in score_dicts.get(i, {}) for i in all_ids)})
    if methods:
        if not args.skip_judge:
            judge_train = judge_pairs(train_pairs, captions, transcripts, cfg)
        Xtr = fusion_features(train_pairs, score_dicts, methods, judge_train)
        ytr = np.array([p.label for p in train_pairs])
        Xte = fusion_features(test_pairs, score_dicts, methods, judge_test)
        # per-column scale so small-magnitude signals (judge margins) compete
        # with large ones (contrastive projections) under the L2 penalty
        col_sd = Xtr.std(axis=0) + 1e-8
        Xtr, Xte = Xtr / col_sd, Xte / col_sd
        # features are favor-A margins, label 1 = B wins; symmetrize so a
        # swapped pair (negated margins) gets the flipped label
        clf = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=False)
        clf.fit(np.concatenate([Xtr, -Xtr]), np.concatenate([ytr, 1 - ytr]))
        prob_b = clf.predict_proba(Xte)[:, 1]
        feat_names = methods + ([] if args.skip_judge else ["pairwise_judge"])
        reports.append(accuracy_report(f"fusion[{len(feat_names)} signals] (v4)",
                                       1.0 - prob_b, prob_b, y))
        print(f"Fusion signals: {feat_names}")
        # -coef: positive number = signal pushes toward the higher-scored side
        weights = {n: round(float(w), 3) for n, w in zip(feat_names, -clf.coef_[0])}
        print(f"Fusion weights: {weights}")
        selected["fusion_signals"] = feat_names
    else:
        print("  No common scores across all creatives — skipping fusion "
              "(run v2 and v3 on this manifest first).")

    # reference: what v2/v3 reported on this manifest, for one-glance compare
    for prev in ("creative_ranking_videos_2", "creative_ranking_videos_3"):
        rp = EXP_DIR.parent / prev / "results" / cfg.manifest.stem / "results.json"
        if rp.exists():
            with open(rp) as f:
                for r in json.load(f).get("reports", []):
                    print(f"  [{prev}] {r['method']}: {r['accuracy']*100:.1f}%")

    print_summary(f"test = {len(test_pairs)} pairs, manifest = {cfg.manifest.name}", reports)

    for method, per_creative in v4_scores.items():
        save_scores(out_dir, method, per_creative)
    save_results(out_dir, {
        "experiment": "creative_ranking_videos_4",
        "config": {
            "preset": args.preset, "vlm_model": cfg.vlm_model,
            "vlm_backend": cfg.vlm_backend, "llm_model": cfg.llm_model,
            "use_audio": cfg.use_audio, "manifest": str(cfg.manifest),
            "seed": cfg.seed,
        },
        "selected": selected,
        "reports": reports,
        "vlm_contrastive_val_table_top10": table_c[:10],
    })


if __name__ == "__main__":
    main()
