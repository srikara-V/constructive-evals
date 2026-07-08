#!/usr/bin/env python3
"""
creative-ranking-videos-2: VLM output tokens -> open-source LLM context ->
contrastive vector ranking of video ad creatives.

Pipeline (per creative):
  1. Sample N frames from the video; a VLM (Qwen2.5-VL by default) generates
     a structured description — these are the "output tokens".
  2. Whisper transcribes the audio track (--no-audio ablates this).
  3. Both are PREPENDED to the context of an open-source base LLM
     (Qwen2.5-7B by default), ending in a performance question.
  4. Pooled hidden states are extracted at every layer.

Training: contrastive vector = mean(activations | winning creatives)
                             - mean(activations | losing creatives),
with layer / pooling / standardization chosen on a validation slice of
train. Test pairs (default 500) are predicted by comparing the projection
of each side onto the vector.

Baselines reported: zero-shot judge logP("Yes")-logP("No") from the same
forward pass, and random (50%).

Usage:
  # real run on the padsplit set (GPU box)
  python experiments/creative_ranking_videos_2/run.py \
      --manifest data/padsplit_videos.json --preset padsplit_7b --n-test 500

  # CPU end-to-end check on synthetic videos (tiny real models)
  python experiments/creative_ranking_videos_2/make_synthetic_data.py
  python experiments/creative_ranking_videos_2/run.py \
      --manifest data/synthetic_videos/manifest.json --preset smoke --n-test 40

  # plumbing-only check (no model downloads)
  python experiments/creative_ranking_videos_2/run.py \
      --manifest data/synthetic_videos/manifest.json --preset mock --n-test 40
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
from video_creative_ranking import contrastive
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
    p.add_argument("--vlm-api-base", default=None)
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
        preset=args.preset,
        manifest=args.manifest,
        n_test=args.n_test,
        n_train=args.n_train,
        vlm_backend=args.vlm_backend,
        vlm_model=args.vlm_model,
        vlm_api_base=args.vlm_api_base,
        llm_model=args.llm_model,
        whisper_model=args.whisper_model,
        device=args.device,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    if args.no_audio:
        cfg.use_audio = False

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    out_dir = args.out_dir or Path(__file__).parent / "results" / cfg.manifest.stem

    print("=" * 70)
    print("CREATIVE-RANKING-VIDEOS-2  (VLM tokens -> LLM contrastive vector)")
    print("=" * 70)
    print(f"VLM: {cfg.vlm_model} ({cfg.vlm_backend})  LLM: {cfg.llm_model}")
    print(f"Audio: {'whisper ' + cfg.whisper_model if cfg.use_audio and not cfg.mock_audio else 'mock' if cfg.use_audio else 'OFF'}")
    print(f"Device: {cfg.resolved_device()}")

    # ---- Stage 1-2: data + per-creative context ingredients ----
    train_pairs, test_pairs = load_manifest(cfg)
    creatives = unique_creatives([train_pairs, test_pairs])
    captions = caption_creatives(creatives, cfg)
    transcripts = transcribe_creatives(creatives, cfg)

    ex = captions[creatives[0].id]["caption"]
    print(f"\nExample caption ({creatives[0].id}): {ex[:220]}...")
    if cfg.use_audio:
        print(f"Example transcript: {transcripts[creatives[0].id][:120]!r}")

    # ---- Stage 3-4: prepended context -> per-layer activations ----
    acts, judges = extract_activations(creatives, captions, transcripts, cfg)

    # ---- Stage 5: contrastive vector (select on val, refit on train) ----
    fit_pairs, val_pairs = split_train_val(train_pairs, cfg.val_frac, cfg.seed)
    print(f"\nSweeping layers/poolings on {len(fit_pairs)} fit / {len(val_pairs)} val pairs...")
    best, table = contrastive.sweep(fit_pairs, val_pairs, acts)
    print(f"Selected: layer {best['layer']}, pooling {best['pooling']}, "
          f"standardize={best['standardize']} (val acc {best['val_acc']*100:.1f}%)")
    print("Top-5 configs by val acc:")
    for r in table[:5]:
        print(f"  L{r['layer']:>2} {r['pooling']:<11} std={str(r['standardize']):<5} "
              f"-> {r['val_acc']*100:.1f}%")

    vec, mu, sd = contrastive.fit_contrastive_vector(
        train_pairs, acts, best["layer"], best["pooling_idx"], best["standardize"])
    sep = contrastive.separation(train_pairs, acts, best["layer"],
                                 best["pooling_idx"], vec, mu, sd)
    print(f"Train winner/loser separation: {sep:.3f} pooled-std units")

    # ---- Stage 6: test evaluation ----
    sa, sb, y = contrastive.score_pairs(test_pairs, acts, best["layer"],
                                        best["pooling_idx"], vec, mu, sd)
    reports = [accuracy_report("contrastive_vector (v2)", sa, sb, y)]

    ja = np.array([judges[p.creative_a.id] for p in test_pairs])
    jb = np.array([judges[p.creative_b.id] for p in test_pairs])
    reports.append(accuracy_report("zero-shot judge logP(Yes/No)", ja, jb, y))

    print_summary(f"test = {len(test_pairs)} pairs, manifest = {cfg.manifest.name}", reports)

    # ---- Persist ----
    all_scores = {}
    for c in creatives:
        x = acts[c.id][best["layer"], best["pooling_idx"]].float().numpy()
        all_scores[c.id] = float(((x - mu) / sd) @ vec)
    save_scores(out_dir, "v2_contrastive", all_scores)
    save_scores(out_dir, "judge", {c.id: judges[c.id] for c in creatives})
    save_results(out_dir, {
        "experiment": "creative_ranking_videos_2",
        "config": {
            "preset": args.preset, "vlm_model": cfg.vlm_model,
            "vlm_backend": cfg.vlm_backend, "llm_model": cfg.llm_model,
            "whisper_model": cfg.whisper_model if cfg.use_audio else None,
            "use_audio": cfg.use_audio, "n_frames": cfg.n_frames,
            "manifest": str(cfg.manifest), "seed": cfg.seed,
        },
        "selected": best,
        "train_separation": sep,
        "n_train_pairs": len(train_pairs),
        "reports": reports,
        "val_table_top20": table[:20],
    })

    vec_file = Path(out_dir) / "contrastive_vector.pt"
    torch.save({"vector": torch.tensor(vec), "mu": torch.tensor(mu),
                "sd": torch.tensor(sd), **best}, vec_file)
    print(f"Contrastive vector saved to {vec_file}")


if __name__ == "__main__":
    main()
