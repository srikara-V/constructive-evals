# creative-ranking-videos-2

Predict which video ad creative will perform better ("human simulation for
videos") — without showing the video to a human or running the ad.

## Method

```
video.mp4 ──► frames (16, uniform) ──► VLM (Qwen2.5-VL) ──► output tokens
         └──► audio track ──────────► Whisper ───────────► transcript
                                                               │
        ┌──────────────────────────────────────────────────────┘
        ▼
  [VIDEO — description] + [AUDIO — transcript] + performance question
        │            (prepended to the context of an open-source LLM)
        ▼
  Qwen2.5-7B (base) forward pass ──► pooled hidden states at every layer
        ▼
  contrastive vector = mean(acts | winners) − mean(acts | losers)
        ▼
  score(creative) = projection onto vector;  rank / compare pairs
```

Same recipe as the repo's persona vectors (`experiments/persona_linkedin`,
`experiments/persona_tweet_virality`), pointed at video creatives: the
LLM never sees pixels, only the VLM's output tokens and the transcript,
and the direction that separates winning from losing creatives in
activation space is the ranking signal.

Layer, pooling (`mean_all` / `mean_last32` / `last_token`), and
standardization are selected on a validation slice of train (15%), then
the vector is refit on all train pairs and evaluated once on the test
pairs (default 500). Reported alongside: a zero-shot judge baseline
`logP("Yes") − logP("No")` on the same context, and bootstrap 95% CIs.

## Running on padsplit

1. Build a manifest. Flat format (pairs are constructed automatically,
   split at the creative level so no video leaks train→test):

```json
{
  "creatives": [
    {
      "id": "ps_0001",
      "video_path": "videos/hook_variant_a.mp4",
      "impressions": 48213,
      "clicks": 1103,
      "campaign": "padsplit_hosts_q2",
      "brand": "PadSplit"
    },
    ...
  ]
}
```

   Relative `video_path`s resolve against the manifest's directory.
   Also accepted: `conversions`, or any precomputed metric field
   (select with `--metric`, default `ctr` = smoothed clicks/impressions).
   Alternatively supply explicit `{"train": [...], "test": [...]}` pairs
   with `creative_a` / `creative_b` / `label` (0 = A won), mirroring
   `data/tweet_pairs.json`.

2. On the GPU box:

```bash
pip install -r video_creative_ranking/requirements.txt
python experiments/creative_ranking_videos_2/run.py \
    --manifest data/padsplit_videos.json --preset padsplit_7b --n-test 500
```

Useful flags:

| flag | effect |
|---|---|
| `--preset padsplit_3b` | 3B models everywhere (fits ~8 GB VRAM) |
| `--preset padsplit_7b_instruct` | instruct LLM + chat template instead of base |
| `--no-audio` | ablate the transcript (measures what audio adds) |
| `--vlm-backend openai_compat --vlm-api-base http://host:8000/v1` | caption through vLLM serving any VLM |
| `--metric conversion_rate` | rank by conversions instead of CTR |

Captions, transcripts, and activations are cached under
`cache/video_creative_ranking/` keyed by video content hash + model, so
re-runs and the v3/v4 experiments only pay for the stages they change.

## Verification runs (synthetic videos, CPU)

`make_synthetic_data.py` builds 20 short mp4s with a planted quality
signal (bright/fast/"SALE 50% OFF"/energetic audio = high CTR) plus
noise; 38 train / 21 test pairs, no video shared across the split. The
whole v2→v3→v4 suite was run end-to-end twice before commit:

| method | mock preset | smoke preset (real models¹) |
|---|---|---|
| v2 contrastive vector | 85.7% | 71.4% |
| v3 independent probe | 85.7% | 71.4% |
| v3 Bradley-Terry probe | 71.4% | 71.4% |
| v3 delta probe | 81.0% | 61.9% |
| v4 VLM-native contrastive | n/a² | **95.2%** |
| v4 VLM-native BT probe | n/a² | **95.2%** |
| v4 fusion (7 signals) | 81.0% | 90.5% |
| v4 pairwise judge (debiased) | 81.0% | 28.6%³ |
| zero-shot Yes/No judge | 52.4% | 66.7% |
| random | 50.0% | 50.0% |

¹ SmolVLM2-256M-Video watching the frames, whisper-tiny, Qwen2.5-0.5B
base — the identical code path as `padsplit_7b`, only smaller checkpoints.
² the mock backend fabricates pseudo hidden states; numbers meaningless.
³ tiny base LLMs can't compare two verbose captions — expect this
baseline to behave only at 7B scale.

At real-model scale the ordering matched the literature exactly:
VLM-native activations (v4) > caption-bottleneck methods (v2/v3) >
zero-shot judges. On 21 pairs the CIs are wide (±~19 pts) — these runs
verify the machinery, not padsplit performance.

## What to expect (calibration from the literature)

Content-only ranking of real ad creatives tops out around **65–75%
pairwise accuracy / Spearman ≈ 0.5–0.7** in published work: multimodal
CTR prediction r = 0.695 (arXiv:2012.11851), AdSEE Spearman 0.512
(arXiv:2309.08159), SnapUGC engagement SROCC ≈ 0.70 (arXiv:2508.02516),
human pairwise agreement itself 68–81%. If padsplit numbers land there,
that is a working system; if they sit at ~50–55%, see
`creative_ranking_videos_3` (trained probes — strictly stronger use of the
same activations) and `_4` (VLM-native activations that skip the caption
bottleneck, an order-debiased pairwise judge, and score fusion).

Two design choices worth knowing:
- **Diff-in-means vs trained probe**: the contrastive vector is worst-case
  optimal and generalizes better out-of-distribution (arXiv:2310.06824,
  arXiv:2306.03341); trained probes usually win in-distribution once a few
  hundred pairs exist. No published crossover — v3 A/Bs it on padsplit.
- **Pooling**: persona-vector work (arXiv:2507.21509) found mean-pooling
  over content tokens beats last-token for direction extraction; the sweep
  covers both, so the choice is made by validation, not by us.

Audio note: the transcript pathway matters most when ads are
voiceover-led (VAST, arXiv:2305.18500); music/mood-led creatives lose
information in any transcript (arXiv:2510.10444). If `--no-audio` barely
moves padsplit accuracy, try a CLAP audio-embedding branch fused in v4
rather than a bigger Whisper.
