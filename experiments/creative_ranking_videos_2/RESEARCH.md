# Research notes: predicting video ad creative performance (2024–2026)

Distilled from an arxiv/web survey run before building
`creative_ranking_videos_2/3/4`. Every claim below carries its source.

## 1. Creative/ad performance prediction from content

- **AdSEE** (KDD'23, arXiv:2309.08159) — CTR from ad cover creatives on
  Tencent QQ (20.5K ads): Spearman **0.512**, NDCG@10 0.685; editing
  creatives toward the predictor's gradient won a live A/B (p=3.7e-5).
- **Alibaba VAM + bandit** (WWW'21, arXiv:2102.04033) — the canonical
  "creative ranking" paper. Lesson: with sparse per-creative data, learn
  to **rank within product/campaign groups**, not regress absolute CTR.
  Released **CreativeRanking** (1.7M creatives, real impressions/clicks) —
  candidate pretraining corpus if padsplit data is thin.
- **Ikeda et al.** (ICPR'20, arXiv:2012.11851) — video-ad CTR from video
  CNN + text + metadata: Pearson **0.695** multimodal vs 0.487 unimodal.
- **SnapUGC engagement** (ICCV'25 workshop winner, arXiv:2508.02516) —
  120K short videos: VideoLLaMA2 with an **MLP head on pooled final-layer
  hidden states** SROCC 0.691 beats the same-model verbalized-score
  approach; content-only ceiling ≈ **SROCC 0.70**.
- **QARM** (Kuaishou, arXiv:2411.11739) — multimodal features gave only
  +0.2-0.35% offline AUC but **+9.6% online ad revenue**, concentrated in
  cold-start — content signals matter exactly when a creative is new,
  which is the creative-testing use case.

## 2. Caption bottleneck vs direct representations

- Same-backbone ablation (arXiv:2508.02516): **feature-based head beats
  token-based scoring** (SROCC 0.674 vs 0.665).
- **LinkedOut** (arXiv:2512.16891): intermediate video-LLM embeddings beat
  text summaries for recommendation (+6.4% HR@10 over the best
  end-to-end baseline); text summaries lose humor/pacing/aesthetics.
- **Language-bottleneck classifiers** (arXiv:2406.15816): caption quality
  flips the sign — detailed captioner slightly beats end-to-end vision,
  weak captioner loses by 2-6 pts. **CaBM** (arXiv:2607.00578): bottleneck
  cost grows with how perceptual/fine-grained the target is.
- Fusion of caption-based and embedding-based predictions beats either.
  → v2/v3 (tokens) and v4 (VLM-native) + fusion cover both sides.

## 3. Activation-based scoring: contrastive vectors vs probes

- **Persona vectors** (Anthropic, arXiv:2507.21509) — diff-in-means over
  residual activations, **mean-pooled over response tokens** (better than
  last-token), layer chosen per-trait by sweep; projection predicts
  behavior at r = 0.75-0.83. The direct template for v2.
- **CAA** (ACL'24, arXiv:2312.06681) — steering vectors peak at **middle
  layers** (13/32 for 7B). **RepE** (arXiv:2310.01405) — last-token +
  middle-to-late layers, PCA of contrast differences.
- **Diff-in-means vs trained probe**: mass-mean probing ≈ logistic
  regression in-distribution but **generalizes better OOD in 7/8 tests**
  (Geometry of Truth, arXiv:2310.06824); ITI (arXiv:2306.03341) works from
  ~40 labels. No published crossover N → v2 vs v3 is a real A/B.
- **ELHSR** (arXiv:2505.12225) — a single linear layer on hidden states
  (<0.005% of RM params, ~6K examples) rivals full reward models →
  v3's Bradley-Terry probe. Apollo deception probes (arXiv:2502.03407):
  linear read-outs hit 0.96+ AUROC OOD.

## 4. Audio

- **VAST** (NeurIPS'23, arXiv:2305.18500) — subtitles dominate when
  speech carries the message (+25 R@1 on narrated video), audio
  embeddings dominate on general video; V+A+S beats all pairs.
- **MindMem** (arXiv:2502.18371, ad memorability) — text 0.589 > video
  0.564 >> audio-only 0.336, but trimodal fusion 0.631: audio is weak
  alone, always additive.
- **Transcripts miss paralinguistics** (arXiv:2510.10444): tone-only
  emotion drops models to near chance. → Whisper transcript first
  (cheap, high-value for voiceover ads); CLAP (arXiv:2206.04769)
  embedding branch is the follow-up if `--no-audio` ≈ no change.

## 5. Pairwise LLM judging

- Judges flip on ~**1/3 of order swaps** (MT-Bench arXiv:2306.05685;
  arXiv:2305.17926) → always judge both orders and average.
- Read **soft log-probs, not argmax** (G-Eval arXiv:2303.16634; GenRM's
  r = p("Yes"), arXiv:2408.15240). Aggregate many pairs with a
  Bradley-Terry MLE + bootstrap (Chatbot Arena, arXiv:2403.04132);
  PairS (arXiv:2403.16950) needs only O(N log N) comparisons.
- Sobering: best MLLM judge gets **49.2%** on 4-way visual preference
  (GenAI-Arena, arXiv:2406.04485) — expect visual judging to be far
  below text-domain numbers. → v4's judge is a floor, not the method.

## 6. What accuracy to expect (pairwise, content-only)

| system | number |
|---|---|
| ImageReward (arXiv:2304.05977) | 65.1% (human-human 73.4%) |
| PickScore (arXiv:2305.01569) | 70.5% (human expert 68.0%) |
| HPSv2 (arXiv:2306.09341) | 83% in-dist → **64-66% OOD** |
| VideoScore (arXiv:2406.15252) | 78.5% pairwise on GenAI-Bench |
| SnapUGC LMMs (arXiv:2508.02516) | SROCC 0.66-0.70 |
| AdSEE ad CTR (arXiv:2309.08159) | Spearman 0.512 |

**Calibration: 65-75% pairwise accuracy on held-out padsplit pairs is a
working system; ~50-55% means the signal isn't reaching the features;
90%+ claims are out of line with everything published.** Trained scorers
lose 10-20 pts off-distribution — validate with campaign-grouped splits
and re-check after any creative-strategy shift.
