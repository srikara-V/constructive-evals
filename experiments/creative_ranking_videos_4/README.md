# creative-ranking-videos-4

Attacks the main structural weakness of v2/v3: the **caption bottleneck**.
There, the LLM only ever sees the VLM's output tokens — pacing, color
energy, production quality, and "scroll-stopping-ness" survive only if the
caption happens to verbalize them. Evidence that this loses real signal:
feature-based heads on pooled VLM hidden states beat token-based scoring
on the same backbone for engagement prediction (SnapUGC winner,
arXiv:2508.02516), and video-LLM inner representations beat text summaries
for recommendation (LinkedOut, arXiv:2512.16891).

Methods:

- **A. VLM-native activations** — pooled hidden states from the VLM's
  language tower *while it watches the actual frames*, captured during the
  same captioning pass v2 uses (cached; marginal cost ≈ one forward per
  creative). Scored with the contrastive vector and the Bradley-Terry
  probe. Requires `--vlm-backend hf` (an API can't expose hidden states).
- **B. Order-debiased pairwise judge** — both creatives' contexts in one
  prompt, "which gets the higher CTR?", margin read from the logits as
  soft `logP("A") − logP("B")` (G-Eval arXiv:2303.16634, GenRM
  arXiv:2408.15240), averaged over **both presentation orders** because
  position bias flips ~1/3 of LLM verdicts (arXiv:2306.05685,
  arXiv:2305.17926). Zero training — the floor every trained method must
  beat. `--skip-judge` disables it (it costs 2 LLM forwards per pair).
- **C. Fusion** — logistic combination of every per-creative score
  available (v4's two + whatever v2/v3 wrote to their `scores.json` for
  this manifest) plus the judge margin, symmetrized and column-scaled.
  Multimodal fusion reliably beats single branches on ad-adjacent tasks
  (MindMem arXiv:2502.18371, VAST arXiv:2305.18500). Fusion weights are
  fit on train pairs whose base scores are in-sample, so treat its edge
  as an upper bound unless re-validated.

```bash
# run v2 and v3 first so fusion has their scores (optional but better)
python experiments/creative_ranking_videos_4/run.py \
    --manifest data/padsplit_videos.json --preset padsplit_7b --n-test 500
```

Smoke verification (synthetic videos, real tiny models — full table in
`../creative_ranking_videos_2/README.md`): VLM-native activations hit
**95.2%** vs 71.4% for every caption-bottleneck method, reproducing the
published ordering end-to-end. The pairwise judge collapsed to 28.6% at
0.5B scale (two verbose captions overwhelm a tiny base model) — treat it
as meaningful only with ≥7B judges.

If v4A beats v2/v3 on padsplit, the caption bottleneck was the problem —
next steps would be bigger VLM contexts (more frames), CLAP audio
embeddings as a fusion branch, or fine-tuning the VLM head directly
(the SnapUGC recipe). If nothing beats ~55%, the honest conclusion is that
creative content alone isn't predictive enough on this account's data —
check label quality (impression counts, tie filtering) before blaming
the models, and prefer campaign-grouped pairs (`campaign` field) so the
model isn't asked to compare ads shown to different audiences.
