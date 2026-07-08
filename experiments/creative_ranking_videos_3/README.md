# creative-ranking-videos-3

Trained linear probes over the **same** cached activations as
`creative_ranking_videos_2` (VLM tokens + transcript prepended to an
open-source LLM). Zero new featurization cost — if v2's caches exist,
this runs in seconds/minutes.

The v2 contrastive vector is a difference of class means: it cannot
downweight noisy dimensions or use feature covariance. With a few hundred
labeled pairs, discriminatively trained linear read-outs are the standard
upgrade (ELHSR, arXiv:2505.12225, shows a single linear layer over hidden
states rivals full reward models; Apollo's deception probes hit 0.96+
AUROC, arXiv:2502.03407). Diff-in-means still tends to transfer better
out-of-distribution (arXiv:2310.06824), so both are reported side by side
on the identical split.

Methods (layer / pooling / regularization all selected on validation):

1. **delta probe** — logistic regression on `z_a − z_b`, symmetrized so
   A/B position carries no information.
2. **independent scorer** — logistic winner=1/loser=0 probe; scores one
   creative at a time, so it can rank a whole candidate set in production.
3. **Bradley-Terry probe** — linear scorer trained with the reward-model
   loss `−log σ(s_winner − s_loser)`; uses pairwise labels as pairwise
   information instead of flattening them.
4. v2 contrastive vector + zero-shot judge, re-run for reference.

```bash
python experiments/creative_ranking_videos_3/run.py \
    --manifest data/padsplit_videos.json --preset padsplit_7b --n-test 500
```

Reading padsplit results: `probes >> contrastive` → the direction exists
but mean-difference is too blunt (keep probes). `everything ≈ 50%` → the
signal didn't survive the caption bottleneck → `creative_ranking_videos_4`.
`contrastive > probes` → you are label-starved; get more pairs before
trusting either.

Smoke verification (synthetic videos, real tiny models — see the table in
`../creative_ranking_videos_2/README.md`): independent probe and BT probe
71.4%, matching the contrastive vector, all well above the 50% floor.
With only ~38 train pairs the probes can't beat diff-in-means yet — the
in-distribution probe advantage needs a few hundred pairs, which padsplit
should provide.
