# Khan Academy-style methodology video

`output/ranking_methodology_khan_academy.mp4` — a ~12.5 minute, fully generated
explainer that builds up this repo's content-ranking methodology from first
principles, in the style of a Khan Academy lesson: dark board, a male narrator,
and hand-drawn diagrams that appear in sync with the narration.

## What the video covers

1. **The problem** — pairwise ranking of content (which of two posts wins?).
2. **Why not just ask the model** — direct A/B answers: 62.3% on LinkedIn,
   ~56.5% on same-author tweet pairs (near coin flip).
3. **Activations** — the residual stream; hooking layer −8; mean-pooling
   response tokens into one snapshot vector `h`.
4. **Contrastive (persona) vectors** — `v = μ⁺ − μ⁻` from 161 correct vs 39
   incorrect training answers; ~3.05 separation on the projection.
5. **Scoring by projection** — `s = h · v` as a free, untrained verifier
   (a linear probe).
6. **Best-of-N sampling** — persona-scored sampling vs majority voting on the
   tweet benchmark (56.5% → 58%).
7. **Power persona sampling** — Metropolis–Hastings over chain-of-thought
   trajectories with suffix-resampling proposals and acceptance rule
   `log r = (α−1)·Δlog p + β·Δs`, targeting `π(y) ∝ p(y|x)^α · e^{β·s(y)}`;
   LinkedIn 62.3% → **88.1%** (95 flips to correct vs 13 to wrong).
8. **Images & video** — the same recipe through a vision-language model
   front end; benchmarked across 400+ images; video via frames.
9. **Recap** — subtraction + dot product.

All quantitative claims come from `experiments/persona_linkedin/results.txt`,
`experiments/persona_tweet_virality/results.json`, and the code in
`power_persona_sampling/`.

## Pipeline

| file | role |
|---|---|
| `narration.py` | the full scene-by-scene narration script |
| `el_audio.py` | ElevenLabs TTS (voice: George, male) via the `with-timestamps` endpoint → per-scene `audio/*.mp3` + character-level alignment JSON |
| `renderer.py` | chalkboard engine: hand-drawn wobbly strokes, progressive reveals, mixed handwriting/symbol fonts, 2× supersampled anti-aliasing |
| `scenes.py` | the choreography: every diagram element is anchored to an exact phrase of the narration and starts drawing when the narrator reaches it |
| `build.py` | orchestrator: `check` (validate + preview stills), `scene <id>`, `all` (render + concat) |

Rebuilding from scratch:

```bash
export ELEVEN_LABS_API=...   # ElevenLabs key
pip install pillow numpy      # plus ffmpeg on the system
cd video
python3 el_audio.py           # regenerate narration + alignments
python3 build.py check        # validate anchors, write preview stills
python3 build.py all          # render all scenes + concat final mp4
```

Fonts (`assets/fonts/`): Kalam and Patrick Hand, from Google Fonts, licensed
under the SIL Open Font License.
