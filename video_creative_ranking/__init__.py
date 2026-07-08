"""
video_creative_ranking: shared pipeline for ranking video ad creatives.

Stages:
  1. video_io      - frame + audio extraction from video files
  2. captioner     - VLM describes the video ("output tokens")
  3. transcriber   - Whisper transcript of the audio track
  4. activations   - VLM tokens (+ transcript) prepended to an open-source LLM
                     context; pooled hidden states extracted at every layer
  5. contrastive   - winner-vs-loser contrastive (persona) vector scoring
  6. probes        - trained linear probes over the same activations
  7. eval_harness  - pairwise accuracy on held-out test pairs

Used by experiments/creative_ranking_videos_2, _3, _4.
"""

from .config import PipelineConfig, PRESETS
from .data import Creative, CreativePair, load_manifest

__all__ = [
    "PipelineConfig",
    "PRESETS",
    "Creative",
    "CreativePair",
    "load_manifest",
]
