#!/usr/bin/env python3
"""Quick test that the video creative ranking setup works.

Checks imports, video encode/decode roundtrip, mock caption/transcript,
and (optionally, --models) that the configured VLM/LLM/Whisper load and
run one tiny forward pass.

    python experiments/creative_ranking_videos_2/test_setup.py
    python experiments/creative_ranking_videos_2/test_setup.py --models --preset smoke
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("=" * 50)
print("Testing imports...")
try:
    import av  # noqa: F401
    import imageio_ffmpeg  # noqa: F401
    import numpy as np
    import sklearn  # noqa: F401
    import torch
    import transformers

    from video_creative_ranking.config import make_config
    print(f"  torch {torch.__version__}, transformers {transformers.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print("  Imports: OK")
except Exception as e:
    print(f"  Imports: FAILED - {e}")
    raise SystemExit(1)

print("\n" + "=" * 50)
print("Testing video encode -> decode -> mock caption...")
try:
    import make_synthetic_data as msd
    import random

    from video_creative_ranking.captioner import MockCaptioner
    from video_creative_ranking.transcriber import MockTranscriber
    from video_creative_ranking.video_io import extract_audio, extract_frames

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.mp4"
        rng = random.Random(0)
        msd.encode_video(msd.render_frames(0.8, rng), msd.synth_audio(0.8, rng), path)
        frames = extract_frames(str(path), 4)
        audio = extract_audio(str(path))
        assert len(frames) == 4 and audio is not None and len(audio) > 1000
        cfg = make_config("mock")
        cap, _ = MockCaptioner(cfg).caption(str(path))
        txt = MockTranscriber(cfg).transcribe(str(path))
        assert "HOOK" in cap and txt
    print("  Video pipeline: OK")
except Exception as e:
    import traceback
    print(f"  Video pipeline: FAILED - {e}")
    traceback.print_exc()

parser = argparse.ArgumentParser()
parser.add_argument("--models", action="store_true", help="also load real models")
parser.add_argument("--preset", default="smoke")
args = parser.parse_args()

if args.models:
    print("\n" + "=" * 50)
    print(f"Testing real models (preset={args.preset})...")
    try:
        from video_creative_ranking.activations import ActivationExtractor
        cfg = make_config(args.preset)
        ext = ActivationExtractor(cfg)
        pooled, judge = ext.extract("Test context for a video ad. Answer (Yes or No):")
        print(f"  LLM activations: {tuple(pooled.shape)}, judge={judge:.3f}")
        print("  LLM: OK")
    except Exception as e:
        import traceback
        print(f"  LLM: FAILED - {e}")
        traceback.print_exc()

print("\n" + "=" * 50)
print("Setup test complete!")
