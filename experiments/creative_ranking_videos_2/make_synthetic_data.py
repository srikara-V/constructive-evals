#!/usr/bin/env python3
"""
Generate a tiny synthetic video-ad dataset to smoke-test the
creative_ranking_videos pipelines end to end (no real ad data needed).

Each "ad" is a short mp4 with audio. A latent quality q in [0,1] drives
attributes that a VLM can genuinely see and describe:
  high q: bright scene, fast-moving red ball, "SALE 50% OFF" + "SHOP NOW"
          overlays, energetic high-pitched jingle
  low q:  dim scene, slow blue square, no text, low hum
CTR is a noisy function of q, so a working pipeline should beat 50%
pairwise accuracy while noise keeps 100% out of reach.

Output: data/synthetic_videos/*.mp4 + data/synthetic_videos/manifest.json
(flat manifest format — pairs are built by video_creative_ranking.data).
"""

import argparse
import json
import math
import random
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

W, H, FPS, SECONDS = 320, 240, 8, 2.5
SR = 16000


def render_frames(q: float, rng: random.Random) -> list[np.ndarray]:
    from PIL import Image, ImageDraw, ImageFont

    n = int(FPS * SECONDS)
    bright = 40 + int(190 * q) + rng.randint(-15, 15)
    bg = (bright, bright, max(0, bright - 20))
    ball_color = (230, 40, 40) if q > 0.5 else (40, 60, 220)
    speed = 2 + 10 * q
    radius = 24
    has_text = q > 0.45 and rng.random() < 0.9

    try:
        font_big = ImageFont.load_default(size=34)
        font_small = ImageFont.load_default(size=22)
    except TypeError:  # Pillow < 10.1
        font_big = font_small = ImageFont.load_default()

    frames = []
    x, y = W // 4, H // 3
    dx, dy = speed, speed * 0.7
    for i in range(n):
        img = Image.new("RGB", (W, H), bg)
        d = ImageDraw.Draw(img)
        x, y = x + dx, y + dy
        if not radius < x < W - radius:
            dx = -dx
            x = max(radius, min(W - radius, x))
        if not radius < y < H - radius:
            dy = -dy
            y = max(radius, min(H - radius, y))
        d.ellipse([x - radius, y - radius, x + radius, y + radius], fill=ball_color)
        if has_text:
            d.text((W // 2 - 90, 18), "SALE 50% OFF", fill=(255, 255, 255),
                   font=font_big, stroke_width=2, stroke_fill=(0, 0, 0))
            if i > n // 2:  # end-card CTA
                d.rectangle([W // 2 - 70, H - 52, W // 2 + 70, H - 16],
                            fill=(20, 120, 30))
                d.text((W // 2 - 52, H - 46), "SHOP NOW", fill=(255, 255, 255),
                       font=font_small)
        frames.append(np.asarray(img, dtype=np.uint8))
    return frames


def synth_audio(q: float, rng: random.Random) -> np.ndarray:
    t = np.arange(int(SR * SECONDS)) / SR
    if q > 0.5:  # energetic arpeggio
        notes = [660, 880, 1100, 1320]
        seg = len(t) // 8
        audio = np.concatenate([
            0.5 * np.sin(2 * math.pi * notes[i % 4] * t[:seg]) for i in range(8)
        ])
        audio = np.resize(audio, len(t))
    else:  # low hum
        audio = 0.25 * np.sin(2 * math.pi * 110 * t)
    audio += 0.02 * rng.random() * np.random.default_rng(rng.randint(0, 9999)).standard_normal(len(t))
    return np.clip(audio, -1, 1)


def encode_video(frames: list[np.ndarray], audio: np.ndarray, out_path: Path):
    """h264 mp4 via PyAV, then mux the wav in with the bundled ffmpeg."""
    import av
    import imageio_ffmpeg

    with tempfile.TemporaryDirectory() as td:
        silent = Path(td) / "silent.mp4"
        wav = Path(td) / "audio.wav"

        with av.open(str(silent), "w") as container:
            stream = container.add_stream("libx264", rate=FPS)
            stream.width, stream.height = W, H
            stream.pix_fmt = "yuv420p"
            for arr in frames:
                frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                container.mux(stream.encode(frame))
            container.mux(stream.encode(None))

        pcm = (audio * 32767).astype("<i2")
        with wave.open(str(wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            wf.writeframes(pcm.tobytes())

        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
             "-i", str(silent), "-i", str(wav),
             "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)],
            check=True,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-videos", type=int, default=20)
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).parent.parent.parent / "data" / "synthetic_videos")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    creatives = []
    for i in range(args.n_videos):
        q = rng.random()
        name = f"synth_{i:03d}.mp4"
        path = args.out_dir / name
        print(f"[{i+1}/{args.n_videos}] {name}  q={q:.2f}")
        frames = render_frames(q, rng)
        audio = synth_audio(q, rng)
        encode_video(frames, audio, path)

        impressions = rng.randint(5000, 50000)
        ctr = (0.004 + 0.030 * q) * math.exp(rng.gauss(0, 0.25))
        creatives.append({
            "id": f"synth_{i:03d}",
            "video_path": name,  # relative to the manifest
            "impressions": impressions,
            "clicks": max(1, int(impressions * ctr)),
            "campaign": "synthetic_smoke",
            "brand": "SynthCo",
            "_latent_quality": round(q, 3),
        })

    manifest = args.out_dir / "manifest.json"
    with open(manifest, "w") as f:
        json.dump({"creatives": creatives}, f, indent=1)
    print(f"\nWrote {args.n_videos} videos + manifest to {manifest}")


if __name__ == "__main__":
    main()
