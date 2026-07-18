"""Build the video.

  python3 build.py check                 build scenes, validate anchors, dump preview stills
  python3 build.py scene <id> [...]      render specific scene(s) to mp4
  python3 build.py all                   render everything + concat final mp4
"""

from __future__ import annotations

import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

from narration import SCENES
from renderer import FPS, Scene, SceneRenderer, W, H  # noqa: F401
from scenes import BUILDERS

ROOT = Path(__file__).parent
AUDIO = ROOT / "audio"
OUT = ROOT / "output"
FINAL = OUT / "ranking_methodology_khan_academy.mp4"


def make_scene(sid: str) -> Scene:
    sc = Scene(sid, AUDIO)
    BUILDERS[sid](sc)
    return sc


def check() -> None:
    prev = OUT / "previews"
    prev.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for meta in SCENES:
        sid = meta["id"]
        sc = make_scene(sid)
        total += sc.duration
        print(f"{sid}: {len(sc.events)} events, {sc.duration:.1f}s")
        r = SceneRenderer(sc)
        for frac in (0.3, 0.6, 0.99):
            t = sc.duration * frac
            img = r.frame(t).resize((W, H))
            img.save(prev / f"{sid}_{int(frac*100):02d}.png")
    print(f"total video: {total/60:.1f} min")


def render_one(sid: str) -> str:
    t0 = time.time()
    sc = make_scene(sid)
    r = SceneRenderer(sc)
    out = OUT / f"scene_{sid}.mp4"
    r.render_to_mp4(out, AUDIO / f"{sid}.mp4".replace(".mp4", ".mp3"))
    dt = time.time() - t0
    n = int(sc.duration * FPS)
    return f"{sid}: {n} frames in {dt:.0f}s ({n/dt:.1f} fps)"


def concat() -> None:
    lst = OUT / "concat.txt"
    lst.write_text("".join(f"file 'scene_{m['id']}.mp4'\n" for m in SCENES))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(FINAL)],
        check=True,
    )
    print(f"final: {FINAL}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        check()
    elif mode == "scene":
        for sid in sys.argv[2:]:
            print(render_one(sid))
    elif mode == "all":
        ids = [m["id"] for m in SCENES]
        with mp.Pool(min(4, mp.cpu_count())) as pool:
            for msg in pool.imap_unordered(render_one, ids):
                print(msg)
        concat()
    else:
        sys.exit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
