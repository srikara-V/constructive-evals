"""Generate narration audio with ElevenLabs, one file per scene.

Uses the /with-timestamps endpoint so every character of the narration gets a
start/end time; scenes.py anchors drawing events to those times.

Voice: George (JBFqnCBsd6RMkjVDRZzb) - warm male storyteller.
Requires env var ELEVEN_LABS_API.
"""

import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from narration import SCENES

VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George - Warm, Captivating Storyteller (male)
MODEL_ID = "eleven_multilingual_v2"
AUDIO_DIR = Path(__file__).parent / "audio"


def tts_with_timestamps(text: str, api_key: str) -> dict:
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/with-timestamps"
        "?output_format=mp3_44100_128"
    )
    body = {
        "text": text,
        "model_id": MODEL_ID,
        "seed": 42,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.1,
            "use_speaker_boost": True,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())
        except Exception as e:  # noqa: BLE001 - retry then surface
            if attempt == 3:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  retry in {wait}s ({e})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def audio_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def main() -> None:
    api_key = os.environ.get("ELEVEN_LABS_API")
    if not api_key:
        sys.exit("ELEVEN_LABS_API env var not set")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    only = set(sys.argv[1:])  # optionally pass scene ids to regenerate
    for scene in SCENES:
        sid = scene["id"]
        if only and sid not in only:
            continue
        mp3_path = AUDIO_DIR / f"{sid}.mp3"
        align_path = AUDIO_DIR / f"{sid}.align.json"
        if mp3_path.exists() and align_path.exists() and not only:
            print(f"[skip] {sid} (exists)")
            continue
        print(f"[tts ] {sid} ({len(scene['text'])} chars)...")
        res = tts_with_timestamps(scene["text"], api_key)
        mp3_path.write_bytes(base64.b64decode(res["audio_base64"]))
        align = res.get("alignment") or res.get("normalized_alignment")
        align_path.write_text(json.dumps(align))
        dur = audio_duration(mp3_path)
        n_chars = len(align["characters"]) if align else 0
        print(f"       {dur:.1f}s audio, {n_chars} aligned chars")

    total = 0.0
    for scene in SCENES:
        p = AUDIO_DIR / f"{scene['id']}.mp3"
        if p.exists():
            d = audio_duration(p)
            total += d
            print(f"{scene['id']}: {d:.1f}s")
    print(f"TOTAL: {total/60:.1f} min")


if __name__ == "__main__":
    main()
