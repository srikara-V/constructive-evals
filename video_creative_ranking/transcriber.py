"""Audio transcription of the video's soundtrack (Whisper via transformers).

The transcript is the second block prepended to the LLM context. Voiceover
copy is often the actual sales pitch of an ad, so this can matter as much
as the visuals; `use_audio=False` ablates it.
"""

import json
import re
from pathlib import Path

import numpy as np

from .video_io import AUDIO_SR, extract_audio


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()


class WhisperTranscriber:
    def __init__(self, cfg):
        from transformers import pipeline

        self.cfg = cfg
        device = cfg.resolved_device()
        print(f"Loading Whisper {cfg.whisper_model} ({device})...")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=cfg.whisper_model,
            device=0 if device == "cuda" else -1,
            chunk_length_s=30,
        )

    def transcribe(self, video_path: str) -> str:
        audio = extract_audio(video_path)
        if audio is None or len(audio) < AUDIO_SR // 4:  # <0.25 s
            return ""
        out = self.pipe({"array": audio, "sampling_rate": AUDIO_SR},
                        generate_kwargs={"language": "english", "task": "transcribe"})
        return out["text"].strip()


class MockTranscriber:
    """Describes audio stats instead of real ASR. Plumbing tests only."""

    def __init__(self, cfg):
        self.cfg = cfg

    def transcribe(self, video_path: str) -> str:
        audio = extract_audio(video_path)
        if audio is None or len(audio) == 0:
            return ""
        energy = float(np.abs(audio).mean())
        spectrum = np.abs(np.fft.rfft(audio[: AUDIO_SR * 4]))
        freqs = np.fft.rfftfreq(min(len(audio), AUDIO_SR * 4), 1 / AUDIO_SR)
        pitch = float((freqs * spectrum).sum() / spectrum.sum()) if spectrum.sum() else 0.0
        tone = ("bright energetic high-pitched jingle" if pitch > 500
                else "calm low-pitched background hum")
        loud = "loud" if energy > 0.1 else "quiet"
        return f"[{loud} {tone}, no discernible speech]"


def transcribe_creatives(creatives, cfg) -> dict[str, str]:
    """Transcribe every creative with caching. Returns {creative_id: text}."""
    from .data import video_cache_key

    if not cfg.use_audio:
        return {c.id: "" for c in creatives}

    cache_dir = Path(cfg.cache_dir) / "transcripts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    backend = "mock" if cfg.mock_audio else _slug(cfg.whisper_model)

    transcriber = None
    results: dict[str, str] = {}
    from tqdm import tqdm
    for c in tqdm(creatives, desc=f"Transcribing ({backend})"):
        cache_file = cache_dir / f"{video_cache_key(c)}_{backend}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                results[c.id] = json.load(f)["text"]
            continue
        if transcriber is None:
            transcriber = MockTranscriber(cfg) if cfg.mock_audio else WhisperTranscriber(cfg)
        text = transcriber.transcribe(c.video_path)
        with open(cache_file, "w") as f:
            json.dump({"text": text}, f)
        results[c.id] = text
    return results
