"""Frame and audio extraction from video files (PyAV, no system ffmpeg)."""

import numpy as np

AUDIO_SR = 16000  # what Whisper expects


def extract_frames(video_path: str, n_frames: int = 16) -> list:
    """Uniformly sample n_frames RGB PIL images across the video."""
    import av
    from PIL import Image

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]

    if not frames:
        raise ValueError(f"No decodable video frames in {video_path}")

    idx = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).astype(int)
    return [Image.fromarray(frames[i]) for i in idx]


def extract_audio(video_path: str, sr: int = AUDIO_SR) -> np.ndarray | None:
    """Decode the audio track to mono float32 at `sr`. None if no audio."""
    import av

    with av.open(video_path) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        # flush resampler
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))

    if not chunks:
        return None
    audio = np.concatenate(chunks).astype(np.float32) / 32768.0
    return audio


def video_stats(video_path: str, n_frames: int = 8) -> dict:
    """Cheap pixel/audio statistics — powers the mock captioner."""
    frames = extract_frames(video_path, n_frames)
    arrs = [np.asarray(f, dtype=np.float32) for f in frames]
    mean_rgb = np.stack([a.reshape(-1, 3).mean(axis=0) for a in arrs]).mean(axis=0)
    brightness = float(mean_rgb.mean())
    motion = 0.0
    if len(arrs) > 1:
        diffs = [np.abs(arrs[i + 1] - arrs[i]).mean() for i in range(len(arrs) - 1)]
        motion = float(np.mean(diffs))

    audio = extract_audio(video_path)
    audio_energy, audio_pitch = 0.0, 0.0
    if audio is not None and len(audio) > 0:
        audio_energy = float(np.abs(audio).mean())
        spectrum = np.abs(np.fft.rfft(audio[: AUDIO_SR * 4]))
        freqs = np.fft.rfftfreq(min(len(audio), AUDIO_SR * 4), 1 / AUDIO_SR)
        if spectrum.sum() > 0:
            audio_pitch = float((freqs * spectrum).sum() / spectrum.sum())

    return {
        "mean_rgb": [float(x) for x in mean_rgb],
        "brightness": brightness,
        "motion": motion,
        "audio_energy": audio_energy,
        "audio_pitch_hz": audio_pitch,
        "n_frames_decoded": len(frames),
    }
