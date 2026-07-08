"""Configuration for the video creative ranking pipeline."""

import torch
from dataclasses import dataclass, field, fields
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CACHE_DIR = REPO_ROOT / "cache" / "video_creative_ranking"


@dataclass
class PipelineConfig:
    # --- data ---
    manifest: Path = REPO_ROOT / "data" / "padsplit_videos.json"
    n_test: int = 500          # test pairs to evaluate on
    n_train: int | None = None  # None = use all train pairs
    val_frac: float = 0.15     # slice of train used for layer/pooling selection
    metric: str = "ctr"        # ctr | conversion_rate | <raw field name>
    min_impressions: int = 500  # ignore creatives with fewer impressions
    min_ratio: float = 1.15    # skip near-tied pairs (winner/loser metric ratio)
    seed: int = 42

    # --- VLM captioning (the "output tokens") ---
    vlm_backend: str = "hf"    # hf | openai_compat | mock
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_api_base: str = "http://localhost:8000/v1"  # for openai_compat (vLLM etc.)
    vlm_api_key: str = "EMPTY"
    n_frames: int = 16         # uniformly sampled frames per video
    max_caption_tokens: int = 350
    # Store the VLM's own pooled hidden states alongside each caption (one
    # extra forward + ~0.5 MB per creative). On by default so a single
    # captioning pass serves v2/v3 (tokens) AND v4 (VLM-native activations).
    # Forced off for openai_compat, which cannot expose hidden states.
    capture_vlm_hidden: bool = True

    # --- audio ---
    use_audio: bool = True
    whisper_model: str = "openai/whisper-small"  # any HF whisper checkpoint
    mock_audio: bool = False   # use mock transcriber (plumbing tests)

    # --- LLM whose context receives the prepended tokens ---
    llm_model: str = "Qwen/Qwen2.5-7B"  # base model by default
    use_chat_template: bool = False     # True for instruct models
    max_context_tokens: int = 1536
    llm_dtype: str = "auto"    # auto | float16 | bfloat16 | float32

    # --- misc ---
    cache_dir: Path = DEFAULT_CACHE_DIR
    device: str = "auto"       # auto | cuda | cpu

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def torch_dtype(self) -> torch.dtype:
        if self.llm_dtype == "float16":
            return torch.float16
        if self.llm_dtype == "bfloat16":
            return torch.bfloat16
        if self.llm_dtype == "float32":
            return torch.float32
        # auto: half precision on GPU, float32 on CPU
        return torch.float16 if self.resolved_device() == "cuda" else torch.float32

    def apply_overrides(self, **kwargs) -> "PipelineConfig":
        valid = {f.name for f in fields(self)}
        for k, v in kwargs.items():
            if v is None:
                continue
            if k not in valid:
                raise ValueError(f"Unknown config field: {k}")
            setattr(self, k, v)
        return self


# Presets keep experiment scripts short. "padsplit_7b" is the intended real
# run; "smoke" runs on CPU with tiny models to verify the pipeline end to end.
PRESETS: dict[str, dict] = {
    "padsplit_7b": dict(
        vlm_model="Qwen/Qwen2.5-VL-7B-Instruct",
        llm_model="Qwen/Qwen2.5-7B",
        whisper_model="openai/whisper-small",
        n_frames=16,
    ),
    "padsplit_3b": dict(
        vlm_model="Qwen/Qwen2.5-VL-3B-Instruct",
        llm_model="Qwen/Qwen2.5-3B",
        whisper_model="openai/whisper-small",
        n_frames=16,
    ),
    # Instruct LLM variant (chat template wraps the prepended context)
    "padsplit_7b_instruct": dict(
        vlm_model="Qwen/Qwen2.5-VL-7B-Instruct",
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        use_chat_template=True,
        whisper_model="openai/whisper-small",
        n_frames=16,
    ),
    # CPU-safe end-to-end check with real (tiny) models
    "smoke": dict(
        vlm_model="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        llm_model="Qwen/Qwen2.5-0.5B",
        whisper_model="openai/whisper-tiny",
        n_frames=6,
        max_caption_tokens=128,
        min_impressions=0,
        min_ratio=1.0,
    ),
    # Pure plumbing check: no model downloads at all
    "mock": dict(
        vlm_backend="mock",
        mock_audio=True,
        llm_model="Qwen/Qwen2.5-0.5B",
        n_frames=8,
        min_impressions=0,
        min_ratio=1.0,
    ),
}


def make_config(preset: str | None = None, **overrides) -> PipelineConfig:
    cfg = PipelineConfig()
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Options: {list(PRESETS)}")
        cfg.apply_overrides(**PRESETS[preset])
    cfg.apply_overrides(**overrides)
    cfg.cache_dir = Path(cfg.cache_dir)
    cfg.manifest = Path(cfg.manifest)
    return cfg
