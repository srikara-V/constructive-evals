"""VLM captioning: turn a video into descriptive output tokens.

The frames-as-images pattern (N uniformly sampled frames sent as an ordered
image sequence in one chat turn) is used for every backend because it works
identically for Qwen2.5-VL, SmolVLM2, and any OpenAI-compatible endpoint.

Backends:
  hf            - local transformers model; can also capture the VLM's own
                  hidden states (used by creative_ranking_videos_4)
  openai_compat - vLLM / any /v1/chat/completions server hosting a VLM
  mock          - deterministic captions from pixel/audio statistics
                  (plumbing tests only; carries almost no semantic signal)
"""

import base64
import hashlib
import io
import json
import re
from pathlib import Path

import torch

from .video_io import extract_frames, video_stats

CAPTION_PROMPT = """You are analyzing a short-form video advertisement. \
The images are {n} frames sampled uniformly across the video, in temporal order.

Describe the ad precisely and concretely:
1. HOOK: what appears in the first moments, and what grabs attention.
2. VISUALS: setting, people (count, apparent age, emotion), product shots, \
dominant colors, lighting, camera style.
3. ON-SCREEN TEXT: quote any captions, headlines, prices, or overlays you can read.
4. PACING: static vs fast-cut, overall energy level.
5. CALL-TO-ACTION: any explicit CTA (button, spoken ask, end card).
6. BRAND: how and when the brand or product is revealed.

Be factual and specific. Do not speculate about how well the ad will perform."""


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()[:8]


class CaptionResult(dict):
    """{caption: str, vlm_hidden_path: str|None}"""


def _from_pretrained_kwargs(dtype, device):
    """transformers v5 renamed torch_dtype -> dtype; support both."""
    import transformers
    major = int(transformers.__version__.split(".")[0])
    key = "dtype" if major >= 5 else "torch_dtype"
    kw = {key: dtype}
    if device == "cuda":
        kw["device_map"] = "auto"
    return kw


class HFVLMCaptioner:
    def __init__(self, cfg):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.cfg = cfg
        self.device = cfg.resolved_device()
        dtype = cfg.torch_dtype()
        print(f"Loading VLM {cfg.vlm_model} ({self.device}, {dtype})...")
        self.processor = AutoProcessor.from_pretrained(cfg.vlm_model)
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.vlm_model, **_from_pretrained_kwargs(dtype, self.device)
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

    def caption(self, video_path: str, capture_hidden: bool = False):
        frames = extract_frames(video_path, self.cfg.n_frames)
        prompt = CAPTION_PROMPT.format(n=len(frames))
        content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=frames, return_tensors="pt")
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_caption_tokens,
                do_sample=False,
            )
        n_prompt = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0, n_prompt:]
        caption = self.processor.decode(gen_ids, skip_special_tokens=True).strip()

        hidden_pooled = None
        if capture_hidden:
            hidden_pooled = self._pooled_hidden(inputs, out_ids, n_prompt)
        return caption, hidden_pooled

    def _pooled_hidden(self, inputs, out_ids, n_prompt: int) -> torch.Tensor:
        """Re-run the full multimodal sequence; pool every layer.

        Returns [n_layers+1, 3, hidden]: pooling 0 = mean over all positions,
        1 = mean over generated caption tokens, 2 = last token.
        """
        fwd = dict(inputs)
        fwd["input_ids"] = out_ids
        fwd["attention_mask"] = torch.ones_like(out_ids)
        with torch.no_grad():
            outputs = self.model(**fwd, output_hidden_states=True, return_dict=True)
        pooled = []
        for h in outputs.hidden_states:
            h = h[0].float()  # [seq, hidden]
            gen = h[n_prompt:] if h.shape[0] > n_prompt else h
            pooled.append(torch.stack([h.mean(0), gen.mean(0), h[-1]]))
        return torch.stack(pooled).to(torch.float16).cpu()


class OpenAICompatCaptioner:
    """Caption through a vLLM/OpenAI-style endpoint (no hidden states)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def caption(self, video_path: str, capture_hidden: bool = False):
        import requests

        if capture_hidden:
            raise ValueError("openai_compat backend cannot expose hidden states; "
                             "use vlm_backend='hf' for creative_ranking_videos_4")
        frames = extract_frames(video_path, self.cfg.n_frames)
        content = []
        for f in frames:
            buf = io.BytesIO()
            f.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": CAPTION_PROMPT.format(n=len(frames))})

        resp = requests.post(
            f"{self.cfg.vlm_api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.cfg.vlm_api_key}"},
            json={
                "model": self.cfg.vlm_model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": self.cfg.max_caption_tokens,
                "temperature": 0.0,
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip(), None


_COLOR_NAMES = [
    ((220, 60, 60), "red"), ((60, 90, 220), "blue"), ((70, 190, 90), "green"),
    ((230, 200, 70), "yellow"), ((160, 90, 200), "purple"), ((240, 150, 60), "orange"),
    ((235, 235, 235), "white"), ((40, 40, 40), "dark gray"),
]


class MockCaptioner:
    """Deterministic caption from raw pixel/audio stats. Plumbing tests only."""

    def __init__(self, cfg):
        self.cfg = cfg

    def caption(self, video_path: str, capture_hidden: bool = False):
        s = video_stats(video_path, self.cfg.n_frames)
        rgb = s["mean_rgb"]
        color = min(_COLOR_NAMES,
                    key=lambda c: sum((a - b) ** 2 for a, b in zip(c[0], rgb)))[1]
        pace = ("fast-cut, high-energy" if s["motion"] > 8
                else "moderately paced" if s["motion"] > 2.5 else "static, slow")
        bright = ("bright" if s["brightness"] > 130
                  else "dim" if s["brightness"] < 80 else "neutrally lit")
        caption = (
            f"1. HOOK: a {color} scene opens the video. "
            f"2. VISUALS: predominantly {color}, {bright} footage. "
            f"3. ON-SCREEN TEXT: none legible. "
            f"4. PACING: {pace} (motion score {s['motion']:.1f}). "
            f"5. CALL-TO-ACTION: none detected. "
            f"6. BRAND: not identified."
        )
        hidden = None
        if capture_hidden:
            # tiny deterministic pseudo-features so v4 plumbing can run
            g = torch.Generator().manual_seed(
                int(hashlib.sha1(json.dumps(rgb).encode()).hexdigest()[:8], 16))
            base = torch.randn(4, 3, 64, generator=g)
            base[:, :, 0] = s["brightness"] / 255.0
            base[:, :, 1] = s["motion"] / 50.0
            hidden = base.to(torch.float16)
        return caption, hidden


def make_captioner(cfg):
    return {"hf": HFVLMCaptioner,
            "openai_compat": OpenAICompatCaptioner,
            "mock": MockCaptioner}[cfg.vlm_backend](cfg)


def caption_creatives(creatives, cfg, capture_hidden: bool | None = None):
    """Caption every creative with caching. Returns {creative_id: CaptionResult}."""
    from .data import video_cache_key

    if capture_hidden is None:
        capture_hidden = cfg.capture_vlm_hidden
    if cfg.vlm_backend == "openai_compat":
        capture_hidden = False  # API backends cannot expose hidden states
    cache_dir = Path(cfg.cache_dir) / "captions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{_slug(cfg.vlm_model)}_{cfg.vlm_backend}_f{cfg.n_frames}_{_prompt_hash(CAPTION_PROMPT)}"

    captioner = None
    results: dict[str, CaptionResult] = {}
    from tqdm import tqdm
    for c in tqdm(creatives, desc=f"Captioning ({cfg.vlm_backend})"):
        key = video_cache_key(c)
        cache_file = cache_dir / f"{key}_{tag}.json"
        hidden_file = cache_dir / f"{key}_{tag}_hidden.pt"
        if cache_file.exists() and (not capture_hidden or hidden_file.exists()):
            with open(cache_file) as f:
                results[c.id] = CaptionResult(json.load(f))
            continue
        if captioner is None:  # lazy: skip model load on full cache hit
            captioner = make_captioner(cfg)
        caption, hidden = captioner.caption(c.video_path, capture_hidden)
        res = CaptionResult(caption=caption,
                            vlm_hidden_path=str(hidden_file) if hidden is not None else None)
        if hidden is not None:
            torch.save(hidden, hidden_file)
        with open(cache_file, "w") as f:
            json.dump(dict(res), f, indent=1)
        results[c.id] = res
    return results
