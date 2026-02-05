from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from power_persona_sampling.io_utils import write_jsonl
from power_persona_sampling.lm.base import SamplingParams
from power_persona_sampling.lm.hf import HFModelConfig, HuggingFaceLM


@dataclass(frozen=True)
class PersonaSample:
    persona: str
    label: int
    prompt: str
    response: str


def build_lm(model_name: str, device: str | None = None, torch_dtype: str | None = None) -> HuggingFaceLM:
    cfg = HFModelConfig(
        model_name_or_path=model_name,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        torch_dtype=torch_dtype,
    )
    return HuggingFaceLM(cfg)


def generate_samples(
    lm: HuggingFaceLM,
    *,
    positive_prompt: str,
    negative_prompt: str,
    trait: str,
    count_per_class: int,
    params: SamplingParams,
    seed: int | None = None,
) -> list[PersonaSample]:
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    prompts: list[tuple[str, int]] = [
        (positive_prompt, 1),
        (negative_prompt, 0),
    ]

    samples: list[PersonaSample] = []
    for prompt, label in prompts:
        prefix_ids = lm.encode(prompt)
        for _ in range(int(count_per_class)):
            suffix = lm.sample_suffix(prefix_ids, params)
            response = lm.decode(suffix).strip()
            samples.append(
                PersonaSample(
                    persona=trait,
                    label=label,
                    prompt=prompt,
                    response=response,
                )
            )

    return samples


def samples_to_rows(samples: Sequence[PersonaSample]) -> Iterable[dict[str, object]]:
    for sample in samples:
        yield {
            "persona": sample.persona,
            "label": sample.label,
            "prompt": sample.prompt,
            "response": sample.response,
        }


def write_samples(path: str, samples: Sequence[PersonaSample]) -> None:
    write_jsonl(path, samples_to_rows(samples))
