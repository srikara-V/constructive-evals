from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Sequence

from power_persona_sampling.config import PersonaSpec
from power_persona_sampling.extract import PersonaExample, PersonaVectorExtractor, iter_examples_jsonl
from power_persona_sampling.persona import PersonaVectorSet

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from persona_creator_lib import build_lm  # noqa: E402


@dataclass(frozen=True)
class PersonaVectorResult:
    vectors: PersonaVectorSet
    examples: list[PersonaExample]


def load_examples(path: str) -> list[PersonaExample]:
    return list(iter_examples_jsonl(path))


def extract_vectors(
    *,
    data_path: str,
    persona_specs: Sequence[PersonaSpec],
    model_name: str,
    batch_size: int = 8,
    normalize: bool = True,
    device: str | None = None,
    torch_dtype: str | None = None,
) -> PersonaVectorResult:
    examples = load_examples(data_path)
    lm = build_lm(model_name, device=device, torch_dtype=torch_dtype)
    extractor = PersonaVectorExtractor(lm, persona_specs, batch_size=batch_size)
    vectors = extractor.extract(examples, normalize=normalize)
    return PersonaVectorResult(vectors=vectors, examples=examples)


def write_vectors(path: str, vectors: PersonaVectorSet) -> None:
    vectors.save(path)
