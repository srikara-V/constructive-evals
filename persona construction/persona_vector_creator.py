from __future__ import annotations

import argparse
import sys
from pathlib import Path

from power_persona_sampling.config import PersonaSpec

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from persona_vector_lib import extract_vectors, write_vectors  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract persona vectors from JSONL data created by persona_creator. "
            "Writes a PersonaVectorSet .pt file compatible with power_persona_sampling."
        )
    )
    parser.add_argument("--data", required=True, help="Path to JSONL persona examples.")
    parser.add_argument("--persona", required=True, help="Persona name present in the JSONL.")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to use for the persona vector.")
    parser.add_argument(
        "--out",
        default="persona_vectors.pt",
        help="Output path for PersonaVectorSet (default: persona_vectors.pt).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model name or path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for scoring examples.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization of the persona vector.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string for torch (e.g., cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        help="Torch dtype string for loading the model (e.g., float16, bfloat16).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    specs = [PersonaSpec(name=args.persona, layer=args.layer, beta=1.0, lam=1.0)]
    result = extract_vectors(
        data_path=args.data,
        persona_specs=specs,
        model_name=args.model,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    write_vectors(args.out, result.vectors)


if __name__ == "__main__":
    main()
