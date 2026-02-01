from __future__ import annotations

import argparse

from power_persona_sampling.lm.base import SamplingParams

from persona_creator_lib import build_lm, generate_samples, write_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic persona examples for power_persona_sampling. "
            "Produces JSONL with prompt/response/persona/label fields."
        )
    )
    parser.add_argument("--positive-prompt", required=True, help="Prompt that elicits the trait (label=1).")
    parser.add_argument("--negative-prompt", required=True, help="Prompt that avoids the trait (label=0).")
    parser.add_argument("--trait", required=True, help="Persona name to assign to all samples.")
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of samples to generate per class (positive and negative).",
    )
    parser.add_argument(
        "--out",
        default="persona_examples.jsonl",
        help="Output JSONL path (default: persona_examples.jsonl).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model name or path.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max tokens to generate per response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
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

    params = SamplingParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    lm = build_lm(args.model, device=args.device, torch_dtype=args.torch_dtype)
    samples = generate_samples(
        lm,
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt,
        trait=args.trait,
        count_per_class=args.count,
        params=params,
        seed=args.seed,
    )
    write_samples(args.out, samples)


if __name__ == "__main__":
    main()
