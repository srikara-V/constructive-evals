from __future__ import annotations

import argparse
import os
import shlex
import subprocess

import modal


APP_NAME = "constructive-evals"
REPO_PATH = "/root"

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "numpy",
        "scipy",
    )
)

app = modal.App(APP_NAME, include_source=True)


def _split_command(cmd: str) -> list[str]:
    parts = shlex.split(cmd)
    if not parts:
        raise ValueError("command must not be empty")
    return parts


def _power_persona_command(args: str) -> list[str]:
    return ["python", "-m", "power_persona_sampling", *_split_command(args)]


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_cli(command: str, *, use_power_persona: bool = True) -> int:
    """
    Run a CLI command inside Modal.
    """
    cmd = _power_persona_command(command) if use_power_persona else _split_command(command)
    result = subprocess.run(cmd, cwd=REPO_PATH, check=False)
    return int(result.returncode)


@app.local_entrypoint()
def main() -> None:
    parser = argparse.ArgumentParser(description="Run constructive-evals CLI on Modal.")
    parser.add_argument(
        "--command",
        default=None,
        help=(
            "Command to run. By default this is appended to "
            "'python -m power_persona_sampling ...'. "
            "If omitted, uses the MODAL_COMMAND environment variable."
        ),
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Run the command as-is instead of routing through power_persona_sampling.",
    )
    args = parser.parse_args()

    command = args.command or os.environ.get("MODAL_COMMAND")
    if not command:
        raise SystemExit("Missing command. Pass --command or set MODAL_COMMAND.")

    rc = run_cli.remote(command, use_power_persona=not args.raw)
    if rc != 0:
        raise SystemExit(rc)
