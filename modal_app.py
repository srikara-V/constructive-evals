from __future__ import annotations

import os
import shlex
import subprocess

import modal


APP_NAME = "constructive-evals"
REPO_PATH = "/root"
DEFAULT_COMMAND = (
    "eval-gsm8k "
    "--model Qwen/Qwen2.5-0.5B-Instruct "
    "--split test "
    "--limit 25 "
    "--out /tmp/gsm8k_mh.jsonl "
    "--mh-steps 10 "
    "--alpha 3.0 "
    "--prompt-style short"
)

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
    command = DEFAULT_COMMAND
    use_power_persona = True
    if "MODAL_COMMAND" in os.environ:
        command = os.environ["MODAL_COMMAND"]
    if os.environ.get("MODAL_RAW") == "1":
        use_power_persona = False

    rc = run_cli.remote(command, use_power_persona=use_power_persona)
    if rc != 0:
        raise SystemExit(rc)
