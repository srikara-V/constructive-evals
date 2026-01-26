from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import modal


APP_NAME = "constructive-evals"
REPO_ROOT = Path(__file__).resolve().parent
REPO_PATH = "/repo"

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

app = modal.App(APP_NAME)


def _mount_from_local_dir(path: Path, *, remote_path: str):
    if hasattr(modal, "Mount"):
        return modal.Mount.from_local_dir(path, remote_path=remote_path)
    mount_module = getattr(modal, "mount", None)
    if mount_module is not None and hasattr(mount_module, "Mount"):
        return mount_module.Mount.from_local_dir(path, remote_path=remote_path)
    raise RuntimeError(
        "Modal Mount API not found. Update the modal package or adjust modal_app.py "
        "to the installed SDK version."
    )


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
    mounts=[_mount_from_local_dir(REPO_ROOT, remote_path=REPO_PATH)],
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
        required=True,
        help=(
            "Command to run. By default this is appended to "
            "'python -m power_persona_sampling ...'."
        ),
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Run the command as-is instead of routing through power_persona_sampling.",
    )
    args = parser.parse_args()

    rc = run_cli.remote(args.command, use_power_persona=not args.raw)
    if rc != 0:
        raise SystemExit(rc)
