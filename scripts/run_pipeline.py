"""Run encryption/decryption from config.json with no IDE arguments."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the ROI-video pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "config.json",
        help="Local JSON configuration; defaults to repository/config.json.",
    )
    return parser.parse_args()


def common_key(config: dict) -> list[str]:
    key = str(config.get("key_hex", ""))
    if len(key) != 64 or key == "REPLACE_WITH_64_HEX_CHARACTERS":
        raise ValueError(
            "config.json key_hex must contain 64 hexadecimal characters."
        )
    return ["--key-hex", key]


def encrypt_command(repo_root: Path, config: dict) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "scripts" / "video_encrypt.py"),
        "--input",
        str(config["input"]),
        "--model",
        str(config["model"]),
        "--output-dir",
        str(config["encrypted_output"]),
        "--pool-size",
        str(config.get("pool_size", 1_000_000)),
        "--rounds",
        str(config.get("rounds", 3)),
        "--confidence",
        str(config.get("confidence", 0.25)),
        "--imgsz",
        str(config.get("imgsz", 960)),
        "--classes",
        *[str(value) for value in config.get("classes", [0, 1, 2])],
        *common_key(config),
    ]
    if config.get("max_frames") is not None:
        command.extend(["--max-frames", str(config["max_frames"])])
    if config.get("preview", False):
        command.append("--preview")
    return command


def decrypt_command(repo_root: Path, config: dict) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "scripts" / "video_decrypt.py"),
        "--input-dir",
        str(config["encrypted_output"]),
        "--output-dir",
        str(config["recovered_output"]),
        *common_key(config),
    ]
    if config.get("enable_recovery", False):
        command.append("--enable-recovery")
    if config.get("preview", False):
        command.append("--preview")
    return command


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    if not config_path.exists():
        template = repo_root / "config.example.json"
        if config_path.parent == repo_root:
            shutil.copyfile(template, config_path)
            print(f"Created local configuration: {config_path}")
            print("Edit its paths and key_hex, then run this script again.")
            return 2
        raise ValueError(f"Configuration file does not exist: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    mode = str(config.get("mode", "encrypt")).lower()
    if mode not in {"encrypt", "decrypt", "roundtrip"}:
        raise ValueError("config.json mode must be encrypt, decrypt, or roundtrip.")

    commands = []
    if mode in {"encrypt", "roundtrip"}:
        commands.append(encrypt_command(repo_root, config))
    if mode in {"decrypt", "roundtrip"}:
        commands.append(decrypt_command(repo_root, config))

    for command in commands:
        print("Running:", " ".join(f'"{part}"' if " " in part else part for part in command))
        completed = subprocess.run(command, cwd=repo_root, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
