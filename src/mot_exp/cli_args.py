"""Shared CLI helpers."""

from __future__ import annotations

import argparse
from pathlib import Path


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def add_trajectories_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trajectories",
        type=str2bool,
        default=False,
        metavar="BOOL",
        help="Draw tracking trajectories/trails (default: false). Pass true or false.",
    )


def with_hyperparam_tags(filename: str | Path, tags: list[str]) -> str:
    """
    Append enabled hyperparam tags to a filename stem.

    Example: ``ultralytics-tracking-seg.mp4`` + ``["-trajectories"]``
    → ``ultralytics-tracking-seg-trajectories.mp4``
    """
    path = Path(filename)
    suffix = path.suffix if path.suffix else ".mp4"
    stem = path.stem
    cleaned = []
    for tag in tags:
        t = tag if tag.startswith("-") else f"-{tag}"
        if t not in stem:
            cleaned.append(t)
    return f"{stem}{''.join(cleaned)}{suffix}"
