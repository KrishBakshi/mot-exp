"""Resolve input/output video paths relative to the repo root."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"


def resolve_input_video(video: str | Path) -> Path:
    """Resolve a video name or path; bare names are looked up under ``data/``."""
    path = Path(video).expanduser()
    if path.is_file():
        return path.resolve()

    candidate = DATA_DIR / path.name
    if candidate.is_file():
        return candidate.resolve()

    if not path.is_absolute():
        from_cwd = (Path.cwd() / path).resolve()
        if from_cwd.is_file():
            return from_cwd

    raise FileNotFoundError(
        f"Video not found: {video!r}. Tried {path} and {candidate}."
    )


def resolve_output_video(
    output: str | Path | None,
    *,
    input_video: Path,
    default_name: str,
) -> Path:
    """
    Resolve where to write the output video.

    - If ``output`` is omitted: ``output/<input_stem>/<default_name>``
    - If ``output`` is a bare filename: ``output/<input_stem>/<filename>``
    - If ``output`` includes a directory: used as-is (created if needed)
    """
    stem = input_video.stem
    if output is None:
        path = OUTPUT_DIR / stem / default_name
    else:
        path = Path(output).expanduser()
        if path.suffix.lower() != ".mp4":
            path = path.with_suffix(".mp4")
        if path.parent == Path("."):
            path = OUTPUT_DIR / stem / path.name

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
