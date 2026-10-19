"""Single-line progress bar that refreshes in place (even when stdout isn't a TTY)."""

from __future__ import annotations

import sys
from typing import Any

from tqdm import tqdm


class _ForceTTY:
    """Wrap a stream so tqdm always uses carriage-return updates instead of newlines."""

    def __init__(self, stream) -> None:
        self._stream = stream

    def write(self, s: str) -> int:
        return self._stream.write(s)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return self._stream.fileno()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def progress_bar(total: int | None, desc: str) -> tqdm:
    return tqdm(
        total=total,
        unit="frame",
        desc=desc,
        file=_ForceTTY(sys.stderr),
        dynamic_ncols=True,
        mininterval=0.1,
        leave=True,
        ascii=True,
    )
