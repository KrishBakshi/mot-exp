"""Shared frame annotation helpers."""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np


def format_track_label(
    track_id: int | None,
    class_name: str,
    confidence: float | None = None,
) -> str:
    """Label like ``#3 person 0.91`` for verifying track persistence."""
    tid = f"#{int(track_id)} " if track_id is not None else ""
    if confidence is not None:
        return f"{tid}{class_name} {float(confidence):.2f}".strip()
    return f"{tid}{class_name}".strip()


def class_counts_from_ids(
    class_ids: list[int] | np.ndarray | None,
    names: dict[int, str] | list[str],
) -> dict[str, int]:
    """Build ``{class_name: count}`` for the current frame."""
    if class_ids is None:
        return {}
    counts: Counter[str] = Counter()
    for cid in class_ids:
        if cid is None:
            continue
        cid_int = int(cid)
        if isinstance(names, dict):
            name = names.get(cid_int, str(cid_int))
        else:
            name = names[cid_int] if 0 <= cid_int < len(names) else str(cid_int)
        counts[name] += 1
    return dict(counts)


def draw_class_count_badge(
    frame: np.ndarray,
    class_counts: dict[str, int],
    frame_width: int,
) -> None:
    """
    Draw a top-right white board listing ``Class: N`` lines.

    The board grows vertically (and horizontally) as more classes appear.
    """
    if class_counts:
        lines = [
            f"{name}: {count}"
            for name, count in sorted(
                class_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]
    else:
        lines = ["None: 0"]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 1.2, 2
    pad_x, pad_y = 16, 12
    line_gap = 10

    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    text_w = max(w for w, _ in sizes)
    text_h = sizes[0][1]
    box_w = text_w + 2 * pad_x
    box_h = pad_y * 2 + len(lines) * text_h + (len(lines) - 1) * line_gap

    x1 = frame_width - box_w - 20
    y1 = 20
    x2, y2 = x1 + box_w, y1 + box_h

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

    y = y1 + pad_y + text_h
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x1 + pad_x, y),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        y += text_h + line_gap
