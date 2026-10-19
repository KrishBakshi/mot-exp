#!/usr/bin/env python3
"""Ultralytics YOLO-seg tracking via solutions.InstanceSegmentation (+ optional trails)."""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from ultralytics import solutions
from ultralytics.engine.results import Results
from ultralytics.solutions.solutions import SolutionAnnotator
from ultralytics.utils.plotting import colors

from mot_exp.annotate import (
    class_counts_from_ids,
    draw_class_count_badge,
    format_track_label,
)
from mot_exp.cli_args import add_trajectories_arg, with_hyperparam_tags
from mot_exp.device import select_device
from mot_exp.progress import progress_bar
from mot_exp.video_paths import REPO_ROOT, resolve_input_video, resolve_output_video

# BoT-SORT: Kalman + CMC (sparseOptFlow) for camera motion / moving objects
# See ultralytics/cfg/trackers/botsort.yaml (gmc_method: sparseOptFlow)
BOTSORT = "botsort.yaml"


def draw_solution_trajectories(segmenter: solutions.InstanceSegmentation, frame) -> None:
    """Draw trails using Ultralytics Solutions ``store_tracking_history``."""
    for box, track_id in zip(segmenter.boxes, segmenter.track_ids):
        segmenter.store_tracking_history(track_id, box)
        history = segmenter.track_history[track_id]
        if len(history) < 2:
            continue
        points = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            frame,
            [points],
            isClosed=False,
            color=colors(int(track_id), True),
            thickness=max(4, segmenter.line_width),
        )


def annotate_frame(segmenter: solutions.InstanceSegmentation, frame):
    """Masks + boxes with ``#id class conf`` labels."""
    segmenter.extract_tracks(frame)
    masks = getattr(segmenter.tracks, "masks", None)

    if masks is None:
        annotated = frame.copy()
    else:
        results = Results(
            frame,
            path=None,
            names=segmenter.names,
            boxes=segmenter.track_data.data,
            masks=masks.data,
        )
        # Masks only here — boxes/labels drawn next with explicit #track ids
        annotated = results.plot(
            line_width=segmenter.line_width,
            boxes=False,
            labels=False,
            masks=True,
            conf=False,
            color_mode="instance",
        )

    annotator = SolutionAnnotator(annotated, line_width=segmenter.line_width)
    for box, track_id, cls, conf in zip(
        segmenter.boxes, segmenter.track_ids, segmenter.clss, segmenter.confs
    ):
        label = format_track_label(track_id, segmenter.names[int(cls)], conf)
        annotator.box_label(box, label=label, color=colors(int(track_id), True))
    return annotator.result()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ultralytics segmentation tracking (BoT-SORT) with masks, boxes, confidence.",
    )
    parser.add_argument(
        "video",
        help="Input video path or filename under data/ (e.g. zeebra.mp4)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output MP4 name or path (default: output/<video_stem>/ultralytics-tracking-seg.mp4)",
    )
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT / "yolo26m-seg.pt"),
        help="YOLO-seg weights path (default: ./yolo26m-seg.pt)",
    )
    add_trajectories_arg(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device()
    print(f"Inference device: {device}")
    print(f"Tracker: BoT-SORT ({BOTSORT}) with camera motion compensation")
    print(f"Trajectories: {args.trajectories}")

    tags = ["-trajectories"] if args.trajectories else []
    default_name = with_hyperparam_tags("ultralytics-tracking-seg.mp4", tags)
    output_arg = (
        with_hyperparam_tags(args.output, tags) if args.output is not None else None
    )

    input_path = resolve_input_video(args.video)
    output_path = resolve_output_video(
        output_arg,
        input_video=input_path,
        default_name=default_name,
    )
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    segmenter = solutions.InstanceSegmentation(
        model=args.model,
        device=device,
        tracker=BOTSORT,
        show=False,
        show_conf=True,
        show_labels=True,
        show_boxes=True,
        verbose=False,
    )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (frame_width, frame_height),
    )
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    try:
        with progress_bar(total, "Ultralytics seg-track") as pbar:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                annotated = annotate_frame(segmenter, frame)

                if args.trajectories:
                    draw_solution_trajectories(segmenter, annotated)

                counts = class_counts_from_ids(segmenter.clss, segmenter.names)
                draw_class_count_badge(annotated, counts, frame_width)
                out.write(annotated)
                pbar.update(1)
    finally:
        cap.release()
        out.release()

    print(f"Done. Wrote {output_path}")


if __name__ == "__main__":
    main()
