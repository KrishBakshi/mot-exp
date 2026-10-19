#!/usr/bin/env python3
"""RF-DETR (Roboflow) segmentation tracking → annotated MP4 (ByteTrack + optional trails)."""

from __future__ import annotations

import argparse

import cv2
import supervision as sv
import torch
from rfdetr.assets.coco_classes import COCO_CLASSES
from trackers import ByteTrackTracker

from mot_exp.annotate import (
    class_counts_from_ids,
    draw_class_count_badge,
    format_track_label,
)
from mot_exp.cli_args import add_trajectories_arg, with_hyperparam_tags
from mot_exp.device import select_device
from mot_exp.progress import progress_bar
from mot_exp.video_paths import resolve_input_video, resolve_output_video

TRAIL_LENGTH = 30


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RF-DETR segmentation tracking (ByteTrack) with masks, boxes, confidence.",
    )
    parser.add_argument(
        "video",
        help="Input video path or filename under data/ (e.g. zeebra.mp4)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output MP4 name or path (default: output/<video_stem>/roboflow-tracking-seg.mp4)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--model-size",
        choices=("nano", "small", "medium", "large", "xlarge", "2xlarge"),
        default="medium",
        help="RF-DETR-Seg size (default: medium)",
    )
    add_trajectories_arg(parser)
    return parser.parse_args(argv)


def load_rfdetr_seg(size: str, device: str):
    from rfdetr import (
        RFDETRSeg2XLarge,
        RFDETRSegLarge,
        RFDETRSegMedium,
        RFDETRSegNano,
        RFDETRSegSmall,
        RFDETRSegXLarge,
    )

    cls = {
        "nano": RFDETRSegNano,
        "small": RFDETRSegSmall,
        "medium": RFDETRSegMedium,
        "large": RFDETRSegLarge,
        "xlarge": RFDETRSegXLarge,
        "2xlarge": RFDETRSeg2XLarge,
    }[size]
    model = cls()
    if getattr(model, "model", None) is not None:
        model.model.device = torch.device(device)
    return model


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device()
    print(f"Inference device: {device}")
    print("Tracker: ByteTrack (trackers.ByteTrackTracker)")
    print(f"Trajectories: {args.trajectories}")

    tags = ["-trajectories"] if args.trajectories else []
    if args.model_size != "medium":
        tags.append(f"-model-size-{args.model_size}")
    default_name = with_hyperparam_tags("roboflow-tracking-seg.mp4", tags)
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

    model = load_rfdetr_seg(args.model_size, device)
    tracker = ByteTrackTracker()
    mask_annotator = sv.MaskAnnotator(opacity=0.5)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    # Supervision TraceAnnotator — only used when --trajectories true
    trace_annotator = (
        sv.TraceAnnotator(trace_length=TRAIL_LENGTH, thickness=4)
        if args.trajectories
        else None
    )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (width, height),
    )
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    try:
        with progress_bar(total, "Roboflow RF-DETR seg-track") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = model.predict(frame, threshold=args.threshold)
                detections = tracker.update(detections)

                class_ids = detections.class_id
                confidences = detections.confidence
                tracker_ids = detections.tracker_id
                labels = []
                if class_ids is not None:
                    for i, cid in enumerate(class_ids):
                        name = COCO_CLASSES[cid] if cid is not None else ""
                        conf = (
                            float(confidences[i])
                            if confidences is not None
                            else None
                        )
                        tid = (
                            int(tracker_ids[i])
                            if tracker_ids is not None
                            else None
                        )
                        labels.append(format_track_label(tid, name, conf))

                frame = mask_annotator.annotate(scene=frame, detections=detections)
                frame = box_annotator.annotate(scene=frame, detections=detections)
                frame = label_annotator.annotate(
                    scene=frame, detections=detections, labels=labels
                )
                if trace_annotator is not None:
                    frame = trace_annotator.annotate(
                        scene=frame, detections=detections
                    )

                counts = class_counts_from_ids(
                    class_ids.tolist() if class_ids is not None else None,
                    COCO_CLASSES,
                )
                draw_class_count_badge(frame, counts, width)
                out.write(frame)
                pbar.update(1)
    finally:
        cap.release()
        out.release()

    print(f"Done. Wrote {output_path}")


if __name__ == "__main__":
    main()
