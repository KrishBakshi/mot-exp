# mot-exp

Experiments with multi-object tracking (MOT) on video using instance segmentation.
Two pipelines are included, both using **BoT-SORT** with camera motion compensation:

- **Ultralytics** — YOLO-seg via `solutions.InstanceSegmentation`
- **Roboflow** — RF-DETR-Seg + the `trackers` library

Each pipeline writes an annotated MP4 with segmentation masks, boxes, `#id class conf`
labels colored per track, a per-frame class count badge, and optional motion trails.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

Inference runs on CUDA, MPS, or CPU, whichever is available (checked in that order).

## Usage

Put input videos in `data/` (or pass any path to a video file).

```sh
# Ultralytics YOLO-seg
uv run mot-ultralytics-track-seg zeebra.mp4

# Roboflow RF-DETR-Seg
uv run mot-roboflow-track-seg zeebra.mp4
```

Output goes to `output/<video_stem>/` by default. You can pass a second positional argument
to set the output filename or path.

### Options

| Option | Pipeline | Description |
| --- | --- | --- |
| `--trajectories BOOL` | both | Draw motion trails (default: `false`) |
| `--model PATH` | Ultralytics | YOLO-seg weights (default: `./yolo26m-seg.pt`) |
| `--model-size SIZE` | Roboflow | `nano`, `small`, `medium`, `large`, `xlarge`, `2xlarge` (default: `medium`) |
| `--threshold FLOAT` | Roboflow | Detection confidence threshold (default: `0.5`) |

Enabled options are added to the output filename, e.g.
`roboflow-tracking-seg-trajectories-model-size-large.mp4`.

The scripts in `ultralytics/` and `roboflow/` are thin wrappers around the same CLIs, and
`roboflow/tracking-segment.ipynb` is an earlier notebook experiment with RF-DETR-Seg and ByteTrack.

## Layout

```
src/mot_exp/
  ultralytics_track_seg.py  # Ultralytics pipeline
  roboflow_track_seg.py     # Roboflow pipeline
  annotate.py               # Labels and class count badge
  cli_args.py               # Shared CLI flags and output naming
  device.py                 # CUDA / MPS / CPU selection
  progress.py               # Progress bar
  video_paths.py            # data/ and output/ path resolution
```

`data/`, `output/`, and model weights (`*.pt`, etc.) are git-ignored.
