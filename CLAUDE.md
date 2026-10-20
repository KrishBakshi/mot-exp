# CLAUDE.md

Multi-object tracking experiments on video with instance segmentation. See `README.md` for usage.

## Commands

```sh
uv sync                                               # install deps (Python 3.13)
uv run mot-ultralytics-track-seg <video> [output]     # YOLO-seg + BoT-SORT
uv run mot-roboflow-track-seg <video> [output]        # RF-DETR-Seg + BoT-SORT
```

There are no tests or linter config. To check a change, run the affected CLI on a short clip in `data/`
and look at the MP4 it writes under `output/<video_stem>/`.

## Structure

- `src/mot_exp/ultralytics_track_seg.py` and `src/mot_exp/roboflow_track_seg.py` are the two pipelines.
  Each one reads frames with OpenCV, runs detection and tracking, draws overlays, and writes an `avc1` MP4.
- Shared helpers: `annotate.py` (labels, class count badge), `cli_args.py` (shared flags, output naming),
  `device.py` (CUDA → MPS → CPU), `progress.py` (tqdm bar), `video_paths.py` (`data/` / `output/` resolution).
- `ultralytics/*.py` and `roboflow/*.py` are thin wrappers that call the package `main()`. Put logic in
  `src/mot_exp/`, not in the wrappers.
- Console scripts are declared in `pyproject.toml` under `[project.scripts]`.

## Conventions

- Keep the two pipelines consistent. Both use BoT-SORT with camera motion compensation (`sparseOptFlow`),
  label tracks as `#id class conf` via `format_track_label`, and draw `draw_class_count_badge` on every frame.
  A feature added to one should usually go into the other too.
- Overlays are colored by tracker ID, not class, so each instance keeps the same color across masks, boxes,
  labels and trails. The Roboflow pipeline drops unconfirmed tracks (`tracker_id < 0`) before drawing.
- Non-default hyperparameters are appended to the output filename with `with_hyperparam_tags`
  (e.g. `-trajectories`, `-model-size-large`). New CLI flags that change the output should add a tag.
- Boolean flags take an explicit value (`--trajectories true`) via `str2bool`, not `store_true`.
- Bare input filenames resolve under `data/`; bare output filenames go under `output/<video_stem>/`.

## Dependencies

- `opencv-python` is pinned `<4.13` and `trackers` to `2.4.x` because `inference` requires them.
  Check those constraints before upgrading either one.
- Manage dependencies with `uv add` / `uv lock`; `uv.lock` is committed.

## Git-ignored

`data/`, `output/`, `tmp/`, model weights (`*.pt`, `*.onnx`, ...), video files, and any path containing
`output`. The Ultralytics pipeline defaults to `./yolo26m-seg.pt` in the repo root.
