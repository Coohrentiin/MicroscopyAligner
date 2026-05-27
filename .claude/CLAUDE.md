# Manual_Registration_App — project context

A **PySide6** GUI for aligning microscopy images: a *template* (reference) image and a *moving*
image are overlaid, and the user finds an affine/similarity transform that registers the moving
image onto the template. Images are `float32` numpy arrays.

## Entry point
- `src/main.py` — builds the `QApplication` (Fusion dark palette) and shows `ImageAligner` from `src/gui.py`.

## Modules (`src/`)
- **`gui.py`** — `ImageAligner(QMainWindow)`, the main window and orchestrator.
  - State attrs: `template_image`, `moving_image`, `transformed_image` (numpy float32),
    `current_transform` (skimage `AffineTransform`/`SimilarityTransform`), `opt_transform`,
    `template_image_file`, `moving_image_file`.
  - Loaders: `load_template()` / `load_moving()` open a `QFileDialog` then delegate to the
    dialog-free `load_template_from_path(path)` / `load_moving_from_path(path)` (the latter
    re-applies `current_transform`, so the transform persists across moving-image swaps).
  - Alignment: `open_keypoints_tool()` (manual point picking), `open_auto_keypoints_tool()`
    (opens `KeyPointsDetectionAndSelection`, blocks on `exec()`, applies result on Accept),
    `optimize_phase_correlation()` (= cross-correlation, fully programmatic, translation only),
    `optimize_brute_force()` (grid search via `BruteForceDialog`).
  - Apply/export: `apply_transform(params)` warps `moving_image` and sets `current_transform`;
    `export_image()` (prompts, tif/png); `load_and_export_stack()` prompts for an input stack and
    delegates to `export_stack_from_path(in_path)` (warps every phase+amp frame, `save_stack`);
    `batch_process()` applies the current transform to a whole folder.
  - Batch Mode entry: `open_batch_mode()` (File menu) creates/shows `BatchModePanel`.
- **`transformControls.py`** — `TransformControls(QWidget)`: rotation/scale/tx/ty sliders+spinboxes,
  center-referenced affine math (`_params_to_affine`, `_affine_to_display_params`). Holds
  `transform_params` dict `{rotation, scale, tx, ty}`; emits `transform_changed` (wired to
  `apply_transform`). Key methods: `set_values_from_params`, `set_values_from_transform`,
  `set_template_shape(shape)`, `reset_transform()`.
- **`imageCanva.py`** — `ImageCanvas(QGraphicsView)`: overlay / side-by-side display, keypoint
  clicking, drag. Emits `point_added`, `template_dragged`.
- **`keyPointsSelection.py`** — manual point-pair dialog + `estimate_transform_keypoints(pairs)`.
- **`keyPointsDetectionAndSelection.py`** — `KeyPointsDetectionAndSelection(parent, template, moving)`:
  automatic detection (AKAZE/KAZE/SIFT/ORB/BRISK) + matching (BF/FLANN) + RANSAC. The detect/match/
  RANSAC logic also lives as Qt-free module functions (`detect_keypoints`, `match_keypoints`,
  `filter_with_ransac`, and the all-in-one `detect_keypoint_pairs(...)`) that the dialog delegates
  to and batch mode uses headlessly.
- **`autoAlignersGui.py`** — `BruteForceDialog`: parameter ranges for brute-force search.
- **`utils_images.py`** — `load_imgfile(path)` (tif/png/jpg/npy → float32),
  `load_wavefront_tif(path, frame_index)` (phase, amplitude, n_frames),
  `save_stack(path, stack, ...)` (ImageJ TZCYX TIFF).
- **`batchMode.py`** — `BatchModePanel(QWidget)`: separate window that **drives** `ImageAligner`.
  Holds an explicit `self.aligner` reference (not a Qt parent). A table of "sequences" (rows):
  each row = one template + several moving images (image paths only). Action sequences
  (`set_matrix`, `auto_keypoints`, `cross_correlation`, `reset`, `save_image`, `save_stack`) are
  keyed **per moving column** (`self.column_actions[col]`), shared by every row's image in that
  column; selecting a moving cell binds the editor to that column, and "Copy from column" copies
  another column's sequence. Browsing a cell uses multi-select (`getOpenFileNames`): picking
  several files `natsorted`-fills that column downward from the clicked row, auto-adding rows
  (load a whole folder of one condition in one shot). Navigation buttons step through actions / moving images / sequences;
  each action calls the corresponding `ImageAligner` method. Config saved/loaded as JSON
  (`{version, sequences, column_actions}`). Transform is kept across moving images and rows
  (only an explicit `reset` action clears it). `save_stack` treats the row's moving cell as the
  input multi-frame stack. Per-action options: `auto_keypoints` can run headlessly (`dialog: false`
  + detector/matcher/ransac params) via `ImageAligner.auto_keypoints_headless(...)`; `save_image` /
  `save_stack` can target a preset `folder` + `suffix` (output = `<stem><suffix><ext>`) via
  `save_image_to_folder` / `export_stack_to_folder`, else they prompt. A "Run All Actions" button
  runs the current moving image's remaining actions in one shot.

## Conventions
- GUI = PySide6 (`from PySide6.QtWidgets import *`, etc.). Transforms via `skimage.transform`.
- Warps use `transform.inverse` with `output_shape = template_image.shape`, `preserve_range=True`,
  cast to `float32`.
