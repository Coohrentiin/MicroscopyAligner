# Copyright (c) 2025 Corentin Soubeiran
# SPDX-License-Identifier: MIT
"""Progressive Folder Alignment panel.

A separate window that drives the main :class:`ImageAligner` across a *folder*
of moving items that each need a **different** transform (e.g. drift across an
acquisition).  This is the opposite case of "Apply to Folder", which applies one
matrix to everything.

Workflow
--------
1. Load one template and a folder of moving items (natsorted).  Each item may be
   a flat image or a multi-frame ImageJ wavefront stack (auto-detected via
   :func:`utils_images.load_wavefront_tif`'s ``n_frames``).
2. Click an item -> its representative 2-D frame (frame 0, phase channel) is sent
   to the main window for manual alignment.
3. Align a few items and mark each as an **anchor** (its final matrix is stored).
4. Build a reusable **action sequence** (set_matrix / auto_keypoints /
   cross_correlation / crop / uncrop / reset), like Batch Mode.  Headless
   ``auto_keypoints`` exposes the same constraints as Batch Mode: lock rotation
   and/or scale (a rigid / similarity fit) plus an optional residual distortion
   model.  ``crop`` restricts the *registration* to a centered percentage of
   both images, so the actions after it ignore the borders (drifting edges,
   vignetting); the fitted matrix is converted back to full-image coordinates
   when the crop is released, and the export still writes the full frame.
5. **Propagate**: for every non-anchor item, set the transform to a prealignment
   matrix interpolated between the two nearest anchors by list index
   (nearest-anchor copy outside the anchor range -- no extrapolation), then
   replay the action sequence on top to refine.
6. Optionally **smooth** the per-frame curves once every item has a matrix: a
   centered sliding window (mean / median / Savitzky-Golay) over the display
   params removes frame-to-frame glitches.  The result is drawn dashed on the 4
   graphs next to the raw curve and stored separately, so it is a preview until
   "Use smoothed at export" is ticked -- the raw matrices are never overwritten,
   and editing any matrix afterwards discards the stale curve.
7. Review per-item status + correlation score, fix outliers, then export aligned
   items + per-item matrices (``matrices.json``).  Items the wavefront loader can
   read -- multi-frame stacks and single-frame 2-channel files alike -- are
   written back as ImageJ stacks keeping phase + amplitude, as Edit > "Apply
   transform to a ImageJ stack" does; the "Save as" selector can force the stack
   or flat writer instead.

A previously exported ``matrices.json`` can be re-imported ("Load matrices.json")
to restore per-item transforms as anchors, optionally populating the item list
from the names it holds.  Items are matched by file name; when the names come
from a different naming scheme but the item counts agree, the import falls back
to the number parsed out of each name, then to plain list order, showing the
proposed pairing for confirmation first.

The prealignment is interpolated on the **center-referenced display params**
(scale / rotation / tx / ty) produced by
:meth:`TransformControls._affine_to_display_params`, which vary smoothly; the
result is turned back into a matrix by
:meth:`TransformControls._params_to_affine`.
"""
import json
import re
from pathlib import Path

import numpy as np
from natsort import natsorted
from scipy.signal import savgol_filter
from skimage import transform as tf

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QTableWidget,
    QTableWidgetItem, QListWidget, QComboBox, QDoubleSpinBox, QSpinBox, QLabel,
    QFileDialog, QMessageBox, QHeaderView, QCheckBox, QProgressDialog, QSplitter,
)
from PySide6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from utils_images import load_imgfile, load_wavefront_tif

IMAGE_FILTER = "Image Files (*.tif *.tiff *.png *.jpg *.npy);;TIFF Files (*.tif *.tiff)"
IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy")

# Subset of Batch Mode actions that make sense to replay during propagation
# (save_image / save_stack are excluded -- export is handled separately).
ACTION_TYPES = [
    ("set_matrix", "Set matrix values (scale / rotation / tx / ty)"),
    ("auto_keypoints", "Auto keypoint detection"),
    ("cross_correlation", "Cross-correlation alignment"),
    ("crop", "Crop to central region (registration only)"),
    ("uncrop", "Release crop (back to full frame)"),
    ("reset", "Reset transform"),
]
ACTION_LABELS = dict(ACTION_TYPES)

LOW_CORR_DEFAULT = 0.5


# --------------------------------------------------------------------- helpers
def _to_2d(img):
    """Reduce a possibly multi-channel image to a single 2-D array.

    ``load_imgfile`` returns ``H x W x 2`` for 2-channel wavefront files
    (channel 0 = phase / real); use channel 0 for alignment & scoring.
    """
    img = np.asarray(img)
    if img.ndim == 3:
        return img[..., 0]
    return img


def _translation(ox, oy, invert=False):
    """3x3 translation matrix; ``invert`` gives the shift *into* crop coords."""
    s = -1.0 if invert else 1.0
    return np.array([[1.0, 0.0, s * ox],
                     [0.0, 1.0, s * oy],
                     [0.0, 0.0, 1.0]], dtype=float)


def representative_frame(path):
    """Return ``(repr2d, n_frames)`` for a folder item.

    Tries the wavefront-stack loader first (frame 0, phase channel); falls back
    to :func:`load_imgfile` for plain png/jpg/npy that it cannot read.
    """
    repr2d, n_frames, _ = probe_item(path)
    return repr2d, n_frames


def probe_item(path):
    """Return ``(repr2d, n_frames, is_wavefront)`` for a folder item.

    ``is_wavefront`` is True when :func:`load_wavefront_tif` can read the file,
    i.e. it carries a size-2 (phase, amplitude) channel axis -- including
    **single-frame** ``(H, W, 2)`` / ``(2, H, W)`` files, which still have to be
    written back through the stack pipeline to keep both channels.
    """
    try:
        phase, _amp, n_frames = load_wavefront_tif(path, frame_index=0)
        return phase.astype(np.float32), int(n_frames), True
    except Exception:
        return _to_2d(load_imgfile(path)).astype(np.float32), 1, False


def correlation_score(template2d, warped2d):
    """Pearson correlation between two 2-D arrays (NaN-safe)."""
    a = np.asarray(template2d, dtype=np.float64).ravel()
    b = np.asarray(warped2d, dtype=np.float64).ravel()
    if a.size != b.size or a.size == 0:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def interpolate_params(anchors, idx):
    """Interpolate center-referenced display params for image ``idx``.

    Parameters
    ----------
    anchors : list of ``(anchor_idx, params_dict)``
        ``params_dict`` has keys ``scale, rotation, tx, ty`` (center-referenced).
        Need not be sorted.
    idx : int
        Target image index.

    Returns
    -------
    params dict, or ``None`` when there are no anchors (caller uses identity).

    Rules: 1 anchor or ``idx`` outside the anchor range -> copy the nearest
    anchor (no extrapolation).  Otherwise linearly interpolate each param
    between the two bracketing anchors; rotation is lerped on the shorter arc.
    """
    if not anchors:
        return None
    anchors = sorted(anchors, key=lambda a: a[0])
    if len(anchors) == 1:
        return dict(anchors[0][1])

    first_idx, last_idx = anchors[0][0], anchors[-1][0]
    if idx <= first_idx:
        return dict(anchors[0][1])
    if idx >= last_idx:
        return dict(anchors[-1][1])

    # Find the bracketing pair.
    for (i0, p0), (i1, p1) in zip(anchors, anchors[1:]):
        if i0 <= idx <= i1:
            if i1 == i0:
                return dict(p0)
            t = (idx - i0) / (i1 - i0)
            return {
                "scale": _lerp(p0["scale"], p1["scale"], t),
                "rotation": _lerp_angle(p0["rotation"], p1["rotation"], t),
                "tx": _lerp(p0["tx"], p1["tx"], t),
                "ty": _lerp(p0["ty"], p1["ty"], t),
            }
    # Shouldn't happen given the range checks above.
    return dict(anchors[-1][1])


def _lerp(a, b, t):
    return a + (b - a) * t


def _lerp_angle(a, b, t):
    """Lerp two angles (degrees) along the shorter arc."""
    diff = ((b - a + 180.0) % 360.0) - 180.0
    return a + diff * t


# ----------------------------------------------------------------------- panel
class ProgressiveFolderPanel(QWidget):
    """Top-level window driving an :class:`ImageAligner` across a folder.

    Data model
    ----------
    ``self.images``: list of dicts ``{path, n_frames, is_wavefront(bool),
    matrix(list|None), matrix_rel(list|None), is_anchor(bool), status, corr}``
    in natsorted order.
    ``matrix`` is the **global** transform (into the first-image / template
    frame, what export & display use); ``matrix_rel`` is the **relative**
    transform to the per-image reference and is only meaningful in sliding mode.
    In fixed-template mode the two are identical.
    ``self.actions``: ordered list of action dicts (shared sequence).

    Reference modes
    ---------------
    - ``MODE_FIXED``: every image aligns to the single loaded template.
    - ``MODE_SLIDING``: image N aligns to image ``N - offset`` (its preceding
      reference); the resulting relative transforms are composed into the
      first-image coordinate frame
      (``global[N] = global[N-offset] @ rel[N]``).  For ``N < offset`` the
      reference clamps to image 0 (``global[N] = rel[N]``); image 0 is identity.
    """

    STATUS_PENDING = "pending"
    STATUS_ANCHOR = "anchor"
    STATUS_PROP = "propagated"
    STATUS_LOW = "low"

    MODE_FIXED = "fixed"
    MODE_SLIDING = "sliding"

    def __init__(self, aligner):
        super().__init__()  # own top-level window (not parented to the aligner)
        self.aligner = aligner
        self.setWindowTitle("Progressive Folder Alignment")
        self.resize(1100, 800)

        self.template_path = ""
        self.images = []      # list of item dicts
        self.actions = []     # shared action sequence
        self._crop_state = None   # live registration crop (see _apply_crop)
        self.smooth_available = False  # a smoothed curve has been computed
        self.current_index = -1
        self.mode = self.MODE_FIXED
        self.offset = 1       # backward offset X for sliding mode

        self._init_ui()
        self._refresh_table()
        self._update_status()

    # --------------------------------------------------------------------- UI
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_load_section())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setSizes([520, 580])
        layout.addWidget(splitter, stretch=1)

        layout.addWidget(self._build_bottom_section())

        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.status_label)

    def _build_load_section(self):
        box = QGroupBox("1. Reference + moving folder")
        v = QVBoxLayout(box)

        h = QHBoxLayout()
        load_tpl = QPushButton("Load Template...")
        load_tpl.clicked.connect(self._load_template)
        load_tpl.setStyleSheet("QPushButton { background-color: #4CAF50; }")
        load_folder = QPushButton("Load Moving Folder...")
        load_folder.clicked.connect(self._load_folder)
        load_files = QPushButton("Load Moving Files...")
        load_files.clicked.connect(self._load_files)
        self.template_label = QLabel("Template: <none>")
        self.template_label.setStyleSheet("color: gray;")
        h.addWidget(load_tpl)
        h.addWidget(load_folder)
        h.addWidget(load_files)
        h.addWidget(self.template_label, stretch=1)
        v.addLayout(h)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Reference mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Fixed template", self.MODE_FIXED)
        self.mode_combo.addItem("Sliding (N − X)", self.MODE_SLIDING)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        self.offset_label = QLabel("Backward offset X:")
        mode_row.addWidget(self.offset_label)
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(1, 1000)
        self.offset_spin.setValue(self.offset)
        self.offset_spin.valueChanged.connect(self._on_offset_changed)
        mode_row.addWidget(self.offset_spin)
        self.mode_hint = QLabel()
        self.mode_hint.setStyleSheet("color: gray;")
        self.mode_hint.setWordWrap(True)
        mode_row.addWidget(self.mode_hint, stretch=1)
        v.addLayout(mode_row)

        self._on_mode_changed()
        return box

    def _on_mode_changed(self):
        self.mode = self.mode_combo.currentData()
        sliding = self.mode == self.MODE_SLIDING
        self.offset_label.setVisible(sliding)
        self.offset_spin.setVisible(sliding)
        if sliding:
            self.mode_hint.setText("Each image N aligns to image N−X; relative transforms are "
                                   "composed into the first image's frame on capture/propagate. "
                                   "The fixed template is still used to score against the global frame.")
        else:
            self.mode_hint.setText("Every image aligns to the single loaded template.")
        # Re-render the reference view for the current selection under the new mode.
        if 0 <= self.current_index < len(self.images):
            self._update_thumbnail(self.current_index)

    def _on_offset_changed(self, value):
        self.offset = int(value)

    def _build_left(self):
        box = QGroupBox("2. Moving items  (click a row to align it in the main window)")
        v = QVBoxLayout(box)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "Status", "Corr", "Anchor"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.currentCellChanged.connect(self._on_row_changed)
        self.table.cellDoubleClicked.connect(lambda r, c: self._send_to_window(r))
        v.addWidget(self.table)

        btns = QHBoxLayout()
        capture = QPushButton("Capture as Anchor")
        capture.setStyleSheet("QPushButton { background-color: #2196F3; }")
        capture.clicked.connect(self._capture_anchor)
        clear = QPushButton("Clear Anchor")
        clear.clicked.connect(self._clear_anchor)
        send = QPushButton("Send to Window")
        send.clicked.connect(lambda: self._send_to_window(self.table.currentRow()))
        btns.addWidget(send)
        btns.addWidget(capture)
        btns.addWidget(clear)
        v.addLayout(btns)

        hint = QLabel("Click a row to send its representative frame to the main window. Align there, "
                      "then 'Capture as Anchor' to store its final matrix. Mark a couple of anchors, "
                      "build the action sequence, then Propagate.")
        hint.setStyleSheet("color: gray;")
        hint.setWordWrap(True)
        v.addWidget(hint)
        return box

    def _build_right(self):
        box = QGroupBox("Visual tools")
        v = QVBoxLayout(box)

        self.thumb_fig = Figure(figsize=(4, 3), tight_layout=True)
        self.thumb_canvas = FigureCanvasQTAgg(self.thumb_fig)
        self.thumb_ax = self.thumb_fig.add_subplot(111)
        self.thumb_ax.set_title("Overlay (template green / moving red)")
        self.thumb_ax.axis("off")
        v.addWidget(self.thumb_canvas, stretch=1)

        self.plot_fig = Figure(figsize=(4, 3), tight_layout=True)
        self.plot_canvas = FigureCanvasQTAgg(self.plot_fig)
        v.addWidget(self.plot_canvas, stretch=1)
        return box

    def _build_bottom_section(self):
        box = QGroupBox("3. Action sequence  ·  Propagate  ·  Export")
        v = QVBoxLayout(box)

        # --- action sequence builder ---
        self.action_list = QListWidget()
        self.action_list.setMaximumHeight(110)
        v.addWidget(self.action_list)

        ctrl = QHBoxLayout()
        self.action_combo = QComboBox()
        for key, label in ACTION_TYPES:
            self.action_combo.addItem(label, key)
        self.action_combo.currentIndexChanged.connect(self._update_option_visibility)
        ctrl.addWidget(self.action_combo)

        # set_matrix options
        self.scale_spin = QDoubleSpinBox(); self.scale_spin.setRange(0.01, 100.0); self.scale_spin.setValue(1.0); self.scale_spin.setPrefix("scale ")
        self.rot_spin = QDoubleSpinBox(); self.rot_spin.setRange(-360.0, 360.0); self.rot_spin.setValue(0.0); self.rot_spin.setPrefix("rot ")
        self.tx_spin = QDoubleSpinBox(); self.tx_spin.setRange(-100000.0, 100000.0); self.tx_spin.setValue(0.0); self.tx_spin.setPrefix("tx ")
        self.ty_spin = QDoubleSpinBox(); self.ty_spin.setRange(-100000.0, 100000.0); self.ty_spin.setValue(0.0); self.ty_spin.setPrefix("ty ")
        self.matrix_spins = [self.scale_spin, self.rot_spin, self.tx_spin, self.ty_spin]
        for s in self.matrix_spins:
            ctrl.addWidget(s)

        # crop options
        self.crop_pct = QDoubleSpinBox()
        self.crop_pct.setRange(1.0, 100.0)
        self.crop_pct.setValue(50.0)
        self.crop_pct.setSuffix(" %")
        self.crop_pct.setPrefix("central ")
        self.crop_pct.setToolTip(
            "Percentage of each side kept, centered (50% = central half in width "
            "and height).\nRegistration only: the actions that follow see just "
            "this region, while the matrix stays in full-image coordinates and "
            "the export still writes the full frame.")
        ctrl.addWidget(self.crop_pct)

        # auto_keypoints options
        self.ak_use_dialog = QCheckBox("Open dialog")
        self.ak_use_dialog.toggled.connect(self._update_option_visibility)
        ctrl.addWidget(self.ak_use_dialog)
        self.ak_detector = QComboBox(); self.ak_detector.addItems(["AKAZE", "KAZE", "SIFT", "ORB", "BRISK"])
        self.ak_matcher = QComboBox(); self.ak_matcher.addItems(["Brute Force", "FLANN"])
        self.ak_ransac = QDoubleSpinBox(); self.ak_ransac.setRange(0.5, 20.0); self.ak_ransac.setValue(5.0); self.ak_ransac.setPrefix("RANSAC ")
        # Constraints (rigid: lock rotation and/or scale) + optional residual
        # distortion, mirroring Batch Mode's auto_keypoints options.
        self.ak_lock_rotation = QCheckBox("Lock rotation")
        self.ak_lock_scale = QCheckBox("Lock scale")
        self.ak_use_distortion = QCheckBox("Distortion")
        self.ak_use_distortion.toggled.connect(self._update_option_visibility)
        self.ak_distortion_model = QComboBox(); self.ak_distortion_model.addItems(["tps", "poly", "radial", "piecewise"])
        self.ak_headless_widgets = [self.ak_detector, self.ak_matcher, self.ak_ransac,
                                    self.ak_lock_rotation, self.ak_lock_scale,
                                    self.ak_use_distortion, self.ak_distortion_model]
        for w in self.ak_headless_widgets:
            ctrl.addWidget(w)

        ctrl.addStretch()
        add_act = QPushButton("Add Action")
        add_act.clicked.connect(self._add_action)
        ctrl.addWidget(add_act)
        v.addLayout(ctrl)

        edit_row = QHBoxLayout()
        rem_act = QPushButton("Remove Action"); rem_act.clicked.connect(self._remove_action)
        up = QPushButton("Move Up"); up.clicked.connect(lambda: self._move_action(-1))
        down = QPushButton("Move Down"); down.clicked.connect(lambda: self._move_action(1))
        for b in (rem_act, up, down):
            edit_row.addWidget(b)
        edit_row.addStretch()
        v.addLayout(edit_row)

        # --- curve smoothing (optional, post-propagation) ---
        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Smoothing:"))
        self.smooth_method = QComboBox()
        self.smooth_method.addItem("Moving average", "mean")
        self.smooth_method.addItem("Median", "median")
        self.smooth_method.addItem("Savitzky-Golay", "savgol")
        self.smooth_method.setToolTip(
            "Moving average: plain sliding-window mean, predictable.\n"
            "Median: rejects a single badly-aligned frame, can produce flat steps.\n"
            "Savitzky-Golay: follows genuine drift without flattening it.")
        smooth_row.addWidget(self.smooth_method)
        smooth_row.addWidget(QLabel("window:"))
        self.smooth_window = QSpinBox()
        self.smooth_window.setRange(3, 999)
        self.smooth_window.setValue(5)
        self.smooth_window.setSingleStep(2)
        self.smooth_window.setToolTip(
            "Number of frames in the sliding window (forced odd so it stays "
            "centered). Larger = smoother, but real motion is flattened too.")
        smooth_row.addWidget(self.smooth_window)
        smooth_btn = QPushButton("Smooth curves")
        smooth_btn.setToolTip(
            "Fit a smoothed curve through the per-frame scale / rotation / tx / ty "
            "once every frame has a matrix.\nDrawn dashed on the 4 graphs for "
            "comparison; nothing is overwritten and the export keeps using the raw "
            "matrices until 'Use smoothed' is ticked.")
        smooth_btn.clicked.connect(self._compute_smoothing)
        smooth_row.addWidget(smooth_btn)
        clear_smooth = QPushButton("Clear")
        clear_smooth.setToolTip("Discard the smoothed curve and go back to the raw matrices.")
        clear_smooth.clicked.connect(self._clear_smoothing)
        smooth_row.addWidget(clear_smooth)
        self.use_smooth = QCheckBox("Use smoothed at export")
        self.use_smooth.setEnabled(False)
        self.use_smooth.setToolTip(
            "When ticked, Export writes the smoothed matrices (and saves them to "
            "matrices.json) instead of the raw per-frame ones.")
        # toggled passes a bool; _update_status takes none.
        self.use_smooth.toggled.connect(lambda _checked: self._update_status())
        smooth_row.addWidget(self.use_smooth)
        smooth_row.addStretch()
        v.addLayout(smooth_row)

        # --- propagate + export ---
        run_row = QHBoxLayout()
        run_row.addWidget(QLabel("Low-corr threshold:"))
        self.low_thresh = QDoubleSpinBox(); self.low_thresh.setRange(0.0, 1.0); self.low_thresh.setSingleStep(0.05); self.low_thresh.setValue(LOW_CORR_DEFAULT)
        run_row.addWidget(self.low_thresh)
        propagate = QPushButton("Propagate (all non-anchors)")
        propagate.setStyleSheet("QPushButton { background-color: #9C27B0; }")
        propagate.clicked.connect(self._propagate)
        run_row.addWidget(propagate)
        export = QPushButton("Export Aligned + Matrices")
        export.setStyleSheet("QPushButton { background-color: #FF9800; }")
        export.clicked.connect(self._export)
        run_row.addWidget(export)
        run_row.addWidget(QLabel("Save as:"))
        self.save_format = QComboBox()
        self.save_format.addItem("Auto (keep input layout)", "auto")
        self.save_format.addItem("ImageJ stack (phase+amp)", "stack")
        self.save_format.addItem("Flat 2-D (phase only)", "flat")
        self.save_format.setToolTip(
            "Auto: multi-channel / multi-frame wavefront items are written back as "
            "ImageJ stacks (phase+amplitude, all frames), flat images stay flat.\n"
            "ImageJ stack: force the stack writer for every wavefront item.\n"
            "Flat 2-D: write only the warped phase channel of frame 0.")
        run_row.addWidget(self.save_format)
        run_row.addStretch()
        save_cfg = QPushButton("Save Config"); save_cfg.clicked.connect(self._save_config)
        load_cfg = QPushButton("Load Config"); load_cfg.clicked.connect(self._load_config)
        load_mat = QPushButton("Load matrices.json")
        load_mat.setToolTip(
            "Re-import per-item matrices exported by a previous run.\n"
            "Matched by file name; when the names differ but the counts agree, "
            "falls back to the number parsed from the names, then to list order "
            "(with a confirmation preview).")
        load_mat.clicked.connect(self._load_matrices)
        run_row.addWidget(save_cfg)
        run_row.addWidget(load_cfg)
        run_row.addWidget(load_mat)
        v.addLayout(run_row)

        self._update_option_visibility()
        return box

    # ------------------------------------------------------- action visibility
    def _set_visible(self, widgets, visible):
        for w in widgets:
            w.setVisible(visible)

    def _update_option_visibility(self):
        atype = self.action_combo.currentData()
        self._set_visible(self.matrix_spins, atype == "set_matrix")
        self.crop_pct.setVisible(atype == "crop")
        is_ak = atype == "auto_keypoints"
        self.ak_use_dialog.setVisible(is_ak)
        headless = is_ak and not self.ak_use_dialog.isChecked()
        self._set_visible(self.ak_headless_widgets, headless)
        # The distortion model combo only matters when distortion is on.
        self.ak_distortion_model.setVisible(headless and self.ak_use_distortion.isChecked())

    # ------------------------------------------------------------ load actions
    def _load_template(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Template", "", IMAGE_FILTER)
        if not path:
            return
        self.template_path = path
        self.template_label.setText(f"Template: {Path(path).name}")
        self.template_label.setStyleSheet("")
        # Push it into the main window immediately so anchoring works.
        self.aligner.load_template_from_path(path)
        self._refresh_plot()

    def _load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Moving Folder")
        if not folder:
            return
        paths = [str(p) for p in Path(folder).iterdir()
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        self._set_images(paths)

    def _load_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Moving Files", "", IMAGE_FILTER)
        if paths:
            self._set_images(list(paths))

    def _set_images(self, paths):
        if not paths:
            return
        paths = natsorted(paths)
        progress = QProgressDialog("Scanning items...", "Cancel", 0, len(paths), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.images = []
        for i, p in enumerate(paths):
            if progress.wasCanceled():
                break
            try:
                _repr, n_frames, is_wf = probe_item(p)
            except Exception:
                n_frames, is_wf = 1, False
            self.images.append({
                "path": p, "n_frames": n_frames, "is_wavefront": is_wf,
                "matrix": None, "matrix_rel": None, "matrix_smooth": None,
                "is_anchor": False, "status": self.STATUS_PENDING, "corr": None,
            })
            progress.setValue(i + 1)
        progress.close()
        self.current_index = -1
        self._refresh_table()
        self._refresh_plot()
        self._update_status()

    # ------------------------------------------------------ table <-> model
    def _refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.images))
        for r, item in enumerate(self.images):
            name = Path(item["path"]).name
            if item["n_frames"] > 1:
                name += f"  [stack ×{item['n_frames']}]"
            elif item.get("is_wavefront"):
                name += "  [wavefront 2ch]"
            cells = [
                name,
                item["status"],
                "" if item["corr"] is None else f"{item['corr']:.3f}",
                "✔" if item["is_anchor"] else "",
            ]
            for c, text in enumerate(cells):
                cell = QTableWidgetItem(text)
                if c == 0:
                    cell.setToolTip(item["path"])
                self.table.setItem(r, c, cell)
        self.table.blockSignals(False)

    def _update_row(self, r):
        """Refresh a single row in place (status / corr / anchor cells)."""
        item = self.images[r]
        self.table.item(r, 1).setText(item["status"])
        self.table.item(r, 2).setText("" if item["corr"] is None else f"{item['corr']:.3f}")
        self.table.item(r, 3).setText("✔" if item["is_anchor"] else "")

    def _on_row_changed(self, row, col, prev_row, prev_col):
        if 0 <= row < len(self.images):
            self._send_to_window(row)

    # ------------------------------------------------------------ window drive
    def _send_to_window(self, row):
        """Load the reference + this item's representative frame into the aligner.

        In sliding mode the reference is image ``N-X`` (pushed in as the aligner
        template) and the transform shown/edited is the **relative** one; in
        fixed mode it is the loaded template and the transform is global.
        """
        if not (0 <= row < len(self.images)):
            return
        if not self.template_path:
            QMessageBox.warning(self, "Progressive Folder", "Load a template first.")
            return
        item = self.images[row]
        self.current_index = row
        # Selecting an item replaces both aligner images, so any crop state kept
        # from an earlier run refers to images that are no longer loaded.
        self._crop_state = None
        if self.aligner.template_image is None:
            self.aligner.load_template_from_path(self.template_path)
        try:
            repr2d, _ = representative_frame(item["path"])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read {Path(item['path']).name}: {e}")
            return
        # Set the per-image reference as the aligner template (sliding mode).
        ref_idx = self._reference_index(row)
        if ref_idx is not None:
            self.aligner.set_template_array(self._reference_repr(row), self.images[ref_idx]["path"])
        elif self.mode == self.MODE_SLIDING:
            # image 0 (or N<offset clamped to 0): reference is image 0 itself /
            # the global origin -> use the loaded template as reference.
            self.aligner.load_template_from_path(self.template_path)
        else:
            self.aligner.load_template_from_path(self.template_path)
        # Apply the stored transform (relative in sliding mode) so the user can edit it.
        edit_matrix = item.get("matrix_rel") if self.mode == self.MODE_SLIDING else item.get("matrix")
        if edit_matrix is not None:
            m = np.array(edit_matrix, dtype=float)
            self.aligner.current_transform = tf.AffineTransform(matrix=m)
            self.aligner.transform_controls.set_values_from_transform(m)
        self.aligner.set_moving_array(repr2d, item["path"])
        ref_txt = (f" vs {Path(self.images[ref_idx]['path']).name}" if ref_idx is not None
                   else " vs template")
        self.aligner.statusBar().showMessage(
            f"Progressive: aligning {Path(item['path']).name}{ref_txt} ({row + 1}/{len(self.images)})")
        self._update_thumbnail(row)
        self._update_status()

    def _store_result(self, row, rel_matrix, status):
        """Record a result for image ``row``.

        ``rel_matrix`` is the transform as produced in the main window
        (relative to the per-image reference in sliding mode, global in fixed
        mode). It is composed into the global frame for ``matrix`` and scored
        against the loaded template. Returns the stored global matrix.
        """
        item = self.images[row]
        rel = np.asarray(rel_matrix, dtype=float)
        global_m = self._compose_global(row, rel)
        item["matrix_rel"] = rel.tolist()
        item["matrix"] = global_m.tolist()
        item["status"] = status
        item["corr"] = self._score_item(item, global_m)
        # The raw curve moved, so any smoothed curve is now stale.
        self._invalidate_smoothing()
        return global_m

    def _capture_anchor(self):
        row = self.table.currentRow()
        if not (0 <= row < len(self.images)):
            return
        if self.aligner.current_transform is None:
            QMessageBox.warning(self, "Progressive Folder",
                                "No transform set in the main window yet.")
            return
        self._store_result(row, self.aligner.current_transform.params, self.STATUS_ANCHOR)
        self.images[row]["is_anchor"] = True
        self._update_row(row)
        self._refresh_plot()
        self._update_thumbnail(row)
        self._update_status()

    def _clear_anchor(self):
        row = self.table.currentRow()
        if not (0 <= row < len(self.images)):
            return
        item = self.images[row]
        item["is_anchor"] = False
        item["matrix"] = None
        item["matrix_rel"] = None
        item["matrix_smooth"] = None
        item["status"] = self.STATUS_PENDING
        item["corr"] = None
        self._invalidate_smoothing()
        self._update_row(row)
        self._refresh_plot()
        self._update_status()

    # ------------------------------------------------------------ action edit
    def _action_summary(self, action):
        t = action.get("type")
        if t == "set_matrix":
            return (f"Set matrix  scale={action.get('scale', 1.0):g}  rot={action.get('rotation', 0.0):g}  "
                    f"tx={action.get('tx', 0.0):g}  ty={action.get('ty', 0.0):g}")
        if t == "crop":
            return (f"Crop to central {action.get('pct', 50.0):g}%  "
                    "(registration only)")
        if t == "auto_keypoints":
            if action.get("dialog", False):
                return "Auto keypoint detection (dialog)"
            extra = []
            if action.get("lock_rotation"):
                extra.append("lock rot")
            if action.get("lock_scale"):
                extra.append("lock scale")
            if action.get("distortion_model"):
                extra.append(f"distortion {action['distortion_model']}")
            extra_txt = (", " + ", ".join(extra)) if extra else ""
            return (f"Auto keypoint detection (headless: {action.get('detector', 'AKAZE')} / "
                    f"{action.get('matcher', 'Brute Force')}, "
                    f"RANSAC={action.get('ransac_threshold', 5.0):g}{extra_txt})")
        return ACTION_LABELS.get(t, t)

    def _add_action(self):
        atype = self.action_combo.currentData()
        if atype == "set_matrix":
            action = {"type": "set_matrix", "scale": self.scale_spin.value(),
                      "rotation": self.rot_spin.value(), "tx": self.tx_spin.value(),
                      "ty": self.ty_spin.value()}
        elif atype == "crop":
            action = {"type": "crop", "pct": self.crop_pct.value()}
        elif atype == "auto_keypoints":
            action = {"type": "auto_keypoints", "dialog": self.ak_use_dialog.isChecked()}
            if not action["dialog"]:
                action.update({"detector": self.ak_detector.currentText(),
                               "matcher": self.ak_matcher.currentText(),
                               "ransac_threshold": self.ak_ransac.value(),
                               "lock_rotation": self.ak_lock_rotation.isChecked(),
                               "lock_scale": self.ak_lock_scale.isChecked(),
                               "distortion_model": (self.ak_distortion_model.currentText()
                                                    if self.ak_use_distortion.isChecked() else None)})
        else:
            action = {"type": atype}
        self.actions.append(action)
        self.action_list.addItem(self._action_summary(action))

    def _remove_action(self):
        idx = self.action_list.currentRow()
        if 0 <= idx < len(self.actions):
            del self.actions[idx]
            self.action_list.takeItem(idx)

    def _move_action(self, delta):
        idx = self.action_list.currentRow()
        new_idx = idx + delta
        if not (0 <= idx < len(self.actions) and 0 <= new_idx < len(self.actions)):
            return
        self.actions[idx], self.actions[new_idx] = self.actions[new_idx], self.actions[idx]
        self._refresh_action_list()
        self.action_list.setCurrentRow(new_idx)

    def _refresh_action_list(self):
        self.action_list.clear()
        for a in self.actions:
            self.action_list.addItem(self._action_summary(a))

    # ---------------------------------------------------------- crop handling
    @staticmethod
    def _crop_box(shape, pct):
        """Centered crop of ``pct`` percent of each side -> (y0, y1, x0, x1).

        The offset ``(x0, y0)`` is what turns cropped coordinates back into
        full-image ones. Always keeps at least a 2x2 region.
        """
        h, w = shape[:2]
        frac = max(1.0, min(100.0, float(pct))) / 100.0
        ch, cw = max(2, int(round(h * frac))), max(2, int(round(w * frac)))
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        return y0, y0 + ch, x0, x0 + cw

    def _apply_crop(self, pct):
        """Restrict the aligner to the central ``pct`` % of template + moving.

        Registration-only: the images handed to the aligner are cropped, so
        cross-correlation and keypoint detection see the central region alone,
        but the crop offsets are remembered so :meth:`_release_crop` can express
        the resulting matrix back in full-image coordinates. Export is untouched.
        """
        a = self.aligner
        if a.template_image is None or a.moving_image is None:
            return
        # Remember the full-frame images so the crop can be released.
        if self._crop_state is None:
            self._crop_state = {
                "template": a.template_image, "template_file": a.template_image_file,
                "moving": a.moving_image, "moving_file": a.moving_image_file,
            }
        full_t = self._crop_state["template"]
        full_m = self._crop_state["moving"]

        # Take the current transform back to full-image coordinates *before*
        # the new offsets replace the old ones -- re-cropping while a crop is
        # already live would otherwise convert from the wrong origin.
        matrix = None
        if a.current_transform is not None:
            matrix = self._to_full_coords(
                np.asarray(a.current_transform.params, dtype=float))

        ty0, ty1, tx0, tx1 = self._crop_box(full_t.shape, pct)
        my0, my1, mx0, mx1 = self._crop_box(full_m.shape, pct)
        self._crop_state["offset_t"] = (float(tx0), float(ty0))
        self._crop_state["offset_m"] = (float(mx0), float(my0))
        self._crop_state["pct"] = pct

        a.set_template_array(full_t[ty0:ty1, tx0:tx1],
                             self._crop_state["template_file"])
        a.set_moving_array(full_m[my0:my1, mx0:mx1], self._crop_state["moving_file"])
        if matrix is not None:
            cropped = self._to_crop_coords(matrix)
            a.transform_controls.set_values_from_transform(cropped)

    def _release_crop(self):
        """Undo :meth:`_apply_crop`, converting the matrix back to full coords.

        Safe to call when no crop is active. Returns True when a crop was live.
        """
        if self._crop_state is None:
            return False
        a = self.aligner
        state, self._crop_state = self._crop_state, None
        matrix = None
        if a.current_transform is not None:
            matrix = np.asarray(a.current_transform.params, dtype=float)
        a.set_template_array(state["template"], state["template_file"])
        a.set_moving_array(state["moving"], state["moving_file"])
        if matrix is not None:
            full = self._to_full_coords(matrix, state)
            a.transform_controls.set_values_from_transform(full)
        return True

    def _to_crop_coords(self, matrix, state=None):
        """Full-image matrix -> cropped-image matrix (``T_t @ M @ inv(T_m)``)."""
        state = state or self._crop_state
        if state is None or "offset_t" not in state:
            return np.asarray(matrix, dtype=float)
        t_t = _translation(*state["offset_t"], invert=True)
        t_m = _translation(*state["offset_m"], invert=True)
        return t_t @ np.asarray(matrix, dtype=float) @ np.linalg.inv(t_m)

    def _to_full_coords(self, matrix, state=None):
        """Cropped-image matrix -> full-image matrix (``inv(T_t) @ M @ T_m``).

        A crop is a pure translation of each image's origin, so a transform
        fitted on the crops differs from the full-frame one whenever rotation or
        scale is involved -- this puts the origin back.
        """
        state = state or self._crop_state
        if state is None or "offset_t" not in state:
            return np.asarray(matrix, dtype=float)
        t_t = _translation(*state["offset_t"], invert=True)
        t_m = _translation(*state["offset_m"], invert=True)
        return np.linalg.inv(t_t) @ np.asarray(matrix, dtype=float) @ t_m

    def _execute_action(self, action):
        """Replay one action on the aligner (mirrors batchMode._execute_action)."""
        t = action.get("type")
        a = self.aligner
        if t == "set_matrix":
            a.transform_controls.set_values_from_params({
                "scale": action.get("scale", 1.0), "rotation": action.get("rotation", 0.0),
                "tx": action.get("tx", 0.0), "ty": action.get("ty", 0.0)})
        elif t == "auto_keypoints":
            if action.get("dialog", False):
                a.open_auto_keypoints_tool()
            else:
                a.auto_keypoints_headless(
                    detector=action.get("detector", "AKAZE"),
                    matcher=action.get("matcher", "Brute Force"),
                    distance_ratio=action.get("distance_ratio", 0.75),
                    use_ransac=action.get("use_ransac", True),
                    ransac_threshold=action.get("ransac_threshold", 5.0),
                    lock_rotation=action.get("lock_rotation", False),
                    lock_scale=action.get("lock_scale", False),
                    distortion_model=action.get("distortion_model"))
        elif t == "cross_correlation":
            a.optimize_phase_correlation()
        elif t == "crop":
            self._apply_crop(action.get("pct", 50.0))
        elif t == "uncrop":
            self._release_crop()
        elif t == "reset":
            a.transform_controls.reset_transform()

    # --------------------------------------------------- sliding-mode helpers
    def _reference_index(self, idx):
        """Index of the reference image for aligning image ``idx``.

        Fixed mode: ``None`` (the loaded template is the reference). Sliding
        mode: ``idx - offset`` clamped to 0; ``None`` when ``idx == 0`` (image 0
        is the global origin, identity transform).
        """
        if self.mode != self.MODE_SLIDING:
            return None
        if idx <= 0:
            return None
        return max(0, idx - self.offset)

    def _reference_repr(self, idx):
        """2-D reference image used to align image ``idx`` (sliding mode), or
        the loaded template (fixed mode / image 0)."""
        ref_idx = self._reference_index(idx)
        if ref_idx is None:
            return _to_2d(self.aligner.template_image) if self.aligner.template_image is not None else None
        repr2d, _ = representative_frame(self.images[ref_idx]["path"])
        return repr2d

    def _global_matrix(self, idx):
        """The stored global matrix for image ``idx`` as an array, or None."""
        if not (0 <= idx < len(self.images)):
            return None
        m = self.images[idx].get("matrix")
        return None if m is None else np.array(m, dtype=float)

    def _compose_global(self, idx, rel_matrix):
        """Compose a relative transform (idx -> its reference) into the global
        (first-image) frame: ``global[idx] = global[ref] @ rel``.

        Fixed mode (or image 0 in sliding mode) returns ``rel`` unchanged.
        """
        ref_idx = self._reference_index(idx)
        if ref_idx is None:
            return np.asarray(rel_matrix, dtype=float)
        ref_global = self._global_matrix(ref_idx)
        if ref_global is None:
            # Reference not aligned yet -> fall back to treating rel as global.
            return np.asarray(rel_matrix, dtype=float)
        return ref_global @ np.asarray(rel_matrix, dtype=float)

    # ------------------------------------------------------------- propagation
    def _anchor_list(self):
        """Anchors as ``(idx, display_params)`` for prealign interpolation.

        Interpolates over the **relative** transform in sliding mode (so the
        per-image refinement starts near the right relative pose) and over the
        global transform in fixed mode.
        """
        out = []
        tc = self.aligner.transform_controls
        key = "matrix_rel" if self.mode == self.MODE_SLIDING else "matrix"
        for i, item in enumerate(self.images):
            m = item.get(key) if item["is_anchor"] else None
            if m is not None:
                out.append((i, tc._affine_to_display_params(np.array(m, dtype=float))))
        return out

    def _propagate(self):
        if self.aligner.template_image is None and self.template_path:
            self.aligner.load_template_from_path(self.template_path)
        if not self.template_path:
            QMessageBox.warning(self, "Progressive Folder", "Load a template first.")
            return
        anchors = self._anchor_list()
        if not anchors:
            QMessageBox.warning(self, "Progressive Folder",
                                "Mark at least one anchor before propagating.")
            return
        # Sliding mode iterates ALL images in index order so each image's
        # reference already has its global transform when we compose -- anchors
        # included (their global is recomposed from the now-available reference,
        # but their relative transform / anchor status are kept). Fixed mode only
        # touches non-anchors and order is irrelevant.
        sliding = self.mode == self.MODE_SLIDING
        if sliding:
            order = list(range(len(self.images)))
        else:
            order = [i for i, it in enumerate(self.images) if not it["is_anchor"]]
        if not order:
            QMessageBox.information(self, "Progressive Folder", "Nothing to propagate.")
            return

        low = self.low_thresh.value()
        tc = self.aligner.transform_controls
        # Never start a run with a crop left over from a previous one.
        self._crop_state = None
        progress = QProgressDialog("Propagating alignment...", "Cancel", 0, len(order), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        for n, i in enumerate(order):
            if progress.wasCanceled():
                break
            item = self.images[i]

            # Anchors: just recompose their global from the (now-aligned)
            # reference, keeping the user-set relative transform.
            if item["is_anchor"]:
                if sliding and item.get("matrix_rel") is not None:
                    global_m = self._compose_global(i, np.array(item["matrix_rel"], dtype=float))
                    item["matrix"] = global_m.tolist()
                    item["corr"] = self._score_item(item, global_m)
                    self._invalidate_smoothing()
                    self._update_row(i)
                progress.setValue(n + 1)
                continue

            try:
                repr2d, _ = representative_frame(item["path"])
            except Exception:
                continue
            # Swap the reference (sliding: image N-X; fixed/image0: the template).
            ref_idx = self._reference_index(i)
            if ref_idx is not None:
                self.aligner.set_template_array(self._reference_repr(i), self.images[ref_idx]["path"])
            elif sliding:
                self.aligner.load_template_from_path(self.template_path)
            # 1. prealign: interpolated (relative) matrix as starting transform
            params = interpolate_params(anchors, i)
            self.aligner.set_moving_array(repr2d, item["path"])
            if params is None:
                tc.reset_transform()
            else:
                tc.set_values_from_transform(tc._params_to_affine(params))
            # 2. replay the action sequence to refine on top
            for action in self.actions:
                self._execute_action(action)
            # A crop left open by the sequence is registration-only: release it
            # so the stored matrix is in full-image coordinates.
            self._release_crop()
            # 3. compose to global, record + score (vs the global template)
            self._store_result(i, self.aligner.current_transform.params, self.STATUS_PROP)
            if (item["corr"] is not None and not np.isnan(item["corr"]) and item["corr"] < low):
                item["status"] = self.STATUS_LOW
            self._update_row(i)
            progress.setValue(n + 1)

        progress.close()
        # A cancel mid-item can leave the aligner holding cropped images.
        self._release_crop()
        # Restore the fixed template in the main window after sliding propagation.
        if self.mode == self.MODE_SLIDING:
            self.aligner.load_template_from_path(self.template_path)
        self._refresh_plot()
        self._update_status()
        n_prop = sum(1 for it in self.images if it["status"] in (self.STATUS_PROP, self.STATUS_LOW))
        self.aligner.statusBar().showMessage(
            f"Progressive: propagated {n_prop} item(s).")

    def _global_template2d(self):
        """The fixed (global-frame) template as a 2-D array, read from disk so
        it is independent of whatever the aligner's template is currently set
        to (sliding mode swaps it per image)."""
        if not self.template_path:
            return None
        try:
            return _to_2d(load_imgfile(self.template_path)).astype(np.float32)
        except Exception:
            return None

    def _score_item(self, item, global_matrix):
        """Correlation of the item's globally-warped frame vs the fixed template."""
        template2d = self._global_template2d()
        if template2d is None:
            return None
        try:
            repr2d, _ = representative_frame(item["path"])
        except Exception:
            return None
        transform = tf.AffineTransform(matrix=np.asarray(global_matrix, dtype=float))
        warped = tf.warp(repr2d, transform.inverse,
                         output_shape=template2d.shape, preserve_range=True)
        return correlation_score(template2d, warped)

    # ------------------------------------------------------------- visual tools
    def _update_thumbnail(self, row):
        self.thumb_ax.clear()
        title = ("Overlay (template green / aligned red)" if self.mode == self.MODE_FIXED
                 else "Overlay vs global frame (img0 green / aligned red)")
        self.thumb_ax.set_title(title, fontsize=8)
        self.thumb_ax.axis("off")
        template2d = self._global_template2d()
        if template2d is not None and 0 <= row < len(self.images):
            try:
                repr2d, _ = representative_frame(self.images[row]["path"])
            except Exception:
                repr2d = None
            if repr2d is not None:
                # Always overlay against the fixed (global) template using the
                # composed global matrix, so the preview is comparable across modes.
                matrix = self.images[row].get("matrix")
                if matrix is not None:
                    transform = tf.AffineTransform(matrix=np.array(matrix, dtype=float))
                    repr2d = tf.warp(repr2d, transform.inverse,
                                     output_shape=template2d.shape, preserve_range=True)
                rgb = np.zeros(template2d.shape + (3,), dtype=np.float32)
                rgb[..., 1] = _norm01(template2d)
                rgb[..., 0] = _norm01(_resize_to(repr2d, template2d.shape))
                self.thumb_ax.imshow(rgb)
        self.thumb_canvas.draw_idle()

    # -------------------------------------------------------------- smoothing
    def _smooth_series(self, ys, window, method, polyorder=2):
        """Smooth one parameter series with a centered sliding window.

        ``ys`` is dense (one value per frame that has a matrix). Edges use a
        shrinking window ("nearest"-style) rather than padding, so the first and
        last frames keep their own value's weight instead of being pulled toward
        a constant. Returns a list the same length as ``ys``.
        """
        arr = np.asarray(ys, dtype=float)
        n = arr.size
        if n == 0:
            return []
        # An even window has no centre sample; bump it so the result is centred.
        w = max(1, int(window))
        if w % 2 == 0:
            w += 1
        if w <= 1 or n == 1:
            return arr.tolist()
        w = min(w, n if n % 2 else n - 1)  # keep it odd and within the data
        w = max(w, 1)

        if method == "savgol":
            po = min(int(polyorder), w - 1)
            if w <= 1 or po < 1:
                return arr.tolist()
            return savgol_filter(arr, w, po, mode="nearest").tolist()

        # Reflect about the endpoints ("odd"/antisymmetric) rather than
        # shrinking the window: a shrinking window averages [y0, y1, y2] at
        # index 0 and drags the first and last frames toward the interior,
        # which is a visible offset on a drifting sequence. Odd reflection
        # continues the local slope, so a straight ramp smooths to itself.
        half = w // 2
        left = 2.0 * arr[0] - arr[1:half + 1][::-1]
        right = 2.0 * arr[-1] - arr[-half - 1:-1][::-1]
        padded = np.concatenate([left, arr, right])
        out = np.empty(n, dtype=float)
        for i in range(n):
            seg = padded[i:i + w]
            out[i] = np.median(seg) if method == "median" else np.mean(seg)
        return out.tolist()

    def _compute_smoothing(self):
        """Build the smoothed matrix for every frame that has one.

        Smoothing runs on the **display params** (scale / rotation / tx / ty),
        the same center-referenced space the prealignment interpolates in, since
        those vary smoothly frame to frame where raw matrix entries do not.
        Results are stored per item as ``matrix_smooth``; nothing else is
        touched, so the raw curve stays available for comparison and the user
        chooses at export which one to write.
        """
        tc = self.aligner.transform_controls
        idxs = [i for i, it in enumerate(self.images) if it["matrix"] is not None]
        if len(idxs) < 2:
            QMessageBox.warning(self, "Progressive Folder",
                                "Need at least two items with a matrix to smooth.")
            return False

        series = {k: [] for k in ("scale", "rotation", "tx", "ty")}
        for i in idxs:
            p = tc._affine_to_display_params(np.array(self.images[i]["matrix"], dtype=float))
            for k in series:
                series[k].append(p[k])

        window = self.smooth_window.value()
        method = self.smooth_method.currentData()
        smoothed = {k: self._smooth_series(v, window, method) for k, v in series.items()}

        for pos, i in enumerate(idxs):
            params = {k: smoothed[k][pos] for k in series}
            self.images[i]["matrix_smooth"] = tc._params_to_affine(params).tolist()
        # Frames without a raw matrix cannot have a smoothed one.
        for i, it in enumerate(self.images):
            if it["matrix"] is None:
                it["matrix_smooth"] = None

        self.smooth_available = True
        self.use_smooth.setEnabled(True)
        self._refresh_plot()
        self._update_status()
        label = self.smooth_method.currentText()
        self.status_label.setText(
            f"Smoothed {len(idxs)} matrices ({label}, window {window}). "
            "Compare the dashed curve, then tick 'Use smoothed' to export it.")
        return True

    def _invalidate_smoothing(self):
        """Drop a stale smoothed curve after the raw matrices changed.

        Cheap and redraw-free: ``_store_result`` calls this once per frame during
        propagation, and its callers already refresh the plot afterwards.
        """
        if not self.smooth_available:
            return
        for it in self.images:
            it["matrix_smooth"] = None
        self.smooth_available = False
        self.use_smooth.setChecked(False)
        self.use_smooth.setEnabled(False)

    def _clear_smoothing(self):
        """Drop the smoothed curve and fall back to the raw matrices."""
        had = self.smooth_available
        self._invalidate_smoothing()
        # Clear unconditionally: a partial curve can outlive the flag.
        for it in self.images:
            it["matrix_smooth"] = None
        self._refresh_plot()
        self._update_status()
        if had:
            self.status_label.setText("Smoothed curve discarded; using raw matrices.")

    def _export_matrix(self, item):
        """The matrix to export for ``item``: smoothed when the user opted in."""
        if self.use_smooth.isChecked() and item.get("matrix_smooth") is not None:
            return np.array(item["matrix_smooth"], dtype=float)
        return np.array(item["matrix"], dtype=float)

    def _refresh_plot(self):
        self.plot_fig.clear()
        if not self.images:
            self.plot_canvas.draw_idle()
            return
        tc = self.aligner.transform_controls
        idxs, scales, rots, txs, tys, anchor_idx = [], [], [], [], [], []
        s_idxs, s_scales, s_rots, s_txs, s_tys = [], [], [], [], []
        for i, item in enumerate(self.images):
            if item["matrix"] is None:
                continue
            p = tc._affine_to_display_params(np.array(item["matrix"], dtype=float))
            idxs.append(i); scales.append(p["scale"]); rots.append(p["rotation"])
            txs.append(p["tx"]); tys.append(p["ty"])
            if item["is_anchor"]:
                anchor_idx.append(i)
            if item.get("matrix_smooth") is not None:
                sp = tc._affine_to_display_params(
                    np.array(item["matrix_smooth"], dtype=float))
                s_idxs.append(i); s_scales.append(sp["scale"]); s_rots.append(sp["rotation"])
                s_txs.append(sp["tx"]); s_tys.append(sp["ty"])

        specs = [("scale", scales, s_scales), ("rotation (deg)", rots, s_rots),
                 ("tx", txs, s_txs), ("ty", tys, s_tys)]
        for k, (title, ys, sys_) in enumerate(specs):
            ax = self.plot_fig.add_subplot(2, 2, k + 1)
            ax.plot(idxs, ys, "-o", markersize=3, label="raw")
            for ai in anchor_idx:
                if ai in idxs:
                    ax.plot(ai, ys[idxs.index(ai)], "s", color="red", markersize=6)
            if sys_:
                # Dashed overlay so raw vs smoothed is readable at a glance; the
                # export follows the "Use smoothed" tick, not what is drawn here.
                ax.plot(s_idxs, sys_, "--", color="#2E7D32", linewidth=1.6,
                        label="smoothed")
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=6)
            if sys_ and k == 0:
                ax.legend(fontsize=6, loc="best")
        self.plot_fig.tight_layout()
        self.plot_canvas.draw_idle()

    # ----------------------------------------------------------------- export
    def _save_as_stack(self, item):
        """Whether ``item`` should be written through the ImageJ stack writer.

        Auto (default): any item the wavefront loader can read -- multi-frame
        stacks *and* single-frame 2-channel wavefronts -- so both phase and
        amplitude survive the round trip, exactly like Edit > "Apply transform
        to a ImageJ stack". Flat png/jpg/npy items always take the flat path.
        """
        mode = self.save_format.currentData()
        if mode == "flat":
            return False
        is_wf = bool(item.get("is_wavefront", item.get("n_frames", 1) > 1))
        if mode == "stack":
            return is_wf
        return is_wf or item.get("n_frames", 1) > 1

    def _export(self):
        if not self.template_path:
            QMessageBox.warning(self, "Progressive Folder", "Load a template first.")
            return
        if not any(it["matrix"] is not None for it in self.images):
            QMessageBox.warning(self, "Progressive Folder", "No aligned items to export.")
            return
        # Ensure the global (fixed) template is the active output frame -- sliding
        # propagation swaps the aligner template per image; export warps every
        # item with its composed GLOBAL matrix into the first-image frame.
        self.aligner.load_template_from_path(self.template_path)
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            return
        out_dir = Path(folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        items = [it for it in self.images if it["matrix"] is not None]
        progress = QProgressDialog("Exporting aligned items...", "Cancel", 0, len(items), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        matrices = {}
        for n, item in enumerate(items):
            if progress.wasCanceled():
                break
            matrix = self._export_matrix(item)
            self.aligner.current_transform = tf.AffineTransform(matrix=matrix)
            matrices[Path(item["path"]).name] = matrix.tolist()
            if self._save_as_stack(item):
                # Wavefront: warp every phase+amp frame via the existing stack
                # pipeline, which preserves the ImageJ channel layout on write.
                self.aligner.export_stack_to_folder(item["path"], str(out_dir), "_aligned")
            else:
                # Flat image: set the moving image, then reuse save_image_to_folder.
                self.aligner.moving_image = _to_2d(load_imgfile(item["path"])).astype(np.float32)
                self.aligner.moving_image_file = item["path"]
                self.aligner.save_image_to_folder(str(out_dir), "_aligned")
            progress.setValue(n + 1)

        progress.close()
        try:
            with open(out_dir / "matrices.json", "w") as f:
                json.dump(matrices, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to write matrices.json: {e}")
            return
        kind = ("smoothed" if (self.use_smooth.isChecked() and self.smooth_available)
                else "raw")
        QMessageBox.information(
            self, "Progressive Folder",
            f"Exported {len(matrices)} item(s) to {out_dir.name} "
            f"using the {kind} matrices.")

    # ------------------------------------------------------------ persistence
    def _load_matrices(self):
        """Re-import a ``matrices.json`` written by a previous export.

        The file maps ``<file name> -> 3x3 global matrix``. Items are matched by
        file name (the export key), falling back to the stem so a run exported
        with a ``_aligned`` suffix still matches, then -- when the counts agree
        but the names come from another scheme -- to the number parsed out of
        the names, then to list order (see :meth:`_match_matrices_to_items`).
        Matched items become anchors holding that global matrix, so Propagate
        can interpolate from them.

        When no item is loaded yet but every key resolves next to the json (or in
        a folder the user picks), the matched files are loaded as the item list.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Load matrices.json", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("expected a {name: matrix} object")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load matrices: {e}")
            return

        # Accept a full config too (it carries an "images" list instead).
        if "images" in data and isinstance(data.get("images"), list):
            QMessageBox.information(
                self, "Progressive Folder",
                "This file looks like a full progressive config; use 'Load Config'.")
            return

        matrices = {}
        for name, m in data.items():
            try:
                arr = np.array(m, dtype=float)
            except Exception:
                continue
            if arr.shape == (3, 3):
                matrices[str(name)] = arr
        if not matrices:
            QMessageBox.warning(self, "Progressive Folder",
                                "No 3x3 matrices found in that file.")
            return

        if not self.images:
            self._adopt_matrix_files(Path(path), matrices)
            if not self.images:
                return

        pairing, strategy = self._match_matrices_to_items(matrices)
        if pairing is None:
            return

        n_matched, n_missed = 0, []
        for i, item in enumerate(self.images):
            item_name = Path(item["path"]).name
            arr = pairing.get(i)
            if arr is None:
                n_missed.append(item_name)
                continue
            item["matrix"] = arr.tolist()
            item["matrix_smooth"] = None
            self._invalidate_smoothing()
            # Imported matrices are global; the relative transform is only
            # meaningful in sliding mode, so derive it from the reference.
            item["matrix_rel"] = self._decompose_relative(i, arr).tolist()
            item["is_anchor"] = True
            item["status"] = self.STATUS_ANCHOR
            item["corr"] = self._score_item(item, arr)
            n_matched += 1

        self._refresh_table()
        self._refresh_plot()
        self._update_status()
        if 0 <= self.current_index < len(self.images):
            self._update_thumbnail(self.current_index)
        msg = (f"Imported {n_matched} matrix/matrices from {Path(path).name} "
               f"({strategy}).")
        if n_missed:
            msg += f"  {len(n_missed)} item(s) unmatched (e.g. {n_missed[0]})."
        self.status_label.setText(msg)
        QMessageBox.information(self, "Progressive Folder", msg)

    # ------------------------------------------------------- matrix matching
    _NUM_RE = re.compile(r"\d+")

    @classmethod
    def _name_numbers(cls, name):
        """Every integer run in a file stem, as ints (``a_03_c12.tif`` -> [3, 12])."""
        return [int(n) for n in cls._NUM_RE.findall(Path(name).stem)]

    @classmethod
    def _numeric_index(cls, name, field):
        """The ``field``-th number of ``name``, or None when it has no such number.

        ``field`` counts from the *end* when negative, so the common
        ``frame_0012_aligned`` / ``img_12`` pair matches on -1 only if both end
        with their index; positions are tried by the caller.
        """
        nums = cls._name_numbers(name)
        if not nums:
            return None
        try:
            return nums[field]
        except IndexError:
            return None

    def _match_by_number(self, matrices):
        """Pair items to matrix keys by a number parsed out of their names.

        Tries each number position (first, second, ..., last) on both sides and
        keeps the first position that yields a *bijection* over all items: every
        item gets a distinct key. Returns ``{item_index: matrix}`` or None.
        """
        item_names = [Path(it["path"]).name for it in self.images]
        keys = list(matrices)
        # Candidate positions: leading ones, then trailing ones.
        positions = [0, -1, 1, -2]
        for ipos in positions:
            item_nums = [self._numeric_index(n, ipos) for n in item_names]
            if any(v is None for v in item_nums) or len(set(item_nums)) != len(item_nums):
                continue
            for kpos in positions:
                key_nums = [self._numeric_index(k, kpos) for k in keys]
                if any(v is None for v in key_nums):
                    continue
                by_num = {}
                for k, v in zip(keys, key_nums):
                    by_num.setdefault(v, k)
                if len(by_num) != len(keys):
                    continue  # duplicate numbers on the json side
                if not all(v in by_num for v in item_nums):
                    continue
                return {i: matrices[by_num[v]] for i, v in enumerate(item_nums)}, \
                       {i: by_num[v] for i, v in enumerate(item_nums)}
        return None

    def _match_matrices_to_items(self, matrices):
        """Resolve ``{name: matrix}`` against the loaded items.

        Cascade: exact file name -> stem (suffixed exports) -> number parsed
        from the names -> positional order. The last two only apply when the
        counts match, and both ask the user to confirm the pairing first.

        Returns ``({item_index: matrix}, strategy_label)``, or ``(None, "")``
        when the user cancels.
        """
        by_name = dict(matrices)
        by_stem, by_base = {}, {}
        for name, arr in matrices.items():
            stem = Path(name).stem
            by_stem.setdefault(stem, arr)
            # A key exported with a suffix ("a_aligned") also answers to "a".
            if "_" in stem:
                by_base.setdefault(stem.rsplit("_", 1)[0], arr)

        pairing, names = {}, {}
        for i, item in enumerate(self.images):
            item_name = Path(item["path"]).name
            stem = Path(item_name).stem
            for key, arr in ((item_name, by_name.get(item_name)),
                             (stem, by_stem.get(stem)),
                             (stem, by_base.get(stem))):
                if arr is not None:
                    pairing[i] = arr
                    names[i] = key
                    break
        if len(pairing) == len(self.images):
            return pairing, "matched by name"
        if pairing and len(matrices) != len(self.images):
            # Partial name match and no equal-count fallback available: keep it.
            return pairing, "matched by name"

        if len(matrices) != len(self.images):
            if not pairing:
                QMessageBox.warning(
                    self, "Progressive Folder",
                    f"No item name matches the {len(matrices)} key(s) in that file, "
                    f"and the counts differ ({len(matrices)} matrices vs "
                    f"{len(self.images)} items), so they cannot be paired by "
                    "order either.")
                return None, ""
            return pairing, "matched by name"

        # Equal counts: try numbers parsed from the names, then plain order.
        numeric = self._match_by_number(matrices)
        if numeric is not None:
            cand_pairing, cand_names = numeric
            label = "matched by number in the name"
        else:
            keys = natsorted(matrices)
            cand_pairing = {i: matrices[k] for i, k in enumerate(keys)}
            cand_names = {i: k for i, k in enumerate(keys)}
            label = "matched by order"

        if not self._confirm_pairing(cand_names, label):
            return None, ""
        return cand_pairing, label

    def _confirm_pairing(self, names, label):
        """Preview a fallback pairing (json key -> item) and ask to apply it."""
        preview = []
        for i in list(names)[:8]:
            preview.append(f"  {names[i]}  ->  {Path(self.images[i]['path']).name}")
        more = len(names) - len(preview)
        if more > 0:
            preview.append(f"  ... and {more} more")
        text = (f"The file names do not match, but both sides hold "
                f"{len(self.images)} item(s).\n\nProposed pairing ({label}):\n\n"
                + "\n".join(preview) + "\n\nApply it?")
        return QMessageBox.question(
            self, "Load matrices.json", text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    def _adopt_matrix_files(self, json_path, matrices):
        """Populate the item list from the names in a matrices.json.

        Looks for the named files beside the json first; if none are there, asks
        for a folder. Names carrying an export suffix (``*_aligned.tif``) are
        also tried with the suffix stripped, since the matrices refer to the
        original moving items.
        """
        search_dirs = [json_path.parent]
        found = self._resolve_matrix_files(search_dirs, matrices)
        if not found:
            folder = QFileDialog.getExistingDirectory(
                self, "Select the folder holding those moving items")
            if not folder:
                return
            found = self._resolve_matrix_files([Path(folder)], matrices)
        if not found:
            QMessageBox.warning(self, "Progressive Folder",
                                "Could not find any of those files on disk. "
                                "Load the moving folder first, then import the matrices.")
            return
        self._set_images(found)

    @staticmethod
    def _resolve_matrix_files(dirs, matrices):
        """Existing paths for the matrix keys, searched across ``dirs``."""
        found = []
        for name in matrices:
            stem, suffix = Path(name).stem, Path(name).suffix
            for d in dirs:
                cand = d / name
                if cand.is_file():
                    found.append(str(cand))
                    break
                # Try the same stem with any supported extension.
                hit = next((c for ext in IMAGE_EXTS
                            if (c := d / f"{stem}{ext}").is_file()), None)
                if hit is not None:
                    found.append(str(hit))
                    break
                # Try dropping a trailing export suffix ("_aligned").
                if "_" in stem:
                    base = stem.rsplit("_", 1)[0]
                    hit = next((c for ext in ((suffix,) if suffix else ()) + IMAGE_EXTS
                                if (c := d / f"{base}{ext}").is_file()), None)
                    if hit is not None:
                        found.append(str(hit))
                        break
        return found

    def _decompose_relative(self, idx, global_matrix):
        """Relative transform for ``idx`` given its global matrix.

        Inverse of :meth:`_compose_global`: ``rel = inv(global[ref]) @ global``.
        Returns ``global_matrix`` unchanged in fixed mode, for image 0, or when
        the reference has no global matrix yet.
        """
        global_matrix = np.asarray(global_matrix, dtype=float)
        ref_idx = self._reference_index(idx)
        if ref_idx is None:
            return global_matrix
        ref_global = self._global_matrix(ref_idx)
        if ref_global is None:
            return global_matrix
        try:
            return np.linalg.inv(ref_global) @ global_matrix
        except np.linalg.LinAlgError:
            return global_matrix

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Progressive Config", "", "JSON Files (*.json)")
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        try:
            with open(path, "w") as f:
                json.dump({"version": 3, "template_path": self.template_path,
                           "mode": self.mode, "offset": self.offset,
                           "images": self.images, "actions": self.actions}, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save config: {e}")
            return
        self.status_label.setText(f"Saved config to {Path(path).name}")

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Progressive Config", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            images = data["images"]
            assert isinstance(images, list)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config: {e}")
            return
        self.template_path = data.get("template_path", "")
        self.images = [{
            "path": it.get("path", ""), "n_frames": int(it.get("n_frames", 1)),
            "is_wavefront": bool(it.get("is_wavefront", int(it.get("n_frames", 1)) > 1)),
            "matrix": it.get("matrix"), "matrix_rel": it.get("matrix_rel", it.get("matrix")),
            "matrix_smooth": it.get("matrix_smooth"),
            "is_anchor": bool(it.get("is_anchor", False)),
            "status": it.get("status", self.STATUS_PENDING), "corr": it.get("corr"),
        } for it in images]
        # A config saved after smoothing brings its curve back with it.
        self.smooth_available = any(it["matrix_smooth"] is not None for it in self.images)
        self.use_smooth.setEnabled(self.smooth_available)
        if not self.smooth_available:
            self.use_smooth.setChecked(False)
        self.actions = list(data.get("actions", []))
        # Restore reference mode + offset (v1 configs default to fixed).
        self.offset = int(data.get("offset", 1))
        self.offset_spin.setValue(self.offset)
        mode = data.get("mode", self.MODE_FIXED)
        idx = self.mode_combo.findData(mode)
        self.mode_combo.setCurrentIndex(idx if idx >= 0 else 0)  # triggers _on_mode_changed
        self.current_index = -1
        if self.template_path:
            self.template_label.setText(f"Template: {Path(self.template_path).name}")
            self.template_label.setStyleSheet("")
            self.aligner.load_template_from_path(self.template_path)
        self._refresh_table()
        self._refresh_action_list()
        self._refresh_plot()
        self._update_status()
        self.status_label.setText(f"Loaded config from {Path(path).name}")

    # ---------------------------------------------------------------- status
    def _update_status(self):
        n = len(self.images)
        n_anchor = sum(1 for it in self.images if it["is_anchor"])
        n_aligned = sum(1 for it in self.images if it["matrix"] is not None)
        n_low = sum(1 for it in self.images if it["status"] == self.STATUS_LOW)
        cur = f"  ·  current: {self.current_index + 1}/{n}" if self.current_index >= 0 else ""
        if self.smooth_available:
            smooth = ("  ·  smoothed: EXPORTING" if self.use_smooth.isChecked()
                      else "  ·  smoothed: preview only")
        else:
            smooth = ""
        self.status_label.setText(
            f"{n} item(s)  ·  {n_anchor} anchor(s)  ·  {n_aligned} aligned  ·  "
            f"{n_low} low-corr  ·  {len(self.actions)} action(s){smooth}{cur}")


# --------------------------------------------------------------- tiny helpers
def _norm01(arr):
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo + 1e-10)


def _resize_to(arr, shape):
    """Resize ``arr`` to ``shape`` (used only for the thumbnail overlay)."""
    if arr.shape == tuple(shape):
        return arr
    return tf.resize(arr, shape, preserve_range=True).astype(np.float32)
