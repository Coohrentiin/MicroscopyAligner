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
   cross_correlation / reset), like Batch Mode.
5. **Propagate**: for every non-anchor item, set the transform to a prealignment
   matrix interpolated between the two nearest anchors by list index
   (nearest-anchor copy outside the anchor range -- no extrapolation), then
   replay the action sequence on top to refine.
6. Review per-item status + correlation score, fix outliers, then export aligned
   items + per-item matrices.

The prealignment is interpolated on the **center-referenced display params**
(scale / rotation / tx / ty) produced by
:meth:`TransformControls._affine_to_display_params`, which vary smoothly; the
result is turned back into a matrix by
:meth:`TransformControls._params_to_affine`.
"""
import json
from pathlib import Path

import numpy as np
from natsort import natsorted
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


def representative_frame(path):
    """Return ``(repr2d, n_frames)`` for a folder item.

    Tries the wavefront-stack loader first (frame 0, phase channel); falls back
    to :func:`load_imgfile` for plain png/jpg/npy that it cannot read.
    """
    try:
        phase, _amp, n_frames = load_wavefront_tif(path, frame_index=0)
        return phase.astype(np.float32), int(n_frames)
    except Exception:
        return _to_2d(load_imgfile(path)).astype(np.float32), 1


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
    ``self.images``: list of dicts ``{path, n_frames, matrix(list|None),
    matrix_rel(list|None), is_anchor(bool), status, corr}`` in natsorted order.
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

        # auto_keypoints options
        self.ak_use_dialog = QCheckBox("Open dialog")
        self.ak_use_dialog.toggled.connect(self._update_option_visibility)
        ctrl.addWidget(self.ak_use_dialog)
        self.ak_detector = QComboBox(); self.ak_detector.addItems(["AKAZE", "KAZE", "SIFT", "ORB", "BRISK"])
        self.ak_matcher = QComboBox(); self.ak_matcher.addItems(["Brute Force", "FLANN"])
        self.ak_ransac = QDoubleSpinBox(); self.ak_ransac.setRange(0.5, 20.0); self.ak_ransac.setValue(5.0); self.ak_ransac.setPrefix("RANSAC ")
        self.ak_headless_widgets = [self.ak_detector, self.ak_matcher, self.ak_ransac]
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
        run_row.addStretch()
        save_cfg = QPushButton("Save Config"); save_cfg.clicked.connect(self._save_config)
        load_cfg = QPushButton("Load Config"); load_cfg.clicked.connect(self._load_config)
        run_row.addWidget(save_cfg)
        run_row.addWidget(load_cfg)
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
        is_ak = atype == "auto_keypoints"
        self.ak_use_dialog.setVisible(is_ak)
        self._set_visible(self.ak_headless_widgets, is_ak and not self.ak_use_dialog.isChecked())

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
                _repr, n_frames = representative_frame(p)
            except Exception:
                n_frames = 1
            self.images.append({
                "path": p, "n_frames": n_frames, "matrix": None, "matrix_rel": None,
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
        item["status"] = self.STATUS_PENDING
        item["corr"] = None
        self._update_row(row)
        self._refresh_plot()
        self._update_status()

    # ------------------------------------------------------------ action edit
    def _action_summary(self, action):
        t = action.get("type")
        if t == "set_matrix":
            return (f"Set matrix  scale={action.get('scale', 1.0):g}  rot={action.get('rotation', 0.0):g}  "
                    f"tx={action.get('tx', 0.0):g}  ty={action.get('ty', 0.0):g}")
        if t == "auto_keypoints":
            if action.get("dialog", False):
                return "Auto keypoint detection (dialog)"
            return (f"Auto keypoint detection (headless: {action.get('detector', 'AKAZE')} / "
                    f"{action.get('matcher', 'Brute Force')}, RANSAC={action.get('ransac_threshold', 5.0):g})")
        return ACTION_LABELS.get(t, t)

    def _add_action(self):
        atype = self.action_combo.currentData()
        if atype == "set_matrix":
            action = {"type": "set_matrix", "scale": self.scale_spin.value(),
                      "rotation": self.rot_spin.value(), "tx": self.tx_spin.value(),
                      "ty": self.ty_spin.value()}
        elif atype == "auto_keypoints":
            action = {"type": "auto_keypoints", "dialog": self.ak_use_dialog.isChecked()}
            if not action["dialog"]:
                action.update({"detector": self.ak_detector.currentText(),
                               "matcher": self.ak_matcher.currentText(),
                               "ransac_threshold": self.ak_ransac.value()})
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
                    ransac_threshold=action.get("ransac_threshold", 5.0))
        elif t == "cross_correlation":
            a.optimize_phase_correlation()
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
            # 3. compose to global, record + score (vs the global template)
            self._store_result(i, self.aligner.current_transform.params, self.STATUS_PROP)
            if (item["corr"] is not None and not np.isnan(item["corr"]) and item["corr"] < low):
                item["status"] = self.STATUS_LOW
            self._update_row(i)
            progress.setValue(n + 1)

        progress.close()
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

    def _refresh_plot(self):
        self.plot_fig.clear()
        if not self.images:
            self.plot_canvas.draw_idle()
            return
        tc = self.aligner.transform_controls
        idxs, scales, rots, txs, tys, anchor_idx = [], [], [], [], [], []
        for i, item in enumerate(self.images):
            if item["matrix"] is None:
                continue
            p = tc._affine_to_display_params(np.array(item["matrix"], dtype=float))
            idxs.append(i); scales.append(p["scale"]); rots.append(p["rotation"])
            txs.append(p["tx"]); tys.append(p["ty"])
            if item["is_anchor"]:
                anchor_idx.append(i)

        specs = [("scale", scales), ("rotation (deg)", rots), ("tx", txs), ("ty", tys)]
        for k, (title, ys) in enumerate(specs):
            ax = self.plot_fig.add_subplot(2, 2, k + 1)
            ax.plot(idxs, ys, "-o", markersize=3)
            for ai in anchor_idx:
                if ai in idxs:
                    ax.plot(ai, ys[idxs.index(ai)], "s", color="red", markersize=6)
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=6)
        self.plot_fig.tight_layout()
        self.plot_canvas.draw_idle()

    # ----------------------------------------------------------------- export
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
            matrix = np.array(item["matrix"], dtype=float)
            self.aligner.current_transform = tf.AffineTransform(matrix=matrix)
            matrices[Path(item["path"]).name] = matrix.tolist()
            if item["n_frames"] > 1:
                # Stack: warp every phase+amp frame via the existing pipeline.
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
        QMessageBox.information(self, "Progressive Folder",
                               f"Exported {len(matrices)} item(s) to {out_dir.name}.")

    # ------------------------------------------------------------ persistence
    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Progressive Config", "", "JSON Files (*.json)")
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        try:
            with open(path, "w") as f:
                json.dump({"version": 2, "template_path": self.template_path,
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
            "matrix": it.get("matrix"), "matrix_rel": it.get("matrix_rel", it.get("matrix")),
            "is_anchor": bool(it.get("is_anchor", False)),
            "status": it.get("status", self.STATUS_PENDING), "corr": it.get("corr"),
        } for it in images]
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
        self.status_label.setText(
            f"{n} item(s)  ·  {n_anchor} anchor(s)  ·  {n_aligned} aligned  ·  "
            f"{n_low} low-corr  ·  {len(self.actions)} action(s){cur}")


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
