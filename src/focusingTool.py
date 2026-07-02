# Copyright (c) 2025 Corentin Soubeiran
# SPDX-License-Identifier: MIT
"""Wavefront Focusing & Alignment panel.

A separate top-level window that drives the main :class:`ImageAligner` to
register two **wavefronts** acquired with *different objectives* (e.g. template
20x NA0.8, moving 10x NA0.45) that differ in scale, focus and lens distortion.

Workflow (configured on one pair, replayed across many via a shared action
sequence -- like :mod:`batchMode` / :mod:`progressiveFolderAlignment`):

1. Load two wavefronts (``utils_images.load_wavefront_tif``); for multi-T files
   pick a frame; per-WF optics (magnification / NA) + shared wavelength, camera
   pixel size and medium index.
2. Refocus each complex field by the Angular Spectrum Method
   (:func:`optics.propagate_asm`) -- manually with a z slider on a chosen ROI,
   or via autofocus (:func:`optics.autofocus`, Gini / Gouy).
3. Align: scale+translation-only keypoints (no rotation), then a non-centered
   distortion correction (thin-plate spline by default), then subpixel NCC.
4. Focus-consistency check: re-propagate both fields over a +/- range and plot
   a similarity curve so the user can confirm the fields co-focus and map onto
   each other.

The optical math lives in :mod:`optics` and operates on complex fields the
aligner never sees; the linear keypoint / NCC steps run *through* the aligner
(reusing its canvas overlay and ``optimize_phase_correlation``). The phase
image is the representative real image pushed into the aligner; the resulting
global similarity + distortion are applied channel-consistently to the complex
field at export and written with :func:`utils_images.save_stack`.
"""
import json
from pathlib import Path

import numpy as np
from natsort import natsorted
from skimage import transform as tf

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QTableWidget,
    QTableWidgetItem, QListWidget, QComboBox, QDoubleSpinBox, QSpinBox, QLabel,
    QFileDialog, QMessageBox, QHeaderView, QCheckBox, QLineEdit, QSplitter,
    QProgressDialog, QSlider,
)
from PySide6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from utils_images import load_wavefront_tif, save_stack
from keyPointsDetectionAndSelection import detect_keypoint_pairs
import optics as opt

WF_FILTER = "Wavefront TIFF (*.tif *.tiff);;All Files (*)"

ALIGN_CMAP = "turbo"  # high-contrast default for the alignment overlay

# User-editable default action sequence (overrides the built-in when present).
DEFAULT_SEQUENCE_PATH = Path.home() / ".manual_registration_app" / "focusing_default_sequence.json"

# Phase-offset bar range: +/- PHASE_OFFSET_MAX radians over PHASE_OFFSET_STEPS ticks.
PHASE_OFFSET_MAX = 2.0 * np.pi
PHASE_OFFSET_STEPS = 360

ACTION_TYPES = [
    ("select_frame", "Select channel (template / moving)"),
    ("set_optics", "Set optics (mag / NA / wavelength / pixel / n)"),
    ("refocus", "Refocus field (manual z OR autofocus)"),
    ("live_feedback", "Live feedback in main panel (on / off)"),
    ("keypoints_scale_translation", "Keypoints align (scale+translation, no rotation)"),
    ("distortion_correct", "Estimate & apply distortion warp"),
    ("ncc_refine", "Subpixel NCC translation refinement"),
    ("focus_check", "Focus-consistency check (z sweep curve)"),
    ("save", "Save wavefront(s)"),
]
ACTION_LABELS = dict(ACTION_TYPES)

DEFAULT_OPTICS = {
    "mag_tpl": 20.0, "na_tpl": 0.8,
    "mag_mov": 10.0, "na_mov": 0.45,
    "wavelength_nm": 660.0, "camera_pixel_um": 5.86, "n": 1.0,
}


def _norm01(arr):
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo + 1e-10)


def _action_summary(action):
    t = action.get("type")
    if t == "select_frame":
        return f"Select channel  template={action.get('template_frame', 0)}  moving={action.get('moving_frame', 0)}"
    if t == "live_feedback":
        return f"Live feedback {'ON' if action.get('enabled', True) else 'OFF'}"
    if t == "set_optics":
        return (f"Set optics  tpl {action.get('mag_tpl', 20):g}x/NA{action.get('na_tpl', 0.8):g}  "
                f"mov {action.get('mag_mov', 10):g}x/NA{action.get('na_mov', 0.45):g}  "
                f"lam={action.get('wavelength_nm', 660):g}nm  px={action.get('camera_pixel_um', 5.86):g}um")
    if t == "refocus":
        if action.get("mode") == "autofocus":
            return (f"Refocus {action.get('target', 'moving')} (autofocus {action.get('metric', 'gini')}, "
                    f"+/-{action.get('half_range_um', 10):g}um step {action.get('step_nm', 100):g}nm)")
        return f"Refocus {action.get('target', 'moving')} (manual z={action.get('z_um', 0.0):g}um)"
    if t == "keypoints_scale_translation":
        return (f"Keypoints scale+translation ({action.get('detector', 'AKAZE')}/"
                f"{action.get('matcher', 'Brute Force')}, RANSAC={action.get('ransac_threshold', 5.0):g})")
    if t == "distortion_correct":
        return f"Distortion correct (model={action.get('model', 'tps')})"
    if t == "ncc_refine":
        return "Subpixel NCC refinement"
    if t == "focus_check":
        return (f"Focus check (tpl ±{action.get('half_tpl_um', 5):g}/{action.get('step_tpl_nm', 100):g}nm, "
                f"mov ±{action.get('half_mov_um', 5):g}/{action.get('step_mov_nm', 100):g}nm, "
                f"{action.get('metric', 'ncc')}, ROI {action.get('roi_frac', 0.5) * 100:g}%)")
    if t == "save":
        what = action.get("what", "moving")
        chan = "all channels" if action.get("all_channels", True) else f"channel {action.get('channel', 0)}"
        folder = action.get("folder")
        dest = f"-> {folder} (name+{action.get('suffix', '')})" if folder else "(prompt)"
        return f"Save {what} [{chan}] {dest}"
    return ACTION_LABELS.get(t, t)


class FocusingPanel(QWidget):
    """Top-level window driving an :class:`ImageAligner` for wavefront focusing.

    Data model
    ----------
    ``self.rows``: list of per-sample dicts (one template + one moving WF):
    ``{template_path, moving_path, n_frames_tpl, n_frames_mov, template_frame,
    moving_frame, optics(dict|None), z_tpl_um, z_mov_um, global_matrix(list|None),
    distortion({model, src, dst}|None), status, corr}``.
    ``self.actions``: ordered shared action sequence replayed across rows.

    Working state (current row only): ``self._tpl_field`` / ``self._mov_field``
    are the live complex fields; ``self._global_matrix`` / ``self._distortion``
    are the alignment being built.
    """

    STATUS_PENDING = "pending"
    STATUS_DONE = "done"
    STATUS_INHERITED = "inherited"

    def __init__(self, aligner):
        super().__init__()
        self.aligner = aligner
        self.setWindowTitle("Wavefront Focusing & Alignment")
        self.resize(1200, 900)

        self.defaults = dict(DEFAULT_OPTICS)
        self.rows = []
        self.actions = []
        self.current_index = -1

        # Live working state for the current row.
        self._tpl_field = None
        self._mov_field = None
        self._global_matrix = None
        self._distortion = None
        self._roi = None          # (y0, y1, x0, x1) in moving-field pixels
        self._roi_selector = None
        self._refocus_target = "moving"
        self._live_feedback = False   # toggled by live_feedback actions
        self._step_index = 0          # next action to run in step-by-step mode
        self._focus_map_window = None  # shared 2-D focus-map window (focus_check)
        self._blocking_run = False     # True during Run All: pause on map/distortion popups

        self._init_ui()
        self._refresh_table()
        self._update_status()

    # --------------------------------------------------------------------- UI
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_optics_section())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_table_section())
        splitter.addWidget(self._build_roi_section())
        splitter.setSizes([520, 680])
        layout.addWidget(splitter, stretch=1)

        layout.addWidget(self._build_action_section())

        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.status_label)

    def _build_optics_section(self):
        box = QGroupBox("1. Wavefronts + optics")
        v = QVBoxLayout(box)

        load_row = QHBoxLayout()
        load_tpl = QPushButton("Load Template WF...")
        load_tpl.clicked.connect(self._load_template)
        load_tpl.setStyleSheet("QPushButton { background-color: #4CAF50; }")
        load_mov = QPushButton("Load Moving WF...")
        load_mov.clicked.connect(self._load_moving)
        load_mov.setStyleSheet("QPushButton { background-color: #2196F3; }")
        load_row.addWidget(load_tpl)
        load_row.addWidget(load_mov)
        load_row.addStretch()
        v.addLayout(load_row)

        # Per-WF channel selector. Each frame is a wavelength channel; the combo
        # lists wavelengths (from metadata) or "frame N" and stores the frame
        # index as item data. Scrolling the combo cycles channels.
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Template channel:"))
        self.tpl_chan_combo = QComboBox()
        self.tpl_chan_combo.currentIndexChanged.connect(lambda _i: self._on_channel_changed("template"))
        frame_row.addWidget(self.tpl_chan_combo)
        frame_row.addSpacing(20)
        frame_row.addWidget(QLabel("Moving channel:"))
        self.mov_chan_combo = QComboBox()
        self.mov_chan_combo.currentIndexChanged.connect(lambda _i: self._on_channel_changed("moving"))
        frame_row.addWidget(self.mov_chan_combo)
        frame_row.addStretch()
        v.addLayout(frame_row)

        # Optics spinboxes.
        opt_row = QHBoxLayout()
        self.mag_tpl = self._spin(0.1, 200, self.defaults["mag_tpl"], " tpl mag")
        self.na_tpl = self._spin(0.01, 1.6, self.defaults["na_tpl"], " tpl NA", step=0.01, decimals=3)
        self.mag_mov = self._spin(0.1, 200, self.defaults["mag_mov"], " mov mag")
        self.na_mov = self._spin(0.01, 1.6, self.defaults["na_mov"], " mov NA", step=0.01, decimals=3)
        self.wavelength_nm = self._spin(100, 2000, self.defaults["wavelength_nm"], " nm")
        self.camera_pixel_um = self._spin(0.1, 100, self.defaults["camera_pixel_um"], " um px", step=0.01, decimals=3)
        self.n_index = self._spin(1.0, 2.0, self.defaults["n"], " n", step=0.001, decimals=3)
        for w in (self.mag_tpl, self.na_tpl, self.mag_mov, self.na_mov,
                  self.wavelength_nm, self.camera_pixel_um, self.n_index):
            w.valueChanged.connect(self._update_derived_label)
            opt_row.addWidget(w)
        v.addLayout(opt_row)

        self.derived_label = QLabel(); self.derived_label.setStyleSheet("color: gray;")
        v.addWidget(self.derived_label)
        self._update_derived_label()
        return box

    @staticmethod
    def _spin(lo, hi, val, suffix, step=None, decimals=2):
        s = QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(decimals)
        s.setValue(val); s.setSuffix(suffix)
        if step is not None:
            s.setSingleStep(step)
        return s

    def _build_table_section(self):
        box = QGroupBox("2. Samples  (click a row to load it)")
        v = QVBoxLayout(box)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Template", "Moving", "Status", "z tpl(um)", "z mov(um)", "Corr", "Sec."])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.currentCellChanged.connect(self._on_row_changed)
        v.addWidget(self.table)

        btns = QHBoxLayout()
        add_tpl = QPushButton("Browse Templates...")
        add_tpl.clicked.connect(lambda: self._browse_column("template"))
        add_mov = QPushButton("Browse Movings...")
        add_mov.clicked.connect(lambda: self._browse_column("moving"))
        add_row = QPushButton("Add Row")
        add_row.clicked.connect(self._add_row)
        rem_row = QPushButton("Remove Row")
        rem_row.clicked.connect(self._remove_row)
        for b in (add_tpl, add_mov, add_row, rem_row):
            btns.addWidget(b)
        v.addLayout(btns)

        sec = QHBoxLayout()
        sec_mov = QPushButton("Secondary Moving(s)...")
        sec_mov.clicked.connect(lambda: self._browse_secondary("moving"))
        sec_tpl = QPushButton("Secondary Template(s)...")
        sec_tpl.clicked.connect(lambda: self._browse_secondary("template"))
        sec_clear = QPushButton("Clear Secondaries")
        sec_clear.clicked.connect(self._clear_secondaries)
        for b in (sec_mov, sec_tpl, sec_clear):
            sec.addWidget(b)
        sec.addStretch()
        v.addLayout(sec)

        hint = QLabel("Browse fills the column downward from the selected row (natsorted, adding rows). "
                      "Click a row to make it the working sample for refocus / actions. "
                      "Secondary images are NOT displayed; at save they get the same transform "
                      "(moving: propagation+matrix+distortion; template: propagation) as the primary.")
        hint.setStyleSheet("color: gray;"); hint.setWordWrap(True)
        v.addWidget(hint)
        return box

    def _build_roi_section(self):
        box = QGroupBox("Refocus + focus check")
        v = QVBoxLayout(box)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Target:"))
        self.target_combo = QComboBox(); self.target_combo.addItems(["moving", "template"])
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        ctrl.addWidget(self.target_combo)
        ctrl.addWidget(QLabel("z (um):"))
        self.z_spin = QDoubleSpinBox(); self.z_spin.setRange(-500, 500); self.z_spin.setDecimals(3)
        self.z_spin.setSingleStep(0.1); self.z_spin.valueChanged.connect(self._on_z_changed)
        ctrl.addWidget(self.z_spin)
        ctrl.addWidget(QLabel("Show:"))
        self.observable_combo = QComboBox(); self.observable_combo.addItems(["amplitude", "phase"])
        self.observable_combo.currentTextChanged.connect(lambda _t: self._draw_field())
        ctrl.addWidget(self.observable_combo)
        self.metric_combo = QComboBox(); self.metric_combo.addItems(["gini", "gouy"])
        ctrl.addWidget(self.metric_combo)
        autofocus = QPushButton("Autofocus ROI")
        autofocus.clicked.connect(self._autofocus_roi)
        ctrl.addWidget(autofocus)
        reset_zoom = QPushButton("Reset Zoom")
        reset_zoom.clicked.connect(self._reset_zoom)
        ctrl.addWidget(reset_zoom)
        ctrl.addStretch()
        v.addLayout(ctrl)

        # Phase-display controls: a constant offset bar (radians) added to the
        # shown phase, and a checkbox to subtract the unwrapped-phase median.
        phase_row = QHBoxLayout()
        self.phase_median_check = QCheckBox("Subtract phase median")
        self.phase_median_check.toggled.connect(lambda _v: self._refresh_phase_view())
        phase_row.addWidget(self.phase_median_check)
        phase_row.addWidget(QLabel("Phase offset:"))
        # QSlider is integer; map [-PHASE_OFFSET_MAX, +PHASE_OFFSET_MAX] rad over
        # PHASE_OFFSET_STEPS ticks each side.
        self.phase_offset_slider = QSlider(Qt.Horizontal)
        self.phase_offset_slider.setRange(-PHASE_OFFSET_STEPS, PHASE_OFFSET_STEPS)
        self.phase_offset_slider.setValue(0)
        self.phase_offset_slider.valueChanged.connect(self._on_phase_offset_changed)
        phase_row.addWidget(self.phase_offset_slider, stretch=1)
        self.phase_offset_label = QLabel("0.00 rad")
        self.phase_offset_label.setMinimumWidth(80)
        phase_row.addWidget(self.phase_offset_label)
        reset_offset = QPushButton("Reset")
        reset_offset.clicked.connect(lambda: self.phase_offset_slider.setValue(0))
        phase_row.addWidget(reset_offset)
        v.addLayout(phase_row)

        self.field_fig = Figure(figsize=(4, 3), tight_layout=True)
        self.field_canvas = FigureCanvasQTAgg(self.field_fig)
        self.field_ax = self.field_fig.add_subplot(111)
        self.field_ax.set_title("Amplitude (drag to set ROI)", fontsize=8)
        self.field_ax.axis("off")
        # Scroll-to-zoom centered on the cursor.
        self.field_canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        v.addWidget(self.field_canvas, stretch=1)

        self.curve_fig = Figure(figsize=(4, 2), tight_layout=True)
        self.curve_canvas = FigureCanvasQTAgg(self.curve_fig)
        v.addWidget(self.curve_canvas, stretch=1)
        return box

    def _build_action_section(self):
        box = QGroupBox("3. Action sequence  ·  Run  ·  Export  ·  Config")
        v = QVBoxLayout(box)

        self.action_list = QListWidget(); self.action_list.setMaximumHeight(110)
        v.addWidget(self.action_list)

        ctrl = QHBoxLayout()
        self.action_combo = QComboBox()
        for key, label in ACTION_TYPES:
            self.action_combo.addItem(label, key)
        self.action_combo.currentIndexChanged.connect(self._update_option_visibility)
        ctrl.addWidget(self.action_combo)

        # refocus options
        self.ro_mode = QComboBox(); self.ro_mode.addItems(["manual", "autofocus"])
        self.ro_mode.currentIndexChanged.connect(self._update_option_visibility)
        self.ro_target = QComboBox(); self.ro_target.addItems(["moving", "template"])
        self.ro_metric = QComboBox(); self.ro_metric.addItems(["gini", "gouy"])
        self.ro_use_roi = QCheckBox("Focus on ROI")
        self.ro_half = self._spin(0.1, 200, 10.0, " +/-um")
        self.ro_step = self._spin(10, 5000, 100.0, " nm step")
        self.refocus_widgets = [self.ro_mode, self.ro_target, self.ro_metric,
                                self.ro_use_roi, self.ro_half, self.ro_step]
        for w in self.refocus_widgets:
            ctrl.addWidget(w)

        # keypoint / distortion options
        self.ak_detector = QComboBox(); self.ak_detector.addItems(["AKAZE", "KAZE", "SIFT", "ORB", "BRISK"])
        self.ak_matcher = QComboBox(); self.ak_matcher.addItems(["Brute Force", "FLANN"])
        self.ak_ransac = self._spin(0.5, 20.0, 5.0, " RANSAC", step=0.5)
        self.ak_border = QSpinBox(); self.ak_border.setRange(0, 1000); self.ak_border.setValue(0)
        self.ak_border.setPrefix("border "); self.ak_border.setSuffix("px")
        self.ak_border.setToolTip("Crop template/moving borders: ignore keypoints within this margin (#2).")
        self.kp_widgets = [self.ak_detector, self.ak_matcher, self.ak_ransac, self.ak_border]
        for w in self.kp_widgets:
            ctrl.addWidget(w)
        self.dist_model = QComboBox(); self.dist_model.addItems(["tps", "poly", "radial", "spherical", "piecewise"])
        ctrl.addWidget(self.dist_model)

        # focus_check options (separate template / moving ranges)
        self.fc_half_tpl = self._spin(0.1, 200, 5.0, " ±tplum")
        self.fc_step_tpl = self._spin(10, 5000, 100.0, " tpl nm")
        self.fc_half_mov = self._spin(0.1, 200, 5.0, " ±movum")
        self.fc_step_mov = self._spin(10, 5000, 100.0, " mov nm")
        self.fc_metric = QComboBox(); self.fc_metric.addItems(["ncc", "l2", "ssim"])
        self.fc_roi_pct = QSpinBox(); self.fc_roi_pct.setRange(5, 100); self.fc_roi_pct.setValue(25)
        self.fc_roi_pct.setPrefix("ROI "); self.fc_roi_pct.setSuffix("%")
        self.fc_roi_pct.setToolTip("Centered crop used for the focus map; z is applied full-frame.")
        self.fc_widgets = [self.fc_half_tpl, self.fc_step_tpl,
                           self.fc_half_mov, self.fc_step_mov, self.fc_metric, self.fc_roi_pct]
        for w in self.fc_widgets:
            ctrl.addWidget(w)

        # live_feedback option
        self.lf_enabled = QComboBox(); self.lf_enabled.addItems(["on", "off"])
        self.lf_widgets = [self.lf_enabled]
        ctrl.addWidget(self.lf_enabled)

        # save options
        self.save_what = QComboBox(); self.save_what.addItems(["moving", "template", "both"])
        self.save_channels = QComboBox(); self.save_channels.addItems(["all channels", "single channel"])
        self.save_channels.currentIndexChanged.connect(self._update_option_visibility)
        self.save_channel_idx = QSpinBox(); self.save_channel_idx.setRange(0, 999); self.save_channel_idx.setPrefix("ch ")
        self.save_folder = QLineEdit(); self.save_folder.setPlaceholderText("output folder (blank = prompt)")
        self.save_suffix = QLineEdit(); self.save_suffix.setPlaceholderText("suffix"); self.save_suffix.setMaximumWidth(120)
        self.save_browse = QPushButton("Browse..."); self.save_browse.clicked.connect(self._browse_save_folder)
        self.save_widgets = [self.save_what, self.save_channels, self.save_channel_idx,
                             self.save_folder, self.save_browse, self.save_suffix]
        for w in self.save_widgets:
            ctrl.addWidget(w)

        add_act = QPushButton("Add Action"); add_act.clicked.connect(self._add_action)
        ctrl.addWidget(add_act)
        v.addLayout(ctrl)

        edit_row = QHBoxLayout()
        rem_act = QPushButton("Remove Action"); rem_act.clicked.connect(self._remove_action)
        up = QPushButton("Move Up"); up.clicked.connect(lambda: self._move_action(-1))
        down = QPushButton("Move Down"); down.clicked.connect(lambda: self._move_action(1))
        step = QPushButton("Step (Next Action)"); step.clicked.connect(self._step_action)
        step.setStyleSheet("QPushButton { background-color: #3F51B5; }")
        reset_steps = QPushButton("Reset Steps"); reset_steps.clicked.connect(self._reset_steps)
        run_all = QPushButton("Run All (current row)"); run_all.clicked.connect(self._run_current_row)
        run_all.setStyleSheet("QPushButton { background-color: #9C27B0; }")
        run_rows = QPushButton("Run All Rows"); run_rows.clicked.connect(self._run_all_rows)
        run_rows.setStyleSheet("QPushButton { background-color: #9C27B0; }")
        for b in (rem_act, up, down, step, reset_steps, run_all, run_rows):
            edit_row.addWidget(b)
        edit_row.addStretch()
        default_seq = QPushButton("Load Default Sequence")
        default_seq.setStyleSheet("QPushButton { background-color: #4CAF50; }")
        default_seq.clicked.connect(self._load_default_sequence)
        edit_row.addWidget(default_seq)
        save_def = QPushButton("Save as Default"); save_def.clicked.connect(self._save_as_default_sequence)
        save_def.setToolTip("Persist the current (edited) action list as the default sequence.")
        edit_row.addWidget(save_def)
        reset_def = QPushButton("Reset Default"); reset_def.clicked.connect(self._reset_default_sequence)
        reset_def.setToolTip("Forget the user default; 'Load Default' reverts to the built-in.")
        edit_row.addWidget(reset_def)
        save_cfg = QPushButton("Save Config"); save_cfg.clicked.connect(self._save_config)
        load_cfg = QPushButton("Load Config"); load_cfg.clicked.connect(self._load_config)
        edit_row.addWidget(save_cfg); edit_row.addWidget(load_cfg)
        v.addLayout(edit_row)

        self._update_option_visibility()
        return box

    # ----------------------------------------------------- option visibility
    def _set_visible(self, widgets, visible):
        for w in widgets:
            w.setVisible(visible)

    def _update_option_visibility(self):
        atype = self.action_combo.currentData()
        self._set_visible(self.refocus_widgets, atype == "refocus")
        if atype == "refocus":
            af = self.ro_mode.currentText() == "autofocus"
            self.ro_metric.setVisible(af); self.ro_half.setVisible(af); self.ro_step.setVisible(af)
            self.ro_use_roi.setVisible(af)  # ROI only used by autofocus scoring
        self._set_visible(self.kp_widgets, atype in ("keypoints_scale_translation", "distortion_correct"))
        self.dist_model.setVisible(atype == "distortion_correct")
        self._set_visible(self.fc_widgets, atype == "focus_check")
        self._set_visible(self.lf_widgets, atype == "live_feedback")
        self._set_visible(self.save_widgets, atype == "save")
        if atype == "save":
            # The single-channel index spin is only relevant for "single channel".
            self.save_channel_idx.setVisible(self.save_channels.currentText() == "single channel")

    def _browse_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.save_folder.setText(folder)

    # ----------------------------------------------------------- optics utils
    def _current_optics(self):
        return {
            "mag_tpl": self.mag_tpl.value(), "na_tpl": self.na_tpl.value(),
            "mag_mov": self.mag_mov.value(), "na_mov": self.na_mov.value(),
            "wavelength_nm": self.wavelength_nm.value(),
            "camera_pixel_um": self.camera_pixel_um.value(), "n": self.n_index.value(),
        }

    def _apply_optics_to_widgets(self, o):
        self.mag_tpl.setValue(o.get("mag_tpl", 20)); self.na_tpl.setValue(o.get("na_tpl", 0.8))
        self.mag_mov.setValue(o.get("mag_mov", 10)); self.na_mov.setValue(o.get("na_mov", 0.45))
        self.wavelength_nm.setValue(o.get("wavelength_nm", 660))
        self.camera_pixel_um.setValue(o.get("camera_pixel_um", 5.86))
        self.n_index.setValue(o.get("n", 1.0))

    def _pixel_sizes(self):
        """Return ``(px_tpl_m, px_mov_m, wavelength_m, n)`` from the widgets."""
        o = self._current_optics()
        cam = o["camera_pixel_um"] * 1e-6
        return (opt.sample_pixel_size(cam, o["mag_tpl"]),
                opt.sample_pixel_size(cam, o["mag_mov"]),
                o["wavelength_nm"] * 1e-9, o["n"])

    def _update_derived_label(self):
        px_tpl, px_mov, _lam, _n = self._pixel_sizes()
        o = self._current_optics()
        scale = opt.magnification_scale(o["mag_tpl"], o["mag_mov"])
        self.derived_label.setText(
            f"Sample pixel: template {px_tpl * 1e9:.1f} nm  ·  moving {px_mov * 1e9:.1f} nm  ·  "
            f"predicted moving display scale = {scale:.4f}")

    # ------------------------------------------------------------ load actions
    def _wf_meta(self, path):
        """Return ``(n_frames, wavelengths_nm or None)`` for a WF file."""
        _phase, _amp, n_frames = load_wavefront_tif(path, frame_index=0)
        return int(n_frames), opt.read_wavefront_wavelengths(path)

    def _load_template(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Template WF", "", WF_FILTER)
        if path:
            self._assign_to_current("template", path)

    def _load_moving(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Moving WF", "", WF_FILTER)
        if path:
            self._assign_to_current("moving", path)

    def _assign_to_current(self, which, path):
        if not self.rows:
            self._add_row()
        row = max(0, self.table.currentRow())
        try:
            n_frames, wl = self._wf_meta(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot read {Path(path).name}: {e}")
            return
        item = self.rows[row]
        if which == "template":
            item["template_path"] = path; item["n_frames_tpl"] = n_frames
            item["wl_tpl"] = wl; item["template_frame"] = 0
        else:
            item["moving_path"] = path; item["n_frames_mov"] = n_frames
            item["wl_mov"] = wl; item["moving_frame"] = 0
        self._refresh_table()
        self.table.setCurrentCell(row, 0)
        self._send_to_window(row)

    def _browse_column(self, which):
        paths, _ = QFileDialog.getOpenFileNames(self, f"Select {which} WF files", "", WF_FILTER)
        if not paths:
            return
        paths = natsorted(paths)
        start = max(0, self.table.currentRow())
        for i, path in enumerate(paths):
            r = start + i
            if r >= len(self.rows):
                self._add_row(refresh=False)
            try:
                n_frames, wl = self._wf_meta(path)
            except Exception:
                n_frames, wl = 1, None
            if which == "template":
                self.rows[r]["template_path"] = path; self.rows[r]["n_frames_tpl"] = n_frames
                self.rows[r]["wl_tpl"] = wl; self.rows[r]["template_frame"] = 0
            else:
                self.rows[r]["moving_path"] = path; self.rows[r]["n_frames_mov"] = n_frames
                self.rows[r]["wl_mov"] = wl; self.rows[r]["moving_frame"] = 0
        self._refresh_table()

    def _browse_secondary(self, which):
        """Attach secondary image(s); multi-select natsort-fills DOWN across rows
        (one per row, adding rows as needed), like 'Browse Movings...' (#3)."""
        paths, _ = QFileDialog.getOpenFileNames(self, f"Select secondary {which} WF files", "", WF_FILTER)
        if not paths:
            return
        paths = natsorted(paths)
        key = "secondary_moving_paths" if which == "moving" else "secondary_template_paths"
        start = max(0, self.table.currentRow())
        for i, path in enumerate(paths):
            r = start + i
            if r >= len(self.rows):
                self._add_row(refresh=False)
            self.rows[r].setdefault(key, [])
            self.rows[r][key].append(path)
        self._refresh_table()
        self.status_label.setText(
            f"Added {len(paths)} secondary {which} image(s) down from row {start + 1}.")

    def _clear_secondaries(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.rows):
            self.rows[row]["secondary_moving_paths"] = []
            self.rows[row]["secondary_template_paths"] = []
            self._refresh_table()

    def _new_row(self):
        return {
            "template_path": "", "moving_path": "",
            "n_frames_tpl": 1, "n_frames_mov": 1,
            "wl_tpl": None, "wl_mov": None,  # per-channel wavelengths (nm) or None
            "template_frame": 0, "moving_frame": 0,
            "optics": None, "z_tpl_um": 0.0, "z_mov_um": 0.0,
            "global_matrix": None, "distortion": None,
            # Secondary images (#1): never displayed; at save each undergoes the
            # SAME transform as its primary (moving: z+matrix+distortion;
            # template: z only).
            "secondary_moving_paths": [], "secondary_template_paths": [],
            "status": self.STATUS_PENDING, "corr": None,
        }

    def _add_row(self, refresh=True):
        self.rows.append(self._new_row())
        if refresh:
            self._refresh_table()
            self.table.setCurrentCell(len(self.rows) - 1, 0)

    def _remove_row(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.rows):
            del self.rows[row]
            self._refresh_table()
            self._update_status()

    # ------------------------------------------------------ table <-> model
    def _refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.rows))
        for r, item in enumerate(self.rows):
            tpl = Path(item["template_path"]).name if item["template_path"] else ""
            if item["n_frames_tpl"] > 1:
                tpl += f"  [×{item['n_frames_tpl']}]"
            mov = Path(item["moving_path"]).name if item["moving_path"] else ""
            if item["n_frames_mov"] > 1:
                mov += f"  [×{item['n_frames_mov']}]"
            n_sec_m = len(item.get("secondary_moving_paths", []))
            n_sec_t = len(item.get("secondary_template_paths", []))
            sec_txt = "" if (n_sec_m + n_sec_t) == 0 else f"m{n_sec_m}/t{n_sec_t}"
            cells = [
                tpl, mov, item["status"],
                f"{item['z_tpl_um']:.3f}", f"{item['z_mov_um']:.3f}",
                "" if item["corr"] is None else f"{item['corr']:.3f}",
                sec_txt,
            ]
            for c, text in enumerate(cells):
                cell = QTableWidgetItem(text)
                if c == 0 and item["template_path"]:
                    cell.setToolTip(item["template_path"])
                if c == 1 and item["moving_path"]:
                    cell.setToolTip(item["moving_path"])
                if c == 6 and (n_sec_m or n_sec_t):
                    cell.setToolTip("Secondary moving:\n" + "\n".join(item.get("secondary_moving_paths", []))
                                    + "\nSecondary template:\n" + "\n".join(item.get("secondary_template_paths", [])))
                self.table.setItem(r, c, cell)
        self.table.blockSignals(False)

    def _update_row(self, r):
        item = self.rows[r]
        self.table.item(r, 2).setText(item["status"])
        self.table.item(r, 3).setText(f"{item['z_tpl_um']:.3f}")
        self.table.item(r, 4).setText(f"{item['z_mov_um']:.3f}")
        self.table.item(r, 5).setText("" if item["corr"] is None else f"{item['corr']:.3f}")

    def _on_row_changed(self, row, col, prev_row, prev_col):
        if 0 <= row < len(self.rows):
            self._send_to_window(row)

    # ------------------------------------------------------------ window drive
    def _inherit_from_previous(self, row):
        """Seed an UN-aligned row from the previous row's transform + distances so
        the user can save as-is when no realignment is needed (#2).

        Only applies when this row has no alignment yet (no matrix/distortion,
        z's at 0, still pending) and the previous row DOES have one. Copies
        z_tpl_um / z_mov_um / global_matrix / distortion / optics.
        """
        if row <= 0 or row >= len(self.rows):
            return
        it = self.rows[row]
        already = (it.get("global_matrix") is not None or it.get("distortion") is not None
                   or it.get("z_tpl_um") or it.get("z_mov_um")
                   or it.get("status") == self.STATUS_DONE)
        if already:
            return
        prev = self.rows[row - 1]
        if prev.get("global_matrix") is None and prev.get("distortion") is None \
                and not prev.get("z_tpl_um") and not prev.get("z_mov_um"):
            return  # previous row has nothing to inherit
        it["z_tpl_um"] = prev.get("z_tpl_um", 0.0)
        it["z_mov_um"] = prev.get("z_mov_um", 0.0)
        it["global_matrix"] = ([list(map(float, r)) for r in prev["global_matrix"]]
                               if prev.get("global_matrix") is not None else None)
        it["distortion"] = (dict(prev["distortion"]) if prev.get("distortion") else None)
        if prev.get("optics"):
            it["optics"] = dict(prev["optics"])
        it["status"] = self.STATUS_INHERITED
        self.status_label.setText(f"Row {row + 1}: inherited transform + focus from row {row}.")

    def _send_to_window(self, row):
        """Make ``row`` the working sample: rebuild its complex fields and show."""
        if not (0 <= row < len(self.rows)):
            return
        self._inherit_from_previous(row)
        item = self.rows[row]
        self.current_index = row
        if item.get("optics"):
            self._apply_optics_to_widgets(item["optics"])
        # Set the high-contrast overlay colormap ONCE when a row is loaded; do NOT
        # re-force it on every step afterwards (#2 -- respect the user's choice).
        self.aligner.template_color.setCurrentText(ALIGN_CMAP)
        self.aligner.moving_color.setCurrentText(ALIGN_CMAP)
        self._populate_channel_combo("template", item)
        self._populate_channel_combo("moving", item)
        self._global_matrix = (np.array(item["global_matrix"], dtype=float)
                               if item["global_matrix"] is not None else None)
        self._distortion = self._distortion_from_dict(item.get("distortion"))
        self.z_spin.setValue(item["z_mov_um"] if self._refocus_target == "moving" else item["z_tpl_um"])
        self._step_index = 0  # step-by-step restarts for the newly selected row
        self._rebuild_fields(row)
        self._update_row(row)  # reflect any inherited z / status in the table
        self._update_status()

    @staticmethod
    def _channel_labels(n_frames, wl):
        """Item labels for a WF's channel combo: wavelengths when known, else 'frame N'."""
        if wl and len(wl) == n_frames:
            return [f"{wl[i]:g} nm" for i in range(n_frames)]
        return [f"frame {i}" for i in range(max(1, n_frames))]

    def _populate_channel_combo(self, which, item):
        combo = self.tpl_chan_combo if which == "template" else self.mov_chan_combo
        n_frames = item["n_frames_tpl"] if which == "template" else item["n_frames_mov"]
        wl = item["wl_tpl"] if which == "template" else item["wl_mov"]
        frame = item["template_frame"] if which == "template" else item["moving_frame"]
        combo.blockSignals(True)
        combo.clear()
        for i, label in enumerate(self._channel_labels(n_frames, wl)):
            combo.addItem(label, i)
        combo.setCurrentIndex(min(frame, combo.count() - 1))
        # A single-channel file leaves nothing to scroll; disable to signal that.
        combo.setEnabled(combo.count() > 1)
        combo.blockSignals(False)

    def _on_channel_changed(self, which):
        if not (0 <= self.current_index < len(self.rows)):
            return
        item = self.rows[self.current_index]
        combo = self.tpl_chan_combo if which == "template" else self.mov_chan_combo
        frame = combo.currentData()
        if frame is None:
            return
        if which == "template":
            item["template_frame"] = int(frame)
        else:
            item["moving_frame"] = int(frame)
        self._rebuild_fields(self.current_index)

    def _rebuild_fields(self, row):
        """Load the chosen frames, build complex fields, pre-scale moving, redraw."""
        item = self.rows[row]
        self._tpl_field = self._mov_field = None
        if item["template_path"]:
            p, a, _ = load_wavefront_tif(item["template_path"], item["template_frame"])
            self._tpl_field = opt.field_from_phase_amp(p, a)
        if item["moving_path"]:
            p, a, _ = load_wavefront_tif(item["moving_path"], item["moving_frame"])
            field = opt.field_from_phase_amp(p, a)
            self._mov_field = self._prescale_moving(field)
        self._draw_field()

    def _prescale_moving(self, field):
        """Resize the moving field to the template *sampling* (no crop/pad).

        Uses :func:`optics.magnification_scale` (template_mag / moving_mag, so a
        20x/10x pair enlarges the moving field 2x); real/imag resized separately
        to keep the complex field consistent.  Cropping/fitting to the template
        *shape* is deferred to :meth:`_fit_to_template` and applied **after**
        propagation, so refocus never sees zero-padded edges (which would ring
        across the FOV in the phase). No-op when no template is loaded.
        """
        if self._tpl_field is None:
            return field
        o = self._current_optics()
        scale = opt.magnification_scale(o["mag_tpl"], o["mag_mov"])
        if abs(scale - 1.0) > 1e-6:
            re = tf.rescale(field.real, scale, order=1, preserve_range=True)
            im = tf.rescale(field.imag, scale, order=1, preserve_range=True)
            field = re + 1j * im
        return field

    @staticmethod
    def _fit_offset(field_shape, shape):
        """``(x0, y0)`` centered-crop offset :meth:`_fit_to_template` applies.

        A moving pixel ``(x, y)`` in the cropped (template-shaped) frame is
        ``(x + x0, y + y0)`` in the full ``field``. Used at save to warp the
        UN-cropped moving with the matrix that was estimated on the crop, so the
        output borders fill from the larger FOV instead of going black (#2).
        """
        h, w = field_shape[:2]
        H, W = shape
        return (max(0, (w - W) // 2), max(0, (h - H) // 2))

    @staticmethod
    def _fit_to_template(field, shape):
        """Center-crop ``field`` to ``shape`` (the common case for scale>1).

        If the field is smaller on an axis it is padded by **edge replication**
        (not zeros) so the complex field has no hard amplitude/phase
        discontinuity at the border. Applied only AFTER propagation.
        """
        if field is None or shape is None or field.shape == tuple(shape):
            return field
        H, W = shape
        h, w = field.shape
        sy = max(0, (h - H) // 2); sx = max(0, (w - W) // 2)
        field = field[sy:sy + min(H, h), sx:sx + min(W, w)]
        h, w = field.shape
        if (h, w) != (H, W):
            pad = ((max(0, (H - h) // 2), max(0, H - h - (H - h) // 2)),
                   (max(0, (W - w) // 2), max(0, W - w - (W - w) // 2)))
            field = np.pad(field, pad, mode="edge")[:H, :W]
        return field

    # ------------------------------------------------------------ refocus view
    def _active_field(self):
        return self._tpl_field if self._refocus_target == "template" else self._mov_field

    def _on_target_changed(self, text):
        self._refocus_target = text
        if 0 <= self.current_index < len(self.rows):
            item = self.rows[self.current_index]
            self.z_spin.blockSignals(True)
            self.z_spin.setValue(item["z_tpl_um"] if text == "template" else item["z_mov_um"])
            self.z_spin.blockSignals(False)
        self._draw_field()

    def _on_z_changed(self, value):
        if 0 <= self.current_index < len(self.rows):
            item = self.rows[self.current_index]
            if self._refocus_target == "template":
                item["z_tpl_um"] = value
            else:
                item["z_mov_um"] = value
            self._update_row(self.current_index)
        self._draw_field()

    def _propagated_active(self, fit=True):
        """The active field propagated to its current z (for display/scoring).

        Propagation runs on the native (uncropped) field; the moving field is
        fitted to the template shape only *after* propagation (``fit=True``) so
        no zero-padding edge is ever propagated.
        """
        field = self._active_field()
        if field is None:
            return None
        item = self.rows[self.current_index] if 0 <= self.current_index < len(self.rows) else None
        z_um = (item["z_tpl_um"] if self._refocus_target == "template" else item["z_mov_um"]) if item else 0.0
        px_tpl, px_mov, lam, n = self._pixel_sizes()
        px = px_tpl if self._refocus_target == "template" else px_mov
        prop = opt.propagate_asm(field, z_um * 1e-6, lam, px, n=n)
        if fit and self._refocus_target == "moving" and self._tpl_field is not None:
            prop = self._fit_to_template(prop, self._tpl_field.shape)
        return prop

    def _draw_field(self, keep_zoom=True):
        # Preserve the current zoom (axis limits) across redraws unless asked not to.
        prev_xlim = self.field_ax.get_xlim() if keep_zoom else None
        prev_ylim = self.field_ax.get_ylim() if keep_zoom else None
        self.field_ax.clear()
        observable = self.observable_combo.currentText()
        self.field_ax.set_title(f"{observable.capitalize()} (drag to set ROI, scroll to zoom)", fontsize=8)
        self.field_ax.axis("off")
        prop = self._propagated_active()
        if prop is not None:
            if observable == "phase":
                phase, vmin, vmax = self._displayed_phase(prop)
                self.field_ax.imshow(phase, cmap=ALIGN_CMAP, vmin=vmin, vmax=vmax)
            else:
                self.field_ax.imshow(np.abs(prop), cmap="gray")
            if prev_xlim is not None and prev_xlim != (0.0, 1.0):
                self.field_ax.set_xlim(prev_xlim)
                self.field_ax.set_ylim(prev_ylim)
            # (Re)create the ROI selector bound to the fresh axes; ax.clear()
            # detaches the previous one's artists. useblit=False so the box
            # survives the frequent imshow redraws. Restore the current ROI box.
            self._roi_selector = RectangleSelector(
                self.field_ax, self._on_roi_select, useblit=False, button=[1],
                minspanx=3, minspany=3, spancoords="pixels", interactive=True)
            if self._roi is not None:
                y0, y1, x0, x1 = self._roi
                try:
                    self._roi_selector.extents = (x0, x1, y0, y1)
                except Exception:
                    pass
        self.field_canvas.draw_idle()

    def _phase_offset_rad(self):
        """Current phase-offset bar value in radians."""
        return self.phase_offset_slider.value() / PHASE_OFFSET_STEPS * PHASE_OFFSET_MAX

    def _subtract_median(self):
        """Whether unwrapped phase should have its median removed."""
        return self.phase_median_check.isChecked()

    def _field_to_observable(self, field):
        """2-D real view of a complex field per the panel's observable combo
        (amplitude or unwrapped phase). Used for the main-window live preview so
        the user can switch the channel they inspect (#2)."""
        if self.observable_combo.currentText() == "phase":
            return opt.unwrapped_phase(field, subtract_median=self._subtract_median())
        return np.abs(field).astype(np.float32)

    def _displayed_phase(self, field):
        """Phase for display with FIXED colour limits so offset/median are visible.

        Returns ``(phase, vmin, vmax)``. The colour limits are fixed to the
        *base* (no-offset, no-median) unwrapped phase range; the offset then
        slides the data against those limits (the colours visibly shift /
        wrap), and median-subtraction re-centres the data within them. Without
        fixed limits ``imshow`` autoscales and a constant offset/median would
        look identical.
        """
        base = opt.unwrapped_phase(field)
        vmin, vmax = float(base.min()), float(base.max())
        if vmax - vmin < 1e-6:           # flat field -> give the bar something to move against
            vmin, vmax = vmin - np.pi, vmax + np.pi
        phase = base
        if self.phase_median_check.isChecked():
            phase = phase - float(np.median(phase))
        phase = phase + self._phase_offset_rad()
        return phase, vmin, vmax

    def _on_phase_offset_changed(self, _value):
        self.phase_offset_label.setText(f"{self._phase_offset_rad():.2f} rad")
        self._refresh_phase_view()

    def _refresh_phase_view(self):
        """Redraw immediately for phase-display controls (offset / median).

        The offset & median only affect the phase view, so switch to it if
        needed, then force an immediate repaint (``draw`` not ``draw_idle``) so
        dragging the slider updates the screen in real time.
        """
        if self.observable_combo.currentText() != "phase":
            self.observable_combo.setCurrentText("phase")  # triggers _draw_field
        else:
            self._draw_field()
        self.field_canvas.draw()

    def _on_scroll_zoom(self, event):
        """Scroll-wheel zoom centered on the cursor over the field axes."""
        if event.inaxes is not self.field_ax or event.xdata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        x0, x1 = self.field_ax.get_xlim()
        y0, y1 = self.field_ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        self.field_ax.set_xlim(cx + (x0 - cx) * scale, cx + (x1 - cx) * scale)
        self.field_ax.set_ylim(cy + (y0 - cy) * scale, cy + (y1 - cy) * scale)
        self.field_canvas.draw_idle()

    def _reset_zoom(self):
        self.field_ax.autoscale(); self._draw_field(keep_zoom=False)

    def _on_roi_select(self, eclick, erelease):
        # Guard against drags that end outside the axes (xdata/ydata None) which
        # would otherwise crash inside matplotlib's callback and silently lose
        # the selection.
        coords = (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)
        if any(c is None or not np.isfinite(c) for c in coords):
            return
        x0, x1 = sorted((int(round(eclick.xdata)), int(round(erelease.xdata))))
        y0, y1 = sorted((int(round(eclick.ydata)), int(round(erelease.ydata))))
        # Clamp to the displayed image bounds.
        shape = self._displayed_shape()
        if shape is not None:
            h, w = shape
            x0, x1 = max(0, x0), min(w, x1)
            y0, y1 = max(0, y0), min(h, y1)
        if x1 - x0 >= 3 and y1 - y0 >= 3:
            self._roi = (y0, y1, x0, x1)
            self.status_label.setText(f"ROI = y[{y0}:{y1}] x[{x0}:{x1}]  (drag again to change)")

    def _displayed_shape(self):
        """(H, W) of the image currently shown in the field axes, or None."""
        prop = self._propagated_active()
        return None if prop is None else prop.shape

    def _autofocus_roi(self):
        field = self._active_field()
        if field is None:
            QMessageBox.warning(self, "Focusing", "Load the field first.")
            return
        px_tpl, px_mov, lam, n = self._pixel_sizes()
        px = px_tpl if self._refocus_target == "template" else px_mov
        half = self.ro_half.value() * 1e-6 if self.ro_half.isVisible() else 10e-6
        step = self.ro_step.value() * 1e-9 if self.ro_step.isVisible() else 100e-9
        z_values = opt.focus_z_values(half, step)
        # Score on the same field the user sees: for the moving target that is
        # the template-shape-fitted field, so the ROI lines up. Gini=amplitude,
        # Gouy=phase; both honour the selected ROI of the FOV.
        best_z, scores, zs = opt.autofocus(
            field, z_values, lam, px, n=n,
            method=self.metric_combo.currentText(), roi=self._roi,
            post=self._autofocus_post())
        self.z_spin.setValue(best_z * 1e6)  # triggers redraw + stores z
        self._plot_curve(zs * 1e6, scores, f"Autofocus ({self.metric_combo.currentText()})", best_z * 1e6)

    def _autofocus_post(self):
        """Post-propagation transform used during autofocus scoring.

        Moving target -> fit to the template shape (so the on-screen ROI lines
        up with the scored pixels); template target -> identity.
        """
        if self._refocus_target == "moving" and self._tpl_field is not None:
            shape = self._tpl_field.shape
            return lambda f: self._fit_to_template(f, shape)
        return None

    def _plot_curve(self, x_um, y, title, mark_um=None):
        self.curve_fig.clear()
        ax = self.curve_fig.add_subplot(111)
        ax.plot(x_um, y, "-o", markersize=3)
        if mark_um is not None:
            ax.axvline(mark_um, color="red", lw=1)
        ax.set_title(title, fontsize=8); ax.set_xlabel("z (um)", fontsize=7)
        ax.tick_params(labelsize=6)
        self.curve_fig.tight_layout()
        self.curve_canvas.draw_idle()

    def _plot_focus_map(self, za_um, zb_um, map2d, z_tpl_cur_um, z_mov_cur_um,
                        peak_za_um, peak_zb_um, metric):
        """2-D focus-consistency map (x = template z, y = moving z).

        Cross-hairs mark the CURRENT focus (z_tpl, z_mov); a marker shows the
        co-focus peak.
        """
        self.curve_fig.clear()
        ax = self.curve_fig.add_subplot(111)
        extent = [za_um[0], za_um[-1], zb_um[-1], zb_um[0]]  # origin upper
        im = ax.imshow(map2d, aspect="auto", cmap="viridis", extent=extent, origin="upper")
        # Current-focus cross-hairs.
        ax.axvline(z_tpl_cur_um, color="white", lw=1, ls="--")
        ax.axhline(z_mov_cur_um, color="white", lw=1, ls="--")
        # Co-focus peak.
        ax.plot(peak_za_um, peak_zb_um, "r+", markersize=10, markeredgewidth=2)
        ax.set_title(f"Focus consistency map ({metric})", fontsize=8)
        ax.set_xlabel("template z (um)", fontsize=7)
        ax.set_ylabel("moving z (um)", fontsize=7)
        ax.tick_params(labelsize=6)
        self.curve_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
        self.curve_fig.tight_layout()
        self.curve_canvas.draw_idle()

    # ------------------------------------------------------------ action edit
    def _add_action(self):
        atype = self.action_combo.currentData()
        if atype == "select_frame":
            t_idx = self.tpl_chan_combo.currentData()
            m_idx = self.mov_chan_combo.currentData()
            action = {"type": atype,
                      "template_frame": int(t_idx) if t_idx is not None else 0,
                      "moving_frame": int(m_idx) if m_idx is not None else 0}
        elif atype == "set_optics":
            action = {"type": atype, **self._current_optics()}
        elif atype == "refocus":
            action = {"type": atype, "target": self.ro_target.currentText(),
                      "mode": self.ro_mode.currentText()}
            if action["mode"] == "autofocus":
                action.update({"metric": self.ro_metric.currentText(),
                               "half_range_um": self.ro_half.value(), "step_nm": self.ro_step.value(),
                               "use_roi": self.ro_use_roi.isChecked()})
                if self.ro_use_roi.isChecked() and self._roi is not None:
                    action["roi"] = list(self._roi)
            else:
                action["z_um"] = self.z_spin.value()
        elif atype in ("keypoints_scale_translation", "distortion_correct"):
            action = {"type": atype, "detector": self.ak_detector.currentText(),
                      "matcher": self.ak_matcher.currentText(),
                      "ransac_threshold": self.ak_ransac.value(),
                      "border_crop": self.ak_border.value()}
            if atype == "distortion_correct":
                action["model"] = self.dist_model.currentText()
        elif atype == "focus_check":
            action = {"type": atype, "metric": self.fc_metric.currentText(),
                      "half_tpl_um": self.fc_half_tpl.value(), "step_tpl_nm": self.fc_step_tpl.value(),
                      "half_mov_um": self.fc_half_mov.value(), "step_mov_nm": self.fc_step_mov.value(),
                      "roi_frac": self.fc_roi_pct.value() / 100.0}
            if self._roi is not None:
                action["roi"] = list(self._roi)
        elif atype == "live_feedback":
            action = {"type": atype, "enabled": self.lf_enabled.currentText() == "on"}
        elif atype == "save":
            action = {"type": atype, "what": self.save_what.currentText(),
                      "all_channels": self.save_channels.currentText() == "all channels",
                      "channel": self.save_channel_idx.value()}
            folder = self.save_folder.text().strip()
            if folder:
                action["folder"] = folder
                action["suffix"] = self.save_suffix.text().strip()
        else:
            action = {"type": atype}
        self.actions.append(action)
        self.action_list.addItem(_action_summary(action))

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
            self.action_list.addItem(_action_summary(a))

    @staticmethod
    def _builtin_default_sequence():
        """The built-in standard focus+align pipeline (used when no user default
        file exists / on reset). Edit + 'Save as Default' overrides this."""
        kp = lambda: {"type": "keypoints_scale_translation", "detector": "AKAZE",
                      "matcher": "Brute Force", "ransac_threshold": 5.0, "border_crop": 0}
        return [
            {"type": "set_optics", **DEFAULT_OPTICS},
            kp(),                              # round 1 (rotation blocked)
            kp(),                              # round 2 (rotation blocked)
            {"type": "refocus", "target": "template", "mode": "manual", "z_um": 0.0},
            {"type": "refocus", "target": "moving", "mode": "manual", "z_um": 0.0},
            kp(),                              # rotation-blocked keypoints
            {"type": "focus_check", "metric": "ncc", "roi_frac": 0.5,  # central ROI
             "half_tpl_um": 2.0, "step_tpl_nm": 200.0,    # 20x template
             "half_mov_um": 20.0, "step_mov_nm": 1000.0},  # 10x moving
            kp(),                              # rotation-blocked keypoints
            {"type": "save", "what": "both", "all_channels": True,
             "channel": 0, "suffix": "_aligned"},  # folder prompted at save
        ]

    def _load_default_sequence(self):
        """Load the default focus+align pipeline -- the user's saved default
        (``DEFAULT_SEQUENCE_PATH``) if present, else the built-in (#1)."""
        actions = None
        src = "built-in"
        if DEFAULT_SEQUENCE_PATH.exists():
            try:
                actions = json.loads(DEFAULT_SEQUENCE_PATH.read_text())
                assert isinstance(actions, list)
                src = "user default"
            except Exception:
                actions = None
        if actions is None:
            actions = self._builtin_default_sequence()
        self.actions = [dict(a) for a in actions]
        self._refresh_action_list()
        self._step_index = 0
        self.observable_combo.setCurrentText("phase")  # refocus is on phase
        self.status_label.setText(
            f"Loaded default sequence ({len(self.actions)} actions, {src}). Edit it and "
            "'Save as Default' to persist; step through it and use the 2-D map's 'Use optimal'.")

    def _save_as_default_sequence(self):
        """Persist the CURRENT action list as the user default (#1)."""
        if not self.actions:
            QMessageBox.information(self, "Default Sequence", "No actions to save as default.")
            return
        try:
            DEFAULT_SEQUENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            DEFAULT_SEQUENCE_PATH.write_text(json.dumps(self.actions, indent=2))
        except Exception as e:
            QMessageBox.critical(self, "Default Sequence", f"Failed to save default: {e}")
            return
        self.status_label.setText(
            f"Saved current {len(self.actions)} action(s) as the default sequence.")

    def _reset_default_sequence(self):
        """Delete the user default so 'Load Default' uses the built-in again."""
        try:
            if DEFAULT_SEQUENCE_PATH.exists():
                DEFAULT_SEQUENCE_PATH.unlink()
        except Exception as e:
            QMessageBox.critical(self, "Default Sequence", f"Failed to reset: {e}")
            return
        self.status_label.setText("Reset to the built-in default sequence.")

    # --------------------------------------------------------- action execute
    def _push_phase_to_aligner(self):
        """Push the current propagated phase images into the aligner (Turbo cmap)."""
        if self._tpl_field is None or self._mov_field is None:
            return False
        item = self.rows[self.current_index]
        px_tpl, px_mov, lam, n = self._pixel_sizes()
        tpl_prop = opt.propagate_asm(self._tpl_field, item["z_tpl_um"] * 1e-6, lam, px_tpl, n=n)
        mov_prop = opt.propagate_asm(self._mov_field, item["z_mov_um"] * 1e-6, lam, px_mov, n=n)
        # Fit moving to the template shape AFTER propagation (never propagate a pad).
        mov_prop = self._fit_to_template(mov_prop, self._tpl_field.shape)
        sub = self._subtract_median()
        tpl_phase = opt.unwrapped_phase(tpl_prop, subtract_median=sub)
        mov_phase = opt.unwrapped_phase(mov_prop, subtract_median=sub)
        # Colormap is set once at row load (_send_to_window); don't re-force it (#2).
        self.aligner.set_template_array(tpl_phase, item["template_path"])
        # Seed the aligner transform from the in-progress global matrix. When there
        # is none yet (e.g. only a refocus happened), reset to IDENTITY so the next
        # step (e.g. ncc_refine) starts from the freshly-refocused moving rather
        # than stale slider/transform values from a previous step or row (#1).
        gm = self._global_matrix if self._global_matrix is not None else np.eye(3)
        self.aligner.current_transform = tf.AffineTransform(matrix=gm)
        self.aligner.transform_controls.set_values_from_transform(gm)
        self.aligner.set_moving_array(mov_phase, item["moving_path"])
        # Also hand the COMPLEX propagated fields to the aligner so the main
        # window's "Observable" combo works while running a sequence (#1).
        self._sync_aligner_fields(tpl_prop, mov_prop, item)
        return True

    def _sync_aligner_fields(self, tpl_complex, mov_complex, item):
        """Expose the panel's propagated complex fields on the aligner so the
        main window's Observable selector can re-derive amplitude/phase/real/
        imag at the SAME focus (z already applied here -> aligner z=0)."""
        a = self.aligner
        a.template_field = np.asarray(tpl_complex)
        a.template_field_file = item["template_path"]
        a.template_z_um = 0.0
        a.moving_field = np.asarray(mov_complex)
        a.moving_field_file = item["moving_path"]
        a.moving_z_um = 0.0
        # Keep optics in sync so any aligner-side propagation uses the same params.
        a.propagation_optics.update(self._current_optics())
        # _push_phase_to_aligner displays the unwrapped PHASE, so start the
        # aligner's observable selector on "phase" (no recompute) -- this keeps
        # the combo consistent with what's shown and makes the first toggle work.
        a.template_observable = a.moving_observable = "phase"
        if hasattr(a, "observable_combo"):
            a.observable_combo.blockSignals(True)
            a.observable_combo.setCurrentText("phase")
            a.observable_combo.blockSignals(False)

    def _read_back_matrix(self, item):
        """Store the aligner's current transform as this row's global matrix.

        Tolerates the aligner producing no transform (e.g. no keypoint pairs
        detected): the existing matrix is left untouched.
        """
        if self.aligner.current_transform is not None:
            self._global_matrix = np.array(self.aligner.current_transform.params, dtype=float)
            item["global_matrix"] = self._global_matrix.tolist()

    def _execute_action(self, action):
        t = action.get("type")
        item = self.rows[self.current_index]
        if t == "select_frame":
            item["template_frame"] = int(action.get("template_frame", 0))
            item["moving_frame"] = int(action.get("moving_frame", 0))
            self._populate_channel_combo("template", item)
            self._populate_channel_combo("moving", item)
            self._rebuild_fields(self.current_index)
        elif t == "set_optics":
            o = {k: action[k] for k in DEFAULT_OPTICS if k in action}
            item["optics"] = {**self._current_optics(), **o}
            self._apply_optics_to_widgets(item["optics"])
            self._rebuild_fields(self.current_index)
        elif t == "refocus":
            self._exec_refocus(action, item)
        elif t == "live_feedback":
            self._live_feedback = bool(action.get("enabled", True))
        elif t == "keypoints_scale_translation":
            if self._push_phase_to_aligner():
                self.aligner.auto_keypoints_scale_translation_headless(
                    detector=action.get("detector", "AKAZE"),
                    matcher=action.get("matcher", "Brute Force"),
                    ransac_threshold=action.get("ransac_threshold", 5.0),
                    border_crop=action.get("border_crop", 0))
                self._read_back_matrix(item)
        elif t == "distortion_correct":
            self._exec_distortion(action, item)
        elif t == "ncc_refine":
            if self._push_phase_to_aligner():
                self.aligner.optimize_phase_correlation()
                self._read_back_matrix(item)
        elif t == "focus_check":
            self._exec_focus_check(action, item)
        elif t == "save":
            self._exec_save(action, item)
        item["status"] = self.STATUS_DONE
        self._update_row(self.current_index)
        # Item 3: while live feedback is on, push the current template & moving
        # phase into the main panel and repaint so the user sees the effect.
        if self._live_feedback and t != "live_feedback":
            self._refresh_main_panel(item)

    def _refresh_main_panel(self, item):
        """Show the current template & moving phase (with the in-progress
        transform) in the main ImageAligner window for visual checking."""
        if self._push_phase_to_aligner():
            self.aligner.update_display()
            self.aligner.raise_()
            self.aligner.activateWindow()

    def _exec_refocus(self, action, item):
        target = action.get("target", "moving")
        field = self._tpl_field if target == "template" else self._mov_field
        if field is None:
            return
        px_tpl, px_mov, lam, n = self._pixel_sizes()
        px = px_tpl if target == "template" else px_mov
        # ROI only when the action opted in (item 1 checkbox); else whole FOV.
        if action.get("use_roi"):
            roi = tuple(action["roi"]) if action.get("roi") else self._roi
        else:
            roi = None
        if action.get("mode") == "autofocus":
            half = action.get("half_range_um", 10.0) * 1e-6
            step = action.get("step_nm", 100.0) * 1e-9
            # Moving target: score on the template-shape-fitted field so the ROI
            # of the FOV matches the display. gini=amplitude, gouy=phase.
            post = (lambda f: self._fit_to_template(f, self._tpl_field.shape)) \
                if target == "moving" and self._tpl_field is not None else None
            best_z, scores, zs = opt.autofocus(
                field, opt.focus_z_values(half, step), lam, px, n=n,
                method=action.get("metric", "gini"), roi=roi, post=post)
            z_um = best_z * 1e6
            self._plot_curve(zs * 1e6, scores, f"Autofocus {target} ({action.get('metric', 'gini')})", z_um)
            if target == "template":
                item["z_tpl_um"] = z_um
            else:
                item["z_mov_um"] = z_um
            if target == self._refocus_target:
                self.z_spin.blockSignals(True); self.z_spin.setValue(z_um); self.z_spin.blockSignals(False)
                self._draw_field()
        else:
            # Manual: open the SAME Change-Distance interface as the main window's
            # method (live slider + 2-D map), pre-targeted to this action's target.
            self._open_distance_dialog(initial_target=target)

    def _open_distance_dialog(self, initial_target="moving"):
        """Open the main-window DistanceDialog driven by a panel-backed
        controller, so manual refocus uses the exact same interface."""
        from gui import DistanceDialog
        if self._tpl_field is None and self._mov_field is None:
            return
        controller = _PanelDistanceController(self, initial_target=initial_target)
        dlg = DistanceDialog(controller)
        # Pre-select the requested target if available.
        idx = dlg.target.findText(initial_target)
        if idx >= 0:
            dlg.target.setCurrentIndex(idx)
        dlg.exec()

    def _exec_distortion(self, action, item):
        if self._tpl_field is None or self._mov_field is None:
            return
        px_tpl, px_mov, lam, n = self._pixel_sizes()
        sub = self._subtract_median()
        tpl_prop = opt.propagate_asm(self._tpl_field, item["z_tpl_um"] * 1e-6, lam, px_tpl, n=n)
        mov_prop = self._fit_to_template(
            opt.propagate_asm(self._mov_field, item["z_mov_um"] * 1e-6, lam, px_mov, n=n),
            self._tpl_field.shape)
        tpl_phase = opt.unwrapped_phase(tpl_prop, subtract_median=sub)
        mov_phase = opt.unwrapped_phase(mov_prop, subtract_median=sub)
        # Detect on the globally-aligned moving phase so distortion is residual.
        moving_for_detect = mov_phase
        if self._global_matrix is not None:
            warped = tf.warp(mov_phase, tf.AffineTransform(matrix=self._global_matrix).inverse,
                             output_shape=tpl_phase.shape, preserve_range=True).astype(np.float32)
            moving_for_detect = warped
        pairs = detect_keypoint_pairs(
            tpl_phase, moving_for_detect, detector=action.get("detector", "AKAZE"),
            matcher=action.get("matcher", "Brute Force"),
            ransac_threshold=action.get("ransac_threshold", 5.0),
            border_crop=action.get("border_crop", 0))
        if len(pairs) < 3:
            self.status_label.setText(f"Distortion: only {len(pairs)} pairs (need >=3) -- skipped.")
            return
        model = action.get("model", "tps")
        prev_distortion = self._distortion
        prev_meta = item.get("distortion")
        try:
            new_distortion = opt.estimate_distortion(pairs, tpl_phase.shape, model=model)
        except Exception as e:
            self.status_label.setText(f"Distortion estimate failed: {e}")
            return
        self._distortion = new_distortion
        item["distortion"] = {
            "model": model,
            "src": [list(p[0]) for p in pairs],
            "dst": [list(p[1]) for p in pairs],
        }
        self.status_label.setText(f"Distortion ({model}) from {len(pairs)} pairs.")
        # Fit-quality popup: grid + keypoints (#2a), NCC before/after (#2c), Cancel (#2d).
        try:
            from gui import show_distortion_fit_quality

            def _ncc(img_a, img_b):
                a = np.asarray(img_a, np.float64).ravel(); b = np.asarray(img_b, np.float64).ravel()
                m = (a != 0) & (b != 0)
                if m.sum() < 16 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
                    return float("nan")
                return float(np.corrcoef(a[m], b[m])[0, 1])

            # moving_for_detect is the globally-aligned moving phase (template frame);
            # 'after' applies the residual distortion to it.
            corr_before = _ncc(tpl_phase, moving_for_detect)
            corrected = tf.warp(moving_for_detect, new_distortion,
                                output_shape=tpl_phase.shape, preserve_range=True)
            corr_after = _ncc(tpl_phase, corrected)

            def _cancel():
                self._distortion = prev_distortion
                item["distortion"] = prev_meta
                self.status_label.setText("Distortion correction cancelled.")

            params = {"detector": action.get("detector", "AKAZE"),
                      "matcher": action.get("matcher", "Brute Force"),
                      "ransac_threshold": action.get("ransac_threshold", 5.0),
                      "border_crop": action.get("border_crop", 0)}
            show_distortion_fit_quality(self, self._distortion, pairs, tpl_phase.shape, model,
                                        corr_before=corr_before, corr_after=corr_after,
                                        on_cancel=_cancel, params=params,
                                        modal=self._blocking_run)
        except Exception as e:
            print(f"Distortion fit-quality popup failed: {e}")

    def _distortion_from_dict(self, d):
        if not d:
            return None
        pairs = [((s[0], s[1]), (t[0], t[1])) for s, t in zip(d["src"], d["dst"])]
        # template_shape only used by piecewise; pass a generic shape.
        shape = (1, 1)
        try:
            return opt.estimate_distortion(pairs, shape, model=d.get("model", "tps"))
        except Exception:
            return None

    def _exec_focus_check(self, action, item):
        if self._tpl_field is None or self._mov_field is None:
            return
        from batchMode import FocusMapWindow
        px_tpl, px_mov, lam, n = self._pixel_sizes()
        tpl_shape = self._tpl_field.shape
        roi = tuple(action["roi"]) if action.get("roi") else self._roi

        # Central-ROI fraction: run the map on a centered crop for speed; the
        # chosen z is applied full-frame on save (#1). Default 25%.
        roi_frac = float(action.get("roi_frac", 0.5))
        tpl_in = opt.center_crop(self._tpl_field, roi_frac)
        mov_in = opt.center_crop(self._mov_field, roi_frac)

        # Alignment-aware: bring each propagated moving frame into the template
        # frame via the global matrix + distortion (or just fit when none). The
        # overlap + 16px border crop inside the map handles edge artifacts. On a
        # centered crop the full-frame linear matrix doesn't apply, so just fit.
        gm = self._global_matrix if self._global_matrix is not None else np.eye(3)
        distortion = self._distortion
        if roi_frac < 1.0:
            crop_shape = tpl_in.shape
            align_b = lambda field: self._fit_to_template(field, crop_shape)
        else:
            def align_b(field):
                if self._global_matrix is not None or distortion is not None:
                    return opt.warp_field_with_distortion(field, gm, distortion, tpl_shape)
                return self._fit_to_template(field, tpl_shape)

        # Separate template / moving ranges, each centred on its current focus.
        z_tpl = item["z_tpl_um"] * 1e-6 + opt.focus_z_values(
            action.get("half_tpl_um", 5.0) * 1e-6, action.get("step_tpl_nm", 100.0) * 1e-9)
        z_mov = item["z_mov_um"] * 1e-6 + opt.focus_z_values(
            action.get("half_mov_um", 5.0) * 1e-6, action.get("step_mov_nm", 100.0) * 1e-9)
        metric = action.get("metric", "ncc")

        win = self._ensure_focus_map_window(FocusMapWindow, item)
        win.show(); win.raise_(); win.activateWindow()

        # When cropping, the rectangular ROI no longer maps onto the crop coords.
        crop_roi = roi if roi_frac >= 1.0 else None
        za, zb, map2d = opt.focus_consistency_map(
            tpl_in, mov_in, z_tpl, z_mov, lam, lam, px_tpl, px_mov,
            n=n, metric=metric, roi=crop_roi, align_b=align_b, border=16,
            progress=win.set_progress)

        ref_col = int(np.argmin(np.abs(za - item["z_tpl_um"] * 1e-6)))
        ref_row = int(np.argmin(np.abs(zb - item["z_mov_um"] * 1e-6)))
        win.set_map(za * 1e6, zb * 1e6, map2d, metric, ref_col=ref_col, ref_row=ref_row)
        if np.any(np.isfinite(map2d)):
            item["corr"] = float(np.nanmax(map2d))
        # Context for click-overlay + "Use optimal". The click overlays the FULL
        # frame, so it must use a full-shape align_b (warp into tpl_shape), NOT
        # the crop-shape align_b used for the (ROI) map -- otherwise the moving
        # appears tiny in a corner.
        def overlay_align_b(field):
            # The global matrix was estimated on the template-shape-FITTED moving
            # (see _push_phase_to_aligner), so fit FIRST, then warp -- otherwise
            # the matrix maps fit-frame coords against a larger prescaled field
            # and the translation lands wrong / looks unapplied (#1 regression).
            field = self._fit_to_template(field, tpl_shape)
            if self._global_matrix is not None or distortion is not None:
                return opt.warp_field_with_distortion(field, gm, distortion, tpl_shape)
            return field
        self._fc_ctx = {"lam": lam, "n": n, "px_tpl": px_tpl, "px_mov": px_mov,
                        "align_b": overlay_align_b, "item": item}
        # During a Run, pause here until the user picks an optimum ('Use optimal')
        # or closes the map window, before the sequence continues.
        if self._blocking_run:
            self.status_label.setText("Focus map: pick an optimum ('Use optimal') or close to continue…")
            win.wait_until_closed()

    def _ensure_focus_map_window(self, FocusMapWindow, item):
        """Create/reuse the shared 2-D focus-map window for focus_check."""
        def use_optimal(z_tpl_um, z_mov_um):
            item["z_tpl_um"] = float(z_tpl_um)
            item["z_mov_um"] = float(z_mov_um)
            self._update_row(self.current_index)
            self.status_label.setText(
                f"Focus: z_tpl={z_tpl_um:.2f} µm, z_mov={z_mov_um:.2f} µm (applied).")
            self._draw_field()

        def on_cell(z_tpl_um, z_mov_um):
            ctx = getattr(self, "_fc_ctx", None)
            if not ctx:
                return
            # APPLY the clicked point to the row so it's what save/next steps use
            # (the last-clicked z is the chosen focus, not just a preview).
            item["z_tpl_um"] = float(z_tpl_um)
            item["z_mov_um"] = float(z_mov_um)
            self._update_row(self.current_index)
            tpl = opt.propagate_asm(self._tpl_field, z_tpl_um * 1e-6, ctx["lam"], ctx["px_tpl"], n=ctx["n"])
            mov = ctx["align_b"](opt.propagate_asm(self._mov_field, z_mov_um * 1e-6, ctx["lam"], ctx["px_mov"], n=ctx["n"]))
            a = self.aligner
            row = self.rows[self.current_index]
            # Render the panel's chosen observable (e.g. PHASE) directly so the
            # click shows the right channel immediately -- no stale-toggle needed
            # (bug #1). Pure display: doesn't mutate the panel's complex fields.
            a.template_image = self._field_to_observable(tpl); a.template_image_file = row["template_path"]
            a.transform_controls.set_template_shape(a.template_image.shape)
            a.moving_image = self._field_to_observable(mov); a.transformed_image = a.moving_image
            a.moving_image_file = row["moving_path"]
            a.onViewModeChanged("overlay"); a.update_display()
            self.status_label.setText(
                f"Focus (clicked): z_tpl={z_tpl_um:.2f} µm, z_mov={z_mov_um:.2f} µm — "
                f"click 'Use optimal' for the auto optimum, or Close to keep this.")

        if self._focus_map_window is None:
            self._focus_map_window = FocusMapWindow(on_use_optimal=use_optimal, on_cell=on_cell)
        else:
            self._focus_map_window._on_use_optimal = use_optimal
            self._focus_map_window._on_cell = on_cell
        return self._focus_map_window

    def _exec_save(self, action, item):
        """Save template and/or moving wavefront(s), all or one channel.

        ``what`` in {"moving", "template", "both"}; ``all_channels`` saves every
        wavelength channel, else only ``channel``. The moving field is warped by
        the global matrix + distortion into the template frame and refocused by
        ``z_mov``; the template is saved in its own frame (refocused by
        ``z_tpl``, no alignment warp -- it is the reference).
        """
        targets = {"moving": ["moving"], "template": ["template"],
                   "both": ["template", "moving"]}.get(action.get("what", "moving"), ["moving"])
        targets = [w for w in targets
                   if (item["moving_path"] if w == "moving" else item["template_path"])]
        if not targets:
            return
        folder = action.get("folder")
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
            if not folder:
                return
        out_dir = Path(folder); out_dir.mkdir(parents=True, exist_ok=True)
        suffix = action.get("suffix", "_focused")
        saved = []
        for which in targets:
            out_path = self._save_target(which, item, action, out_dir, suffix)
            if out_path:
                saved.append(Path(out_path).name)
            # Secondary images of this target: same transform, never displayed (#1).
            sec_key = "secondary_moving_paths" if which == "moving" else "secondary_template_paths"
            for sec_path in item.get(sec_key, []):
                sp = self._save_target(which, item, action, out_dir, suffix, path_override=sec_path)
                if sp:
                    saved.append(Path(sp).name)
        if saved:
            self.status_label.setText(f"Saved {', '.join(saved)}")

    def _warp_moving_for_save(self, field, z_um, lam, px, n, out_shape):
        """Refocus + align a moving field for export, FILLING borders (#2).

        Scale to template sampling, refocus on the native grid, then warp the
        **un-cropped** field into the template-shaped output. The alignment
        matrix was estimated on the centered crop, so the warp uses
        ``input_offset = crop offset`` to sample the larger FOV -- output borders
        fill from the moving periphery instead of going black."""
        field = self._prescale_moving(field)
        field = opt.propagate_asm(field, z_um * 1e-6, lam, px, n=n)
        if out_shape is None:
            return field
        offset = self._fit_offset(field.shape, out_shape)
        gm = self._global_matrix if self._global_matrix is not None else np.eye(3)
        return opt.warp_field_with_distortion(
            field, gm, self._distortion, out_shape, input_offset=offset)

    def _save_target(self, which, item, action, out_dir, suffix, path_override=None):
        """Warp + save one target ('template' or 'moving'); return output path.

        ``path_override`` saves a SECONDARY image of that target with the SAME
        transform (#1); its frame count is read from the file (it may differ
        from the primary)."""
        path = path_override or (item["moving_path"] if which == "moving" else item["template_path"])
        if not path:
            return None
        if path_override:
            try:
                _p, _a, n_frames = load_wavefront_tif(path, frame_index=0)
            except Exception as e:
                self.status_label.setText(f"Secondary {Path(path).name} skipped: {e}")
                return None
        else:
            n_frames = item["n_frames_mov"] if which == "moving" else item["n_frames_tpl"]
        z_um = item["z_mov_um"] if which == "moving" else item["z_tpl_um"]
        px_tpl, px_mov, lam, n = self._pixel_sizes()
        px = px_mov if which == "moving" else px_tpl
        out_shape = self._tpl_field.shape if self._tpl_field is not None else None

        if action.get("all_channels", True):
            channels = list(range(max(1, n_frames)))
        else:
            channels = [min(int(action.get("channel", 0)), max(0, n_frames - 1))]

        frames = []
        for fi in channels:
            p, a, _ = load_wavefront_tif(path, fi)
            field = opt.field_from_phase_amp(p, a)
            if which == "moving":
                field = self._warp_moving_for_save(field, z_um, lam, px, n, out_shape)
            else:
                # Template: refocus only (it is the reference frame).
                field = opt.propagate_asm(field, z_um * 1e-6, lam, px, n=n)
            phase, amp = opt.phase_amp_from_field(field, unwrap=True,
                                                  subtract_median=self._subtract_median())
            frames.append(np.stack([phase, amp], axis=0))  # (2[phase,amp], H, W)
        stack = np.moveaxis(np.stack(frames, axis=0), 1, -1)  # (N, H, W, C) for save_stack

        all_wl = opt.read_wavefront_wavelengths(path)
        if all_wl and len(all_wl) >= len(channels):
            wavelengths = tuple(all_wl[i] for i in channels) if not action.get("all_channels", True) else tuple(all_wl)
        else:
            wavelengths = (self.wavelength_nm.value(),)
        chan_tag = "" if action.get("all_channels", True) else f"_ch{channels[0]}"
        out_path = str(out_dir / f"{Path(path).stem}{suffix}{chan_tag}.tif")
        metas = {"optics": self._current_optics(), "target": which, "z_um": z_um,
                 "global_matrix": (self._global_matrix.tolist() if which == "moving"
                                   and self._global_matrix is not None else None),
                 "distortion_model": (item["distortion"] or {}).get("model") if which == "moving" else None}
        save_stack(out_path, stack, wavelengths=wavelengths, source_path=path, metas=metas)
        return out_path

    # ------------------------------------------------------------------ runners
    def _step_action(self):
        """Run the CURRENTLY HIGHLIGHTED action of the current row, then advance
        the highlight to the next one.

        The highlighted row in the action list drives the step (the user can
        click any action to jump there); if nothing is selected it falls back to
        ``_step_index``. After running, the highlight (and ``_step_index``) move
        to the following action.
        """
        if not (0 <= self.current_index < len(self.rows)):
            QMessageBox.warning(self, "Focusing", "Select a sample row first.")
            return
        if not self.actions:
            QMessageBox.information(self, "Focusing", "No actions in the sequence.")
            return
        idx = self.action_list.currentRow()
        if not (0 <= idx < len(self.actions)):
            idx = self._step_index
        if idx >= len(self.actions):
            QMessageBox.information(self, "Focusing", "All actions done. Use 'Reset Steps' to start over.")
            return
        self.action_list.setCurrentRow(idx)
        self._execute_action(self.actions[idx])
        # Advance the highlight to the next action.
        self._step_index = idx + 1
        self.action_list.setCurrentRow(min(self._step_index, len(self.actions) - 1))
        self._update_status()

    def _reset_steps(self):
        """Rewind the step-by-step cursor / highlight to the first action."""
        self._step_index = 0
        self.action_list.setCurrentRow(0 if self.actions else -1)
        self._update_status()

    def _run_current_row(self):
        if not (0 <= self.current_index < len(self.rows)):
            QMessageBox.warning(self, "Focusing", "Select a sample row first.")
            return
        if not self.actions:
            QMessageBox.information(self, "Focusing", "No actions in the sequence.")
            return
        # Blocking run: pause on the 2-D map / distortion popups until the user
        # acts (Use optimal / Close / Cancel) before the next step.
        self._blocking_run = True
        try:
            for action in self.actions:
                self._execute_action(action)
        finally:
            self._blocking_run = False
        self._step_index = len(self.actions)
        self._update_status()

    def _run_all_rows(self):
        if not self.actions:
            QMessageBox.information(self, "Focusing", "No actions in the sequence.")
            return
        rows = [r for r in range(len(self.rows)) if self.rows[r]["moving_path"]]
        progress = QProgressDialog("Running focusing pipeline...", "Cancel", 0, len(rows), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        self._blocking_run = True
        try:
            for n, r in enumerate(rows):
                if progress.wasCanceled():
                    break
                self.table.setCurrentCell(r, 0)  # _send_to_window rebuilds working state
                self._send_to_window(r)
                for action in self.actions:
                    self._execute_action(action)
                progress.setValue(n + 1)  # last step done -> advance to next row
        finally:
            self._blocking_run = False
        progress.close()
        self._update_status()

    # ------------------------------------------------------------ persistence
    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Focusing Config", "", "JSON Files (*.json)")
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        try:
            with open(path, "w") as f:
                json.dump({"version": 1, "tool": "focusing",
                           "defaults": self._current_optics(),
                           "rows": self.rows, "actions": self.actions}, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save config: {e}")
            return
        self.status_label.setText(f"Saved config to {Path(path).name}")

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Focusing Config", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            rows = data["rows"]
            assert isinstance(rows, list)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config: {e}")
            return
        self.defaults = {**DEFAULT_OPTICS, **data.get("defaults", {})}
        self._apply_optics_to_widgets(self.defaults)
        self.rows = []
        for it in rows:
            row = self._new_row()
            row.update({k: it.get(k, row[k]) for k in row})
            self.rows.append(row)
        self.actions = list(data.get("actions", []))
        self.current_index = -1
        self._refresh_table()
        self._refresh_action_list()
        self._update_status()
        self.status_label.setText(f"Loaded config from {Path(path).name}")

    # ---------------------------------------------------------------- status
    def _update_status(self):
        n = len(self.rows)
        n_done = sum(1 for it in self.rows if it["status"] == self.STATUS_DONE)
        cur = f"  ·  current: {self.current_index + 1}/{n}" if self.current_index >= 0 else ""
        step = (f"  ·  step {min(self._step_index + 1, len(self.actions))}/{len(self.actions)}"
                if self.actions else "")
        lf = "  ·  live feedback ON" if self._live_feedback else ""
        self.status_label.setText(
            f"{n} sample(s)  ·  {n_done} done  ·  {len(self.actions)} action(s){step}{cur}{lf}")


class _PanelDistanceController:
    """Adapts :class:`FocusingPanel` (current row) to the ``DistanceDialog``
    controller API, so the manual-refocus step reuses the SAME dialog as the
    main window's Change-Distance method.

    Note: the panel's ``_mov_field`` is already pre-scaled to the TEMPLATE
    sampling, so propagation for BOTH targets uses the template pixel size, and
    ``align_b`` only fits + warps (no magnification rescale).
    """

    def __init__(self, panel, initial_target="moving"):
        self.p = panel
        self.parent_widget = panel

    def targets(self):
        out = []
        if self.p._mov_field is not None:
            out.append("moving")
        if self.p._tpl_field is not None:
            out.append("template")
        return out

    def optics(self):
        return self.p._current_optics()

    def magnification(self, target):
        o = self.p._current_optics()
        return o["mag_tpl"] if target == "template" else o["mag_mov"]

    def set_optics(self, target, mag, wl, px, n):
        # Write back into the panel's optics widgets.
        o = self.p._current_optics()
        o["mag_tpl" if target == "template" else "mag_mov"] = mag
        o.update({"wavelength_nm": wl, "camera_pixel_um": px, "n": n})
        self.p._apply_optics_to_widgets(o)

    def _item(self):
        i = self.p.current_index
        return self.p.rows[i] if 0 <= i < len(self.p.rows) else None

    def current_z(self, target):
        it = self._item()
        if it is None:
            return 0.0
        return it["z_tpl_um"] if target == "template" else it["z_mov_um"]

    def set_distance(self, target, z_um, roi_frac=1.0):
        it = self._item()
        if it is None:
            return
        if target == "template":
            it["z_tpl_um"] = float(z_um)
        else:
            it["z_mov_um"] = float(z_um)
        self.p._update_row(self.p.current_index)
        # Reflect on the panel's own refocus view.
        if target == self.p._refocus_target:
            self.p.z_spin.blockSignals(True); self.p.z_spin.setValue(float(z_um)); self.p.z_spin.blockSignals(False)
            self.p._draw_field()
        # Live feedback in the MAIN window while sliding -- on the central ROI
        # crop for speed (#1/#3). PRESERVE the user's channel/colormap/opacity
        # (#2); favored opacity is applied only on target switch.
        self._render_main(active_target=target, set_opacity=False, roi_frac=roi_frac)

    def _render_main(self, active_target, set_opacity=True, roi_frac=1.0):
        """Render the current row's propagated template+moving in the main window.

        ``set_opacity`` (target-switch only) favors the adjusted target
        (template α=100/moving α=0, or inverse); during slider drags it is False
        so the user's manual opacity/colormap choices are kept (#2). ``roi_frac``
        restricts the live preview to a centered crop for speed."""
        it = self._item()
        if it is None or self.p._tpl_field is None or self.p._mov_field is None:
            return
        a = self.p.aligner
        if set_opacity:
            # Target switch: ensure overlay mode + favored opacity (resets to the
            # template/moving emphasis), then the user may re-tweak alpha/channel.
            a.onViewModeChanged("overlay")
            if active_target == "template":
                a.template_opacity.setValue(100); a.moving_opacity.setValue(0)
            else:
                a.template_opacity.setValue(0); a.moving_opacity.setValue(100)
        self.overlay_cell(it["z_tpl_um"], it["z_mov_um"], self.align_b(roi_frac),
                          force_colormap=False, roi_frac=roi_frac)
        a.raise_(); a.activateWindow()

    def reset(self, target, roi_frac=1.0):
        self.set_distance(target, 0.0, roi_frac=roi_frac)

    def on_target_shown(self, target, roi_frac=1.0):
        """Refresh the main-window live preview when the dialog target changes,
        favoring the newly selected target's opacity (no z change)."""
        self._render_main(active_target=target, roi_frac=roi_frac)

    def both_wavefronts(self):
        return self.p._tpl_field is not None and self.p._mov_field is not None

    def fields(self, roi_frac=1.0):
        # Both fields are at template sampling (moving is pre-scaled), so a
        # centered crop of each is a valid co-focus comparison region (#1).
        return (opt.center_crop(self.p._tpl_field, roi_frac),
                opt.center_crop(self.p._mov_field, roi_frac))

    def pixel_sizes(self):
        px_tpl, _px_mov, lam, n = self.p._pixel_sizes()
        # Moving field is pre-scaled to template sampling -> use px_tpl for both.
        return px_tpl, px_tpl, lam, n

    def align_b(self, roi_frac=1.0):
        p = self.p
        gm = p._global_matrix if p._global_matrix is not None else np.eye(3)
        distortion = p._distortion
        if roi_frac is not None and roi_frac < 1.0:
            # Focus search on the centered crop: the full-frame linear matrix
            # doesn't apply to crop coords, so just fit moving to the cropped
            # template shape (focus is what matters over a centered ROI).
            cropped_tpl_shape = opt.center_crop(p._tpl_field, roi_frac).shape
            return lambda field: p._fit_to_template(field, cropped_tpl_shape)
        tpl_shape = p._tpl_field.shape

        def f(field):
            field = p._fit_to_template(field, tpl_shape)
            if p._global_matrix is not None or distortion is not None:
                field = opt.warp_field_with_distortion(field, gm, distortion, tpl_shape)
            return field
        return f

    def apply_optimal(self, z_tpl_um, z_mov_um):
        self.set_distance("template", z_tpl_um)
        self.set_distance("moving", z_mov_um)

    def overlay_cell(self, z_tpl_um, z_mov_um, align_b, force_colormap=True, roi_frac=1.0):
        """Render template@z_tpl + aligned moving@z_mov into the main window.

        ``force_colormap`` (used by the 2-D map click) sets Turbo; the live
        distance preview passes False so the user's channel/colormap/alpha are
        preserved (#2). The displayed observable follows the panel's selectors.
        ``roi_frac`` < 1 propagates only the centered crop (much faster for the
        live distance slider) — the preview then shows that ROI; the chosen z is
        applied to the full frame elsewhere (save / 2-D map 'Use optimal')."""
        p = self.p
        px_tpl, px_mov, lam, n = self.pixel_sizes()
        a = p.aligner
        # Crop BEFORE propagation so the FFT runs on the small ROI (#3).
        tpl_in = opt.center_crop(p._tpl_field, roi_frac)
        mov_in = opt.center_crop(p._mov_field, roi_frac)
        tpl = opt.propagate_asm(tpl_in, z_tpl_um * 1e-6, lam, px_tpl, n=n)
        mov = align_b(opt.propagate_asm(mov_in, z_mov_um * 1e-6, lam, px_mov, n=n))
        # Colormap set once at row load; don't re-force here (#2). force_colormap kept
        # for API compat but no longer changes the user's choice.
        a.template_image = p._field_to_observable(tpl)
        a.transform_controls.set_template_shape(a.template_image.shape)
        a.moving_image = p._field_to_observable(mov); a.transformed_image = a.moving_image
        a.moving_image_file = p.rows[p.current_index]["moving_path"]
        a.template_image_file = p.rows[p.current_index]["template_path"]
        # Hand the (already-propagated, already-aligned) complex fields to the
        # aligner so its Observable selector re-derives this exact view (#1).
        # The moving is already aligned -> clear the aligner transform so the
        # observable recompute does NOT warp it again (this is a display preview;
        # the real matrix/distortion stay on the panel in _global_matrix/_distortion).
        a.current_transform = None
        a.distortion_transform = None
        a.template_field = np.asarray(tpl); a.template_z_um = 0.0
        a.moving_field = np.asarray(mov); a.moving_z_um = 0.0
        a.propagation_optics.update(p._current_optics())
        a.update_display()  # do NOT onViewModeChanged -> keeps user's opacity
