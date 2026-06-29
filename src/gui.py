# Copyright (c) 2025 Corentin Soubeiran
# SPDX-License-Identifier: MIT
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from skimage import transform as tf
import tifffile
from PIL import Image

from autoAlignersGui import BruteForceDialog
from imageCanva import ImageCanvas
from transformControls import TransformControls
from keyPointsSelection import KeyPointsSelection, estimate_transform_keypoints, estimate_constrained_transform
from keyPointsDetectionAndSelection import KeyPointsDetectionAndSelection, detect_keypoint_pairs
from utils_images import load_imgfile, load_wavefront_tif, save_stack
from optics import (
    estimate_scale_translation, estimate_distortion,
    field_from_phase_amp, phase_amp_from_field, propagate_asm,
    sample_pixel_size, warp_field_with_distortion, unwrapped_phase,
)

from skimage.registration import phase_cross_correlation
import cv2
import matplotlib.pyplot as plt

# Matplotlib colormaps
available_colormaps = ["gray", "red", "green", "blue", "cyan", "magenta"] + plt.colormaps()


class AlignerDistanceController:
    """Adapts :class:`ImageAligner` to the :class:`DistanceDialog` controller API
    so the same dialog can drive either the main window or another panel.

    A controller exposes: ``parent_widget``, ``targets()``, ``optics()``,
    ``magnification(target)``, ``set_optics(target, mag, wl, px, n)``,
    ``current_z(target)``, ``set_distance(target, z_um)``, ``reset(target)``,
    ``both_wavefronts()``, ``fields()`` -> (template_field, moving_field),
    ``align_b()`` -> callable, ``pixel_sizes()`` -> (px_tpl, px_mov, lam, n),
    ``apply_optimal(z_tpl_um, z_mov_um)`` and ``overlay_cell(z_tpl_um, z_mov_um,
    align_b)``.
    """

    def __init__(self, aligner):
        self.a = aligner
        self.parent_widget = aligner

    def targets(self):
        out = []
        if self.a.moving_field is not None:
            out.append("moving")
        if self.a.template_field is not None:
            out.append("template")
        return out

    def optics(self):
        return self.a.propagation_optics

    def magnification(self, target):
        o = self.a.propagation_optics
        return o["mag_tpl"] if target == "template" else o["mag_mov"]

    def set_optics(self, target, mag, wl, px, n):
        o = self.a.propagation_optics
        o["mag_tpl" if target == "template" else "mag_mov"] = mag
        o.update({"wavelength_nm": wl, "camera_pixel_um": px, "n": n})

    def current_z(self, target):
        return self.a._wf_z(target)

    def set_distance(self, target, z_um, roi_frac=1.0):
        # Main-window slider re-derives the full image via the aligner's own
        # (GPU) propagation; roi_frac is a focusing-panel preview concern.
        self.a.set_propagation_distance(z_um, target=target)

    def reset(self, target, roi_frac=1.0):
        self.a.reset_propagation(target=target)

    def both_wavefronts(self):
        return self.a.template_field is not None and self.a.moving_field is not None

    def fields(self, roi_frac=1.0):
        # Aligner's moving field is raw (different sampling than template), so the
        # ROI crop is applied inside align_b after rescale+fit; return full fields.
        return self.a.template_field, self.a.moving_field

    def pixel_sizes(self):
        o = self.a.propagation_optics
        cam = o["camera_pixel_um"] * 1e-6
        return (sample_pixel_size(cam, o["mag_tpl"]), sample_pixel_size(cam, o["mag_mov"]),
                o["wavelength_nm"] * 1e-9, o["n"])

    def align_b(self, roi_frac=1.0):
        a = self.a
        o = a.propagation_optics
        scale = magnification_scale_safe(o["mag_tpl"], o["mag_mov"])
        gm = a.current_transform.params if a.current_transform is not None else np.eye(3)
        distortion = a.distortion_transform
        tpl_shape = a.template_field.shape

        def f(field):
            if abs(scale - 1.0) > 1e-6:
                field = (tf.rescale(field.real, scale, order=1, preserve_range=True)
                         + 1j * tf.rescale(field.imag, scale, order=1, preserve_range=True))
            field = DistanceDialog._fit_shape(field, tpl_shape)
            if a.current_transform is not None or distortion is not None:
                field = warp_field_with_distortion(field, gm, distortion, tpl_shape)
            return field
        return f

    def apply_optimal(self, z_tpl_um, z_mov_um):
        self.a.set_propagation_distance(z_tpl_um, target="template")
        self.a.set_propagation_distance(z_mov_um, target="moving")

    def overlay_cell(self, z_tpl_um, z_mov_um, align_b):
        # Pure display preview (does NOT mutate the aligner's raw fields/transform,
        # so 'Use optimal' still propagates from the raw fields). Renders the
        # aligner's chosen observable so a click shows phase immediately.
        a = self.a
        px_tpl, px_mov, lam, n = self.pixel_sizes()
        tpl = propagate_asm(a.template_field, z_tpl_um * 1e-6, lam, px_tpl, n=n)
        mov = align_b(propagate_asm(a.moving_field, z_mov_um * 1e-6, lam, px_mov, n=n))
        a.template_image = a._observable_image(tpl, "template")
        a.transform_controls.set_template_shape(a.template_image.shape)
        a.moving_image = a._observable_image(mov, "moving")
        a.transformed_image = a.moving_image
        a.onViewModeChanged("overlay"); a.update_display()


def magnification_scale_safe(mag_tpl, mag_mov):
    from optics import magnification_scale
    return magnification_scale(mag_tpl, mag_mov)


class DistanceDialog(QDialog):
    """Change-distance dialog: target (template/moving) + optics + a live slider.

    Drives a *controller* (default :class:`AlignerDistanceController`) so the
    SAME dialog can change propagation in the main window or in another panel
    (e.g. the Focusing tool's manual-refocus step). Each change re-propagates
    the raw field and re-derives that image; 'Reset to 0' reloads that file.
    """

    Z_STEPS = 500  # slider ticks across +/- half-range

    def __init__(self, controller):
        # Accept either a controller or a bare ImageAligner (back-compat).
        if not hasattr(controller, "targets"):
            controller = AlignerDistanceController(controller)
        self.ctrl = controller
        super().__init__(controller.parent_widget)
        self.setWindowTitle("Change Propagation Distance")
        self.setModal(False)
        v = QVBoxLayout(self)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target:"))
        self.target = QComboBox()
        for t in self.ctrl.targets():
            self.target.addItem(t)
        self.target.currentTextChanged.connect(self._on_target_changed)
        target_row.addWidget(self.target)
        v.addLayout(target_row)

        o = self.ctrl.optics()
        opt_row = QHBoxLayout()
        self.mag = self._spin(o["mag_mov"], " mag×", 0.1, 200)
        self.wl = self._spin(o["wavelength_nm"], " nm", 100, 2000)
        self.px = self._spin(o["camera_pixel_um"], " µm px", 0.1, 100, 0.01, 3)
        self.n = self._spin(o["n"], " n", 1.0, 2.0, 0.001, 3)
        for w in (self.mag, self.wl, self.px, self.n):
            opt_row.addWidget(w)
        v.addLayout(opt_row)

        rng_row = QHBoxLayout()
        rng_row.addWidget(QLabel("Half-range (µm):"))
        self.half = QDoubleSpinBox(); self.half.setRange(0.1, 500); self.half.setValue(20.0)
        self.half.valueChanged.connect(self._update_label)
        rng_row.addWidget(self.half)
        # Central-ROI fraction: the focus search (slider preview + 2-D map) runs
        # on this centered crop for speed; the chosen z is applied full-frame.
        rng_row.addWidget(QLabel("Focus ROI %:"))
        self.roi_pct = QSpinBox(); self.roi_pct.setRange(5, 100); self.roi_pct.setValue(25)
        self.roi_pct.setToolTip("Centered crop used for the focus search; z is applied to the full frame.")
        rng_row.addWidget(self.roi_pct)
        v.addLayout(rng_row)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(-self.Z_STEPS, self.Z_STEPS)
        self.slider.valueChanged.connect(self._on_slider)
        v.addWidget(self.slider)

        self.z_label = QLabel(); self.z_label.setStyleSheet("font-weight: bold;")
        v.addWidget(self.z_label)

        # --- 2D propagation map (co-focus of template vs moving) ---
        map_box = QGroupBox("2D propagation map (template vs moving co-focus)")
        mv = QVBoxLayout(map_box)
        mrow = QHBoxLayout()
        self.map_half_tpl = self._spin(5.0, " ±tplµm", 0.1, 200)
        self.map_step_tpl = self._spin(200.0, " tpl nm", 10, 5000)
        self.map_half_mov = self._spin(5.0, " ±movµm", 0.1, 200)
        self.map_step_mov = self._spin(200.0, " mov nm", 10, 5000)
        self.map_metric = QComboBox(); self.map_metric.addItems(["ncc", "l2", "ssim"])
        for w in (self.map_half_tpl, self.map_step_tpl, self.map_half_mov, self.map_step_mov, self.map_metric):
            mrow.addWidget(w)
        mv.addLayout(mrow)
        self.map_btn = QPushButton("Compute 2D Map")
        self.map_btn.setStyleSheet("QPushButton { background-color: #2196F3; }")
        self.map_btn.clicked.connect(self._compute_map)
        mv.addWidget(self.map_btn)
        map_hint = QLabel("Needs BOTH images to be wavefronts. The map window has "
                          "focal-point modes, go-to-optimal, click→overlay; 'Use optimal' "
                          "applies the distances to both targets.")
        map_hint.setStyleSheet("color: gray;"); map_hint.setWordWrap(True)
        mv.addWidget(map_hint)
        # Disable if either image is not a wavefront.
        if not self.ctrl.both_wavefronts():
            map_box.setEnabled(False)
            map_box.setToolTip("Both template and moving must be wavefronts.")
        v.addWidget(map_box)

        btns = QHBoxLayout()
        reset = QPushButton("Reset to 0 (reload file)")
        reset.clicked.connect(self._reset)
        close = QPushButton("Close"); close.clicked.connect(self.accept)
        btns.addWidget(reset); btns.addStretch(); btns.addWidget(close)
        v.addLayout(btns)
        self._focus_map_window = None
        self._on_target_changed(self.target.currentText())

    @staticmethod
    def _spin(val, suffix, lo, hi, step=None, decimals=2):
        s = QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(decimals)
        s.setValue(val); s.setSuffix(suffix)
        if step is not None:
            s.setSingleStep(step)
        return s

    def _tgt(self):
        return self.target.currentText() or "moving"

    def _on_target_changed(self, _text):
        # Load the selected target's magnification + current distance into the UI.
        tgt = self._tgt()
        if not tgt:
            return
        self.mag.blockSignals(True)
        self.mag.setValue(self.ctrl.magnification(tgt))
        self.mag.blockSignals(False)
        cur = self.ctrl.current_z(tgt)
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(cur / max(self.half.value(), 1e-9) * self.Z_STEPS)))
        self.slider.blockSignals(False)
        self._update_label()
        # Optional: let the controller refresh any live preview for this target.
        if hasattr(self.ctrl, "on_target_shown"):
            self.ctrl.on_target_shown(tgt, roi_frac=self._roi_frac())

    def _z_um(self):
        return self.slider.value() / self.Z_STEPS * self.half.value()

    def _push_optics(self):
        self.ctrl.set_optics(self._tgt(), self.mag.value(), self.wl.value(),
                             self.px.value(), self.n.value())

    def _roi_frac(self):
        return self.roi_pct.value() / 100.0

    def _on_slider(self, _v):
        self._push_optics()
        self.ctrl.set_distance(self._tgt(), self._z_um(), roi_frac=self._roi_frac())
        self._update_label()

    def _update_label(self):
        roi = self._roi_frac()
        roi_txt = "" if roi >= 1.0 else f"  ·  preview on central {roi * 100:g}% ROI"
        self.z_label.setText(f"{self._tgt()} distance: {self._z_um():.3f} µm{roi_txt}")

    def _reset(self):
        self.slider.blockSignals(True); self.slider.setValue(0); self.slider.blockSignals(False)
        self.ctrl.reset(self._tgt(), roi_frac=self._roi_frac())
        self._update_label()

    def _compute_map(self):
        """Compute the 2-D template-vs-moving co-focus map and show the shared
        FocusMapWindow (modes / go-to-optimal / click-overlay / progress)."""
        if not self.ctrl.both_wavefronts():
            QMessageBox.warning(self, "2D Map", "Both template and moving must be wavefronts.")
            return
        self._push_optics()
        from batchMode import FocusMapWindow
        import optics as o
        roi_frac = self.roi_pct.value() / 100.0
        tpl_field, mov_field = self.ctrl.fields(roi_frac)
        px_tpl, px_mov, lam, n = self.ctrl.pixel_sizes()
        align_b = self.ctrl.align_b(roi_frac)
        # On-cell live overlay uses the FULL fields (z applied to whole frame).
        overlay_align_b = self.ctrl.align_b(1.0)
        cur_tpl, cur_mov = self.ctrl.current_z("template"), self.ctrl.current_z("moving")
        z_tpl = cur_tpl * 1e-6 + o.focus_z_values(
            self.map_half_tpl.value() * 1e-6, self.map_step_tpl.value() * 1e-9)
        z_mov = cur_mov * 1e-6 + o.focus_z_values(
            self.map_half_mov.value() * 1e-6, self.map_step_mov.value() * 1e-9)
        metric = self.map_metric.currentText()

        def use_optimal(z_tpl_um, z_mov_um):
            self.ctrl.apply_optimal(z_tpl_um, z_mov_um)
            self._on_target_changed(self.target.currentText())

        def on_cell(z_tpl_um, z_mov_um):
            self.ctrl.overlay_cell(z_tpl_um, z_mov_um, overlay_align_b)

        if self._focus_map_window is None:
            self._focus_map_window = FocusMapWindow(on_use_optimal=use_optimal, on_cell=on_cell)
        else:
            self._focus_map_window._on_use_optimal = use_optimal
            self._focus_map_window._on_cell = on_cell
        win = self._focus_map_window
        win.show(); win.raise_(); win.activateWindow()

        za, zb, map2d = o.focus_consistency_map(
            tpl_field, mov_field, z_tpl, z_mov, lam, lam, px_tpl, px_mov,
            n=n, metric=metric, align_b=align_b, border=16, progress=win.set_progress)
        ref_col = int(np.argmin(np.abs(za - cur_tpl * 1e-6)))
        ref_row = int(np.argmin(np.abs(zb - cur_mov * 1e-6)))
        win.set_map(za * 1e6, zb * 1e6, map2d, metric, ref_col=ref_col, ref_row=ref_row)

    @staticmethod
    def _fit_shape(field, shape):
        """Center-crop/edge-pad a complex field to ``shape`` (map comparison)."""
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


class DistortionFitDialog(QDialog):
    """Fit-quality popup for an estimated distortion: numeric metrics, a
    deformation-grid figure with the keypoint pairs (origin->destination), a
    before/after image-correlation sanity check, and a Cancel button to discard
    the estimate (#2a/#2c/#2d)."""

    def __init__(self, parent, distortion_tf, pairs, shape, model,
                 residual_matrix=None, corr_before=None, corr_after=None,
                 on_cancel=None, params=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        import optics as o
        self._on_cancel = on_cancel
        self.setWindowTitle(f"Distortion fit quality — {model}")
        self.resize(600, 700)
        v = QVBoxLayout(self)

        q = o.distortion_fit_quality(distortion_tf, pairs, residual_matrix=residual_matrix)
        lines = [f"<b>Model:</b> {model}",
                 f"<b>Pairs:</b> {q['n']}",
                 f"<b>Residual RMS:</b> {q['rms']:.3f} px",
                 f"<b>Max residual:</b> {q['max_err']:.3f} px",
                 f"<b>R² (displacement explained):</b> {q['r2'] * 100:.1f}%"]
        # Detection parameters used to obtain the pairs (#2).
        if params:
            order = [("detector", "Detector"), ("matcher", "Matcher"),
                     ("distance_ratio", "Distance ratio"), ("ransac_threshold", "RANSAC px"),
                     ("border_crop", "Border crop px"), ("max_features", "Max features")]
            pstr = "  ·  ".join(f"{label}: {params[key]}" for key, label in order if key in params)
            if pstr:
                lines.append(f"<b>Keypoints:</b> {pstr}")
        if "k" in q:
            lines.append(f"<b>Curvature k:</b> {q['k']:.3e} px⁻²")
        if q.get("effective_radius_px") not in (None, float("inf")):
            lines.append(f"<b>Eff. radius:</b> {q['effective_radius_px']:.0f} px")
        quality = ("Excellent" if q["rms"] < 1 else "Good" if q["rms"] < 3
                   else "Fair" if q["rms"] < 8 else "Poor")
        lines.append(f"<b>Residual assessment:</b> {quality}")
        # Image-correlation sanity check (#2c).
        if corr_before is not None and corr_after is not None:
            delta = corr_after - corr_before
            verdict = "improved" if delta > 1e-3 else ("no change" if abs(delta) <= 1e-3 else "WORSE")
            color = "#2e7d32" if delta > 1e-3 else ("#888" if abs(delta) <= 1e-3 else "#c62828")
            lines.append(f"<b>Image NCC:</b> before {corr_before:.4f} → after {corr_after:.4f} "
                         f"<span style='color:{color}'>({verdict})</span>")
            if delta < -1e-3:
                lines.append("<span style='color:#c62828'>Correction lowers correlation — "
                             "likely a bad estimate; consider Cancel.</span>")
        lbl = QLabel("<br>".join(lines))
        lbl.setTextFormat(Qt.RichText); lbl.setWordWrap(True)
        v.addWidget(lbl)

        fig = Figure(figsize=(5, 4), tight_layout=True)
        canvas = FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(111)
        gx, gy, dx, dy = o.deformation_grid(distortion_tf, shape, n=16)
        for r in range(dx.shape[0]):
            ax.plot(dx[r, :], dy[r, :], color="#1f77b4", lw=0.6)
        for c in range(dx.shape[1]):
            ax.plot(dx[:, c], dy[:, c], color="#1f77b4", lw=0.6)
        ax.quiver(gx, gy, dx - gx, dy - gy, angles="xy", scale_units="xy",
                  scale=1, color="#d62728", width=0.003, label="grid deformation")
        # Keypoint pairs: template (origin) -> moving (destination) (#2a).
        if pairs:
            tp = np.asarray([p[0] for p in pairs], dtype=float)
            mp = np.asarray([p[1] for p in pairs], dtype=float)
            ax.scatter(tp[:, 0], tp[:, 1], s=10, c="#2e7d32", marker="o", label="template pt")
            ax.scatter(mp[:, 0], mp[:, 1], s=10, c="#ff9800", marker="x", label="moving pt")
            ax.quiver(tp[:, 0], tp[:, 1], mp[:, 0] - tp[:, 0], mp[:, 1] - tp[:, 1],
                      angles="xy", scale_units="xy", scale=1, color="#9c27b0",
                      width=0.002, alpha=0.6)
        ax.set_title("Deformation grid + keypoint pairs (template→moving)", fontsize=9)
        ax.set_xlim(0, shape[1]); ax.set_ylim(shape[0], 0)
        ax.set_aspect("equal"); ax.tick_params(labelsize=6)
        ax.legend(loc="upper right", fontsize=6, framealpha=0.6)
        v.addWidget(canvas, stretch=1)

        btns = QHBoxLayout(); btns.addStretch()
        if on_cancel is not None:
            cancel = QPushButton("Cancel (discard correction)")
            cancel.setStyleSheet("QPushButton { background-color: #c62828; }")
            cancel.clicked.connect(self._cancel)
            btns.addWidget(cancel)
        close = QPushButton("Close (keep)"); close.clicked.connect(self.accept)
        btns.addWidget(close)
        v.addLayout(btns)

    def _cancel(self):
        if self._on_cancel is not None:
            self._on_cancel()
        self.reject()


def show_distortion_fit_quality(parent, distortion_tf, pairs, shape, model,
                                residual_matrix=None, corr_before=None,
                                corr_after=None, on_cancel=None, params=None,
                                modal=False):
    """Show the distortion fit-quality popup (best-effort; never raises).

    ``modal`` blocks (``exec``) until the user clicks Close or Cancel -- used
    during a sequence Run so the next step waits for the user's decision."""
    try:
        dlg = DistortionFitDialog(parent, distortion_tf, pairs, shape, model,
                                  residual_matrix=residual_matrix,
                                  corr_before=corr_before, corr_after=corr_after,
                                  on_cancel=on_cancel, params=params)
        parent._distortion_fit_dialog = dlg  # keep a ref so it isn't GC'd
        if modal:
            dlg.setModal(True)
            dlg.exec()
        else:
            dlg.show()
    except Exception as e:
        print(f"Distortion fit-quality popup failed: {e}")


class ImageAligner(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.template_image = None
        self.moving_image = None
        self.transformed_image = None
        self.current_transform = None
        self.opt_transform = None
        # Optional non-rigid distortion warp (template->moving) applied on top of
        # current_transform, e.g. from the keypoint dialog's distortion option.
        self.distortion_transform = None
        # Raw WAVEFRONTS (complex) + per-target propagation state. When a loaded
        # file is a 2-channel wavefront, that image is DERIVED from
        # propagate(field, z) via the chosen observable; changing z re-derives
        # the whole field. Both template and moving can be wavefronts. None for
        # plain (non-wavefront) images.
        self.moving_field = None
        self.moving_field_file = None
        self.moving_z_um = 0.0
        self.moving_observable = "amplitude"  # real / imaginary / amplitude / phase
        self.template_field = None
        self.template_field_file = None
        self.template_z_um = 0.0
        self.template_observable = "amplitude"
        self.propagation_optics = {
            "mag_tpl": 20.0, "na_tpl": 0.8, "mag_mov": 10.0, "na_mov": 0.45,
            "wavelength_nm": 660.0, "camera_pixel_um": 5.86, "n": 1.0,
        }
        self.optimizer_dialog_open = False
        self.keypoints_dialog = None
        self.auto_keypoints_dialog = None
        self.batch_panel = None
        self.progressive_panel = None
        self.focusing_panel = None
        self.init_ui()
        self.update_canvas_drag_state()
        
    def init_ui(self):
        self.setWindowTitle("Microscopy Image Alignment Tool")
        self.setGeometry(100, 100, 1920, 1080)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Image display
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Image controls
        img_controls = QHBoxLayout()
        
        load_template_btn = QPushButton("Load Template")
        load_template_btn.clicked.connect(self.load_template)
        load_template_btn.setStyleSheet("QPushButton { background-color: #4CAF50; }")
        self.load_template_btn = load_template_btn

        load_moving_btn = QPushButton("Load Moving Image")
        load_moving_btn.clicked.connect(self.load_moving)
        load_moving_btn.setStyleSheet("QPushButton { background-color: #4CAF50; }")
        self.load_moving_btn = load_moving_btn

        view_mode = QComboBox()
        view_mode.addItems(["Overlay", "Side by Side"])
        view_mode.currentTextChanged.connect(lambda text: self.onViewModeChanged(text.lower().replace(" ", "_")))

        img_controls.addWidget(load_template_btn)
        img_controls.addWidget(load_moving_btn)
        img_controls.addWidget(QLabel("View Mode:"))
        img_controls.addWidget(view_mode)
        
        # Colormap controls
        color_controls = QHBoxLayout()
        
        color_controls.addWidget(QLabel("Template:"))
        self.template_color = QComboBox()
        self.template_color.addItems(available_colormaps)
        self.template_color.setCurrentText("green")
        self.template_color.currentTextChanged.connect(self.update_display)
        color_controls.addWidget(self.template_color)
        
        color_controls.addWidget(QLabel("Moving:"))
        self.moving_color = QComboBox()
        self.moving_color.addItems(available_colormaps)
        self.moving_color.setCurrentText("red")
        self.moving_color.currentTextChanged.connect(self.update_display)
        color_controls.addWidget(self.moving_color)
        
        # Opacity controls
        opacity_controls = QHBoxLayout()
        
        opacity_controls.addWidget(QLabel("Template Opacity:"))
        self.template_opacity = QSlider(Qt.Orientation.Horizontal)
        self.template_opacity.setRange(0, 100)
        self.template_opacity.setValue(50)
        self.template_opacity.valueChanged.connect(self.update_display)
        opacity_controls.addWidget(self.template_opacity)
        
        opacity_controls.addWidget(QLabel("Moving Opacity:"))
        self.moving_opacity = QSlider(Qt.Orientation.Horizontal)
        self.moving_opacity.setRange(0, 100)
        self.moving_opacity.setValue(50)
        self.moving_opacity.valueChanged.connect(self.update_display)
        opacity_controls.addWidget(self.moving_opacity)
        
        # Canvas
        self.canvas = ImageCanvas()
        self.canvas.setMinimumSize(600, 600)
        self.canvas.template_dragged.connect(self.on_template_drag)
        
        left_layout.addLayout(img_controls)
        left_layout.addLayout(color_controls)
        left_layout.addLayout(opacity_controls)
        left_layout.addWidget(self.canvas)
        
        # Right panel - Controls
        right_panel = QWidget()
        right_panel.setMaximumWidth(400)
        right_layout = QVBoxLayout(right_panel)
        
        # Transform controls
        self.transform_controls = TransformControls()
        self.transform_controls.transform_changed.connect(self.apply_transform)

        # Wavefront / propagation controls (active for wavefront images). The
        # target combo selects whether the observable / distance apply to the
        # template or the moving wavefront.
        wf_group = QGroupBox("Wavefront")
        wf_layout = QVBoxLayout()
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target:"))
        self.wf_target_combo = QComboBox()
        self.wf_target_combo.addItems(["moving", "template"])
        self.wf_target_combo.currentTextChanged.connect(self._on_wf_target_changed)
        target_row.addWidget(self.wf_target_combo)
        wf_layout.addLayout(target_row)
        obs_row = QHBoxLayout()
        obs_row.addWidget(QLabel("Observable:"))
        self.observable_combo = QComboBox()
        self.observable_combo.addItems(["amplitude", "phase", "real", "imaginary"])
        self.observable_combo.currentTextChanged.connect(self._on_observable_changed)
        obs_row.addWidget(self.observable_combo)
        wf_layout.addLayout(obs_row)
        self.distance_label = QLabel("Propagation (moving): 0.000 µm")
        self.distance_label.setStyleSheet("color: gray;")
        wf_layout.addWidget(self.distance_label)
        reset_dist_btn = QPushButton("Reset distance (reload file)")
        reset_dist_btn.clicked.connect(lambda: self.reset_propagation(self._wf_target()))
        wf_layout.addWidget(reset_dist_btn)
        remove_dist_btn = QPushButton("Remove distortion")
        remove_dist_btn.clicked.connect(self.remove_distortion)
        wf_layout.addWidget(remove_dist_btn)
        wf_group.setLayout(wf_layout)

        # Optimization controls
        opt_group = QGroupBox("Optimization")
        opt_layout = QVBoxLayout()

        self.opt_method = QComboBox()
        self.opt_method.addItems(["Manual Pairs of Points", "Auto Detect Keypoints", "Phase Cross-Correlation", "Brute Force", "Distortion Correction", "Change Distance"])
        
        optimize_btn = QPushButton("Optimize Alignment")
        optimize_btn.setStyleSheet("QPushButton { background-color: #2196F3; }")
        optimize_btn.clicked.connect(self.optimize_alignment)
        
        opt_layout.addWidget(QLabel("Method:"))
        opt_layout.addWidget(self.opt_method)
        opt_layout.addWidget(optimize_btn)
        opt_group.setLayout(opt_layout)
        self.opt_layout = opt_layout
        self.opt_group = opt_group
        
        # Export controls
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        
        # self.crop_check = QCheckBox("Crop to overlap")
        # self.crop_check.setChecked(True)
        
        export_btn = QPushButton("Export Transformed Image")
        export_btn.setStyleSheet("QPushButton { background-color: #FF9800; }")
        export_btn.clicked.connect(self.export_image)
        
        batch_btn = QPushButton("Apply to Folder")
        batch_btn.setStyleSheet("QPushButton { background-color: #9C27B0; }")
        batch_btn.clicked.connect(self.batch_process)
        
        # export_layout.addWidget(self.crop_check)
        export_layout.addWidget(export_btn)
        export_layout.addWidget(batch_btn)
        export_group.setLayout(export_layout)
        
        right_layout.addWidget(self.transform_controls)
        right_layout.addWidget(wf_group)
        right_layout.addWidget(opt_group)
        right_layout.addWidget(export_group)
        right_layout.addStretch()
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(right_panel, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready to load images")

        # Create menu bar
        self.setup_menubar()
        self.update_canvas_drag_state()

    def setup_menubar(self): 
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        edit_menu = menubar.addMenu("Edit")
        # File menu actions
        load_template_action = QAction("Load Template Image", self)
        load_template_action.triggered.connect(self.load_template)
        file_menu.addAction(load_template_action)
        load_moving_action = QAction("Load Moving Image", self)
        load_moving_action.triggered.connect(self.load_moving)
        file_menu.addAction(load_moving_action)
        file_menu.addSeparator()
        save_transform_action = QAction("Save Transformation Matrix", self)
        save_transform_action.triggered.connect(self.save_transform_matrix)
        file_menu.addAction(save_transform_action)
        load_transform_action = QAction("Load Transformation Matrix", self)
        load_transform_action.triggered.connect(self.load_transform_matrix)
        file_menu.addAction(load_transform_action)
        file_menu.addSeparator()
        export_image_action = QAction("Export Transformed Image", self)
        export_image_action.triggered.connect(self.export_image)
        file_menu.addAction(export_image_action)
        batch_process_action = QAction("Batch Process Folder", self)
        batch_process_action.triggered.connect(self.batch_process)
        file_menu.addAction(batch_process_action)
        open_batch_mode_action = QAction("Open Batch Mode Panel", self)
        open_batch_mode_action.triggered.connect(self.open_batch_mode)
        file_menu.addAction(open_batch_mode_action)
        open_progressive_action = QAction("Open Progressive Folder Alignment", self)
        open_progressive_action.triggered.connect(self.open_progressive_folder)
        file_menu.addAction(open_progressive_action)
        open_focusing_action = QAction("Open Wavefront Focusing && Alignment", self)
        open_focusing_action.triggered.connect(self.open_focusing_tool)
        file_menu.addAction(open_focusing_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        # Edit menu actions
        reset_transform_action = QAction("Reset Transformation", self)
        reset_transform_action.triggered.connect(self.reset_transformation)
        edit_menu.addAction(reset_transform_action)
        load_export_stack_action = QAction("Apply transform to a ImageJ stack", self)
        load_export_stack_action.triggered.connect(self.load_and_export_stack)
        edit_menu.addAction(load_export_stack_action)


    def load_template(self):
        """Load template (reference) image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Template Image", "", "Image Files (*.tif *.tiff *.png *.jpg)"
        )
        if file_path:
            self.load_template_from_path(file_path)

    def load_template_from_path(self, file_path):
        """Load template (reference) image from an explicit path (no dialog).

        Like the moving loader, keeps the raw complex field when the file is a
        2-channel wavefront so the template can also be re-propagated."""
        self.template_field = None
        self.template_field_file = None
        self.template_z_um = 0.0
        try:
            phase, amp, _n = load_wavefront_tif(file_path, frame_index=0)
            self.template_field = field_from_phase_amp(phase, amp)
            self.template_field_file = file_path
        except Exception:
            self.template_field = None
        if self.template_field is not None:
            self.set_template_array(self._observable_image(self.template_field, "template"), file_path)
            self.statusBar().showMessage(
                f"Loaded template wavefront: {Path(file_path).name} (propagation enabled)")
        else:
            self.set_template_array(load_imgfile(file_path).astype(np.float32), file_path)
            self.statusBar().showMessage(f"Loaded template: {Path(file_path).name}")
        self._update_distance_label()

    def set_template_array(self, arr, source_path):
        """Set the template (reference) image from an in-memory array (no file
        read). Used by Progressive Folder Alignment's sliding mode to push the
        per-image reference frame (image N-X) in as the alignment reference.
        """
        self.template_image_file = source_path
        self.template_image = arr.astype(np.float32)
        self.transform_controls.set_template_shape(self.template_image.shape)
        self.canvas.set_template(
            self.template_image,
            self.template_color.currentText(),
            self.template_opacity.value() / 100
        )
        self.load_template_btn.setStyleSheet(f"QPushButton {{ background-color: {QApplication.instance().palette().button().color().name()}; }}")
        self.update_canvas_drag_state()

    def load_moving(self):
        """Load moving image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Moving Image", "", "Image Files (*.tif *.tiff *.png *.jpg)"
        )
        if file_path:
            self.load_moving_from_path(file_path)

    def load_moving_from_path(self, file_path):
        """Load moving image from an explicit path (no dialog).

        If the file is a 2-channel wavefront, keep the raw complex field so the
        propagation distance can be changed later (the moving image is then
        derived from ``propagate(field, z)`` via the chosen observable). Plain
        images load as before with no propagation.
        """
        self.moving_field = None
        self.moving_field_file = None
        self.moving_z_um = 0.0
        try:
            phase, amp, _n = load_wavefront_tif(file_path, frame_index=0)
            self.moving_field = field_from_phase_amp(phase, amp)
            self.moving_field_file = file_path
        except Exception:
            self.moving_field = None  # not a wavefront -> plain image path
        self._update_distance_label()
        if self.moving_field is not None:
            self.set_moving_array(self._observable_image(self.moving_field, "moving"), file_path)
            self.statusBar().showMessage(
                f"Loaded moving wavefront: {Path(file_path).name} (propagation enabled)")
        else:
            self.set_moving_array(load_imgfile(file_path).astype(np.float32), file_path)
            self.statusBar().showMessage(f"Loaded moving image: {Path(file_path).name}")

    # ---- per-target wavefront/propagation helpers (target in {template, moving}) ----
    def _wf_field(self, target):
        return self.template_field if target == "template" else self.moving_field

    def _wf_observable(self, target):
        return self.template_observable if target == "template" else self.moving_observable

    def _wf_z(self, target):
        return self.template_z_um if target == "template" else self.moving_z_um

    def _observable_image(self, field, target):
        """2-D real view of a complex field per the target's observable."""
        obs = self._wf_observable(target)
        if obs == "phase":
            return unwrapped_phase(field).astype(np.float32)
        if obs == "real":
            return np.real(field).astype(np.float32)
        if obs == "imaginary":
            return np.imag(field).astype(np.float32)
        return np.abs(field).astype(np.float32)  # amplitude (default)

    def _propagated_field(self, target):
        """The raw field for ``target`` propagated to its current distance, or None."""
        field = self._wf_field(target)
        if field is None:
            return None
        o = self.propagation_optics
        mag = o["mag_tpl"] if target == "template" else o["mag_mov"]
        px = sample_pixel_size(o["camera_pixel_um"] * 1e-6, mag)
        lam = o["wavelength_nm"] * 1e-9
        return propagate_asm(field, self._wf_z(target) * 1e-6, lam, px, n=o["n"])

    def _recompute_from_field(self, target):
        """Re-derive the target's displayed image from its propagated field +
        observable, keeping the current transform."""
        field = self._propagated_field(target)
        if field is None:
            return
        if target == "template":
            self.set_template_array(self._observable_image(field, "template"), self.template_field_file)
            # Template shape may matter for the transform; re-apply on the moving.
            if self.current_transform is not None and self.moving_image is not None:
                self.apply_transform(self.transform_controls.transform_params)
        else:
            self.set_moving_array(self._observable_image(field, "moving"), self.moving_field_file)
        self._update_distance_label()

    def _update_distance_label(self):
        target = self._wf_target()
        field = self._wf_field(target)
        if field is None:
            self.distance_label.setText(f"Propagation ({target}): n/a (not a wavefront)")
        else:
            self.distance_label.setText(f"Propagation ({target}): {self._wf_z(target):.3f} µm")

    def _wf_target(self):
        """Currently selected wavefront target in the panel (template/moving)."""
        return self.wf_target_combo.currentText() if hasattr(self, "wf_target_combo") else "moving"

    def _on_wf_target_changed(self, _text):
        # Reflect the selected target's observable + distance in the panel.
        target = self._wf_target()
        obs = self._wf_observable(target)
        self.observable_combo.blockSignals(True)
        self.observable_combo.setCurrentText(obs)
        self.observable_combo.blockSignals(False)
        self._update_distance_label()

    def _on_observable_changed(self, _text):
        target = self._wf_target()
        obs = self.observable_combo.currentText()
        if target == "template":
            self.template_observable = obs
        else:
            self.moving_observable = obs
        if self._wf_field(target) is not None:
            self._recompute_from_field(target)

    def set_propagation_distance(self, z_um, target="moving"):
        """Set the propagation distance (µm) for ``target`` and re-derive it.

        Changes the entire wavefront the transform acts on (moving) or the
        reference frame (template)."""
        if self._wf_field(target) is None:
            QMessageBox.warning(self, "Propagation",
                                f"The {target} image is not a wavefront; propagation is disabled.")
            return
        if target == "template":
            self.template_z_um = float(z_um)
        else:
            self.moving_z_um = float(z_um)
        self._recompute_from_field(target)
        self.statusBar().showMessage(f"Propagation distance ({target}): {self._wf_z(target):.3f} µm")

    def reset_propagation(self, target="moving"):
        """Reset the target's distance to 0 by reloading its file from disk."""
        path = self.template_field_file if target == "template" else self.moving_field_file
        if not path:
            return
        if target == "template":
            self.load_template_from_path(path)
        else:
            self.load_moving_from_path(path)
        self.statusBar().showMessage(f"Propagation ({target}) reset to 0 µm (reloaded).")

    def remove_distortion(self):
        """Clear any distortion warp (keep the linear transform)."""
        if self.distortion_transform is None:
            self.statusBar().showMessage("No distortion to remove.")
            return
        self.distortion_transform = None
        self.apply_transform(self.transform_controls.transform_params)
        self.statusBar().showMessage("Distortion correction removed.")

    def set_moving_array(self, arr, source_path):
        """Set the moving image from an in-memory array (no file read).

        Refreshes the canvas and re-applies the current transform if any. Used
        by Progressive Folder Alignment to push a stack's representative frame
        in directly. ``source_path`` is recorded as ``moving_image_file`` so
        exports keep the original name/extension.
        """
        self.moving_image_file = source_path
        self.moving_image = arr.astype(np.float32)
        self.canvas.set_moving(
            self.moving_image,
            self.moving_color.currentText(),
            self.moving_opacity.value() / 100
        )
        self.load_moving_btn.setStyleSheet(f"QPushButton {{ background-color: {QApplication.instance().palette().button().color().name()}; }}")
        # When loading a new moving image, apply current transform if available
        if self.current_transform is not None:
            self.apply_transform(self.transform_controls.transform_params)
        self.update_canvas_drag_state()
        
    def onViewModeChanged(self, mode):
        """Handle view mode change"""
        if mode == "overlay":
            #Enable the opacity sliders
            self.template_opacity.setValue(50)
            self.moving_opacity.setValue(50)
            self.template_opacity.setEnabled(True)
            self.moving_opacity.setEnabled(True)
            self.canvas.set_view_mode(mode)
        else:
            #Block the opacity sliders and set them to 100%
            self.template_opacity.setValue(100)
            self.moving_opacity.setValue(100)
            self.template_opacity.setEnabled(False)
            self.moving_opacity.setEnabled(False)
            self.canvas.set_view_mode(mode)
        self.update_display()
        self.update_canvas_drag_state()

    def open_keypoints_tool(self):
        if self.template_image is None or self.moving_image is None:
            QMessageBox.warning(self, "Warning", "Load both images first")
            return
        self.set_optimizer_dialog_state(True)
        # Force side by side mode for keypoint selection
        self.onViewModeChanged("side_by_side")
        self.canvas.enable_keypoint_mode(True)
        # dialog = KeyPointsSelection(self)

        self.keypoints_dialog = KeyPointsSelection(self)
        self.statusBar().showMessage(f"Select corresponding points in both images. Click 'Done' when finished.")
        self.canvas.point_added.connect(self.keypoints_dialog.add_pair)
        self.keypoints_dialog.show()   # non-blocking
        # Freeze the transform controls while selecting points
        self.transform_controls.setEnabled(False)
        # Add a grayed overlay layout to indicate that the control panel is inactive
        overlay = QFrame(self.transform_controls)
        overlay.setStyleSheet("background-color: rgba(85, 85, 85, 0.5);")
        overlay.setGeometry(self.transform_controls.rect())
        overlay.raise_()
        overlay.show()

        result = self.keypoints_dialog.exec()
        if result == QDialog.Accepted:
            # User clicked Done → get the pairs
            pairs = self.keypoints_dialog.point_pairs
            print("Collected pairs:", pairs)
            self.canvas.clear_keypoints()
            
            if len(pairs)>0:
                matrix = estimate_transform_keypoints(pairs)
                print("Estimated transformation matrix:\n", matrix)
                self.opt_transform = matrix
                transform = tf.AffineTransform(matrix=matrix)
                # Collect current matrix to combine it with the calculated one: 
                if self.current_transform is not None:
                    matrix = np.dot(transform.params, self.current_transform.params)
                    transform = tf.AffineTransform(matrix=matrix)
                print(f"Transform matrix:\n{transform.params}")
                self.onViewModeChanged("overlay")
                self.transform_controls.set_values_from_transform(transform.params)
                self.statusBar().showMessage(f"Estimated transform from {len(pairs)} point pairs.")
            else:
                self.statusBar().showMessage("No points selected. Operation cancelled.")
        #Release the transform controls
        self.transform_controls.setEnabled(True)
        #Remove the overlay
        overlay.deleteLater()
        self.keypoints_dialog = None
        self.set_optimizer_dialog_state(False)

    def open_auto_keypoints_tool(self):
        """Open automatic keypoint detection and selection tool."""
        if self.template_image is None or self.moving_image is None:
            QMessageBox.warning(self, "Warning", "Load both images first")
            return
        self.set_optimizer_dialog_state(True)
        
        # Force side by side mode for keypoint selection
        self.onViewModeChanged("side_by_side")
        self.canvas.enable_keypoint_mode(False)  # No manual clicking needed
        
        # Use transformed image if available, otherwise use original moving image
        moving_img_for_detection = self.transformed_image if self.transformed_image is not None else self.moving_image
        
        # Create dialog with images
        self.auto_keypoints_dialog = KeyPointsDetectionAndSelection(
            self, self.template_image, moving_img_for_detection
        )
        
        self.statusBar().showMessage("Configure and detect keypoints automatically.")
        self.auto_keypoints_dialog.show()  # non-blocking
        
        # Freeze the transform controls while working with keypoints
        self.transform_controls.setEnabled(False)
        overlay = QFrame(self.transform_controls)
        overlay.setStyleSheet("background-color: rgba(85, 85, 85, 0.5);")
        overlay.setGeometry(self.transform_controls.rect())
        overlay.raise_()
        overlay.show()
        
        result = self.auto_keypoints_dialog.exec()
        if result == QDialog.Accepted:
            # User clicked Done → get the pairs
            pairs = self.auto_keypoints_dialog.point_pairs
            print("Collected pairs:", len(pairs))
            self.canvas.clear_keypoints()
            
            if len(pairs) > 0:
                lock_rotation, lock_scale = self.auto_keypoints_dialog.constraints()
                matrix = estimate_constrained_transform(
                    pairs, lock_rotation=lock_rotation, lock_scale=lock_scale)
                print("Estimated transformation matrix:\n", matrix)
                self.opt_transform = matrix
                transform = tf.AffineTransform(matrix=matrix)

                # Combine with current matrix if exists
                if self.current_transform is not None:
                    matrix = np.dot(transform.params, self.current_transform.params)
                    transform = tf.AffineTransform(matrix=matrix)

                print(f"Transform matrix:\n{transform.params}")
                self.onViewModeChanged("overlay")
                # Optional distortion correction: residual of the incremental
                # keypoint fit (opt_transform), so it composes with the live
                # current_transform rather than overriding the scale/rotation.
                self._apply_keypoint_distortion(pairs, self.opt_transform)
                self.transform_controls.set_values_from_transform(transform.params)
                msg = f"Estimated transform from {len(pairs)} detected keypoint pairs."
                if self.distortion_transform is not None:
                    msg += " Distortion correction applied."
                self.statusBar().showMessage(msg)
            else:
                self.statusBar().showMessage("No keypoint pairs detected. Operation cancelled.")
        
        # Release the transform controls
        self.transform_controls.setEnabled(True)
        overlay.deleteLater()
        self.auto_keypoints_dialog = None
        self.set_optimizer_dialog_state(False)

    def open_distortion_tool(self):
        """Distortion-correction method: a keypoint-like dialog that estimates &
        applies ONLY a residual distortion warp (the linear transform is left
        as-is). Mirrors the auto-keypoints dialog (point 3)."""
        if self.template_image is None or self.moving_image is None:
            QMessageBox.warning(self, "Warning", "Load both images first")
            return
        self.set_optimizer_dialog_state(True)
        self.onViewModeChanged("side_by_side")
        self.canvas.enable_keypoint_mode(False)
        moving_for_detection = self.transformed_image if self.transformed_image is not None else self.moving_image
        self.auto_keypoints_dialog = KeyPointsDetectionAndSelection(
            self, self.template_image, moving_for_detection)
        # This tool is specifically about distortion -> default it ON.
        self.auto_keypoints_dialog.use_distortion.setChecked(True)
        self.auto_keypoints_dialog._update_distortion_enabled(True)
        self.statusBar().showMessage("Detect keypoints, then apply distortion correction.")
        self.auto_keypoints_dialog.show()
        result = self.auto_keypoints_dialog.exec()
        if result == QDialog.Accepted:
            pairs = self.auto_keypoints_dialog.point_pairs
            self.canvas.clear_keypoints()
            if len(pairs) >= 3:
                enabled, model = self.auto_keypoints_dialog.distortion_request()
                # Keypoints were detected on the ALREADY-TRANSFORMED moving image
                # (``transformed_image``), so the moving points are ALREADY in the
                # template frame -> the residual reference is IDENTITY. (Applying
                # current_transform again here double-counted it and made the
                # estimated distortion far too strong.)
                self._estimate_keypoint_distortion(pairs, np.eye(3),
                                                   model if enabled else "tps", warn=True,
                                                   show_quality=True)
                self.onViewModeChanged("overlay")
                self.apply_transform(self.transform_controls.transform_params)
                if self.distortion_transform is not None:
                    self.statusBar().showMessage(
                        f"Distortion correction applied from {len(pairs)} pairs ({model}).")
            else:
                self.statusBar().showMessage(f"Need >= 3 pairs for distortion ({len(pairs)} found).")
        self.auto_keypoints_dialog = None
        self.set_optimizer_dialog_state(False)

    def open_distance_tool(self):
        """Change-distance method: a dialog with a target (template/moving)
        selector, optics, and a live z slider that re-propagates the chosen
        wavefront in real time (point 4)."""
        if self.moving_field is None and self.template_field is None:
            QMessageBox.warning(self, "Change Distance",
                                "Neither image is a wavefront; propagation is disabled.")
            return
        dialog = DistanceDialog(self)
        dialog.exec()

    def auto_keypoints_headless(self, detector="AKAZE", matcher="Brute Force",
                                max_features=500, distance_ratio=0.75,
                                use_ransac=True, ransac_threshold=5.0,
                                lock_rotation=False, lock_scale=False,
                                distortion_model=None, border_crop=0):
        """Run automatic keypoint alignment without opening the dialog.

        Mirrors the matrix-combination logic of ``open_auto_keypoints_tool`` so
        batch mode can align headlessly. ``lock_rotation`` / ``lock_scale``
        constrain the estimated transform (via
        :func:`keyPointsSelection.estimate_constrained_transform`); a non-None
        ``distortion_model`` (tps/poly/radial/piecewise) additionally fits and
        applies a residual distortion warp. Returns the number of pairs used.
        """
        if self.template_image is None or self.moving_image is None:
            QMessageBox.warning(self, "Warning", "Load both images first")
            return 0

        moving_img_for_detection = (
            self.transformed_image if self.transformed_image is not None
            else self.moving_image
        )
        pairs = detect_keypoint_pairs(
            self.template_image, moving_img_for_detection,
            detector=detector, matcher=matcher, max_features=max_features,
            distance_ratio=distance_ratio, use_ransac=use_ransac,
            ransac_threshold=ransac_threshold, border_crop=border_crop,
        )
        if len(pairs) == 0:
            self.statusBar().showMessage("Auto keypoints: no pairs detected (transform unchanged).")
            return 0

        matrix = estimate_constrained_transform(
            pairs, lock_rotation=lock_rotation, lock_scale=lock_scale)
        self.opt_transform = matrix
        # Residual distortion (composes with the live linear transform).
        self._estimate_keypoint_distortion(pairs, self.opt_transform, distortion_model)
        transform = tf.AffineTransform(matrix=matrix)
        if self.current_transform is not None:
            matrix = np.dot(transform.params, self.current_transform.params)
            transform = tf.AffineTransform(matrix=matrix)
        self.onViewModeChanged("overlay")
        self.transform_controls.set_values_from_transform(transform.params)
        self.statusBar().showMessage(
            f"Auto keypoints (headless): estimated transform from {len(pairs)} pairs.")
        return len(pairs)

    def auto_keypoints_scale_translation_headless(self, detector="AKAZE",
                                                  matcher="Brute Force",
                                                  max_features=500, distance_ratio=0.75,
                                                  use_ransac=True, ransac_threshold=5.0,
                                                  border_crop=0):
        """Headless keypoint alignment constrained to scale + translation only.

        Mirrors :meth:`auto_keypoints_headless` but estimates an isotropic
        scale + translation (no rotation) via
        :func:`optics.estimate_scale_translation`, used by the Focusing tool
        where the two objectives share orientation. Returns the pairs used.
        """
        if self.template_image is None or self.moving_image is None:
            QMessageBox.warning(self, "Warning", "Load both images first")
            return 0

        moving_img_for_detection = (
            self.transformed_image if self.transformed_image is not None
            else self.moving_image
        )
        pairs = detect_keypoint_pairs(
            self.template_image, moving_img_for_detection,
            detector=detector, matcher=matcher, max_features=max_features,
            distance_ratio=distance_ratio, use_ransac=use_ransac,
            ransac_threshold=ransac_threshold, border_crop=border_crop,
        )
        if len(pairs) == 0:
            self.statusBar().showMessage("Auto keypoints: no pairs detected (transform unchanged).")
            return 0

        matrix = estimate_scale_translation(pairs)
        self.opt_transform = matrix
        transform = tf.AffineTransform(matrix=matrix)
        if self.current_transform is not None:
            matrix = np.dot(transform.params, self.current_transform.params)
            transform = tf.AffineTransform(matrix=matrix)
        self.onViewModeChanged("overlay")
        self.transform_controls.set_values_from_transform(transform.params)
        self.statusBar().showMessage(
            f"Auto keypoints scale+translation (headless): from {len(pairs)} pairs.")
        return len(pairs)

    def apply_transform(self, params):
        """Apply transformation to moving image"""
        if self.moving_image is None:
            return

        # # Build transformation matrix
        transform = tf.SimilarityTransform(scale=params['scale'],
                                           rotation=np.radians(params['rotation']),
                                           translation=[params['tx'], params['ty']])

        self.current_transform = transform

        print(f"Transform matrix (apply_transform):\n{transform.params}")
        # Apply transformation (+ optional distortion warp on top)
        self.transformed_image = self._warp_with_distortion(
            self.moving_image,
            self.template_image.shape if self.template_image is not None else self.moving_image.shape,
        )

        self.update_display()

    def _warp_with_distortion(self, image, output_shape, transform=None):
        """Warp ``image`` into the template frame, honouring ``distortion_transform``.

        ``current_transform`` maps moving->template; its ``.inverse`` is the
        usual ``tf.warp`` inverse_map (output/template coords -> moving coords).
        ``distortion_transform`` is a **residual** template->template remap (the
        non-rigid part after the linear fit), so the two **compose**:
        ``inverse_map(coords) = current_transform.inverse(distortion(coords))``.
        This keeps the linear transform live -- editing scale / rotation / tx /
        ty in the controls still re-warps correctly while the distortion rides on
        top. Used by display and every export path.
        """
        transform = transform if transform is not None else self.current_transform
        if transform is None:
            return None
        distortion = self.distortion_transform
        if distortion is None:
            inverse_map = transform.inverse
        else:
            def inverse_map(coords):
                return transform.inverse(distortion(coords))
        return tf.warp(
            image, inverse_map, output_shape=output_shape, preserve_range=True
        ).astype(np.float32)

    def _apply_keypoint_distortion(self, pairs, linear_matrix):
        """Estimate a distortion warp from keypoint pairs if the dialog asked.

        Fits the **residual** non-rigid remap left over after the linear keypoint
        fit ``linear_matrix`` (the incremental moving->template estimate from
        these pairs) via :func:`optics.estimate_distortion`, and stores it as
        ``distortion_transform``. Because it is a residual, it **composes** with
        the live ``current_transform`` in :meth:`_warp_with_distortion` (so the
        scale / rotation / tx / ty in the controls remain effective). Cleared on
        failure / when not requested.
        """
        dialog = self.auto_keypoints_dialog
        if dialog is None or not hasattr(dialog, "distortion_request"):
            self.distortion_transform = None
            return
        enabled, model = dialog.distortion_request()
        self._estimate_keypoint_distortion(pairs, linear_matrix,
                                           model if enabled else None, warn=True,
                                           show_quality=enabled)

    def _estimate_keypoint_distortion(self, pairs, linear_matrix, model, warn=False,
                                      show_quality=False):
        """Set ``distortion_transform`` to the residual warp for ``model`` (or None).

        Shared by the dialog path and the headless/batch path. ``model`` None
        clears it; <3 pairs skips (warning optional). The residual composes with
        the live linear transform in :meth:`_warp_with_distortion`. When
        ``show_quality`` is set, a fit-quality popup (RMS / R² + deformation
        grid) is shown after estimation."""
        self.distortion_transform = None
        if not model:
            return
        if len(pairs) < 3:
            if warn:
                QMessageBox.warning(self, "Distortion",
                                    f"Need >= 3 pairs for distortion ({len(pairs)} found); skipped.")
            return
        shape = self.template_image.shape if self.template_image is not None else self.moving_image.shape
        prev_distortion = self.distortion_transform
        try:
            new_distortion = estimate_distortion(
                pairs, shape, model=model, residual_matrix=linear_matrix)
        except Exception as e:
            if warn:
                QMessageBox.warning(self, "Distortion", f"Distortion estimate failed: {e}")
            self.distortion_transform = None
            return
        self.distortion_transform = new_distortion
        if show_quality:
            # Sanity check (#2c): image correlation before vs after the correction.
            corr_before = self._distortion_corr(use_distortion=False)
            corr_after = self._distortion_corr(use_distortion=True)

            def _cancel():
                self.distortion_transform = prev_distortion
                self.apply_transform(self.transform_controls.transform_params)
                self.statusBar().showMessage("Distortion correction cancelled.")

            # Detection params from the keypoint dialog (for the popup listing #2).
            params = None
            dlg = self.auto_keypoints_dialog
            if dlg is not None:
                params = {"detector": dlg.detection_method.currentText(),
                          "matcher": dlg.matching_method.currentText(),
                          "distance_ratio": round(dlg.distance_ratio.value(), 2),
                          "ransac_threshold": dlg.ransac_threshold.value(),
                          "border_crop": dlg.border_crop_px(),
                          "max_features": dlg.max_features.value()}
            show_distortion_fit_quality(
                self, self.distortion_transform, pairs, shape, model,
                residual_matrix=linear_matrix, corr_before=corr_before,
                corr_after=corr_after, on_cancel=_cancel, params=params)

    def _distortion_corr(self, use_distortion):
        """NCC of the template vs the moving warped with/without the current
        distortion (for the fit-quality sanity check). Returns nan on failure."""
        if self.template_image is None or self.moving_image is None:
            return float("nan")
        saved = self.distortion_transform
        if not use_distortion:
            self.distortion_transform = None
        try:
            warped = self._warp_with_distortion(self.moving_image, self.template_image.shape)
        finally:
            self.distortion_transform = saved
        if warped is None:
            return float("nan")
        a = self.template_image.astype(np.float64).ravel()
        b = warped.astype(np.float64).ravel()
        m = (a != 0) & (b != 0)  # ignore the zero borders
        if m.sum() < 16 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
            return float("nan")
        return float(np.corrcoef(a[m], b[m])[0, 1])

    def reset_transformation(self):
        """Reset the linear transform and distortion, but KEEP propagation.

        Propagation lives in ``moving_field``/``moving_z_um`` (and the template
        equivalents): the propagated wavefront is the image the transform acts
        on, so a transform reset leaves the current propagation distances in
        place (use 'Reset distance' per target to undo propagation)."""
        self.distortion_transform = None
        self.transform_controls.reset_transform()  # emits transform_changed -> apply_transform

    def update_display(self):
        """Update image display with current settings"""
        if self.template_image is not None:
            self.canvas.set_template(
                self.template_image,
                self.template_color.currentText(),
                self.template_opacity.value() / 100
            )
        if self.transformed_image is not None:
            self.canvas.set_moving(
                self.transformed_image,
                self.moving_color.currentText(),
                self.moving_opacity.value() / 100
            )
        elif self.moving_image is not None:
            self.canvas.set_moving(
                self.moving_image,
                self.moving_color.currentText(),
                self.moving_opacity.value() / 100
            )

    def optimize_alignment(self):
        """Optimize alignment using selected method"""
        if self.template_image is None or self.moving_image is None:
            QMessageBox.warning(self, "Warning", "Please load both images first")
            return
            
        method = self.opt_method.currentText()
        
        if method == "Manual Pairs of Points":
            self.open_keypoints_tool()
        elif method == "Auto Detect Keypoints":
            self.open_auto_keypoints_tool()
        elif method == "Phase Cross-Correlation":
            self.optimize_phase_correlation()
        elif method == "Brute Force":
            self.optimize_brute_force()
        elif method == "Distortion Correction":
            self.open_distortion_tool()
        elif method == "Change Distance":
            self.open_distance_tool()
        else:
            self.optimize_enhanced_correlation()
            
    def optimize_phase_correlation(self):
        """Use phase cross-correlation for optimization"""
        # Get current transform as starting point
        current = self.transform_controls.transform_params
        
        moving_mask = np.ones_like(self.moving_image, dtype=bool)
        print(moving_mask)
        # Apply initial transform
        tform = tf.AffineTransform(scale=current['scale'],
                                      rotation=np.radians(current['rotation']),
                                      translation=[current['tx'], current['ty']]
        )
        
        transformed = tf.warp(
            self.moving_image,
            tform.inverse,
            output_shape=self.template_image.shape,
            preserve_range=True
        )
        moving_mask = tf.warp(
            moving_mask.astype(np.float32),
            tform.inverse,
            output_shape=self.template_image.shape,
            preserve_range=True
        ) > 0.5
        print(moving_mask)
        # Find translation using phase correlation
        shift, _, _ = phase_cross_correlation(
            self.template_image,
            transformed,
            reference_mask=np.ones_like(self.template_image, dtype=bool),
            moving_mask=moving_mask,
            upsample_factor=10
        )
        print(f"Phase correlation shift: {shift}")
        # Update translation
        new_params = current.copy()
        print("Current params before update:", current)
        new_params['ty'] = current['ty'] + shift[0]
        new_params['tx'] = current['tx'] + shift[1]
        
        # Apply optimized transform
        self.transform_controls.set_values_from_params(new_params)
        self.statusBar().showMessage("Optimization complete (Phase Correlation - Can capture translation only)")
        
    def optimize_brute_force(self):
        """Brute force optimization with user-defined ranges"""
        dialog = BruteForceDialog(self, current_params=self.transform_controls.transform_params)
        self.set_optimizer_dialog_state(True)
        result = dialog.exec()
        self.set_optimizer_dialog_state(False)
        if result == QDialog.DialogCode.Accepted:
            ranges = dialog.get_ranges()
            
            # Create progress dialog
            progress = QProgressDialog("Optimizing alignment from current position", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            
            best_score = -np.inf
            best_params = self.transform_controls.transform_params.copy()
            
            # Calculate total iterations
            total = (ranges['rot_steps'] * ranges['scale_steps'] * 
                    ranges['tx_steps'] * ranges['ty_steps'])
            current_iter = 0
            
            for rot in np.linspace(ranges['rot_min'], ranges['rot_max'], ranges['rot_steps']):
                for scale in np.linspace(ranges['scale_min'], ranges['scale_max'], ranges['scale_steps']):
                    for tx in np.linspace(ranges['tx_min'], ranges['tx_max'], ranges['tx_steps']):
                        for ty in np.linspace(ranges['ty_min'], ranges['ty_max'], ranges['ty_steps']):
                            if progress.wasCanceled():
                                return
                                
                            # Test these parameters
                            params = {
                                'rotation': rot,
                                'scale': scale,
                                'tx':  tx,
                                'ty':  ty
                            }
                            
                            # Apply transform
                            transform = tf.SimilarityTransform(translation=[params['tx'], params['ty']], 
                                                               scale=params['scale'], 
                                                               rotation=np.radians(params['rotation']))

                            transformed = tf.warp(
                                self.moving_image,
                                transform.inverse,
                                output_shape=self.template_image.shape,
                                preserve_range=True
                            )
                            
                            # Calculate correlation
                            score = np.corrcoef(self.template_image.flatten(), transformed.flatten())[0, 1]
                            
                            if score > best_score:
                                best_score = score
                                best_params = params
                                print(f"New best score: {best_score:.4f} with params: {best_params}")
                                
                                
                            current_iter += 1
                            progress.setValue(int(100 * current_iter / total))
                            QApplication.processEvents()
                            
            # Apply best parameters
            self.transform_controls.set_transform(best_params)
            self.statusBar().showMessage(f"Optimization complete (Brute Force). Score: {best_score:.3f}")
            
    def optimize_enhanced_correlation(self):
        """Enhanced correlation using multi-scale approach"""
        if self.template_image is None or self.moving_image is None:
            return
            
        # Multi-scale optimization
        scales = [0.25, 0.5, 1.0]
        current_params = self.transform_controls.transform_params.copy()
        
        for scale_factor in scales:
            # Downsample images
            template_scaled = cv2.resize(
                self.template_image,
                None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_AREA
            )
            moving_scaled = cv2.resize(
                self.moving_image,
                None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_AREA
            )
            
            # Apply current transform
            angle_rad = np.radians(current_params['rotation'])
            s = current_params['scale']
            
            M = cv2.getRotationMatrix2D(
                (moving_scaled.shape[1]/2, moving_scaled.shape[0]/2),
                current_params['rotation'],
                s
            )
            M[0, 2] += current_params['tx'] * scale_factor
            M[1, 2] += current_params['ty'] * scale_factor
            
            transformed = cv2.warpAffine(
                moving_scaled,
                M,
                (template_scaled.shape[1], template_scaled.shape[0])
            )
            
            # Template matching for fine adjustment
            result = cv2.matchTemplate(template_scaled, transformed, cv2.TM_CCORR_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            
            # Update parameters
            if scale_factor < 1.0:
                current_params['tx'] += (max_loc[0] - template_scaled.shape[1]/2) / scale_factor * 0.5
                current_params['ty'] += (max_loc[1] - template_scaled.shape[0]/2) / scale_factor * 0.5
                
        self.transform_controls.set_transform(current_params)
        self.statusBar().showMessage("Optimization complete (Enhanced Correlation)")

    def on_template_drag(self, dx, dy):
        """Handle canvas drag events by nudging translation parameters."""
        if (self.template_image is None or self.moving_image is None or
                not self.canvas.template_drag_enabled):
            return
        # Dragging template visually corresponds to moving the transform opposite direction
        self.transform_controls.nudge_translation(dx, dy)

    def update_canvas_drag_state(self):
        allow = (
            self.template_image is not None and
            self.moving_image is not None and
            not self.optimizer_dialog_open and
            getattr(self.canvas, 'view_mode', 'overlay') == 'overlay'
        )
        self.canvas.set_template_drag_enabled(allow)

    def set_optimizer_dialog_state(self, active: bool):
        self.optimizer_dialog_open = active
        self.update_canvas_drag_state()

    def save_transform_matrix(self):
        """Export the tranformation matrix as a float 32 Text file"""
        if self.transformed_image is None:
            QMessageBox.warning(self, "Warning", "No transformed image to export")
            return
            
        file_path, filetype_ext = QFileDialog.getSaveFileName(
            self, "Save Transformation", "", "Text File (*.txt)"
        )
        if file_path and not file_path.endswith('.txt'):
            file_path += '.txt'
        
        if file_path and self.current_transform is not None:
            # Save matrix
            matrix = self.current_transform.params.astype(np.float32)
            np.savetxt(file_path, matrix, delimiter=',', fmt='%.6f')
            self.statusBar().showMessage(f"Transformation matrix saved to: {Path(file_path).name}")        

    def load_transform_matrix(self):
        """Load a transformation matrix from a Text file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Transformation", "", "Text File (*.txt)"
        )
        if file_path and not file_path.endswith('.txt'):
            file_path += '.txt'
        if file_path:
            try:
                matrix = np.loadtxt(file_path, delimiter=',').astype(np.float32)
                if matrix.shape != (3, 3):
                    raise ValueError("Invalid matrix shape")
                    
                transform = tf.AffineTransform(matrix=matrix)
                self.current_transform = transform
                self.transform_controls.set_values_from_transform(matrix)
                self.apply_transform(self.transform_controls.transform_params)
                self.statusBar().showMessage(f"Loaded transformation from: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load transformation: {e}")

    def export_image(self):
        """Export transformed image"""
        if self.transformed_image is None:
            QMessageBox.warning(self, "Warning", "No transformed image to export")
            return
            
        file_path, filetype_ext = QFileDialog.getSaveFileName(
            self, "Save Transformed Image", "", "TIF Files (*.tif);;TIFF Files (*.tiff);;PNG Files (*.png)"
        )
        
        # We re-apply the transform to ensure full resolution export and the same output shape
        img_to_save = None
        if self.current_transform is not None and self.moving_image is not None:
            img_to_save = tf.warp(
                self.moving_image,
                self.current_transform.inverse,
                output_shape=self.template_image.shape if self.template_image is not None else self.moving_image.shape,
                preserve_range=True
            ).astype(np.float32)

        if file_path and img_to_save is not None:
            # Save image
            if file_path.endswith('.tif') or file_path.endswith('.tiff'):
                tifffile.imwrite(file_path, img_to_save.astype(np.float32))
            elif filetype_ext in ['TIF Files (*.tif)', 'TIFF Files (*.tiff)']:
                extension = '.tif' if filetype_ext == 'TIF Files (*.tif)' else '.tiff'
                tifffile.imwrite(file_path + extension, img_to_save.astype(np.float32))
            elif file_path.endswith('.png'):
                img_normalized = ((img_to_save - img_to_save.min()) / 
                                 (img_to_save.max() - img_to_save.min() + 1e-10) * 255).astype(np.uint8)
                Image.fromarray(img_normalized).save(file_path)
            elif filetype_ext in ['PNG Files (*.png)']:
                img_normalized = ((img_to_save - img_to_save.min()) / 
                                 (img_to_save.max() - img_to_save.min() + 1e-10) * 255).astype(np.uint8)
                Image.fromarray(img_normalized).save(file_path + '.png')
            else:
                QMessageBox.warning(self, "Warning", "Unsupported file format")
                return
                
            self.statusBar().showMessage(f"Exported to: {Path(file_path).name}")

    def _warp_moving_for_export(self):
        """Re-warp the moving image at full resolution for export, or None."""
        if self.current_transform is None or self.moving_image is None:
            return None
        return self._warp_with_distortion(
            self.moving_image,
            self.template_image.shape if self.template_image is not None else self.moving_image.shape,
        )

    def save_image_to_folder(self, folder, suffix=""):
        """Save the transformed moving image to ``folder`` keeping its original
        name plus ``suffix`` and extension (no dialog). For batch mode."""
        img_to_save = self._warp_moving_for_export()
        if img_to_save is None:
            QMessageBox.warning(self, "Warning", "No transformed image to export")
            return
        out_dir = Path(folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        src = Path(self.moving_image_file) if self.moving_image_file else Path("moving.tif")
        ext = src.suffix.lower() if src.suffix else ".tif"
        out_path = out_dir / f"{src.stem}{suffix}{ext}"

        if ext in ('.tif', '.tiff'):
            tifffile.imwrite(str(out_path), img_to_save.astype(np.float32))
        else:  # png/jpg -> normalize to 8-bit
            img_normalized = ((img_to_save - img_to_save.min()) /
                              (img_to_save.max() - img_to_save.min() + 1e-10) * 255).astype(np.uint8)
            Image.fromarray(img_normalized).save(str(out_path))
        self.statusBar().showMessage(f"Exported to: {out_path.name}")

    def load_and_export_stack(self):
        """Apply the current transform to every frame of an ImageJ wavefront
        stack and save the result with the same channel layout."""
        if self.current_transform is None:
            QMessageBox.warning(self, "Warning", "Please set up a transformation first")
            return

        in_path, _ = QFileDialog.getOpenFileName(
            self, "Load ImageJ Stack", "", "TIFF Files (*.tif *.tiff)"
        )
        if not in_path:
            return
        self.export_stack_from_path(in_path)

    def export_stack_from_path(self, in_path):
        """Apply the current transform to every frame of the given ImageJ wavefront
        stack file and save the result (prompts only for the output path)."""
        if self.current_transform is None:
            QMessageBox.warning(self, "Warning", "Please set up a transformation first")
            return

        out_path, filetype_ext = QFileDialog.getSaveFileName(
            self, "Save Transformed Stack", "", "TIFF Files (*.tif *.tiff)"
        )
        if not out_path:
            return
        if not (out_path.endswith('.tif') or out_path.endswith('.tiff')):
            out_path += '.tif'
        self._process_and_save_stack(in_path, out_path)

    def export_stack_to_folder(self, in_path, folder, suffix=""):
        """Apply the current transform to the stack at ``in_path`` and save it to
        ``folder`` as ``<stem><suffix>.tif`` (no dialog). For batch mode."""
        if self.current_transform is None:
            QMessageBox.warning(self, "Warning", "Please set up a transformation first")
            return
        out_dir = Path(folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(in_path).stem
        out_path = str(out_dir / f"{stem}{suffix}.tif")
        self._process_and_save_stack(in_path, out_path)

    def _process_and_save_stack(self, in_path, out_path):
        """Warp every frame of the wavefront stack at ``in_path`` and write ``out_path``."""
        try:
            _, _, n_frames = load_wavefront_tif(in_path, frame_index=0)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load stack: {e}")
            return

        ref_shape = (
            self.template_image.shape if self.template_image is not None else None
        )

        progress = QProgressDialog(
            "Processing stack frames...", "Cancel", 0, n_frames, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        frames = []
        for t in range(n_frames):
            if progress.wasCanceled():
                self.statusBar().showMessage("Stack export canceled")
                return
            try:
                phase, amp, _ = load_wavefront_tif(in_path, frame_index=t)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read frame {t}: {e}")
                return

            output_shape = ref_shape if ref_shape is not None else phase.shape
            phase_t = self._warp_with_distortion(phase, output_shape)
            amp_t = self._warp_with_distortion(amp, output_shape)

            # Channel order matches load_wavefront_tif: 0=phase, 1=amplitude
            frames.append(np.stack([phase_t, amp_t], axis=-1))
            progress.setValue(t + 1)
            QApplication.processEvents()

        stack = np.stack(frames, axis=0)  # (N, H, W, 2)

        try:
            save_stack(out_path, stack, source_path=in_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save stack: {e}")
            return

        self.statusBar().showMessage(
            f"Exported transformed stack ({n_frames} frames) to: {Path(out_path).name}"
        )

    def export_propagated_stack_to_folder(self, in_path, folder, suffix, optics,
                                          z_tpl_um=0.0, z_mov_um=0.0,
                                          template_path=None, save_template=True):
        """Propagate + transform + distort a moving stack, write it (and optionally
        the propagated template) to ``folder``.

        For the **moving** stack at ``in_path``: each frame's complex field is
        ASM-propagated by ``z_mov_um`` (at the moving sample-plane pixel size),
        then warped into the template frame by the current transform + distortion
        (channel-consistent via :func:`optics.warp_field_with_distortion`), split
        back to phase+amp, saved as ``<stem><suffix>.tif``.

        When ``save_template`` and ``template_path`` are given, the **template**
        stack is ASM-propagated by ``z_tpl_um`` (no transform -- it is the
        reference) and written as ``<tpl_stem><suffix>_template.tif``.

        ``optics`` = ``{mag_tpl, na_tpl, mag_mov, na_mov, wavelength_nm,
        camera_pixel_um, n}``. Used by Batch Mode's propagation-map save.
        """
        out_dir = Path(folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        cam = float(optics.get("camera_pixel_um", 5.86)) * 1e-6
        lam = float(optics.get("wavelength_nm", 660.0)) * 1e-9
        n = float(optics.get("n", 1.0))
        px_mov = sample_pixel_size(cam, optics.get("mag_mov", 10.0))
        px_tpl = sample_pixel_size(cam, optics.get("mag_tpl", 20.0))
        out_shape = self.template_image.shape if self.template_image is not None else None
        gm = self.current_transform.params if self.current_transform is not None else np.eye(3)
        distortion = self.distortion_transform

        def _save_one(path, z_um, px, warp):
            try:
                _p, _a, n_frames = load_wavefront_tif(path, frame_index=0)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load stack {Path(path).name}: {e}")
                return None
            shape = out_shape if (warp and out_shape is not None) else None
            frames = []
            for t in range(n_frames):
                phase, amp, _ = load_wavefront_tif(path, frame_index=t)
                field = field_from_phase_amp(phase, amp)
                field = propagate_asm(field, z_um * 1e-6, lam, px, n=n)
                if warp:
                    field = warp_field_with_distortion(
                        field, gm, distortion, shape if shape is not None else field.shape)
                ph, am = phase_amp_from_field(field, unwrap=False)
                frames.append(np.stack([ph, am], axis=-1))
            return np.stack(frames, axis=0)  # (N, H, W, 2)

        mov_stack = _save_one(in_path, z_mov_um, px_mov, warp=True)
        if mov_stack is None:
            return
        mov_out = str(out_dir / f"{Path(in_path).stem}{suffix}.tif")
        save_stack(mov_out, mov_stack, source_path=in_path,
                   metas={"optics": optics, "z_mov_um": z_mov_um, "z_tpl_um": z_tpl_um})
        saved = [Path(mov_out).name]

        if save_template and template_path:
            tpl_stack = _save_one(template_path, z_tpl_um, px_tpl, warp=False)
            if tpl_stack is not None:
                tpl_out = str(out_dir / f"{Path(template_path).stem}{suffix}_template.tif")
                save_stack(tpl_out, tpl_stack, source_path=template_path,
                           metas={"optics": optics, "z_tpl_um": z_tpl_um})
                saved.append(Path(tpl_out).name)

        self.statusBar().showMessage(f"Exported propagated stack(s): {', '.join(saved)}")

    def batch_process(self):
        """Apply transformation to a folder of images"""
        if self.current_transform is None:
            QMessageBox.warning(self, "Warning", "Please set up a transformation first")
            return
            
        # Select input folder
        input_folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if not input_folder:
            return
            
        # Select output folder
        output_folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_folder:
            return
            
        # Find all image files
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        image_files = list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff")) + \
                     list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
                     
        if not image_files:
            QMessageBox.warning(self, "Warning", "No image files found in selected folder")
            return
            
        # Process images with progress dialog
        progress = QProgressDialog("Processing images...", "Cancel", 0, len(image_files), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        for i, img_file in enumerate(image_files):
            if progress.wasCanceled():
                break
                
            # Load image
            img = load_imgfile(str(img_file)).astype(np.float32)
            
            # Apply transformation
            transformed = tf.warp(
                img,
                self.current_transform.inverse,
                output_shape=self.template_image.shape if self.template_image is not None else img.shape,
                preserve_range=True
            ).astype(np.float32)
            
            # Save transformed image
            output_file = output_path / f"transformed_{img_file.name}"
            if img_file.suffix in ['.tif', '.tiff']:
                tifffile.imwrite(str(output_file), transformed)
            else:
                img_normalized = ((transformed - transformed.min()) / 
                                (transformed.max() - transformed.min() + 1e-10) * 255).astype(np.uint8)
                Image.fromarray(img_normalized).save(str(output_file))
                
            progress.setValue(i + 1)
            QApplication.processEvents()
            
        self.statusBar().showMessage(f"Batch processing complete. Processed {len(image_files)} images")

    def open_batch_mode(self):
        """Open the Batch Mode panel that drives this window across many images."""
        from batchMode import BatchModePanel
        if self.batch_panel is None:
            self.batch_panel = BatchModePanel(self)
        self.batch_panel.show()
        self.batch_panel.raise_()
        self.batch_panel.activateWindow()

    def open_progressive_folder(self):
        """Open the Progressive Folder Alignment panel that drives this window."""
        from progressiveFolderAlignment import ProgressiveFolderPanel
        if self.progressive_panel is None:
            self.progressive_panel = ProgressiveFolderPanel(self)
        self.progressive_panel.show()
        self.progressive_panel.raise_()
        self.progressive_panel.activateWindow()

    def open_focusing_tool(self):
        """Open the Wavefront Focusing & Alignment panel that drives this window."""
        from focusingTool import FocusingPanel
        if self.focusing_panel is None:
            self.focusing_panel = FocusingPanel(self)
        self.focusing_panel.show()
        self.focusing_panel.raise_()
        self.focusing_panel.activateWindow()
  