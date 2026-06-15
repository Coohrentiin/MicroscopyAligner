# Copyright (c) 2025 Corentin Soubeiran
# SPDX-License-Identifier: MIT
"""Batch Mode panel.

A separate window that drives the main :class:`ImageAligner` window across many
images.  The user lays out, in a table, several *sequences* (rows); each row has
a template image and one or more moving images.

Action sequences are keyed **per moving column** (``Moving 1``, ``Moving 2`` ...)
and are shared by every row's image in that column.  Selecting any cell of a
moving column shows / edits that column's action sequence; a "Copy from column"
button copies another column's sequence into the current one.

Navigation buttons step through the actions / moving images / sequences while
the panel triggers the real alignment and export code on the main window.

The transform is intentionally kept across moving images and rows; only an
explicit ``reset`` action clears it.
"""
import json
from pathlib import Path

from natsort import natsorted

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QTableWidget,
    QTableWidgetItem, QListWidget, QComboBox, QDoubleSpinBox, QSpinBox, QLabel,
    QFileDialog, QMessageBox, QHeaderView, QCheckBox, QLineEdit,
)
from PySide6.QtCore import Qt


IMAGE_FILTER = "Image Files (*.tif *.tiff *.png *.jpg);;TIFF Files (*.tif *.tiff)"

# Action type -> human readable label shown in the action editor combo box.
ACTION_TYPES = [
    ("set_matrix", "Set matrix values (scale / rotation / tx / ty)"),
    ("auto_keypoints", "Auto keypoint detection"),
    ("cross_correlation", "Cross-correlation alignment"),
    ("reset", "Reset transform"),
    ("save_image", "Save moving image"),
    ("save_stack", "Save on stack"),
]
ACTION_LABELS = dict(ACTION_TYPES)


def _action_summary(action):
    """One-line readable description of an action dict (for the list widget)."""
    t = action.get("type")
    if t == "set_matrix":
        return (
            f"Set matrix  scale={action.get('scale', 1.0):g}  "
            f"rot={action.get('rotation', 0.0):g}  "
            f"tx={action.get('tx', 0.0):g}  ty={action.get('ty', 0.0):g}"
        )
    if t == "auto_keypoints":
        if action.get("dialog", True):
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
                f"{action.get('matcher', 'Brute Force')}, RANSAC={action.get('ransac_threshold', 5.0):g}{extra_txt})")
    if t in ("save_image", "save_stack"):
        label = ACTION_LABELS.get(t, t)
        folder = action.get("folder")
        if folder:
            suffix = action.get("suffix", "")
            return f"{label} -> {folder} (name{('+' + suffix) if suffix else ''})"
        return f"{label} (prompt)"
    return ACTION_LABELS.get(t, t)


class BatchModePanel(QWidget):
    """Top-level window driving an :class:`ImageAligner` instance.

    Data model
    ----------
    ``self.sequences``: list of rows, each ``{"template_path": str,
    "moving_paths": [str, ...]}`` (image paths only, variable length).

    ``self.column_actions``: list indexed by moving column (0-based, i.e. column
    0 == "Moving 1"), each an ordered list of action dicts shared by every row's
    image in that column.
    """

    def __init__(self, aligner):
        super().__init__()  # not parented to the aligner -> own top-level window
        self.aligner = aligner
        self.setWindowTitle("Batch Mode")
        self.resize(900, 700)

        # Image table model (paths only)
        self.sequences = [self._new_sequence()]
        # Per-column action sequences (index 0 == "Moving 1")
        self.column_actions = [[]]

        # Navigation state
        self.current_row = 0
        self.current_moving = 0     # also the active moving-column index
        self.current_action = 0

        # Which moving column the editor is bound to
        self.edit_column = 0

        self._init_ui()
        self._rebuild_table_from_model()
        self.table.setCurrentCell(0, 1)  # select first moving cell
        self._populate_action_editor()
        self._update_status()

    # ------------------------------------------------------------------ model
    @staticmethod
    def _new_sequence():
        return {"template_path": "", "moving_paths": [""]}

    def _n_moving_columns(self):
        """Number of moving columns = len(column_actions), kept >= 1."""
        return max(len(self.column_actions), 1)

    def _ensure_columns(self, n):
        """Ensure column_actions and every row's moving_paths have >= n columns."""
        while len(self.column_actions) < n:
            self.column_actions.append([])
        for s in self.sequences:
            while len(s["moving_paths"]) < n:
                s["moving_paths"].append("")

    # --------------------------------------------------------------------- UI
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_table_section(), stretch=3)
        layout.addWidget(self._build_action_section(), stretch=2)
        layout.addWidget(self._build_nav_section())

    def _build_table_section(self):
        box = QGroupBox("1. Sequences  (row = one template + its moving images)")
        v = QVBoxLayout(box)

        self.table = QTableWidget(0, 2)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self._browse_cell)
        self.table.currentCellChanged.connect(self._on_current_cell_changed)
        v.addWidget(self.table)

        btn_row = QHBoxLayout()
        add_seq = QPushButton("Add Sequence")
        add_seq.clicked.connect(self._add_sequence)
        rem_seq = QPushButton("Remove Sequence")
        rem_seq.clicked.connect(self._remove_sequence)
        add_col = QPushButton("+ Moving Column")
        add_col.clicked.connect(self._add_moving_column)
        browse = QPushButton("Browse Cell...")
        browse.clicked.connect(self._browse_selected_cell)
        for b in (add_seq, rem_seq, add_col, browse):
            btn_row.addWidget(b)
        btn_row.addStretch()
        v.addLayout(btn_row)

        hint = QLabel("Double-click a cell (or select it and use 'Browse Cell') to assign image file(s). "
                      "Selecting multiple files fills this column downward from the clicked row "
                      "(natsorted, adding rows as needed). Select a moving cell to edit that column's action sequence.")
        hint.setStyleSheet("color: gray;")
        hint.setWordWrap(True)
        v.addWidget(hint)
        return box

    def _build_action_section(self):
        box = QGroupBox("2. Action sequence")
        v = QVBoxLayout(box)

        self.action_header = QLabel()
        self.action_header.setStyleSheet("font-weight: bold;")
        v.addWidget(self.action_header)

        self.action_list = QListWidget()
        v.addWidget(self.action_list)

        # Action-type chooser
        ctrl = QHBoxLayout()
        self.action_combo = QComboBox()
        for key, label in ACTION_TYPES:
            self.action_combo.addItem(label, key)
        self.action_combo.currentIndexChanged.connect(self._update_option_visibility)
        ctrl.addWidget(self.action_combo)
        ctrl.addStretch()
        add_act = QPushButton("Add Action")
        add_act.clicked.connect(self._add_action)
        ctrl.addWidget(add_act)
        v.addLayout(ctrl)

        # --- set_matrix options ---
        self.matrix_row = QHBoxLayout()
        self.scale_spin = QDoubleSpinBox(); self.scale_spin.setRange(0.01, 100.0); self.scale_spin.setValue(1.0); self.scale_spin.setPrefix("scale ")
        self.rot_spin = QDoubleSpinBox(); self.rot_spin.setRange(-360.0, 360.0); self.rot_spin.setValue(0.0); self.rot_spin.setPrefix("rot ")
        self.tx_spin = QDoubleSpinBox(); self.tx_spin.setRange(-100000.0, 100000.0); self.tx_spin.setValue(0.0); self.tx_spin.setPrefix("tx ")
        self.ty_spin = QDoubleSpinBox(); self.ty_spin.setRange(-100000.0, 100000.0); self.ty_spin.setValue(0.0); self.ty_spin.setPrefix("ty ")
        self.matrix_spins = [self.scale_spin, self.rot_spin, self.tx_spin, self.ty_spin]
        for s in self.matrix_spins:
            self.matrix_row.addWidget(s)
        self.matrix_row.addStretch()
        v.addLayout(self.matrix_row)

        # --- auto_keypoints options ---
        self.auto_row = QHBoxLayout()
        self.ak_use_dialog = QCheckBox("Open dialog")
        self.ak_use_dialog.setChecked(True)
        self.ak_use_dialog.toggled.connect(self._update_option_visibility)
        self.auto_row.addWidget(self.ak_use_dialog)
        self.ak_detector = QComboBox(); self.ak_detector.addItems(["AKAZE", "KAZE", "SIFT", "ORB", "BRISK"])
        self.ak_matcher = QComboBox(); self.ak_matcher.addItems(["Brute Force", "FLANN"])
        self.ak_ransac = QDoubleSpinBox(); self.ak_ransac.setRange(0.5, 20.0); self.ak_ransac.setValue(5.0); self.ak_ransac.setPrefix("RANSAC ")
        # Constraints (lock rotation / scale) + optional residual distortion.
        self.ak_lock_rotation = QCheckBox("Lock rotation")
        self.ak_lock_scale = QCheckBox("Lock scale")
        self.ak_use_distortion = QCheckBox("Distortion")
        self.ak_use_distortion.toggled.connect(self._update_option_visibility)
        self.ak_distortion_model = QComboBox(); self.ak_distortion_model.addItems(["tps", "poly", "radial", "piecewise"])
        self.ak_headless_widgets = [self.ak_detector, self.ak_matcher, self.ak_ransac,
                                    self.ak_lock_rotation, self.ak_lock_scale,
                                    self.ak_use_distortion, self.ak_distortion_model]
        for w in self.ak_headless_widgets:
            self.auto_row.addWidget(w)
        self.auto_row.addStretch()
        v.addLayout(self.auto_row)

        # --- save options (save_image / save_stack) ---
        self.save_row = QHBoxLayout()
        self.save_use_folder = QCheckBox("Save to folder")
        self.save_use_folder.toggled.connect(self._update_option_visibility)
        self.save_row.addWidget(self.save_use_folder)
        self.save_folder_edit = QLineEdit(); self.save_folder_edit.setPlaceholderText("output folder...")
        self.save_row.addWidget(self.save_folder_edit, stretch=1)
        self.save_folder_btn = QPushButton("Browse...")
        self.save_folder_btn.clicked.connect(self._browse_save_folder)
        self.save_row.addWidget(self.save_folder_btn)
        self.save_suffix_edit = QLineEdit(); self.save_suffix_edit.setPlaceholderText("suffix (e.g. _aligned)")
        self.save_suffix_edit.setMaximumWidth(160)
        self.save_row.addWidget(self.save_suffix_edit)
        v.addLayout(self.save_row)

        edit_row = QHBoxLayout()
        rem_act = QPushButton("Remove Action")
        rem_act.clicked.connect(self._remove_action)
        up = QPushButton("Move Up")
        up.clicked.connect(lambda: self._move_action(-1))
        down = QPushButton("Move Down")
        down.clicked.connect(lambda: self._move_action(1))
        edit_row.addWidget(rem_act)
        edit_row.addWidget(up)
        edit_row.addWidget(down)
        edit_row.addSpacing(20)
        edit_row.addWidget(QLabel("Copy from:"))
        self.copy_from_combo = QComboBox()
        edit_row.addWidget(self.copy_from_combo)
        copy_btn = QPushButton("Copy from column")
        copy_btn.clicked.connect(self._copy_from_column)
        edit_row.addWidget(copy_btn)
        edit_row.addStretch()
        v.addLayout(edit_row)

        self._update_option_visibility()
        return box

    def _build_nav_section(self):
        box = QGroupBox("3. Navigation")
        v = QVBoxLayout(box)

        nav = QHBoxLayout()
        prev_act = QPushButton("< Previous Action")
        prev_act.clicked.connect(self._previous_action)
        run_all = QPushButton("Run All Actions")
        run_all.clicked.connect(self._run_all_actions)
        next_act = QPushButton("Next Action >")
        next_act.clicked.connect(self._next_action)
        next_mov = QPushButton("Next Moving Image >>")
        next_mov.clicked.connect(self._next_moving)
        next_seq = QPushButton("Next Sequence >>>")
        next_seq.clicked.connect(self._next_sequence)
        for b in (prev_act, run_all, next_act, next_mov, next_seq):
            nav.addWidget(b)
        v.addLayout(nav)

        persist = QHBoxLayout()
        save_cfg = QPushButton("Save Config (JSON)")
        save_cfg.clicked.connect(self._save_config)
        load_cfg = QPushButton("Load Config (JSON)")
        load_cfg.clicked.connect(self._load_config)
        persist.addWidget(save_cfg)
        persist.addWidget(load_cfg)
        persist.addStretch()
        v.addLayout(persist)

        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-weight: bold;")
        v.addWidget(self.status_label)
        return box

    # ------------------------------------------------------- table <-> model
    def _rebuild_table_from_model(self):
        ncols = 1 + self._n_moving_columns()
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setColumnCount(ncols)
        self.table.setRowCount(len(self.sequences))
        headers = ["Template"] + [f"Moving {i + 1}" for i in range(ncols - 1)]
        self.table.setHorizontalHeaderLabels(headers)
        for r, seq in enumerate(self.sequences):
            self._set_cell(r, 0, seq["template_path"])
            for c in range(ncols - 1):
                path = seq["moving_paths"][c] if c < len(seq["moving_paths"]) else ""
                self._set_cell(r, c + 1, path)
        self.table.blockSignals(False)

    def _set_cell(self, row, col, path):
        item = QTableWidgetItem(Path(path).name if path else "")
        item.setData(Qt.UserRole, path)
        if path:
            item.setToolTip(path)
        self.table.setItem(row, col, item)

    def _sync_model_from_table(self):
        """Read every cell's stored path back into self.sequences (preserve column slots)."""
        n_mov_cols = self.table.columnCount() - 1
        for r in range(min(self.table.rowCount(), len(self.sequences))):
            tpl_item = self.table.item(r, 0)
            self.sequences[r]["template_path"] = (tpl_item.data(Qt.UserRole) if tpl_item else "") or ""
            movings = []
            for c in range(1, self.table.columnCount()):
                item = self.table.item(r, c)
                movings.append((item.data(Qt.UserRole) if item else "") or "")
            self.sequences[r]["moving_paths"] = movings if movings else [""]
        # Keep the per-column action list aligned with the visible columns.
        self._ensure_columns(n_mov_cols)

    # --------------------------------------------------------- table actions
    def _add_sequence(self):
        self._sync_model_from_table()
        new = self._new_sequence()
        new["moving_paths"] = ["" for _ in range(self._n_moving_columns())]
        self.sequences.append(new)
        self._rebuild_table_from_model()
        self.table.setCurrentCell(len(self.sequences) - 1, 1)

    def _remove_sequence(self):
        row = self.table.currentRow()
        if row < 0 or len(self.sequences) <= 1:
            return
        self._sync_model_from_table()
        del self.sequences[row]
        self._rebuild_table_from_model()
        self.table.setCurrentCell(min(row, len(self.sequences) - 1), 1)

    def _add_moving_column(self):
        self._sync_model_from_table()
        self.column_actions.append([])
        self._ensure_columns(len(self.column_actions))
        self._rebuild_table_from_model()
        # Bind the editor to the new column.
        self.table.setCurrentCell(self.table.currentRow() if self.table.currentRow() >= 0 else 0,
                                  self.table.columnCount() - 1)

    def _browse_selected_cell(self):
        row, col = self.table.currentRow(), self.table.currentColumn()
        if row >= 0 and col >= 0:
            self._browse_cell(row, col)

    def _browse_cell(self, row, col):
        title = "Select Template Image(s)" if col == 0 else "Select Moving Image(s) / Stack(s)"
        paths, _ = QFileDialog.getOpenFileNames(self, title, "", IMAGE_FILTER)
        if not paths:
            return
        # natsort so e.g. img2 < img10; fill column `col` downward from `row`,
        # auto-adding rows when the selection runs past the table.
        paths = natsorted(paths)
        self._sync_model_from_table()
        for i, path in enumerate(paths):
            r = row + i
            if r >= len(self.sequences):
                new = self._new_sequence()
                new["moving_paths"] = ["" for _ in range(self._n_moving_columns())]
                self.sequences.append(new)
            if col == 0:
                self.sequences[r]["template_path"] = path
            else:
                mp = self.sequences[r]["moving_paths"]
                while len(mp) <= col - 1:
                    mp.append("")
                mp[col - 1] = path
        self._rebuild_table_from_model()

    # --------------------------------------------------------- action editor
    def _on_current_cell_changed(self, row, col, prev_row, prev_col):
        # Moving cells are columns 1..N; map to action column index 0..N-1.
        if col >= 1:
            self.edit_column = col - 1
            self._populate_action_editor()

    def _populate_action_editor(self):
        self.action_list.clear()
        self.action_header.setText(f"Editing actions for: Moving {self.edit_column + 1} "
                                   f"(applies to every row's image in this column)")
        if 0 <= self.edit_column < len(self.column_actions):
            for action in self.column_actions[self.edit_column]:
                self.action_list.addItem(_action_summary(action))
        self._refresh_copy_combo()

    def _refresh_copy_combo(self):
        self.copy_from_combo.blockSignals(True)
        self.copy_from_combo.clear()
        for c in range(len(self.column_actions)):
            if c != self.edit_column:
                self.copy_from_combo.addItem(f"Moving {c + 1}", c)
        self.copy_from_combo.blockSignals(False)

    def _set_row_visible(self, layout, visible):
        for i in range(layout.count()):
            w = layout.itemAt(i).widget()
            if w is not None:
                w.setVisible(visible)

    def _update_option_visibility(self):
        atype = self.action_combo.currentData()
        self._set_row_visible(self.matrix_row, atype == "set_matrix")
        self._set_row_visible(self.auto_row, atype == "auto_keypoints")
        self._set_row_visible(self.save_row, atype in ("save_image", "save_stack"))
        # Headless detector params only relevant when the dialog is OFF.
        if atype == "auto_keypoints":
            headless = not self.ak_use_dialog.isChecked()
            for w in self.ak_headless_widgets:
                w.setVisible(headless)
            # The distortion model combo only matters when distortion is on.
            self.ak_distortion_model.setVisible(headless and self.ak_use_distortion.isChecked())
        # Folder/suffix only relevant when 'Save to folder' is on.
        if atype in ("save_image", "save_stack"):
            use_folder = self.save_use_folder.isChecked()
            self.save_folder_edit.setVisible(use_folder)
            self.save_folder_btn.setVisible(use_folder)
            self.save_suffix_edit.setVisible(use_folder)

    def _browse_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.save_folder_edit.setText(folder)

    def _edit_actions(self):
        """The action list currently bound to the editor."""
        if 0 <= self.edit_column < len(self.column_actions):
            return self.column_actions[self.edit_column]
        return None

    def _add_action(self):
        actions = self._edit_actions()
        if actions is None:
            return
        atype = self.action_combo.currentData()
        if atype == "set_matrix":
            action = {
                "type": "set_matrix",
                "scale": self.scale_spin.value(),
                "rotation": self.rot_spin.value(),
                "tx": self.tx_spin.value(),
                "ty": self.ty_spin.value(),
            }
        elif atype == "auto_keypoints":
            action = {"type": "auto_keypoints", "dialog": self.ak_use_dialog.isChecked()}
            if not action["dialog"]:
                action.update({
                    "detector": self.ak_detector.currentText(),
                    "matcher": self.ak_matcher.currentText(),
                    "ransac_threshold": self.ak_ransac.value(),
                    "lock_rotation": self.ak_lock_rotation.isChecked(),
                    "lock_scale": self.ak_lock_scale.isChecked(),
                    "distortion_model": (self.ak_distortion_model.currentText()
                                         if self.ak_use_distortion.isChecked() else None),
                })
        elif atype in ("save_image", "save_stack"):
            action = {"type": atype}
            if self.save_use_folder.isChecked() and self.save_folder_edit.text().strip():
                action["folder"] = self.save_folder_edit.text().strip()
                action["suffix"] = self.save_suffix_edit.text().strip()
        else:
            action = {"type": atype}
        actions.append(action)
        self.action_list.addItem(_action_summary(action))

    def _remove_action(self):
        actions = self._edit_actions()
        idx = self.action_list.currentRow()
        if actions is None or idx < 0:
            return
        del actions[idx]
        self.action_list.takeItem(idx)

    def _move_action(self, delta):
        actions = self._edit_actions()
        idx = self.action_list.currentRow()
        if actions is None or idx < 0:
            return
        new_idx = idx + delta
        if not (0 <= new_idx < len(actions)):
            return
        actions[idx], actions[new_idx] = actions[new_idx], actions[idx]
        self._populate_action_editor()
        self.action_list.setCurrentRow(new_idx)

    def _copy_from_column(self):
        src = self.copy_from_combo.currentData()
        if src is None or not (0 <= src < len(self.column_actions)):
            return
        # Deep-ish copy: actions are flat dicts, so copy each dict.
        self.column_actions[self.edit_column] = [dict(a) for a in self.column_actions[src]]
        self._populate_action_editor()

    # ------------------------------------------------------------ navigation
    def _current_sequence(self):
        if 0 <= self.current_row < len(self.sequences):
            return self.sequences[self.current_row]
        return None

    def _current_moving_path(self):
        seq = self._current_sequence()
        if seq and 0 <= self.current_moving < len(seq["moving_paths"]):
            return seq["moving_paths"][self.current_moving]
        return ""

    def _current_actions(self):
        if 0 <= self.current_moving < len(self.column_actions):
            return self.column_actions[self.current_moving]
        return []

    def _load_current_pair(self):
        """Load the current row's template and current moving image into the aligner."""
        self._sync_model_from_table()
        seq = self._current_sequence()
        if seq is None:
            return False
        tpl = seq["template_path"]
        mov = self._current_moving_path()
        if not tpl or not mov:
            QMessageBox.warning(self, "Batch Mode",
                                "Both a template and a moving image must be set for this step.")
            return False
        self.aligner.load_template_from_path(tpl)
        self.aligner.load_moving_from_path(mov)
        # Ensure a transformed image exists (identity) so save actions can run.
        if self.aligner.current_transform is None:
            self.aligner.apply_transform(self.aligner.transform_controls.transform_params)
        return True

    def _execute_action(self, action):
        t = action.get("type")
        a = self.aligner
        if t == "set_matrix":
            a.transform_controls.set_values_from_params({
                "scale": action.get("scale", 1.0),
                "rotation": action.get("rotation", 0.0),
                "tx": action.get("tx", 0.0),
                "ty": action.get("ty", 0.0),
            })
        elif t == "auto_keypoints":
            if action.get("dialog", True):
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
                    distortion_model=action.get("distortion_model"),
                )
        elif t == "cross_correlation":
            a.optimize_phase_correlation()
        elif t == "reset":
            a.transform_controls.reset_transform()
        elif t == "save_image":
            folder = action.get("folder")
            if folder:
                a.save_image_to_folder(folder, action.get("suffix", ""))
            else:
                a.export_image()
        elif t == "save_stack":
            folder = action.get("folder")
            if folder:
                a.export_stack_to_folder(self._current_moving_path(), folder, action.get("suffix", ""))
            else:
                a.export_stack_from_path(self._current_moving_path())

    def _next_action(self):
        actions = self._current_actions()
        if not actions:
            QMessageBox.information(self, "Batch Mode",
                                    f"Moving {self.current_moving + 1} has no actions.")
            return
        if self.aligner.moving_image is None or self.aligner.template_image is None:
            if not self._load_current_pair():
                return
        if self.current_action >= len(actions):
            QMessageBox.information(self, "Batch Mode", "All actions done for this moving image.")
            return
        self._execute_action(actions[self.current_action])
        self.current_action = min(self.current_action + 1, len(actions))
        self._highlight_action()
        self._update_status()

    def _run_all_actions(self):
        """Run the remaining actions of the current moving image in sequence."""
        actions = self._current_actions()
        if not actions:
            QMessageBox.information(self, "Batch Mode",
                                    f"Moving {self.current_moving + 1} has no actions.")
            return
        if self.aligner.moving_image is None or self.aligner.template_image is None:
            if not self._load_current_pair():
                return
        while self.current_action < len(actions):
            self._execute_action(actions[self.current_action])
            self.current_action += 1
        self._highlight_action()
        self._update_status()

    def _previous_action(self):
        self.current_action = max(0, self.current_action - 1)
        self._highlight_action()
        self._update_status()

    def _next_moving(self):
        seq = self._current_sequence()
        if seq is None:
            return
        if self.current_moving + 1 < len(seq["moving_paths"]):
            self.current_moving += 1
            self.current_action = 0
            self.edit_column = self.current_moving
            self.table.setCurrentCell(self.current_row, self.current_moving + 1)
            self._populate_action_editor()
            self._load_current_pair()  # transform NOT reset (kept across images)
        else:
            QMessageBox.information(self, "Batch Mode", "Last moving image of this sequence.")
        self._update_status()

    def _next_sequence(self):
        self._sync_model_from_table()
        if self.current_row + 1 < len(self.sequences):
            self.current_row += 1
            self.current_moving = 0
            self.current_action = 0
            self.edit_column = 0
            self.table.setCurrentCell(self.current_row, 1)
            self._populate_action_editor()
            self._load_current_pair()  # transform NOT reset (kept across rows)
        else:
            QMessageBox.information(self, "Batch Mode", "Last sequence reached.")
        self._update_status()

    def _highlight_action(self):
        # Only highlight when the editor is bound to the column being navigated.
        if self.edit_column == self.current_moving:
            idx = min(self.current_action, self.action_list.count() - 1)
            if idx >= 0:
                self.action_list.setCurrentRow(idx)

    def _update_status(self):
        seq = self._current_sequence()
        n_seq = len(self.sequences)
        n_mov = len(seq["moving_paths"]) if seq else 0
        n_act = len(self._current_actions())
        self.status_label.setText(
            f"Sequence {self.current_row + 1}/{n_seq}  ·  "
            f"Moving {self.current_moving + 1}/{n_mov}  ·  "
            f"Action {min(self.current_action + 1, max(n_act, 1))}/{max(n_act, 0)}"
        )

    # ------------------------------------------------------------ persistence
    def _save_config(self):
        self._sync_model_from_table()
        path, _ = QFileDialog.getSaveFileName(self, "Save Batch Config", "", "JSON Files (*.json)")
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        try:
            with open(path, "w") as f:
                json.dump({
                    "version": 2,
                    "sequences": self.sequences,
                    "column_actions": self.column_actions,
                }, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save config: {e}")
            return
        self.status_label.setText(f"Saved config to {Path(path).name}")

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Batch Config", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            sequences = data["sequences"]
            assert isinstance(sequences, list) and sequences
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config: {e}")
            return

        self.sequences = []
        for s in sequences:
            self.sequences.append({
                "template_path": s.get("template_path", ""),
                "moving_paths": list(s.get("moving_paths", [""])) or [""],
            })

        # column_actions: present in v2. For v1 (per-row "actions"), migrate the
        # first row's actions onto column 0 as a best-effort fallback.
        if "column_actions" in data:
            self.column_actions = [list(c) for c in data["column_actions"]]
        else:
            first = sequences[0].get("actions", []) if sequences else []
            self.column_actions = [list(first)]

        n_mov_cols = max((len(s["moving_paths"]) for s in self.sequences), default=1)
        self._ensure_columns(max(n_mov_cols, len(self.column_actions), 1))

        self.current_row = self.current_moving = self.current_action = 0
        self.edit_column = 0
        self._rebuild_table_from_model()
        self.table.setCurrentCell(0, 1)
        self._populate_action_editor()
        self._update_status()
        self.status_label.setText(f"Loaded config from {Path(path).name}")
