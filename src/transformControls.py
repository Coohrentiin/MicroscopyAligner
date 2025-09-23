# Copyright (c) 2025 Corentin Soubeiran
# SPDX-License-Identifier: MIT
import numpy as np
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *


class TransformControls(QWidget):
    """Widget for transformation controls"""
    
    transform_changed = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.transform_params = {
            'rotation': 0.0,
            'scale': 1.0,
            'tx': 0.0,
            'ty': 0.0
        }
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Transform Controls")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)
        
        # Rotation control
        rot_group = QGroupBox("Rotation")
        rot_layout = QVBoxLayout()
        
        self.rot_slider = QSlider(Qt.Orientation.Horizontal)
        self.rot_slider.setRange(-180, 180)
        self.rot_slider.setValue(0.)
        self.rot_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.rot_slider.setTickInterval(45)
        self.rot_slider.setSingleStep(0.1)
        
        self.rot_spin = QDoubleSpinBox()
        self.rot_spin.setRange(-180, 180)
        self.rot_spin.setSuffix("°")
        self.rot_spin.setSingleStep(0.1)
        
        rot_buttons = QHBoxLayout()
        rot_m90 = QPushButton("-90°")
        rot_m90.clicked.connect(lambda: self.quick_rotate(-90))
        rot_p90 = QPushButton("+90°")
        rot_p90.clicked.connect(lambda: self.quick_rotate(90))
        rot_buttons.addWidget(rot_m90)
        rot_buttons.addWidget(rot_p90)
        
        rot_layout.addWidget(self.rot_slider)
        rot_layout.addWidget(self.rot_spin)
        rot_layout.addLayout(rot_buttons)
        rot_group.setLayout(rot_layout)
        
        # Scale control
        scale_group = QGroupBox("Scale")
        scale_layout = QVBoxLayout()
        
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(25, 400)
        self.scale_slider.setValue(100)
        self.scale_slider.setSingleStep(0.1)    
        self.scale_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.scale_slider.setTickInterval(50)
        
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.25, 4.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setPrefix("×")
        
        scale_layout.addWidget(self.scale_slider)
        scale_layout.addWidget(self.scale_spin)
        scale_group.setLayout(scale_layout)
        
        # # Translation controls
        # trans_group = QGroupBox("Translation")
        # trans_layout = QGridLayout()
        
        # trans_layout.addWidget(QLabel("X:"), 0, 0)
        # self.tx_spin = QSpinBox()
        # self.tx_spin.setRange(-500, 500)
        # self.tx_spin.setSuffix(" px")
        # trans_layout.addWidget(self.tx_spin, 0, 1)
        
        # trans_layout.addWidget(QLabel("Y:"), 1, 0)
        # self.ty_spin = QSpinBox()
        # self.ty_spin.setRange(-500, 500)
        # self.ty_spin.setSuffix(" px")
        # trans_layout.addWidget(self.ty_spin, 1, 1)
        
        # trans_group.setLayout(trans_layout)

        trans_group = QGroupBox("Translation")
        trans_layout = QVBoxLayout()

        # --- X Translation ---
        trans_layout.addWidget(QLabel("X:"))

        self.tx_slider = QSlider(Qt.Horizontal)
        self.tx_slider.setRange(-500, 500)
        self.tx_slider.setValue(0.)
        self.tx_slider.setSingleStep(0.1)

        self.tx_spin = QDoubleSpinBox()
        self.tx_spin.setRange(-500, 500)
        self.tx_spin.setSuffix(" px")
        self.tx_spin.setValue(0.)
        self.tx_spin.setSingleStep(0.1)

        # link slider and spinbox
        self.tx_slider.valueChanged.connect(self.tx_spin.setValue)
        self.tx_spin.valueChanged.connect(self.tx_slider.setValue)

        # quick buttons
        tx_buttons = QHBoxLayout()
        tx_m10 = QPushButton("-10 px")
        tx_m10.clicked.connect(lambda: self.tx_spin.setValue(self.tx_spin.value() - 10))
        tx_p10 = QPushButton("+10 px")
        tx_p10.clicked.connect(lambda: self.tx_spin.setValue(self.tx_spin.value() + 10))
        tx_buttons.addWidget(tx_m10)
        tx_buttons.addWidget(tx_p10)

        # add widgets to layout
        trans_layout.addWidget(self.tx_slider)
        trans_layout.addWidget(self.tx_spin)
        trans_layout.addLayout(tx_buttons)

        # # --- Y Translation ---
        trans_layout.addWidget(QLabel("Y:"))

        self.ty_slider = QSlider(Qt.Horizontal)
        self.ty_slider.setRange(-500, 500)
        self.ty_slider.setValue(0.0)
        self.ty_slider.setSingleStep(0.1)

        self.ty_spin = QDoubleSpinBox()
        self.ty_spin.setRange(-500, 500)
        self.ty_spin.setSuffix(" px")
        self.ty_spin.setValue(0.0)
        self.ty_spin.setSingleStep(0.1)

        # link slider and spinbox
        self.ty_slider.valueChanged.connect(self.ty_spin.setValue)
        self.ty_spin.valueChanged.connect(self.ty_slider.setValue)

        # quick buttons
        ty_buttons = QHBoxLayout()
        ty_m10 = QPushButton("-10 px")
        ty_m10.clicked.connect(lambda: self.ty_spin.setValue(self.ty_spin.value() - 10))
        ty_p10 = QPushButton("+10 px")
        ty_p10.clicked.connect(lambda: self.ty_spin.setValue(self.ty_spin.value() + 10))
        ty_buttons.addWidget(ty_m10)
        ty_buttons.addWidget(ty_p10)

        # add widgets to layout
        trans_layout.addWidget(self.ty_slider)
        trans_layout.addWidget(self.ty_spin)
        trans_layout.addLayout(ty_buttons)

        trans_group.setLayout(trans_layout)
        # Matrix display
        matrix_group = QGroupBox("Transformation Matrix")
        matrix_layout = QVBoxLayout()
        
        self.matrix_display = QTextEdit()
        self.matrix_display.setReadOnly(True)
        self.matrix_display.setMaximumHeight(100)
        self.matrix_display.setFont(QFont("Courier", 9))
        
        reset_btn = QPushButton("Reset Transform")
        reset_btn.setStyleSheet("QPushButton { background-color: #e60b0b; color: white; }")
        reset_btn.clicked.connect(self.reset_transform)
        
        matrix_layout.addWidget(self.matrix_display)
        matrix_layout.addWidget(reset_btn)
        matrix_group.setLayout(matrix_layout)
        
        # Add all to main layout
        layout.addWidget(rot_group)
        layout.addWidget(scale_group)
        layout.addWidget(trans_group)
        layout.addWidget(matrix_group)
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Connect signals
        self.rot_slider.valueChanged.connect(self.rot_spin.setValue)
        self.rot_spin.valueChanged.connect(self.rot_slider.setValue)
        self.rot_spin.valueChanged.connect(self.on_transform_changed)
        
        self.scale_slider.valueChanged.connect(lambda v: self.scale_spin.setValue(v/100))
        self.scale_spin.valueChanged.connect(lambda v: self.scale_slider.setValue(int(v*100)))
        self.scale_spin.valueChanged.connect(self.on_transform_changed)
        
        self.tx_spin.valueChanged.connect(self.on_transform_changed)
        self.ty_spin.valueChanged.connect(self.on_transform_changed)
        
    def quick_rotate(self, angle):
        current = self.rot_spin.value()
        self.rot_spin.setValue(current + angle)
        
    def on_transform_changed(self):
        self.transform_params = {
            'rotation': self.rot_spin.value(),
            'scale': self.scale_spin.value(),
            'tx': self.tx_spin.value(),
            'ty': self.ty_spin.value()
        }
        self.update_matrix_display()
        self.transform_changed.emit(self.transform_params)
    
    def set_values_from_transform(self, transfrom_mat):
        """Set the control values based on a given transformation matrix"""
        # Extract parameters from the affine matrix
        a, b, tx = transfrom_mat[0]
        c, d, ty = transfrom_mat[1]
        
        scale = np.sqrt(a**2 + c**2)
        rotation = np.degrees(np.arctan2(c, a))
        
        #Set value wuthout triggering signals
        self.rot_spin.blockSignals(True)
        self.scale_spin.blockSignals(True)
        self.tx_spin.blockSignals(True)
        self.ty_spin.blockSignals(True)
        self.rot_spin.setValue(rotation)
        self.scale_spin.setValue(scale)
        self.tx_spin.setValue(tx)
        self.ty_spin.setValue(ty)
        self.rot_spin.blockSignals(False)
        self.scale_spin.blockSignals(False)
        self.tx_spin.blockSignals(False)
        self.ty_spin.blockSignals(False)
        self.on_transform_changed()

    def set_values_from_params(self, params):
        """Set the control values based on given parameters dict"""
        self.rot_spin.blockSignals(True)
        self.scale_spin.blockSignals(True)
        self.tx_spin.blockSignals(True)
        self.ty_spin.blockSignals(True)
        self.rot_spin.setValue(params.get('rotation', 0))
        self.scale_spin.setValue(params.get('scale', 1.0))
        self.tx_spin.setValue(params.get('tx', 0))
        self.ty_spin.setValue(params.get('ty', 0))
        self.rot_spin.blockSignals(False)
        self.scale_spin.blockSignals(False)
        self.tx_spin.blockSignals(False)
        self.ty_spin.blockSignals(False)
        self.on_transform_changed()

    def update_matrix_display(self):
        """Update the transformation matrix display"""
        angle_rad = np.radians(self.transform_params['rotation'])
        s = self.transform_params['scale']
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        matrix = np.array([
            [s * cos_a, -s * sin_a, self.transform_params['tx']],
            [s * sin_a, s * cos_a, self.transform_params['ty']],
            [0, 0, 1]
        ])
        
        matrix_str = "[\n"
        for row in matrix:
            matrix_str += "  " + " ".join(f"{val:7.3f}" for val in row) + "\n"
        matrix_str += "]"
        
        self.matrix_display.setPlainText(matrix_str)
        
    def reset_transform(self):
        self.rot_spin.setValue(0)
        self.scale_spin.setValue(1.0)
        self.tx_spin.setValue(0)
        self.ty_spin.setValue(0)
        
    def set_transform(self, params):
        """Set transform parameters programmatically"""
        self.rot_spin.setValue(params.get('rotation', 0))
        self.scale_spin.setValue(params.get('scale', 1.0))
        self.tx_spin.setValue(params.get('tx', 0))
        self.ty_spin.setValue(params.get('ty', 0))
