# Copyright (c) 2025 Corentin Soubeiran
# SPDX-License-Identifier: MIT
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

class BruteForceDialog(QDialog):
    """Dialog for brute force optimization parameters"""
    
    def __init__(self, parent=None, current_params=None):
        super().__init__(parent)
        self.setWindowTitle("Brute Force Optimization Parameters")
        self.setModal(True)
        self.init_ui(current_params)
        
    def init_ui(self,current_params=None):
        layout = QVBoxLayout()
        
        # Create parameter inputs
        params_layout = QGridLayout()
        
        # Rotation range
        params_layout.addWidget(QLabel("Rotation Range (°):"), 0, 0)
        self.rot_min = QSpinBox()
        self.rot_min.setRange(-45, 45)
        if current_params is not None:
            self.rot_min.setValue(int(current_params.get('rotation', 0)) - 10)
        else:
            self.rot_min.setValue(-10)
        params_layout.addWidget(self.rot_min, 0, 1)
        
        params_layout.addWidget(QLabel("to"), 0, 2)
        self.rot_max = QSpinBox()
        self.rot_max.setRange(-45, 45)
        if current_params is not None:
            self.rot_max.setValue(int(current_params.get('rotation', 0)) + 10)
        else:
            self.rot_max.setValue(10)
        params_layout.addWidget(self.rot_max, 0, 3)
        
        params_layout.addWidget(QLabel("Steps:"), 0, 4)
        self.rot_steps = QSpinBox()
        self.rot_steps.setRange(1, 20)
        self.rot_steps.setValue(20)
        params_layout.addWidget(self.rot_steps, 0, 5)
        
        # Scale range
        params_layout.addWidget(QLabel("Scale Range:"), 1, 0)
        self.scale_min = QDoubleSpinBox()
        self.scale_min.setRange(0.5, 2.0)
        if current_params is not None:
            self.scale_min.setValue(current_params.get('scale', 1.0) - 0.1)
        else:
            self.scale_min.setValue(0.9)
        self.scale_min.setSingleStep(0.1)
        params_layout.addWidget(self.scale_min, 1, 1)
        
        params_layout.addWidget(QLabel("to"), 1, 2)
        self.scale_max = QDoubleSpinBox()
        self.scale_max.setRange(0.5, 2.0)
        if current_params is not None:
            self.scale_max.setValue(current_params.get('scale', 1.0) + 0.1)
        else:
            self.scale_max.setValue(1.1)
        self.scale_max.setSingleStep(0.1)
        params_layout.addWidget(self.scale_max, 1, 3)
        
        params_layout.addWidget(QLabel("Steps:"), 1, 4)
        self.scale_steps = QSpinBox()
        self.scale_steps.setRange(1, 20)
        self.scale_steps.setValue(20)
        params_layout.addWidget(self.scale_steps, 1, 5)
        
        # Translation X range
        params_layout.addWidget(QLabel("Translation X (px):"), 2, 0)
        self.tx_min = QSpinBox()
        self.tx_min.setRange(-100, 100)
        if current_params is not None:
            self.tx_min.setValue(current_params.get('tx', 0) - 10)
        else:
            self.tx_min.setValue(-10)
        params_layout.addWidget(self.tx_min, 2, 1)
        
        params_layout.addWidget(QLabel("to"), 2, 2)
        self.tx_max = QSpinBox()
        self.tx_max.setRange(-100, 100)
        if current_params is not None:
            self.tx_max.setValue(current_params.get('tx', 0) + 10)
        else:
            self.tx_max.setValue(10)
        params_layout.addWidget(self.tx_max, 2, 3)
        
        params_layout.addWidget(QLabel("Steps:"), 2, 4)
        self.tx_steps = QSpinBox()
        self.tx_steps.setRange(1, 20)
        self.tx_steps.setValue(5)
        params_layout.addWidget(self.tx_steps, 2, 5)
        
        # Translation Y range
        params_layout.addWidget(QLabel("Translation Y (px):"), 3, 0)
        self.ty_min = QSpinBox()
        self.ty_min.setRange(-100, 100)
        if current_params is not None:
            self.ty_min.setValue(current_params.get('ty', 0) - 10)
        else:
            self.ty_min.setValue(-10)
        params_layout.addWidget(self.ty_min, 3, 1)
        
        params_layout.addWidget(QLabel("to"), 3, 2)
        self.ty_max = QSpinBox()
        self.ty_max.setRange(-100, 100)
        if current_params is not None:
            self.ty_max.setValue(current_params.get('ty', 0) + 10)
        else:
            self.ty_max.setValue(10)
        params_layout.addWidget(self.ty_max, 3, 3)
        
        params_layout.addWidget(QLabel("Steps:"), 3, 4)
        self.ty_steps = QSpinBox()
        self.ty_steps.setRange(1, 20)
        self.ty_steps.setValue(5)
        params_layout.addWidget(self.ty_steps, 3, 5)
        
        layout.addLayout(params_layout)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
    def get_ranges(self):
        """Get the parameter ranges"""
        return {
            'rot_min': self.rot_min.value(),
            'rot_max': self.rot_max.value(),
            'rot_steps': self.rot_steps.value(),
            'scale_min': self.scale_min.value(),
            'scale_max': self.scale_max.value(),
            'scale_steps': self.scale_steps.value(),
            'tx_min': self.tx_min.value(),
            'tx_max': self.tx_max.value(),
            'tx_steps': self.tx_steps.value(),
            'ty_min': self.ty_min.value(),
            'ty_max': self.ty_max.value(),
            'ty_steps': self.ty_steps.value()
        }
 