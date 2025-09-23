# Copyright (c) 2025 Corentin Soubeiran
# SPDX-License-Identifier: MIT
import sys
import numpy as np
import os
from pathlib import Path

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from skimage import transform as tf
import tifffile
from PIL import Image

from gui import ImageAligner

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark theme palette 
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(dark_palette)
    app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")


    # Set application icon
    icon_path = Path(__file__).parent.parent / "src" / "resources" / "icon.png"
    app.setWindowIcon(QIcon(str(icon_path)))

    window = ImageAligner()
    window.setWindowIcon(QIcon(str(icon_path)))
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()