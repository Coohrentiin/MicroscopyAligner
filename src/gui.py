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
from keyPointsSelection import KeyPointsSelection, estimate_transform_keypoints
from utils_images import load_imgfile

from skimage.registration import phase_cross_correlation
import cv2
import matplotlib.pyplot as plt

# Matplotlib colormaps
available_colormaps = ["gray", "red", "green", "blue", "cyan", "magenta"] + plt.colormaps()
class ImageAligner(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.template_image = None
        self.moving_image = None
        self.transformed_image = None
        self.current_transform = None
        self.opt_transform = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Microscopy Image Alignment Tool")
        self.setGeometry(100, 100, 1400, 900)
        
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

        # Optimization controls
        opt_group = QGroupBox("Optimization")
        opt_layout = QVBoxLayout()
        
        self.opt_method = QComboBox()
        self.opt_method.addItems(["Manual Pairs of Points","Phase Cross-Correlation","Brute Force"]) #["Manual Pairs of Points","Phase Cross-Correlation", "Brute Force", "Enhanced Correlation"]
        
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
        
        self.crop_check = QCheckBox("Crop to overlap")
        self.crop_check.setChecked(True)
        
        export_btn = QPushButton("Export Transformed Image")
        export_btn.setStyleSheet("QPushButton { background-color: #FF9800; }")
        export_btn.clicked.connect(self.export_image)
        
        batch_btn = QPushButton("Apply to Folder")
        batch_btn.setStyleSheet("QPushButton { background-color: #9C27B0; }")
        batch_btn.clicked.connect(self.batch_process)
        
        export_layout.addWidget(self.crop_check)
        export_layout.addWidget(export_btn)
        export_layout.addWidget(batch_btn)
        export_group.setLayout(export_layout)
        
        right_layout.addWidget(self.transform_controls)
        right_layout.addWidget(opt_group)
        right_layout.addWidget(export_group)
        right_layout.addStretch()
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(right_panel, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready to load images")
        
    def load_template(self):
        """Load template (reference) image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Template Image", "", "Image Files (*.tif *.tiff *.png *.jpg)"
        )
        if file_path:
            self.template_image_file = file_path
            self.template_image = load_imgfile(file_path).astype(np.float32)
            self.canvas.set_template(
                self.template_image,
                self.template_color.currentText(),
                self.template_opacity.value() / 100
            )
            self.statusBar().showMessage(f"Loaded template: {Path(file_path).name}")
        if self.template_image is not None:
            self.load_template_btn.setStyleSheet(f"QPushButton {{ background-color: {QApplication.instance().palette().button().color().name()}; }}")

    def load_moving(self):
        """Load moving image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Moving Image", "", "Image Files (*.tif *.tiff *.png *.jpg)"
        )
        if file_path:
            self.moving_image_file = file_path
            self.moving_image = load_imgfile(file_path).astype(np.float32)
            self.canvas.set_moving(
                self.moving_image,
                self.moving_color.currentText(),
                self.moving_opacity.value() / 100
            )
            self.statusBar().showMessage(f"Loaded moving image: {Path(file_path).name}")
        if self.moving_image is not None:
            self.load_moving_btn.setStyleSheet(f"QPushButton {{ background-color: {QApplication.instance().palette().button().color().name()}; }}")
        # When loading a new moving image, apply current transform if available
        if self.current_transform is not None:
            self.apply_transform(self.transform_controls.transform_params)
        
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

    def open_keypoints_tool(self):
        if self.template_image is None or self.moving_image is None:
            QMessageBox.warning(self, "Warning", "Load both images first")
            return
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

        if self.keypoints_dialog.exec() == QDialog.Accepted:
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
        # Apply transformation
        self.transformed_image = tf.warp(
            self.moving_image,
            transform.inverse,
            output_shape=self.template_image.shape if self.template_image is not None else self.moving_image.shape,
            preserve_range=True
        ).astype(np.float32)
        
        self.update_display()
        
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
        elif method == "Phase Cross-Correlation":
            self.optimize_phase_correlation()
        elif method == "Brute Force":
            self.optimize_brute_force()
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
        if dialog.exec() == QDialog.DialogCode.Accepted:
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
            else:
                img_normalized = ((img_to_save - img_to_save.min()) / 
                                 (img_to_save.max() - img_to_save.min() + 1e-10) * 255).astype(np.uint8)
                Image.fromarray(img_normalized).save(file_path)
                
            self.statusBar().showMessage(f"Exported to: {Path(file_path).name}")
            
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
            img = self.load_image(str(img_file))
            
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
  