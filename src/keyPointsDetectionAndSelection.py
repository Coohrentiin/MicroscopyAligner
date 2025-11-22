# Copyright (c) 2025 Corentin Soubeiran
# SPDX-License-Identifier: MIT
import numpy as np
import cv2
from pathlib import Path

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from keyPointsSelection import estimate_transform_keypoints


class KeyPointsDetectionAndSelection(QDialog):
    """Dialog to detect, match, and manage automatic keypoint pairs for alignment."""

    def __init__(self, parent=None, template_image=None, moving_image=None):
        super().__init__(parent)
        self.setWindowTitle("Automatic Keypoints Detection & Selection")
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.resize(400, 600)

        self.template_image = template_image
        self.moving_image = moving_image
        
        # Data: list of (template_point, moving_point)
        self.point_pairs = []
        self.keypoints_template = []
        self.keypoints_moving = []
        self.descriptors_template = None
        self.descriptors_moving = None
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Detection method selection
        detection_group = QGroupBox("Keypoint Detection")
        detection_layout = QVBoxLayout()
        
        detection_layout.addWidget(QLabel("Detection Method:"))
        self.detection_method = QComboBox()
        self.detection_method.addItems(["AKAZE", "KAZE", "SIFT", "ORB", "BRISK"])
        self.detection_method.setCurrentText("AKAZE")
        detection_layout.addWidget(self.detection_method)
        
        # Number of features
        features_layout = QHBoxLayout()
        features_layout.addWidget(QLabel("Max Features:"))
        self.max_features = QSpinBox()
        self.max_features.setRange(10, 10000)
        self.max_features.setValue(500)
        self.max_features.setSingleStep(50)
        features_layout.addWidget(self.max_features)
        detection_layout.addLayout(features_layout)
        
        detection_group.setLayout(detection_layout)
        
        # Matching method selection
        matching_group = QGroupBox("Keypoint Matching")
        matching_layout = QVBoxLayout()
        
        matching_layout.addWidget(QLabel("Matching Method:"))
        self.matching_method = QComboBox()
        self.matching_method.addItems(["Brute Force", "FLANN"])
        self.matching_method.setCurrentText("Brute Force")
        matching_layout.addWidget(self.matching_method)
        
        # Distance ratio for Lowe's test
        ratio_layout = QHBoxLayout()
        ratio_layout.addWidget(QLabel("Distance Ratio:"))
        self.distance_ratio = QDoubleSpinBox()
        self.distance_ratio.setRange(0.1, 1.0)
        self.distance_ratio.setValue(0.75)
        self.distance_ratio.setSingleStep(0.05)
        self.distance_ratio.setDecimals(2)
        ratio_layout.addWidget(self.distance_ratio)
        matching_layout.addLayout(ratio_layout)
        
        matching_group.setLayout(matching_layout)
        
        # RANSAC filtering
        ransac_group = QGroupBox("RANSAC Filtering")
        ransac_layout = QVBoxLayout()
        
        self.use_ransac = QCheckBox("Enable RANSAC Filtering")
        self.use_ransac.setChecked(True)
        ransac_layout.addWidget(self.use_ransac)
        
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("RANSAC Threshold:"))
        self.ransac_threshold = QDoubleSpinBox()
        self.ransac_threshold.setRange(0.5, 20.0)
        self.ransac_threshold.setValue(5.0)
        self.ransac_threshold.setSingleStep(0.5)
        self.ransac_threshold.setDecimals(1)
        threshold_layout.addWidget(self.ransac_threshold)
        ransac_layout.addLayout(threshold_layout)
        
        ransac_group.setLayout(ransac_layout)
        
        # Detect button
        self.detect_btn = QPushButton("Detect & Match Keypoints")
        self.detect_btn.setStyleSheet("QPushButton { background-color: #2196F3; font-weight: bold; }")
        self.detect_btn.clicked.connect(self.detect_and_match)
        
        # Points list
        list_label = QLabel("Detected Keypoint Pairs:")
        list_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.points_list = QListWidget()
        
        # Control buttons
        self.remove_btn = QPushButton("Remove Selected Pair")
        self.clear_btn = QPushButton("Clear All")
        self.done_btn = QPushButton("Done")
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.done_btn)
        
        # Add all to main layout
        layout.addWidget(detection_group)
        layout.addWidget(matching_group)
        layout.addWidget(ransac_group)
        layout.addWidget(self.detect_btn)
        layout.addWidget(list_label)
        layout.addWidget(self.points_list)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
        
        # Connections
        self.remove_btn.clicked.connect(self.remove_point)
        self.clear_btn.clicked.connect(self.clear_all)
        self.done_btn.clicked.connect(self.accept)

    def detect_and_match(self):
        """Detect keypoints, match them, and optionally filter with RANSAC."""
        if self.template_image is None or self.moving_image is None:
            QMessageBox.warning(self, "Warning", "Both images must be loaded")
            return
        
        # Show progress
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Detecting keypoints...") if hasattr(self, 'statusBar') else None
        
        try:
            # Step 1: Detect keypoints
            self.keypoints_template, self.descriptors_template = self.detect_keypoints(
                self.template_image, self.detection_method.currentText()
            )
            self.keypoints_moving, self.descriptors_moving = self.detect_keypoints(
                self.moving_image, self.detection_method.currentText()
            )
            
            if len(self.keypoints_template) == 0 or len(self.keypoints_moving) == 0:
                QMessageBox.warning(self, "Warning", "No keypoints detected in one or both images")
                QApplication.restoreOverrideCursor()
                return
            
            # Step 2: Match keypoints
            matches = self.match_keypoints(
                self.descriptors_template,
                self.descriptors_moving,
                self.matching_method.currentText()
            )
            
            if len(matches) == 0:
                QMessageBox.warning(self, "Warning", "No keypoint matches found")
                QApplication.restoreOverrideCursor()
                return
            
            # Step 3: Extract matched points
            template_pts = np.float32([self.keypoints_template[m.queryIdx].pt for m in matches])
            moving_pts = np.float32([self.keypoints_moving[m.trainIdx].pt for m in matches])
            
            # Step 4: Apply RANSAC filtering if enabled
            if self.use_ransac.isChecked():
                template_pts, moving_pts = self.filter_with_ransac(
                    template_pts, moving_pts, self.ransac_threshold.value()
                )
            
            if len(template_pts) == 0:
                QMessageBox.warning(self, "Warning", "No keypoint pairs survived RANSAC filtering")
                QApplication.restoreOverrideCursor()
                return
            
            # Clear existing pairs
            self.clear_all()
            
            # Add matched pairs to the list
            for i in range(len(template_pts)):
                template_pt = tuple(template_pts[i])
                moving_pt = tuple(moving_pts[i])
                self.add_pair(template_pt, moving_pt)
            
            QMessageBox.information(
                self, "Success", 
                f"Detected {len(self.point_pairs)} keypoint pairs"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Keypoint detection failed: {str(e)}")
            print(f"Error in detect_and_match: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            QApplication.restoreOverrideCursor()

    def detect_keypoints(self, image, method):
        """Detect keypoints and compute descriptors using the specified method."""
        # Normalize image to uint8
        if image.dtype != np.uint8:
            img_norm = ((image - image.min()) / (image.max() - image.min() + 1e-10) * 255).astype(np.uint8)
        else:
            img_norm = image
        
        # Select detector
        if method == "AKAZE":
            detector = cv2.AKAZE_create()
        elif method == "KAZE":
            detector = cv2.KAZE_create()
        elif method == "SIFT":
            detector = cv2.SIFT_create(nfeatures=self.max_features.value())
        elif method == "ORB":
            detector = cv2.ORB_create(nfeatures=self.max_features.value())
        elif method == "BRISK":
            detector = cv2.BRISK_create()
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        # Detect and compute
        keypoints, descriptors = detector.detectAndCompute(img_norm, None)
        
        # Limit number of keypoints if not SIFT/ORB (they have built-in limits)
        if method not in ["SIFT", "ORB"] and len(keypoints) > self.max_features.value():
            # Sort by response strength and keep top N
            keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:self.max_features.value()]
            # Recompute descriptors for selected keypoints
            descriptors = descriptors[:self.max_features.value()]
        
        return keypoints, descriptors

    def match_keypoints(self, desc1, desc2, method):
        """Match keypoints using the specified method."""
        if desc1 is None or desc2 is None:
            return []
        
        # Check descriptor type for matcher selection
        if desc1.dtype == np.uint8:
            # Binary descriptors (ORB, BRISK, AKAZE)
            norm_type = cv2.NORM_HAMMING
        else:
            # Float descriptors (SIFT, KAZE)
            norm_type = cv2.NORM_L2
        
        if method == "Brute Force":
            matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        elif method == "FLANN":
            if norm_type == cv2.NORM_HAMMING:
                # FLANN parameters for binary descriptors
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                   table_number=6,
                                   key_size=12,
                                   multi_probe_level=1)
            else:
                # FLANN parameters for float descriptors
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unknown matching method: {method}")
        
        # Match descriptors using k-nearest neighbors
        matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.distance_ratio.value() * n.distance:
                    good_matches.append(m)
        
        return good_matches

    def filter_with_ransac(self, template_pts, moving_pts, threshold):
        """Filter matched points using RANSAC."""
        if len(template_pts) < 4:
            return template_pts, moving_pts
        
        # Use OpenCV's findHomography with RANSAC
        _, mask = cv2.findHomography(
            moving_pts, template_pts, 
            cv2.RANSAC, 
            ransacReprojThreshold=threshold
        )
        
        if mask is None:
            return template_pts, moving_pts
        
        # Keep only inliers
        mask = mask.ravel().astype(bool)
        template_pts_filtered = template_pts[mask]
        moving_pts_filtered = moving_pts[mask]
        
        return template_pts_filtered, moving_pts_filtered

    def add_pair(self, template_pt, moving_pt):
        """Add a new point pair to the list."""
        self.point_pairs.append((template_pt, moving_pt))
        item_text = f"Pair {len(self.point_pairs)}: T({template_pt[0]:.1f},{template_pt[1]:.1f}) | M({moving_pt[0]:.1f},{moving_pt[1]:.1f})"
        self.points_list.addItem(item_text)
        
        # Draw points on canvas if available
        if self.parent() and hasattr(self.parent(), 'canvas'):
            canvas = self.parent().canvas
            point_num = len(self.point_pairs)
            
            # Draw template point (blue for paired)
            canvas.draw_point(template_pt, color=Qt.blue, number=point_num)
            
            # Offset moving point x-coordinate for side-by-side display
            if canvas.template_array is not None:
                moving_pt_display = (moving_pt[0] + canvas.template_array.shape[1], moving_pt[1])
            else:
                moving_pt_display = moving_pt
            
            # Draw moving point (blue for paired)
            canvas.draw_point(moving_pt_display, color=Qt.blue, number=point_num)
            
            # Draw line connecting the two points
            line = canvas.scene.addLine(template_pt[0], template_pt[1], 
                                       moving_pt_display[0], moving_pt_display[1], 
                                       QPen(Qt.blue))
            line.setZValue(1)
            
            # Add line to points_items
            if point_num in canvas.points_items:
                canvas.points_items[point_num].append((line, None))
            else:
                canvas.points_items[point_num] = [(line, None)]

    def remove_point(self):
        """Remove selected point pair."""
        row = self.points_list.currentRow()
        if row >= 0:
            self.points_list.takeItem(row)
            del self.point_pairs[row]
            
            # Clear and redraw all points
            if self.parent() and hasattr(self.parent(), 'canvas'):
                self.parent().canvas.clear_keypoints()
                canvas = self.parent().canvas
                for i, (template_pt, moving_pt) in enumerate(self.point_pairs, 1):
                    # Draw template point (blue for paired)
                    canvas.draw_point(template_pt, color=Qt.blue, number=i)
                    
                    # Offset moving point x-coordinate for side-by-side display
                    if canvas.template_array is not None:
                        moving_pt_display = (moving_pt[0] + canvas.template_array.shape[1], moving_pt[1])
                    else:
                        moving_pt_display = moving_pt
                    
                    # Draw moving point (blue for paired)
                    canvas.draw_point(moving_pt_display, color=Qt.blue, number=i)
                    
                    # Draw line connecting the two points
                    line = canvas.scene.addLine(template_pt[0], template_pt[1], 
                                               moving_pt_display[0], moving_pt_display[1], 
                                               QPen(Qt.blue))
                    line.setZValue(1)
                    
                    # Add line to points_items
                    if i in canvas.points_items:
                        canvas.points_items[i].append((line, None))
                    else:
                        canvas.points_items[i] = [(line, None)]
            
            # Update list labels
            self.points_list.clear()
            for i, (template_pt, moving_pt) in enumerate(self.point_pairs, 1):
                item_text = f"Pair {i}: T({template_pt[0]:.1f},{template_pt[1]:.1f}) | M({moving_pt[0]:.1f},{moving_pt[1]:.1f})"
                self.points_list.addItem(item_text)

    def clear_all(self):
        """Remove all point pairs."""
        self.points_list.clear()
        self.point_pairs.clear()
        if self.parent() and hasattr(self.parent(), 'canvas'):
            self.parent().canvas.clear_keypoints()

    def update_pair(self, index, template_pt, moving_pt):
        """Update an existing point pair."""
        if 0 <= index < len(self.point_pairs):
            self.point_pairs[index] = (template_pt, moving_pt)
            item_text = f"Pair {index+1}: T({template_pt[0]:.1f},{template_pt[1]:.1f}) | M({moving_pt[0]:.1f},{moving_pt[1]:.1f})"
            self.points_list.item(index).setText(item_text)

    def closeEvent(self, event):
        """Handle dialog close event."""
        print("KeyPointsDetectionAndSelection closed")
        parent = self.parent()
        if parent and hasattr(parent, "canvas"):
            parent.canvas.enable_keypoint_mode(False)
            parent.canvas.clear_keypoints()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    
    # Test the dialog
    app = QApplication(sys.argv)
    
    # Create dummy images
    template = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
    moving = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
    
    dialog = KeyPointsDetectionAndSelection(None, template, moving)
    dialog.show()
    
    sys.exit(app.exec())
