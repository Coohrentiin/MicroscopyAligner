# Copyright (c) 2025 Corentin Soubeiran
# SPDX-License-Identifier: MIT
import numpy as np
import cv2
from pathlib import Path

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from keyPointsSelection import estimate_transform_keypoints


def detect_keypoints(image, method, max_features=500):
    """Detect keypoints and compute descriptors using the specified method.

    Standalone (Qt-free) so it can run headlessly from batch mode.
    """
    # Normalize image to uint8
    if image.dtype != np.uint8:
        img_norm = ((image - image.min()) / (image.max() - image.min() + 1e-10) * 255).astype(np.uint8)
    else:
        img_norm = image

    if method == "AKAZE":
        detector = cv2.AKAZE_create()
    elif method == "KAZE":
        detector = cv2.KAZE_create()
    elif method == "SIFT":
        detector = cv2.SIFT_create(nfeatures=max_features)
    elif method == "ORB":
        detector = cv2.ORB_create(nfeatures=max_features)
    elif method == "BRISK":
        detector = cv2.BRISK_create()
    else:
        raise ValueError(f"Unknown detection method: {method}")

    keypoints, descriptors = detector.detectAndCompute(img_norm, None)

    # Limit number of keypoints if not SIFT/ORB (they have built-in limits)
    if method not in ["SIFT", "ORB"] and len(keypoints) > max_features:
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:max_features]
        descriptors = descriptors[:max_features]

    return keypoints, descriptors


def match_keypoints(desc1, desc2, method, distance_ratio=0.75):
    """Match descriptors with Lowe's ratio test. Standalone (Qt-free)."""
    if desc1 is None or desc2 is None:
        return []

    if desc1.dtype == np.uint8:
        norm_type = cv2.NORM_HAMMING  # binary descriptors (ORB, BRISK, AKAZE)
    else:
        norm_type = cv2.NORM_L2       # float descriptors (SIFT, KAZE)

    if method == "Brute Force":
        matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    elif method == "FLANN":
        if norm_type == cv2.NORM_HAMMING:
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6,
                                key_size=12, multi_probe_level=1)
        else:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError(f"Unknown matching method: {method}")

    matches = matcher.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < distance_ratio * n.distance:
                good_matches.append(m)
    return good_matches


def filter_with_ransac(template_pts, moving_pts, threshold):
    """Filter matched points using RANSAC homography. Standalone (Qt-free)."""
    if len(template_pts) < 4:
        return template_pts, moving_pts
    _, mask = cv2.findHomography(moving_pts, template_pts, cv2.RANSAC,
                                 ransacReprojThreshold=threshold)
    if mask is None:
        return template_pts, moving_pts
    mask = mask.ravel().astype(bool)
    return template_pts[mask], moving_pts[mask]


def detect_keypoint_pairs(template_image, moving_image, detector="AKAZE",
                          matcher="Brute Force", max_features=500,
                          distance_ratio=0.75, use_ransac=True,
                          ransac_threshold=5.0, border_crop=0):
    """Run the full detect -> match -> (RANSAC) pipeline headlessly.

    ``border_crop`` (px) drops matched pairs whose template OR moving point lies
    within that margin of the respective image edge, avoiding spurious matches
    from edge/border artifacts (e.g. a propagated/zero-padded frame).

    Returns a list of ``(template_point, moving_point)`` tuples (possibly empty).
    """
    kp_t, desc_t = detect_keypoints(template_image, detector, max_features)
    kp_m, desc_m = detect_keypoints(moving_image, detector, max_features)
    if len(kp_t) == 0 or len(kp_m) == 0:
        return []

    matches = match_keypoints(desc_t, desc_m, matcher, distance_ratio)
    if len(matches) == 0:
        return []

    template_pts = np.float32([kp_t[m.queryIdx].pt for m in matches])
    moving_pts = np.float32([kp_m[m.trainIdx].pt for m in matches])

    if border_crop and border_crop > 0:
        th, tw = template_image.shape[:2]
        mh, mw = moving_image.shape[:2]
        b = float(border_crop)
        keep = []
        for i in range(len(template_pts)):
            tx, ty = template_pts[i]; mx, my = moving_pts[i]
            if (b <= tx <= tw - b and b <= ty <= th - b and
                    b <= mx <= mw - b and b <= my <= mh - b):
                keep.append(i)
        if not keep:
            return []
        template_pts = template_pts[keep]; moving_pts = moving_pts[keep]

    if use_ransac:
        template_pts, moving_pts = filter_with_ransac(
            template_pts, moving_pts, ransac_threshold)

    return [(tuple(template_pts[i]), tuple(moving_pts[i]))
            for i in range(len(template_pts))]


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

        # Border crop: drop matched pairs within N px of an image edge, to avoid
        # border / FFT-padding artifacts being matched as keypoints.
        border_layout = QHBoxLayout()
        border_layout.addWidget(QLabel("Crop template borders (px):"))
        self.border_crop = QSpinBox()
        self.border_crop.setRange(0, 1000)
        self.border_crop.setValue(0)
        self.border_crop.setToolTip("Ignore keypoints within this margin of the image edges.")
        border_layout.addWidget(self.border_crop)
        ransac_layout.addLayout(border_layout)

        ransac_group.setLayout(ransac_layout)

        # Transform constraints: optionally forbid the keypoints from changing
        # rotation and/or scale (e.g. objectives share orientation / known mag).
        constrain_group = QGroupBox("Transform Constraints")
        constrain_layout = QVBoxLayout()
        self.lock_rotation = QCheckBox("Lock rotation (no rotation from keypoints)")
        self.lock_scale = QCheckBox("Lock scale (no scale change from keypoints)")
        constrain_layout.addWidget(self.lock_rotation)
        constrain_layout.addWidget(self.lock_scale)
        chint = QLabel("Locking both leaves translation only; locking rotation "
                       "leaves scale+translation; locking scale leaves rotation+translation.")
        chint.setStyleSheet("color: gray;"); chint.setWordWrap(True)
        constrain_layout.addWidget(chint)
        constrain_group.setLayout(constrain_layout)

        # Distortion correction (non-rigid warp from the matched pairs, on top of
        # the linear transform -- e.g. to correct a non-centered objective bend).
        distortion_group = QGroupBox("Distortion Correction")
        distortion_layout = QVBoxLayout()
        self.use_distortion = QCheckBox("Estimate & apply distortion warp")
        self.use_distortion.setChecked(False)
        self.use_distortion.toggled.connect(self._update_distortion_enabled)
        distortion_layout.addWidget(self.use_distortion)
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.distortion_model = QComboBox()
        self.distortion_model.addItems(["tps", "poly", "radial", "spherical", "piecewise"])
        self.distortion_model.setEnabled(False)
        model_layout.addWidget(self.distortion_model)
        distortion_layout.addLayout(model_layout)
        hint = QLabel("Needs >= 3 pairs (TPS/piecewise want more, well spread). "
                      "Applied after the linear fit; persists until 'Reset Transformation'.")
        hint.setStyleSheet("color: gray;"); hint.setWordWrap(True)
        distortion_layout.addWidget(hint)
        distortion_group.setLayout(distortion_layout)

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
        layout.addWidget(constrain_group)
        layout.addWidget(distortion_group)
        layout.addWidget(self.detect_btn)
        layout.addWidget(list_label)
        layout.addWidget(self.points_list)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
        
        # Connections
        self.remove_btn.clicked.connect(self.remove_point)
        self.clear_btn.clicked.connect(self.clear_all)
        self.done_btn.clicked.connect(self.accept)

    def _update_distortion_enabled(self, checked):
        self.distortion_model.setEnabled(checked)

    def distortion_request(self):
        """Return ``(enabled, model)`` for the distortion-correction step."""
        return self.use_distortion.isChecked(), self.distortion_model.currentText()

    def constraints(self):
        """Return ``(lock_rotation, lock_scale)`` for the estimated transform."""
        return self.lock_rotation.isChecked(), self.lock_scale.isChecked()

    def border_crop_px(self):
        """Border margin (px) to ignore keypoints near image edges."""
        return int(self.border_crop.value())

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

            # Step 3b: Drop pairs near the image borders (avoid edge artifacts).
            b = self.border_crop_px()
            if b > 0:
                th, tw = self.template_image.shape[:2]
                mh, mw = self.moving_image.shape[:2]
                keep = [i for i in range(len(template_pts))
                        if (b <= template_pts[i][0] <= tw - b and b <= template_pts[i][1] <= th - b
                            and b <= moving_pts[i][0] <= mw - b and b <= moving_pts[i][1] <= mh - b)]
                template_pts = template_pts[keep]; moving_pts = moving_pts[keep]
                if len(template_pts) == 0:
                    QMessageBox.warning(self, "Warning", "No keypoints left after border crop")
                    QApplication.restoreOverrideCursor()
                    return

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
        """Detect keypoints and compute descriptors (uses widget Max Features)."""
        return detect_keypoints(image, method, self.max_features.value())

    def match_keypoints(self, desc1, desc2, method):
        """Match keypoints (uses widget Distance Ratio)."""
        return match_keypoints(desc1, desc2, method, self.distance_ratio.value())

    def filter_with_ransac(self, template_pts, moving_pts, threshold):
        """Filter matched points using RANSAC."""
        return filter_with_ransac(template_pts, moving_pts, threshold)

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
