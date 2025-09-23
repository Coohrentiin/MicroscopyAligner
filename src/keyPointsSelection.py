# Copyright (c) 2025 Corentin Soubeiran
# SPDX-License-Identifier: MIT
import numpy as np
from pathlib import Path

# from sklearn.linear_model import RANSACRegressor
# from sklearn.preprocessing import StandardScaler

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

class KeyPointsSelection(QDialog):
    """Dialog to manage manual keypoint pairs for alignment."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keypoints Selection")
        # self.setModal(True)
        # Important: prevent blocking canvas mouse events
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)

        self.points_list = QListWidget()
        self.remove_btn = QPushButton("Remove Point")
        self.clear_btn = QPushButton("Clear All")
        self.done_btn = QPushButton("Done")

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.done_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.points_list)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # Data: list of (template_point, moving_point)
        self.point_pairs = []

        # Connections
        self.remove_btn.clicked.connect(self.remove_point)
        self.clear_btn.clicked.connect(self.clear_all)
        self.done_btn.clicked.connect(self.accept)

    def add_pair(self, template_pt, moving_pt):
        """Add a new point pair to the list."""
        self.point_pairs.append((template_pt, moving_pt))
        item_text = f"T: ({template_pt[0]:.1f},{template_pt[1]:.1f}) | M: ({moving_pt[0]:.1f},{moving_pt[1]:.1f})"
        self.points_list.addItem(item_text)
        # Recolor points on canvas:

    def remove_point(self):
        """Remove selected point pair."""
        row = self.points_list.currentRow()
        if row >= 0:
            self.points_list.takeItem(row)
            del self.point_pairs[row]
            self.parent().canvas.clear_keypoint(row+1)

    def clear_all(self):
        """Remove all point pairs."""
        self.points_list.clear()
        self.point_pairs.clear()
        self.parent().canvas.clear_keypoints()
    
    def closeEvent(self, event):
        print("KeyPointsSelection closed")
        parent = self.parent()
        if parent and hasattr(parent, "canvas"):
            parent.canvas.enable_keypoint_mode(False)
            parent.canvas.clear_keypoints()
        super().closeEvent(event)


def estimate_transform_keypoints(point_pairs):
    """
    Estimate transformation matrix from point pairs.
    
    Args:
        point_pairs: List of tuples [((x1, y1), (x2, y2)), ...]
                    where (x1,y1) are template points and (x2,y2) are moving points
    
    Returns:
        3x3 transformation matrix
    """
    if not point_pairs:
        raise ValueError("At least one point pair is required")
    
    n_points = len(point_pairs)
    
    # Convert to numpy arrays for easier manipulation
    template_pts = np.array([pair[1] for pair in point_pairs])
    moving_pts = np.array([pair[0] for pair in point_pairs])
    
    if n_points == 1:
        # Translation only
        print(f"Estimating translation from 1 point pair using translation only")
        return translation_transform(template_pts, moving_pts)
    elif n_points == 2:
        # Similarity transform (translation + rotation + uniform scale)
        print(f"Estimating similarity transform from 2 point pairs (considering translation, rotation, scale)")
        return similarity_transform(template_pts, moving_pts)
    elif n_points == 3:
        # Affine transform (6 DOF)
        print(f"Estimating affine transform from 3 point pairs (considering translation, rotation, scale)")
        return affine_transform(template_pts, moving_pts)
    else:
        # 4+ points: Use RANSAC for robust estimation
        print(f"Estimating robust affine transform from {n_points} point pairs using RANSAC")
        return ransac_affine_transform(template_pts, moving_pts)

def translation_transform(template_pts, moving_pts):
    """Estimate translation only from point pairs"""
    # Calculate average translation
    translation = np.mean(moving_pts - template_pts, axis=0)
    
    # Build transformation matrix
    matrix = np.eye(3)
    matrix[0, 2] = translation[0]
    matrix[1, 2] = translation[1]
    
    return matrix

def similarity_transform(template_pts, moving_pts):
    """Estimate similarity transformation (translation + rotation + uniform scale)"""
    # Center the points
    template_center = np.mean(template_pts, axis=0)
    moving_center = np.mean(moving_pts, axis=0)
    
    template_centered = template_pts - template_center
    moving_centered = moving_pts - moving_center
    
    # Calculate scale
    template_norm = np.linalg.norm(template_centered, axis=1)
    moving_norm = np.linalg.norm(moving_centered, axis=1)
    scale = np.mean(moving_norm / template_norm)
    
    # Calculate rotation using complex numbers representation
    # For 2 points, we can use the angle between the vectors
    if len(template_pts) == 2:
        vec_template = template_pts[1] - template_pts[0]
        vec_moving = moving_pts[1] - moving_pts[0]
        
        angle_template = np.arctan2(vec_template[1], vec_template[0])
        angle_moving = np.arctan2(vec_moving[1], vec_moving[0])
        rotation_angle = angle_moving - angle_template
    else:
        # Alternative method using SVD (more robust for multiple points)
        H = template_centered.T @ moving_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        rotation_angle = np.arctan2(R[1, 0], R[0, 0])
    
    # Build transformation matrix
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    
    matrix = np.eye(3)
    matrix[0, 0] = scale * cos_theta
    matrix[0, 1] = -scale * sin_theta
    matrix[1, 0] = scale * sin_theta
    matrix[1, 1] = scale * cos_theta
    
    # Calculate translation
    rotation_matrix = matrix[:2, :2]
    translation = moving_center - rotation_matrix @ template_center
    matrix[0, 2] = translation[0]
    matrix[1, 2] = translation[1]
    
    return matrix

def affine_transform(template_pts, moving_pts):
    """Estimate affine transformation (6 DOF) using least squares"""
    n = len(template_pts)
    
    # Build the system of equations: A * params = b
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    
    for i in range(n):
        x, y = template_pts[i]
        A[2*i, :] = [x, y, 1, 0, 0, 0]
        A[2*i+1, :] = [0, 0, 0, x, y, 1]
        
        b[2*i] = moving_pts[i, 0]
        b[2*i+1] = moving_pts[i, 1]
    
    # Solve using least squares
    params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Build transformation matrix
    matrix = np.eye(3)
    matrix[0, :] = params[:3]
    matrix[1, :] = params[3:]
    
    return matrix

def ransac_affine_transform(template_pts, moving_pts, min_samples=3, max_trials=100, residual_threshold=5.0):
    """Estimate affine transformation using RANSAC for robustness"""
    n_samples = len(template_pts)
    
    if n_samples < min_samples:
        return affine_transform(template_pts, moving_pts)
    
    # Prepare data for RANSAC
    X = template_pts
    y = moving_pts
    
    # Custom estimator for affine transformation
    class AffineEstimator:
        def fit(self, X, y):
            return affine_transform(X, y)
        
        def predict(self, X, transform_matrix):
            # Apply transformation to points
            homogeneous_pts = np.column_stack([X, np.ones(len(X))])
            transformed = (transform_matrix @ homogeneous_pts.T).T
            return transformed[:, :2]
    
    estimator = AffineEstimator()
    
    # Simple RANSAC implementation
    best_matrix = None
    best_inliers = 0
    best_residual = float('inf')
    
    for _ in range(max_trials):
        # Randomly select min_samples points
        idx = np.random.choice(n_samples, min_samples, replace=False)
        X_sample = X[idx]
        y_sample = y[idx]
        
        try:
            # Fit model to sample
            matrix = affine_transform(X_sample, y_sample)
            
            # Calculate residuals for all points
            homogeneous_pts = np.column_stack([X, np.ones(len(X))])
            predicted = (matrix @ homogeneous_pts.T).T[:, :2]
            residuals = np.linalg.norm(predicted - y, axis=1)
            
            # Count inliers
            inliers = residuals < residual_threshold
            n_inliers = np.sum(inliers)
            mean_residual = np.mean(residuals[inliers]) if n_inliers > 0 else float('inf')
            
            # Update best model
            if n_inliers > best_inliers or (n_inliers == best_inliers and mean_residual < best_residual):
                best_inliers = n_inliers
                best_residual = mean_residual
                best_matrix = matrix
                
        except np.linalg.LinAlgError:
            continue
    
    # If RANSAC failed, fall back to regular affine transform
    if best_matrix is None:
        return affine_transform(template_pts, moving_pts)
    
    # Optional: Refit using all inliers
    if best_inliers > min_samples:
        homogeneous_pts = np.column_stack([X, np.ones(len(X))])
        predicted = (best_matrix @ homogeneous_pts.T).T[:, :2]
        residuals = np.linalg.norm(predicted - y, axis=1)
        inlier_mask = residuals < residual_threshold
        
        if np.sum(inlier_mask) > min_samples:
            best_matrix = affine_transform(X[inlier_mask], y[inlier_mask])
    
    return best_matrix

# Alternative simpler version without RANSAC for 4+ points:
def projective_transform(template_pts, moving_pts):
    """Estimate projective transformation (8 DOF) using DLT algorithm"""
    n = len(template_pts)
    
    if n < 4:
        return affine_transform(template_pts, moving_pts)
    
    # Build the homogeneous system: A * h = 0
    A = np.zeros((2 * n, 9))
    
    for i in range(n):
        x, y = template_pts[i]
        u, v = moving_pts[i]
        
        A[2*i, :] = [x, y, 1, 0, 0, 0, -u*x, -u*y, -u]
        A[2*i+1, :] = [0, 0, 0, x, y, 1, -v*x, -v*y, -v]
    
    # Solve using SVD
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]  # Last row of Vt corresponds to smallest singular value
    
    # Normalize and reshape to 3x3 matrix
    matrix = h.reshape(3, 3)
    matrix = matrix / matrix[2, 2]  # Normalize
    
    return matrix

# Usage example:
if __name__ == "__main__":
    # Test with different numbers of points
    point_pairs_1 = [((10, 20), (15, 25))]  # 1 point - translation
    point_pairs_2 = [((10, 20), (15, 25)), ((30, 40), (40, 45))]  # 2 points - similarity
    point_pairs_3 = [((10, 20), (15, 25)), ((30, 40), (40, 45)), ((50, 60), (65, 70))]  # 3 points - affine
    point_pairs_4 = [((10, 20), (15, 25)), ((30, 40), (40, 45)), 
                    ((50, 60), (65, 70)), ((70, 80), (90, 95))]  # 4+ points - RANSAC
    
    for i, pairs in enumerate([point_pairs_1, point_pairs_2, point_pairs_3, point_pairs_4], 1):
        try:
            matrix = estimate_transform_keypoints(pairs)
            print(f"Transformation matrix for {i} point(s):")
            print(matrix)
            print()
        except Exception as e:
            print(f"Error with {i} points: {e}")