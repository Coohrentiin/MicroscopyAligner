# Copyright (c) 2025 Corentin Soubeiran
# SPDX-License-Identifier: MIT
import numpy as np

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import matplotlib.pyplot as plt


class ImageCanvas(QGraphicsView):
    """Custom canvas for displaying overlaid images"""
    point_added = Signal(tuple, tuple)  # (template_point, moving_point)

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.template_item = None
        self.moving_item = None
        self.template_array = None
        self.moving_array = None
        self.transformed_array = None
        self.view_mode = 'overlay'  # or 'side_by_side'
        self.keypoint_mode = False
        self.pending_point = None   # store template point before moving point
        self.points_items = {}      # dict of (ellipse_item, text_item)

    def set_template(self, img_array, colormap='gray', opacity=0.5):
        """Set template (reference) image"""
        self.template_array = img_array
        qimg = self.array_to_qimage(img_array, colormap)
        if self.template_item:
            self.scene.removeItem(self.template_item)

        pixmap = QPixmap.fromImage(qimg)
        self.template_item = self.scene.addPixmap(pixmap)
        self.template_item.setOpacity(opacity)
        self.template_item.setZValue(0)

        if self.view_mode == 'overlay':
            self.template_item.setPos(0, 0)  # ensure template is at origin
            self.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)



    def set_moving(self, img_array, colormap='green', opacity=0.5):
        """Set moving image with transformation"""
        self.moving_array = img_array
        qimg = self.array_to_qimage(img_array, colormap)
        if self.moving_item:
            self.scene.removeItem(self.moving_item)

        pixmap = QPixmap.fromImage(qimg)
        self.moving_item = self.scene.addPixmap(pixmap)
        self.moving_item.setOpacity(opacity)
        self.moving_item.setZValue(1)

        if self.view_mode == 'overlay':
            # Overlay → same position
            print("Overlay mode: setting moving image position to (0,0)")
            self.moving_item.setPos(0, 0)
            self.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        elif self.view_mode == 'side_by_side' and self.template_array is not None:
            # Side-by-side → place moving image to the right of template
            w = self.template_array.shape[1]
            print("Side-by-side mode: setting template width:", w)
            self.template_item.setPos(0, 0)
            self.moving_item.setPos(w, 0)
            # Adjust scene rect to include both images
            total_width = self.template_array.shape[1] + self.moving_array.shape[1]
            total_height = max(self.template_array.shape[0], self.moving_array.shape[0])
            self.scene.setSceneRect(0, 0, total_width, total_height)
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_view_mode(self, mode):
        """Set view mode: 'overlay' or 'side_by_side'"""
        self.view_mode = mode

    # def update_moving_transform(self, transformed_array, colormap='green', opacity=0.5):
    #     """Update moving image with transformation applied"""
    #     self.transformed_array = transformed_array
    #     qimg = self.array_to_qimage(transformed_array, colormap)
    #     if self.moving_item:
    #         self.scene.removeItem(self.moving_item)
    #     pixmap = QPixmap.fromImage(qimg)
    #     self.moving_item = self.scene.addPixmap(pixmap)
    #     self.moving_item.setOpacity(opacity)
    #     self.moving_item.setZValue(1)

    # def enable_keypoint_mode(self, enable=True):
    #     self.keypoint_mode = enable
    #     self.pending_point = None
    def enable_keypoint_mode(self, enabled: bool):
        self.keypoint_mode = enabled
        if not enabled:
            self.clear_keypoints()

    def mousePressEvent(self, event):
        print("Mouse press event at:", event.pos(), "Keypoint mode:", self.keypoint_mode, "View mode:", self.view_mode)
        if self.keypoint_mode and self.view_mode == "side_by_side":
            scene_pos = self.mapToScene(event.pos())
            x, y = scene_pos.x(), scene_pos.y()

            # Check if click was in template or moving image
            if self.template_array is not None and x < self.template_array.shape[1]:
                # template side
                self.pending_point = (x, y)
                self.draw_point((x, y), color=Qt.red) #, number=len(self.points_items)+1)
            elif self.moving_array is not None and self.template_array is not None:
                if x >= self.template_array.shape[1]:
                    # moving side → adjust x relative to moving image
                    rel_x = x - self.template_array.shape[1]
                    moving_pt = (rel_x, y)

                    if self.pending_point is not None:
                        # finalize pair
                        self.point_added.emit(self.pending_point, moving_pt)
                        self.draw_point((x, y), color=Qt.blue, number=len(self.points_items)) # same number as template point
                        # Redraw the template point un blue and add a line between them
                        template_point = self.pending_point
                        self.draw_point(template_point, color=Qt.blue, number=len(self.points_items))
                        line = self.scene.addLine(template_point[0], template_point[1], x, y, QPen(Qt.blue))
                        self.points_items[len(self.points_items)].append((line, None))
                        line.setZValue(1)
                        self.pending_point = None
            return
        super().mousePressEvent(event)

    def draw_point(self, pt, color=Qt.blue, number=None):
        """Draw a point at (x,y) with optional label."""
        r = 4
        print("Drawing point at:", pt)
        ellipse = self.scene.addEllipse(pt[0]-r, pt[1]-r, 2*r, 2*r,
                                        QPen(color), QBrush(color))
        ellipse.setZValue(1)

        text = None
        if number is not None:
            text = self.scene.addText(str(number))
            text.setDefaultTextColor(color)
            text.setPos(pt[0]+5, pt[1]-5)
            text.setZValue(1)
        if number is not None and number not in self.points_items:
            self.points_items[number] = [(ellipse, text)]
        elif number is not None:
            self.points_items[number].append((ellipse, text))
        else:
            self.points_items[len(self.points_items)+1] = [(ellipse, text)]


    def clear_keypoints(self):
        """Remove all keypoint markers from the scene"""
        print("Clearing all keypoints")
        for item_list in self.points_items.values():
            for ellipse, text in item_list:
                self.scene.removeItem(ellipse)
                if text:
                    self.scene.removeItem(text)
        self.points_items.clear()
        self.pending_point = None
        

    def clear_keypoint(self,item_number):
        """Remove specific keypoint markers from the scene"""
        print("Clearing keypoint number:", item_number)
        if item_number in self.points_items:
            print("Found keypoint items:", self.points_items[item_number])
            for ellipse, text in self.points_items[item_number]:
                self.scene.removeItem(ellipse)
                if text:
                    self.scene.removeItem(text)
            del self.points_items[item_number]
        return    
    

    def array_to_qimage(self, arr, colormap):
        """Convert numpy array to QImage with colormap"""
        # Normalize to 0-255
        arr_norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-10) * 255).astype(np.uint8)
        h, w = arr_norm.shape

        # Create alpha mask: transparent where original value == 0
        alpha = np.where(arr == 0, 0, 255).astype(np.uint8)

        if colormap == 'gray':
            rgba = np.stack([arr_norm, arr_norm, arr_norm, alpha], axis=-1)
        elif colormap == 'green':
            rgba = np.stack([np.zeros_like(arr_norm),
                            arr_norm,
                            np.zeros_like(arr_norm),
                            alpha], axis=-1)
        elif colormap == 'red':
            rgba = np.stack([arr_norm,
                            np.zeros_like(arr_norm),
                            np.zeros_like(arr_norm),
                            alpha], axis=-1)
        elif colormap == 'blue':
            rgba = np.stack([np.zeros_like(arr_norm),
                            np.zeros_like(arr_norm),
                            arr_norm,
                            alpha], axis=-1)
        elif colormap == 'cyan':
            rgba = np.stack([np.zeros_like(arr_norm),
                            arr_norm,
                            arr_norm,
                            alpha], axis=-1)
        elif colormap == 'magenta':
            rgba = np.stack([arr_norm,
                            np.zeros_like(arr_norm),
                            arr_norm,
                            alpha], axis=-1)
        else:
            # Use matplotlib for other colormaps
            cmap = plt.get_cmap(colormap)
            colored = (cmap(arr_norm / 255.0)[:, :, :3] * 255).astype(np.uint8)
            rgba = np.concatenate([colored, alpha[..., None]], axis=-1)

        img = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        return img
