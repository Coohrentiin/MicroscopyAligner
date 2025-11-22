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
        
        # For dragging keypoints
        self.dragging_point = None  # (point_number, is_template)
        self.drag_start_pos = None

        # View settings
        self.setRenderHints(self.renderHints() | 
                            QPainter.Antialiasing | 
                            QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)  # enables scrolling with mouse drag
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # zoom around mouse
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # Zoom parameters
        self._zoom_factor = 1.25
        self._current_zoom = 0

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

    def wheelEvent(self, event: QWheelEvent):
        """Handle zoom with mouse wheel"""
        if event.angleDelta().y() > 0:
            zoom = self._zoom_factor
            self._current_zoom += 1
        else:
            zoom = 1 / self._zoom_factor
            self._current_zoom -= 1

        if self._current_zoom < -10:  # optional limit
            self._current_zoom = -10
            return
        elif self._current_zoom > 30:
            self._current_zoom = 30
            return

        self.scale(zoom, zoom)
        
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
        scene_pos = self.mapToScene(event.pos())
        x, y = scene_pos.x(), scene_pos.y()
        
        # Check if we're clicking on an existing keypoint to drag it
        if self.view_mode == "side_by_side":
            clicked_point = self.find_point_at_position(x, y)
            if clicked_point is not None:
                self.dragging_point = clicked_point
                self.drag_start_pos = (x, y)
                self.setCursor(Qt.ClosedHandCursor)
                return
        
        if self.keypoint_mode and self.view_mode == "side_by_side":
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
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging keypoints."""
        if self.dragging_point is not None:
            scene_pos = self.mapToScene(event.pos())
            x, y = scene_pos.x(), scene_pos.y()
            
            point_num, is_template = self.dragging_point
            self.update_point_position(point_num, is_template, x, y)
            return
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release for dropping keypoints."""
        if self.dragging_point is not None:
            scene_pos = self.mapToScene(event.pos())
            x, y = scene_pos.x(), scene_pos.y()
            
            point_num, is_template = self.dragging_point
            self.update_point_position(point_num, is_template, x, y)
            
            # Notify parent dialog about the update
            self.notify_point_update(point_num, is_template, x, y)
            
            self.dragging_point = None
            self.drag_start_pos = None
            self.setCursor(Qt.ArrowCursor)
            return
        
        super().mouseReleaseEvent(event)

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
            for item_tuple in self.points_items[item_number]:
                if isinstance(item_tuple, tuple) and len(item_tuple) >= 1:
                    # Remove first item (ellipse or line)
                    if item_tuple[0] is not None:
                        self.scene.removeItem(item_tuple[0])
                    # Remove second item (text) if exists
                    if len(item_tuple) > 1 and item_tuple[1] is not None:
                        self.scene.removeItem(item_tuple[1])
            del self.points_items[item_number]
        return
    
    def find_point_at_position(self, x, y, threshold=10):
        """Find if there's a keypoint near the given position.
        Returns (point_number, is_template) or None."""
        for point_num, items_list in self.points_items.items():
            for item_tuple in items_list:
                ellipse = item_tuple[0] if isinstance(item_tuple, tuple) else item_tuple
                if ellipse is None:
                    continue
                    
                # Get ellipse position (center)
                rect = ellipse.boundingRect() #.rect()
                cx = rect.center().x()
                cy = rect.center().y()
                
                # Check distance
                dist = np.sqrt((cx - x)**2 + (cy - y)**2)
                if dist < threshold:
                    # Determine if template or moving side
                    if self.template_array is not None:
                        is_template = cx < self.template_array.shape[1]
                        return (point_num, is_template)
        return None
    
    def update_point_position(self, point_num, is_template, new_x, new_y):
        """Update the position of a keypoint by removing old graphics and redrawing."""
        if point_num not in self.points_items:
            return
        
        # Get the items list for this point pair
        items_list = self.points_items[point_num]
        
        # Identify which items to keep and which to replace
        # Structure: [(template_ellipse, template_text), (moving_ellipse, moving_text), (line, None)]
        template_items = None
        moving_items = None
        line_item = None
        
        # Separate items based on position
        for item_tuple in items_list:
            if len(item_tuple) < 1 or item_tuple[0] is None:
                continue
            
            first_item = item_tuple[0]
            
            # Check if it's a line
            if hasattr(first_item, 'line'):
                line_item = first_item
                continue
            
            # It's an ellipse - check position to determine template vs moving
            if hasattr(first_item, 'rect'):
                rect = first_item.rect()
                cx = rect.center().x()
                
                if self.template_array is not None:
                    if cx < self.template_array.shape[1]:
                        template_items = item_tuple
                    else:
                        moving_items = item_tuple
        
        # Determine color (blue for paired points)
        color = Qt.blue
        
        # Remove and recreate the item being moved
        if is_template and template_items is not None:
            # Remove old template graphics
            ellipse, text = template_items
            print(f"Removing old template ellipse at {ellipse.rect().center()}")
            self.scene.removeItem(ellipse)
            if text is not None:
                print(f"Removing old template text")
                self.scene.removeItem(text)
            
            # Create new template graphics at new position
            r = 4
            new_ellipse = self.scene.addEllipse(new_x - r, new_y - r, 2*r, 2*r,
                                                QPen(color), QBrush(color))
            new_ellipse.setZValue(1)
            print(f"Created new template ellipse at ({new_x}, {new_y})")
            
            new_text = None
            if text is not None:  # Keep text if it existed
                new_text = self.scene.addText(str(point_num))
                new_text.setDefaultTextColor(color)
                new_text.setPos(new_x + 5, new_y - 5)
                new_text.setZValue(1)
            
            # Update items_list in the dictionary
            template_idx = items_list.index(template_items)
            self.points_items[point_num][template_idx] = (new_ellipse, new_text)
            print(f"Updated point {point_num} template items at index {template_idx}")
            
            # Force scene update
            self.scene.update()
            
            # Update line if it exists
            if line_item is not None:
                self.scene.removeItem(line_item)
                if moving_items is not None:
                    moving_ellipse = moving_items[0]
                    moving_rect = moving_ellipse.rect()
                    moving_cx = moving_rect.center().x()
                    moving_cy = moving_rect.center().y()
                    new_line = self.scene.addLine(new_x, new_y, moving_cx, moving_cy, QPen(color))
                    new_line.setZValue(1)
                    # Update line in items_list
                    for i, item in enumerate(self.points_items[point_num]):
                        if len(item) > 0 and item[0] == line_item:
                            self.points_items[point_num][i] = (new_line, None)
                            break
                    
        elif not is_template and moving_items is not None:
            # Remove old moving graphics
            ellipse, text = moving_items
            self.scene.removeItem(ellipse)
            if text is not None:
                self.scene.removeItem(text)
            
            # Create new moving graphics at new position
            r = 4
            new_ellipse = self.scene.addEllipse(new_x - r, new_y - r, 2*r, 2*r,
                                                QPen(color), QBrush(color))
            new_ellipse.setZValue(1)
            
            new_text = None
            if text is not None:  # Keep text if it existed
                new_text = self.scene.addText(str(point_num))
                new_text.setDefaultTextColor(color)
                new_text.setPos(new_x + 5, new_y - 5)
                new_text.setZValue(1)
            
            # Update items_list in the dictionary
            moving_idx = items_list.index(moving_items)
            self.points_items[point_num][moving_idx] = (new_ellipse, new_text)
            
            # Force scene update
            self.scene.update()
            
            # Update line if it exists
            if line_item is not None:
                self.scene.removeItem(line_item)
                if template_items is not None:
                    template_ellipse = template_items[0]
                    template_rect = template_ellipse.rect()
                    template_cx = template_rect.center().x()
                    template_cy = template_rect.center().y()
                    new_line = self.scene.addLine(template_cx, template_cy, new_x, new_y, QPen(color))
                    new_line.setZValue(1)
                    # Update line in items_list
                    for i, item in enumerate(self.points_items[point_num]):
                        if len(item) > 0 and item[0] == line_item:
                            self.points_items[point_num][i] = (new_line, None)
                            break
    
    def notify_point_update(self, point_num, is_template, new_x, new_y):
        """Notify parent dialog about keypoint position update."""
        parent = self.window()
        
        # Adjust coordinates for moving side
        if not is_template and self.template_array is not None:
            new_x -= self.template_array.shape[1]
        
        # Find the appropriate dialog
        if hasattr(parent, 'auto_keypoints_dialog') and parent.auto_keypoints_dialog is not None:
            dialog = parent.auto_keypoints_dialog
            if hasattr(dialog, 'point_pairs') and 0 <= point_num - 1 < len(dialog.point_pairs):
                old_template, old_moving = dialog.point_pairs[point_num - 1]
                if is_template:
                    new_template = (new_x, new_y)
                    new_moving = old_moving
                else:
                    new_template = old_template
                    new_moving = (new_x, new_y)
                dialog.update_pair(point_num - 1, new_template, new_moving)
        elif hasattr(parent, 'keypoints_dialog') and parent.keypoints_dialog is not None:
            dialog = parent.keypoints_dialog
            if hasattr(dialog, 'point_pairs') and 0 <= point_num - 1 < len(dialog.point_pairs):
                old_template, old_moving = dialog.point_pairs[point_num - 1]
                if is_template:
                    new_template = (new_x, new_y)
                    new_moving = old_moving
                else:
                    new_template = old_template
                    new_moving = (new_x, new_y)
                dialog.update_pair(point_num - 1, new_template, new_moving)

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
