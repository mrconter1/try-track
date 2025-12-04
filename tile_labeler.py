"""Tile Labeler - Label tiles across video frames."""

import sys
import os
import random
import argparse
import json
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy, 
                              QSpinBox, QListWidget, QListWidgetItem)
from PyQt6.QtCore import Qt, QPoint, QRectF
from PyQt6.QtGui import QPixmap, QImage, QShortcut, QKeySequence, QPainter, QPen, QColor, QBrush


STYLE = """
QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0f0f1a, stop:1 #1a1a2e);
}
QLabel {
    color: #888;
    font-family: 'Consolas', monospace;
}
QLabel#title {
    color: #00d4ff;
    font-size: 18px;
    font-weight: bold;
    font-family: 'Segoe UI', sans-serif;
}
QLabel#info {
    color: #666;
    font-size: 11px;
}
QLabel#frameLabel {
    color: #00d4ff;
    font-size: 12px;
    padding: 5px;
}
QLabel#videoInfo {
    color: #aaa;
    font-size: 12px;
    padding: 10px;
    background: #0a0a12;
    border-radius: 6px;
}
QFrame#frameBox {
    background: #0a0a12;
    border: 2px solid #2a2a3e;
    border-radius: 10px;
}
QFrame#controlPanel {
    background: #12121f;
    border-left: 1px solid #2a2a3e;
}
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #00d4ff, stop:1 #0099cc);
    color: #000;
    font-size: 13px;
    font-weight: bold;
    font-family: 'Segoe UI', sans-serif;
    padding: 14px 30px;
    border: none;
    border-radius: 6px;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #33e0ff, stop:1 #00b8e6);
}
QPushButton:pressed {
    background: #0088aa;
}
QPushButton#stepBtn {
    padding: 0;
    font-size: 18px;
    font-weight: bold;
}
QPushButton#smallBtn {
    padding: 8px 15px;
    font-size: 11px;
}
QPushButton#deleteBtn {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff6b6b, stop:1 #cc5555);
    padding: 8px 15px;
    font-size: 11px;
}
QPushButton#deleteBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff8888, stop:1 #dd6666);
}
QFrame#separator {
    background: #2a2a3e;
}
QSpinBox {
    background: #0a0a12;
    color: #00d4ff;
    border: 1px solid #2a2a3e;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 14px;
    font-family: 'Consolas', monospace;
}
QSpinBox::up-button, QSpinBox::down-button {
    width: 0;
    height: 0;
    border: none;
}
QListWidget {
    background: #0a0a12;
    color: #aaa;
    border: 1px solid #2a2a3e;
    border-radius: 4px;
    font-family: 'Consolas', monospace;
    font-size: 11px;
}
QListWidget::item {
    padding: 6px;
}
QListWidget::item:selected {
    background: #00d4ff;
    color: #000;
}
QListWidget::item:hover {
    background: #1a1a3e;
}
"""

TILE_COLORS = [
    (255, 100, 100),
    (100, 255, 100),
    (100, 100, 255),
    (255, 255, 100),
    (255, 100, 255),
    (100, 255, 255),
    (255, 180, 100),
    (180, 100, 255),
]

SAVE_FILE = 'tile_labels.json'


class FrameDisplay(QFrame):
    def __init__(self, frame_index, parent_sampler):
        super().__init__()
        self.frame_index = frame_index
        self.parent_sampler = parent_sampler
        self.setObjectName("frameBox")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        self.image_label = InteractiveImageLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setStyleSheet("background: #050508; border-radius: 6px;")
        self.image_label.setMinimumSize(200, 200)
        layout.addWidget(self.image_label)
        
        self.frame_label = QLabel("—")
        self.frame_label.setObjectName("frameLabel")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setFixedHeight(25)
        layout.addWidget(self.frame_label)
        
        self.current_frame = None
        self.current_frame_num = None
        
        self.img_offset_x = 0
        self.img_offset_y = 0
        self.img_scale = 1.0
        self.img_w = 0
        self.img_h = 0
    
    def set_frame(self, frame, frame_num):
        self.current_frame = frame.copy()
        self.current_frame_num = frame_num
        self.frame_label.setText(f"Frame {frame_num}")
        self._update_display()
    
    def _update_display(self):
        if self.current_frame is None:
            return
        
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        if label_w < 10 or label_h < 10:
            return
        
        frame = self.current_frame
        h, w = frame.shape[:2]
        self.img_w, self.img_h = w, h
        
        scale = min(label_w / w, label_h / h)
        self.img_scale = scale
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        self.img_offset_x = (label_w - new_w) // 2
        self.img_offset_y = (label_h - new_h) // 2
        
        canvas = np.full((label_h, label_w, 3), 5, dtype=np.uint8)
        canvas[self.img_offset_y:self.img_offset_y+new_h, self.img_offset_x:self.img_offset_x+new_w] = resized
        
        qimg = QImage(canvas.data, label_w, label_h, 3 * label_w, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))
        self.image_label.update()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()
    
    def clear(self):
        self.current_frame = None
        self.image_label.clear()
        self.frame_label.setText("—")
    
    def img_to_widget(self, img_x, img_y):
        wx = self.img_offset_x + img_x * self.img_scale
        wy = self.img_offset_y + img_y * self.img_scale
        return wx, wy
    
    def widget_to_img(self, wx, wy):
        img_x = (wx - self.img_offset_x) / self.img_scale
        img_y = (wy - self.img_offset_y) / self.img_scale
        return img_x, img_y


class InteractiveImageLabel(QLabel):
    def __init__(self, frame_display):
        super().__init__()
        self.frame_display = frame_display
        self.setMouseTracking(True)
        self.dragging_corner = None
        self.dragging_tile = None
        self.corner_radius_normal = 4
        self.corner_radius_dragging = 1
    
    def get_tiles_for_frame(self):
        sampler = self.frame_display.parent_sampler
        frame_num = self.frame_display.current_frame_num
        if frame_num is None:
            return []
        
        result = []
        for tile in sampler.current_tiles:
            for inst in tile['instances']:
                if inst['frame'] == frame_num:
                    result.append((tile, inst))
                    break
        return result
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self.frame_display.current_frame is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        tiles = self.get_tiles_for_frame()
        
        for tile, inst in tiles:
            color_idx = tile.get('color_idx', 0) % len(TILE_COLORS)
            r, g, b = TILE_COLORS[color_idx]
            up_edge = inst.get('up_edge', 0)
            
            corners = inst['corners']
            widget_corners = [self.frame_display.img_to_widget(c[0], c[1]) for c in corners]
            
            # Draw filled quad
            painter.setBrush(QBrush(QColor(r, g, b, 50)))
            painter.setPen(QPen(QColor(r, g, b, 200), 2))
            
            from PyQt6.QtGui import QPolygonF
            from PyQt6.QtCore import QPointF
            polygon = QPolygonF([QPointF(c[0], c[1]) for c in widget_corners])
            painter.drawPolygon(polygon)
            
            # Draw "up" edge in blue
            up_c1 = widget_corners[up_edge]
            up_c2 = widget_corners[(up_edge + 1) % 4]
            painter.setPen(QPen(QColor(50, 150, 255), 3))
            painter.drawLine(QPointF(up_c1[0], up_c1[1]), QPointF(up_c2[0], up_c2[1]))
            
            # Draw arrow along the up edge
            mid_x = (up_c1[0] + up_c2[0]) / 2
            mid_y = (up_c1[1] + up_c2[1]) / 2
            dx = up_c2[0] - up_c1[0]
            dy = up_c2[1] - up_c1[1]
            length = (dx**2 + dy**2) ** 0.5
            if length > 0:
                dx, dy = dx / length, dy / length
                arrow_len = min(20, length / 3)
                # Arrow tip
                tip_x = mid_x + dx * arrow_len / 2
                tip_y = mid_y + dy * arrow_len / 2
                # Arrow base
                base_x = mid_x - dx * arrow_len / 2
                base_y = mid_y - dy * arrow_len / 2
                # Arrow head
                head_size = 6
                perp_x, perp_y = -dy, dx
                painter.setBrush(QBrush(QColor(50, 150, 255)))
                arrow_head = QPolygonF([
                    QPointF(tip_x, tip_y),
                    QPointF(tip_x - dx * head_size + perp_x * head_size / 2, tip_y - dy * head_size + perp_y * head_size / 2),
                    QPointF(tip_x - dx * head_size - perp_x * head_size / 2, tip_y - dy * head_size - perp_y * head_size / 2)
                ])
                painter.drawPolygon(arrow_head)
                painter.drawLine(QPointF(base_x, base_y), QPointF(tip_x - dx * head_size / 2, tip_y - dy * head_size / 2))
            
            # Draw corners
            painter.setBrush(QBrush(QColor(r, g, b, 255)))
            painter.setPen(QPen(QColor(r, g, b, 200), 2))
            for i, (wx, wy) in enumerate(widget_corners):
                is_dragging_this = (self.dragging_corner and 
                                    self.dragging_corner[0] == tile['id'] and 
                                    self.dragging_corner[1] == i)
                radius = self.corner_radius_dragging if is_dragging_this else self.corner_radius_normal
                painter.drawEllipse(QPointF(wx, wy), radius, radius)
            
            # Draw tile ID
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
            center_x = sum(c[0] for c in widget_corners) / 4
            center_y = sum(c[1] for c in widget_corners) / 4
            painter.drawText(QPointF(center_x - 15, center_y + 5), tile['id'])
        
        painter.end()
    
    def find_corner_at(self, pos):
        tiles = self.get_tiles_for_frame()
        
        for tile, inst in tiles:
            corners = inst['corners']
            for i, corner in enumerate(corners):
                wx, wy = self.frame_display.img_to_widget(corner[0], corner[1])
                dist = ((pos.x() - wx) ** 2 + (pos.y() - wy) ** 2) ** 0.5
                if dist <= self.corner_radius_normal + 6:
                    return (tile['id'], i, inst)
        return None
    
    def find_tile_at(self, pos):
        tiles = self.get_tiles_for_frame()
        
        for tile, inst in tiles:
            corners = inst['corners']
            widget_corners = [self.frame_display.img_to_widget(c[0], c[1]) for c in corners]
            
            x, y = pos.x(), pos.y()
            n = len(widget_corners)
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = widget_corners[i]
                xj, yj = widget_corners[j]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            
            if inside:
                return (tile, inst)
        return None
    
    def find_edge_at(self, pos):
        """Find if click is near an edge (but not a corner). Returns (tile, inst, edge_index) or None."""
        tiles = self.get_tiles_for_frame()
        
        for tile, inst in tiles:
            corners = inst['corners']
            widget_corners = [self.frame_display.img_to_widget(c[0], c[1]) for c in corners]
            
            # First check we're not on a corner
            for corner in corners:
                wx, wy = self.frame_display.img_to_widget(corner[0], corner[1])
                dist = ((pos.x() - wx) ** 2 + (pos.y() - wy) ** 2) ** 0.5
                if dist <= self.corner_radius_normal + 6:
                    return None  # On a corner, not an edge
            
            # Check distance to each edge
            for i in range(4):
                c1 = widget_corners[i]
                c2 = widget_corners[(i + 1) % 4]
                
                # Point to line segment distance
                x, y = pos.x(), pos.y()
                x1, y1 = c1
                x2, y2 = c2
                
                dx, dy = x2 - x1, y2 - y1
                length_sq = dx * dx + dy * dy
                if length_sq == 0:
                    continue
                
                t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / length_sq))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = ((x - proj_x) ** 2 + (y - proj_y) ** 2) ** 0.5
                
                if dist <= 10 and t > 0.1 and t < 0.9:  # Near edge but not at corners
                    return (tile, inst, i)
        
        return None
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            corner_result = self.find_corner_at(event.pos())
            if corner_result:
                tile_id, corner_idx, inst = corner_result
                self.dragging_corner = (tile_id, corner_idx, inst)
                return
            
            # Check if clicking on an edge to set up direction
            edge_result = self.find_edge_at(event.pos())
            if edge_result:
                tile, inst, edge_idx = edge_result
                inst['up_edge'] = edge_idx
                self.update()
                self.frame_display.parent_sampler.auto_save()
                return
            
            tile_result = self.find_tile_at(event.pos())
            if tile_result:
                tile, inst = tile_result
                original_corners = [[c[0], c[1]] for c in inst['corners']]
                img_x, img_y = self.frame_display.widget_to_img(event.pos().x(), event.pos().y())
                self.dragging_tile = (inst, (img_x, img_y), original_corners)
                return
            
            img_x, img_y = self.frame_display.widget_to_img(event.pos().x(), event.pos().y())
            
            if 0 <= img_x <= self.frame_display.img_w and 0 <= img_y <= self.frame_display.img_h:
                self.frame_display.parent_sampler.add_tile_at(img_x, img_y)
    
    def mouseMoveEvent(self, event):
        if self.dragging_corner:
            tile_id, corner_idx, inst = self.dragging_corner
            img_x, img_y = self.frame_display.widget_to_img(event.pos().x(), event.pos().y())
            
            img_x = max(0, min(self.frame_display.img_w, img_x))
            img_y = max(0, min(self.frame_display.img_h, img_y))
            
            inst['corners'][corner_idx] = [img_x, img_y]
            self.update()
        
        elif self.dragging_tile:
            inst, (start_x, start_y), original_corners = self.dragging_tile
            img_x, img_y = self.frame_display.widget_to_img(event.pos().x(), event.pos().y())
            
            dx = img_x - start_x
            dy = img_y - start_y
            
            for i, orig in enumerate(original_corners):
                new_x = orig[0] + dx
                new_y = orig[1] + dy
                new_x = max(0, min(self.frame_display.img_w, new_x))
                new_y = max(0, min(self.frame_display.img_h, new_y))
                inst['corners'][i] = [new_x, new_y]
            
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            was_dragging = self.dragging_corner or self.dragging_tile
            self.dragging_corner = None
            self.dragging_tile = None
            self.update()
            
            if was_dragging:
                self.frame_display.parent_sampler.auto_save()


class TileLabeler(QMainWindow):
    def __init__(self, videos_dir="videos"):
        super().__init__()
        self.videos_dir = videos_dir
        
        self.videos = []
        self.total_frames = 0
        self._scan_videos()
        
        self.history = []
        self.history_index = -1
        
        self.current_video_path = None
        self.current_start_frame = None
        self.current_total_frames = None
        
        self.current_tiles = []
        self.tile_counter = 0
        
        self.all_tiles = {}
        
        self.setWindowTitle("Tile Labeler")
        self.setStyleSheet(STYLE)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        frames_widget = QWidget()
        frames_layout = QHBoxLayout(frames_widget)
        frames_layout.setContentsMargins(15, 15, 15, 15)
        frames_layout.setSpacing(12)
        
        self.displays = []
        for i in range(3):
            display = FrameDisplay(i, self)
            self.displays.append(display)
            frames_layout.addWidget(display)
        
        main_layout.addWidget(frames_widget, stretch=1)
        
        control_panel = QFrame()
        control_panel.setObjectName("controlPanel")
        control_panel.setFixedWidth(280)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(20, 25, 20, 25)
        control_layout.setSpacing(10)
        
        title = QLabel("Tile Labeler")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(title)
        
        sep1 = QFrame()
        sep1.setObjectName("separator")
        sep1.setFixedHeight(1)
        control_layout.addWidget(sep1)
        
        info_label = QLabel("Videos Folder")
        info_label.setObjectName("info")
        control_layout.addWidget(info_label)
        
        folder_label = QLabel(f"📁  {self.videos_dir}")
        folder_label.setStyleSheet("color: #aaa; font-size: 12px; padding: 8px; background: #0a0a12; border-radius: 4px;")
        folder_label.setWordWrap(True)
        control_layout.addWidget(folder_label)
        
        count_label = QLabel(f"{len(self.videos)} videos, {self.total_frames:,} frames")
        count_label.setObjectName("info")
        count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(count_label)
        
        sep2 = QFrame()
        sep2.setObjectName("separator")
        sep2.setFixedHeight(1)
        control_layout.addWidget(sep2)
        
        step_label = QLabel("Frame Step Size  [Q/E]")
        step_label.setObjectName("info")
        control_layout.addWidget(step_label)
        
        step_row = QHBoxLayout()
        step_row.setSpacing(8)
        
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 100)
        self.step_spin.setValue(1)
        self.step_spin.setFixedSize(80, 38)
        self.step_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.step_spin.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.step_spin.valueChanged.connect(self._on_step_changed)
        step_row.addWidget(self.step_spin)
        
        minus_btn = QPushButton("−")
        minus_btn.setObjectName("stepBtn")
        minus_btn.setFixedSize(38, 38)
        minus_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        minus_btn.clicked.connect(lambda: self.step_spin.setValue(self.step_spin.value() - 1))
        step_row.addWidget(minus_btn)
        
        plus_btn = QPushButton("+")
        plus_btn.setObjectName("stepBtn")
        plus_btn.setFixedSize(38, 38)
        plus_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        plus_btn.clicked.connect(lambda: self.step_spin.setValue(self.step_spin.value() + 1))
        step_row.addWidget(plus_btn)
        
        step_row.addStretch()
        control_layout.addLayout(step_row)
        
        sep3 = QFrame()
        sep3.setObjectName("separator")
        sep3.setFixedHeight(1)
        control_layout.addWidget(sep3)
        
        current_label = QLabel("Current Sample")
        current_label.setObjectName("info")
        control_layout.addWidget(current_label)
        
        self.video_info = QLabel("Loading...")
        self.video_info.setObjectName("videoInfo")
        self.video_info.setWordWrap(True)
        self.video_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(self.video_info)
        
        self.history_label = QLabel("")
        self.history_label.setObjectName("info")
        self.history_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(self.history_label)
        
        sep4 = QFrame()
        sep4.setObjectName("separator")
        sep4.setFixedHeight(1)
        control_layout.addWidget(sep4)
        
        tiles_label = QLabel("Tile Labels")
        tiles_label.setObjectName("info")
        control_layout.addWidget(tiles_label)
        
        self.stats_label = QLabel("0 labeled samples")
        self.stats_label.setObjectName("info")
        self.stats_label.setStyleSheet("color: #00d4ff; font-size: 10px;")
        control_layout.addWidget(self.stats_label)
        
        self.tile_list = QListWidget()
        self.tile_list.setFixedHeight(120)
        self.tile_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        control_layout.addWidget(self.tile_list)
        
        tile_btn_row = QHBoxLayout()
        tile_btn_row.setSpacing(8)
        
        add_tile_btn = QPushButton("+ Add Tile")
        add_tile_btn.setObjectName("smallBtn")
        add_tile_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_tile_btn.clicked.connect(self.add_tile)
        tile_btn_row.addWidget(add_tile_btn)
        
        del_tile_btn = QPushButton("Delete")
        del_tile_btn.setObjectName("deleteBtn")
        del_tile_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        del_tile_btn.clicked.connect(self.delete_selected_tile)
        tile_btn_row.addWidget(del_tile_btn)
        
        control_layout.addLayout(tile_btn_row)
        
        control_layout.addStretch()
        
        btn = QPushButton("⟳  Sample  [D]")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(self.sample_new)
        control_layout.addWidget(btn)
        
        shortcuts_label = QLabel("A/D: prev/next  •  Q/E: step")
        shortcuts_label.setObjectName("info")
        shortcuts_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(shortcuts_label)
        
        main_layout.addWidget(control_panel)
        
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self.sample_new)
        QShortcut(QKeySequence(Qt.Key.Key_R), self, self.sample_new)
        QShortcut(QKeySequence(Qt.Key.Key_D), self, self.next_sample)
        QShortcut(QKeySequence(Qt.Key.Key_A), self, self.prev_sample)
        QShortcut(QKeySequence(Qt.Key.Key_Q), self, lambda: self.step_spin.setValue(self.step_spin.value() - 1))
        QShortcut(QKeySequence(Qt.Key.Key_E), self, lambda: self.step_spin.setValue(self.step_spin.value() + 1))
        QShortcut(QKeySequence(Qt.Key.Key_T), self, self.add_tile)
        
        self._load_from_file()
    
    def _scan_videos(self):
        if not os.path.isdir(self.videos_dir):
            return
        
        for fname in os.listdir(self.videos_dir):
            if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                path = os.path.join(self.videos_dir, fname)
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if frame_count > 0:
                        self.videos.append((path, frame_count))
                        self.total_frames += frame_count
                cap.release()
        
        self.videos.sort(key=lambda x: x[0])
    
    def _sample_video_weighted(self):
        if not self.videos or self.total_frames == 0:
            return None, 0
        
        target = random.randint(0, self.total_frames - 1)
        
        cumulative = 0
        for path, frame_count in self.videos:
            if cumulative + frame_count > target:
                return path, frame_count
            cumulative += frame_count
        
        return self.videos[-1]
    
    def _get_sample_key(self):
        if self.current_video_path and self.current_start_frame is not None:
            return f"{self.current_video_path}:{self.current_start_frame}"
        return None
    
    def _save_current_tiles(self):
        key = self._get_sample_key()
        if key:
            if self.current_tiles:
                self.all_tiles[key] = [tile.copy() for tile in self.current_tiles]
            elif key in self.all_tiles:
                del self.all_tiles[key]
    
    def _load_tiles_for_sample(self):
        key = self._get_sample_key()
        if key and key in self.all_tiles:
            self.current_tiles = [tile.copy() for tile in self.all_tiles[key]]
        else:
            self.current_tiles = []
        self._update_tile_list()
    
    def _update_tile_list(self):
        self.tile_list.clear()
        for tile in self.current_tiles:
            color_idx = tile.get('color_idx', 0) % len(TILE_COLORS)
            r, g, b = TILE_COLORS[color_idx]
            item = QListWidgetItem(f"● {tile['id']}")
            item.setForeground(QColor(r, g, b))
            self.tile_list.addItem(item)
        self._update_stats()
    
    def _update_stats(self):
        total_samples = len(self.all_tiles)
        total_tiles = sum(len(tiles) for tiles in self.all_tiles.values())
        self.stats_label.setText(f"{total_samples} samples, {total_tiles} tiles")
    
    def _update_history_label(self):
        if self.history:
            self.history_label.setText(f"Sample {self.history_index + 1} / {len(self.history)}")
        else:
            self.history_label.setText("")
    
    def _on_step_changed(self):
        if self.current_video_path is None:
            return
        self._save_current_tiles()
        self._load_frames(self.current_video_path, self.current_start_frame, self.current_total_frames)
        
        # Update history entry with new step size
        if self.history_index >= 0 and self.history_index < len(self.history):
            step = self.step_spin.value()
            entry = self.history[self.history_index]
            self.history[self.history_index] = (entry[0], entry[1], entry[2], step)
        
        self.auto_save()
    
    def _load_frames(self, path, start, total):
        step = self.step_spin.value()
        frame_indices = [start, start + step, start + 2 * step]
        
        if frame_indices[-1] >= total:
            self.video_info.setText(f"Step too large\n(frame {frame_indices[-1]} >= {total})")
            return
        
        cap = cv2.VideoCapture(path)
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.displays[i].set_frame(frame, frame_idx)
        
        cap.release()
        
        name = os.path.basename(path)
        frames_str = ", ".join(str(f) for f in frame_indices)
        self.video_info.setText(f"{name}\n\nFrames: {frames_str}\n({total} total, step={step})")
        
        self._update_tile_frame_numbers(frame_indices)
    
    def _update_tile_frame_numbers(self, frame_indices):
        for tile in self.current_tiles:
            while len(tile['instances']) < 3:
                if tile['instances']:
                    new_inst = {'frame': 0, 'corners': [c.copy() for c in tile['instances'][-1]['corners']]}
                else:
                    new_inst = {'frame': 0, 'corners': [[100, 100], [200, 100], [200, 200], [100, 200]]}
                tile['instances'].append(new_inst)
            
            for i, frame_num in enumerate(frame_indices):
                tile['instances'][i]['frame'] = frame_num
        
        for display in self.displays:
            display.image_label.update()
    
    def _load_sample(self, path, start, total, step=1):
        self._save_current_tiles()
        
        self.current_video_path = path
        self.current_start_frame = start
        self.current_total_frames = total
        
        self.step_spin.blockSignals(True)
        self.step_spin.setValue(step)
        self.step_spin.blockSignals(False)
        
        self._load_tiles_for_sample()
        self._load_frames(path, start, total)
        self._update_history_label()
    
    def sample_new(self):
        if not self.videos:
            self.video_info.setText("No videos found!")
            return
        
        self._save_current_tiles()
        
        path, total = self._sample_video_weighted()
        if path is None:
            return
        
        self.step_spin.blockSignals(True)
        self.step_spin.setValue(1)
        self.step_spin.blockSignals(False)
        
        step = 1
        min_frames_needed = 1 + 2 * step
        
        if total < min_frames_needed:
            self.video_info.setText(f"Video too short\n({total} frames, need {min_frames_needed})")
            return
        
        start = random.randint(0, total - min_frames_needed)
        
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        
        self.history.append((path, start, total, step))
        self.history_index = len(self.history) - 1
        
        self._load_sample(path, start, total, step)
        self.auto_save()
    
    def next_sample(self):
        if self.history_index < len(self.history) - 1:
            self._save_current_tiles()
            self.history_index += 1
            entry = self.history[self.history_index]
            path, start, total = entry[0], entry[1], entry[2]
            step = entry[3] if len(entry) > 3 else 1
            self._load_sample(path, start, total, step)
            self.auto_save()
        else:
            self.sample_new()
    
    def prev_sample(self):
        if self.history_index > 0:
            self._save_current_tiles()
            self.history_index -= 1
            entry = self.history[self.history_index]
            path, start, total = entry[0], entry[1], entry[2]
            step = entry[3] if len(entry) > 3 else 1
            self._load_sample(path, start, total, step)
            self.auto_save()
    
    def add_tile(self):
        if self.current_video_path is None:
            return
        
        self.tile_counter += 1
        tile_id = f"T{self.tile_counter:03d}"
        
        frame_nums = [d.current_frame_num for d in self.displays]
        
        cx, cy = 200, 200
        size = 80
        default_corners = [
            [cx - size, cy - size],
            [cx + size, cy - size],
            [cx + size, cy + size],
            [cx - size, cy + size]
        ]
        
        tile = {
            'id': tile_id,
            'color_idx': len(self.current_tiles) % len(TILE_COLORS),
            'instances': [
                {'frame': frame_nums[i], 'corners': [c.copy() for c in default_corners], 'up_edge': 0}
                for i in range(3)
            ]
        }
        
        self.current_tiles.append(tile)
        self._update_tile_list()
        
        for display in self.displays:
            display.image_label.update()
        
        self.auto_save()
    
    def add_tile_at(self, cx, cy):
        if self.current_video_path is None:
            return
        
        self.tile_counter += 1
        tile_id = f"T{self.tile_counter:03d}"
        
        frame_nums = [d.current_frame_num for d in self.displays]
        
        size = 60
        default_corners = [
            [cx - size, cy - size],
            [cx + size, cy - size],
            [cx + size, cy + size],
            [cx - size, cy + size]
        ]
        
        tile = {
            'id': tile_id,
            'color_idx': len(self.current_tiles) % len(TILE_COLORS),
            'instances': [
                {'frame': frame_nums[i], 'corners': [c.copy() for c in default_corners], 'up_edge': 0}
                for i in range(3)
            ]
        }
        
        self.current_tiles.append(tile)
        self._update_tile_list()
        
        for display in self.displays:
            display.image_label.update()
        
        self.auto_save()
    
    def delete_selected_tile(self):
        row = self.tile_list.currentRow()
        if row >= 0 and row < len(self.current_tiles):
            del self.current_tiles[row]
            self._update_tile_list()
            for display in self.displays:
                display.image_label.update()
            self.auto_save()
    
    def auto_save(self):
        self._save_current_tiles()
        self._update_stats()
        
        save_data = {
            'tiles': self.all_tiles,
            'tile_counter': self.tile_counter,
            'history': self.history,
            'history_index': self.history_index,
            'current': {
                'video_path': self.current_video_path,
                'start_frame': self.current_start_frame,
                'total_frames': self.current_total_frames,
                'step_size': self.step_spin.value()
            }
        }
        
        with open(SAVE_FILE, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def _load_from_file(self):
        if not os.path.exists(SAVE_FILE):
            self.video_info.setText("Press D or Sample to begin")
            return
        
        try:
            with open(SAVE_FILE, 'r') as f:
                data = json.load(f)
            
            self.all_tiles = data.get('tiles', {})
            self.tile_counter = data.get('tile_counter', 0)
            self.history = data.get('history', [])
            
            # Always go to last sample in history
            if self.history:
                self.history_index = len(self.history) - 1
                entry = self.history[self.history_index]
                path, start, total = entry[0], entry[1], entry[2]
                step = entry[3] if len(entry) > 3 else 1
                
                if os.path.exists(path):
                    self.current_video_path = path
                    self.current_start_frame = start
                    self.current_total_frames = total
                    
                    self.step_spin.blockSignals(True)
                    self.step_spin.setValue(step)
                    self.step_spin.blockSignals(False)
                    
                    self._load_tiles_for_sample()
                    self._load_frames(path, start, total)
                    self._update_history_label()
                    self._update_stats()
                    return
            
            self.video_info.setText("Press D or Sample to begin")
        except Exception as e:
            print(f"Error loading: {e}")
            self.video_info.setText("Press D or Sample to begin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="videos", help="Videos directory")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    window = TileLabeler(args.videos)
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
