"""Random Video Frame Sampler - Shows 3 consecutive frames from a random video with tile labeling."""

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

# Colors for different tiles
TILE_COLORS = [
    (255, 100, 100),  # Red
    (100, 255, 100),  # Green
    (100, 100, 255),  # Blue
    (255, 255, 100),  # Yellow
    (255, 100, 255),  # Magenta
    (100, 255, 255),  # Cyan
    (255, 180, 100),  # Orange
    (180, 100, 255),  # Purple
]


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
        
        # Image display transform info
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
        """Convert image coordinates to widget coordinates."""
        wx = self.img_offset_x + img_x * self.img_scale
        wy = self.img_offset_y + img_y * self.img_scale
        return wx, wy
    
    def widget_to_img(self, wx, wy):
        """Convert widget coordinates to image coordinates."""
        img_x = (wx - self.img_offset_x) / self.img_scale
        img_y = (wy - self.img_offset_y) / self.img_scale
        return img_x, img_y


class InteractiveImageLabel(QLabel):
    def __init__(self, frame_display):
        super().__init__()
        self.frame_display = frame_display
        self.setMouseTracking(True)
        self.dragging_corner = None  # (tile_id, corner_index, inst)
        self.dragging_tile = None    # (inst, start_pos, original_corners)
        self.corner_radius = 4
    
    def get_tiles_for_frame(self):
        """Get tiles that have data for this frame."""
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
            
            corners = inst['corners']
            widget_corners = [self.frame_display.img_to_widget(c[0], c[1]) for c in corners]
            
            # Draw filled quad
            painter.setBrush(QBrush(QColor(r, g, b, 50)))
            painter.setPen(QPen(QColor(r, g, b, 200), 2))
            
            from PyQt6.QtGui import QPolygonF
            from PyQt6.QtCore import QPointF
            polygon = QPolygonF([QPointF(c[0], c[1]) for c in widget_corners])
            painter.drawPolygon(polygon)
            
            # Draw corners
            painter.setBrush(QBrush(QColor(r, g, b, 255)))
            for wx, wy in widget_corners:
                painter.drawEllipse(QPointF(wx, wy), self.corner_radius, self.corner_radius)
            
            # Draw tile ID
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
            center_x = sum(c[0] for c in widget_corners) / 4
            center_y = sum(c[1] for c in widget_corners) / 4
            painter.drawText(QPointF(center_x - 15, center_y + 5), tile['id'])
        
        painter.end()
    
    def find_corner_at(self, pos):
        """Find if there's a corner at the given position."""
        tiles = self.get_tiles_for_frame()
        
        for tile, inst in tiles:
            corners = inst['corners']
            for i, corner in enumerate(corners):
                wx, wy = self.frame_display.img_to_widget(corner[0], corner[1])
                dist = ((pos.x() - wx) ** 2 + (pos.y() - wy) ** 2) ** 0.5
                if dist <= self.corner_radius + 6:
                    return (tile['id'], i, inst)
        return None
    
    def find_tile_at(self, pos):
        """Find if point is inside a tile polygon."""
        tiles = self.get_tiles_for_frame()
        
        for tile, inst in tiles:
            corners = inst['corners']
            widget_corners = [self.frame_display.img_to_widget(c[0], c[1]) for c in corners]
            
            # Point in polygon test (ray casting)
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
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # First check corners (priority)
            corner_result = self.find_corner_at(event.pos())
            if corner_result:
                tile_id, corner_idx, inst = corner_result
                self.dragging_corner = (tile_id, corner_idx, inst)
                return
            
            # Then check if inside a tile
            tile_result = self.find_tile_at(event.pos())
            if tile_result:
                tile, inst = tile_result
                # Store original corners for relative dragging
                original_corners = [[c[0], c[1]] for c in inst['corners']]
                img_x, img_y = self.frame_display.widget_to_img(event.pos().x(), event.pos().y())
                self.dragging_tile = (inst, (img_x, img_y), original_corners)
    
    def mouseMoveEvent(self, event):
        if self.dragging_corner:
            tile_id, corner_idx, inst = self.dragging_corner
            img_x, img_y = self.frame_display.widget_to_img(event.pos().x(), event.pos().y())
            
            # Clamp to image bounds
            img_x = max(0, min(self.frame_display.img_w, img_x))
            img_y = max(0, min(self.frame_display.img_h, img_y))
            
            inst['corners'][corner_idx] = [img_x, img_y]
            self.update()
        
        elif self.dragging_tile:
            inst, (start_x, start_y), original_corners = self.dragging_tile
            img_x, img_y = self.frame_display.widget_to_img(event.pos().x(), event.pos().y())
            
            # Calculate delta
            dx = img_x - start_x
            dy = img_y - start_y
            
            # Move all corners
            for i, orig in enumerate(original_corners):
                new_x = orig[0] + dx
                new_y = orig[1] + dy
                # Clamp
                new_x = max(0, min(self.frame_display.img_w, new_x))
                new_y = max(0, min(self.frame_display.img_h, new_y))
                inst['corners'][i] = [new_x, new_y]
            
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging_corner = None
            self.dragging_tile = None


class RandomFrameSampler(QMainWindow):
    def __init__(self, videos_dir="videos"):
        super().__init__()
        self.videos_dir = videos_dir
        
        # Video info: list of (path, frame_count)
        self.videos = []
        self.total_frames = 0
        self._scan_videos()
        
        # Sample history: list of (path, start_frame, total_frames)
        self.history = []
        self.history_index = -1
        
        # Current state
        self.current_video_path = None
        self.current_start_frame = None
        self.current_total_frames = None
        
        # Tile labels for current sample
        self.current_tiles = []  # List of tile dicts
        self.tile_counter = 0
        
        # All saved tiles (keyed by sample identifier)
        self.all_tiles = {}  # { "video_path:start_frame": [tiles] }
        
        self.setWindowTitle("Random Frame Sampler")
        self.setStyleSheet(STYLE)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Left: Frames area
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
        
        # Right: Control panel
        control_panel = QFrame()
        control_panel.setObjectName("controlPanel")
        control_panel.setFixedWidth(280)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(20, 25, 20, 25)
        control_layout.setSpacing(10)
        
        title = QLabel("Frame Sampler")
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
        
        # Step size control
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
        
        self.video_info = QLabel("Press D or Sample to begin")
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
        
        # Tile labeling section
        tiles_label = QLabel("Tile Labels")
        tiles_label.setObjectName("info")
        control_layout.addWidget(tiles_label)
        
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
        
        save_btn = QPushButton("💾 Save Tiles")
        save_btn.setObjectName("smallBtn")
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.clicked.connect(self.save_tiles)
        control_layout.addWidget(save_btn)
        
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
        
        # Shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self.sample_new)
        QShortcut(QKeySequence(Qt.Key.Key_R), self, self.sample_new)
        QShortcut(QKeySequence(Qt.Key.Key_D), self, self.next_sample)
        QShortcut(QKeySequence(Qt.Key.Key_A), self, self.prev_sample)
        QShortcut(QKeySequence(Qt.Key.Key_Q), self, lambda: self.step_spin.setValue(self.step_spin.value() - 1))
        QShortcut(QKeySequence(Qt.Key.Key_E), self, lambda: self.step_spin.setValue(self.step_spin.value() + 1))
        QShortcut(QKeySequence(Qt.Key.Key_T), self, self.add_tile)
        
        # Load existing tiles
        self._load_tiles_from_file()
    
    def _scan_videos(self):
        """Scan videos folder and get frame counts for weighted sampling."""
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
        """Sample a video weighted by frame count."""
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
        """Get unique key for current sample."""
        if self.current_video_path and self.current_start_frame is not None:
            return f"{self.current_video_path}:{self.current_start_frame}"
        return None
    
    def _save_current_tiles(self):
        """Save current tiles to the all_tiles dict."""
        key = self._get_sample_key()
        if key and self.current_tiles:
            self.all_tiles[key] = self.current_tiles.copy()
    
    def _load_tiles_for_sample(self):
        """Load tiles for current sample from all_tiles dict."""
        key = self._get_sample_key()
        if key and key in self.all_tiles:
            self.current_tiles = self.all_tiles[key].copy()
        else:
            self.current_tiles = []
        self._update_tile_list()
    
    def _update_tile_list(self):
        """Update the tile list widget."""
        self.tile_list.clear()
        for tile in self.current_tiles:
            color_idx = tile.get('color_idx', 0) % len(TILE_COLORS)
            r, g, b = TILE_COLORS[color_idx]
            item = QListWidgetItem(f"● {tile['id']}")
            item.setForeground(QColor(r, g, b))
            self.tile_list.addItem(item)
    
    def _update_history_label(self):
        if self.history:
            self.history_label.setText(f"Sample {self.history_index + 1} / {len(self.history)}")
        else:
            self.history_label.setText("")
    
    def _on_step_changed(self):
        """Update frames 2 and 3 when step size changes."""
        if self.current_video_path is None:
            return
        self._save_current_tiles()
        self._load_frames(self.current_video_path, self.current_start_frame, self.current_total_frames)
    
    def _load_frames(self, path, start, total):
        """Load 3 frames from video."""
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
        
        # Update tile instances for new frame numbers
        self._update_tile_frame_numbers(frame_indices)
    
    def _update_tile_frame_numbers(self, frame_indices):
        """Update tile instances to match current frame numbers."""
        for tile in self.current_tiles:
            # Ensure we have 3 instances
            while len(tile['instances']) < 3:
                # Copy from last instance or create default
                if tile['instances']:
                    new_inst = {'frame': 0, 'corners': [c.copy() for c in tile['instances'][-1]['corners']]}
                else:
                    new_inst = {'frame': 0, 'corners': [[100, 100], [200, 100], [200, 200], [100, 200]]}
                tile['instances'].append(new_inst)
            
            # Update frame numbers
            for i, frame_num in enumerate(frame_indices):
                tile['instances'][i]['frame'] = frame_num
        
        # Refresh displays
        for display in self.displays:
            display.image_label.update()
    
    def _load_sample(self, path, start, total):
        """Load a sample and update state."""
        self._save_current_tiles()
        
        self.current_video_path = path
        self.current_start_frame = start
        self.current_total_frames = total
        
        self._load_tiles_for_sample()
        self._load_frames(path, start, total)
        self._update_history_label()
    
    def sample_new(self):
        """Generate a new random sample."""
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
        
        self.history.append((path, start, total))
        self.history_index = len(self.history) - 1
        
        self._load_sample(path, start, total)
    
    def next_sample(self):
        """Go to next sample or generate new."""
        if self.history_index < len(self.history) - 1:
            self._save_current_tiles()
            self.history_index += 1
            path, start, total = self.history[self.history_index]
            self.step_spin.blockSignals(True)
            self.step_spin.setValue(1)
            self.step_spin.blockSignals(False)
            self._load_sample(path, start, total)
        else:
            self.sample_new()
    
    def prev_sample(self):
        """Go to previous sample."""
        if self.history_index > 0:
            self._save_current_tiles()
            self.history_index -= 1
            path, start, total = self.history[self.history_index]
            self.step_spin.blockSignals(True)
            self.step_spin.setValue(1)
            self.step_spin.blockSignals(False)
            self._load_sample(path, start, total)
    
    def add_tile(self):
        """Add a new tile label."""
        if self.current_video_path is None:
            return
        
        self.tile_counter += 1
        tile_id = f"T{self.tile_counter:03d}"
        
        # Get current frame numbers
        frame_nums = [d.current_frame_num for d in self.displays]
        
        # Default quad in center of image
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
                {'frame': frame_nums[i], 'corners': [c.copy() for c in default_corners]}
                for i in range(3)
            ]
        }
        
        self.current_tiles.append(tile)
        self._update_tile_list()
        
        # Refresh displays
        for display in self.displays:
            display.image_label.update()
    
    def delete_selected_tile(self):
        """Delete the selected tile."""
        row = self.tile_list.currentRow()
        if row >= 0 and row < len(self.current_tiles):
            del self.current_tiles[row]
            self._update_tile_list()
            for display in self.displays:
                display.image_label.update()
    
    def save_tiles(self):
        """Save all tiles to JSON file."""
        self._save_current_tiles()
        
        # Convert to serializable format
        save_data = {}
        for key, tiles in self.all_tiles.items():
            save_data[key] = tiles
        
        with open('tile_labels.json', 'w') as f:
            json.dump(save_data, f, indent=2)
        
        self.video_info.setText("Tiles saved to\ntile_labels.json")
    
    def _load_tiles_from_file(self):
        """Load tiles from JSON file."""
        if os.path.exists('tile_labels.json'):
            try:
                with open('tile_labels.json', 'r') as f:
                    self.all_tiles = json.load(f)
                # Find max tile counter
                for tiles in self.all_tiles.values():
                    for tile in tiles:
                        try:
                            num = int(tile['id'][1:])
                            self.tile_counter = max(self.tile_counter, num)
                        except:
                            pass
            except:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="videos", help="Videos directory")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    window = RandomFrameSampler(args.videos)
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
