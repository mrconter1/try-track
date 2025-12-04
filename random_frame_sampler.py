"""Random Video Frame Sampler - Shows 3 consecutive frames from a random video."""

import sys
import os
import random
import argparse
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy, QSpinBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QShortcut, QKeySequence


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
"""


class FrameDisplay(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("frameBox")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        self.image_label = QLabel()
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
        scale = min(label_w / w, label_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        canvas = np.full((label_h, label_w, 3), 5, dtype=np.uint8)
        y, x = (label_h - new_h) // 2, (label_w - new_w) // 2
        canvas[y:y+new_h, x:x+new_w] = resized
        
        qimg = QImage(canvas.data, label_w, label_h, 3 * label_w, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()
    
    def clear(self):
        self.current_frame = None
        self.image_label.clear()
        self.frame_label.setText("—")


class RandomFrameSampler(QMainWindow):
    def __init__(self, videos_dir="videos"):
        super().__init__()
        self.videos_dir = videos_dir
        
        # Video info: list of (path, frame_count)
        self.videos = []
        self.total_frames = 0
        self._scan_videos()
        
        # Current state
        self.current_video_path = None
        self.current_start_frame = None
        self.current_total_frames = None
        
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
        for _ in range(3):
            display = FrameDisplay()
            self.displays.append(display)
            frames_layout.addWidget(display)
        
        main_layout.addWidget(frames_widget, stretch=1)
        
        # Right: Control panel
        control_panel = QFrame()
        control_panel.setObjectName("controlPanel")
        control_panel.setFixedWidth(280)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(20, 25, 20, 25)
        control_layout.setSpacing(15)
        
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
        
        control_layout.addSpacing(10)
        
        sep2 = QFrame()
        sep2.setObjectName("separator")
        sep2.setFixedHeight(1)
        control_layout.addWidget(sep2)
        
        control_layout.addSpacing(5)
        
        # Step size control
        step_label = QLabel("Frame Step Size")
        step_label.setObjectName("info")
        control_layout.addWidget(step_label)
        
        step_row = QHBoxLayout()
        step_row.setSpacing(8)
        
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 100)
        self.step_spin.setValue(3)
        self.step_spin.setFixedSize(80, 38)
        self.step_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        
        control_layout.addSpacing(10)
        
        sep3 = QFrame()
        sep3.setObjectName("separator")
        sep3.setFixedHeight(1)
        control_layout.addWidget(sep3)
        
        control_layout.addSpacing(5)
        
        current_label = QLabel("Current Sample")
        current_label.setObjectName("info")
        control_layout.addWidget(current_label)
        
        self.video_info = QLabel("Press Sample to begin")
        self.video_info.setObjectName("videoInfo")
        self.video_info.setWordWrap(True)
        self.video_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(self.video_info)
        
        control_layout.addStretch()
        
        btn = QPushButton("⟳  Sample  [Space]")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(self.sample)
        control_layout.addWidget(btn)
        
        shortcuts_label = QLabel("Shortcuts: Space, R")
        shortcuts_label.setObjectName("info")
        shortcuts_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(shortcuts_label)
        
        main_layout.addWidget(control_panel)
        
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self.sample)
        QShortcut(QKeySequence(Qt.Key.Key_R), self, self.sample)
    
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
        """Sample a video weighted by frame count (more frames = higher chance)."""
        if not self.videos or self.total_frames == 0:
            return None, 0
        
        # Pick a random frame index across all videos
        target = random.randint(0, self.total_frames - 1)
        
        # Find which video contains this frame
        cumulative = 0
        for path, frame_count in self.videos:
            if cumulative + frame_count > target:
                return path, frame_count
            cumulative += frame_count
        
        # Fallback
        return self.videos[-1]
    
    def _on_step_changed(self):
        """Update frames 2 and 3 when step size changes, keeping frame 1."""
        if self.current_video_path is None:
            return
        self._load_frames(self.current_video_path, self.current_start_frame, self.current_total_frames)
    
    def _load_frames(self, path, start, total):
        """Load 3 frames from video starting at 'start' with current step."""
        step = self.step_spin.value()
        frame_indices = [start, start + step, start + 2 * step]
        
        # Check if frames are valid
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
    
    def sample(self):
        if not self.videos:
            self.video_info.setText("No videos found!")
            return
        
        path, total = self._sample_video_weighted()
        if path is None:
            return
        
        step = self.step_spin.value()
        min_frames_needed = 1 + 2 * step
        
        if total < min_frames_needed:
            self.video_info.setText(f"Video too short\n({total} frames, need {min_frames_needed})")
            return
        
        start = random.randint(0, total - min_frames_needed)
        
        # Store current state
        self.current_video_path = path
        self.current_start_frame = start
        self.current_total_frames = total
        
        self._load_frames(path, start, total)


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
