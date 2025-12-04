"""Random Video Frame Sampler - Shows 3 consecutive frames from a random video."""

import sys
import os
import random
import argparse
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QPushButton, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QFont, QShortcut, QKeySequence


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
    font-size: 24px;
    font-weight: bold;
    font-family: 'Segoe UI', sans-serif;
}
QLabel#info {
    color: #666;
    font-size: 12px;
}
QLabel#frameLabel {
    color: #00d4ff;
    font-size: 13px;
    padding: 8px;
}
QFrame#frameBox {
    background: #0a0a12;
    border: 2px solid #2a2a3e;
    border-radius: 12px;
}
QFrame#frameBox:hover {
    border-color: #00d4ff;
}
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #00d4ff, stop:1 #0099cc);
    color: #000;
    font-size: 14px;
    font-weight: bold;
    font-family: 'Segoe UI', sans-serif;
    padding: 12px 40px;
    border: none;
    border-radius: 8px;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #33e0ff, stop:1 #00b8e6);
}
QPushButton:pressed {
    background: #0088aa;
}
"""


class FrameDisplay(QFrame):
    def __init__(self, size=350):
        super().__init__()
        self.setObjectName("frameBox")
        self.size = size
        self.setFixedSize(size + 20, size + 60)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        self.image_label = QLabel()
        self.image_label.setFixedSize(size, size)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background: #050508; border-radius: 8px;")
        layout.addWidget(self.image_label)
        
        self.frame_label = QLabel("—")
        self.frame_label.setObjectName("frameLabel")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.frame_label)
    
    def set_frame(self, frame, frame_num):
        h, w = frame.shape[:2]
        scale = self.size / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        
        canvas = np.full((self.size, self.size, 3), 5, dtype=np.uint8)
        y, x = (self.size - frame.shape[0]) // 2, (self.size - frame.shape[1]) // 2
        canvas[y:y+frame.shape[0], x:x+frame.shape[1]] = frame
        
        h, w, ch = canvas.shape
        qimg = QImage(canvas.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))
        self.frame_label.setText(f"Frame {frame_num}")
    
    def clear(self):
        self.image_label.clear()
        self.frame_label.setText("—")


class RandomFrameSampler(QMainWindow):
    def __init__(self, videos_dir="videos"):
        super().__init__()
        self.videos_dir = videos_dir
        self.videos = [os.path.join(videos_dir, f) for f in os.listdir(videos_dir) 
                       if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))] if os.path.isdir(videos_dir) else []
        
        self.setWindowTitle("Random Frame Sampler")
        self.setStyleSheet(STYLE)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 30, 40, 30)
        
        title = QLabel("Random Frame Sampler")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        self.info = QLabel(f"{len(self.videos)} videos found in '{self.videos_dir}'")
        self.info.setObjectName("info")
        self.info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info)
        
        layout.addStretch()
        
        frames_row = QHBoxLayout()
        frames_row.setSpacing(20)
        frames_row.addStretch()
        
        self.displays = []
        for _ in range(3):
            display = FrameDisplay(350)
            self.displays.append(display)
            frames_row.addWidget(display)
        
        frames_row.addStretch()
        layout.addLayout(frames_row)
        
        layout.addStretch()
        
        self.video_info = QLabel("Press Space or click Sample to load frames")
        self.video_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_info.setStyleSheet("color: #555; font-size: 13px;")
        layout.addWidget(self.video_info)
        
        btn = QPushButton("Sample  [Space]")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(self.sample)
        btn.setFixedWidth(200)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self.sample)
        QShortcut(QKeySequence(Qt.Key.Key_R), self, self.sample)
    
    def sample(self):
        if not self.videos:
            self.video_info.setText("No videos found!")
            return
        
        path = random.choice(self.videos)
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total < 3:
            cap.release()
            self.video_info.setText("Video too short")
            return
        
        start = random.randint(0, total - 3)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        for i in range(3):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.displays[i].set_frame(frame, start + i)
        
        cap.release()
        self.video_info.setText(f"{os.path.basename(path)}  •  Frames {start}–{start+2} of {total}")


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
