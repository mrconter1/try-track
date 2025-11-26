import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import argparse
import os


class FrameOverlayViewer:
    """GUI to view current frame overlaid on previous frame at 50% opacity."""
    
    def __init__(self, root, video_path):
        self.root = root
        self.root.title("Frame Overlay Viewer")
        self.root.state('zoomed')
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_name = os.path.basename(video_path)
        
        # Frame state
        self.current_idx = 1  # Start at frame 1 so we have a previous frame
        self.prev_frame = None
        self.curr_frame = None
        self.photo_image = None
        
        # Opacity control
        self.opacity_var = tk.DoubleVar(value=0.5)
        
        self._build_ui()
        self._load_frames()
        
        # Bind keys
        self.root.bind("<Right>", lambda e: self.next_frame())
        self.root.bind("<Left>", lambda e: self.prev_frame_nav())
        self.root.bind("<space>", lambda e: self.next_frame())
        self.root.bind("d", lambda e: self.next_frame())
        self.root.bind("a", lambda e: self.prev_frame_nav())
        self.root.bind("<Home>", lambda e: self.goto_frame(1))
        self.root.bind("<End>", lambda e: self.goto_frame(self.total_frames - 1))
    
    def _build_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Canvas for image display
        self.canvas = tk.Canvas(main_frame, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # Controls panel at bottom
        controls = ttk.Frame(main_frame, padding=10)
        controls.grid(row=1, column=0, sticky="ew")
        
        # Navigation buttons
        btn_frame = ttk.Frame(controls)
        btn_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(btn_frame, text="◀ Previous (A)", command=self.prev_frame_nav, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Next (D) ▶", command=self.next_frame, width=15).pack(side=tk.LEFT, padx=2)
        
        # Frame slider
        slider_frame = ttk.Frame(controls)
        slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)
        
        ttk.Label(slider_frame, text="Frame:").pack(side=tk.LEFT)
        self.frame_slider = ttk.Scale(slider_frame, from_=1, to=self.total_frames - 1, orient=tk.HORIZONTAL, command=self._on_slider_change)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.frame_slider.set(1)
        
        # Frame label
        self.frame_label = ttk.Label(slider_frame, text="1 / ?", width=20)
        self.frame_label.pack(side=tk.LEFT)
        
        # Opacity control
        opacity_frame = ttk.Frame(controls)
        opacity_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(opacity_frame, text="Current Frame Opacity:").pack(side=tk.LEFT)
        opacity_slider = ttk.Scale(opacity_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.opacity_var, command=self._on_opacity_change)
        opacity_slider.pack(side=tk.LEFT, padx=5)
        self.opacity_label = ttk.Label(opacity_frame, text="50%", width=5)
        self.opacity_label.pack(side=tk.LEFT)
        
        # Info label
        self.info_label = ttk.Label(controls, text="")
        self.info_label.pack(side=tk.RIGHT, padx=10)
        
        # Bind resize
        self.canvas.bind("<Configure>", self._on_resize)
    
    def _load_frames(self):
        """Load previous and current frames."""
        if self.current_idx < 1:
            self.current_idx = 1
        if self.current_idx >= self.total_frames:
            self.current_idx = self.total_frames - 1
        
        # Load previous frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_idx - 1)
        ret, prev = self.cap.read()
        if ret:
            self.prev_frame = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
        
        # Load current frame
        ret, curr = self.cap.read()
        if ret:
            self.curr_frame = cv2.cvtColor(curr, cv2.COLOR_BGR2RGB)
        
        self._update_display()
        self._update_labels()
    
    def _update_display(self):
        """Blend frames and display."""
        if self.prev_frame is None or self.curr_frame is None:
            return
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2:
            return
        
        # Blend frames
        opacity = self.opacity_var.get()
        blended = cv2.addWeighted(
            self.prev_frame.astype(np.float32), 1.0 - opacity,
            self.curr_frame.astype(np.float32), opacity,
            0
        ).astype(np.uint8)
        
        # Scale to fit canvas
        img_h, img_w = blended.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        disp_w, disp_h = int(img_w * scale), int(img_h * scale)
        
        resized = cv2.resize(blended, (disp_w, disp_h))
        
        # Display
        self.photo_image = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.delete("all")
        offset_x = (canvas_w - disp_w) // 2
        offset_y = (canvas_h - disp_h) // 2
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.photo_image)
    
    def _update_labels(self):
        """Update frame and info labels."""
        self.frame_label.config(text=f"{self.current_idx} / {self.total_frames - 1}")
        
        # Calculate timestamp
        if self.fps > 0:
            timestamp = self.current_idx / self.fps
            mins = int(timestamp // 60)
            secs = timestamp % 60
            time_str = f"{mins:02d}:{secs:05.2f}"
        else:
            time_str = "N/A"
        
        self.info_label.config(text=f"{self.video_name} | Time: {time_str}")
        self.opacity_label.config(text=f"{int(self.opacity_var.get() * 100)}%")
        
        # Update slider without triggering callback
        self.frame_slider.set(self.current_idx)
    
    def next_frame(self):
        """Go to next frame."""
        if self.current_idx < self.total_frames - 1:
            self.current_idx += 1
            self._load_frames()
    
    def prev_frame_nav(self):
        """Go to previous frame."""
        if self.current_idx > 1:
            self.current_idx -= 1
            self._load_frames()
    
    def goto_frame(self, idx):
        """Go to specific frame."""
        self.current_idx = max(1, min(idx, self.total_frames - 1))
        self._load_frames()
    
    def _on_slider_change(self, value):
        """Handle slider change."""
        new_idx = int(float(value))
        if new_idx != self.current_idx:
            self.current_idx = new_idx
            self._load_frames()
    
    def _on_opacity_change(self, value):
        """Handle opacity slider change."""
        self.opacity_label.config(text=f"{int(float(value) * 100)}%")
        self._update_display()
    
    def _on_resize(self, event):
        """Handle window resize."""
        self._update_display()


def main():
    parser = argparse.ArgumentParser(description="View frames with previous frame overlay")
    parser.add_argument("--video", type=str, default="videos/20251117_171939.mp4", help="Path to video file")
    args = parser.parse_args()
    
    if not os.path.isfile(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    root = tk.Tk()
    root.geometry("1200x800")
    
    try:
        app = FrameOverlayViewer(root, args.video)
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

