import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import argparse
import os


class FrameOverlayViewer:
    """GUI to view current frame overlaid on previous frame with perspective adjustment."""
    
    def __init__(self, root, video_path):
        self.root = root
        self.root.title("Frame Overlay Viewer - Drag corners to adjust perspective")
        self.root.state('zoomed')
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_name = os.path.basename(video_path)
        
        # Frame state
        self.current_idx = 1
        self.prev_frame = None
        self.curr_frame = None
        self.photo_image = None
        
        # Display scaling
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.canvas_scale = 1.0
        self.img_w = 0
        self.img_h = 0
        
        # Corner points (in image coordinates) - start at corners
        # Order: top-left, top-right, bottom-right, bottom-left
        self.corner_points = None  # Will be set when frame loads
        self.dragging_point = None
        self.drag_start_pos = None
        self.drag_start_point = None
        
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
        self.root.bind("r", lambda e: self.reset_corners())
        self.root.bind("R", lambda e: self.reset_corners())
    
    def _build_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Canvas for image display
        self.canvas = tk.Canvas(main_frame, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # Bind mouse events for dragging
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        
        # Controls panel at bottom
        controls = ttk.Frame(main_frame, padding=10)
        controls.grid(row=1, column=0, sticky="ew")
        
        # Navigation buttons
        btn_frame = ttk.Frame(controls)
        btn_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(btn_frame, text="◀ Prev (A)", command=self.prev_frame_nav, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Next (D) ▶", command=self.next_frame, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Reset (R)", command=self.reset_corners, width=10).pack(side=tk.LEFT, padx=2)
        
        # Frame slider
        slider_frame = ttk.Frame(controls)
        slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)
        
        ttk.Label(slider_frame, text="Frame:").pack(side=tk.LEFT)
        self.frame_slider = ttk.Scale(slider_frame, from_=1, to=self.total_frames - 1, orient=tk.HORIZONTAL, command=self._on_slider_change)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.frame_slider.set(1)
        
        self.frame_label = ttk.Label(slider_frame, text="1 / ?", width=15)
        self.frame_label.pack(side=tk.LEFT)
        
        # Opacity control
        opacity_frame = ttk.Frame(controls)
        opacity_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(opacity_frame, text="Opacity:").pack(side=tk.LEFT)
        opacity_slider = ttk.Scale(opacity_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.opacity_var, command=self._on_opacity_change, length=100)
        opacity_slider.pack(side=tk.LEFT, padx=5)
        self.opacity_label = ttk.Label(opacity_frame, text="50%", width=4)
        self.opacity_label.pack(side=tk.LEFT)
        
        # Info label
        self.info_label = ttk.Label(controls, text="Drag corners to adjust perspective")
        self.info_label.pack(side=tk.RIGHT, padx=10)
        
        # Bind resize
        self.canvas.bind("<Configure>", self._on_resize)
    
    def _init_corners(self):
        """Initialize corner points at frame corners."""
        if self.curr_frame is None:
            return
        h, w = self.curr_frame.shape[:2]
        self.img_w, self.img_h = w, h
        # Order: top-left, top-right, bottom-right, bottom-left
        self.corner_points = [
            [0.0, 0.0],           # TL
            [float(w), 0.0],      # TR
            [float(w), float(h)], # BR
            [0.0, float(h)]       # BL
        ]
    
    def reset_corners(self):
        """Reset corners to original positions."""
        self._init_corners()
        self._update_display()
    
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
        
        # Reset corners for new frame
        self._init_corners()
        self._update_display()
        self._update_labels()
    
    def _warp_current_frame(self):
        """Apply perspective transform to current frame based on corner points."""
        if self.curr_frame is None or self.corner_points is None:
            return self.curr_frame
        
        h, w = self.curr_frame.shape[:2]
        
        # Source points (original corners)
        src_pts = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])
        
        # Destination points (user-adjusted corners)
        dst_pts = np.float32(self.corner_points)
        
        # Check if corners have moved
        if np.allclose(src_pts, dst_pts, atol=1.0):
            return self.curr_frame
        
        # Compute perspective transform
        try:
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(self.curr_frame, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            return warped
        except:
            return self.curr_frame
    
    def _update_display(self):
        """Blend frames and display."""
        if self.prev_frame is None or self.curr_frame is None:
            return
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2:
            return
        
        # Warp current frame
        warped_curr = self._warp_current_frame()
        
        # Blend frames
        opacity = self.opacity_var.get()
        blended = cv2.addWeighted(
            self.prev_frame.astype(np.float32), 1.0 - opacity,
            warped_curr.astype(np.float32), opacity,
            0
        ).astype(np.uint8)
        
        # Scale to fit canvas (75% to leave room for dragging outside bounds)
        img_h, img_w = blended.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h) * 0.75
        disp_w, disp_h = int(img_w * scale), int(img_h * scale)
        
        self.canvas_scale = scale
        self.canvas_offset_x = (canvas_w - disp_w) // 2
        self.canvas_offset_y = (canvas_h - disp_h) // 2
        
        resized = cv2.resize(blended, (disp_w, disp_h))
        
        # Display
        self.photo_image = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y, anchor="nw", image=self.photo_image)
        
        # Draw corner points
        self._draw_corners()
    
    def _draw_corners(self):
        """Draw draggable corner points."""
        if self.corner_points is None:
            return
        
        colors = ["#FF5555", "#55FF55", "#5555FF", "#FFFF55"]  # TL=red, TR=green, BR=blue, BL=yellow
        labels = ["TL", "TR", "BR", "BL"]
        
        for i, (px, py) in enumerate(self.corner_points):
            # Convert image coords to canvas coords
            cx = self.canvas_offset_x + px * self.canvas_scale
            cy = self.canvas_offset_y + py * self.canvas_scale
            
            # Draw point
            r = 10
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=colors[i], outline="white", width=2, tags="corner")
            self.canvas.create_text(cx, cy - 18, text=labels[i], fill=colors[i], font=("Arial", 10, "bold"), tags="corner")
        
        # Draw lines connecting corners
        pts = []
        for px, py in self.corner_points:
            cx = self.canvas_offset_x + px * self.canvas_scale
            cy = self.canvas_offset_y + py * self.canvas_scale
            pts.append((cx, cy))
        
        for i in range(4):
            j = (i + 1) % 4
            self.canvas.create_line(pts[i][0], pts[i][1], pts[j][0], pts[j][1], fill="cyan", width=2, dash=(4, 4), tags="corner")
    
    def _on_mouse_down(self, event):
        """Handle mouse button press."""
        if self.corner_points is None:
            return
        
        # Check if clicking near a corner point
        for i, (px, py) in enumerate(self.corner_points):
            cx = self.canvas_offset_x + px * self.canvas_scale
            cy = self.canvas_offset_y + py * self.canvas_scale
            
            dist = np.sqrt((event.x - cx)**2 + (event.y - cy)**2)
            if dist < 20:
                self.dragging_point = i
                self.drag_start_pos = (event.x, event.y)
                self.drag_start_point = list(self.corner_points[i])
                self.canvas.config(cursor="none")
                return
    
    def _on_mouse_drag(self, event):
        """Handle mouse drag."""
        if self.dragging_point is None:
            return
        
        # Calculate delta in canvas coords
        dx = event.x - self.drag_start_pos[0]
        dy = event.y - self.drag_start_pos[1]
        
        # Convert to image coords
        new_x = self.drag_start_point[0] + dx / self.canvas_scale
        new_y = self.drag_start_point[1] + dy / self.canvas_scale
        
        # Clamp to image bounds (with large margin for stretching outside)
        margin = self.img_w // 2
        new_x = max(-margin, min(self.img_w + margin, new_x))
        new_y = max(-margin, min(self.img_h + margin, new_y))
        
        self.corner_points[self.dragging_point] = [new_x, new_y]
        self._update_display()
    
    def _on_mouse_up(self, event):
        """Handle mouse button release."""
        self.dragging_point = None
        self.drag_start_pos = None
        self.drag_start_point = None
        self.canvas.config(cursor="")
    
    def _update_labels(self):
        """Update frame and info labels."""
        self.frame_label.config(text=f"{self.current_idx} / {self.total_frames - 1}")
        
        if self.fps > 0:
            timestamp = self.current_idx / self.fps
            mins = int(timestamp // 60)
            secs = timestamp % 60
            time_str = f"{mins:02d}:{secs:05.2f}"
        else:
            time_str = "N/A"
        
        self.info_label.config(text=f"{self.video_name} | {time_str} | Drag corners to adjust")
        self.opacity_label.config(text=f"{int(self.opacity_var.get() * 100)}%")
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
    parser = argparse.ArgumentParser(description="View frames with perspective adjustment overlay")
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
