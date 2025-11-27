"""
Pipeline Viewer - Visualize each step of the line detection and grid unwrapping pipeline.

Usage: python pipeline_viewer.py <video_path>

Controls:
  - A / Left Arrow: Previous frame
  - D / Right Arrow: Next frame
  - Slider: Jump to frame
"""

import argparse
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models


class MobileUNet(nn.Module):
    """MobileNetV2-based U-Net for line segmentation."""
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            mobilenet = models.mobilenet_v2(weights=None)
        self.encoder = mobilenet.features
        
        self.up1 = nn.ConvTranspose2d(1280, 96, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(96 + 96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        
        self.up2 = nn.ConvTranspose2d(96, 32, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        
        self.up3 = nn.ConvTranspose2d(32, 24, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(24 + 24, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True))
        
        self.up4 = nn.ConvTranspose2d(24, 16, 2, stride=2)
        self.dec4 = nn.Sequential(nn.Conv2d(16 + 16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        
        self.final_up = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.out = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid())
    
    def forward(self, x):
        skip_connections = []
        skip_indices = [1, 3, 6, 13]
        
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            if idx in skip_indices:
                skip_connections.append(x)
        
        x = self.up1(x)
        x = torch.cat([x, skip_connections[3]], dim=1)
        x = self.dec1(x)
        
        x = self.up2(x)
        x = torch.cat([x, skip_connections[2]], dim=1)
        x = self.dec2(x)
        
        x = self.up3(x)
        x = torch.cat([x, skip_connections[1]], dim=1)
        x = self.dec3(x)
        
        x = self.up4(x)
        x = torch.cat([x, skip_connections[0]], dim=1)
        x = self.dec4(x)
        
        x = self.final_up(x)
        x = self.out(x)
        return x


class PipelineViewer:
    def __init__(self, video_path, model_path="line_detector_unet_best.pth"):
        self.video_path = video_path
        self.model_path = model_path
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.current_frame_idx = 0
        
        # Pipeline state
        self.raw_frame = None
        self.mask = None
        self.lines = []
        self.crossings = []
        self.grid_crossings = []
        self.unwrapped = None
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MobileUNet(pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Settings
        self.tile_size = 100
        
        # Photo references
        self.photo_refs = {}
        
        # Build UI
        self.root = tk.Tk()
        self.root.title(f"Pipeline Viewer - {video_path}")
        self.root.geometry("1400x900")
        self._build_ui()
        
        # Bindings
        self.root.bind("<a>", lambda e: self.step_frame(-1))
        self.root.bind("<d>", lambda e: self.step_frame(1))
        self.root.bind("<Left>", lambda e: self.step_frame(-1))
        self.root.bind("<Right>", lambda e: self.step_frame(1))
        
        # Load first frame
        self.load_and_process_frame(0)
    
    def _build_ui(self):
        # Top controls
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X)
        
        ttk.Label(control_frame, text="Frame:").pack(side=tk.LEFT, padx=5)
        
        self.frame_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(control_frame, from_=0, to=max(1, self.total_frames - 1),
                                variable=self.frame_var, orient="horizontal",
                                command=self._on_slider_changed)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.lbl_frame = ttk.Label(control_frame, text=f"0 / {self.total_frames}")
        self.lbl_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Tile Size:").pack(side=tk.LEFT, padx=(20, 5))
        self.tile_size_var = tk.IntVar(value=100)
        tile_slider = ttk.Scale(control_frame, from_=50, to=200, variable=self.tile_size_var,
                                orient="horizontal", command=self._on_tile_size_changed, length=100)
        tile_slider.pack(side=tk.LEFT, padx=5)
        self.lbl_tile_size = ttk.Label(control_frame, text="100px")
        self.lbl_tile_size.pack(side=tk.LEFT)
        
        # Main content - 5 panels
        content = ttk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top row: Raw Frame, Mask, Lines
        top_row = ttk.Frame(content)
        top_row.pack(fill=tk.BOTH, expand=True)
        
        # Panel 1: Raw Frame
        p1 = ttk.LabelFrame(top_row, text="1. Raw Frame", padding=5)
        p1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.canvas_raw = tk.Canvas(p1, bg="#1a1a1a", highlightthickness=0)
        self.canvas_raw.pack(fill=tk.BOTH, expand=True)
        
        # Panel 2: Mask
        p2 = ttk.LabelFrame(top_row, text="2. Mask (Neural Net)", padding=5)
        p2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.canvas_mask = tk.Canvas(p2, bg="#1a1a1a", highlightthickness=0)
        self.canvas_mask.pack(fill=tk.BOTH, expand=True)
        
        # Panel 3: Lines
        p3 = ttk.LabelFrame(top_row, text="3. Lines (HoughP)", padding=5)
        p3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.canvas_lines = tk.Canvas(p3, bg="#1a1a1a", highlightthickness=0)
        self.canvas_lines.pack(fill=tk.BOTH, expand=True)
        
        # Bottom row: Crossings, Unwrapped
        bottom_row = ttk.Frame(content)
        bottom_row.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Panel 4: Crossings
        p4 = ttk.LabelFrame(bottom_row, text="4. Crossings", padding=5)
        p4.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.canvas_crossings = tk.Canvas(p4, bg="#1a1a1a", highlightthickness=0)
        self.canvas_crossings.pack(fill=tk.BOTH, expand=True)
        self.lbl_crossings = ttk.Label(p4, text="0 crossings")
        self.lbl_crossings.pack()
        
        # Panel 5: Unwrapped
        p5 = ttk.LabelFrame(bottom_row, text="5. Unwrapped (Homography)", padding=5)
        p5.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.canvas_unwrapped = tk.Canvas(p5, bg="#1a1a2e", highlightthickness=0)
        self.canvas_unwrapped.pack(fill=tk.BOTH, expand=True)
        self.lbl_unwrap_status = ttk.Label(p5, text="")
        self.lbl_unwrap_status.pack()
    
    def _on_slider_changed(self, value):
        frame_idx = int(float(value))
        if frame_idx != self.current_frame_idx:
            self.load_and_process_frame(frame_idx)
    
    def _on_tile_size_changed(self, value):
        self.tile_size = int(float(value))
        self.lbl_tile_size.config(text=f"{self.tile_size}px")
        # Recompute unwrap with new tile size
        self._step5_unwrap()
        self._display_all()
    
    def step_frame(self, delta):
        new_idx = max(0, min(self.total_frames - 1, self.current_frame_idx + delta))
        if new_idx != self.current_frame_idx:
            self.frame_var.set(new_idx)
            self.load_and_process_frame(new_idx)
    
    def load_and_process_frame(self, frame_idx):
        self.current_frame_idx = frame_idx
        self.lbl_frame.config(text=f"{frame_idx} / {self.total_frames}")
        
        # Step 1: Load raw frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return
        self.raw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Step 2: Generate mask
        self._step2_mask()
        
        # Step 3: Detect lines
        self._step3_lines()
        
        # Step 4: Find crossings
        self._step4_crossings()
        
        # Step 5: Unwrap
        self._step5_unwrap()
        
        # Display all panels
        self._display_all()
    
    def _step2_mask(self):
        """Run neural network to get mask."""
        frame = self.raw_frame
        h, w = frame.shape[:2]
        
        # Pad to multiple of 32
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            frame_padded = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            frame_padded = frame
        
        # Inference
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).reshape(1, 3, 1, 1)
        
        with torch.no_grad():
            input_tensor = torch.from_numpy(frame_padded.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            input_tensor = (input_tensor.to(self.device) - mean) / std
            prediction = self.model(input_tensor)
            pred_np = prediction.squeeze().cpu().numpy()
        
        # Crop and convert to uint8
        self.mask = (pred_np[:h, :w] * 255).astype(np.uint8)
    
    def _step3_lines(self):
        """Detect lines from mask using HoughLinesP."""
        if self.mask is None:
            self.lines = []
            return
        
        # Threshold mask
        _, binary = cv2.threshold(self.mask, 127, 255, cv2.THRESH_BINARY)
        
        # Detect lines
        lines_raw = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180, threshold=50,
                                     minLineLength=50, maxLineGap=20)
        
        if lines_raw is None:
            self.lines = []
            return
        
        # Convert to list of (x1, y1, x2, y2)
        self.lines = [tuple(line[0]) for line in lines_raw]
    
    def _step4_crossings(self):
        """Find crossings between horizontal and vertical lines."""
        if len(self.lines) < 2:
            self.crossings = []
            self.grid_crossings = []
            return
        
        h, w = self.raw_frame.shape[:2]
        
        # Group lines by angle
        horizontal = []
        vertical = []
        
        for x1, y1, x2, y2 in self.lines:
            dx, dy = x2 - x1, y2 - y1
            angle_deg = abs(np.arctan2(dy, dx) * 180 / np.pi)
            if angle_deg > 90:
                angle_deg = 180 - angle_deg
            
            if angle_deg < 45:
                horizontal.append((x1, y1, x2, y2))
            else:
                vertical.append((x1, y1, x2, y2))
        
        # Sort
        horizontal.sort(key=lambda l: (l[1] + l[3]) / 2)
        vertical.sort(key=lambda l: (l[0] + l[2]) / 2)
        
        def line_intersection(l1, l2):
            x1, y1, x2, y2 = l1
            x3, y3, x4, y4 = l2
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                return None
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (int(x), int(y))
        
        self.crossings = []
        self.grid_crossings = []
        
        for i, h_line in enumerate(horizontal):
            for j, v_line in enumerate(vertical):
                pt = line_intersection(h_line, v_line)
                if pt is not None:
                    x, y = pt
                    if -50 <= x < w + 50 and -50 <= y < h + 50:
                        self.crossings.append((x, y))
                        self.grid_crossings.append(((x, y), (i, j)))
    
    def _step5_unwrap(self):
        """Compute homography and unwarp the frame."""
        if len(self.grid_crossings) < 4:
            self.unwrapped = None
            return
        
        src_points = []
        dst_points = []
        
        for (x, y), (i, j) in self.grid_crossings:
            src_points.append([x, y])
            dst_points.append([j * self.tile_size, i * self.tile_size])
        
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        try:
            H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            if H is None:
                self.unwrapped = None
                return
            
            max_i = max(i for (x, y), (i, j) in self.grid_crossings)
            max_j = max(j for (x, y), (i, j) in self.grid_crossings)
            out_w = (max_j + 1) * self.tile_size
            out_h = (max_i + 1) * self.tile_size
            
            frame_bgr = cv2.cvtColor(self.raw_frame, cv2.COLOR_RGB2BGR)
            unwarped = cv2.warpPerspective(frame_bgr, H, (out_w, out_h))
            self.unwrapped = cv2.cvtColor(unwarped, cv2.COLOR_BGR2RGB)
            
            # Draw grid
            for i in range(out_h // self.tile_size + 1):
                y = i * self.tile_size
                cv2.line(self.unwrapped, (0, y), (out_w, y), (100, 100, 255), 1)
            for j in range(out_w // self.tile_size + 1):
                x = j * self.tile_size
                cv2.line(self.unwrapped, (x, 0), (x, out_h), (100, 100, 255), 1)
        except Exception:
            self.unwrapped = None
    
    def _display_all(self):
        """Display all pipeline stages."""
        self.root.after(10, self._do_display_all)
    
    def _do_display_all(self):
        # 1. Raw frame
        if self.raw_frame is not None:
            self._display_on_canvas(self.canvas_raw, self.raw_frame, "raw")
        
        # 2. Mask (colorize)
        if self.mask is not None:
            mask_rgb = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2RGB)
            self._display_on_canvas(self.canvas_mask, mask_rgb, "mask")
        
        # 3. Lines on frame
        if self.raw_frame is not None:
            lines_img = self.raw_frame.copy()
            for x1, y1, x2, y2 in self.lines:
                cv2.line(lines_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.circle(lines_img, (x1, y1), 4, (255, 0, 0), -1)
                cv2.circle(lines_img, (x2, y2), 4, (255, 0, 0), -1)
            self._display_on_canvas(self.canvas_lines, lines_img, "lines")
        
        # 4. Crossings
        if self.raw_frame is not None:
            cross_img = self.raw_frame.copy()
            # Draw lines faintly
            for x1, y1, x2, y2 in self.lines:
                cv2.line(cross_img, (x1, y1), (x2, y2), (100, 100, 100), 1)
            # Draw crossings with grid indices
            for (x, y), (i, j) in self.grid_crossings:
                cv2.circle(cross_img, (x, y), 8, (0, 255, 0), -1)
                cv2.circle(cross_img, (x, y), 8, (0, 0, 0), 2)
                cv2.putText(cross_img, f"{i},{j}", (x + 10, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            self._display_on_canvas(self.canvas_crossings, cross_img, "crossings")
            self.lbl_crossings.config(text=f"{len(self.crossings)} crossings")
        
        # 5. Unwrapped
        if self.unwrapped is not None:
            self._display_on_canvas(self.canvas_unwrapped, self.unwrapped, "unwrapped")
            self.lbl_unwrap_status.config(text="OK", foreground="green")
        else:
            self.canvas_unwrapped.delete("all")
            cw = self.canvas_unwrapped.winfo_width()
            ch = self.canvas_unwrapped.winfo_height()
            if cw > 10 and ch > 10:
                self.canvas_unwrapped.create_text(cw // 2, ch // 2,
                    text="Need 4+ crossings", fill="#666666", font=("Arial", 12))
            self.lbl_unwrap_status.config(text=f"Need 4+ crossings (have {len(self.crossings)})", foreground="orange")
    
    def _display_on_canvas(self, canvas, img, key):
        """Scale and display image on canvas."""
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            return
        
        img_h, img_w = img.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h) * 0.95
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_pil = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(img_pil)
        
        self.photo_refs[key] = photo
        
        offset_x = (canvas_w - new_w) // 2
        offset_y = (canvas_h - new_h) // 2
        
        canvas.delete("all")
        canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=photo)
    
    def run(self):
        self.root.mainloop()
        self.cap.release()


def main():
    parser = argparse.ArgumentParser(description="Pipeline Viewer")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--model", default="line_detector_unet_best.pth", help="Path to model file")
    args = parser.parse_args()
    
    viewer = PipelineViewer(args.video, args.model)
    viewer.run()


if __name__ == "__main__":
    main()

