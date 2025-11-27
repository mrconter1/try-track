"""
Crossing Detector Viewer

View predictions from trained crossing detector model on video frames.
Uses sliding window to process full frames.

Usage: python crossing_viewer.py --model cross_net_v1_best.pth --video videos/video.mp4
"""

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torchvision.models as models


class MobileUNet(nn.Module):
    """Mobile-optimized U-Net with MobileNetV2 backbone."""
    
    def __init__(self, pretrained=False):
        super().__init__()
        
        mobilenet = models.mobilenet_v2(weights=None)
        self.encoder = mobilenet.features
        
        self.up1 = nn.ConvTranspose2d(1280, 96, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(96 + 96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(96, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(32, 24, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(24 + 24, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.ConvTranspose2d(24, 16, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.final_up = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.out = nn.Sequential(
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
    
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


class CrossingViewer:
    """GUI for viewing crossing predictions on video frames."""
    
    def __init__(self, model_path: str, video_path: str = None, videos_dir: str = "videos", start_frame: int = 0):
        self.root = tk.Tk()
        self.root.title("Crossing Detector Viewer")
        self.root.state('zoomed')
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        self.model = MobileUNet(pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Loaded model from {model_path}")
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Find videos
        self.videos_dir = videos_dir
        self.video_paths = self._find_videos(videos_dir)
        if video_path and os.path.exists(video_path):
            if video_path not in self.video_paths:
                self.video_paths.insert(0, video_path)
        
        self.start_frame = start_frame
        
        # State
        self.cap = None
        self.current_frame = None
        self.current_heatmap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.photo_image = None
        self.stride = 32  # 75% overlap with 128 patch
        self.threshold = 0.1
        self.show_heatmap = True
        
        # Build UI
        self._build_ui()
        
        # Load first video
        if self.video_paths:
            self._load_video(self.video_paths[0])
        
        # Bindings
        self.root.bind("<Left>", lambda e: self._step_frame(-1))
        self.root.bind("<Right>", lambda e: self._step_frame(1))
        self.root.bind("<space>", lambda e: self._step_frame(10))
        self.root.bind("<h>", lambda e: self._toggle_heatmap())
    
    def _find_videos(self, videos_dir: str):
        video_paths = []
        if os.path.isdir(videos_dir):
            for fname in os.listdir(videos_dir):
                if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_paths.append(os.path.join(videos_dir, fname))
        return sorted(video_paths)
    
    def _build_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas
        self.canvas = tk.Canvas(main_frame, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Sidebar
        sidebar = ttk.Frame(main_frame, width=280, padding=10)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        # Video selection
        video_frame = ttk.LabelFrame(sidebar, text="Video", padding=10)
        video_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.video_var = tk.StringVar()
        video_names = [os.path.basename(v) for v in self.video_paths]
        self.video_combo = ttk.Combobox(video_frame, textvariable=self.video_var, values=video_names, state='readonly')
        self.video_combo.pack(fill=tk.X)
        self.video_combo.bind('<<ComboboxSelected>>', self._on_video_changed)
        if video_names:
            self.video_combo.current(0)
        
        # Frame slider
        slider_frame = ttk.LabelFrame(sidebar, text="Frame", padding=10)
        slider_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.frame_var = tk.IntVar(value=0)
        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=100, variable=self.frame_var, command=self._on_slider_changed)
        self.frame_slider.pack(fill=tk.X)
        
        self.frame_label = ttk.Label(slider_frame, text="Frame: 0 / 0")
        self.frame_label.pack(anchor='w')
        
        # Detection settings
        detect_frame = ttk.LabelFrame(sidebar, text="Detection", padding=10)
        detect_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(detect_frame, text="Stride:").pack(anchor='w')
        self.stride_var = tk.IntVar(value=32)
        stride_combo = ttk.Combobox(detect_frame, textvariable=self.stride_var, values=[32, 64, 96, 128], state='readonly', width=10)
        stride_combo.pack(anchor='w', pady=(0, 5))
        stride_combo.bind('<<ComboboxSelected>>', lambda e: self._predict_current())
        
        ttk.Label(detect_frame, text="Threshold:").pack(anchor='w')
        self.threshold_var = tk.DoubleVar(value=0.1)
        threshold_scale = ttk.Scale(detect_frame, from_=0.1, to=0.9, variable=self.threshold_var, command=self._on_threshold_changed)
        threshold_scale.pack(fill=tk.X, pady=(0, 5))
        
        self.threshold_label = ttk.Label(detect_frame, text="Threshold: 0.50")
        self.threshold_label.pack(anchor='w')
        
        # View controls
        view_frame = ttk.LabelFrame(sidebar, text="View", padding=10)
        view_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.heatmap_var = tk.BooleanVar(value=True)
        heatmap_check = ttk.Checkbutton(view_frame, text="Show Heatmap (H)", variable=self.heatmap_var, command=self._redraw)
        heatmap_check.pack(anchor='w')
        
        # Info
        info_frame = ttk.LabelFrame(sidebar, text="Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.crossings_label = ttk.Label(info_frame, text="Crossings: 0")
        self.crossings_label.pack(anchor='w')
        
        # Keyboard shortcuts
        help_frame = ttk.LabelFrame(sidebar, text="Shortcuts", padding=10)
        help_frame.pack(fill=tk.X)
        
        ttk.Label(help_frame, text="← → : Prev/Next frame").pack(anchor='w')
        ttk.Label(help_frame, text="Space : +10 frames").pack(anchor='w')
        ttk.Label(help_frame, text="H : Toggle heatmap").pack(anchor='w')
    
    def _load_video(self, video_path):
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0
        
        self.frame_slider.configure(to=max(1, self.total_frames - 1))
        
        # Use start_frame if set
        start = min(self.start_frame, self.total_frames - 1)
        self.frame_var.set(start)
        self.start_frame = 0  # Only use once
        
        self._load_frame(start)
    
    def _load_frame(self, frame_idx):
        if not self.cap:
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret and frame is not None:
            self.current_frame_idx = frame_idx
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_label.configure(text=f"Frame: {frame_idx} / {self.total_frames}")
            self._predict_current()
    
    def _predict_current(self):
        if self.current_frame is None:
            return
        
        self.stride = self.stride_var.get()
        self.current_heatmap = self._predict_frame(self.current_frame)
        self._redraw()
    
    def _predict_frame(self, frame):
        """Run batched sliding window prediction on full frame."""
        h, w = frame.shape[:2]
        patch_size = 128
        stride = self.stride
        batch_size = 32
        
        # Create output heatmap
        pred_sum = np.zeros((h, w), dtype=np.float32)
        pred_count = np.zeros((h, w), dtype=np.float32)
        
        # Build list of all patch positions
        y_positions = list(range(0, h - patch_size + 1, stride))
        if y_positions and y_positions[-1] + patch_size < h:
            y_positions.append(h - patch_size)
        x_positions = list(range(0, w - patch_size + 1, stride))
        if x_positions and x_positions[-1] + patch_size < w:
            x_positions.append(w - patch_size)
        
        all_positions = [(y, x) for y in y_positions for x in x_positions]
        
        # Prepare normalization tensors
        mean = torch.tensor(self.mean).reshape(3, 1, 1).to(self.device)
        std = torch.tensor(self.std).reshape(3, 1, 1).to(self.device)
        
        # Process in batches
        with torch.no_grad():
            for batch_start in range(0, len(all_positions), batch_size):
                batch_end = min(batch_start + batch_size, len(all_positions))
                batch_positions = all_positions[batch_start:batch_end]
                
                # Extract patches
                batch_patches = []
                for y, x in batch_positions:
                    patch = frame[y:y+patch_size, x:x+patch_size]
                    patch_tensor = torch.from_numpy(patch.astype(np.float32) / 255.0).permute(2, 0, 1)
                    batch_patches.append(patch_tensor)
                
                # Stack and normalize
                batch_tensor = torch.stack(batch_patches).to(self.device)
                batch_tensor = (batch_tensor - mean) / std
                
                # Predict
                predictions = self.model(batch_tensor)
                pred_patches = predictions.squeeze(1).cpu().numpy()
                
                # Accumulate results
                for idx, (y, x) in enumerate(batch_positions):
                    pred_sum[y:y+patch_size, x:x+patch_size] += pred_patches[idx]
                    pred_count[y:y+patch_size, x:x+patch_size] += 1
        
        # Average overlapping predictions
        pred_count[pred_count == 0] = 1
        heatmap = pred_sum / pred_count
        
        return heatmap
    
    def _find_peaks(self, heatmap, threshold):
        """Find local maxima above threshold."""
        from scipy import ndimage
        
        # Threshold
        binary = heatmap > threshold
        
        # Find local maxima
        max_filter = ndimage.maximum_filter(heatmap, size=10)
        peaks = (heatmap == max_filter) & binary
        
        # Get coordinates
        coords = np.where(peaks)
        points = list(zip(coords[1], coords[0]))  # (x, y)
        
        return points
    
    def _redraw(self):
        if self.current_frame is None:
            return
        
        # Create display image
        display = self.current_frame.copy()
        
        # Overlay heatmap if enabled
        if self.heatmap_var.get() and self.current_heatmap is not None:
            # Convert heatmap to color
            heatmap_color = cv2.applyColorMap((self.current_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            
            # Blend
            alpha = 0.4
            display = cv2.addWeighted(display, 1 - alpha, heatmap_color, alpha, 0)
        
        # Find and draw peaks
        threshold = self.threshold_var.get()
        try:
            peaks = self._find_peaks(self.current_heatmap, threshold)
            for x, y in peaks:
                cv2.circle(display, (x, y), 8, (0, 255, 0), 2)
                cv2.circle(display, (x, y), 3, (0, 255, 0), -1)
            self.crossings_label.configure(text=f"Crossings: {len(peaks)}")
        except:
            self.crossings_label.configure(text="Crossings: 0")
        
        self.threshold_label.configure(text=f"Threshold: {threshold:.2f}")
        
        # Scale to fit canvas
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            self.root.after(100, self._redraw)
            return
        
        img_h, img_w = display.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h) * 0.95
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        
        display = cv2.resize(display, (new_w, new_h))
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(display)
        self.photo_image = ImageTk.PhotoImage(img_pil)
        
        # Draw
        self.canvas.delete("all")
        offset_x = (canvas_w - new_w) // 2
        offset_y = (canvas_h - new_h) // 2
        self.canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.photo_image)
    
    def _on_video_changed(self, event=None):
        idx = self.video_combo.current()
        if 0 <= idx < len(self.video_paths):
            self._load_video(self.video_paths[idx])
    
    def _on_slider_changed(self, value):
        frame_idx = int(float(value))
        if frame_idx != self.current_frame_idx:
            self._load_frame(frame_idx)
    
    def _on_threshold_changed(self, value):
        self._redraw()
    
    def _step_frame(self, delta):
        new_idx = max(0, min(self.total_frames - 1, self.current_frame_idx + delta))
        self.frame_var.set(new_idx)
        self._load_frame(new_idx)
    
    def _toggle_heatmap(self):
        self.heatmap_var.set(not self.heatmap_var.get())
        self._redraw()
    
    def run(self):
        self.root.mainloop()
        if self.cap:
            self.cap.release()


def main():
    parser = argparse.ArgumentParser(description="Crossing Detector Viewer")
    parser.add_argument("video", nargs='?', default=None, help="Path to video file")
    parser.add_argument("--model", default="cross_net_v1_best.pth", help="Path to trained model (.pth)")
    parser.add_argument("--frame", type=int, default=0, help="Starting frame number")
    parser.add_argument("--videos", default="videos", help="Videos directory")
    args = parser.parse_args()
    
    app = CrossingViewer(model_path=args.model, video_path=args.video, videos_dir=args.videos, start_frame=args.frame)
    app.run()


if __name__ == "__main__":
    main()

