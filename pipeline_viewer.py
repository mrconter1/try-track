"""
Pipeline Viewer - Visualize each step of the line detection and grid unwrapping pipeline.

Usage: python pipeline_viewer.py <video_path> [--frame N] [--model path]

Pipeline Steps:
  1. Raw Frame - Load frame from video
  2. Mask - Neural network line prediction
  3. Lines - HoughLinesP detection
  4. Merged Lines - Rho-theta clustering to merge collinear segments
  5. Crossings - Find intersections between horizontal/vertical lines
  6. Unwrapped - Homography-based perspective correction

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
    def __init__(self, video_path, model_path="line_detector_unet_best.pth", start_frame=0):
        self.video_path = video_path
        self.model_path = model_path
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.current_frame_idx = 0
        self.start_frame = max(0, min(start_frame, self.total_frames - 1))
        
        # Pipeline state
        self.raw_frame = None
        self.mask = None
        self.lines = []  # Raw segments from HoughP
        self.lines_merged = []  # Segments (for display)
        self.infinite_lines = []  # List of (rho, theta) for infinite lines
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
        self.root.state('zoomed')  # Start maximized (Windows)
        self._build_ui()
        
        # Bindings
        self.root.bind("<a>", lambda e: self.step_frame(-1))
        self.root.bind("<d>", lambda e: self.step_frame(1))
        self.root.bind("<Left>", lambda e: self.step_frame(-1))
        self.root.bind("<Right>", lambda e: self.step_frame(1))
        self.root.bind("<A>", lambda e: self.step_frame(-500))  # Shift+A
        self.root.bind("<D>", lambda e: self.step_frame(500))   # Shift+D
        
        # Load starting frame after window is shown (so canvases have dimensions)
        self.frame_var.set(self.start_frame)
        self.root.after(100, lambda: self.load_and_process_frame(self.start_frame))
    
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
        
        # Main content - 6 panels (3x2 grid)
        content = ttk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top row: Raw Frame, Mask, Lines (raw)
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
        
        # Panel 3: Lines (raw from HoughP)
        p3 = ttk.LabelFrame(top_row, text="3. Lines (HoughP)", padding=5)
        p3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.canvas_lines = tk.Canvas(p3, bg="#1a1a1a", highlightthickness=0)
        self.canvas_lines.pack(fill=tk.BOTH, expand=True)
        self.lbl_lines = ttk.Label(p3, text="0 lines")
        self.lbl_lines.pack()
        
        # Bottom row: Merged Lines, Crossings, Unwrapped
        bottom_row = ttk.Frame(content)
        bottom_row.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Panel 4: Infinite Lines
        p4 = ttk.LabelFrame(bottom_row, text="4. Infinite Lines (ρ,θ)", padding=5)
        p4.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.canvas_merged = tk.Canvas(p4, bg="#1a1a1a", highlightthickness=0)
        self.canvas_merged.pack(fill=tk.BOTH, expand=True)
        self.lbl_merged = ttk.Label(p4, text="0 lines")
        self.lbl_merged.pack()
        
        # Panel 5: Crossings
        p5 = ttk.LabelFrame(bottom_row, text="5. Crossings", padding=5)
        p5.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.canvas_crossings = tk.Canvas(p5, bg="#1a1a1a", highlightthickness=0)
        self.canvas_crossings.pack(fill=tk.BOTH, expand=True)
        self.lbl_crossings = ttk.Label(p5, text="0 crossings")
        self.lbl_crossings.pack()
        
        # Panel 6: Unwrapped
        p6 = ttk.LabelFrame(bottom_row, text="6. Unwrapped (Homography)", padding=5)
        p6.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.canvas_unwrapped = tk.Canvas(p6, bg="#1a1a2e", highlightthickness=0)
        self.canvas_unwrapped.pack(fill=tk.BOTH, expand=True)
        self.lbl_unwrap_status = ttk.Label(p6, text="")
        self.lbl_unwrap_status.pack()
    
    def _on_slider_changed(self, value):
        frame_idx = int(float(value))
        if frame_idx != self.current_frame_idx:
            self.load_and_process_frame(frame_idx)
    
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
        
        # Step 4: Merge collinear lines
        self._step4_merge_lines()
        
        # Step 5: Find crossings
        self._step5_crossings()
        
        # Step 6: Unwrap
        self._step6_unwrap()
        
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
    
    def _step4_merge_lines(self):
        """Merge collinear segments. Simple: angle_diff < 10° AND dist_diff < 25px → same bin."""
        if len(self.lines) < 2:
            self.lines_merged = self.lines.copy()
            self.infinite_lines = []
            return
        
        # For each segment: (angle_deg, perp_dist, segment, midpoint)
        line_data = []
        for seg in self.lines:
            x1, y1, x2, y2 = seg
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            if length < 1:
                continue
            
            # Angle in degrees [0, 180)
            angle = np.arctan2(dy, dx) * 180 / np.pi
            if angle < 0:
                angle += 180
            
            # Midpoint
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Perpendicular distance from origin (signed)
            # Normal = (-sin(angle), cos(angle))
            angle_rad = angle * np.pi / 180
            nx, ny = -np.sin(angle_rad), np.cos(angle_rad)
            dist = mx * nx + my * ny
            
            line_data.append((angle, dist, seg, (mx, my)))
        
        print(f"[DEBUG] {len(line_data)} segments")
        
        # Cluster: angle_diff < 10° AND dist_diff < 25px
        ANGLE_TOL = 10
        DIST_TOL = 25
        
        clusters = []
        used = [False] * len(line_data)
        
        for i in range(len(line_data)):
            if used[i]:
                continue
            
            angle_i, dist_i, _, _ = line_data[i]
            cluster = [line_data[i]]
            used[i] = True
            
            for j in range(i + 1, len(line_data)):
                if used[j]:
                    continue
                
                angle_j, dist_j, _, _ = line_data[j]
                
                # Angle diff with wraparound
                angle_diff = abs(angle_i - angle_j)
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff
                
                if angle_diff < ANGLE_TOL and abs(dist_i - dist_j) < DIST_TOL:
                    cluster.append(line_data[j])
                    used[j] = True
            
            clusters.append(cluster)
        
        print(f"[DEBUG] {len(clusters)} clusters")
        
        # Average each cluster → infinite line (angle_deg, dist)
        # Only keep bins with at least 3 segments
        MIN_SEGMENTS = 3
        
        self.infinite_lines = []
        self.lines_merged = []
        kept_clusters = []
        
        for cluster in clusters:
            if len(cluster) < MIN_SEGMENTS:
                continue
            
            kept_clusters.append(cluster)
            avg_angle = np.mean([c[0] for c in cluster])
            avg_dist = np.mean([c[1] for c in cluster])
            self.infinite_lines.append((avg_angle, avg_dist))
            
            # Merged segment: find extreme endpoints
            angle_rad = avg_angle * np.pi / 180
            dir_x, dir_y = np.cos(angle_rad), np.sin(angle_rad)
            
            all_pts = []
            for _, _, (x1, y1, x2, y2), _ in cluster:
                all_pts.extend([(x1, y1), (x2, y2)])
            
            projs = [(p, p[0]*dir_x + p[1]*dir_y) for p in all_pts]
            projs.sort(key=lambda x: x[1])
            p1, p2 = projs[0][0], projs[-1][0]
            self.lines_merged.append((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
        
        skipped = len(clusters) - len(kept_clusters)
        print(f"[DEBUG] {len(self.infinite_lines)} infinite lines (skipped {skipped} with <{MIN_SEGMENTS} segs):")
        for i, cluster in enumerate(kept_clusters):
            angle, dist = self.infinite_lines[i]
            print(f"  angle={angle:5.1f}°, dist={dist:7.1f}, {len(cluster)} segments")
    
    def _step5_crossings(self):
        """Find crossings between infinite lines. Lines are (angle_deg, dist)."""
        if len(self.infinite_lines) < 2:
            self.crossings = []
            self.grid_crossings = []
            return
        
        h, w = self.raw_frame.shape[:2]
        
        # Group by angle: < 90° vs >= 90°
        group1 = [(a, d) for a, d in self.infinite_lines if a < 90]
        group2 = [(a, d) for a, d in self.infinite_lines if a >= 90]
        
        # Sort by dist
        group1.sort(key=lambda x: x[1])
        group2.sort(key=lambda x: x[1])
        
        def intersect(angle1_deg, dist1, angle2_deg, dist2):
            """Intersect two infinite lines. Line eq: -x*sin(a) + y*cos(a) = dist"""
            a1 = angle1_deg * np.pi / 180
            a2 = angle2_deg * np.pi / 180
            
            # Coefficients: -sin(a)*x + cos(a)*y = dist
            A1, B1 = -np.sin(a1), np.cos(a1)
            A2, B2 = -np.sin(a2), np.cos(a2)
            
            det = A1 * B2 - A2 * B1
            if abs(det) < 1e-6:
                return None
            
            x = (dist1 * B2 - dist2 * B1) / det
            y = (A1 * dist2 - A2 * dist1) / det
            return (int(x), int(y))
        
        self.crossings = []
        self.grid_crossings = []
        
        for i, (a1, d1) in enumerate(group1):
            for j, (a2, d2) in enumerate(group2):
                pt = intersect(a1, d1, a2, d2)
                if pt:
                    x, y = pt
                    if -50 <= x < w + 50 and -50 <= y < h + 50:
                        self.crossings.append((x, y))
                        self.grid_crossings.append(((x, y), (i, j)))
        
        print(f"[DEBUG] Groups: {len(group1)} lines <90°, {len(group2)} lines >=90°")
        print(f"[DEBUG] Found {len(self.crossings)} crossings")
    
    def _step6_unwrap(self):
        """Find all quads, collect point correspondences, RANSAC global homography."""
        h, w = self.raw_frame.shape[:2]
        
        if len(self.grid_crossings) < 4:
            self.unwrapped = None
            self.quads = []
            self.inlier_quads = []
            self.outlier_quads = []
            self.H = None
            return
        
        # Build lookup: (i, j) -> (x, y)
        crossing_map = {(i, j): (x, y) for (x, y), (i, j) in self.grid_crossings}
        
        # Find all valid quads: 4 corners at (i,j), (i+1,j), (i,j+1), (i+1,j+1)
        self.quads = []
        all_i = sorted(set(i for (i, j) in crossing_map.keys()))
        all_j = sorted(set(j for (i, j) in crossing_map.keys()))
        
        for i in all_i:
            for j in all_j:
                if (i, j) in crossing_map and (i+1, j) in crossing_map and \
                   (i, j+1) in crossing_map and (i+1, j+1) in crossing_map:
                    corners = [
                        crossing_map[(i, j)],
                        crossing_map[(i+1, j)],
                        crossing_map[(i+1, j+1)],
                        crossing_map[(i, j+1)]
                    ]
                    self.quads.append({'corners': corners, 'grid_pos': (i, j)})
        
        print(f"[DEBUG] Found {len(self.quads)} complete quads")
        
        # Collect ALL point correspondences from all quads
        # Each quad corner: image (x,y) -> grid (col*tile, row*tile)
        src_pts = []  # image points
        dst_pts = []  # grid points
        
        for quad in self.quads:
            i, j = quad['grid_pos']
            corners = quad['corners']
            # corners order: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
            grid_corners = [
                (j * self.tile_size, i * self.tile_size),           # (i,j)
                (j * self.tile_size, (i+1) * self.tile_size),       # (i+1,j)
                ((j+1) * self.tile_size, (i+1) * self.tile_size),   # (i+1,j+1)
                ((j+1) * self.tile_size, i * self.tile_size)        # (i,j+1)
            ]
            for (img_x, img_y), (grid_x, grid_y) in zip(corners, grid_corners):
                src_pts.append([img_x, img_y])
                dst_pts.append([grid_x, grid_y])
        
        print(f"[DEBUG] Collected {len(src_pts)} point correspondences from {len(self.quads)} quads")
        
        if len(src_pts) < 4:
            self.unwrapped = None
            self.H = None
            self.inlier_quads = []
            self.outlier_quads = []
            return
        
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)
        
        # RANSAC to find global homography
        self.H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if self.H is None:
            print(f"[DEBUG] RANSAC failed to find homography")
            self.unwrapped = None
            self.inlier_quads = []
            self.outlier_quads = []
            return
        
        n_inliers = np.sum(inlier_mask) if inlier_mask is not None else 0
        print(f"[DEBUG] RANSAC: {n_inliers}/{len(src_pts)} inliers")
        
        # Compute output size from grid bounds
        min_i = min(i for (i, j) in crossing_map.keys())
        max_i = max(i for (i, j) in crossing_map.keys())
        min_j = min(j for (i, j) in crossing_map.keys())
        max_j = max(j for (i, j) in crossing_map.keys())
        
        out_w = (max_j + 1) * self.tile_size
        out_h = (max_i + 1) * self.tile_size
        
        # Warp entire frame with global homography
        frame_bgr = cv2.cvtColor(self.raw_frame, cv2.COLOR_RGB2BGR)
        unwarped = cv2.warpPerspective(frame_bgr, self.H, (out_w, out_h))
        self.unwrapped = cv2.cvtColor(unwarped, cv2.COLOR_BGR2RGB)
        
        # Draw grid lines
        for i in range(out_h // self.tile_size + 1):
            y = i * self.tile_size
            cv2.line(self.unwrapped, (0, y), (out_w, y), (100, 100, 255), 1)
        for j in range(out_w // self.tile_size + 1):
            x = j * self.tile_size
            cv2.line(self.unwrapped, (x, 0), (x, out_h), (100, 100, 255), 1)
        
        # Mark which quads were inliers vs outliers
        self.inlier_quads = []
        self.outlier_quads = []
        if inlier_mask is not None:
            pts_per_quad = 4
            for q_idx, quad in enumerate(self.quads):
                quad_inliers = inlier_mask[q_idx*pts_per_quad:(q_idx+1)*pts_per_quad]
                if np.all(quad_inliers):
                    self.inlier_quads.append(quad)
                else:
                    self.outlier_quads.append(quad)
        
        print(f"[DEBUG] {len(self.inlier_quads)} inlier quads, {len(self.outlier_quads)} outlier quads")
    
    def _refine_homography(self, H_init, mask, n_lines=8, samples_per_line=50):
        """Refine homography by maximizing alignment with mask.
        
        Projects a regular grid through inverse homography onto the image,
        and optimizes to maximize mask values along projected grid lines.
        """
        from scipy.optimize import minimize
        
        h, w = mask.shape[:2]
        
        # Determine grid bounds from detected crossings
        if self.grid_crossings:
            min_i = min(i for (x, y), (i, j) in self.grid_crossings)
            max_i = max(i for (x, y), (i, j) in self.grid_crossings)
            min_j = min(j for (x, y), (i, j) in self.grid_crossings)
            max_j = max(j for (x, y), (i, j) in self.grid_crossings)
            grid_x_min = min_j * self.tile_size
            grid_x_max = (max_j + 1) * self.tile_size
            grid_y_min = min_i * self.tile_size
            grid_y_max = (max_i + 1) * self.tile_size
        else:
            grid_x_min, grid_x_max = 0, n_lines * self.tile_size
            grid_y_min, grid_y_max = 0, n_lines * self.tile_size
        
        print(f"[DEBUG] Grid bounds: x=[{grid_x_min}, {grid_x_max}], y=[{grid_y_min}, {grid_y_max}]")
        
        # Parameterize homography as 8 values (H[2,2] = 1 for normalization)
        def h_to_params(H):
            return np.array([H[0,0], H[0,1], H[0,2], H[1,0], H[1,1], H[1,2], H[2,0], H[2,1]])
        
        def params_to_h(params):
            H = np.zeros((3, 3), dtype=np.float64)
            H[0, 0] = params[0]
            H[0, 1] = params[1]
            H[0, 2] = params[2]
            H[1, 0] = params[3]
            H[1, 1] = params[4]
            H[1, 2] = params[5]
            H[2, 0] = params[6]
            H[2, 1] = params[7]
            H[2, 2] = 1.0
            return H
        
        def score_homography(params, debug=False):
            """Negative score (for minimization). Higher mask alignment = better."""
            H = params_to_h(params)
            try:
                H_inv = np.linalg.inv(H)
            except:
                return 1e10
            
            total_score = 0.0
            count = 0
            out_of_bounds = 0
            
            # Project horizontal grid lines onto image
            for i in range(n_lines):
                y_grid = grid_y_min + i * (grid_y_max - grid_y_min) / (n_lines - 1) if n_lines > 1 else grid_y_min
                for s in range(samples_per_line):
                    x_grid = grid_x_min + s * (grid_x_max - grid_x_min) / (samples_per_line - 1) if samples_per_line > 1 else grid_x_min
                    
                    # Transform from grid space to image space
                    pt = np.array([x_grid, y_grid, 1.0])
                    pt_img = H_inv @ pt
                    if abs(pt_img[2]) < 1e-6:
                        continue
                    px, py = pt_img[0] / pt_img[2], pt_img[1] / pt_img[2]
                    
                    # Sample mask
                    if 0 <= int(py) < h and 0 <= int(px) < w:
                        total_score += mask[int(py), int(px)]
                        count += 1
                    else:
                        out_of_bounds += 1
            
            # Project vertical grid lines onto image
            for j in range(n_lines):
                x_grid = grid_x_min + j * (grid_x_max - grid_x_min) / (n_lines - 1) if n_lines > 1 else grid_x_min
                for s in range(samples_per_line):
                    y_grid = grid_y_min + s * (grid_y_max - grid_y_min) / (samples_per_line - 1) if samples_per_line > 1 else grid_y_min
                    
                    pt = np.array([x_grid, y_grid, 1.0])
                    pt_img = H_inv @ pt
                    if abs(pt_img[2]) < 1e-6:
                        continue
                    px, py = pt_img[0] / pt_img[2], pt_img[1] / pt_img[2]
                    
                    if 0 <= int(py) < h and 0 <= int(px) < w:
                        total_score += mask[int(py), int(px)]
                        count += 1
                    else:
                        out_of_bounds += 1
            
            if debug or count == 0:
                print(f"[DEBUG] Score: count={count}, out_of_bounds={out_of_bounds}, mask shape={mask.shape}, h={h}, w={w}")
            
            if count == 0:
                return 1e10
            
            return -total_score / count  # Negative for minimization
        
        # Debug: check initial homography
        print(f"[DEBUG] H_init:\n{H_init}")
        try:
            H_inv = np.linalg.inv(H_init)
            print(f"[DEBUG] H_inv computed OK")
            # Test a few points
            for test_pt in [[0, 0, 1], [100, 0, 1], [0, 100, 1], [100, 100, 1]]:
                pt_img = H_inv @ np.array(test_pt, dtype=np.float64)
                if abs(pt_img[2]) > 1e-6:
                    px, py = pt_img[0] / pt_img[2], pt_img[1] / pt_img[2]
                    print(f"[DEBUG] Grid {test_pt[:2]} -> Image ({px:.1f}, {py:.1f})")
        except Exception as e:
            print(f"[DEBUG] H_inv failed: {e}")
        
        # Optimize
        init_params = h_to_params(H_init)
        init_score = -score_homography(init_params, debug=True)
        
        result = minimize(score_homography, init_params, method='Powell',
                         options={'maxiter': 100, 'ftol': 1e-4})
        
        final_score = -result.fun
        print(f"[DEBUG] Homography refinement: score {init_score:.1f} -> {final_score:.1f}")
        
        if final_score > init_score:
            return params_to_h(result.x)
        else:
            return H_init
    
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
        
        # 3. Raw lines from HoughP
        if self.raw_frame is not None:
            lines_img = self.raw_frame.copy()
            for x1, y1, x2, y2 in self.lines:
                cv2.line(lines_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.circle(lines_img, (x1, y1), 4, (255, 0, 0), -1)
                cv2.circle(lines_img, (x2, y2), 4, (255, 0, 0), -1)
            self._display_on_canvas(self.canvas_lines, lines_img, "lines")
            self.lbl_lines.config(text=f"{len(self.lines)} lines")
        
        # 4. Infinite lines (draw extending to frame edges)
        if self.raw_frame is not None:
            merged_img = self.raw_frame.copy()
            h, w = merged_img.shape[:2]
            
            # Draw infinite lines (angle_deg, dist)
            for angle_deg, dist in self.infinite_lines:
                angle_rad = angle_deg * np.pi / 180
                # Direction along line
                dir_x = np.cos(angle_rad)
                dir_y = np.sin(angle_rad)
                # Normal direction
                nx, ny = -np.sin(angle_rad), np.cos(angle_rad)
                
                # Point on line at perpendicular distance from origin
                base_x = dist * nx
                base_y = dist * ny
                
                # Extend far in both directions
                t = max(w, h) * 2
                x1 = int(base_x - t * dir_x)
                y1 = int(base_y - t * dir_y)
                x2 = int(base_x + t * dir_x)
                y2 = int(base_y + t * dir_y)
                
                cv2.line(merged_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            self._display_on_canvas(self.canvas_merged, merged_img, "merged")
            self.lbl_merged.config(text=f"{len(self.infinite_lines)} infinite lines (was {len(self.lines)} segs)")
        
        # 5. Crossings (using merged lines) + detected quads (inliers green, outliers red)
        if self.raw_frame is not None:
            cross_img = self.raw_frame.copy()
            # Draw inlier quads (green)
            if hasattr(self, 'inlier_quads'):
                for quad in self.inlier_quads:
                    corners = np.array(quad['corners'], dtype=np.int32)
                    cv2.fillPoly(cross_img, [corners], (50, 120, 50))
                    cv2.polylines(cross_img, [corners], True, (0, 255, 0), 2)
            # Draw outlier quads (red)
            if hasattr(self, 'outlier_quads'):
                for quad in self.outlier_quads:
                    corners = np.array(quad['corners'], dtype=np.int32)
                    cv2.fillPoly(cross_img, [corners], (50, 50, 120))
                    cv2.polylines(cross_img, [corners], True, (0, 0, 255), 2)
            # Draw merged lines faintly
            for x1, y1, x2, y2 in self.lines_merged:
                cv2.line(cross_img, (x1, y1), (x2, y2), (100, 100, 100), 1)
            # Draw crossings with grid indices
            for (x, y), (i, j) in self.grid_crossings:
                cv2.circle(cross_img, (x, y), 8, (0, 255, 0), -1)
                cv2.circle(cross_img, (x, y), 8, (0, 0, 0), 2)
                cv2.putText(cross_img, f"{i},{j}", (x + 10, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            n_in = len(self.inlier_quads) if hasattr(self, 'inlier_quads') else 0
            n_out = len(self.outlier_quads) if hasattr(self, 'outlier_quads') else 0
            self._display_on_canvas(self.canvas_crossings, cross_img, "crossings")
            self.lbl_crossings.config(text=f"{len(self.crossings)} crossings, {n_in} inlier / {n_out} outlier quads")
        
        # 6. Unwrapped (global homography from RANSAC)
        n_in = len(self.inlier_quads) if hasattr(self, 'inlier_quads') else 0
        if self.unwrapped is not None and self.H is not None:
            self._display_on_canvas(self.canvas_unwrapped, self.unwrapped, "unwrapped")
            self.lbl_unwrap_status.config(text=f"RANSAC: {n_in} inlier quads", foreground="green")
        else:
            self.canvas_unwrapped.delete("all")
            cw = self.canvas_unwrapped.winfo_width()
            ch = self.canvas_unwrapped.winfo_height()
            if cw > 10 and ch > 10:
                self.canvas_unwrapped.create_text(cw // 2, ch // 2,
                    text="No homography", fill="#666666", font=("Arial", 12))
            self.lbl_unwrap_status.config(text=f"No homography (need quads)", foreground="orange")
    
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
    parser.add_argument("--frame", "-f", type=int, default=0, help="Starting frame index")
    args = parser.parse_args()
    
    viewer = PipelineViewer(args.video, args.model, start_frame=args.frame)
    viewer.run()


if __name__ == "__main__":
    main()

