import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import os
import random
import bisect
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# --- Helper Functions (from annotator) ---

def find_videos_in_paths(paths):
    video_files = []
    supported_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    for path in paths:
        path = os.path.abspath(path)
        if os.path.isfile(path):
            if os.path.splitext(path)[1].lower() in supported_extensions:
                video_files.append(path)
        elif os.path.isdir(path):
            for item in os.listdir(path):
                full_path = os.path.join(path, item)
                if os.path.isfile(full_path) and os.path.splitext(full_path)[1].lower() in supported_extensions:
                    video_files.append(full_path)
    print(f"Found {len(video_files)} videos to process.")
    return sorted(list(set(video_files)))

def get_video_props(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

# --- Model Definition ---

class CrossDetectorModel(nn.Module):
    """Mobile-friendly cross detection model."""
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        self.head = nn.Sequential(
            nn.Linear(576, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

# --- GUI Application ---

class InferenceViewer:
    def __init__(self, root, video_paths, model_path, stride, threshold, cluster_radius, min_hits):
        self.root = root
        self.model_path = model_path
        self.stride = stride
        self.threshold = threshold
        self.cluster_radius = cluster_radius
        self.min_hits = min_hits

        self.video_paths = [os.path.abspath(p) for p in video_paths]
        self.video_frame_counts = {path: get_video_props(path) for path in self.video_paths}

        # Proportional Sampling Setup
        self.cumulative_frames = []
        current_total = 0
        for path in self.video_paths:
            count = self.video_frame_counts.get(path, 0)
            current_total += count
            self.cumulative_frames.append(current_total)
        self.total_combined_frames = current_total

        # Track current sample
        self.current_global_frame_idx = None

        self.root.title("Inference Viewer")
        self.root.state('zoomed')
        self._build_ui()
        self._load_model()
        self.root.after(100, self.run_new_inference)

    def _build_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(main_frame, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=5, padx=5)

        self.run_button = ttk.Button(controls_frame, text="Run Inference on New Random Frame", command=self.run_new_inference)
        self.run_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.info_label = ttk.Label(controls_frame, text="Click the button to start.")
        self.info_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.root.bind("<Configure>", self._on_resize)
        self.root.bind("<a>", lambda e: self.previous_sample())
        self.root.bind("<d>", lambda e: self.next_sample())
        self.root.bind("<r>", lambda e: self.run_new_inference())
        self.photo_image = None
        self.current_frame_with_detections = None

    def _load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossDetectorModel().to(self.device)
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print(f"[Info] Model '{self.model_path}' loaded successfully on {self.device}.")
        except FileNotFoundError:
            messagebox.showerror("Error", f"Model file not found: {self.model_path}")
            self.run_button.config(state=tk.DISABLED)

    def run_new_inference(self):
        if self.total_combined_frames <= 0:
            messagebox.showerror("Error", "No frames found in videos.")
            return

        # 1. Pick a random frame proportionally
        global_frame_idx = random.randint(0, self.total_combined_frames - 1)
        self.current_global_frame_idx = global_frame_idx
        self._load_and_display_frame(global_frame_idx)

    def _load_and_display_frame(self, global_frame_idx):
        if global_frame_idx < 0 or global_frame_idx >= self.total_combined_frames:
            messagebox.showerror("Error", "Frame index out of range.")
            return

        video_idx = bisect.bisect_left(self.cumulative_frames, global_frame_idx)
        video_path = self.video_paths[video_idx]
        previous_cumulative = self.cumulative_frames[video_idx - 1] if video_idx > 0 else 0
        frame_idx = global_frame_idx - previous_cumulative

        # 2. Load the frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Error", f"Failed to read frame {frame_idx} from {video_path}")
            return

        # 3. Run inference logic
        # Pass grayscale frame for grid fitting intensity check
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_with_detections, num_detections = self._run_inference_on_frame(frame, frame_gray)
        self.current_frame_with_detections = frame_with_detections
        
        # 4. Display results
        self._display_frame()
        self.info_label.config(text=f"Video: {os.path.basename(video_path)}\nFrame: {frame_idx}\nDetections: {num_detections}")

    def next_sample(self):
        if self.current_global_frame_idx is None:
            return
        next_idx = self.current_global_frame_idx + 1
        if next_idx < self.total_combined_frames:
            self.current_global_frame_idx = next_idx
            self._load_and_display_frame(next_idx)

    def previous_sample(self):
        if self.current_global_frame_idx is None:
            return
        prev_idx = self.current_global_frame_idx - 1
        if prev_idx >= 0:
            self.current_global_frame_idx = prev_idx
            self._load_and_display_frame(prev_idx)

    def _distance_point_to_segment(self, p, a, b):
        """Calculate minimum distance from point p to line segment ab."""
        x, y = p
        x1, y1 = a
        x2, y2 = b
        
        # Vector from a to b
        dx = x2 - x1
        dy = y2 - y1
        
        # If segment is a point
        if dx == 0 and dy == 0:
            return np.sqrt((x - x1)**2 + (y - y1)**2)
        
        # Parameter t for closest point on segment
        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx**2 + dy**2)))
        
        # Closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return np.sqrt((x - closest_x)**2 + (y - closest_y)**2)

    def _find_grid_tiles(self, detections, frame_gray=None):
        """
        Global Grid Fitting using Homography RANSAC.
        Finds a homography H that maps detections to integer grid coordinates.
        Uses pixel intensity along grid lines to prefer dark grout lines over bright diagonals.
        """
        if len(detections) < 4:
            return []
            
        points = np.array(detections)
        n = len(points)
        
        # Need grayscale frame for line sampling. 
        # If not passed (in current architecture), we can't do it.
        # We need to refactor call site to pass frame.
        # Assuming self.current_frame_gray exists or we can access it?
        # Actually, looking at call site in _run_inference_on_frame, 
        # we have 'frame' available but this function signature doesn't take it.
        # I will assume we update the call site or pass it here.
        # For now, let's rely on a passed 'frame_gray' or fail gracefully.
        
        # RANSAC parameters
        best_H = None
        best_inliers = []
        best_score = -1000 # Use negative starting score as darkness is a penalty/reward
        
        iterations = 2000
        
        # Pre-calculate intensity lookups if possible, but random access is fast enough.
        
        for _ in range(iterations):
            indices = np.random.choice(n, 3, replace=False)
            p0, p1, p2 = points[indices]
            
            if np.abs(np.cross(p1-p0, p2-p0)) < 1e-3: continue
            
            p3_est = p1 + p2 - p0 
            
            src_pts = np.array([p0, p1, p2, p3_est], dtype=np.float32)
            dst_pts = np.array([[0,0], [1,0], [0,1], [1,1]], dtype=np.float32)
            
            try:
                H, _ = cv2.findHomography(src_pts, dst_pts)
            except:
                continue
            if H is None: continue
            
            # --- Validate Geometry ---
            H_inv = np.linalg.inv(H)
            unit_pts = np.array([[0,0], [1,0], [0,1]], dtype=np.float32).reshape(-1, 1, 2)
            back_pts = cv2.perspectiveTransform(unit_pts, H_inv)
            
            v1 = back_pts[1][0] - back_pts[0][0] # Vector for x-axis
            v2 = back_pts[2][0] - back_pts[0][0] # Vector for y-axis
            
            d_x = np.linalg.norm(v1)
            d_y = np.linalg.norm(v2)
            
            if d_x < 40 or d_x > 800 or d_y < 40 or d_y > 800: continue
            
            # Check aspect ratio (tiles should be roughly square-ish, not 1:10)
            ratio = d_x / d_y if d_y > 0 else 999
            if ratio < 0.5 or ratio > 2.0: continue
            
            # --- Check Angle between Axes ---
            # v1 is vector (1,0) in image space
            # v2 is vector (0,1) in image space
            # Calculate angle between them
            def vec_angle(a, b):
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a == 0 or norm_b == 0: return 0
                cos_theta = np.dot(a, b) / (norm_a * norm_b)
                return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            
            angle_deg = vec_angle(v1, v2)
            
            # In a diagonal fit (rhombus), the axes are diagonals of the square,
            # so they are perpendicular in 3D space too?
            # Wait. If we fit the diagonals as the axes (0,1) and (1,0),
            # then the "tiles" become the 45-degree rotated squares.
            # In that case, the angle is ALSO 90 degrees in 3D!
            
            # BUT: If we fit the diagonals, the "scale" is different.
            # However, the visual difference is that the diagonal grid
            # often results in oddly shaped tiles in perspective if the homography is forced.
            
            # Actually, looking at your image, the cyan lines form a grid of DIAMONDS/RHOMBUSES.
            # The angle at the vertex of those cyan polygons is clearly acute (like 60 deg).
            # The REAL grid has angles closer to 90.
            
            # So we enforce that the angle between the basis vectors in image space
            # must be 'reasonable' (e.g. > 60 degrees).
            # 45-degree diagonals will often result in ~45 degree angles in image space.
            
            if angle_deg < 65 or angle_deg > 115:
                 continue

            # --- Calculate Geometric Inliers ---
            ones = np.ones((n, 1))
            pts_homo = np.hstack([points, ones])
            projected = (H @ pts_homo.T).T
            with np.errstate(divide='ignore', invalid='ignore'):
                projected /= projected[:, 2:3]
            
            grid_coords = projected[:, :2]
            nearest_int = np.round(grid_coords)
            dist = np.linalg.norm(grid_coords - nearest_int, axis=1)
            
            current_inliers = []
            geo_score = 0
            for i in range(n):
                if dist[i] < 0.15:
                    current_inliers.append(i)
                    geo_score += 1
            
            if geo_score < 4: continue
            
            # --- Calculate Photometric Score (Grout Darkness) ---
            photo_score = 0
            if frame_gray is not None:
                # Calculate global median brightness once
                # (Actually, compute it only once outside the loop if possible, but safe here)
                global_median = np.median(frame_gray)

                # We want to check if the lines defined by this grid align with dark streaks (grout)
                # vs passing through bright areas (diagonal crossing tiles).
                
                H_inv_curr = np.linalg.inv(H)
                inlier_grid_coords = np.round(grid_coords[current_inliers]).astype(int)
                
                # Find connected pairs in this hypothesis
                connected_pairs = []
                inlier_indices_local = current_inliers
                num_inliers = len(inlier_indices_local)
                if num_inliers > 1:
                    coords = inlier_grid_coords
                    # fast pairwise check
                    for i in range(num_inliers):
                        for j in range(i+1, num_inliers):
                            du = coords[i][0] - coords[j][0]
                            dv = coords[i][1] - coords[j][1]
                            if du*du + dv*dv == 1:
                                connected_pairs.append((inlier_indices_local[i], inlier_indices_local[j]))
                
                segments_to_check = []
                if connected_pairs:
                    import random
                    if len(connected_pairs) > 5:
                        indices_to_check = random.sample(range(len(connected_pairs)), 5)
                        for idx in indices_to_check:
                            pA = points[connected_pairs[idx][0]]
                            pB = points[connected_pairs[idx][1]]
                            segments_to_check.append((pA, pB))
                    else:
                        for pA_idx, pB_idx in connected_pairs:
                            segments_to_check.append((points[pA_idx], points[pB_idx]))
                else:
                    pts_basis = cv2.perspectiveTransform(np.array([[[0,0],[1,0],[0,1]]], dtype=np.float32), H_inv_curr)[0]
                    segments_to_check.append((pts_basis[0], pts_basis[1]))
                    segments_to_check.append((pts_basis[0], pts_basis[2]))
                
                dark_votes = 0
                
                for pStart, pEnd in segments_to_check:
                    num_s = 20
                    xs = np.linspace(pStart[0], pEnd[0], num_s)
                    ys = np.linspace(pStart[1], pEnd[1], num_s)
                    
                    valid_s = 0
                    dark_s = 0
                    h_img, w_img = frame_gray.shape
                    
                    for k in range(num_s):
                        x, y = int(xs[k]), int(ys[k])
                        if 0 <= x < w_img and 0 <= y < h_img:
                            val = frame_gray[y, x]
                            valid_s += 1
                            # Use stricter threshold: grout is usually significantly darker
                            if val < global_median * 0.9: 
                                dark_s += 1
                    
                    if valid_s > 5:
                        ratio = dark_s / valid_s
                        # A grout line should be mostly dark.
                        if ratio > 0.3: 
                            dark_votes += 3
                        else:
                            dark_votes -= 2 # Strong penalty for bright lines (diagonals)
                            
                photo_score = dark_votes
                
            final_score = geo_score + photo_score
            
            if final_score > best_score:
                best_score = final_score
                best_H = H
                best_inliers = current_inliers
        
        if best_H is None or len(best_inliers) < 4:
            return []
            
        # --- Re-Fit H with all inliers ---
        src_refine = []
        dst_refine = []
        
        ones = np.ones((len(best_inliers), 1))
        inlier_pts = points[best_inliers]
        inlier_pts_homo = np.hstack([inlier_pts, ones])
        projected = (best_H @ inlier_pts_homo.T).T
        projected /= projected[:, 2:3]
        grid_coords = np.round(projected[:, :2])
        
        for k, idx in enumerate(best_inliers):
            src_refine.append(points[idx])
            dst_refine.append(grid_coords[k])
            
        src_refine = np.array(src_refine, dtype=np.float32)
        dst_refine = np.array(dst_refine, dtype=np.float32)
        
        H_final, _ = cv2.findHomography(src_refine, dst_refine)
        if H_final is None: return []
        
        # --- Generate Tiles ---
        # Calculate grid bounds based on the IMAGE CORNERS
        # Map image corners (0,0), (w,0), (w,h), (0,h) to grid space
        img_h, img_w = frame_gray.shape
        img_corners = np.array([
            [0, 0],
            [img_w, 0],
            [img_w, img_h],
            [0, img_h]
        ], dtype=np.float32)
        
        img_corners_homo = np.hstack([img_corners, np.ones((4, 1))])
        grid_corners_proj = (H_final @ img_corners_homo.T).T
        grid_corners_proj /= grid_corners_proj[:, 2:3]
        grid_corners = grid_corners_proj[:, :2]
        
        u_min_img = int(np.floor(np.min(grid_corners[:, 0])))
        u_max_img = int(np.ceil(np.max(grid_corners[:, 0])))
        v_min_img = int(np.floor(np.min(grid_corners[:, 1])))
        v_max_img = int(np.ceil(np.max(grid_corners[:, 1])))
        
        # Sanity check limits to avoid hanging on infinite planes
        # Limit to a reasonable range around detected points
        u_mean = np.mean(dst_refine[:, 0])
        v_mean = np.mean(dst_refine[:, 1])
        range_limit = 30 # Max 30 tiles away from center
        
        u_start = max(u_min_img, int(u_mean - range_limit))
        u_end = min(u_max_img, int(u_mean + range_limit))
        v_start = max(v_min_img, int(v_mean - range_limit))
        v_end = min(v_max_img, int(v_mean + range_limit))
        
        H_inv = np.linalg.inv(H_final)
        tiles = []
        
        for u in range(u_start, u_end + 1):
            for v in range(v_start, v_end + 1):
                quad_grid = np.array([[u,v],[u+1,v],[u+1,v+1],[u,v+1]], dtype=np.float32).reshape(-1, 1, 2)
                quad_img = cv2.perspectiveTransform(quad_grid, H_inv).reshape(4, 2)
                
                # Check if tile is visible on screen
                # Simple AABB check
                q_min_x, q_min_y = np.min(quad_img, axis=0)
                q_max_x, q_max_y = np.max(quad_img, axis=0)
                
                if (q_max_x < 0 or q_min_x > img_w or 
                    q_max_y < 0 or q_min_y > img_h):
                    continue
                
                tiles.append(quad_img)
                    
        return tiles

    def _is_valid_tile(self, all_points, indices):
        pts = all_points[list(indices)]
        
        # Sort points radially to ensure we traverse the perimeter
        center = np.mean(pts, axis=0)
        angles_rad = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
        order = np.argsort(angles_rad)
        pts = pts[order]
        
        # Calculate side vectors
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[1]
        v3 = pts[3] - pts[2]
        v4 = pts[0] - pts[3]
        
        # Calculate side lengths
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)
        d3 = np.linalg.norm(v3)
        d4 = np.linalg.norm(v4)
        sides = np.array([d1, d2, d3, d4])
        
        # Diagonals
        diag1 = np.linalg.norm(pts[0] - pts[2])
        diag2 = np.linalg.norm(pts[1] - pts[3])
        
        # --- CRITERIA 1: Diagonals must be longer than ALL sides ---
        # In a rectangle (even rotated), diagonals are the longest segments.
        # In those 45-degree mistakenly connected shapes, one "side" is usually a diagonal 
        # of the real grid, and the "diagonal" of the mistake is just a grid edge.
        if diag1 < np.max(sides) * 1.05 or diag2 < np.max(sides) * 1.05:
            return False

        # --- CRITERIA 2: Check internal angles ---
        # A real perspective square shouldn't have super acute/obtuse angles 
        # unless the camera is grazing the floor.
        # Cosine rule or dot product to find angles.
        def get_angle(vA, vB):
            # Angle between vector vA and vB (outgoing from vertex)
            # vA and vB should be normalized
            uA = vA / (np.linalg.norm(vA) + 1e-6)
            uB = vB / (np.linalg.norm(vB) + 1e-6)
            return np.arccos(np.clip(np.dot(uA, uB), -1.0, 1.0))

        # Vectors pointing OUT from each vertex
        angles = []
        angles.append(get_angle(pts[1]-pts[0], pts[3]-pts[0])) # Angle at 0
        angles.append(get_angle(pts[0]-pts[1], pts[2]-pts[1])) # Angle at 1
        angles.append(get_angle(pts[1]-pts[2], pts[3]-pts[2])) # Angle at 2
        angles.append(get_angle(pts[2]-pts[3], pts[0]-pts[3])) # Angle at 3
        
        angles_deg = np.degrees(angles)
        
        # Filter out if any angle is too sharp (e.g. < 60 degrees) or too wide (> 120)
        # 45-degree triangles usually result in 45-45-90 or similar, 
        # so a 45 deg angle is a dead giveaway of a bad connection.
        if np.any(angles_deg < 60) or np.any(angles_deg > 120):
            return False

        # --- CRITERIA 3: Convexity/Side Consistency ---
        mean_side = np.mean(sides)
        if np.std(sides) > 0.3 * mean_side:
             return False

        return True
    
    def _closest_distance_between_segments(self, p1, p2, p3, p4):
        """Calculate the minimum distance between two line segments."""
        # Distance from p1 to segment p3-p4
        d1 = self._distance_point_to_segment(p1, p3, p4)
        # Distance from p2 to segment p3-p4
        d2 = self._distance_point_to_segment(p2, p3, p4)
        # Distance from p3 to segment p1-p2
        d3 = self._distance_point_to_segment(p3, p1, p2)
        # Distance from p4 to segment p1-p2
        d4 = self._distance_point_to_segment(p4, p1, p2)
        
        return min(d1, d2, d3, d4)

    def _line_intersection(self, p1, p2, p3, p4, threshold=32):
        """Check if two line segments (p1-p2) and (p3-p4) intersect with tolerance threshold."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            # Parallel or coincident lines - check if they're close
            min_dist = self._closest_distance_between_segments(p1, p2, p3, p4)
            return min_dist <= threshold
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Check if intersection is within both segments (with threshold tolerance)
        if -threshold/100 < t < 1 + threshold/100 and -threshold/100 < u < 1 + threshold/100:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return True
        
        # If not a perfect intersection, check if segments are close enough
        min_dist = self._closest_distance_between_segments(p1, p2, p3, p4)
        return min_dist <= threshold

    def _run_inference_on_frame(self, frame, frame_gray=None):
        img_h, img_w = frame.shape[:2]
        tile_size = 128
        tiles, tile_coords = [], []

        for y in range(0, img_h - tile_size + 1, self.stride):
            for x in range(0, img_w - tile_size + 1, self.stride):
                tiles.append(frame[y:y+tile_size, x:x+tile_size])
                tile_coords.append((x, y))
        
        if not tiles: return frame, 0

        batch = []
        for tile in tiles:
            img = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            img = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
            batch.append(torch.from_numpy(img))
        
        batch_tensor = torch.stack(batch).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch_tensor)

        scores = torch.sigmoid(outputs[:, 0])
        raw_detections = []
        for i in range(len(scores)):
            if scores[i] > self.threshold:
                tile_x, tile_y = tile_coords[i]
                pred_x_norm, pred_y_norm = outputs[i, 1:].cpu().numpy()
                global_x = tile_x + pred_x_norm * tile_size
                global_y = tile_y + pred_y_norm * tile_size
                raw_detections.append((global_x, global_y))

        # --- Cluster raw detections ---
        final_detections = self._cluster_detections(raw_detections, radius=self.cluster_radius, min_hits=self.min_hits)

        output_image = frame.copy()
        
        # Find and draw grid tiles (using RANSAC + Intensity)
        grid_tiles = self._find_grid_tiles(final_detections, frame_gray)
        
        for tile in grid_tiles:
            # Sort points radially to draw the polygon correctly
            center = np.mean(tile, axis=0)
            angles = np.arctan2(tile[:,1] - center[1], tile[:,0] - center[0])
            order = np.argsort(angles)
            tile_ordered = tile[order].astype(np.int32)
            
            # Draw filled polygon with low opacity or just thick lines
            # Let's draw thick Cyan lines
            cv2.polylines(output_image, [tile_ordered], isClosed=True, color=(255, 255, 0), thickness=2)
            
            # Optional: Draw diagonals faintly
            cv2.line(output_image, tuple(tile_ordered[0]), tuple(tile_ordered[2]), (0, 128, 128), 1)
            cv2.line(output_image, tuple(tile_ordered[1]), tuple(tile_ordered[3]), (0, 128, 128), 1)
        
        # Draw crosses at detection points
        for x, y in final_detections:
            px, py = int(x), int(y)
            cv2.line(output_image, (px - 15, py), (px + 15, py), (0, 255, 0), 2)
            cv2.line(output_image, (px, py - 15), (px, py + 15), (0, 255, 0), 2)
        
        return output_image, len(final_detections)

    def _cluster_detections(self, detections, radius=32, min_hits=3):
        """Group nearby detections into clusters and average them."""
        clusters = []
        for (x, y) in detections:
            found_cluster = False
            for cluster in clusters:
                # Check distance to the cluster's center
                center_x = np.mean([p[0] for p in cluster])
                center_y = np.mean([p[1] for p in cluster])
                if np.sqrt((x - center_x)**2 + (y - center_y)**2) < radius:
                    cluster.append((x, y))
                    found_cluster = True
                    break
            if not found_cluster:
                clusters.append([(x, y)])
        
        # Average the points in each cluster to get the final detection
        final_detections = []
        for cluster in clusters:
            if len(cluster) >= min_hits:
                avg_x = np.mean([p[0] for p in cluster])
                avg_y = np.mean([p[1] for p in cluster])
                final_detections.append((avg_x, avg_y))
            
        return final_detections

    def _display_frame(self):
        if self.current_frame_with_detections is None: return

        frame_rgb = cv2.cvtColor(self.current_frame_with_detections, cv2.COLOR_BGR2RGB)
        
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2: return

        img_h, img_w = frame_rgb.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        disp_w, disp_h = int(img_w * scale), int(img_h * scale)
        offset_x, offset_y = (canvas_w - disp_w) // 2, (canvas_h - disp_h) // 2

        resized = cv2.resize(frame_rgb, (disp_w, disp_h))
        self.photo_image = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.photo_image)
    
    def _on_resize(self, event):
        self._display_frame()

def main():
    parser = argparse.ArgumentParser(description="Run cross detection inference on random video frames.")
    parser.add_argument("--input", nargs='+', required=True, help="Path to video file(s) or folder(s) containing videos.")
    parser.add_argument("--model", type=str, default="cross_detector_best.pth", help="Path to the trained model .pth file.")
    parser.add_argument("--stride", type=int, default=64, help="Stride for overlapping tiles.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Confidence threshold for detection.")
    parser.add_argument("--cluster-radius", type=int, default=32, help="Radius in pixels to group multiple detections into a single cluster.")
    parser.add_argument("--min-hits", type=int, default=3, help="Minimum number of raw detections required to form a valid cluster.")
    args = parser.parse_args()

    video_paths = find_videos_in_paths(args.input)
    if not video_paths:
        print("[Error] No video files found in the specified paths.")
        return

    root = tk.Tk()
    root.geometry("1200x800")
    app = InferenceViewer(root, video_paths, args.model, args.stride, args.threshold, args.cluster_radius, args.min_hits)
    root.mainloop()

if __name__ == "__main__":
    main()
