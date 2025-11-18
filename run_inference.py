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
        frame_with_detections, num_detections = self._run_inference_on_frame(frame)
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

    def _run_inference_on_frame(self, frame):
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
        
        # Generate all pairs of dots
        if len(final_detections) > 1:
            detections_array = np.array(final_detections)
            all_lines = []
            
            for i in range(len(final_detections)):
                for j in range(i + 1, len(final_detections)):
                    all_lines.append((i, j))
            
            # Calculate perpendicular distance score for each line (sum of distances to other lines within 16px)
            distance_scores = {}
            proximity_threshold = 16
            
            for idx, (i, j) in enumerate(all_lines):
                p1 = final_detections[i]
                p2 = final_detections[j]
                score = 0.0
                
                for other_idx, (k, l) in enumerate(all_lines):
                    if idx != other_idx:
                        p3 = final_detections[k]
                        p4 = final_detections[l]
                        dist = self._closest_distance_between_segments(p1, p2, p3, p4)
                        if dist < proximity_threshold:
                            score += dist
                
                distance_scores[idx] = score
            
            # Normalize scores to 0-1 range for color mapping
            if distance_scores:
                min_score = min(distance_scores.values())
                max_score = max(distance_scores.values())
                score_range = max_score - min_score if max_score > min_score else 1
                
                normalized_scores = {}
                for idx in distance_scores:
                    normalized_scores[idx] = (distance_scores[idx] - min_score) / score_range if score_range > 0 else 0
                
                # Highlight the line with highest score in bright red with thicker stroke
                max_score_idx = max(normalized_scores, key=normalized_scores.get)
                i, j = all_lines[max_score_idx]
                x1, y1 = final_detections[i]
                x2, y2 = final_detections[j]
                cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                print(f"[Best Line] Line {max_score_idx} (points {i}-{j}) score: {distance_scores[max_score_idx]:.2f}")
        
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
