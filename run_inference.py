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
        
        # Draw lines to three closest neighbors for each point
        if len(final_detections) > 1:
            detections_array = np.array(final_detections)
            drawn_connections = set()
            edges_from_point = {i: [] for i in range(len(final_detections))}
            
            def is_angle_valid(point_idx, other_idx):
                """Check if adding edge to other_idx maintains minimum 60 degree angles."""
                if len(edges_from_point[point_idx]) == 0:
                    return True
                
                point = detections_array[point_idx]
                other = detections_array[other_idx]
                new_vec = other - point
                
                for existing_idx in edges_from_point[point_idx]:
                    existing = detections_array[existing_idx]
                    existing_vec = existing - point
                    
                    # Calculate angle between vectors
                    dot_product = np.dot(new_vec, existing_vec)
                    mag_new = np.linalg.norm(new_vec)
                    mag_existing = np.linalg.norm(existing_vec)
                    
                    if mag_new > 0 and mag_existing > 0:
                        cos_angle = dot_product / (mag_new * mag_existing)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle_rad = np.arccos(cos_angle)
                        angle_deg = np.degrees(angle_rad)
                        
                        if angle_deg < 60:
                            return False
                return True
            
            for i, (x, y) in enumerate(final_detections):
                # Calculate distances to all other points
                distances = np.sqrt(np.sum((detections_array - np.array([x, y]))**2, axis=1))
                # Get indices of three closest neighbors (excluding self)
                closest_indices = np.argsort(distances)[1:4]  # Skip self (index 0), take next 3
                for j in closest_indices:
                    if j < len(final_detections):
                        # Create a canonical connection key (sorted tuple to avoid duplicates)
                        connection_key = tuple(sorted([i, j]))
                        if connection_key not in drawn_connections:
                            # Check angle constraints for both points
                            if is_angle_valid(i, j) and is_angle_valid(j, i):
                                drawn_connections.add(connection_key)
                                edges_from_point[i].append(j)
                                edges_from_point[j].append(i)
                                neighbor_x, neighbor_y = final_detections[j]
                                px1, py1 = int(x), int(y)
                                px2, py2 = int(neighbor_x), int(neighbor_y)
                                cv2.line(output_image, (px1, py1), (px2, py2), (255, 0, 0), 2)
        
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
