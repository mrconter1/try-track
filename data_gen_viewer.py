"""
Crossing Detection Data Generator

Generates augmented training patches for crossing/intersection detection.
Each crossing is rendered as a soft 2D Gaussian blob (sigma ~3.5) in the mask.

Usage: python data_gen_viewer.py [--videos VIDEOS_DIR] [--annotations ANNOTATIONS_FILE]
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
import random
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import json
import threading


@dataclass
class Line:
    """Represents a line annotation with start and end points (crop-relative coords)."""
    start: Tuple[float, float]  # (x, y)
    end: Tuple[float, float]    # (x, y)


@dataclass
class Sample:
    """Represents a single viewed sample in order."""
    video_path: str
    frame_idx: int
    crop_rect: Tuple[int, int, int, int]
    lines: List[Line] = field(default_factory=list)
    
    def add_line(self, start: Tuple[float, float], end: Tuple[float, float]):
        self.lines.append(Line(start, end))


@dataclass
class AnnotationDatabase:
    """Sequential list of all viewed samples."""
    samples: List[Sample] = field(default_factory=list)
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        def to_relative_path(path):
            basename = path.replace('\\', '/').split('/')[-1]
            return f"videos/{basename}"
        
        return {
            "samples": [
                {
                    "video_path": to_relative_path(sample.video_path),
                    "frame_idx": sample.frame_idx,
                    "crop_rect": list(sample.crop_rect),
                    "lines": [
                        {"start": list(line.start), "end": list(line.end)}
                        for line in sample.lines
                    ]
                }
                for sample in self.samples
            ]
        }
    
    @classmethod
    def load(cls, filepath: str, base_dir: str = ".") -> 'AnnotationDatabase':
        """Load from JSON file."""
        if not os.path.exists(filepath):
            return cls()
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        db = cls()
        for sample_data in data.get("samples", []):
            # Resolve relative video path
            video_path = sample_data["video_path"]
            if not os.path.isabs(video_path):
                video_path = os.path.join(base_dir, video_path)
            
            sample = Sample(
                video_path=video_path,
                frame_idx=sample_data["frame_idx"],
                crop_rect=tuple(sample_data["crop_rect"])
            )
            for line_data in sample_data.get("lines", []):
                sample.add_line(
                    tuple(line_data["start"]),
                    tuple(line_data["end"])
                )
            db.samples.append(sample)
        
        return db


def find_line_intersections(lines: List[Line]) -> List[Tuple[float, float]]:
    """
    Find all intersection points between line segments.
    Returns list of (x, y) intersection coordinates.
    """
    intersections = []
    
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            
            # Line 1: from (x1, y1) to (x2, y2)
            x1, y1 = line1.start
            x2, y2 = line1.end
            
            # Line 2: from (x3, y3) to (x4, y4)
            x3, y3 = line2.start
            x4, y4 = line2.end
            
            # Calculate intersection using parametric form
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            
            if abs(denom) < 1e-10:
                continue  # Lines are parallel
            
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
            
            # Check if intersection is within both line segments
            if 0 <= t <= 1 and 0 <= u <= 1:
                ix = x1 + t * (x2 - x1)
                iy = y1 + t * (y2 - y1)
                intersections.append((ix, iy))
    
    return intersections


def render_gaussian_blob(mask: np.ndarray, cx: float, cy: float, sigma: float = 3.5):
    """
    Render a 2D Gaussian blob onto the mask at (cx, cy).
    Adds to existing values (for overlapping blobs).
    """
    h, w = mask.shape[:2]
    
    # Only render within a reasonable radius (3*sigma covers 99.7%)
    radius = int(np.ceil(3 * sigma))
    
    x_min = max(0, int(cx - radius))
    x_max = min(w, int(cx + radius) + 1)
    y_min = max(0, int(cy - radius))
    y_max = min(h, int(cy + radius) + 1)
    
    if x_min >= x_max or y_min >= y_max:
        return
    
    # Create coordinate grids for the local region
    y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
    
    # Compute Gaussian
    dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
    gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
    
    # Add to mask (use maximum to avoid over-saturation with overlapping blobs)
    mask[y_min:y_max, x_min:x_max] = np.maximum(
        mask[y_min:y_max, x_min:x_max],
        gaussian
    )


class DataGenViewer:
    """Standalone viewer for data generation / augmented training patches."""
    
    def __init__(self, videos_dir: str = "videos", annotations_file: str = "line_annotations.json"):
        self.root = tk.Tk()
        self.root.title("Crossing Detection Data Generator")
        self.root.geometry("1200x800")
        
        # Find videos
        self.videos_dir = videos_dir
        self.video_paths = self._find_videos(videos_dir)
        
        # Load annotations
        base_dir = os.path.dirname(annotations_file) if os.path.dirname(annotations_file) else "."
        self.db = AnnotationDatabase.load(annotations_file, base_dir)
        
        # State
        self.generated_patches = []
        self.show_gen_mask = False
        self.gen_photo_image = None
        
        # Build UI
        self._build_ui()
        
        # Keybindings
        self.root.bind("<r>", lambda e: self.generate_training_patches())
        self.root.bind("<m>", lambda e: self.toggle_generation_view())
        
        # Generate initial patches after UI is ready
        self.root.after(500, self._generate_initial_training_patches)
    
    def _find_videos(self, videos_dir: str) -> List[str]:
        """Find all video files in the given directory."""
        video_paths = []
        if os.path.isdir(videos_dir):
            for fname in os.listdir(videos_dir):
                if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_paths.append(os.path.join(videos_dir, fname))
        return sorted(video_paths)
    
    def _build_ui(self):
        """Build the UI."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas Area (Left)
        self.gen_canvas = tk.Canvas(main_frame, bg="#222222", highlightthickness=0)
        self.gen_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Sidebar (Right)
        sidebar = ttk.Frame(main_frame, width=300, padding=10)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        # Generation controls
        gen_frame = ttk.LabelFrame(sidebar, text="Generation Controls", padding=10)
        gen_frame.pack(fill=tk.X, pady=(10, 10))
        
        ttk.Label(gen_frame, text="Patch size: 128x128").pack(anchor="w", pady=2)
        ttk.Label(gen_frame, text="Grid: 3x3 (9 patches)").pack(anchor="w", pady=2)
        
        btn_generate = ttk.Button(gen_frame, text="Generate Patches (R)", command=self.generate_training_patches)
        btn_generate.pack(fill=tk.X, pady=5)
        
        self.btn_toggle_gen_view = ttk.Button(gen_frame, text="Show: Images", command=self.toggle_generation_view)
        self.btn_toggle_gen_view.pack(fill=tk.X, pady=5)
        
        # Generation info
        info_frame = ttk.LabelFrame(sidebar, text="Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_gen_count = ttk.Label(info_frame, text="Patches: 0")
        self.lbl_gen_count.pack(anchor="w", pady=2)
        
        self.lbl_samples = ttk.Label(info_frame, text=f"Labeled samples: {len([s for s in self.db.samples if s.lines])}")
        self.lbl_samples.pack(anchor="w", pady=2)
        
        self.lbl_videos = ttk.Label(info_frame, text=f"Videos: {len(self.video_paths)}")
        self.lbl_videos.pack(anchor="w", pady=2)
        
        # Keyboard shortcuts
        help_frame = ttk.LabelFrame(sidebar, text="Keyboard Shortcuts", padding=10)
        help_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(help_frame, text="R - Regenerate patches").pack(anchor="w", pady=1)
        ttk.Label(help_frame, text="M - Toggle crossings/image").pack(anchor="w", pady=1)
    
    def run(self):
        """Start the application."""
        self.root.mainloop()
    
    def _generate_single_patch(self, sample, frame, include_visualization=False):
        """Generate a single augmented training patch + mask from a sample."""
        x, y, w, h = sample.crop_rect
        crop = frame[y:y+h, x:x+w]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Random augmentation parameters
        zoom_factor = random.uniform(0.75, 2.0)  # Extended zoom range for more variety
        rotation_angle = random.uniform(-180, 180)
        stretch_x = random.uniform(0.9, 1.1)
        stretch_y = random.uniform(0.9, 1.1)
        
        perspective_strength = 0.15
        perspective_corners = [
            (random.uniform(-perspective_strength, perspective_strength),
             random.uniform(-perspective_strength, perspective_strength))
            for _ in range(4)
        ]
        
        flip_horizontal = random.random() < 0.5
        flip_vertical = random.random() < 0.5
        
        # Increase buffer for larger zoom range
        buffer_factor = 3.0
        initial_size = int(128 * buffer_factor)
        
        if w < initial_size or h < initial_size:
            patch_x, patch_y = 0, 0
            patch_w, patch_h = w, h
        else:
            patch_x = random.randint(0, w - initial_size)
            patch_y = random.randint(0, h - initial_size)
            patch_w, patch_h = initial_size, initial_size
        
        large_patch = crop_rgb[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
        
        # Find line intersections (crossings)
        intersections = find_line_intersections(sample.lines)
        
        # Translate intersections to large patch coordinates
        intersections_local = [(ix - patch_x, iy - patch_y) for ix, iy in intersections]
        
        max_attempts = 10
        for attempt in range(max_attempts):
            if attempt > 0:
                perspective_corners = [
                    (random.uniform(-perspective_strength, perspective_strength),
                     random.uniform(-perspective_strength, perspective_strength))
                    for _ in range(4)
                ]
            
            center_x, center_y = patch_w / 2, patch_h / 2
            rad = np.deg2rad(rotation_angle)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)
            scale_x = zoom_factor * stretch_x
            scale_y = zoom_factor * stretch_y
            
            src_corners = np.array([
                [0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]
            ], dtype=np.float32)
            
            dst_corners = []
            for i, (sx, sy) in enumerate(src_corners):
                x_c = sx - center_x
                y_c = sy - center_y
                xr = x_c * cos_a - y_c * sin_a
                yr = x_c * sin_a + y_c * cos_a
                xs = xr * scale_x
                ys = yr * scale_y
                xf = xs + center_x
                yf = ys + center_y
                px, py = perspective_corners[i]
                xf += px * patch_w
                yf += py * patch_h
                dst_corners.append([xf, yf])
            
            dst_corners = np.array(dst_corners, dtype=np.float32)
            H = cv2.getPerspectiveTransform(src_corners, dst_corners)
            H_inverse = np.linalg.inv(H)
            crop_x_offset = (patch_w - 128) // 2
            crop_y_offset = (patch_h - 128) // 2
            
            output_corners = np.array([
                [crop_x_offset, crop_y_offset],
                [crop_x_offset + 128, crop_y_offset],
                [crop_x_offset + 128, crop_y_offset + 128],
                [crop_x_offset, crop_y_offset + 128]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            source_corners_check = cv2.perspectiveTransform(output_corners, H_inverse).reshape(-1, 2)
            
            margin = 2
            valid = True
            for sx, sy in source_corners_check:
                if sx < margin or sx > patch_w - margin or sy < margin or sy > patch_h - margin:
                    valid = False
                    break
            
            if valid:
                break
        
        # Transform image
        transformed_img = cv2.warpPerspective(large_patch, H, (patch_w, patch_h),
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        # Crop to 128x128
        final_image = transformed_img[crop_y_offset:crop_y_offset+128, crop_x_offset:crop_x_offset+128]
        
        # Transform intersection points through homography and filter by edge margin
        edge_margin = 128 // 10  # 1/10 of patch width = ~12 pixels
        valid_crossings = []
        
        if intersections_local:
            # Convert points to homogeneous coordinates and transform
            points = np.array(intersections_local, dtype=np.float32).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(points, H).reshape(-1, 2)
            
            # Convert to final 128x128 coordinates and filter by edge margin
            for tx, ty in transformed_points:
                # Translate to 128x128 crop coordinates
                fx = tx - crop_x_offset
                fy = ty - crop_y_offset
                
                # Check edge margin (must be at least edge_margin pixels from any edge)
                if fx >= edge_margin and fx <= 128 - edge_margin and fy >= edge_margin and fy <= 128 - edge_margin:
                    valid_crossings.append((fx, fy))
        
        # Create mask with Gaussian blobs at valid crossing points
        final_mask = np.zeros((128, 128), dtype=np.float32)
        for cx, cy in valid_crossings:
            render_gaussian_blob(final_mask, cx, cy, sigma=3.5)
        
        # Apply flips
        if flip_horizontal:
            final_image = cv2.flip(final_image, 1)
            final_mask = cv2.flip(final_mask, 1)
            # Update crossing positions for visualization
            valid_crossings = [(128 - cx, cy) for cx, cy in valid_crossings]
        if flip_vertical:
            final_image = cv2.flip(final_image, 0)
            final_mask = cv2.flip(final_mask, 0)
            valid_crossings = [(cx, 128 - cy) for cx, cy in valid_crossings]
        
        # Convert mask to uint8 for display (0-255 grayscale)
        final_mask = np.clip(final_mask, 0, 1)
        final_mask = (final_mask * 255).astype(np.uint8)
        
        # Track if this patch has valid crossings (for balancing)
        has_crossing = len(valid_crossings) > 0
        
        final_image = final_image.astype(np.float32)
        
        brightness = random.uniform(-0.3, 0.3)
        final_image = final_image + brightness * 255
        
        contrast = random.uniform(0.7, 1.3)
        mean = np.mean(final_image)
        final_image = (final_image - mean) * contrast + mean
        
        gamma = random.uniform(0.7, 1.5)
        final_image = np.clip(final_image, 0, 255)
        final_image = 255.0 * np.power(final_image / 255.0, gamma)
        
        noise_sigma = random.uniform(0, 25)
        if noise_sigma > 0:
            noise = np.random.normal(0, noise_sigma, final_image.shape)
            final_image = final_image + noise
        
        if random.random() < 0.5:
            blur_sigma = random.uniform(0.5, 1.5)
            final_image = cv2.GaussianBlur(final_image.astype(np.float32), (0, 0), blur_sigma)
        
        final_image = np.clip(final_image, 0, 255).astype(np.uint8)
        
        result = {
            'image': final_image,
            'mask': final_mask,
            'has_crossing': has_crossing,
            'num_crossings': len(valid_crossings)
        }
        
        if include_visualization:
            # Create visualization mask showing ALL crossings in source (before edge filtering)
            full_source_mask = np.zeros((h, w), dtype=np.float32)
            for ix, iy in intersections:
                render_gaussian_blob(full_source_mask, ix, iy, sigma=3.5)
            full_source_mask = np.clip(full_source_mask, 0, 1)
            full_source_mask = (full_source_mask * 255).astype(np.uint8)
            full_source_mask = cv2.cvtColor(full_source_mask, cv2.COLOR_GRAY2RGB)
            
            result.update({
                'source_video': os.path.basename(sample.video_path),
                'source_frame': sample.frame_idx,
                'source_crop': sample.crop_rect,
                'source_mask': full_source_mask,
                'patch_offset': (patch_x, patch_y, patch_w, patch_h),
                'total_intersections': len(intersections),
                'valid_crossings': valid_crossings,
                'step3_params': {
                    'rotation': rotation_angle,
                    'zoom': zoom_factor,
                    'stretch_x': stretch_x,
                    'stretch_y': stretch_y,
                    'perspective': perspective_corners,
                    'flip_h': flip_horizontal,
                    'flip_v': flip_vertical
                },
                'homography': H,
                'step5_final': final_image
            })
        
        return result
    
    def _generate_initial_training_patches(self):
        """Generate initial training patches automatically on app start."""
        labeled_samples = [s for s in self.db.samples if len(s.lines) > 0]
        if not labeled_samples:
            return
        threading.Thread(target=self.generate_training_patches, daemon=True).start()
    
    def generate_training_patches(self):
        """Generate random 128x128 patches with 50/50 balance of positive/negative."""
        self.root.after(0, lambda: self.lbl_gen_count.config(text="Generating..."))
        
        self.generated_patches = []
        
        # Separate samples: those that CAN have crossings (2+ lines) vs those that can't
        samples_with_crossings = [s for s in self.db.samples if len(s.lines) >= 2]
        samples_no_crossings = [s for s in self.db.samples if len(s.lines) == 1]
        all_labeled = samples_with_crossings + samples_no_crossings
        
        if not all_labeled:
            self.root.after(0, lambda: messagebox.showwarning("No Labeled Data", "No labeled samples found."))
            return
        
        print(f"Generating balanced patches: {len(samples_with_crossings)} samples with 2+ lines, {len(samples_no_crossings)} with 1 line...")
        
        # Target: 5 positive, 4 negative (or vice versa) for 9 total
        target_positive = 5
        target_negative = 4
        max_attempts = 200  # Prevent infinite loop
        
        positive_patches = []
        negative_patches = []
        frame_cache = {}
        attempts = 0
        
        while (len(positive_patches) < target_positive or len(negative_patches) < target_negative) and attempts < max_attempts:
            attempts += 1
            
            # Choose sample strategically based on what we still need
            if len(positive_patches) < target_positive and samples_with_crossings:
                # Try to get a positive - use sample with 2+ lines
                sample = random.choice(samples_with_crossings)
            elif len(negative_patches) < target_negative:
                # Get a negative - can use any sample (even 2+ lines might not produce crossing in patch)
                sample = random.choice(all_labeled)
            else:
                break
            
            # Load frame
            cache_key = (sample.video_path, sample.frame_idx)
            if cache_key not in frame_cache:
                cap = cv2.VideoCapture(sample.video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_idx)
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    frame_cache[cache_key] = frame
            
            if cache_key not in frame_cache:
                continue
            
            frame = frame_cache[cache_key]
            patch_data = self._generate_single_patch(sample, frame, include_visualization=True)
            
            # Categorize by whether it has valid crossings
            if patch_data['has_crossing'] and len(positive_patches) < target_positive:
                positive_patches.append(patch_data)
            elif not patch_data['has_crossing'] and len(negative_patches) < target_negative:
                negative_patches.append(patch_data)
        
        # Combine and shuffle
        new_patches = positive_patches + negative_patches
        random.shuffle(new_patches)
        
        print(f"Generated {len(new_patches)} patches: {len(positive_patches)} positive, {len(negative_patches)} negative ({attempts} attempts)")
        
        def update_ui():
            self.generated_patches = new_patches
            if self.generated_patches:
                self.show_gen_mask = False
                self.display_gen_grid()
        
        self.root.after(0, update_ui)
    
    def display_gen_grid(self):
        """Display 3x3 grid showing: Full region with green highlight | Final 128x128 warped patch."""
        if not self.generated_patches:
            return
        
        num_positive = sum(1 for p in self.generated_patches if p.get('has_crossing', False))
        num_negative = len(self.generated_patches) - num_positive
        total_crossings = sum(p.get('num_crossings', 0) for p in self.generated_patches)
        self.lbl_gen_count.config(text=f"+{num_positive}/-{num_negative} | {total_crossings} crossings")
        
        if self.show_gen_mask:
            self.btn_toggle_gen_view.config(text="Show: Crossing Masks")
        else:
            self.btn_toggle_gen_view.config(text="Show: Images")
        
        num_rows = 3
        num_cols = 3
        region_size = 180
        patch_size = 128
        padding = 6
        cell_spacing = 12
        
        cell_width = region_size + padding + patch_size
        cell_height = region_size
        
        grid_width = num_cols * cell_width + (num_cols - 1) * cell_spacing + 2 * padding
        grid_height = num_rows * cell_height + (num_rows - 1) * cell_spacing + 2 * padding
        
        grid_img = np.full((grid_height, grid_width, 3), 32, dtype=np.uint8)
        
        patch_idx = 0
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                if patch_idx >= len(self.generated_patches):
                    break
                
                patch = self.generated_patches[patch_idx]
                patch_idx += 1
                
                cell_x = padding + col_idx * (cell_width + cell_spacing)
                cell_y = padding + row_idx * (cell_height + cell_spacing)
                
                source_video = patch.get("source_video")
                source_frame = patch.get("source_frame")
                source_crop = patch.get("source_crop")
                patch_offset = patch.get("patch_offset")
                
                left_img = None
                if source_video and source_frame is not None and source_crop:
                    video_path = None
                    for vp in self.video_paths:
                        if os.path.basename(vp) == source_video:
                            video_path = vp
                            break
                    
                    if video_path:
                        cap = cv2.VideoCapture(video_path)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
                        ret, frame = cap.read()
                        cap.release()
                        
                        if ret and frame is not None:
                            x, y, w, h = source_crop
                            left_img = frame[y:y+h, x:x+w]
                            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
                
                if left_img is None:
                    left_img = np.full((region_size, region_size, 3), 64, dtype=np.uint8)
                
                orig_h, orig_w = left_img.shape[:2]
                left_img = cv2.resize(left_img, (region_size, region_size), interpolation=cv2.INTER_LINEAR)
                left_img_display = left_img.copy()
                
                display_corners = None
                if patch_offset and source_crop:
                    patch_x, patch_y, patch_w, patch_h = patch_offset
                    H = patch.get("homography")
                    
                    if H is not None:
                        H_inverse = np.linalg.inv(H)
                        crop_x_offset = (patch_w - 128) // 2
                        crop_y_offset = (patch_h - 128) // 2
                        
                        output_corners = np.array([
                            [crop_x_offset, crop_y_offset],
                            [crop_x_offset + 128, crop_y_offset],
                            [crop_x_offset + 128, crop_y_offset + 128],
                            [crop_x_offset, crop_y_offset + 128]
                        ], dtype=np.float32).reshape(-1, 1, 2)
                        
                        source_corners = cv2.perspectiveTransform(output_corners, H_inverse).reshape(-1, 2)
                        source_corners[:, 0] += patch_x
                        source_corners[:, 1] += patch_y
                        
                        _, _, crop_w, crop_h = source_crop
                        scale_x_display = region_size / crop_w
                        scale_y_display = region_size / crop_h
                        
                        display_corners = source_corners.copy()
                        display_corners[:, 0] *= scale_x_display
                        display_corners[:, 1] *= scale_y_display
                        display_corners = display_corners.astype(np.int32)
                
                if self.show_gen_mask:
                    source_mask = patch.get("source_mask")
                    if source_mask is not None:
                        mask_resized = cv2.resize(source_mask, (region_size, region_size), interpolation=cv2.INTER_LINEAR)
                        mask_gray = cv2.cvtColor(mask_resized, cv2.COLOR_RGB2GRAY) if len(mask_resized.shape) == 3 else mask_resized
                        left_img_display[mask_gray > 128] = [255, 255, 255]
                    if display_corners is not None:
                        cv2.polylines(left_img_display, [display_corners], isClosed=True, color=(0, 255, 0), thickness=2)
                else:
                    if display_corners is not None:
                        cv2.polylines(left_img_display, [display_corners], isClosed=True, color=(0, 255, 0), thickness=2)
                
                step3_params = patch.get("step3_params", {})
                flip_h = step3_params.get("flip_h", False)
                flip_v = step3_params.get("flip_v", False)
                
                if (flip_h or flip_v) and display_corners is not None:
                    cx = int(np.mean(display_corners[:, 0]))
                    cy = int(np.mean(display_corners[:, 1]))
                    
                    if flip_h:
                        cv2.arrowedLine(left_img_display, (cx - 15, cy - 10), (cx + 15, cy - 10), (255, 255, 0), 2, tipLength=0.3)
                        cv2.arrowedLine(left_img_display, (cx + 15, cy - 10), (cx - 15, cy - 10), (255, 255, 0), 2, tipLength=0.3)
                    
                    if flip_v:
                        cv2.arrowedLine(left_img_display, (cx, cy - 15 + (10 if flip_h else 0)), (cx, cy + 15 + (10 if flip_h else 0)), (255, 0, 255), 2, tipLength=0.3)
                        cv2.arrowedLine(left_img_display, (cx, cy + 15 + (10 if flip_h else 0)), (cx, cy - 15 + (10 if flip_h else 0)), (255, 0, 255), 2, tipLength=0.3)
                
                grid_img[cell_y:cell_y+region_size, cell_x:cell_x+region_size] = left_img_display
                
                if self.show_gen_mask:
                    right_img = patch.get("mask")
                    if right_img is None:
                        right_img = np.full((patch_size, patch_size, 3), 64, dtype=np.uint8)
                    if len(right_img.shape) == 2:
                        right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2RGB)
                else:
                    right_img = patch.get("step5_final")
                    if right_img is None:
                        right_img = np.full((patch_size, patch_size, 3), 64, dtype=np.uint8)
                
                rh, rw = right_img.shape[:2]
                if rh != patch_size or rw != patch_size:
                    right_img = cv2.resize(right_img, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
                
                right_x = cell_x + region_size + padding
                right_y_offset = (region_size - patch_size) // 2
                grid_img[cell_y + right_y_offset:cell_y + right_y_offset + patch_size,
                         right_x:right_x + patch_size] = right_img
        
        self._draw_generated_image(grid_img)
    
    def _draw_generated_image(self, img_arr):
        """Draw a generated patch on the generation canvas."""
        img_h, img_w = img_arr.shape[:2]
        
        canvas_w = self.gen_canvas.winfo_width()
        canvas_h = self.gen_canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            self.root.after(100, lambda: self._draw_generated_image(img_arr))
            return
        
        scale_w = canvas_w * 0.95 / img_w
        scale_h = canvas_h * 0.95 / img_h
        scale = min(scale_w, scale_h)
        
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        interp = cv2.INTER_NEAREST if scale > 1.5 else cv2.INTER_LINEAR
        resized = cv2.resize(img_arr, (new_w, new_h), interpolation=interp)
        
        img_pil = Image.fromarray(resized)
        self.gen_photo_image = ImageTk.PhotoImage(img_pil)
        
        offset_x = (canvas_w - new_w) // 2
        offset_y = (canvas_h - new_h) // 2
        
        self.gen_canvas.delete("all")
        self.gen_canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.gen_photo_image)
    
    def toggle_generation_view(self):
        """Toggle between images and masks view in data generation."""
        self.show_gen_mask = not self.show_gen_mask
        self.display_gen_grid()


def main():
    parser = argparse.ArgumentParser(description="Standalone Data Generation Viewer")
    parser.add_argument("--videos", default="videos", help="Path to videos directory")
    parser.add_argument("--annotations", default="line_annotations.json", help="Path to annotations JSON file")
    args = parser.parse_args()
    
    app = DataGenViewer(videos_dir=args.videos, annotations_file=args.annotations)
    app.run()


if __name__ == "__main__":
    main()

