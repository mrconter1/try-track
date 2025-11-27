"""
Standalone Data Generation Viewer - Extracted from line_annotator_gui.py

Visualizes augmented training patches from labeled annotation data.
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


class DataGenViewer:
    """Standalone viewer for data generation / augmented training patches."""
    
    def __init__(self, videos_dir: str = "videos", annotations_file: str = "line_annotations.json"):
        self.root = tk.Tk()
        self.root.title("Data Generation Viewer")
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
        ttk.Label(help_frame, text="M - Toggle mask/image").pack(anchor="w", pady=1)
    
    def run(self):
        """Start the application."""
        self.root.mainloop()
    
    def _generate_single_patch(self, sample, frame, include_visualization=False):
        """Generate a single augmented training patch + mask from a sample."""
        x, y, w, h = sample.crop_rect
        crop = frame[y:y+h, x:x+w]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Random augmentation parameters
        zoom_factor = random.uniform(0.75, 1.25)
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
        
        buffer_factor = 2.5
        initial_size = int(128 * buffer_factor)
        
        if w < initial_size or h < initial_size:
            patch_x, patch_y = 0, 0
            patch_w, patch_h = w, h
        else:
            patch_x = random.randint(0, w - initial_size)
            patch_y = random.randint(0, h - initial_size)
            patch_w, patch_h = initial_size, initial_size
        
        large_patch = crop_rgb[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
        
        mask_large = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
        for line in sample.lines:
            p1_x, p1_y = line.start
            p2_x, p2_y = line.end
            p1_patch = (int(p1_x - patch_x), int(p1_y - patch_y))
            p2_patch = (int(p2_x - patch_x), int(p2_y - patch_y))
            cv2.line(mask_large, p1_patch, p2_patch, (255, 255, 255), thickness=1)
        
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
        
        transformed_img = cv2.warpPerspective(large_patch, H, (patch_w, patch_h),
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        transformed_mask = cv2.warpPerspective(mask_large, H, (patch_w, patch_h),
                                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        final_image = transformed_img[crop_y_offset:crop_y_offset+128, crop_x_offset:crop_x_offset+128]
        final_mask = transformed_mask[crop_y_offset:crop_y_offset+128, crop_x_offset:crop_x_offset+128]
        
        if flip_horizontal:
            final_image = cv2.flip(final_image, 1)
            final_mask = cv2.flip(final_mask, 1)
        if flip_vertical:
            final_image = cv2.flip(final_image, 0)
            final_mask = cv2.flip(final_mask, 0)
        
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
        
        result = {'image': final_image, 'mask': final_mask}
        
        if include_visualization:
            full_source_mask = np.zeros((h, w, 3), dtype=np.uint8)
            for line in sample.lines:
                p1 = (int(line.start[0]), int(line.start[1]))
                p2 = (int(line.end[0]), int(line.end[1]))
                cv2.line(full_source_mask, p1, p2, (255, 255, 255), thickness=1)
            
            result.update({
                'source_video': os.path.basename(sample.video_path),
                'source_frame': sample.frame_idx,
                'source_crop': sample.crop_rect,
                'source_mask': full_source_mask,
                'patch_offset': (patch_x, patch_y, patch_w, patch_h),
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
        """Generate random 128x128 patches from labeled samples for visualization."""
        self.root.after(0, lambda: self.lbl_gen_count.config(text="Generating..."))
        
        self.generated_patches = []
        new_patches = []
        
        labeled_samples = [s for s in self.db.samples if len(s.lines) > 0]
        
        if not labeled_samples:
            self.root.after(0, lambda: messagebox.showwarning("No Labeled Data", "No labeled samples found."))
            return
        
        print(f"Generating patches from {len(labeled_samples)} labeled samples...")
        
        frame_cache = {}
        selected_samples = [random.choice(labeled_samples) for _ in range(9)]
        
        for sample in selected_samples:
            cache_key = (sample.video_path, sample.frame_idx)
            if cache_key not in frame_cache:
                cap = cv2.VideoCapture(sample.video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_idx)
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    frame_cache[cache_key] = frame
        
        for sample in selected_samples:
            cache_key = (sample.video_path, sample.frame_idx)
            if cache_key not in frame_cache:
                continue
            
            frame = frame_cache[cache_key]
            patch_data = self._generate_single_patch(sample, frame, include_visualization=True)
            new_patches.append(patch_data)
        
        print(f"Generated {len(new_patches)} patches")
        
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
        
        self.lbl_gen_count.config(text=f"Patches: {len(self.generated_patches)}")
        
        if self.show_gen_mask:
            self.btn_toggle_gen_view.config(text="Show: Masks")
        else:
            self.btn_toggle_gen_view.config(text="Show: Step 5 Results")
        
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

