"""
Crossing Detection Data Generator

Generates augmented training patches for crossing/intersection detection.
Each crossing is rendered as a soft 2D Gaussian blob (sigma ~3.5) in the mask.

Usage: 
  GUI mode:   python data_gen_viewer.py
  Train mode: python data_gen_viewer.py --train 10000 --epochs 50 --batch-size 32
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
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

# PyTorch imports for training
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models


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


# ============================================================================
# Neural Network Model and Training
# ============================================================================

class MobileUNet(nn.Module):
    """Mobile-optimized U-Net with MobileNetV2 backbone for crossing detection."""
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained MobileNetV2 as encoder
        if pretrained:
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            mobilenet = models.mobilenet_v2(weights=None)
        self.encoder = mobilenet.features
        
        # Decoder (lightweight upsampling path)
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
        
        # Final upsampling to original resolution
        self.final_up = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.out = nn.Sequential(
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()  # Output 0-1 for grayscale heatmap
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


class CrossingDataset(Dataset):
    """Dataset for crossing detection training."""
    
    def __init__(self, samples):
        self.samples = samples
        # ImageNet normalization for pretrained MobileNetV2
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image'].astype(np.float32) / 255.0
        mask = sample['mask'].astype(np.float32) / 255.0
        
        # Convert to tensors (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Apply ImageNet normalization
        mean = torch.from_numpy(self.mean).reshape(3, 1, 1)
        std = torch.from_numpy(self.std).reshape(3, 1, 1)
        image = (image - mean) / std
        
        # Mask is single channel
        if len(mask.shape) == 2:
            mask = torch.from_numpy(mask).unsqueeze(0)
        else:
            mask = torch.from_numpy(mask[:, :, 0]).unsqueeze(0)
        
        return image, mask


class DataGenViewer:
    """Standalone viewer for data generation / augmented training patches."""
    
    def __init__(self, videos_dir: str = "videos", annotations_file: str = "line_annotations.json"):
        self.root = tk.Tk()
        self.root.title("Crossing Detection Data Generator")
        self.root.state('zoomed')  # Launch maximized
        
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
    
    def _generate_single_patch(self, sample, frame, include_visualization=False, force_crossing=False):
        """Generate a single augmented training patch + mask from a sample.
        
        Args:
            sample: Sample object with lines
            frame: Video frame (BGR)
            include_visualization: Include extra data for GUI display
            force_crossing: If True, center on a crossing and retry until valid
        """
        x, y, w, h = sample.crop_rect
        crop = frame[y:y+h, x:x+w]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Find line intersections (crossings)
        intersections = find_line_intersections(sample.lines)
        
        # Determine if we should center on a crossing
        center_on_crossing = force_crossing and len(intersections) > 0
        target_crossing = random.choice(intersections) if center_on_crossing else None
        
        buffer_factor = 3.0
        initial_size = int(128 * buffer_factor)
        
        # Patch placement - center on crossing if requested
        if center_on_crossing:
            # Center the patch on the target crossing
            target_x, target_y = target_crossing
            patch_x = int(target_x - initial_size / 2)
            patch_y = int(target_y - initial_size / 2)
            # Clamp to valid range
            patch_x = max(0, min(w - initial_size, patch_x))
            patch_y = max(0, min(h - initial_size, patch_y))
            patch_w, patch_h = min(initial_size, w), min(initial_size, h)
        else:
            # Random placement
            if w < initial_size or h < initial_size:
                patch_x, patch_y = 0, 0
                patch_w, patch_h = w, h
            else:
                patch_x = random.randint(0, w - initial_size)
                patch_y = random.randint(0, h - initial_size)
                patch_w, patch_h = initial_size, initial_size
        
        large_patch = crop_rgb[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
        
        # Translate intersections to large patch coordinates
        intersections_local = [(ix - patch_x, iy - patch_y) for ix, iy in intersections]
        
        edge_margin = 128 // 10  # 1/10 of patch width = ~12 pixels
        max_augment_attempts = 20 if center_on_crossing else 10
        
        valid_crossings = []
        final_image = None
        H = None
        
        for attempt in range(max_augment_attempts):
            # Random augmentation parameters (re-roll each attempt)
            zoom_factor = random.uniform(0.75, 2.0)
            rotation_angle = random.uniform(-180, 180)
            stretch_x = random.uniform(0.9, 1.1)
            stretch_y = random.uniform(0.9, 1.1)
            
            perspective_strength = 0.15
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
            
            # Check if output region maps to valid source pixels
            output_corners = np.array([
                [crop_x_offset, crop_y_offset],
                [crop_x_offset + 128, crop_y_offset],
                [crop_x_offset + 128, crop_y_offset + 128],
                [crop_x_offset, crop_y_offset + 128]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            source_corners_check = cv2.perspectiveTransform(output_corners, H_inverse).reshape(-1, 2)
            
            margin = 2
            valid_transform = True
            for sx, sy in source_corners_check:
                if sx < margin or sx > patch_w - margin or sy < margin or sy > patch_h - margin:
                    valid_transform = False
                    break
            
            if not valid_transform:
                continue
            
            # Check if crossing is in valid zone (for force_crossing mode)
            valid_crossings = []
            if intersections_local:
                points = np.array(intersections_local, dtype=np.float32).reshape(-1, 1, 2)
                transformed_points = cv2.perspectiveTransform(points, H).reshape(-1, 2)
                
                for tx, ty in transformed_points:
                    fx = tx - crop_x_offset
                    fy = ty - crop_y_offset
                    
                    if fx >= edge_margin and fx <= 128 - edge_margin and fy >= edge_margin and fy <= 128 - edge_margin:
                        valid_crossings.append((fx, fy))
            
            # If force_crossing, we need at least one valid crossing
            if center_on_crossing and len(valid_crossings) == 0:
                continue  # Retry with new augmentation
            
            # Success - apply transform
            transformed_img = cv2.warpPerspective(large_patch, H, (patch_w, patch_h),
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            final_image = transformed_img[crop_y_offset:crop_y_offset+128, crop_x_offset:crop_x_offset+128]
            break
        
        # Fallback if no valid augmentation found
        if final_image is None:
            crop_x_offset = (patch_w - 128) // 2
            crop_y_offset = (patch_h - 128) // 2
            final_image = large_patch[crop_y_offset:crop_y_offset+128, crop_x_offset:crop_x_offset+128]
            if final_image.shape[0] < 128 or final_image.shape[1] < 128:
                final_image = cv2.resize(large_patch, (128, 128))
            valid_crossings = []
        
        # Apply flips
        flip_horizontal = random.random() < 0.5
        flip_vertical = random.random() < 0.5
        
        # Create mask with Gaussian blobs at valid crossing points
        final_mask = np.zeros((128, 128), dtype=np.float32)
        for cx, cy in valid_crossings:
            render_gaussian_blob(final_mask, cx, cy, sigma=3.5)
        
        if flip_horizontal:
            final_image = cv2.flip(final_image, 1)
            final_mask = cv2.flip(final_mask, 1)
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
            
            # For positive samples, use force_crossing=True to center on crossing
            want_positive = len(positive_patches) < target_positive and len(sample.lines) >= 2
            patch_data = self._generate_single_patch(sample, frame, include_visualization=True, force_crossing=want_positive)
            
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


# ============================================================================
# Standalone Training Functions
# ============================================================================

# Global cache for multiprocessing workers
_worker_frame_cache = None
_worker_video_paths = None

def _init_worker(frame_cache, video_paths):
    """Initialize worker process with shared frame cache."""
    global _worker_frame_cache, _worker_video_paths
    _worker_frame_cache = frame_cache
    _worker_video_paths = video_paths

def _generate_one_sample(args):
    """Worker function - inlined for speed."""
    sample_data, cache_key, force_crossing = args
    
    global _worker_frame_cache
    
    frame = _worker_frame_cache.get(cache_key)
    if frame is None:
        return None
    
    # Inline the patch generation for speed (avoid object creation)
    crop_rect = sample_data['crop_rect']
    lines_data = sample_data['lines']
    
    x, y, w, h = crop_rect
    crop = frame[y:y+h, x:x+w]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    # Find intersections inline
    intersections = []
    for i in range(len(lines_data)):
        for j in range(i + 1, len(lines_data)):
            l1, l2 = lines_data[i], lines_data[j]
            x1, y1 = l1['start']
            x2, y2 = l1['end']
            x3, y3 = l2['start']
            x4, y4 = l2['end']
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:
                continue
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
            if 0 <= t <= 1 and 0 <= u <= 1:
                intersections.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    
    center_on_crossing = force_crossing and len(intersections) > 0
    target_crossing = random.choice(intersections) if center_on_crossing else None
    
    buffer_factor = 3.0
    initial_size = int(128 * buffer_factor)
    
    if center_on_crossing:
        target_x, target_y = target_crossing
        patch_x = max(0, min(w - initial_size, int(target_x - initial_size / 2)))
        patch_y = max(0, min(h - initial_size, int(target_y - initial_size / 2)))
        patch_w, patch_h = min(initial_size, w), min(initial_size, h)
    else:
        if w < initial_size or h < initial_size:
            patch_x, patch_y = 0, 0
            patch_w, patch_h = w, h
        else:
            patch_x = random.randint(0, w - initial_size)
            patch_y = random.randint(0, h - initial_size)
            patch_w, patch_h = initial_size, initial_size
    
    large_patch = crop_rgb[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
    intersections_local = [(ix - patch_x, iy - patch_y) for ix, iy in intersections]
    
    edge_margin = 12
    max_attempts = 20 if center_on_crossing else 10
    valid_crossings = []
    final_image = None
    
    for _ in range(max_attempts):
        zoom = random.uniform(0.75, 2.0)
        angle = random.uniform(-180, 180)
        stretch_x, stretch_y = random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)
        
        cx, cy = patch_w / 2, patch_h / 2
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        sx, sy = zoom * stretch_x, zoom * stretch_y
        
        src = np.array([[0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]], dtype=np.float32)
        dst = []
        for i, (px, py) in enumerate(src):
            rx, ry = (px - cx) * cos_a - (py - cy) * sin_a, (px - cx) * sin_a + (py - cy) * cos_a
            fx = rx * sx + cx + random.uniform(-0.15, 0.15) * patch_w
            fy = ry * sy + cy + random.uniform(-0.15, 0.15) * patch_h
            dst.append([fx, fy])
        
        dst = np.array(dst, dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        
        crop_off = (patch_w - 128) // 2
        
        # Check crossings
        valid_crossings = []
        if intersections_local:
            pts = np.array(intersections_local, dtype=np.float32).reshape(-1, 1, 2)
            tpts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
            for tx, ty in tpts:
                fx, fy = tx - crop_off, ty - crop_off
                if edge_margin <= fx <= 128 - edge_margin and edge_margin <= fy <= 128 - edge_margin:
                    valid_crossings.append((fx, fy))
        
        if center_on_crossing and not valid_crossings:
            continue
        
        transformed = cv2.warpPerspective(large_patch, H, (patch_w, patch_h), borderMode=cv2.BORDER_CONSTANT)
        final_image = transformed[crop_off:crop_off+128, crop_off:crop_off+128]
        break
    
    if final_image is None:
        final_image = cv2.resize(large_patch, (128, 128))
        valid_crossings = []
    
    # Mask
    mask = np.zeros((128, 128), dtype=np.float32)
    for cx, cy in valid_crossings:
        render_gaussian_blob(mask, cx, cy, sigma=3.5)
    
    # Flips
    if random.random() < 0.5:
        final_image = cv2.flip(final_image, 1)
        mask = cv2.flip(mask, 1)
    if random.random() < 0.5:
        final_image = cv2.flip(final_image, 0)
        mask = cv2.flip(mask, 0)
    
    # Quick augmentations
    img = final_image.astype(np.float32)
    img = img + random.uniform(-0.3, 0.3) * 255
    img = (img - img.mean()) * random.uniform(0.7, 1.3) + img.mean()
    img = np.clip(img, 0, 255)
    img = 255.0 * np.power(img / 255.0, random.uniform(0.7, 1.5))
    if random.random() < 0.5:
        img = img + np.random.normal(0, random.uniform(0, 25), img.shape)
    
    return {
        'image': np.clip(img, 0, 255).astype(np.uint8),
        'mask': (np.clip(mask, 0, 1) * 255).astype(np.uint8),
        'has_crossing': len(valid_crossings) > 0,
        'num_crossings': len(valid_crossings)
    }


def generate_training_samples(db: AnnotationDatabase, video_paths: List[str], 
                               num_samples: int, balance_ratio: float = 0.5) -> List[dict]:
    """
    Generate training samples with specified positive/negative balance.
    Uses pre-cached frames and parallel processing for speed.
    """
    samples_with_crossings = [s for s in db.samples if len(s.lines) >= 2]  # Can have crossings
    samples_for_negatives = [s for s in db.samples]  # All samples can be negatives (including 0 lines)
    
    if not db.samples:
        raise ValueError("No samples found in database")
    
    print(f"Sample pool: {len(samples_with_crossings)} with 2+ lines (for positives), {len(db.samples)} total (for negatives)")
    
    target_positive = int(num_samples * balance_ratio)
    target_negative = num_samples - target_positive
    
    # Step 1: Pre-cache ALL unique frames upfront (parallel)
    print("Pre-caching frames...")
    unique_frames = list(set((s.video_path, s.frame_idx) for s in db.samples))
    
    def load_frame(args):
        video_path, frame_idx = args
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return ((video_path, frame_idx), frame)
        return None
    
    frame_cache = {}
    num_workers = min(multiprocessing.cpu_count(), 8)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i, result in enumerate(executor.map(load_frame, unique_frames)):
            if result:
                frame_cache[result[0]] = result[1]
            if (i + 1) % 100 == 0:
                print(f"  Cached {i + 1}/{len(unique_frames)} frames")
    
    print(f"  Cached {len(frame_cache)} frames total")
    
    # Step 2: Prepare tasks
    print(f"Generating {num_samples} samples: {target_positive} positive, {target_negative} negative...")
    
    def sample_to_dict(s):
        return {
            'video_path': s.video_path,
            'frame_idx': s.frame_idx,
            'crop_rect': s.crop_rect,
            'lines': [{'start': l.start, 'end': l.end} for l in s.lines]
        }
    
    # Create tasks (don't pass frames, just cache keys)
    oversample = 1.5
    positive_tasks = []
    negative_tasks = []
    
    for _ in range(int(target_positive * oversample)):
        if samples_with_crossings:
            s = random.choice(samples_with_crossings)
            cache_key = (s.video_path, s.frame_idx)
            if cache_key in frame_cache:
                positive_tasks.append((sample_to_dict(s), cache_key, True))
    
    for _ in range(int(target_negative * oversample)):
        s = random.choice(samples_for_negatives)
        cache_key = (s.video_path, s.frame_idx)
        if cache_key in frame_cache:
            negative_tasks.append((sample_to_dict(s), cache_key, False))
    
    # Step 3: Process with multiprocessing
    num_workers = min(multiprocessing.cpu_count(), 8)
    print(f"Processing with {num_workers} workers...")
    
    # Use ThreadPoolExecutor (shares memory, no pickle overhead for frames)
    positive_samples = []
    negative_samples = []
    
    # Initialize global cache for workers
    global _worker_frame_cache, _worker_video_paths
    _worker_frame_cache = frame_cache
    _worker_video_paths = video_paths
    
    # Process in chunks with progress
    chunk_size = 2500
    
    print("Generating positive samples...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in range(0, len(positive_tasks), chunk_size):
            chunk = positive_tasks[i:i+chunk_size]
            results = list(executor.map(_generate_one_sample, chunk))
            for r in results:
                if r and r['has_crossing'] and len(positive_samples) < target_positive:
                    positive_samples.append(r)
            print(f"  Positive: {len(positive_samples)}/{target_positive} ({i+len(chunk)}/{len(positive_tasks)} processed)")
            if len(positive_samples) >= target_positive:
                break
    
    print("Generating negative samples...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in range(0, len(negative_tasks), chunk_size):
            chunk = negative_tasks[i:i+chunk_size]
            results = list(executor.map(_generate_one_sample, chunk))
            for r in results:
                if r and not r['has_crossing'] and len(negative_samples) < target_negative:
                    negative_samples.append(r)
            print(f"  Negative: {len(negative_samples)}/{target_negative} ({i+len(chunk)}/{len(negative_tasks)} processed)")
            if len(negative_samples) >= target_negative:
                break
    
    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)
    
    print(f"Generated {len(all_samples)} samples ({len(positive_samples)} pos, {len(negative_samples)} neg)")
    return all_samples


class _SampleGenerator:
    """Minimal class for generating samples without GUI."""
    
    def __init__(self, video_paths: List[str]):
        self.video_paths = video_paths
    
    def generate_patch(self, sample, frame, force_crossing=False):
        """Generate a single patch with optional crossing-centered placement."""
        x, y, w, h = sample.crop_rect
        crop = frame[y:y+h, x:x+w]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Find intersections
        intersections = find_line_intersections(sample.lines)
        
        # Determine if we should center on a crossing
        center_on_crossing = force_crossing and len(intersections) > 0
        target_crossing = random.choice(intersections) if center_on_crossing else None
        
        buffer_factor = 3.0
        initial_size = int(128 * buffer_factor)
        
        # Patch placement
        if center_on_crossing:
            target_x, target_y = target_crossing
            patch_x = int(target_x - initial_size / 2)
            patch_y = int(target_y - initial_size / 2)
            patch_x = max(0, min(w - initial_size, patch_x))
            patch_y = max(0, min(h - initial_size, patch_y))
            patch_w, patch_h = min(initial_size, w), min(initial_size, h)
        else:
            if w < initial_size or h < initial_size:
                patch_x, patch_y = 0, 0
                patch_w, patch_h = w, h
            else:
                patch_x = random.randint(0, w - initial_size)
                patch_y = random.randint(0, h - initial_size)
                patch_w, patch_h = initial_size, initial_size
        
        large_patch = crop_rgb[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
        intersections_local = [(ix - patch_x, iy - patch_y) for ix, iy in intersections]
        
        edge_margin = 128 // 10
        max_augment_attempts = 20 if center_on_crossing else 10
        
        valid_crossings = []
        final_image = None
        
        for attempt in range(max_augment_attempts):
            # Random augmentation parameters
            zoom_factor = random.uniform(0.75, 2.0)
            rotation_angle = random.uniform(-180, 180)
            stretch_x = random.uniform(0.9, 1.1)
            stretch_y = random.uniform(0.9, 1.1)
            
            perspective_strength = 0.15
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
            valid_transform = True
            for sx, sy in source_corners_check:
                if sx < margin or sx > patch_w - margin or sy < margin or sy > patch_h - margin:
                    valid_transform = False
                    break
            
            if not valid_transform:
                continue
            
            # Check crossing validity
            valid_crossings = []
            if intersections_local:
                points = np.array(intersections_local, dtype=np.float32).reshape(-1, 1, 2)
                transformed_points = cv2.perspectiveTransform(points, H).reshape(-1, 2)
                
                for tx, ty in transformed_points:
                    fx = tx - crop_x_offset
                    fy = ty - crop_y_offset
                    
                    if fx >= edge_margin and fx <= 128 - edge_margin and fy >= edge_margin and fy <= 128 - edge_margin:
                        valid_crossings.append((fx, fy))
            
            # If force_crossing, need at least one valid crossing
            if center_on_crossing and len(valid_crossings) == 0:
                continue
            
            # Success
            transformed_img = cv2.warpPerspective(large_patch, H, (patch_w, patch_h),
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            final_image = transformed_img[crop_y_offset:crop_y_offset+128, crop_x_offset:crop_x_offset+128]
            break
        
        # Fallback
        if final_image is None:
            crop_x_offset = (patch_w - 128) // 2
            crop_y_offset = (patch_h - 128) // 2
            final_image = large_patch[crop_y_offset:crop_y_offset+128, crop_x_offset:crop_x_offset+128]
            if final_image.shape[0] < 128 or final_image.shape[1] < 128:
                final_image = cv2.resize(large_patch, (128, 128))
            valid_crossings = []
        
        # Create mask
        final_mask = np.zeros((128, 128), dtype=np.float32)
        for cx, cy in valid_crossings:
            render_gaussian_blob(final_mask, cx, cy, sigma=3.5)
        
        # Apply flips
        flip_horizontal = random.random() < 0.5
        flip_vertical = random.random() < 0.5
        
        if flip_horizontal:
            final_image = cv2.flip(final_image, 1)
            final_mask = cv2.flip(final_mask, 1)
        if flip_vertical:
            final_image = cv2.flip(final_image, 0)
            final_mask = cv2.flip(final_mask, 0)
        
        # Image augmentations
        final_image = final_image.astype(np.float32)
        
        brightness = random.uniform(-0.3, 0.3)
        final_image = final_image + brightness * 255
        
        contrast = random.uniform(0.7, 1.3)
        mean_val = np.mean(final_image)
        final_image = (final_image - mean_val) * contrast + mean_val
        
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
        final_mask = np.clip(final_mask, 0, 1)
        final_mask = (final_mask * 255).astype(np.uint8)
        
        return {
            'image': final_image,
            'mask': final_mask,
            'has_crossing': len(valid_crossings) > 0,
            'num_crossings': len(valid_crossings)
        }


def train_crossing_detector(args):
    """Main training function."""
    print("=" * 60)
    print("Crossing Detector Training")
    print("=" * 60)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Load annotations
    base_dir = os.path.dirname(args.annotations) if os.path.dirname(args.annotations) else "."
    db = AnnotationDatabase.load(args.annotations, base_dir)
    print(f"Loaded {len(db.samples)} samples from {args.annotations}")
    
    # Find videos
    video_paths = []
    if os.path.isdir(args.videos):
        for fname in os.listdir(args.videos):
            if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_paths.append(os.path.join(args.videos, fname))
    print(f"Found {len(video_paths)} videos in {args.videos}")
    
    # Generate training samples
    print(f"\nGenerating {args.train} training samples...")
    all_samples = generate_training_samples(db, video_paths, args.train, balance_ratio=0.5)
    
    # Split into train/val (90/10)
    val_size = max(1, len(all_samples) // 10)
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Create datasets and loaders
    train_dataset = CrossingDataset(train_samples)
    val_dataset = CrossingDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print("\nInitializing MobileUNet model...")
    model = MobileUNet(pretrained=True).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()  # Regression loss for grayscale heatmap
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save best model
        model_path = f"{args.model_name}_best.pth"
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            save_marker = " *"
        else:
            save_marker = ""
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.6f}{save_marker}")
    
    print("-" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Crossing Detection Data Generator & Trainer")
    parser.add_argument("--videos", default="videos", help="Path to videos directory")
    parser.add_argument("--annotations", default="line_annotations.json", help="Path to annotations JSON file")
    
    # Training arguments
    parser.add_argument("--train", type=int, default=None, help="Number of samples to generate for training (enables training mode)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model-name", default="crossing_detector", help="Model name (saves as {name}_best.pth)")
    
    args = parser.parse_args()
    
    if args.train is not None:
        # Training mode
        train_crossing_detector(args)
    else:
        # GUI mode
        app = DataGenViewer(videos_dir=args.videos, annotations_file=args.annotations)
        app.run()


if __name__ == "__main__":
    main()

