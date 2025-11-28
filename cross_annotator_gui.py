"""
Crossing Point Annotator GUI

Annotate crossing/intersection points on video frames.
Click to add points, right-click or Delete to remove selected point.

Usage: python cross_annotator_gui.py
       python cross_annotator_gui.py videos/
       python cross_annotator_gui.py --annotations cross_annotations.json
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
import random
import bisect
import argparse
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from scipy.ndimage import maximum_filter
import multiprocessing

# Global frame cache for worker processes
_worker_frame_cache = {}
_worker_video_paths = []


def render_gaussian_blob(mask: np.ndarray, cx: float, cy: float, sigma: float = 5.0):
    """
    Render a 2D Gaussian blob onto the mask at (cx, cy).
    Uses maximum to avoid over-saturation with overlapping blobs.
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


class MobileUNet(nn.Module):
    """Mobile-optimized U-Net with MobileNetV2 backbone for crossing detection."""
    
    def __init__(self, pretrained=False):
        super().__init__()
        
        if pretrained:
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
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


class CrossingDataset(Dataset):
    """Dataset for crossing detection training."""
    
    def __init__(self, samples):
        self.samples = samples
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image'].astype(np.float32) / 255.0
        mask = sample['mask'].astype(np.float32) / 255.0
        
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        mean = torch.from_numpy(self.mean).reshape(3, 1, 1)
        std = torch.from_numpy(self.std).reshape(3, 1, 1)
        image = (image - mean) / std
        
        if len(mask.shape) == 2:
            mask = torch.from_numpy(mask).unsqueeze(0)
        else:
            mask = torch.from_numpy(mask[:, :, 0]).unsqueeze(0)
        
        return image, mask


def generate_augmented_patch(crop_rgb, crossings, force_crossing=False, include_visualization=False):
    """
    Shared function for generating augmented 128x128 training patches.
    
    Args:
        crop_rgb: RGB image (numpy array)
        crossings: List of (x, y) crossing coordinates in crop_rgb space
        force_crossing: If True, retry until at least one crossing is in the patch
        include_visualization: If True, include extra data for visualization
        
    Returns:
        Dictionary with 'image', 'mask', 'has_crossing', 'num_crossings', etc.
    """
    h, w = crop_rgb.shape[:2]
    
    # Determine if we should center on a crossing
    center_on_crossing = force_crossing and len(crossings) > 0
    target_crossing = random.choice(crossings) if center_on_crossing else None
    
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
    crossings_local = [(cx - patch_x, cy - patch_y) for cx, cy in crossings]
    
    edge_margin = 12
    max_attempts = 20 if center_on_crossing else 10
    
    all_crossings_in_patch = []  # All crossings anywhere in 128x128
    interior_crossings = []       # Only crossings >12px from edge (get blobs)
    final_image = None
    H = None
    
    # Store augmentation params for visualization
    aug_params = {}
    
    for attempt in range(max_attempts):
        zoom = random.uniform(0.75, 2.0)
        angle = random.uniform(-180, 180)
        stretch_x, stretch_y = random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)
        
        # Perspective
        perspective_corners = [
            (random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15))
            for _ in range(4)
        ]
        
        cx, cy = patch_w / 2, patch_h / 2
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        sx, sy = zoom * stretch_x, zoom * stretch_y
        
        src = np.array([[0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]], dtype=np.float32)
        dst = []
        for i, (px, py) in enumerate(src):
            rx, ry = (px - cx) * cos_a - (py - cy) * sin_a, (px - cx) * sin_a + (py - cy) * cos_a
            fx = rx * sx + cx + perspective_corners[i][0] * patch_w
            fy = ry * sy + cy + perspective_corners[i][1] * patch_h
            dst.append([fx, fy])
        
        dst = np.array(dst, dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        
        crop_off = (patch_w - 128) // 2
        
        # Transform and classify crossings
        all_crossings_in_patch = []
        interior_crossings = []
        
        if crossings_local:
            pts = np.array(crossings_local, dtype=np.float32).reshape(-1, 1, 2)
            tpts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
            for tx, ty in tpts:
                fx, fy = tx - crop_off, ty - crop_off
                # Check if anywhere in the 128x128 patch
                if 0 <= fx <= 128 and 0 <= fy <= 128:
                    all_crossings_in_patch.append((fx, fy))
                    # Check if in interior (>12px from edge) - only these get blobs
                    if edge_margin <= fx <= 128 - edge_margin and edge_margin <= fy <= 128 - edge_margin:
                        interior_crossings.append((fx, fy))
        
        # For positive samples, require at least one crossing anywhere in patch
        if center_on_crossing and not all_crossings_in_patch:
            continue
        
        # Check transform validity
        H_inverse = np.linalg.inv(H)
        output_corners = np.array([
            [crop_off, crop_off],
            [crop_off + 128, crop_off],
            [crop_off + 128, crop_off + 128],
            [crop_off, crop_off + 128]
        ], dtype=np.float32).reshape(-1, 1, 2)
        source_corners_check = cv2.perspectiveTransform(output_corners, H_inverse).reshape(-1, 2)
        
        margin = 2
        valid_transform = True
        for scx, scy in source_corners_check:
            if scx < margin or scx > patch_w - margin or scy < margin or scy > patch_h - margin:
                valid_transform = False
                break
        
        if not valid_transform:
            continue
        
        transformed = cv2.warpPerspective(large_patch, H, (patch_w, patch_h), borderMode=cv2.BORDER_CONSTANT)
        final_image = transformed[crop_off:crop_off+128, crop_off:crop_off+128]
        
        aug_params = {
            'rotation': angle,
            'zoom': zoom,
            'stretch_x': stretch_x,
            'stretch_y': stretch_y,
            'perspective': perspective_corners
        }
        break
    
    if final_image is None:
        final_image = cv2.resize(large_patch, (128, 128))
        all_crossings_in_patch = []
        interior_crossings = []
    
    # Flips
    flip_h = random.random() < 0.5
    flip_v = random.random() < 0.5
    
    if flip_h:
        final_image = cv2.flip(final_image, 1)
        all_crossings_in_patch = [(128 - x, y) for x, y in all_crossings_in_patch]
        interior_crossings = [(128 - x, y) for x, y in interior_crossings]
    if flip_v:
        final_image = cv2.flip(final_image, 0)
        all_crossings_in_patch = [(x, 128 - y) for x, y in all_crossings_in_patch]
        interior_crossings = [(x, 128 - y) for x, y in interior_crossings]
    
    aug_params['flip_h'] = flip_h
    aug_params['flip_v'] = flip_v
    
    # Create mask - only for INTERIOR crossings (>12px from edge)
    mask = np.zeros((128, 128), dtype=np.float32)
    for ccx, ccy in interior_crossings:
        render_gaussian_blob(mask, ccx, ccy, sigma=5.0)
    
    # Photometric augmentations (image only)
    img = final_image.astype(np.float32)
    img = img + random.uniform(-0.3, 0.3) * 255
    img = (img - img.mean()) * random.uniform(0.7, 1.3) + img.mean()
    img = np.clip(img, 0, 255)
    img = 255.0 * np.power(img / 255.0, random.uniform(0.7, 1.5))
    if random.random() < 0.5:
        img = img + np.random.normal(0, random.uniform(0, 25), img.shape)
    
    final_image = np.clip(img, 0, 255).astype(np.uint8)
    mask_uint8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    
    result = {
        'image': final_image,
        'mask': mask_uint8,
        'has_crossing': len(all_crossings_in_patch) > 0,  # Valid if ANY crossing in patch
        'num_crossings': len(interior_crossings),  # Count only interior (with blobs)
        'all_crossings': all_crossings_in_patch,
        'interior_crossings': interior_crossings
    }
    
    if include_visualization:
        result.update({
            'patch_offset': (patch_x, patch_y, patch_w, patch_h),
            'homography': H,
            'step3_params': aug_params,
            'step5_final': final_image
        })
    
    return result


def _generate_one_sample(args):
    """Worker function for parallel sample generation."""
    sample_data, cache_key, force_crossing = args
    
    global _worker_frame_cache
    
    frame = _worker_frame_cache.get(cache_key)
    if frame is None:
        return None
    
    crop_rect = sample_data['crop_rect']
    crossings = sample_data['crossings']
    
    x, y, w, h = crop_rect
    crop = frame[y:y+h, x:x+w]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    return generate_augmented_patch(crop_rgb, crossings, force_crossing=force_crossing)


def generate_training_samples(db, video_paths, num_samples, balance_ratio=0.5):
    """Generate training samples with positive/negative balance."""
    global _worker_frame_cache
    
    samples_with_crossings = [s for s in db.samples if len(s.crossings) >= 1]
    samples_for_negatives = [s for s in db.samples]
    
    if not db.samples:
        raise ValueError("No samples found in database")
    
    print(f"Sample pool: {len(samples_with_crossings)} with crossings, {len(db.samples)} total")
    
    target_positive = int(num_samples * balance_ratio)
    target_negative = num_samples - target_positive
    
    # Pre-cache frames
    print("Pre-caching frames...")
    unique_frames = list(set((s.video_path, s.frame_idx) for s in db.samples))
    
    def load_frame(args):
        video_path, frame_idx = args
        original_path = video_path
        # Resolve relative path
        if not os.path.isabs(video_path):
            for vp in video_paths:
                if os.path.basename(vp) == os.path.basename(video_path):
                    video_path = vp
                    break
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Warning: Could not open video: {os.path.basename(video_path)}")
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return ((original_path, frame_idx), frame)
        else:
            print(f"  Warning: Could not read frame {frame_idx} from: {os.path.basename(video_path)}")
            return None
    
    frame_cache = {}
    num_workers = min(multiprocessing.cpu_count(), 8)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i, result in enumerate(executor.map(load_frame, unique_frames)):
            if result:
                frame_cache[result[0]] = result[1]
            if (i + 1) % 50 == 0:
                print(f"  Cached {i + 1}/{len(unique_frames)} frames")
    
    print(f"  Cached {len(frame_cache)} frames total")
    _worker_frame_cache = frame_cache
    
    # Create tasks
    def sample_to_dict(s):
        return {
            'video_path': s.video_path,
            'frame_idx': s.frame_idx,
            'crop_rect': s.crop_rect,
            'crossings': list(s.crossings)
        }
    
    oversample_positive = 2.5
    oversample_negative = 1.5
    positive_tasks = []
    negative_tasks = []
    
    for _ in range(int(target_positive * oversample_positive)):
        if samples_with_crossings:
            s = random.choice(samples_with_crossings)
            cache_key = (s.video_path, s.frame_idx)
            if cache_key in frame_cache:
                positive_tasks.append((sample_to_dict(s), cache_key, True))
    
    for _ in range(int(target_negative * oversample_negative)):
        s = random.choice(samples_for_negatives)
        cache_key = (s.video_path, s.frame_idx)
        if cache_key in frame_cache:
            negative_tasks.append((sample_to_dict(s), cache_key, False))
    
    # Generate samples
    print(f"Generating {num_samples} samples: {target_positive} positive, {target_negative} negative...")
    
    positive_samples = []
    negative_samples = []
    chunk_size = 2500
    
    print("Generating positive samples...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in range(0, len(positive_tasks), chunk_size):
            chunk = positive_tasks[i:i+chunk_size]
            results = list(executor.map(_generate_one_sample, chunk))
            for r in results:
                if r and r['has_crossing'] and len(positive_samples) < target_positive:
                    positive_samples.append(r)
            print(f"  Positive: {len(positive_samples)}/{target_positive}")
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
            print(f"  Negative: {len(negative_samples)}/{target_negative}")
            if len(negative_samples) >= target_negative:
                break
    
    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)
    
    print(f"Generated {len(all_samples)} samples ({len(positive_samples)} pos, {len(negative_samples)} neg)")
    return all_samples


def train_crossing_detector(args):
    """Train crossing detector from CLI."""
    print("\n" + "=" * 60)
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
    db = CrossingDatabase.load(args.annotations)
    print(f"Loaded {len(db.samples)} samples from {args.annotations}")
    
    # Find videos
    video_paths = find_videos([args.videos] if os.path.exists(args.videos) else [])
    print(f"Found {len(video_paths)} videos")
    
    # Generate training samples
    print(f"\nGenerating {args.train} training samples...")
    all_samples = generate_training_samples(db, video_paths, args.train, balance_ratio=0.5)
    
    # Split train/val (90/10)
    val_size = max(1, len(all_samples) // 10)
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Create datasets and loaders
    train_dataset = CrossingDataset(train_samples)
    val_dataset = CrossingDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Create model
    print("\nInitializing MobileUNet model...")
    model = MobileUNet(pretrained=True).to(device)
    
    # Weighted MSE loss
    def weighted_mse_loss(pred, target, pos_weight=20.0):
        weights = torch.ones_like(target)
        weights[target > 0.1] = pos_weight
        loss = weights * (pred - target) ** 2
        return loss.mean()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = weighted_mse_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = weighted_mse_loss(outputs, masks)
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


@dataclass
class Sample:
    """Represents a single sample with crossing points."""
    video_path: str
    frame_idx: int
    crop_rect: Tuple[int, int, int, int]
    crossings: List[Tuple[float, float]] = field(default_factory=list)
    num_lines: int = 0  # Original line count (for reference)
    
    def add_crossing(self, x: float, y: float):
        self.crossings.append((x, y))
    
    def remove_crossing(self, idx: int):
        if 0 <= idx < len(self.crossings):
            del self.crossings[idx]


@dataclass
class CrossingDatabase:
    """Database of crossing annotations."""
    samples: List[Sample] = field(default_factory=list)
    
    def find_sample(self, video_path: str, frame_idx: int, 
                    crop_rect: Tuple[int, int, int, int]) -> Sample:
        """Find existing sample or create new one."""
        for sample in self.samples:
            if (sample.video_path == video_path and 
                sample.frame_idx == frame_idx and 
                sample.crop_rect == crop_rect):
                return sample
        
        # Create new sample
        new_sample = Sample(video_path, frame_idx, crop_rect)
        self.samples.append(new_sample)
        return new_sample
    
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
                    "crossings": [list(c) for c in sample.crossings],
                    "num_lines": sample.num_lines
                }
                for sample in self.samples
            ],
            "stats": {
                "total_samples": len(self.samples),
                "samples_with_crossings": sum(1 for s in self.samples if s.crossings),
                "total_crossings": sum(len(s.crossings) for s in self.samples)
            }
        }
    
    def save(self, filepath: str):
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CrossingDatabase':
        """Load from JSON file."""
        if not os.path.exists(filepath):
            return cls()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        db = cls()
        for sample_data in data.get("samples", []):
            sample = Sample(
                video_path=sample_data["video_path"],
                frame_idx=sample_data["frame_idx"],
                crop_rect=tuple(sample_data["crop_rect"]),
                num_lines=sample_data.get("num_lines", 0)
            )
            for crossing in sample_data.get("crossings", []):
                sample.add_crossing(crossing[0], crossing[1])
            db.samples.append(sample)
        
        return db


def get_video_props(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


class CrossingAnnotator:
    def __init__(self, root, video_paths, annotations_file="cross_annotations.json", patch_size=400, 
                 model_path="cross_net_v1_best.pth"):
        self.root = root
        self.video_paths = [os.path.abspath(p) for p in video_paths]
        self.patch_size = patch_size
        self.annotations_file = annotations_file
        self.model_path = model_path
        
        # PyTorch setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        self.model = None
        
        # Pre-calculate frame counts
        self.video_frame_counts = {}
        print("Scanning videos...")
        
        max_workers = min(8, len(self.video_paths))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(get_video_props, path): path for path in self.video_paths}
            
            for future in as_completed(futures):
                path = futures[future]
                try:
                    count = future.result()
                    if count > 0:
                        self.video_frame_counts[path] = count
                        print(f"  {os.path.basename(path)}: {count} frames")
                except Exception as e:
                    print(f"  Error scanning {os.path.basename(path)}: {e}")
        
        # Setup cumulative distribution for proportional sampling
        self.cumulative_frames = []
        self.active_video_paths = []
        self.total_combined_frames = 0
        current_total = 0
        
        for path in self.video_paths:
            if path in self.video_frame_counts:
                count = self.video_frame_counts[path]
                current_total += count
                self.cumulative_frames.append(current_total)
                self.active_video_paths.append(path)
        
        self.total_combined_frames = current_total
        print(f"Total frames: {self.total_combined_frames}")
        
        # Load annotation database
        self.db = CrossingDatabase.load(annotations_file)
        print(f"Loaded {len(self.db.samples)} samples from {annotations_file}")
        
        # State
        self.current_patch_info = None
        self.history = []
        self.history_idx = -1
        
        self.photo_image = None
        self.crossings = []  # List of (x, y) in image coords
        self.selected_crossing_idx = None
        self.dragging_idx = None  # Index of crossing being dragged
        
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.display_scale = 1.0
        
        # Undo stack
        self.undo_stack = []
        
        # Data generation state
        self.generated_patches = []
        self.show_gen_mask = False
        self.gen_photo_image = None
        
        # Inference state
        self.inference_samples = []
        self.inference_photo_image = None
        self.show_inference_prediction = False
        self.inference_running = False
        self.inference_selected_indices = set()
        
        # UI Setup
        self.root.title("Crossing Point Annotator")
        self._build_ui()
        
        # Bindings
        self.root.bind("<Configure>", self.on_resize)
        self.root.bind("<a>", lambda e: self.prev_patch())
        self.root.bind("<d>", lambda e: self.next_patch())
        self.root.bind("<Left>", lambda e: self.prev_patch())
        self.root.bind("<Right>", lambda e: self.next_patch())
        self.root.bind("<Delete>", lambda e: self.delete_selected_crossing())
        self.root.bind("<Control-z>", lambda e: self.undo_last_action())
        self.root.bind("<r>", lambda e: self.on_r_key())
        self.root.bind("<m>", lambda e: self.on_m_key())
        self.root.bind("<g>", lambda e: self.on_g_key())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        
        # Load history from database
        self._load_history_from_db()
        
        # Initial patch
        if self.total_combined_frames > 0:
            if len(self.history) > 0:
                self.history_idx = len(self.history) - 1
                self.current_patch_info = self.history[self.history_idx]
                self._load_current_crossings()
                self.display_current_patch()
            else:
                self.next_patch()
        else:
            messagebox.showerror("Error", "No valid frames found in the provided videos.")
        
        # Auto-load model for inference
        self.root.after(100, self._auto_load_model)
    
    def _auto_load_model(self):
        """Automatically load model on startup (silent, no messagebox)."""
        try:
            if os.path.exists(self.model_path):
                if self.model is None:
                    self.model = MobileUNet(pretrained=False).to(self.device)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                model_name = os.path.basename(self.model_path)
                self.lbl_inference_model.config(text=f"Model: {model_name} ✓", foreground="green")
                print(f"Auto-loaded model: {model_name}")
        except Exception as e:
            print(f"Auto-load model failed: {e}")
    
    def _load_sample_image(self, patch_info):
        """Load image for a patch_info dict (lazy loading)."""
        if patch_info.get("image") is not None:
            return True  # Already loaded
        
        cap = cv2.VideoCapture(patch_info["video_path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, patch_info["frame_idx"])
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            x, y, w, h = patch_info["crop_rect"]
            patch = frame[y:y+h, x:x+w]
            patch_info["image"] = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            return True
        return False
    
    def _load_history_from_db(self):
        """Load history metadata from saved samples (lazy - no images yet)."""
        if not self.db.samples:
            return
        
        print(f"Loading {len(self.db.samples)} sample metadata (lazy)...")
        
        for sample in self.db.samples:
            # Just store metadata, image will be loaded on-demand
            self.history.append({
                "video_path": sample.video_path,
                "frame_idx": sample.frame_idx,
                "crop_rect": sample.crop_rect,
                "image": None  # Loaded lazily
            })
        
        print(f"Loaded {len(self.history)} sample entries")
    
    def _build_ui(self):
        """Build the UI with tabs."""
        # Create tab control
        self.tab_control = ttk.Notebook(self.root)
        self.tab_control.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.labelling_tab = ttk.Frame(self.tab_control)
        self.data_gen_tab = ttk.Frame(self.tab_control)
        self.inference_tab = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.labelling_tab, text="Labelling")
        self.tab_control.add(self.data_gen_tab, text="Data Generation")
        self.tab_control.add(self.inference_tab, text="Inference")
        
        # Build each tab
        self._build_labelling_tab()
        self._build_data_generation_tab()
        self._build_inference_tab()
    
    def _build_labelling_tab(self):
        """Build the labelling tab UI."""
        main_frame = ttk.Frame(self.labelling_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas
        self.canvas = tk.Canvas(main_frame, bg="#222222", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Sidebar
        sidebar = ttk.Frame(main_frame, width=300, padding=10)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        # Info Panel
        info_frame = ttk.LabelFrame(sidebar, text="Current Sample", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.lbl_video = ttk.Label(info_frame, text="Video: -", wraplength=260)
        self.lbl_video.pack(anchor="w", pady=2)
        
        self.lbl_frame = ttk.Label(info_frame, text="Frame: -")
        self.lbl_frame.pack(anchor="w", pady=2)
        
        self.lbl_coords = ttk.Label(info_frame, text="Crop: -")
        self.lbl_coords.pack(anchor="w", pady=2)
        
        # Statistics Panel
        stats_frame = ttk.LabelFrame(sidebar, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.lbl_total_samples = ttk.Label(stats_frame, text="Total samples: 0")
        self.lbl_total_samples.pack(anchor="w", pady=2)
        
        self.lbl_labeled_samples = ttk.Label(stats_frame, text="Samples with crossings: 0")
        self.lbl_labeled_samples.pack(anchor="w", pady=2)
        
        self.lbl_total_crossings = ttk.Label(stats_frame, text="Total crossings: 0")
        self.lbl_total_crossings.pack(anchor="w", pady=2)
        
        # Crossings List Panel
        crossings_frame = ttk.LabelFrame(sidebar, text="Crossings", padding=10)
        crossings_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        list_container = ttk.Frame(crossings_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.crossings_listbox = tk.Listbox(list_container, yscrollcommand=scrollbar.set, height=10)
        self.crossings_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.crossings_listbox.yview)
        self.crossings_listbox.bind('<<ListboxSelect>>', self.on_crossing_select)
        
        self.btn_delete = ttk.Button(crossings_frame, text="Delete Crossing", command=self.delete_selected_crossing)
        self.btn_delete.pack(pady=5)
        
        # Navigation Panel
        nav_frame = ttk.LabelFrame(sidebar, text="Navigation", padding=10)
        nav_frame.pack(fill=tk.X, pady=(0, 20))
        
        btn_prev = ttk.Button(nav_frame, text="<< Previous (A)", command=self.prev_patch)
        btn_prev.pack(fill=tk.X, pady=5)
        
        btn_next = ttk.Button(nav_frame, text="Next Random (D) >>", command=self.next_patch)
        btn_next.pack(fill=tk.X, pady=5)
        
        btn_resample = ttk.Button(nav_frame, text="Resample (R)", command=self.resample_current_patch)
        btn_resample.pack(fill=tk.X, pady=5)
        
        # Sample management
        sample_frame = ttk.LabelFrame(sidebar, text="Sample Management", padding=10)
        sample_frame.pack(fill=tk.X, pady=(0, 20))
        
        btn_delete_sample = ttk.Button(sample_frame, text="Delete Sample", command=self.delete_current_sample)
        btn_delete_sample.pack(fill=tk.X, pady=5)
        
        # Help
        help_frame = ttk.LabelFrame(sidebar, text="Controls", padding=10)
        help_frame.pack(fill=tk.X)
        
        ttk.Label(help_frame, text="Left click: Add crossing").pack(anchor="w")
        ttk.Label(help_frame, text="Right click: Delete nearest").pack(anchor="w")
        ttk.Label(help_frame, text="Delete: Remove selected").pack(anchor="w")
        ttk.Label(help_frame, text="Ctrl+Z: Undo").pack(anchor="w")
    
    def _build_data_generation_tab(self):
        """Build the UI for the data generation tab."""
        main_frame = ttk.Frame(self.data_gen_tab)
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
        
        btn_generate = ttk.Button(gen_frame, text="Generate 9 Random Patches (R)", command=self.generate_training_patches)
        btn_generate.pack(fill=tk.X, pady=5)
        
        self.btn_toggle_gen_view = ttk.Button(gen_frame, text="Show: Images (M)", command=self.toggle_generation_view)
        self.btn_toggle_gen_view.pack(fill=tk.X, pady=5)
        
        # Generation info
        info_frame = ttk.LabelFrame(sidebar, text="Grid Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_gen_count = ttk.Label(info_frame, text="Patches: 0")
        self.lbl_gen_count.pack(anchor="w", pady=2)
        
        samples_with_crossings = sum(1 for s in self.db.samples if s.crossings)
        self.lbl_gen_samples = ttk.Label(info_frame, text=f"Labeled samples: {samples_with_crossings}")
        self.lbl_gen_samples.pack(anchor="w", pady=2)
        
        self.lbl_gen_videos = ttk.Label(info_frame, text=f"Videos: {len(self.video_paths)}")
        self.lbl_gen_videos.pack(anchor="w", pady=2)
        
        # Augmentation info
        aug_frame = ttk.LabelFrame(sidebar, text="Augmentations", padding=10)
        aug_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(aug_frame, text="Geometric:").pack(anchor="w")
        ttk.Label(aug_frame, text="  • Zoom: 0.75-2.0x").pack(anchor="w")
        ttk.Label(aug_frame, text="  • Rotation: -180° to 180°").pack(anchor="w")
        ttk.Label(aug_frame, text="  • Perspective warp").pack(anchor="w")
        ttk.Label(aug_frame, text="  • H/V flip: 50%").pack(anchor="w")
        ttk.Label(aug_frame, text="Photometric:").pack(anchor="w", pady=(5, 0))
        ttk.Label(aug_frame, text="  • Brightness: ±30%").pack(anchor="w")
        ttk.Label(aug_frame, text="  • Contrast: 0.7-1.3x").pack(anchor="w")
        ttk.Label(aug_frame, text="  • Gamma: 0.7-1.5").pack(anchor="w")
        ttk.Label(aug_frame, text="  • Noise: σ 0-25").pack(anchor="w")
        
        # Keyboard shortcuts
        help_frame = ttk.LabelFrame(sidebar, text="Keyboard Shortcuts", padding=10)
        help_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(help_frame, text="R: Generate new patches").pack(anchor="w")
        ttk.Label(help_frame, text="M: Toggle image/mask view").pack(anchor="w")
    
    def get_random_frame_location(self):
        """Select a video and frame proportionally."""
        if self.total_combined_frames == 0:
            return None, None
        
        global_idx = random.randint(0, self.total_combined_frames - 1)
        video_idx = bisect.bisect_left(self.cumulative_frames, global_idx)
        
        if video_idx >= len(self.active_video_paths):
            video_idx = len(self.active_video_paths) - 1
        
        video_path = self.active_video_paths[video_idx]
        prev_cumulative = self.cumulative_frames[video_idx - 1] if video_idx > 0 else 0
        frame_idx = global_idx - prev_cumulative
        
        return video_path, frame_idx
    
    def generate_new_patch(self):
        """Generate a new random patch."""
        if self.total_combined_frames == 0:
            return None
        
        for _ in range(10):
            video_path, frame_idx = self.get_random_frame_location()
            if not video_path:
                continue
            
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                continue
            
            h, w = frame.shape[:2]
            
            if h < self.patch_size or w < self.patch_size:
                x, y = 0, 0
                cw, ch = min(w, self.patch_size), min(h, self.patch_size)
            else:
                x = random.randint(0, w - self.patch_size)
                y = random.randint(0, h - self.patch_size)
                cw, ch = self.patch_size, self.patch_size
            
            patch = frame[y:y+ch, x:x+cw]
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            
            return {
                "video_path": video_path,
                "frame_idx": frame_idx,
                "crop_rect": (x, y, cw, ch),
                "image": patch_rgb
            }
        
        return None
    
    def next_patch(self):
        """Go to next patch."""
        self._save_current_crossings()
        
        if self.history_idx < len(self.history) - 1:
            self.history_idx += 1
            self.current_patch_info = self.history[self.history_idx]
        else:
            new_info = self.generate_new_patch()
            if new_info:
                self.history.append(new_info)
                self.history_idx = len(self.history) - 1
                self.current_patch_info = new_info
        
        self._load_current_crossings()
        self.display_current_patch()
    
    def prev_patch(self):
        """Go to previous patch."""
        self._save_current_crossings()
        
        if self.history_idx > 0:
            self.history_idx -= 1
            self.current_patch_info = self.history[self.history_idx]
            self._load_current_crossings()
            self.display_current_patch()
    
    def _save_current_crossings(self):
        """Save current crossings to database."""
        if not self.current_patch_info:
            return
        
        info = self.current_patch_info
        sample = self.db.find_sample(
            info['video_path'],
            info['frame_idx'],
            info['crop_rect']
        )
        
        sample.crossings.clear()
        for x, y in self.crossings:
            sample.add_crossing(x, y)
    
    def _load_current_crossings(self):
        """Load crossings from database for current patch."""
        self.crossings = []
        self.selected_crossing_idx = None
        self.undo_stack = []
        
        if not self.current_patch_info:
            return
        
        info = self.current_patch_info
        sample = self.db.find_sample(
            info['video_path'],
            info['frame_idx'],
            info['crop_rect']
        )
        
        for x, y in sample.crossings:
            self.crossings.append((x, y))
    
    def display_current_patch(self):
        """Display current patch."""
        if not self.current_patch_info:
            return
        
        info = self.current_patch_info
        
        # Lazy load image if needed
        if info.get("image") is None:
            if not self._load_sample_image(info):
                self.lbl_video.config(text="Error loading image")
                return
        
        self.lbl_video.config(text=f"Video: {os.path.basename(info['video_path'])}")
        self.lbl_frame.config(text=f"Frame: {info['frame_idx']}")
        x, y, w, h = info['crop_rect']
        self.lbl_coords.config(text=f"Crop: x={x}, y={y} ({w}x{h})")
        
        self.update_statistics()
        self.update_crossings_list()
        self.draw_image()
    
    def draw_image(self):
        """Draw the image with crossing points."""
        if not self.current_patch_info:
            return
        
        img_arr = self.current_patch_info['image']
        img_h, img_w = img_arr.shape[:2]
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            self.root.after(100, self.draw_image)
            return
        
        # Scale to fit
        target_height = int(canvas_h * 0.90)
        scale = target_height / img_h
        
        interp = cv2.INTER_NEAREST if scale > 1.5 else cv2.INTER_LINEAR
        
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized = cv2.resize(img_arr, (new_w, new_h), interpolation=interp)
        
        self.photo_image = ImageTk.PhotoImage(Image.fromarray(resized))
        
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        
        self.canvas_offset_x = x_offset
        self.canvas_offset_y = y_offset
        self.display_scale = scale
        
        self.canvas.delete("all")
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.photo_image)
        
        # Draw crossings
        self.draw_crossings_on_canvas()
    
    def draw_crossings_on_canvas(self):
        """Draw crossing points on canvas."""
        for idx, (x, y) in enumerate(self.crossings):
            canvas_x, canvas_y = self.image_to_canvas_coords(x, y)
            
            is_selected = (self.selected_crossing_idx == idx)
            
            if is_selected:
                color = "lime"
                size = 8
            else:
                color = "red"
                size = 6
            
            # Draw crosshair
            self.canvas.create_line(canvas_x - size, canvas_y, canvas_x + size, canvas_y, 
                                   fill=color, width=2)
            self.canvas.create_line(canvas_x, canvas_y - size, canvas_x, canvas_y + size, 
                                   fill=color, width=2)
            
            # Draw circle
            self.canvas.create_oval(canvas_x - size, canvas_y - size,
                                   canvas_x + size, canvas_y + size,
                                   outline=color, width=2)
            
            # Draw index label
            self.canvas.create_text(canvas_x + size + 5, canvas_y - size - 5,
                                   text=str(idx + 1), fill=color, 
                                   font=("Arial", 10, "bold"), anchor="sw")
    
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to image coordinates."""
        img_x = (canvas_x - self.canvas_offset_x) / self.display_scale
        img_y = (canvas_y - self.canvas_offset_y) / self.display_scale
        return img_x, img_y
    
    def image_to_canvas_coords(self, img_x, img_y):
        """Convert image coordinates to canvas coordinates."""
        canvas_x = self.canvas_offset_x + img_x * self.display_scale
        canvas_y = self.canvas_offset_y + img_y * self.display_scale
        return canvas_x, canvas_y
    
    def on_canvas_click(self, event):
        """Handle left click - start dragging existing crossing or add new one."""
        if not self.current_patch_info:
            return
        
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        
        # Check if clicking near existing crossing (to start dragging)
        click_threshold = 15 / self.display_scale
        
        for idx, (x, y) in enumerate(self.crossings):
            dist = ((x - img_x)**2 + (y - img_y)**2)**0.5
            if dist < click_threshold:
                # Start dragging this crossing
                self._push_undo_state("Move crossing")
                self.dragging_idx = idx
                self.selected_crossing_idx = idx
                self.update_crossings_list()
                self.draw_image()
                return
        
        # Add new crossing and start dragging it
        self._push_undo_state("Add crossing")
        self.crossings.append((img_x, img_y))
        self.selected_crossing_idx = len(self.crossings) - 1
        self.dragging_idx = len(self.crossings) - 1  # Start dragging the new dot
        self.update_statistics()
        self.update_crossings_list()
        self.draw_image()
    
    def on_canvas_drag(self, event):
        """Handle mouse drag - move crossing if dragging."""
        if self.dragging_idx is None or not self.current_patch_info:
            return
        
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        
        # Clamp to image bounds
        if self.current_patch_info:
            img_arr = self.current_patch_info['image']
            img_h, img_w = img_arr.shape[:2]
            img_x = max(0, min(img_x, img_w))
            img_y = max(0, min(img_y, img_h))
        
        # Update crossing position
        self.crossings[self.dragging_idx] = (img_x, img_y)
        self.update_crossings_list()
        self.draw_image()
    
    def on_canvas_release(self, event):
        """Handle mouse release - finish dragging."""
        if self.dragging_idx is not None:
            # Save the new position
            self.save_annotations(show_message=False)
            self.update_statistics()
            self.dragging_idx = None
    
    def on_canvas_right_click(self, event):
        """Handle right click - delete nearest crossing."""
        if not self.current_patch_info or not self.crossings:
            return
        
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        
        # Find nearest crossing
        min_dist = float('inf')
        nearest_idx = -1
        
        for idx, (x, y) in enumerate(self.crossings):
            dist = ((x - img_x)**2 + (y - img_y)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        
        # Delete if within reasonable distance
        if nearest_idx >= 0 and min_dist < 50 / self.display_scale:
            self._push_undo_state("Delete crossing")
            del self.crossings[nearest_idx]
            self.selected_crossing_idx = None
            self.save_annotations(show_message=False)
            self.update_statistics()
            self.update_crossings_list()
            self.draw_image()
    
    def on_crossing_select(self, event):
        """Handle selection from listbox."""
        selection = self.crossings_listbox.curselection()
        if selection:
            self.selected_crossing_idx = selection[0]
            self.draw_image()
    
    def delete_selected_crossing(self):
        """Delete the selected crossing."""
        if self.selected_crossing_idx is not None and 0 <= self.selected_crossing_idx < len(self.crossings):
            self._push_undo_state("Delete crossing")
            del self.crossings[self.selected_crossing_idx]
            self.selected_crossing_idx = None
            self.save_annotations(show_message=False)
            self.update_statistics()
            self.update_crossings_list()
            self.draw_image()
    
    def _push_undo_state(self, action_name=""):
        """Save current state for undo."""
        state = {
            'crossings': list(self.crossings),
            'selected_idx': self.selected_crossing_idx,
            'action': action_name
        }
        self.undo_stack.append(state)
    
    def undo_last_action(self):
        """Undo the last action."""
        if not self.current_patch_info or not self.undo_stack:
            return
        
        state = self.undo_stack.pop()
        self.crossings = list(state['crossings'])
        self.selected_crossing_idx = state['selected_idx']
        
        self.save_annotations(show_message=False)
        self.update_statistics()
        self.update_crossings_list()
        self.draw_image()
    
    def update_statistics(self):
        """Update statistics labels."""
        total_samples = len(self.db.samples)
        labeled_samples = sum(1 for s in self.db.samples if s.crossings)
        total_crossings = sum(len(s.crossings) for s in self.db.samples)
        
        current_sample = self.history_idx + 1 if self.history_idx >= 0 else 0
        
        self.lbl_total_samples.config(text=f"Total samples: {current_sample} / {total_samples}")
        self.lbl_labeled_samples.config(text=f"Samples with crossings: {labeled_samples}")
        self.lbl_total_crossings.config(text=f"Total crossings: {total_crossings}")
    
    def update_crossings_list(self):
        """Update crossings listbox."""
        self.crossings_listbox.delete(0, tk.END)
        
        if not self.crossings:
            self.crossings_listbox.insert(tk.END, "No crossings")
        else:
            for i, (x, y) in enumerate(self.crossings):
                text = f"Crossing {i+1}: ({x:.1f}, {y:.1f})"
                self.crossings_listbox.insert(tk.END, text)
            
            if self.selected_crossing_idx is not None and self.selected_crossing_idx < len(self.crossings):
                self.crossings_listbox.selection_clear(0, tk.END)
                self.crossings_listbox.selection_set(self.selected_crossing_idx)
                self.crossings_listbox.see(self.selected_crossing_idx)
    
    def save_annotations(self, show_message=True):
        """Save annotations to file."""
        self._save_current_crossings()
        self.db.save(self.annotations_file)
        if show_message:
            print(f"Annotations saved to {self.annotations_file}")
    
    def delete_current_sample(self):
        """Delete the current sample."""
        if not self.current_patch_info or len(self.history) == 0:
            return
        
        result = messagebox.askyesno(
            "Delete Sample",
            "Delete this sample and all its crossings?",
            icon='warning'
        )
        if not result:
            return
        
        info = self.current_patch_info
        for i, sample in enumerate(self.db.samples):
            if (sample.video_path == info['video_path'] and
                sample.frame_idx == info['frame_idx'] and
                sample.crop_rect == info['crop_rect']):
                del self.db.samples[i]
                break
        
        if self.history_idx >= 0 and self.history_idx < len(self.history):
            del self.history[self.history_idx]
        
        if len(self.history) == 0:
            self.history_idx = -1
            self.current_patch_info = None
            self.next_patch()
        elif self.history_idx >= len(self.history):
            self.history_idx = len(self.history) - 1
            self.current_patch_info = self.history[self.history_idx]
            self._load_current_crossings()
            self.display_current_patch()
        else:
            self.current_patch_info = self.history[self.history_idx]
            self._load_current_crossings()
            self.display_current_patch()
        
        self.save_annotations(show_message=False)
    
    def resample_current_patch(self):
        """Replace current sample with a new random patch."""
        if self.total_combined_frames == 0:
            return
        
        if self.current_patch_info:
            old_info = self.current_patch_info
            for i, sample in enumerate(self.db.samples):
                if (sample.video_path == old_info['video_path'] and
                    sample.frame_idx == old_info['frame_idx'] and
                    sample.crop_rect == old_info['crop_rect']):
                    del self.db.samples[i]
                    break
        
        new_info = self.generate_new_patch()
        if not new_info:
            return
        
        if self.history_idx >= 0 and self.history_idx < len(self.history):
            self.history[self.history_idx] = new_info
        else:
            self.history.append(new_info)
            self.history_idx = len(self.history) - 1
        
        self.current_patch_info = new_info
        self.crossings = []
        self.selected_crossing_idx = None
        self.undo_stack = []
        
        self.save_annotations(show_message=False)
        self.display_current_patch()
    
    def on_resize(self, event):
        """Handle window resize."""
        if self.current_patch_info:
            self.draw_image()
    
    # ========== Data Generation Methods ==========
    
    def on_r_key(self):
        """Handle 'r' key press - resample in labelling tab, regenerate in data gen tab."""
        current_tab = self.tab_control.index(self.tab_control.select())
        if current_tab == 0:  # Labelling tab
            self.resample_current_patch()
        elif current_tab == 1:  # Data Generation tab
            self.generate_training_patches()
    
    def _generate_single_patch(self, sample, frame, include_visualization=False):
        """
        Generate a single augmented training patch + mask from a sample.
        Uses the shared generate_augmented_patch function.
        
        Args:
            sample: Sample object with video_path, frame_idx, crop_rect, crossings
            frame: The video frame (BGR format)
            include_visualization: If True, include extra data for visualization
            
        Returns:
            Dictionary with 'image' (128x128 RGB), 'mask' (128x128 grayscale)
        """
        # Extract the original crop
        x, y, w, h = sample.crop_rect
        crop = frame[y:y+h, x:x+w]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Use shared generation function
        result = generate_augmented_patch(
            crop_rgb, 
            list(sample.crossings), 
            force_crossing=False,
            include_visualization=include_visualization
        )
        
        # Add sample metadata for visualization
        if include_visualization:
            result.update({
                'source_video': os.path.basename(sample.video_path),
                'source_frame': sample.frame_idx,
                'source_crop': sample.crop_rect
            })
        
        return result
    
    def _generate_initial_training_patches(self):
        """Generate initial training patches automatically on app start."""
        labeled_samples = [s for s in self.db.samples if len(s.crossings) > 0]
        if not labeled_samples:
            return
        
        threading.Thread(target=self.generate_training_patches, daemon=True).start()
    
    def generate_training_patches(self):
        """Generate 9 random 128x128 patches from labeled samples for visualization."""
        self.root.after(0, lambda: self.lbl_gen_count.config(text="Generating..."))
        
        self.generated_patches = []
        new_patches = []
        
        labeled_samples = [s for s in self.db.samples if len(s.crossings) > 0]
        
        if not labeled_samples:
            self.root.after(0, lambda: messagebox.showwarning("No Labeled Data", 
                "No labeled samples found. Please label some crossings first."))
            return
        
        print(f"Generating 9 patches from {len(labeled_samples)} labeled samples...")
        
        # Pre-load frames
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
        """Display 3x3 grid showing: Full region with green highlight | Final 128x128 patch."""
        if not self.generated_patches:
            return
        
        self.lbl_gen_count.config(text=f"Patches: {len(self.generated_patches)}")
        
        if self.show_gen_mask:
            self.btn_toggle_gen_view.config(text="Show: Masks (M)")
        else:
            self.btn_toggle_gen_view.config(text="Show: Images (M)")
        
        # Layout: 3x3 grid
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
                
                # LEFT: Source region
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
                
                left_img = cv2.resize(left_img, (region_size, region_size), interpolation=cv2.INTER_LINEAR)
                left_img_display = left_img.copy()
                
                # Draw green polygon for source region
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
                        
                        cv2.polylines(left_img_display, [display_corners], isClosed=True, 
                                      color=(0, 255, 0), thickness=2)
                        
                        # Draw flip indicators
                        step3_params = patch.get("step3_params", {})
                        flip_h = step3_params.get("flip_h", False)
                        flip_v = step3_params.get("flip_v", False)
                        
                        if flip_h or flip_v:
                            cx = int(np.mean(display_corners[:, 0]))
                            cy = int(np.mean(display_corners[:, 1]))
                            
                            if flip_h:
                                # Draw horizontal double arrow (↔)
                                cv2.arrowedLine(left_img_display, (cx - 15, cy - 10), (cx + 15, cy - 10), 
                                              (255, 255, 0), 2, tipLength=0.3)
                                cv2.arrowedLine(left_img_display, (cx + 15, cy - 10), (cx - 15, cy - 10), 
                                              (255, 255, 0), 2, tipLength=0.3)
                            
                            if flip_v:
                                # Draw vertical double arrow (↕)
                                y_off = 10 if flip_h else 0
                                cv2.arrowedLine(left_img_display, (cx, cy - 15 + y_off), (cx, cy + 15 + y_off), 
                                              (255, 0, 255), 2, tipLength=0.3)
                                cv2.arrowedLine(left_img_display, (cx, cy + 15 + y_off), (cx, cy - 15 + y_off), 
                                              (255, 0, 255), 2, tipLength=0.3)
                
                grid_img[cell_y:cell_y+region_size, cell_x:cell_x+region_size] = left_img_display
                
                # RIGHT: Final 128x128 patch
                if self.show_gen_mask:
                    right_img = patch.get("mask")
                    if right_img is None:
                        right_img = np.full((patch_size, patch_size), 64, dtype=np.uint8)
                    if len(right_img.shape) == 2:
                        right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2RGB)
                else:
                    right_img = patch.get("image")
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
        
        scale_w = canvas_w * 0.9 / img_w
        scale_h = canvas_h * 0.9 / img_h
        scale = min(scale_w, scale_h)
        
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        if scale > 1.5:
            interp = cv2.INTER_NEAREST
        else:
            interp = cv2.INTER_LINEAR
        
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
    
    def on_m_key(self):
        """Handle 'm' key - toggle masks in data gen or inference tab."""
        current_tab = self.tab_control.index(self.tab_control.select())
        if current_tab == 1:  # Data Generation tab
            self.toggle_generation_view()
        elif current_tab == 2:  # Inference tab
            self.toggle_inference_view()
    
    # ========== Inference Tab ==========
    
    def _build_inference_tab(self):
        """Build the UI for the inference tab."""
        main_frame = ttk.Frame(self.inference_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas Area (Left)
        self.inference_canvas = tk.Canvas(main_frame, bg="#222222", highlightthickness=0)
        self.inference_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bindings for selection
        self.inference_canvas.bind("<Button-1>", self.on_inference_click)
        self.inference_canvas.bind("<Button-3>", self.on_inference_right_click)
        
        # Context Menu
        self.inference_context_menu = tk.Menu(self.root, tearoff=0)
        self.inference_context_menu.add_command(label="Add selected to Labeling Queue", 
                                                command=self.add_selected_to_labeling)
        
        # Sidebar (Right)
        sidebar = ttk.Frame(main_frame, width=300, padding=10)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        # Model controls
        model_frame = ttk.LabelFrame(sidebar, text="Model", padding=10)
        model_frame.pack(fill=tk.X, pady=(10, 10))
        
        self.lbl_inference_model = ttk.Label(model_frame, text="Model: Not loaded")
        self.lbl_inference_model.pack(anchor="w", pady=5)
        
        btn_load_model = ttk.Button(model_frame, text="Load Model", command=self.load_model_for_inference)
        btn_load_model.pack(fill=tk.X, pady=5)
        
        # Sample controls
        sample_frame = ttk.LabelFrame(sidebar, text="Inference Settings", padding=10)
        sample_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Mode selection
        ttk.Label(sample_frame, text="Inference Mode:").pack(anchor="w", pady=2)
        self.inference_mode_var = tk.StringVar(value="Random Regions")
        mode_combo = ttk.Combobox(sample_frame, textvariable=self.inference_mode_var, 
                                  values=["Random Regions", "Full Frame"], state="readonly")
        mode_combo.pack(fill=tk.X, pady=5)
        
        # Even video sampling
        self.even_video_sampling_var = tk.BooleanVar(value=False)
        even_sampling_check = ttk.Checkbutton(sample_frame, text="Sample evenly across videos",
                                               variable=self.even_video_sampling_var)
        even_sampling_check.pack(anchor="w", pady=2)
        
        # Number of samples
        ttk.Label(sample_frame, text="Number of frames:").pack(anchor="w", pady=2)
        self.inference_samples_var = tk.IntVar(value=25)
        samples_spinbox = ttk.Spinbox(sample_frame, from_=1, to=100, increment=1, 
                                       textvariable=self.inference_samples_var, width=10)
        samples_spinbox.pack(anchor="w", pady=5)
        
        btn_generate_inference = ttk.Button(sample_frame, text="Generate Predictions (G)", 
                                           command=self.generate_inference_samples)
        btn_generate_inference.pack(fill=tk.X, pady=5)
        
        # View controls
        view_frame = ttk.LabelFrame(sidebar, text="View", padding=10)
        view_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_toggle_inference = ttk.Button(view_frame, text="Show: Image", 
                                               command=self.toggle_inference_view)
        self.btn_toggle_inference.pack(fill=tk.X, pady=5)
        
        # Show detected crossings toggle
        self.show_detected_crossings_var = tk.BooleanVar(value=False)
        show_crossings_check = ttk.Checkbutton(view_frame, text="Show detected crossings", 
                                               variable=self.show_detected_crossings_var,
                                               command=self.display_inference_sample)
        show_crossings_check.pack(fill=tk.X, pady=5)
        
        # Peak Detection Settings
        peak_frame = ttk.LabelFrame(sidebar, text="Peak Detection", padding=10)
        peak_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(peak_frame, text="Threshold:").pack(anchor="w")
        self.peak_threshold_var = tk.DoubleVar(value=0.3)
        threshold_spinbox = ttk.Spinbox(peak_frame, from_=0.1, to=0.9, increment=0.05, 
                                         textvariable=self.peak_threshold_var, width=10)
        threshold_spinbox.pack(anchor="w", pady=2)
        
        ttk.Label(peak_frame, text="Min Distance:").pack(anchor="w")
        self.peak_distance_var = tk.IntVar(value=10)
        distance_spinbox = ttk.Spinbox(peak_frame, from_=3, to=50, increment=1, 
                                        textvariable=self.peak_distance_var, width=10)
        distance_spinbox.pack(anchor="w", pady=2)
        
        btn_redetect = ttk.Button(peak_frame, text="Re-detect Peaks", command=self.redetect_peaks)
        btn_redetect.pack(fill=tk.X, pady=5)
        
        # Info
        info_frame = ttk.LabelFrame(sidebar, text="Grid Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_inference_idx = ttk.Label(info_frame, text="Samples: 0")
        self.lbl_inference_idx.pack(anchor="w", pady=5)
        
        # Keyboard shortcut help
        help_frame = ttk.LabelFrame(sidebar, text="Shortcuts", padding=10)
        help_frame.pack(fill=tk.X)
        
        ttk.Label(help_frame, text="G: Generate predictions").pack(anchor="w")
        ttk.Label(help_frame, text="Click: Select sample").pack(anchor="w")
        ttk.Label(help_frame, text="Right-click: Context menu").pack(anchor="w")
    
    def load_model_for_inference(self):
        """Load a trained model for inference."""
        try:
            if self.model is None:
                self.model = MobileUNet(pretrained=False).to(self.device)
            
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            model_name = os.path.basename(self.model_path)
            self.lbl_inference_model.config(text=f"Model: {model_name} ✓", foreground="green")
            messagebox.showinfo("Success", f"Model '{model_name}' loaded successfully!")
        except FileNotFoundError:
            messagebox.showerror("Error", f"Model file '{self.model_path}' not found.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def generate_inference_samples(self):
        """Generate predictions."""
        if self.model is None:
            messagebox.showwarning("No Model", "Please load a model first.")
            return
        
        if self.inference_running:
            return
        
        self.inference_running = True
        self.lbl_inference_idx.config(text="Generating predictions...")
        
        thread = threading.Thread(target=self._generate_inference_thread, daemon=True)
        thread.start()
    
    def _generate_inference_thread(self):
        """Background thread for generating inference samples with parallel loading and batch inference."""
        try:
            mode = self.inference_mode_var.get()
            num_samples = self.inference_samples_var.get()
            even_sampling = self.even_video_sampling_var.get()
            
            self.model.eval()
            
            # Normalization tensors
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).reshape(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).reshape(1, 3, 1, 1)
            
            # Step 1: Collect frame locations
            self.root.after(0, lambda: self.lbl_inference_idx.config(text="Selecting frames..."))
            frame_requests = []
            for _ in range(num_samples):
                video_path, frame_idx = self.get_random_frame_location_inference(even_sampling)
                if video_path:
                    frame_requests.append((video_path, frame_idx))
            
            # Step 2: Load frames in parallel
            self.root.after(0, lambda: self.lbl_inference_idx.config(text=f"Loading {len(frame_requests)} frames..."))
            
            def load_frame(args):
                video_path, frame_idx = args
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return (video_path, frame_idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                return None
            
            loaded_frames = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(load_frame, frame_requests))
                for r in results:
                    if r:
                        loaded_frames.append(r)
            
            self.root.after(0, lambda n=len(loaded_frames): self.lbl_inference_idx.config(
                text=f"Loaded {n} frames, running inference..."))
            
            # Step 3: Prepare data for batch inference
            samples_data = []  # Store metadata
            
            if mode == "Random Regions":
                patch_size = self.patch_size
                for video_path, frame_idx, frame_rgb in loaded_frames:
                    h, w = frame_rgb.shape[:2]
                    if h < patch_size or w < patch_size:
                        continue
                    
                    x = random.randint(0, w - patch_size)
                    y = random.randint(0, h - patch_size)
                    patch = frame_rgb[y:y+patch_size, x:x+patch_size]
                    
                    samples_data.append({
                        'frame': patch,
                        'video_path': video_path,
                        'frame_idx': frame_idx,
                        'location': (x, y),
                        'size': (patch_size, patch_size),
                        'type': 'patch'
                    })
            else:  # Full Frame
                for video_path, frame_idx, frame_rgb in loaded_frames:
                    h, w = frame_rgb.shape[:2]
                    if h < 128 or w < 128:
                        continue
                    
                    samples_data.append({
                        'frame': frame_rgb,
                        'video_path': video_path,
                        'frame_idx': frame_idx,
                        'size': (w, h),
                        'type': 'full'
                    })
            
            # Step 4: Batch inference
            samples = []
            batch_size = 8 if mode == "Random Regions" else 2  # Smaller batch for full frames (more memory)
            
            with torch.no_grad():
                for batch_start in range(0, len(samples_data), batch_size):
                    batch_end = min(batch_start + batch_size, len(samples_data))
                    batch = samples_data[batch_start:batch_end]
                    
                    self.root.after(0, lambda s=batch_start, e=len(samples_data): 
                        self.lbl_inference_idx.config(text=f"Inference: {s}/{e}..."))
                    
                    # Prepare batch tensors
                    batch_tensors = []
                    batch_info = []
                    
                    for item in batch:
                        img = item['frame']
                        h, w = img.shape[:2]
                        
                        pad_h = (32 - h % 32) % 32
                        pad_w = (32 - w % 32) % 32
                        
                        if pad_h > 0 or pad_w > 0:
                            img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                        else:
                            img_padded = img
                        
                        tensor = torch.from_numpy(img_padded.astype(np.float32) / 255.0).permute(2, 0, 1)
                        batch_tensors.append(tensor)
                        batch_info.append({'item': item, 'h': h, 'w': w})
                    
                    # Stack and run batch
                    batch_input = torch.stack(batch_tensors).to(self.device)
                    batch_input = (batch_input - mean) / std
                    predictions = self.model(batch_input)
                    
                    # Process results
                    for idx, (pred, info) in enumerate(zip(predictions, batch_info)):
                        item = info['item']
                        h, w = info['h'], info['w']
                        
                        pred_np = pred.squeeze().cpu().numpy()
                        pred_mask = pred_np[:h, :w]
                        detected_crossings = self._detect_peaks(pred_mask)
                        
                        result = {
                            'frame': item['frame'],
                            'prediction': pred_mask,
                            'detected_crossings': detected_crossings,
                            'source_video': os.path.basename(item['video_path']),
                            'source_frame': item['frame_idx'],
                            'size': item['size'],
                            'type': item['type']
                        }
                        if 'location' in item:
                            result['location'] = item['location']
                        
                        samples.append(result)
            
            self.inference_samples = samples
            self.inference_selected_indices = set()
            
            def finish_inference():
                self.inference_running = False
                if self.inference_samples:
                    self.show_inference_prediction = False
                    self.display_inference_sample()
                else:
                    self.lbl_inference_idx.config(text="No samples generated")
            
            self.root.after(0, finish_inference)
            
        except Exception as e:
            error_msg = str(e)
            print(f"Inference error: {error_msg}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda msg=error_msg: self.lbl_inference_idx.config(text=f"Error: {msg}"))
            self.inference_running = False
            self.inference_running = False
    
    def get_random_frame_location_inference(self, even_across_videos=False):
        """Select a video and frame for inference."""
        if self.total_combined_frames == 0:
            return None, None
        
        if even_across_videos and self.active_video_paths:
            video_path = random.choice(self.active_video_paths)
            frame_count = self.video_frame_counts.get(video_path, 1)
            frame_idx = random.randint(0, frame_count - 1)
            return video_path, frame_idx
        else:
            global_idx = random.randint(0, self.total_combined_frames - 1)
            video_idx = bisect.bisect_left(self.cumulative_frames, global_idx)
            
            if video_idx >= len(self.active_video_paths):
                video_idx = len(self.active_video_paths) - 1
            
            video_path = self.active_video_paths[video_idx]
            prev_cumulative = self.cumulative_frames[video_idx - 1] if video_idx > 0 else 0
            frame_idx = global_idx - prev_cumulative
            
            return video_path, frame_idx
    
    def _detect_peaks(self, heatmap):
        """Detect crossing peaks in heatmap using local maxima."""
        threshold = self.peak_threshold_var.get()
        min_distance = self.peak_distance_var.get()
        
        # Find local maxima
        local_max = maximum_filter(heatmap, size=min_distance * 2 + 1)
        peaks = (heatmap == local_max) & (heatmap > threshold)
        
        # Get coordinates
        coords = np.where(peaks)
        crossings = list(zip(coords[1], coords[0]))  # (x, y) format
        
        return crossings
    
    def redetect_peaks(self):
        """Re-detect peaks with current settings."""
        if not self.inference_samples:
            return
        
        for sample in self.inference_samples:
            pred_mask = sample['prediction']
            sample['detected_crossings'] = self._detect_peaks(pred_mask)
        
        self.display_inference_sample()
    
    def display_inference_sample(self):
        """Display predictions in grid layout."""
        if not self.inference_samples:
            return
        
        num_samples = len(self.inference_samples)
        mode = self.inference_mode_var.get()
        self.lbl_inference_idx.config(text=f"{mode}: {num_samples}")
        
        if self.show_inference_prediction:
            self.btn_toggle_inference.config(text="Show: Predictions")
        else:
            self.btn_toggle_inference.config(text="Show: Images")
        
        # Calculate grid
        if mode == "Random Regions":
            num_cols = max(1, int(math.ceil(math.sqrt(num_samples))))
            num_rows = max(1, int(math.ceil(num_samples / num_cols)))
        else:
            num_cols = min(3, num_samples)
            num_rows = (num_samples + num_cols - 1) // num_cols
        
        canvas_w = self.inference_canvas.winfo_width()
        canvas_h = self.inference_canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            self.root.after(100, self.display_inference_sample)
            return
        
        padding = 10
        available_w = canvas_w - padding * (num_cols + 1)
        available_h = canvas_h - padding * (num_rows + 1)
        cell_w = available_w // num_cols
        cell_h = available_h // num_rows
        
        self.inference_grid_geometry = {
            'num_cols': num_cols,
            'cell_w': cell_w,
            'cell_h': cell_h,
            'padding': padding
        }
        
        grid_img = np.full((canvas_h, canvas_w, 3), 32, dtype=np.uint8)
        
        for idx, sample in enumerate(self.inference_samples):
            row = idx // num_cols
            col = idx % num_cols
            
            x_start = padding + col * (cell_w + padding)
            y_start = padding + row * (cell_h + padding)
            
            # Selection highlight
            if idx in self.inference_selected_indices:
                cv2.rectangle(grid_img, 
                             (x_start - 4, y_start - 4), 
                             (x_start + cell_w + 4, y_start + cell_h + 4), 
                             (0, 255, 0), 4)
            
            # Choose image or prediction
            if self.show_inference_prediction:
                pred = sample['prediction']
                img_data = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
                img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
            else:
                img_data = sample['frame'].copy()
            
            # Draw detected crossings
            if self.show_detected_crossings_var.get():
                for cx, cy in sample.get('detected_crossings', []):
                    cv2.circle(img_data, (int(cx), int(cy)), 5, (0, 255, 0), 2)
                    cv2.drawMarker(img_data, (int(cx), int(cy)), (0, 255, 0), 
                                   cv2.MARKER_CROSS, 10, 2)
            
            # Resize to fit cell
            img_h, img_w = img_data.shape[:2]
            scale = min(cell_w / img_w, cell_h / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            
            if new_w > 0 and new_h > 0:
                resized = cv2.resize(img_data, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                x_off = (cell_w - new_w) // 2
                y_off = (cell_h - new_h) // 2
                
                grid_img[y_start + y_off:y_start + y_off + new_h,
                         x_start + x_off:x_start + x_off + new_w] = resized
        
        # Display
        img_pil = Image.fromarray(grid_img)
        self.inference_photo_image = ImageTk.PhotoImage(img_pil)
        
        self.inference_canvas.delete("all")
        self.inference_canvas.create_image(0, 0, anchor=tk.NW, image=self.inference_photo_image)
    
    def toggle_inference_view(self):
        """Toggle between image and prediction view."""
        self.show_inference_prediction = not self.show_inference_prediction
        self.display_inference_sample()
    
    def on_inference_click(self, event):
        """Handle click on inference canvas."""
        if not self.inference_samples or not hasattr(self, 'inference_grid_geometry'):
            return
        
        geom = self.inference_grid_geometry
        padding = geom['padding']
        cell_w = geom['cell_w']
        cell_h = geom['cell_h']
        num_cols = geom['num_cols']
        
        # Calculate which cell was clicked
        col = (event.x - padding) // (cell_w + padding)
        row = (event.y - padding) // (cell_h + padding)
        
        idx = row * num_cols + col
        
        if 0 <= idx < len(self.inference_samples):
            if idx in self.inference_selected_indices:
                self.inference_selected_indices.remove(idx)
            else:
                self.inference_selected_indices.add(idx)
            self.display_inference_sample()
    
    def on_inference_right_click(self, event):
        """Show context menu."""
        if self.inference_selected_indices:
            self.inference_context_menu.post(event.x_root, event.y_root)
    
    def add_selected_to_labeling(self):
        """Add selected inference samples to the labeling database."""
        if not self.inference_selected_indices:
            messagebox.showinfo("No Selection", "No samples selected. Left-click samples to select them first.")
            return
        
        num_selected = len(self.inference_selected_indices)
        
        # Confirmation dialog
        result = messagebox.askyesno(
            "Add to Labeling Queue",
            f"Add {num_selected} selected sample(s) to labeling queue?\n\nSelected indices: {sorted(self.inference_selected_indices)}",
            icon='question'
        )
        if not result:
            return
        
        count = 0
        for idx in self.inference_selected_indices:
            if idx < len(self.inference_samples):
                sample_data = self.inference_samples[idx]
                
                # Find full path from basename
                video_path = None
                for vp in self.video_paths:
                    if os.path.basename(vp) == sample_data['source_video']:
                        video_path = vp
                        break
                
                if not video_path:
                    continue
                
                frame_idx = sample_data['source_frame']
                w, h = sample_data['size']
                
                # Use stored location or default to 0,0
                if 'location' in sample_data:
                    x, y = sample_data['location']
                    crop_rect = (x, y, w, h)
                else:
                    crop_rect = (0, 0, w, h)
                
                # Add to database
                self.db.find_sample(video_path, frame_idx, crop_rect)
                
                # Also add to history so it shows up immediately in labelling tab
                frame_rgb = sample_data['frame']
                
                history_entry = {
                    "video_path": video_path,
                    "frame_idx": frame_idx,
                    "crop_rect": crop_rect,
                    "image": frame_rgb
                }
                self.history.append(history_entry)
                
                count += 1
        
        # Save database
        self.db.save(self.annotations_file)
        
        # Update statistics in labelling tab
        self.update_statistics()
        
        # Clear selection
        self.inference_selected_indices.clear()
        self.display_inference_sample()
        
        # Notify user
        messagebox.showinfo("Success", f"Added {count} of {num_selected} selected samples to labeling queue.\n\nGo to Labelling tab to see them (they're at the end).")
    
    def on_g_key(self):
        """Handle 'g' key - generate inference samples."""
        current_tab = self.tab_control.index(self.tab_control.select())
        if current_tab == 2:  # Inference tab
            self.generate_inference_samples()
    
    def on_close(self):
        """Handle window close."""
        self.save_annotations(show_message=False)
        self.root.destroy()


def find_videos(input_paths):
    """Find all video files in the given paths."""
    video_files = []
    supported_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    
    for path in input_paths:
        if os.path.isfile(path):
            if os.path.splitext(path)[1].lower() in supported_extensions:
                video_files.append(path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if os.path.splitext(file)[1].lower() in supported_extensions:
                        video_files.append(os.path.join(root, file))
    
    return sorted(list(set(video_files)))


def main():
    parser = argparse.ArgumentParser(description="Crossing point annotation and training tool")
    parser.add_argument("videos", nargs="*", help="Video files or directories")
    parser.add_argument("--annotations", "-a", default="cross_annotations.json", help="Annotations file")
    parser.add_argument("--patch-size", type=int, default=400, help="Patch size (default: 400)")
    parser.add_argument("--model", default="cross_net_v1_best.pth", help="Model file for inference")
    
    # Training arguments
    parser.add_argument("--train", type=int, default=None, 
                        help="Number of samples to generate for training (enables training mode)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model-name", default="cross_net", help="Model name (saves as {name}_best.pth)")
    parser.add_argument("--videos-dir", default="videos", help="Videos directory for training")
    
    args = parser.parse_args()
    
    # Training mode
    if args.train is not None:
        args.videos = args.videos_dir
        train_crossing_detector(args)
        return
    
    # GUI mode
    video_inputs = args.videos
    if not video_inputs:
        if os.path.exists("videos"):
            video_inputs = ["videos"]
        else:
            print("No video paths provided and 'videos' folder not found.")
            return
    
    video_paths = find_videos(video_inputs)
    
    if not video_paths:
        print("No video files found.")
        return
    
    print(f"Found {len(video_paths)} videos.")
    
    root = tk.Tk()
    root.state('zoomed')
    
    app = CrossingAnnotator(root, video_paths, annotations_file=args.annotations, 
                            patch_size=args.patch_size, model_path=args.model)
    
    root.mainloop()


if __name__ == "__main__":
    main()

