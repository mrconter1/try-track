import argparse
import json
import os
import random
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models


class TrainingDataBrowser:
    def __init__(self, root, annotations_path):
        self.root = root
        self.annotations_path = annotations_path
        self.positive_samples = []
        self.negative_samples = []
        self.hard_negative_sources = []
        self.current_images = []
        self.current_info = ""
        self.photo_images = []
        self.canvases = []
        self.display_size = 200
        self.grid_cols = 5
        self.grid_rows = 2
        
        self.root.title("Training Data Browser")
        self._load_annotations()
        self._build_ui()
        
    def _load_annotations(self):
        """Load and parse annotations.json to build positive and negative sample lists."""
        if not os.path.exists(self.annotations_path):
            messagebox.showerror("Error", f"Annotations file not found: {self.annotations_path}")
            return
        
        try:
            with open(self.annotations_path, "r") as f:
                data = json.load(f)
            
            for video in data.get("videos", []):
                video_path = video.get("video_path", "")
                if not os.path.exists(video_path):
                    print(f"[Warning] Video not found: {video_path}")
                    continue
                
                for frame_data in video.get("frames", []):
                    frame_idx = frame_data["frame_idx"]
                    crosses = frame_data.get("crosses", [])
                    negative_rects = frame_data.get("negative_rects", [])
                    
                    # Add positive samples (frames with crosses)
                    if crosses:
                        for cross in crosses:
                            self.positive_samples.append({
                                "video_path": video_path,
                                "frame_idx": frame_idx,
                                "cross": cross
                            })
                    
                    # Add negative samples (frames with negative rects)
                    if negative_rects:
                        for rect in negative_rects:
                            self.negative_samples.append({
                                "video_path": video_path,
                                "frame_idx": frame_idx,
                                "rect": rect
                            })

                    # Add sources for hard negatives (frames with both)
                    if crosses and negative_rects:
                        for cross in crosses:
                            for rect in negative_rects:
                                self.hard_negative_sources.append({
                                    "video_path": video_path,
                                    "frame_idx": frame_idx,
                                    "cross": cross,
                                    "rect": rect
                                })
            
            print(f"[Info] Loaded {len(self.positive_samples)} positive samples, {len(self.negative_samples)} negative samples, and {len(self.hard_negative_sources)} hard negative sources.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load annotations: {e}")
    
    def _build_ui(self):
        container = ttk.Frame(self.root, padding=20)
        container.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        try:
            self.root.state("zoomed")
        except tk.TclError:
            self.root.geometry("1200x700")
        
        self.root.bind("<p>", lambda e: self.generate_positive_batch())
        self.root.bind("<P>", lambda e: self.generate_positive_batch())
        self.root.bind("<n>", lambda e: self.generate_negative_batch())
        self.root.bind("<N>", lambda e: self.generate_negative_batch())
        self.root.bind("<h>", lambda e: self.generate_hard_negative_batch())
        self.root.bind("<H>", lambda e: self.generate_hard_negative_batch())
        
        # Info label
        self.info_label = ttk.Label(
            container,
            text="Press 'Generate Positive' or 'Generate Negative' to create 10 training samples",
            justify="center",
            font=("Segoe UI", 11)
        )
        self.info_label.grid(row=0, column=0, pady=(0, 10))
        
        # Grid frame for canvases
        grid_frame = ttk.Frame(container)
        grid_frame.grid(row=1, column=0, pady=10)
        
        # Create 5x2 grid of canvases
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                canvas = tk.Canvas(
                    grid_frame,
                    width=self.display_size,
                    height=self.display_size,
                    bg="black",
                    highlightthickness=1,
                    highlightbackground="#333"
                )
                canvas.grid(row=row, column=col, padx=2, pady=2)
                self.canvases.append(canvas)
        
        # Buttons
        button_frame = ttk.Frame(container)
        button_frame.grid(row=2, column=0, pady=(10, 0))
        
        self.positive_button = ttk.Button(
            button_frame,
            text="Generate Positive (P)",
            command=self.generate_positive_batch
        )
        self.positive_button.pack(side=tk.LEFT, padx=5)
        
        self.negative_button = ttk.Button(
            button_frame,
            text="Generate Negative (N)",
            command=self.generate_negative_batch
        )
        self.negative_button.pack(side=tk.LEFT, padx=5)

        self.hard_negative_button = ttk.Button(
            button_frame,
            text="Generate Hard Negative (H)",
            command=self.generate_hard_negative_batch
        )
        self.hard_negative_button.pack(side=tk.LEFT, padx=5)
        
        # Training controls
        train_frame = ttk.Frame(container)
        train_frame.grid(row=3, column=0, pady=(20, 0))
        
        ttk.Label(train_frame, text="Positive:").pack(side=tk.LEFT, padx=(10, 2))
        self.num_positive_var = tk.StringVar(value="5000")
        self.num_positive_entry = ttk.Entry(train_frame, textvariable=self.num_positive_var, width=8)
        self.num_positive_entry.pack(side=tk.LEFT)
        
        ttk.Label(train_frame, text="Hard Neg:").pack(side=tk.LEFT, padx=(10, 2))
        self.num_hard_negative_var = tk.StringVar(value="5000")
        self.num_hard_negative_entry = ttk.Entry(train_frame, textvariable=self.num_hard_negative_var, width=8)
        self.num_hard_negative_entry.pack(side=tk.LEFT)

        ttk.Label(train_frame, text="Easy Neg:").pack(side=tk.LEFT, padx=(10, 2))
        self.num_negative_var = tk.StringVar(value="15000")
        self.num_negative_entry = ttk.Entry(train_frame, textvariable=self.num_negative_var, width=8)
        self.num_negative_entry.pack(side=tk.LEFT)
        
        self.train_button = ttk.Button(
            train_frame,
            text="Train Model",
            command=self.train_model
        )
        self.train_button.pack(side=tk.LEFT, padx=10)
    
    def generate_positive_batch(self):
        """Generate and display 10 positive training samples."""
        if not self.positive_samples:
            messagebox.showwarning("No Data", "No positive samples available")
            return
        
        self.current_images = []
        num_samples = self.grid_cols * self.grid_rows
        
        for _ in range(num_samples):
            try:
                img = self._generate_single_positive()
                self.current_images.append(img)
            except Exception as e:
                print(f"[Warning] Failed to generate positive sample: {e}")
                # Add black placeholder
                self.current_images.append(np.zeros((128, 128, 3), dtype=np.uint8))
        
        self.current_info = f"Type: POSITIVE (Has Cross) - {num_samples} samples"
        self._display_batch()
    
    def generate_negative_batch(self):
        """Generate and display 10 negative training samples."""
        if not self.negative_samples:
            messagebox.showwarning("No Data", "No negative samples available")
            return
        
        self.current_images = []
        num_samples = self.grid_cols * self.grid_rows
        
        for _ in range(num_samples):
            try:
                img = self._generate_single_negative()
                self.current_images.append(img)
            except Exception as e:
                print(f"[Warning] Failed to generate negative sample: {e}")
                # Add black placeholder
                self.current_images.append(np.zeros((128, 128, 3), dtype=np.uint8))
        
        self.current_info = f"Type: NEGATIVE (No Cross) - {num_samples} samples"
        self._display_batch()

    def generate_hard_negative_batch(self):
        """Generate and display 10 hard negative training samples."""
        if not self.hard_negative_sources:
            messagebox.showwarning("No Data", "No frames with both crosses and negative rects found. Cannot generate hard negatives.")
            return
        
        self.current_images = []
        num_samples = self.grid_cols * self.grid_rows
        
        for _ in range(num_samples):
            try:
                img = self._generate_single_hard_negative()
                self.current_images.append(img)
            except Exception as e:
                print(f"[Warning] Failed to generate hard negative sample: {e}")
                self.current_images.append(np.zeros((128, 128, 3), dtype=np.uint8))
        
        self.current_info = f"Type: HARD NEGATIVE - {num_samples} samples"
        self._display_batch()
    
    def _generate_single_positive(self):
        """Generate a single positive training sample (has cross)."""
        sample = random.choice(self.positive_samples)
        
        # Load frame
        frame = self._load_frame(sample["video_path"], sample["frame_idx"])
        h, w = frame.shape[:2]
        
        # Denormalize cross coordinates
        cross_x = sample["cross"]["x"] * w
        cross_y = sample["cross"]["y"] * h
        
        # Generate random offset so cross is not always centered
        # Cross must be at least 10px from edge of 128x128 crop
        crop_size = 128
        margin = 10
        
        # Calculate valid offset range
        offset_x = random.uniform(-(crop_size//2 - margin), (crop_size//2 - margin))
        offset_y = random.uniform(-(crop_size//2 - margin), (crop_size//2 - margin))
        
        # Calculate crop center
        center_x = cross_x - offset_x
        center_y = cross_y - offset_y
        
        # Clamp crop center to keep crop fully inside frame
        half = crop_size / 2
        center_x = max(half, min(w - half, center_x))
        center_y = max(half, min(h - half, center_y))
        
        # Extract crop (no padding, guaranteed to be inside frame)
        crop = self._extract_crop_clamped(frame, center_x, center_y, crop_size)
        
        # Calculate cross position within crop
        cross_in_crop_x = cross_x - (center_x - half)
        cross_in_crop_y = cross_y - (center_y - half)
        
        # Apply augmentations
        augmented, cross_aug_x, cross_aug_y = self._augment_image(
            crop, cross_in_crop_x, cross_in_crop_y
        )
        
        # Draw crosshair overlay
        augmented = self._draw_crosshair(augmented, cross_aug_x, cross_aug_y)
        
        return augmented
    
    def _generate_single_hard_negative(self):
        """
        Generate a single hard negative sample.
        This is a sample from a 'no-cross' rect, but biased to be close to a real cross.
        """
        if not self.hard_negative_sources:
            raise ValueError("No hard negative sources available.")

        # Hard negatives are derived from positives
        sample = random.choice(self.hard_negative_sources)
        
        # Load frame
        frame = self._load_frame(sample["video_path"], sample["frame_idx"])
        h, w = frame.shape[:2]
        
        # Denormalize cross and rect coordinates
        cross = sample["cross"]
        rect = sample["rect"]
        
        cx = cross["x"] * w
        cy = cross["y"] * h
        
        x0 = rect["x0"] * w
        y0 = rect["y0"] * h
        x1 = rect["x1"] * w
        y1 = rect["y1"] * h

        crop_size = 128
        half = crop_size / 2
        
        # Ensure the rect is large enough to source a crop from
        if (x1 - x0) < crop_size or (y1 - y0) < crop_size:
            # Fallback to a random easy negative if the rect is too small
            return self._generate_single_negative()

        # Find the point in the rect closest to the cross
        # This becomes the ideal center of our hard negative crop
        target_x = np.clip(cx, x0, x1)
        target_y = np.clip(cy, y0, y1)
        
        # Add some jitter so we don't always sample the exact same spot
        jitter_x = random.uniform(-crop_size / 4, crop_size / 4)
        jitter_y = random.uniform(-crop_size / 4, crop_size / 4)
        
        center_x = target_x + jitter_x
        center_y = target_y + jitter_y
        
        # Clamp the center so the entire crop stays inside the negative rect
        center_x = np.clip(center_x, x0 + half, x1 - half)
        center_y = np.clip(center_y, y0 + half, y1 - half)

        # Extract crop
        crop = self._extract_crop_clamped(frame, center_x, center_y, crop_size)
        
        # Apply augmentations (no cross to track)
        augmented, _, _ = self._augment_image(crop, None, None)
        
        return augmented
    
    def _generate_single_negative(self):
        """Generate a single negative training sample (no cross)."""
        sample = random.choice(self.negative_samples)
        
        # Load frame
        frame = self._load_frame(sample["video_path"], sample["frame_idx"])
        h, w = frame.shape[:2]
        
        # Denormalize rect coordinates
        rect = sample["rect"]
        x0 = rect["x0"] * w
        y0 = rect["y0"] * h
        x1 = rect["x1"] * w
        y1 = rect["y1"] * h
        
        rect_w = x1 - x0
        rect_h = y1 - y0
        
        crop_size = 128
        
        # Check if rect is large enough
        if rect_w < crop_size or rect_h < crop_size:
            # Try another sample
            return self._generate_single_negative()
        
        # Random position within rect for crop center
        center_x = random.uniform(x0 + crop_size/2, x1 - crop_size/2)
        center_y = random.uniform(y0 + crop_size/2, y1 - crop_size/2)
        
        # Extract crop (guaranteed to be inside rect, which is inside frame)
        crop = self._extract_crop_clamped(frame, center_x, center_y, crop_size)
        
        # Apply augmentations (no cross to track)
        augmented, _, _ = self._augment_image(crop, None, None)
        
        return augmented
    
    def _load_frame(self, video_path, frame_idx):
        """Load a specific frame from video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx}")
        return frame
    
    def _extract_crop_clamped(self, frame, center_x, center_y, size):
        """Extract a crop centered at (center_x, center_y), guaranteed to be inside frame."""
        half = size // 2
        
        # Calculate crop bounds
        x0 = int(center_x - half)
        y0 = int(center_y - half)
        x1 = x0 + size
        y1 = y0 + size
        
        crop = frame[y0:y1, x0:x1]
        return crop
    
    def _augment_image(self, image, point_x=None, point_y=None):
        """
        Apply augmentations to image and optionally track a point through transformations.
        Returns: (augmented_image, transformed_point_x, transformed_point_y)
        """
        h, w = image.shape[:2]
        
        # Random flip
        flip_h = random.random() < 0.5
        flip_v = random.random() < 0.5
        
        if flip_h:
            image = cv2.flip(image, 1)
            if point_x is not None:
                point_x = w - point_x
        
        if flip_v:
            image = cv2.flip(image, 0)
            if point_y is not None:
                point_y = h - point_y
        
        # Random rotation
        angle = random.uniform(-15, 15)
        
        # Random scale
        scale = random.uniform(0.9, 1.1)
        
        # Build transformation matrix
        center = (w / 2, h / 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Apply rotation+scale
        image = cv2.warpAffine(image, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        
        # Transform point if provided
        if point_x is not None and point_y is not None:
            point = np.array([[[point_x, point_y]]], dtype=np.float32)
            point_transformed = cv2.transform(point, M_rot)
            point_x = point_transformed[0, 0, 0]
            point_y = point_transformed[0, 0, 1]
        
        # Random brightness/contrast
        alpha = random.uniform(0.8, 1.2)  # contrast
        beta = random.uniform(-20, 20)    # brightness
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # Random blur and noise augmentations
        if random.random() < 0.3:
            ksize = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
            
        if random.random() < 0.3:
            noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
            image = cv2.add(image, noise)

        return image, point_x, point_y
    
    def _draw_crosshair(self, image, x, y):
        """Draw a small crosshair at (x, y)."""
        image = image.copy()
        x = int(round(x))
        y = int(round(y))
        size = 8
        color = (0, 255, 0)  # Green
        thickness = 2
        
        cv2.line(image, (x - size, y), (x + size, y), color, thickness)
        cv2.line(image, (x, y - size), (x, y + size), color, thickness)
        
        return image
    
    def _display_batch(self):
        """Display current batch of samples."""
        if not self.current_images:
            return
        
        self.photo_images = []
        
        for idx, img in enumerate(self.current_images):
            # Convert to RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Scale up for display
            display_img = cv2.resize(
                rgb,
                (self.display_size, self.display_size),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Convert to PhotoImage
            pil_img = Image.fromarray(display_img)
            photo = ImageTk.PhotoImage(pil_img)
            self.photo_images.append(photo)
            
            # Update corresponding canvas
            canvas = self.canvases[idx]
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)
        
        # Update info label
        self.info_label.config(text=self.current_info)
    
    def train_model(self):
        """Train the cross detection model."""
        try:
            num_positive = int(self.num_positive_var.get())
            num_hard_negative = int(self.num_hard_negative_var.get())
            num_negative = int(self.num_negative_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for sample counts")
            return
        
        if num_positive <= 0 or num_negative < 0 or num_hard_negative < 0:
            messagebox.showerror("Error", "Sample counts must be positive (or >= 0 for negatives)")
            return
        
        # Run training in a separate thread to keep UI responsive
        thread = threading.Thread(
            target=self._train_worker,
            args=(num_positive, num_hard_negative, num_negative),
            daemon=True
        )
        thread.start()
        messagebox.showinfo("Training", f"Training started with {num_positive} pos, {num_hard_negative} hard neg, and {num_negative} easy neg samples.\nCheck console for progress.")
    
    def _train_worker(self, num_positive, num_hard_negative, num_negative):
        """Background worker for training."""
        try:
            print(f"\n[Train] Generating {num_positive} pos, {num_hard_negative} hard neg, {num_negative} easy neg samples...")
            
            # Step 1: Preload all unique frames
            print("[Train] Preloading unique frames...")
            frame_cache = self._preload_frames()
            print(f"[Train] Loaded {len(frame_cache)} unique frames into memory")
            
            # Step 2: Generate samples in parallel
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing
            
            num_workers = max(1, multiprocessing.cpu_count() - 1)
            print(f"[Train] Using {num_workers} parallel workers for sample generation")
            
            all_samples = []
            
            # Generate positive samples in parallel
            print("[Train] Generating positive samples...")
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                chunk_size = max(1, num_positive // (num_workers * 4))
                
                for i in range(0, num_positive, chunk_size):
                    count = min(chunk_size, num_positive - i)
                    future = executor.submit(
                        generate_positive_samples_batch,
                        self.positive_samples,
                        frame_cache,
                        count
                    )
                    futures.append(future)
                
                completed = 0
                total_futures = len(futures)
                for idx, future in enumerate(as_completed(futures), 1):
                    try:
                        samples = future.result()
                        all_samples.extend(samples)
                        completed += len(samples)
                        print(f"[Train] Positive: {completed}/{num_positive} samples ({idx}/{total_futures} batches)")
                    except Exception as e:
                        print(f"[Warning] Batch failed: {e}")

            # Generate hard negative samples in parallel
            if num_hard_negative > 0:
                print("[Train] Generating hard negative samples...")
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    chunk_size = max(1, num_hard_negative // (num_workers * 4))
                    
                    for i in range(0, num_hard_negative, chunk_size):
                        count = min(chunk_size, num_hard_negative - i)
                        future = executor.submit(
                            generate_hard_negative_samples_batch,
                            self.hard_negative_sources,
                            frame_cache,
                            count
                        )
                        futures.append(future)
                    
                    completed = 0
                    total_futures = len(futures)
                    for idx, future in enumerate(as_completed(futures), 1):
                        try:
                            samples = future.result()
                            all_samples.extend(samples)
                            completed += len(samples)
                            print(f"[Train] Hard Negative: {completed}/{num_hard_negative} samples ({idx}/{total_futures} batches)")
                        except Exception as e:
                            print(f"[Warning] Batch failed: {e}")

            # Generate negative samples in parallel
            print("[Train] Generating easy negative samples...")
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                chunk_size = max(1, num_negative // (num_workers * 4))
                
                for i in range(0, num_negative, chunk_size):
                    count = min(chunk_size, num_negative - i)
                    future = executor.submit(
                        generate_negative_samples_batch,
                        self.negative_samples,
                        frame_cache,
                        count
                    )
                    futures.append(future)
                
                completed = 0
                total_futures = len(futures)
                for idx, future in enumerate(as_completed(futures), 1):
                    try:
                        samples = future.result()
                        all_samples.extend(samples)
                        completed += len(samples)
                        print(f"[Train] Negative: {completed}/{num_negative} samples ({idx}/{total_futures} batches)")
                    except Exception as e:
                        print(f"[Warning] Batch failed: {e}")
            
            print(f"[Train] Generated {len(all_samples)} total samples")
            
            # Split by source frame to avoid data leakage
            train_samples, test_samples = self._split_by_frame(all_samples, test_ratio=0.2)
            
            print(f"[Train] Split: {len(train_samples)} train, {len(test_samples)} test")
            
            # Create datasets and dataloaders
            train_dataset = CrossDataset(train_samples)
            test_dataset = CrossDataset(test_samples)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Create model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[Train] Using device: {device}")
            
            model = CrossDetectorModel().to(device)
            
            # Calculate pos_weight for class imbalance
            num_pos = num_positive
            num_neg = num_negative + num_hard_negative
            pos_weight = num_neg / max(1, num_pos)
            print(f"[Train] Class balance - pos_weight: {pos_weight:.2f}")
            
            # Loss and optimizer
            criterion = CrossDetectionLoss(regression_weight=5.0, pos_weight=pos_weight)
            optimizer = optim.Adam(model.parameters(), lr=0.0003)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3
            )
            
            # Training loop
            num_epochs = 30
            best_mae = float('inf')
            
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                
                for batch in train_loader:
                    images = batch["image"].to(device)
                    has_cross = batch["has_cross"].to(device)
                    coords = batch["coords"].to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, has_cross, coords)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                
                # Evaluate on test set
                metrics = self._evaluate(model, test_loader, device)
                
                print(f"[Train] Epoch {epoch+1}/{num_epochs} "
                      f"loss={avg_train_loss:.4f} "
                      f"acc={metrics['accuracy']:.3f} "
                      f"prec={metrics['precision']:.3f} "
                      f"rec={metrics['recall']:.3f} "
                      f"mae={metrics['mae_pixels']:.2f}px")
                
                # Save best model based on MAE
                if metrics['mae_pixels'] > 0 and metrics['mae_pixels'] < best_mae:
                    best_mae = metrics['mae_pixels']
                    best_model_path = "cross_detector_best.pth"
                    torch.save(model.state_dict(), best_model_path)
                    print(f"[Train] ✓ New best MAE: {best_mae:.2f}px - saved to {best_model_path}")
                
                # Learning rate scheduling
                scheduler.step(avg_train_loss)
            
            # Final evaluation
            print("\n[Train] Training complete! Final test metrics:")
            final_metrics = self._evaluate(model, test_loader, device)
            print(f"  Accuracy:  {final_metrics['accuracy']:.3f}")
            print(f"  Precision: {final_metrics['precision']:.3f}")
            print(f"  Recall:    {final_metrics['recall']:.3f}")
            print(f"  F1 Score:  {final_metrics['f1']:.3f}")
            print(f"  MAE:       {final_metrics['mae_pixels']:.2f} pixels")
            print(f"  Within 5px:  {final_metrics['within_5px']:.1f}%")
            print(f"  Within 10px: {final_metrics['within_10px']:.1f}%")
            
            # Save model
            model_path = "cross_detector.pth"
            torch.save(model.state_dict(), model_path)
            print(f"\n[Train] Model saved to {model_path}")
            
        except Exception as e:
            print(f"[Error] Training failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_single_positive_with_label(self):
        """Generate a positive sample and return it with labels."""
        sample = random.choice(self.positive_samples)
        
        # Load frame
        frame = self._load_frame(sample["video_path"], sample["frame_idx"])
        h, w = frame.shape[:2]
        
        # Denormalize cross coordinates
        cross_x = sample["cross"]["x"] * w
        cross_y = sample["cross"]["y"] * h
        
        crop_size = 128
        margin = 10
        
        # Calculate valid offset range
        offset_x = random.uniform(-(crop_size//2 - margin), (crop_size//2 - margin))
        offset_y = random.uniform(-(crop_size//2 - margin), (crop_size//2 - margin))
        
        # Calculate crop center
        center_x = cross_x - offset_x
        center_y = cross_y - offset_y
        
        # Clamp crop center to keep crop fully inside frame
        half = crop_size / 2
        center_x = max(half, min(w - half, center_x))
        center_y = max(half, min(h - half, center_y))
        
        # Extract crop
        crop = self._extract_crop_clamped(frame, center_x, center_y, crop_size)
        
        # Calculate cross position within crop
        cross_in_crop_x = cross_x - (center_x - half)
        cross_in_crop_y = cross_y - (center_y - half)
        
        # Apply augmentations
        augmented, cross_aug_x, cross_aug_y = self._augment_image(
            crop, cross_in_crop_x, cross_in_crop_y
        )
        
        # Normalize coordinates to 0-1
        norm_x = cross_aug_x / crop_size
        norm_y = cross_aug_y / crop_size
        
        return {
            "image": augmented,
            "has_cross": 1,
            "x": float(norm_x),
            "y": float(norm_y),
            "frame_key": f"{sample['video_path']}_{sample['frame_idx']}"
        }
    
    def _split_by_frame(self, samples, test_ratio=0.2):
        """Split samples by source frame to avoid data leakage."""
        # Group by frame
        frame_groups = {}
        for sample in samples:
            key = sample.get("frame_key", "unknown")
            if key not in frame_groups:
                frame_groups[key] = []
            frame_groups[key].append(sample)
        
        # Shuffle frame keys
        frame_keys = list(frame_groups.keys())
        random.shuffle(frame_keys)
        
        # Split frames
        split_idx = int(len(frame_keys) * (1 - test_ratio))
        train_keys = frame_keys[:split_idx]
        test_keys = frame_keys[split_idx:]
        
        # Collect samples
        train_samples = []
        test_samples = []
        
        for key in train_keys:
            train_samples.extend(frame_groups[key])
        
        for key in test_keys:
            test_samples.extend(frame_groups[key])
        
        return train_samples, test_samples
    
    def _evaluate(self, model, dataloader, device):
        """Evaluate model on a dataset."""
        model.eval()
        
        all_preds_class = []
        all_labels_class = []
        all_preds_coords = []
        all_labels_coords = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device)
                has_cross = batch["has_cross"].cpu().numpy()
                coords = batch["coords"].cpu().numpy()
                
                outputs = model(images)
                has_cross_logits = outputs[:, 0].cpu().numpy()
                pred_coords = outputs[:, 1:].cpu().numpy()
                
                # Classification predictions
                has_cross_preds = (has_cross_logits > 0).astype(int)
                
                all_preds_class.extend(has_cross_preds)
                all_labels_class.extend(has_cross)
                
                # Regression only for true positives
                for i in range(len(has_cross)):
                    if has_cross[i] == 1:
                        all_preds_coords.append(pred_coords[i])
                        all_labels_coords.append(coords[i])
        
        all_preds_class = np.array(all_preds_class)
        all_labels_class = np.array(all_labels_class)
        
        # Classification metrics
        tp = np.sum((all_preds_class == 1) & (all_labels_class == 1))
        fp = np.sum((all_preds_class == 1) & (all_labels_class == 0))
        tn = np.sum((all_preds_class == 0) & (all_labels_class == 0))
        fn = np.sum((all_preds_class == 0) & (all_labels_class == 1))
        
        accuracy = (tp + tn) / max(1, len(all_labels_class))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        
        # Regression metrics (only on true positives)
        mae_pixels = 0.0
        within_5px = 0.0
        within_10px = 0.0
        
        if len(all_preds_coords) > 0:
            all_preds_coords = np.array(all_preds_coords)
            all_labels_coords = np.array(all_labels_coords)
            
            # Convert to pixels (128x128 image)
            pred_pixels = all_preds_coords * 128
            label_pixels = all_labels_coords * 128
            
            # Euclidean distance
            distances = np.sqrt(np.sum((pred_pixels - label_pixels) ** 2, axis=1))
            mae_pixels = np.mean(distances)
            within_5px = 100 * np.mean(distances <= 5)
            within_10px = 100 * np.mean(distances <= 10)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mae_pixels": mae_pixels,
            "within_5px": within_5px,
            "within_10px": within_10px
        }
    
    def _preload_frames(self):
        """Preload all unique frames from positive and negative samples."""
        frame_cache = {}
        unique_frames = set()
        
        # Collect unique frames from positive samples
        for sample in self.positive_samples:
            key = (sample["video_path"], sample["frame_idx"])
            unique_frames.add(key)
        
        # Collect unique frames from negative samples
        for sample in self.negative_samples:
            key = (sample["video_path"], sample["frame_idx"])
            unique_frames.add(key)
        
        # Load all unique frames
        for video_path, frame_idx in unique_frames:
            try:
                frame = self._load_frame(video_path, frame_idx)
                frame_cache[(video_path, frame_idx)] = frame
            except Exception as e:
                print(f"[Warning] Failed to load frame {frame_idx} from {video_path}: {e}")
        
        return frame_cache


# Global functions for parallel processing
def generate_positive_samples_batch(positive_samples, frame_cache, count):
    """Generate a batch of positive samples. Must be top-level for pickling."""
    import random
    import cv2
    import numpy as np
    
    samples = []
    for _ in range(count):
        try:
            sample = random.choice(positive_samples)
            video_path = sample["video_path"]
            frame_idx = sample["frame_idx"]
            
            # Get frame from cache
            frame = frame_cache.get((video_path, frame_idx))
            if frame is None:
                continue
            
            h, w = frame.shape[:2]
            
            # Denormalize cross coordinates
            cross_x = sample["cross"]["x"] * w
            cross_y = sample["cross"]["y"] * h
            
            crop_size = 128
            margin = 10
            
            # Calculate valid offset range
            offset_x = random.uniform(-(crop_size//2 - margin), (crop_size//2 - margin))
            offset_y = random.uniform(-(crop_size//2 - margin), (crop_size//2 - margin))
            
            # Calculate crop center
            center_x = cross_x - offset_x
            center_y = cross_y - offset_y
            
            # Clamp crop center to keep crop fully inside frame
            half = crop_size / 2
            center_x = max(half, min(w - half, center_x))
            center_y = max(half, min(h - half, center_y))
            
            # Extract crop
            x0 = int(center_x - half)
            y0 = int(center_y - half)
            crop = frame[y0:y0+crop_size, x0:x0+crop_size]
            
            # Calculate cross position within crop
            cross_in_crop_x = cross_x - (center_x - half)
            cross_in_crop_y = cross_y - (center_y - half)
            
            # Apply augmentations
            augmented, cross_aug_x, cross_aug_y = augment_image(
                crop, cross_in_crop_x, cross_in_crop_y
            )
            
            # Normalize coordinates to 0-1
            norm_x = cross_aug_x / crop_size
            norm_y = cross_aug_y / crop_size
            
            samples.append({
                "image": augmented,
                "has_cross": 1,
                "x": float(norm_x),
                "y": float(norm_y),
                "frame_key": f"{video_path}_{frame_idx}"
            })
        except Exception as e:
            pass
    
    return samples


def generate_negative_samples_batch(negative_samples, frame_cache, count):
    """Generate a batch of negative samples. Must be top-level for pickling."""
    import random
    import cv2
    import numpy as np
    
    samples = []
    for _ in range(count):
        try:
            sample = random.choice(negative_samples)
            video_path = sample["video_path"]
            frame_idx = sample["frame_idx"]
            
            # Get frame from cache
            frame = frame_cache.get((video_path, frame_idx))
            if frame is None:
                continue
            
            h, w = frame.shape[:2]
            
            # Denormalize rect coordinates
            rect = sample["rect"]
            x0 = rect["x0"] * w
            y0 = rect["y0"] * h
            x1 = rect["x1"] * w
            y1 = rect["y1"] * h
            
            rect_w = x1 - x0
            rect_h = y1 - y0
            
            crop_size = 128
            
            # Check if rect is large enough
            if rect_w < crop_size or rect_h < crop_size:
                continue
            
            # Random position within rect for crop center
            center_x = random.uniform(x0 + crop_size/2, x1 - crop_size/2)
            center_y = random.uniform(y0 + crop_size/2, y1 - crop_size/2)
            
            # Extract crop
            half = crop_size // 2
            cx0 = int(center_x - half)
            cy0 = int(center_y - half)
            crop = frame[cy0:cy0+crop_size, cx0:cx0+crop_size]
            
            # Apply augmentations
            augmented, _, _ = augment_image(crop, None, None)
            
            samples.append({
                "image": augmented,
                "has_cross": 0,
                "x": 0.0,
                "y": 0.0,
                "frame_key": "negative"
            })
        except Exception as e:
            pass
    
    return samples


def generate_hard_negative_samples_batch(hard_negative_sources, frame_cache, count):
    """Generate a batch of hard negative samples. Must be top-level for pickling."""
    import random
    import cv2
    import numpy as np
    
    samples = []
    if not hard_negative_sources:
        return samples

    for _ in range(count):
        try:
            sample = random.choice(hard_negative_sources)
            video_path = sample["video_path"]
            frame_idx = sample["frame_idx"]
            
            frame = frame_cache.get((video_path, frame_idx))
            if frame is None:
                continue
                
            h, w = frame.shape[:2]
            
            cross = sample["cross"]
            rect = sample["rect"]
            
            cx = cross["x"] * w
            cy = cross["y"] * h
            
            x0, y0, x1, y1 = rect["x0"] * w, rect["y0"] * h, rect["x1"] * w, rect["y1"] * h
            
            crop_size = 128
            half = crop_size / 2
            
            if (x1 - x0) < crop_size or (y1 - y0) < crop_size:
                continue

            target_x = np.clip(cx, x0, x1)
            target_y = np.clip(cy, y0, y1)
            
            jitter_x = random.uniform(-crop_size / 4, crop_size / 4)
            jitter_y = random.uniform(-crop_size / 4, crop_size / 4)
            
            center_x = target_x + jitter_x
            center_y = target_y + jitter_y
            
            center_x = np.clip(center_x, x0 + half, x1 - half)
            center_y = np.clip(center_y, y0 + half, y1 - half)

            cx0, cy0 = int(center_x - half), int(center_y - half)
            crop = frame[cy0:cy0+crop_size, cx0:cx0+crop_size]
            
            augmented, _, _ = augment_image(crop, None, None)
            
            samples.append({
                "image": augmented, "has_cross": 0, "x": 0.0, "y": 0.0, "frame_key": "negative_hard"
            })
        except Exception:
            pass
            
    return samples


def augment_image(image, point_x=None, point_y=None):
    """Apply augmentations. Must be top-level for pickling."""
    import random
    import cv2
    import numpy as np
    
    h, w = image.shape[:2]
    
    # Random flip
    flip_h = random.random() < 0.5
    flip_v = random.random() < 0.5
    
    if flip_h:
        image = cv2.flip(image, 1)
        if point_x is not None:
            point_x = w - point_x
    
    if flip_v:
        image = cv2.flip(image, 0)
        if point_y is not None:
            point_y = h - point_y
    
    # Random rotation
    angle = random.uniform(-15, 15)
    
    # Random scale
    scale = random.uniform(0.9, 1.1)
    
    # Build transformation matrix
    center = (w / 2, h / 2)
    M_rot = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Apply rotation+scale
    image = cv2.warpAffine(image, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    
    # Transform point if provided
    if point_x is not None and point_y is not None:
        point = np.array([[[point_x, point_y]]], dtype=np.float32)
        point_transformed = cv2.transform(point, M_rot)
        point_x = point_transformed[0, 0, 0]
        point_y = point_transformed[0, 0, 1]
    
    # Random brightness/contrast
    alpha = random.uniform(0.8, 1.2)  # contrast
    beta = random.uniform(-20, 20)    # brightness
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Random blur and noise augmentations
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        
    if random.random() < 0.3:
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)

    return image, point_x, point_y


class CrossDataset(Dataset):
    """Dataset for cross detection."""
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample["image"]
        
        # Convert BGR to RGB and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        return {
            "image": torch.from_numpy(image),
            "has_cross": torch.tensor(sample["has_cross"], dtype=torch.float32),
            "coords": torch.tensor([sample["x"], sample["y"]], dtype=torch.float32)
        }


class CrossDetectorModel(nn.Module):
    """Mobile-friendly cross detection model."""
    def __init__(self):
        super().__init__()
        
        # Use MobileNetV3-Small as backbone
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # Remove classifier
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        
        # Custom head for cross detection
        # Output: [has_cross_logit, x_normalized, y_normalized]
        self.head = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class CrossDetectionLoss(nn.Module):
    """Combined loss for classification + regression."""
    def __init__(self, regression_weight=5.0, pos_weight=5.0):
        super().__init__()
        self.regression_weight = regression_weight
        pos_weight_tensor = torch.tensor([pos_weight])
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.smooth_l1 = nn.SmoothL1Loss()
    
    def forward(self, outputs, has_cross, coords):
        # Classification loss
        has_cross_logits = outputs[:, 0]
        cls_loss = self.bce(has_cross_logits, has_cross)
        
        # Regression loss (only for samples with crosses)
        mask = has_cross > 0.5
        if mask.sum() > 0:
            pred_coords = outputs[mask, 1:]
            true_coords = coords[mask]
            reg_loss = self.smooth_l1(pred_coords, true_coords)
        else:
            reg_loss = torch.tensor(0.0, device=outputs.device)
        
        return cls_loss + self.regression_weight * reg_loss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Browse and visualize training data samples from annotations."
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="annotations.json",
        help="Path to annotations JSON file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    browser = TrainingDataBrowser(root, args.annotations)
    root.mainloop()


if __name__ == "__main__":
    main()

