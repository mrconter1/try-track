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
from concurrent.futures import ProcessPoolExecutor, as_completed


class TrainingDataBrowser:
    def __init__(self, root, annotations_path):
        self.root = root
        self.annotations_path = annotations_path
        self.sample_sources = [] # Changed from positive_samples, negative_samples, hard_negative_sources
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
        
        with open(self.annotations_path, 'r') as f:
            data = json.load(f)

        # A flat list of all annotated regions, which are our sample sources
        self.sample_sources = []

        for video_data in data.get("videos", []):
            video_path = os.path.abspath(video_data["video_path"])
            for frame_data in video_data.get("frames", []):
                frame_idx = frame_data["frame_idx"]
                for region_data in frame_data.get("regions", []):
                    if region_data.get("crosses"):
                        self.sample_sources.append({
                            "video_path": video_path,
                            "frame_idx": frame_idx,
                            "rect": region_data["rect"],
                            "crosses": region_data["crosses"] # Normalized
                        })
        
        print(f"[Info] Loaded {len(self.sample_sources)} annotated regions as sample sources.")

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
        
        # --- Info Label ---
        self.info_label = ttk.Label(container, text="")
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
        
        # --- Buttons ---
        button_frame = ttk.Frame(container)
        button_frame.grid(row=2, column=0, pady=10)
        
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

        # --- Training controls ---
        train_frame = ttk.Frame(container)
        train_frame.grid(row=3, column=0, pady=(20, 0))
        
        ttk.Label(train_frame, text="Positive:").pack(side=tk.LEFT, padx=(10, 2))
        self.num_positive_var = tk.StringVar(value="25000")
        self.num_positive_entry = ttk.Entry(train_frame, textvariable=self.num_positive_var, width=8)
        self.num_positive_entry.pack(side=tk.LEFT)
        
        ttk.Label(train_frame, text="Negative:").pack(side=tk.LEFT, padx=(10, 2))
        self.num_negative_var = tk.StringVar(value="25000")
        self.num_negative_entry = ttk.Entry(train_frame, textvariable=self.num_negative_var, width=8)
        self.num_negative_entry.pack(side=tk.LEFT)
        
        self.train_button = ttk.Button(
            train_frame,
            text="Train Model",
            command=self.train_model
        )
        self.train_button.pack(side=tk.LEFT, padx=10)
        
        # --- Progress Bar ---
        self.progress_bar = ttk.Progressbar(container, mode='indeterminate')
        self.progress_bar.grid(row=4, column=0, sticky="ew", pady=(10, 0))
    
    def generate_positive_batch(self):
        """Generate and display 10 positive training samples."""
        if not self.sample_sources:
            messagebox.showwarning("No Data", "No annotated regions found.")
            return
        
        self.current_images = []
        num_samples = self.grid_cols * self.grid_rows
        
        for _ in range(num_samples):
            try:
                img, x, y = self._generate_single_positive()
                img_with_cross = draw_crosshair(img.copy(), x, y)
                self.current_images.append(img_with_cross)
            except Exception as e:
                print(f"[Warning] Failed to generate positive sample: {e}")
                self.current_images.append(np.zeros((128, 128, 3), dtype=np.uint8))
            
        self.current_info = f"Type: POSITIVE - {num_samples} samples"
        self._display_batch()
    
    def generate_negative_batch(self):
        """Generate and display 10 negative training samples."""
        if not self.sample_sources:
            messagebox.showwarning("No Data", "No annotated regions found.")
            return
        
        self.current_images = []
        num_samples = self.grid_cols * self.grid_rows
        
        for _ in range(num_samples):
            try:
                img = self._generate_single_negative()
                self.current_images.append(img)
            except Exception as e:
                print(f"[Warning] Failed to generate negative sample: {e}")
                self.current_images.append(np.zeros((128, 128, 3), dtype=np.uint8))
        
        self.current_info = f"Type: NEGATIVE - {num_samples} samples"
        self._display_batch()

    def _generate_single_positive(self):
        if not self.sample_sources:
            raise ValueError("No sample sources available.")

        source = random.choice(self.sample_sources)
        frame = self._load_frame(source["video_path"], source["frame_idx"])

        # Extract the region
        x, y, w, h = source["rect"]
        region_frame = frame[y:y+h, x:x+w]
        if region_frame.shape[0] < 1 or region_frame.shape[1] < 1:
            raise ValueError("Region is empty")

        crop_size = 128
        if region_frame.shape[0] < crop_size or region_frame.shape[1] < crop_size:
            # If the source region itself is too small, we can't get a valid crop.
            # This is an edge case with very low-res videos or tiny regions.
            raise ValueError("Source region is smaller than crop size")

        # Denormalize the cross to be relative to the region
        cross = random.choice(source["crosses"])
        cross_x = cross["x"] * w
        cross_y = cross["y"] * h
        
        # Add random offset so the cross is not always dead-center
        offset_x = random.uniform(-20, 20)
        offset_y = random.uniform(-20, 20)
        center_x = cross_x + offset_x
        center_y = cross_y + offset_y

        # Extract a 128x128 crop around this new center, clamped to region bounds
        crop, new_center = extract_crop_clamped(region_frame, center_x, center_y, crop_size)
        
        # The new cross position is relative to the top-left of the *crop*
        final_cross_x = cross_x - (new_center[0] - crop_size / 2)
        final_cross_y = cross_y - (new_center[1] - crop_size / 2)

        augmented_crop, augmented_x, augmented_y = augment_image(crop, final_cross_x, final_cross_y)
        return augmented_crop, augmented_x, augmented_y
    
    def _generate_single_negative(self):
        if not self.sample_sources:
            raise ValueError("No sample sources available.")

        source = random.choice(self.sample_sources)
        frame = self._load_frame(source["video_path"], source["frame_idx"])

        x, y, w, h = source["rect"]
        region_frame = frame[y:y+h, x:x+w]
        if region_frame.shape[0] < 1 or region_frame.shape[1] < 1:
            raise ValueError("Region is empty")
        
        crop_size = 128
        if region_frame.shape[0] < crop_size or region_frame.shape[1] < crop_size:
            raise ValueError("Source region is smaller than crop size")
        
        # Denormalize all crosses in this region to check for overlaps
        crosses = [[c["x"] * w, c["y"] * h] for c in source["crosses"]]
        
        w_region, h_region = region_frame.shape[:2]
        half = crop_size / 2
        
        # Try a few times to find a crop that doesn't overlap with any cross
        for _ in range(20): # Max 20 attempts
            # Pick a random center for the crop
            center_x = random.uniform(half, w_region - half)
            center_y = random.uniform(half, h_region - half)
            
            # Crop boundaries
            x0, y0 = center_x - half, center_y - half
            x1, y1 = center_x + half, center_y + half
            
            # Check for overlap with any cross
            is_valid = True
            for cross_x, cross_y in crosses:
                if x0 < cross_x < x1 and y0 < cross_y < y1:
                    is_valid = False
                    break
            
            if is_valid:
                crop = extract_crop_clamped(region_frame, center_x, center_y, crop_size)[0]
                augmented_crop, _, _ = augment_image(crop, None, None)
                return augmented_crop
        
        # If we failed to find a valid crop after 20 tries, just return a black image
        return np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

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
            noise = np.random.normal(0, 10, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

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
            num_negative = int(self.num_negative_var.get())
            if num_positive <= 0 or num_negative <= 0:
                raise ValueError("Number of samples must be positive.")
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return
            
        if not self.sample_sources:
            messagebox.showwarning("No Data", "No annotated regions found to generate data from.")
            return

        self.train_button.config(state=tk.DISABLED, text="Training...")
        self.progress_bar.start()
        
        # Run training in a separate thread
        thread = threading.Thread(
            target=self._train_worker,
            args=(num_positive, num_negative),
            daemon=True
        )
        thread.start()

    def _train_worker(self, num_positive, num_negative):
        # --- 1. Preload Frames ---
        frame_cache = self._preload_frames()
        
        all_samples = []
        num_workers = os.cpu_count() - 1 or 1

        # --- 2. Generate Positive Samples ---
        print("[Train] Generating positive samples...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            chunk_size = max(1, num_positive // (num_workers * 4))
            for i in range(0, num_positive, chunk_size):
                count = min(chunk_size, num_positive - i)
                future = executor.submit(generate_positive_samples_batch, self.sample_sources, frame_cache, count)
                futures.append(future)

            completed = 0
            for idx, future in enumerate(as_completed(futures), 1):
                try:
                    samples = future.result()
                    all_samples.extend(samples)
                    completed += len(samples)
                    print(f"[Train] Positive: {completed}/{num_positive} samples ({idx}/{len(futures)} batches)")
                except Exception as e:
                    print(f"[Warning] Batch failed: {e}")

        # --- 3. Generate Negative Samples ---
        print("[Train] Generating negative samples...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            chunk_size = max(1, num_negative // (num_workers * 4))
            for i in range(0, num_negative, chunk_size):
                count = min(chunk_size, num_negative - i)
                future = executor.submit(generate_negative_samples_batch, self.sample_sources, frame_cache, count)
                futures.append(future)

            completed = 0
            for idx, future in enumerate(as_completed(futures), 1):
                try:
                    samples = future.result()
                    all_samples.extend(samples)
                    completed += len(samples)
                    print(f"[Train] Negative: {completed}/{num_negative} samples ({idx}/{len(futures)} batches)")
                except Exception as e:
                    print(f"[Warning] Batch failed: {e}")
        
        if not all_samples:
            print("[Error] No training samples were generated. Aborting training.")
            self.root.after(100, self._on_train_finish)
            return

        # --- 4. Setup for Training ---
        random.shuffle(all_samples)
        print(f"[Train] Generated {len(all_samples)} total samples")
        
        # Split by source frame to avoid data leakage
        train_samples, test_samples = self._split_by_frame(all_samples, test_ratio=0.2)
        print(f"[Train] Split: {len(train_samples)} train, {len(test_samples)} test")
        
        if not train_samples or not test_samples:
            print("[Error] Could not create a valid train/test split. Need more diverse frames.")
            self.root.after(100, self._on_train_finish)
            return

        # Create datasets and dataloaders
        train_dataset = CrossDataset(train_samples)
        test_dataset = CrossDataset(test_samples)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # --- 5. Training Loop ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Train] Using device: {device}")

        model = CrossDetectorModel().to(device)
        
        # Calculate pos_weight for class imbalance
        num_pos = sum(1 for s in all_samples if s["has_cross"] == 1)
        num_neg = len(all_samples) - num_pos
        pos_weight = num_neg / max(1, num_pos)
        print(f"[Train] Class balance - pos_weight: {pos_weight:.2f}")

        criterion = CrossDetectionLoss(pos_weight=pos_weight, device=device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        best_mae = float('inf')
        epochs = 30
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            
            # Validation
            val_metrics = self._evaluate(model, test_loader, device)
            val_loss = val_metrics["loss"]
            
            print(f"[Train] Epoch {epoch+1}/{epochs} loss={avg_loss:.4f} "
                  f"acc={val_metrics['acc']:.5f} prec={val_metrics['prec']:.5f} rec={val_metrics['rec']:.5f} "
                  f"mae={val_metrics['mae_px']:.2f}px")
            
            scheduler.step(val_loss)
            
            if val_metrics['mae_px'] < best_mae:
                best_mae = val_metrics['mae_px']
                print(f"[Train] ✓ New best MAE: {best_mae:.2f}px - saved to cross_detector_best.pth")
                torch.save(model.state_dict(), "cross_detector_best.pth")

        print("[Train] Finished training.")
        self.root.after(100, self._on_train_finish)
    
    def _on_train_finish(self):
        """Callback when training thread is done."""
        self.train_button.config(state=tk.NORMAL, text="Train Model")
        self.progress_bar.stop()
        messagebox.showinfo("Success", "Model training finished. Best model saved to cross_detector_best.pth")
    
    def _preload_frames(self):
        """Load all unique frames needed for generation into memory."""
        frame_cache = {}
        unique_frames = set()
        
        for source in self.sample_sources:
            unique_frames.add((source["video_path"], source["frame_idx"]))
        
        print(f"[Train] Preloading {len(unique_frames)} unique frames into memory...")
        
        for video_path, frame_idx in unique_frames:
            try:
                frame = self._load_frame(video_path, frame_idx)
                frame_cache[(video_path, frame_idx)] = frame
            except Exception as e:
                print(f"[Warning] Failed to load frame {frame_idx} from {video_path}: {e}")
        
        return frame_cache

    def _split_by_frame(self, samples, test_ratio=0.2):
        """
        Split samples into train/test sets, ensuring all crops from a
        single source frame go into the same set to prevent data leakage.
        """
        frames = {}
        for s in samples:
            key = s["frame_key"]
            if key not in frames:
                frames[key] = []
            frames[key].append(s)
            
        frame_keys = list(frames.keys())
        random.shuffle(frame_keys)
        
        split_idx = int(len(frame_keys) * (1 - test_ratio))
        train_keys = frame_keys[:split_idx]
        test_keys = frame_keys[split_idx:]
        
        train_samples = [s for key in train_keys for s in frames[key]]
        test_samples = [s for key in test_keys for s in frames[key]]
        
        return train_samples, test_samples
    
    def _evaluate(self, model, loader, device):
        model.eval()
        all_preds, all_labels = [], []
        
        criterion = CrossDetectionLoss(pos_weight=1.0, device=device) # No weighting for eval
        total_loss = 0
        
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(images)
                total_loss += criterion(outputs, labels).item()
                
                # Store predictions and labels for metrics
                preds_sigmoid = torch.sigmoid(outputs[:, 0])
                all_preds.append(torch.cat([
                    preds_sigmoid.unsqueeze(1),
                    outputs[:, 1:]
                ], dim=1).cpu())
                
                all_labels.append(labels.cpu())
                
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        # Classification metrics
        has_cross_preds = all_preds[:, 0] > 0.5
        has_cross_labels = all_labels[:, 0] > 0.5
        
        accuracy = np.mean(has_cross_preds == has_cross_labels)
        precision = np.sum((has_cross_preds == 1) & (has_cross_labels == 1)) / np.sum(has_cross_preds == 1) if np.sum(has_cross_preds == 1) > 0 else 0
        recall = np.sum((has_cross_preds == 1) & (has_cross_labels == 1)) / np.sum(has_cross_labels == 1) if np.sum(has_cross_labels == 1) > 0 else 0
        
        # Regression metrics (only on true positives)
        pos_preds = all_preds[has_cross_labels == 1, 1:]
        pos_labels = all_labels[has_cross_labels == 1, 1:]
        
        if len(pos_preds) > 0:
            mae_normalized = np.mean(np.abs(pos_preds - pos_labels))
            mae_px = mae_normalized * 128
        else:
            mae_px = float('inf')
            
        return {
            "loss": total_loss / len(loader),
            "acc": accuracy,
            "prec": precision,
            "rec": recall,
            "mae_px": mae_px
        }

# Data handling classes and functions
class CrossDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample["image"]
        
        # Normalize and transpose image
        img_tensor = torch.from_numpy(np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)))
        
        # Create label tensor
        label_tensor = torch.tensor([
            float(sample["has_cross"]),
            float(sample["x"]),
            float(sample["y"])
        ], dtype=torch.float32)
        
        return {"image": img_tensor, "label": label_tensor}
        
class CrossDetectionLoss(nn.Module):
    def __init__(self, pos_weight, device, regression_weight=0.5):
        super().__init__()
        self.regression_weight = regression_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
        self.l1 = nn.SmoothL1Loss()

    def forward(self, outputs, labels):
        has_cross_out = outputs[:, 0]
        coords_out = outputs[:, 1:]
        
        has_cross_label = labels[:, 0]
        coords_label = labels[:, 1:]
        
        # Classification loss (for all samples)
        class_loss = self.bce(has_cross_out, has_cross_label)
        
        # Regression loss (only for positive samples)
        pos_mask = has_cross_label > 0.5
        pos_coords_out = coords_out[pos_mask]
        pos_coords_label = coords_label[pos_mask]
        
        if pos_coords_out.shape[0] > 0:
            reg_loss = self.l1(pos_coords_out, pos_coords_label)
            total_loss = class_loss + self.regression_weight * reg_loss
        else:
            total_loss = class_loss
            
        return total_loss


class CrossDetectorModel(nn.Module):
    """Mobile-friendly cross detection model."""
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        self.head = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # [has_cross_logit, x_norm, y_norm]
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def extract_crop_clamped(image, center_x, center_y, crop_size):
    """Helper for multiprocessing. Extracts a crop, handling boundary conditions by clamping."""
    h, w = image.shape[:2]
    
    # Clamp the center so the crop is always fully inside the image
    clamped_center_x = np.clip(center_x, crop_size / 2, w - crop_size / 2)
    clamped_center_y = np.clip(center_y, crop_size / 2, h - crop_size / 2)
    
    x0 = int(clamped_center_x - crop_size / 2)
    y0 = int(clamped_center_y - crop_size / 2)
    
    crop = image[y0:y0+crop_size, x0:x0+crop_size]
    return crop, (clamped_center_x, clamped_center_y)


def generate_negative_samples_batch(sample_sources, frame_cache, count):
    """Generate a batch of negative samples. Must be top-level for pickling."""
    import random
    import cv2
    import numpy as np
    
    samples = []
    if not sample_sources: return samples

    for _ in range(count):
        try:
            source = random.choice(sample_sources)
            video_path = source["video_path"]
            frame_idx = source["frame_idx"]
            frame = frame_cache.get((video_path, frame_idx))
            if frame is None: continue

            x, y, w, h = source["rect"]
            region_frame = frame[y:y+h, x:x+w]
            if region_frame.shape[0] < 1 or region_frame.shape[1] < 1: continue
            
            crop_size = 128
            if region_frame.shape[0] < crop_size or region_frame.shape[1] < crop_size: continue

            crosses = [[c["x"] * w, c["y"] * h] for c in source["crosses"]]
            
            w_region, h_region = region_frame.shape[:2]
            
            is_valid = False
            for _ in range(20):
                # Pick a random center for the crop
                center_x = random.uniform(0, w_region)
                center_y = random.uniform(0, h_region)
                
                # Get a random transform centered on this point
                transform_matrix = get_random_affine_transform((center_x, center_y), (crop_size, crop_size))

                # Check if any cross would land inside the crop
                transformed_crosses = cv2.transform(np.array([[c] for c in crosses]), transform_matrix)
                
                if not any(0 <= tc[0][0] < crop_size and 0 <= tc[0][1] < crop_size for tc in transformed_crosses):
                    is_valid = True
                    break
            
            if is_valid:
                crop = cv2.warpAffine(region_frame, transform_matrix, (crop_size, crop_size), borderMode=cv2.BORDER_REFLECT_101)
                augmented_crop, _, _ = augment_image(crop, None, None)
                samples.append({
                    "image": augmented_crop, "has_cross": 0, "x": 0.0, "y": 0.0, "frame_key": "negative"
                })
        except Exception:
            pass
    return samples


def augment_image(image, point_x=None, point_y=None):
    """Apply color, noise, and flip augmentations. Must be top-level for pickling."""
    import random
    import cv2
    import numpy as np
    
    h, w = image.shape[:2]
    
    # Random flip
    if random.random() < 0.5:
        image = cv2.flip(image, 1) # Horizontal
        if point_x is not None:
            point_x = w - point_x
    
    if random.random() < 0.5:
        image = cv2.flip(image, 0) # Vertical
        if point_y is not None:
            point_y = h - point_y
            
    # Random brightness/contrast
    alpha = random.uniform(0.8, 1.2)  # contrast
    beta = random.uniform(-20, 20)    # brightness
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Random blur and noise
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        
    if random.random() < 0.3:
        noise = np.random.normal(0, 10, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image, point_x, point_y


def get_random_affine_transform(center, output_size):
    """Generate a random affine transformation matrix."""
    import random
    import cv2
    import numpy as np

    out_w, out_h = output_size
    
    # Random rotation
    angle = random.uniform(0, 360)
    
    # Random scale (zoom)
    scale = random.uniform(0.9, 1.1)
    
    # Random non-uniform scale (stretch)
    scale_x = scale * random.uniform(0.9, 1.1)
    scale_y = scale * random.uniform(0.9, 1.1)
    
    # Random shear
    shear_x = random.uniform(-0.1, 0.1)
    shear_y = random.uniform(-0.1, 0.1)
    
    # --- Build the matrix ---
    # 1. Start with translation to origin
    T1 = np.float32([[1, 0, -center[0]], [0, 1, -center[1]]])
    
    # 2. Add Shear
    S = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
    
    # 3. Add Rotation and Scale
    R_mat = cv2.getRotationMatrix2D((0,0), angle, 1.0)
    R = np.vstack([R_mat, [0, 0, 1]]) # to 3x3 for matrix multiplication
    Sc = np.float32([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
    
    # Combine Scale, Shear, Rotation
    M = (S @ R @ Sc)[:2, :]
    
    # 4. Translate to center of output image
    T2 = np.float32([[1, 0, out_w / 2], [0, 1, out_h / 2]])
    
    # 5. Combine all transformations
    # The transformation for warpAffine is T2 * M * T1
    # cv2.transform needs a 3x3, so we build it up then slice
    T1_3x3 = np.vstack([T1, [0,0,1]])
    T2_3x3 = np.vstack([T2, [0,0,1]])
    M_3x3 = np.vstack([M, [0,0,1]])
    
    final_M = (T2_3x3 @ M_3x3 @ T1_3x3)
    
    return final_M[:2, :]


def draw_crosshair(image, x, y):
    """Draws a crosshair on the image for visualization."""
    if x is not None and y is not None:
        px, py = int(x), int(y)
        # Green cross with black outline for visibility
        cv2.line(image, (px - 8, py), (px + 8, py), (0, 0, 0), 3)
        cv2.line(image, (px, py - 8), (px, py + 8), (0, 0, 0), 3)
        cv2.line(image, (px - 8, py), (px + 8, py), (0, 255, 0), 1)
        cv2.line(image, (px, py - 8), (px, py + 8), (0, 255, 0), 1)
    return image


def generate_positive_samples_batch(sample_sources, frame_cache, count):
    """Generate a batch of positive samples. Must be top-level for pickling."""
    import random
    import cv2
    import numpy as np
    
    samples = []
    if not sample_sources: return samples

    for _ in range(count):
        try:
            source = random.choice(sample_sources)
            video_path = source["video_path"]
            frame_idx = source["frame_idx"]
            frame = frame_cache.get((video_path, frame_idx))
            if frame is None: continue

            x, y, w, h = source["rect"]
            region_frame = frame[y:y+h, x:x+w]
            if region_frame.shape[0] < 1 or region_frame.shape[1] < 1: continue
            
            crop_size = 128
            if region_frame.shape[0] < crop_size or region_frame.shape[1] < crop_size: continue

            cross = random.choice(source["crosses"])
            cross_x, cross_y = cross["x"] * w, cross["y"] * h
            
            # Add random offset so the cross is not always dead-center
            offset_x = random.uniform(-20, 20)
            offset_y = random.uniform(-20, 20)
            center_x, center_y = cross_x + offset_x, cross_y + offset_y

            # Get the random affine transformation
            transform_matrix = get_random_affine_transform((center_x, center_y), (crop_size, crop_size))
            
            # Warp the image to get the augmented crop
            crop = cv2.warpAffine(region_frame, transform_matrix, (crop_size, crop_size), borderMode=cv2.BORDER_REFLECT_101)
            
            # Transform the original cross coordinate to find its new position
            original_cross_point = np.array([[[cross_x, cross_y]]])
            transformed_cross = cv2.transform(original_cross_point, transform_matrix)
            aug_x, aug_y = transformed_cross[0,0]

            # Apply color/noise augmentations
            augmented_crop, aug_x, aug_y = augment_image(crop, aug_x, aug_y)
            
            if aug_x is not None:
                samples.append({
                    "image": augmented_crop,
                    "has_cross": 1,
                    "x": aug_x / crop_size,
                    "y": aug_y / crop_size,
                    "frame_key": f"{os.path.basename(source['video_path'])}_{source['frame_idx']}"
                })
        except Exception:
            pass
    return samples


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

