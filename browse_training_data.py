import argparse
import json
import os
import random
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


class TrainingDataBrowser:
    def __init__(self, root, annotations_path):
        self.root = root
        self.annotations_path = annotations_path
        self.positive_samples = []
        self.negative_samples = []
        self.current_image = None
        self.current_info = ""
        self.photo_image = None
        self.display_size = 384
        
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
            
            print(f"[Info] Loaded {len(self.positive_samples)} positive samples and {len(self.negative_samples)} negative samples")
            
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
            self.root.geometry("800x900")
        
        self.root.bind("<p>", lambda e: self.generate_positive())
        self.root.bind("<P>", lambda e: self.generate_positive())
        self.root.bind("<n>", lambda e: self.generate_negative())
        self.root.bind("<N>", lambda e: self.generate_negative())
        
        # Info label
        self.info_label = ttk.Label(
            container,
            text="Press 'Generate Positive' or 'Generate Negative' to create a training sample",
            justify="center",
            font=("Segoe UI", 11)
        )
        self.info_label.grid(row=0, column=0, pady=(0, 10))
        
        # Canvas for displaying the sample
        self.canvas = tk.Canvas(
            container,
            width=self.display_size,
            height=self.display_size,
            bg="black",
            highlightthickness=0
        )
        self.canvas.grid(row=1, column=0, pady=10)
        
        # Buttons
        button_frame = ttk.Frame(container)
        button_frame.grid(row=2, column=0, pady=(10, 0))
        
        self.positive_button = ttk.Button(
            button_frame,
            text="Generate Positive (P)",
            command=self.generate_positive
        )
        self.positive_button.pack(side=tk.LEFT, padx=5)
        
        self.negative_button = ttk.Button(
            button_frame,
            text="Generate Negative (N)",
            command=self.generate_negative
        )
        self.negative_button.pack(side=tk.LEFT, padx=5)
    
    def generate_positive(self):
        """Generate and display a positive training sample (has cross)."""
        if not self.positive_samples:
            messagebox.showwarning("No Data", "No positive samples available")
            return
        
        sample = random.choice(self.positive_samples)
        
        try:
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
            
            # Extract crop (with padding to handle edges)
            crop = self._extract_crop(frame, center_x, center_y, crop_size)
            
            # Calculate cross position within crop
            cross_in_crop_x = crop_size / 2 + offset_x
            cross_in_crop_y = crop_size / 2 + offset_y
            
            # Apply augmentations
            augmented, cross_aug_x, cross_aug_y = self._augment_image(
                crop, cross_in_crop_x, cross_in_crop_y
            )
            
            # Draw crosshair overlay
            augmented = self._draw_crosshair(augmented, cross_aug_x, cross_aug_y)
            
            # Display
            self.current_image = augmented
            video_name = os.path.basename(sample["video_path"])
            self.current_info = (
                f"Type: POSITIVE (Has Cross)\n"
                f"Video: {video_name}\n"
                f"Frame: {sample['frame_idx']}\n"
                f"Cross position in crop: ({cross_aug_x:.1f}, {cross_aug_y:.1f})"
            )
            self._display_sample()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate positive sample: {e}")
    
    def generate_negative(self):
        """Generate and display a negative training sample (no cross)."""
        if not self.negative_samples:
            messagebox.showwarning("No Data", "No negative samples available")
            return
        
        sample = random.choice(self.negative_samples)
        
        try:
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
                return self.generate_negative()
            
            # Random position within rect for crop center
            center_x = random.uniform(x0 + crop_size/2, x1 - crop_size/2)
            center_y = random.uniform(y0 + crop_size/2, y1 - crop_size/2)
            
            # Extract crop
            crop = self._extract_crop(frame, center_x, center_y, crop_size)
            
            # Apply augmentations (no cross to track)
            augmented, _, _ = self._augment_image(crop, None, None)
            
            # Display
            self.current_image = augmented
            video_name = os.path.basename(sample["video_path"])
            self.current_info = (
                f"Type: NEGATIVE (No Cross)\n"
                f"Video: {video_name}\n"
                f"Frame: {sample['frame_idx']}\n"
                f"Crop from negative rect ({rect_w:.0f}×{rect_h:.0f})"
            )
            self._display_sample()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate negative sample: {e}")
    
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
    
    def _extract_crop(self, frame, center_x, center_y, size):
        """Extract a crop centered at (center_x, center_y) with padding if needed."""
        h, w = frame.shape[:2]
        half = size // 2
        
        # Calculate crop bounds
        x0 = int(center_x - half)
        y0 = int(center_y - half)
        x1 = x0 + size
        y1 = y0 + size
        
        # Pad frame if crop extends beyond boundaries
        pad_top = max(0, -y0)
        pad_bottom = max(0, y1 - h)
        pad_left = max(0, -x0)
        pad_right = max(0, x1 - w)
        
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            frame = cv2.copyMakeBorder(
                frame,
                pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_REFLECT_101
            )
            x0 += pad_left
            y0 += pad_top
            x1 += pad_left
            y1 += pad_top
        
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
    
    def _display_sample(self):
        """Display current sample and info."""
        if self.current_image is None:
            return
        
        # Convert to RGB
        rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # Scale up for display
        display_img = cv2.resize(
            rgb,
            (self.display_size, self.display_size),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Convert to PhotoImage
        pil_img = Image.fromarray(display_img)
        self.photo_image = ImageTk.PhotoImage(pil_img)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)
        
        # Update info label
        self.info_label.config(text=self.current_info)


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

