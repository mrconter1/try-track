import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
import random
import bisect
import argparse
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
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
        return {
            "samples": [
                {
                    "video_path": sample.video_path,
                    "frame_idx": sample.frame_idx,
                    "crop_rect": list(sample.crop_rect),
                    "lines": [
                        {
                            "start": list(line.start),
                            "end": list(line.end)
                        }
                        for line in sample.lines
                    ]
                }
                for sample in self.samples
            ]
        }
    
    def save(self, filepath: str):
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'AnnotationDatabase':
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
                crop_rect=tuple(sample_data["crop_rect"])
            )
            for line_data in sample_data.get("lines", []):
                sample.add_line(
                    tuple(line_data["start"]),
                    tuple(line_data["end"])
                )
            db.samples.append(sample)
        
        return db

# Mobile-optimized U-Net with MobileNetV2 backbone
class MobileUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained MobileNetV2 as encoder
        if pretrained:
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            mobilenet = models.mobilenet_v2(weights=None)
        self.encoder = mobilenet.features
        
        # MobileNetV2 output channels at different stages (after specific layers):
        # Layer 1: 16 channels (stride 2, 64x64)
        # Layer 3: 24 channels (stride 4, 32x32)
        # Layer 6: 32 channels (stride 8, 16x16)
        # Layer 13: 96 channels (stride 16, 8x8)
        # Layer 18 (final): 1280 channels (stride 32, 4x4)
        
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
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encoder with skip connections
        skip_connections = []
        
        # Extract features at different scales
        # Correct layer indices for MobileNetV2: [1, 3, 6, 13, -1]
        skip_indices = [1, 3, 6, 13]
        
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            if idx in skip_indices:
                skip_connections.append(x)
        
        # x is now the final output (1280 channels @ 4x4)
        # skip_connections: [16ch@64x64, 24ch@32x32, 32ch@16x16, 96ch@8x8]
        
        # Decoder with skip connections
        x = self.up1(x)  # 1280 -> 96, 4x4 -> 8x8
        x = torch.cat([x, skip_connections[3]], dim=1)  # concat with 96ch@8x8
        x = self.dec1(x)
        
        x = self.up2(x)  # 96 -> 32, 8x8 -> 16x16
        x = torch.cat([x, skip_connections[2]], dim=1)  # concat with 32ch@16x16
        x = self.dec2(x)
        
        x = self.up3(x)  # 32 -> 24, 16x16 -> 32x32
        x = torch.cat([x, skip_connections[1]], dim=1)  # concat with 24ch@32x32
        x = self.dec3(x)
        
        x = self.up4(x)  # 24 -> 16, 32x32 -> 64x64
        x = torch.cat([x, skip_connections[0]], dim=1)  # concat with 16ch@64x64
        x = self.dec4(x)
        
        x = self.final_up(x)  # 16 -> 16, 64x64 -> 128x128
        x = self.out(x)
        
        return x

# Dataset class
class LineDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image'].astype(np.float32) / 255.0
        mask = sample['mask'][:, :, 0].astype(np.float32) / 255.0
        
        # Convert to tensors (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask

def get_video_props(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

class RandomPatchViewer:
    def __init__(self, root, video_paths, patch_size=400):
        self.root = root
        self.video_paths = [os.path.abspath(p) for p in video_paths]
        self.patch_size = patch_size
        
        # Pre-calculate frame counts for proportional sampling
        self.video_frame_counts = {}
        print("Scanning videos...")
        for path in self.video_paths:
            count = get_video_props(path)
            if count > 0:
                self.video_frame_counts[path] = count
                print(f"  {os.path.basename(path)}: {count} frames")
        
        # Setup proportional sampling (Cumulative Distribution)
        self.cumulative_frames = []
        self.active_video_paths = [] # Only videos with >0 frames
        self.total_combined_frames = 0
        current_total = 0
        
        for path in self.video_paths:
            if path in self.video_frame_counts:
                count = self.video_frame_counts[path]
                current_total += count
                self.cumulative_frames.append(current_total)
                self.active_video_paths.append(path)
                
        self.total_combined_frames = current_total
        print(f"Total frames across {len(self.active_video_paths)} videos: {self.total_combined_frames}")

        # Annotation database - load existing or create new
        self.db = AnnotationDatabase.load("line_annotations.json")
        
        # State
        self.current_patch_info = None # {video_path, frame_idx, crop_rect: (x,y,w,h), image}
        self.history = [] # List of patch_info dicts (will be populated from db)
        self.history_idx = -1
        self.history_loaded = False  # Track if we've loaded history from db
        
        self.photo_image = None
        self.scale = 1.0
        
        # Line drawing state (working copy for current region)
        self.lines = [] # List of [(x1, y1), (x2, y2)] in image coords (crop-relative)
        self.first_point = None # The locked-in first point
        self.current_point = None # The point being dragged right now
        self.is_dragging = False
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.display_scale = 1.0
        self.show_mask_mode = False  # Toggle between normal and mask view
        self.show_lines = True  # Toggle line visibility
        self.selected_line_idx = None  # Index of currently selected line
        
        # State for editing existing points
        self.editing_point = None  # (line_index, point_index) being edited, or None
        
        # Data generation state
        self.generated_patches = []  # List of {image, mask, source_info}
        self.gen_patch_idx = -1
        self.show_gen_mask = False  # Toggle between image and mask in gen tab
        
        # Training state
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_training = False
        self.training_samples = []
        
        # Inference state
        self.inference_samples = []
        self.inference_idx = 0
        self.show_inference_prediction = False
        
        # UI Setup
        self.root.title(f"Random Patch Viewer ({patch_size}x{patch_size})")
        self._build_ui()
        
        # Bindings
        self.root.bind("<Configure>", self.on_resize)
        self.root.bind("<a>", lambda e: self.prev_patch())
        self.root.bind("<d>", lambda e: self.next_patch())
        self.root.bind("<Left>", lambda e: self.prev_patch())
        self.root.bind("<Right>", lambda e: self.next_patch())
        self.root.bind("<m>", lambda e: self.on_toggle_mask_key())
        self.root.bind("<l>", lambda e: self.toggle_lines())
        self.root.bind("<Delete>", lambda e: self.delete_selected_line())
        self.root.bind("<Control-s>", lambda e: self.save_annotations())
        self.root.bind("<r>", lambda e: self.on_regenerate_key())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.canvas.bind("<Button-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Load history from database
        self._load_history_from_db()
        
        # Initial patch - either last from history or generate new
        if self.total_combined_frames > 0:
            if len(self.history) > 0:
                # Start at the end of loaded history
                self.history_idx = len(self.history) - 1
                self.current_patch_info = self.history[self.history_idx]
                self._load_current_lines()
                self.display_current_patch()
            else:
                # No history, generate first patch
                self.next_patch()
        else:
            messagebox.showerror("Error", "No valid frames found in the provided videos.")

    def _load_history_from_db(self):
        """Reconstruct history from saved samples in database."""
        for sample in self.db.samples:
            # Need to reload the actual image for each sample
            cap = cv2.VideoCapture(sample.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                x, y, w, h = sample.crop_rect
                patch = frame[y:y+h, x:x+w]
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                
                patch_info = {
                    "video_path": sample.video_path,
                    "frame_idx": sample.frame_idx,
                    "crop_rect": sample.crop_rect,
                    "image": patch_rgb
                }
                self.history.append(patch_info)
            else:
                print(f"Warning: Could not reload sample from {sample.video_path} frame {sample.frame_idx}")
        
        print(f"Loaded {len(self.history)} samples from database")
    
    def _build_ui(self):
        # Tab control for whole GUI
        self.tab_control = ttk.Notebook(self.root)
        self.tab_control.pack(fill=tk.BOTH, expand=True)
        
        # Labelling Tab
        self.labelling_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.labelling_tab, text="Labelling")
        
        # Data Generation Tab
        self.data_gen_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.data_gen_tab, text="Data Generation")
        
        # Training Tab
        self.training_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.training_tab, text="Training")
        
        # Inference Tab
        self.inference_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.inference_tab, text="Inference")
        
        self._build_labelling_tab()
        self._build_data_generation_tab()
        self._build_training_tab()
        self._build_inference_tab()
    
    def _build_labelling_tab(self):
        """Build the UI for the labelling tab."""
        # Main frame with canvas and sidebar
        main_frame = ttk.Frame(self.labelling_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas Area (Left)
        self.canvas = tk.Canvas(main_frame, bg="#222222", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Sidebar (Right)
        sidebar = ttk.Frame(main_frame, width=300, padding=10)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False) # Force width
        
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
        stats_frame = ttk.LabelFrame(sidebar, text="Dataset Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.lbl_total_samples = ttk.Label(stats_frame, text="Total samples: 0")
        self.lbl_total_samples.pack(anchor="w", pady=2)
        
        self.lbl_labeled_samples = ttk.Label(stats_frame, text="Labeled samples: 0")
        self.lbl_labeled_samples.pack(anchor="w", pady=2)
        
        self.lbl_total_lines = ttk.Label(stats_frame, text="Total lines: 0")
        self.lbl_total_lines.pack(anchor="w", pady=2)
        
        # Lines List Panel
        lines_frame = ttk.LabelFrame(sidebar, text="Lines", padding=10)
        lines_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Create scrollable list for lines
        list_container = ttk.Frame(lines_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.lines_listbox = tk.Listbox(list_container, yscrollcommand=scrollbar.set, height=10)
        self.lines_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.lines_listbox.yview)
        self.lines_listbox.bind('<<ListboxSelect>>', self.on_line_select)
        
        # Delete button
        self.btn_delete = ttk.Button(lines_frame, text="Delete Line", command=self.delete_selected_line)
        self.btn_delete.pack(pady=5)
        
        # Navigation Panel
        nav_frame = ttk.LabelFrame(sidebar, text="Navigation", padding=10)
        nav_frame.pack(fill=tk.X, pady=(0, 20))
        
        btn_prev = ttk.Button(nav_frame, text="<< Previous (A)", command=self.prev_patch)
        btn_prev.pack(fill=tk.X, pady=5)
        
        btn_next = ttk.Button(nav_frame, text="Next Random (D) >>", command=self.next_patch)
        btn_next.pack(fill=tk.X, pady=5)
        
        self.btn_toggle_mask = ttk.Button(nav_frame, text="Show Mask (M)", command=self.toggle_mask)
        self.btn_toggle_mask.pack(fill=tk.X, pady=5)
        
        btn_save = ttk.Button(nav_frame, text="Save (Ctrl+S)", command=self.save_annotations)
        btn_save.pack(fill=tk.X, pady=5)

        # Instructions
        ttk.Label(sidebar, text="Instructions:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(20, 5))
        ttk.Label(sidebar, text="• Press 'D' or Right Arrow for a new random sample\n• Press 'A' or Left Arrow to go back\n• Click to place lines\n• Delete key to remove selected line\n• Ctrl+S to save").pack(anchor="w")
    
    def _build_data_generation_tab(self):
        """Build the UI for the data generation tab."""
        # Main frame with canvas and sidebar
        main_frame = ttk.Frame(self.data_gen_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas Area (Left)
        self.gen_canvas = tk.Canvas(main_frame, bg="#222222", highlightthickness=0)
        self.gen_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Sidebar (Right)
        sidebar = ttk.Frame(main_frame, width=300, padding=10)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False) # Force width
        
        # Generation controls
        gen_frame = ttk.LabelFrame(sidebar, text="Generation Controls", padding=10)
        gen_frame.pack(fill=tk.X, pady=(10, 10))
        
        ttk.Label(gen_frame, text="Patch size: 128x128").pack(anchor="w", pady=2)
        ttk.Label(gen_frame, text="Grid: 4x4 (16 patches)").pack(anchor="w", pady=2)
        
        btn_generate = ttk.Button(gen_frame, text="Generate 16 Random Patches", command=self.generate_training_patches)
        btn_generate.pack(fill=tk.X, pady=5)
        
        self.btn_toggle_gen_view = ttk.Button(gen_frame, text="Show: Images", command=self.toggle_generation_view)
        self.btn_toggle_gen_view.pack(fill=tk.X, pady=5)
        
        # Generation info
        info_frame = ttk.LabelFrame(sidebar, text="Grid Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_gen_count = ttk.Label(info_frame, text="Patches: 0")
        self.lbl_gen_count.pack(anchor="w", pady=2)
        
        # Instructions
        ttk.Label(sidebar, text="Instructions:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(20, 5))
        ttk.Label(sidebar, text="• Press 'R' to generate new patches\n• Press 'M' to toggle images/masks\n• Generates 4x4 grid from labeled data").pack(anchor="w")
    
    def _build_training_tab(self):
        """Build the UI for the training tab."""
        # Main frame
        main_frame = ttk.Frame(self.training_tab, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Training controls
        controls_frame = ttk.LabelFrame(main_frame, text="Training Configuration", padding=10)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Number of samples
        ttk.Label(controls_frame, text="Training Samples:").grid(row=0, column=0, sticky="w", pady=5)
        self.train_samples_var = tk.IntVar(value=5000)
        samples_spinbox = ttk.Spinbox(controls_frame, from_=100, to=10000, increment=100, 
                                       textvariable=self.train_samples_var, width=10)
        samples_spinbox.grid(row=0, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # Epochs
        ttk.Label(controls_frame, text="Epochs:").grid(row=1, column=0, sticky="w", pady=5)
        self.epochs_var = tk.IntVar(value=50)
        epochs_spinbox = ttk.Spinbox(controls_frame, from_=10, to=200, increment=10, 
                                      textvariable=self.epochs_var, width=10)
        epochs_spinbox.grid(row=1, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # Batch size
        ttk.Label(controls_frame, text="Batch Size:").grid(row=2, column=0, sticky="w", pady=5)
        self.batch_size_var = tk.IntVar(value=16)
        batch_spinbox = ttk.Spinbox(controls_frame, from_=4, to=64, increment=4, 
                                     textvariable=self.batch_size_var, width=10)
        batch_spinbox.grid(row=2, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # Learning rate
        ttk.Label(controls_frame, text="Learning Rate:").grid(row=3, column=0, sticky="w", pady=5)
        self.lr_var = tk.DoubleVar(value=0.001)
        lr_entry = ttk.Entry(controls_frame, textvariable=self.lr_var, width=10)
        lr_entry.grid(row=3, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # Device info
        device_text = f"Device: {self.device}"
        ttk.Label(controls_frame, text=device_text, foreground="blue").grid(row=4, column=0, columnspan=2, sticky="w", pady=5)
        
        # Model info
        ttk.Label(controls_frame, text="Backbone: MobileNetV2 (ImageNet)", foreground="green").grid(row=5, column=0, columnspan=2, sticky="w", pady=2)
        
        # Train button
        self.btn_train = ttk.Button(controls_frame, text="Start Training", command=self.start_training)
        self.btn_train.grid(row=6, column=0, columnspan=2, sticky="ew", pady=10)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Training Progress", padding=10)
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Status label
        self.lbl_train_status = ttk.Label(progress_frame, text="Ready to train")
        self.lbl_train_status.pack(anchor="w", pady=5)
        
        # Loss display
        self.lbl_train_loss = ttk.Label(progress_frame, text="Train Loss: -")
        self.lbl_train_loss.pack(anchor="w", pady=2)
        
        self.lbl_val_loss = ttk.Label(progress_frame, text="Val Loss: -")
        self.lbl_val_loss.pack(anchor="w", pady=2)
        
        # Log text area
        log_frame = ttk.Frame(progress_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.train_log = tk.Text(log_frame, height=15, yscrollcommand=scrollbar.set, state='disabled')
        self.train_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.train_log.yview)
        
        # Model info
        model_frame = ttk.LabelFrame(main_frame, text="Model Info", padding=10)
        model_frame.pack(fill=tk.X)
        
        self.lbl_model_status = ttk.Label(model_frame, text="Model: Not trained")
        self.lbl_model_status.pack(anchor="w", pady=2)
        
        btn_save_model = ttk.Button(model_frame, text="Save Model", command=self.save_model)
        btn_save_model.pack(fill=tk.X, pady=5)
    
    def _build_inference_tab(self):
        """Build the UI for the inference tab."""
        # Main frame with canvas and sidebar
        main_frame = ttk.Frame(self.inference_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas Area (Left)
        self.inference_canvas = tk.Canvas(main_frame, bg="#222222", highlightthickness=0)
        self.inference_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
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
        sample_frame = ttk.LabelFrame(sidebar, text="Test Samples", padding=10)
        sample_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(sample_frame, text="Number of samples:").pack(anchor="w", pady=2)
        self.inference_samples_var = tk.IntVar(value=16)
        samples_spinbox = ttk.Spinbox(sample_frame, from_=1, to=100, increment=1, 
                                       textvariable=self.inference_samples_var, width=10)
        samples_spinbox.pack(anchor="w", pady=5)
        
        btn_generate_inference = ttk.Button(sample_frame, text="Generate Test Samples", 
                                           command=self.generate_inference_samples)
        btn_generate_inference.pack(fill=tk.X, pady=5)
        
        # View controls
        view_frame = ttk.LabelFrame(sidebar, text="View", padding=10)
        view_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_toggle_inference = ttk.Button(view_frame, text="Show: Image", 
                                               command=self.toggle_inference_view)
        self.btn_toggle_inference.pack(fill=tk.X, pady=5)
        
        # Info
        info_frame = ttk.LabelFrame(sidebar, text="Grid Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_inference_idx = ttk.Label(info_frame, text="Samples: 0")
        self.lbl_inference_idx.pack(anchor="w", pady=5)
        
        # Instructions
        ttk.Label(sidebar, text="Instructions:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(20, 5))
        ttk.Label(sidebar, text="• Load trained model\n• Generate 16 test samples\n• Toggle between images/predictions\n• View as 4x4 grid").pack(anchor="w")

    def get_random_frame_location(self):
        """Select a video and frame index proportional to frame count."""
        if self.total_combined_frames == 0:
            return None, None
            
        global_idx = random.randint(0, self.total_combined_frames - 1)
        video_idx = bisect.bisect_left(self.cumulative_frames, global_idx)
        
        # Safety check
        if video_idx >= len(self.active_video_paths):
            video_idx = len(self.active_video_paths) - 1
            
        video_path = self.active_video_paths[video_idx]
        
        prev_cumulative = self.cumulative_frames[video_idx - 1] if video_idx > 0 else 0
        frame_idx = global_idx - prev_cumulative # Local frame index
        
        return video_path, frame_idx

    def generate_new_patch(self):
        if self.total_combined_frames == 0:
            return None

        # Try up to 10 times to get a valid frame/patch (in case of read errors)
        for _ in range(10):
            video_path, frame_idx = self.get_random_frame_location()
            if not video_path:
                        continue
                    
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                print(f"Warning: Could not read frame {frame_idx} from {video_path}")
                continue
                
            h, w = frame.shape[:2]
            
            # Ensure frame is large enough
            if h < self.patch_size or w < self.patch_size:
                # If frame is too small, just take center crop or resize? 
                # User asked for 250x250. If smaller, let's skip or take whole.
                # Let's just pad it if smaller, or skip. Skipping is safer for "random 250x250 area"
                if h < 10 or w < 10: # Extremely small
                    continue
                
                # If slightly smaller, just use 0,0 and min size
                x, y = 0, 0
                cw, ch = min(w, self.patch_size), min(h, self.patch_size)
            else:
                x = random.randint(0, w - self.patch_size)
                y = random.randint(0, h - self.patch_size)
                cw, ch = self.patch_size, self.patch_size
            
            # Extract patch
            patch = frame[y:y+ch, x:x+cw]
            
            # Convert BGR to RGB
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            
            return {
                "video_path": video_path,
                "frame_idx": frame_idx,
                "crop_rect": (x, y, cw, ch),
                "image": patch_rgb
            }
            
        return None

    def next_patch(self):
        # Save current lines to database before moving
        self._save_current_lines()
        
        if self.history_idx < len(self.history) - 1:
            # Move forward in history
            self.history_idx += 1
            self.current_patch_info = self.history[self.history_idx]
        else:
            # Generate new
            new_info = self.generate_new_patch()
            if new_info:
                self.history.append(new_info)
                self.history_idx = len(self.history) - 1
                self.current_patch_info = new_info
        
        # Load lines for the new patch
        self._load_current_lines()
        self.display_current_patch()
    
    def prev_patch(self):
        # Save current lines to database before moving
        self._save_current_lines()
        
        if self.history_idx > 0:
            self.history_idx -= 1
            self.current_patch_info = self.history[self.history_idx]
            
            # Load lines for this patch
            self._load_current_lines()
            self.display_current_patch()

    def _save_current_lines(self):
        """Save current working lines to the database."""
        if not self.current_patch_info:
            return
        
        info = self.current_patch_info
        sample = self.db.find_sample(
            info['video_path'],
            info['frame_idx'],
            info['crop_rect']
        )
        
        # Clear existing lines and add current ones
        sample.lines.clear()
        for line in self.lines:
            sample.add_line(line[0], line[1])
    
    def _load_current_lines(self):
        """Load lines from database for current patch."""
        self.lines = []
        self.selected_line_idx = None
        self.first_point = None
        self.current_point = None
        self.editing_point = None
        
        if not self.current_patch_info:
            return
        
        info = self.current_patch_info
        sample = self.db.find_sample(
            info['video_path'],
            info['frame_idx'],
            info['crop_rect']
        )
        
        # Convert from Line dataclass to list format
        for line in sample.lines:
            self.lines.append([line.start, line.end])
    
    def display_current_patch(self):
        if not self.current_patch_info:
            return
            
        info = self.current_patch_info
        
        # Update Info Labels
        self.lbl_video.config(text=f"Video: {os.path.basename(info['video_path'])}")
        self.lbl_frame.config(text=f"Frame: {info['frame_idx']}")
        x, y, w, h = info['crop_rect']
        self.lbl_coords.config(text=f"Crop: x={x}, y={y} ({w}x{h})")
        
        # Update Statistics
        self.update_statistics()
        
        # Update Lines List
        self.update_lines_list()
        
        # Display Image on Canvas
        self.draw_image()

    def draw_image(self):
        if not self.current_patch_info:
            return
            
        img_arr = self.current_patch_info['image']
        img_h, img_w = img_arr.shape[:2]
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            # Wait for layout
            self.root.after(100, self.draw_image)
            return
        
        # Choose what to display based on mode
        if self.show_mask_mode:
            # Create mask view
            display_arr = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            # Draw each line as 3px wide white line
            for line in self.lines:
                p1, p2 = line
                x1, y1 = int(round(p1[0])), int(round(p1[1]))
                x2, y2 = int(round(p2[0])), int(round(p2[1]))
                cv2.line(display_arr, (x1, y1), (x2, y2), (255, 255, 255), thickness=3)
        else:
            # Normal view - show image
            display_arr = img_arr
        
        # Scale image to ~33% of screen height while maintaining aspect ratio
        target_height = int(canvas_h * 0.33)
        scale = target_height / img_h
        
        # Use Nearest Neighbor for sharp upscaling, Linear for downscaling
        interp = cv2.INTER_NEAREST if scale > 1.5 else cv2.INTER_LINEAR
        
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized = cv2.resize(display_arr, (new_w, new_h), interpolation=interp)
        
        self.photo_image = ImageTk.PhotoImage(Image.fromarray(resized))
        
        # Center horizontally and vertically
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        
        self.canvas_offset_x = x_offset
        self.canvas_offset_y = y_offset
        self.display_scale = scale
        
        self.canvas.delete("all")
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.photo_image)
        
        # Only draw UI overlays in normal mode
        if not self.show_mask_mode:
            # Draw existing lines (if visible)
            if self.show_lines:
                self.draw_lines_on_canvas()
            
            # Draw locked first point if it exists
            if self.first_point is not None:
                self.draw_point_on_canvas(self.first_point, "lime", 8)
            
            # Draw the point being currently dragged
            if self.current_point is not None:
                if self.first_point is None:
                    # Dragging first point
                    self.draw_point_on_canvas(self.current_point, "yellow", 8)
                else:
                    # Dragging second point - also show preview line
                    self.draw_point_on_canvas(self.current_point, "red", 8)
                    canvas_x1, canvas_y1 = self.image_to_canvas_coords(self.first_point[0], self.first_point[1])
                    canvas_x2, canvas_y2 = self.image_to_canvas_coords(self.current_point[0], self.current_point[1])
                    self.canvas.create_line(
                        canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                        fill="yellow", width=2, dash=(4, 4)
                    )

    def on_resize(self, event):
        # Debounce or just redraw
        if self.current_patch_info:
            self.draw_image()
    
    def update_statistics(self):
        """Update the statistics labels."""
        total_samples = len(self.db.samples)
        labeled_samples = sum(1 for sample in self.db.samples if len(sample.lines) > 0)
        total_lines = sum(len(sample.lines) for sample in self.db.samples)
        
        self.lbl_total_samples.config(text=f"Total samples: {total_samples}")
        self.lbl_labeled_samples.config(text=f"Labeled samples: {labeled_samples}")
        self.lbl_total_lines.config(text=f"Total lines: {total_lines}")
    
    def update_lines_list(self):
        """Update the lines listbox with current lines."""
        self.lines_listbox.delete(0, tk.END)
        
        if not self.lines:
            self.lines_listbox.insert(tk.END, "No lines yet")
        else:
            for i, line in enumerate(self.lines):
                p1, p2 = line
                text = f"Line {i+1}: ({p1[0]:.1f}, {p1[1]:.1f}) → ({p2[0]:.1f}, {p2[1]:.1f})"
                self.lines_listbox.insert(tk.END, text)
            
            # Select the currently selected line in the listbox
            if self.selected_line_idx is not None and self.selected_line_idx < len(self.lines):
                self.lines_listbox.selection_clear(0, tk.END)
                self.lines_listbox.selection_set(self.selected_line_idx)
                self.lines_listbox.see(self.selected_line_idx)
    
    def toggle_mask(self):
        """Toggle between normal view and mask view."""
        self.show_mask_mode = not self.show_mask_mode
        
        # Update button text
        if self.show_mask_mode:
            self.btn_toggle_mask.config(text="Show Image (M)")
        else:
            self.btn_toggle_mask.config(text="Show Mask (M)")
        
        # Redraw
        self.draw_image()
    
    def toggle_lines(self):
        """Toggle visibility of lines on the canvas."""
        self.show_lines = not self.show_lines
        self.draw_image()
    
    def on_line_select(self, event):
        """Handle selection of a line from the listbox."""
        selection = self.lines_listbox.curselection()
        if selection:
            # selection is a tuple of indices
            self.selected_line_idx = selection[0]
            self.draw_image()
    
    def delete_selected_line(self):
        """Delete the currently selected line."""
        if self.selected_line_idx is not None and 0 <= self.selected_line_idx < len(self.lines):
            del self.lines[self.selected_line_idx]
            self.selected_line_idx = None
            self._save_current_lines()  # Save after deletion
            self.update_statistics()  # Update stats after deletion
            self.update_lines_list()
            self.draw_image()
    
    def save_annotations(self, show_message=True):
        """Save all annotations to file."""
        self._save_current_lines()  # Save current work
        self.db.save("line_annotations.json")
        if show_message:
            print("Annotations saved to line_annotations.json")
    
    def on_close(self):
        """Handle window close event."""
        self.save_annotations(show_message=False)
        self.root.destroy()
    
    # Data Generation Methods
    
    def generate_training_patches(self):
        """Generate 16 random 128x128 patches from labeled samples for 4x4 grid."""
        self.generated_patches = []
        
        # Only use samples with lines
        labeled_samples = [s for s in self.db.samples if len(s.lines) > 0]
        
        if not labeled_samples:
            messagebox.showwarning("No Labeled Data", "No labeled samples found. Please label some data first.")
            return
        
        print(f"Generating 16 patches from {len(labeled_samples)} labeled samples...")
        
        for i in range(16):
            # Pick random labeled sample
            sample = random.choice(labeled_samples)
            
            # Load the frame
            cap = cv2.VideoCapture(sample.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                print(f"Warning: Could not load frame for patch {i+1}")
                continue
            
            # Extract the original crop
            x, y, w, h = sample.crop_rect
            crop = frame[y:y+h, x:x+w]
            
            # Apply random augmentations
            # Random zoom (±5%)
            zoom_factor = random.uniform(0.95, 1.05)
            
            # Random rotation (±180 degrees)
            rotation_angle = random.uniform(-180, 180)
            
            # Random stretch in x and y
            stretch_x = random.uniform(0.9, 1.1)
            stretch_y = random.uniform(0.9, 1.1)
            
            # To ensure the final 128x128 patch is fully filled after transformation,
            # we need to sample from a larger region initially
            # The required size depends on rotation and zoom
            # Worst case: 45° rotation requires sqrt(2) * size, plus zoom/stretch
            buffer_factor = 1.8  # Conservative factor to ensure full coverage
            initial_size = int(128 * buffer_factor)
            
            # Random location for the larger initial patch
            if w < initial_size or h < initial_size:
                # If crop is too small, work with what we have
                patch_x, patch_y = 0, 0
                patch_w, patch_h = w, h
            else:
                patch_x = random.randint(0, w - initial_size)
                patch_y = random.randint(0, h - initial_size)
                patch_w, patch_h = initial_size, initial_size
            
            # Extract larger patch
            large_patch = crop[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
            large_patch_rgb = cv2.cvtColor(large_patch, cv2.COLOR_BGR2RGB)
            
            # Calculate center of large patch
            center_x, center_y = patch_w / 2, patch_h / 2
            
            # Build transformation matrix
            M_center = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]], dtype=np.float32)
            
            # Rotation matrix
            rad = np.deg2rad(rotation_angle)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)
            M_rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
            
            # Scale/stretch/zoom matrix
            scale_x = zoom_factor * stretch_x
            scale_y = zoom_factor * stretch_y
            M_scale = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float32)
            
            # Translate back
            M_back = np.array([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]], dtype=np.float32)
            
            # Combine transformations
            M_combined = M_back @ M_scale @ M_rot @ M_center
            M_2x3 = M_combined[:2, :]
            
            # Apply transformation to large image
            transformed = cv2.warpAffine(large_patch_rgb, M_2x3, (patch_w, patch_h), 
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            
            # Crop center 128x128 region from transformed image
            crop_x = (patch_w - 128) // 2
            crop_y = (patch_h - 128) // 2
            patch_img_aug = transformed[crop_y:crop_y+128, crop_x:crop_x+128]
            
            # Create mask with same transformation
            mask_large = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
            
            # Draw lines on large mask with transformation
            for line in sample.lines:
                # Line coords are relative to the original crop
                p1_x, p1_y = line.start
                p2_x, p2_y = line.end
                
                # Translate to large patch coordinates
                p1_patch = np.array([p1_x - patch_x, p1_y - patch_y, 1], dtype=np.float32)
                p2_patch = np.array([p2_x - patch_x, p2_y - patch_y, 1], dtype=np.float32)
                
                # Apply transformation
                p1_transformed = M_combined @ p1_patch
                p2_transformed = M_combined @ p2_patch
                
                # Draw transformed line on large mask
                p1_final = (int(p1_transformed[0]), int(p1_transformed[1]))
                p2_final = (int(p2_transformed[0]), int(p2_transformed[1]))
                
                cv2.line(mask_large, p1_final, p2_final, (255, 255, 255), thickness=3)
            
            # Crop center 128x128 region from mask
            mask = mask_large[crop_y:crop_y+128, crop_x:crop_x+128]
            
            patch_data = {
                "image": patch_img_aug,
                "mask": mask,
                "source_video": os.path.basename(sample.video_path),
                "source_frame": sample.frame_idx,
                "source_crop": sample.crop_rect,
                "patch_offset": (patch_x, patch_y, patch_w, patch_h)
            }
            
            self.generated_patches.append(patch_data)
        
        print(f"Generated {len(self.generated_patches)} patches")
        
        if self.generated_patches:
            self.show_gen_mask = False
            self.display_gen_grid()
    
    def display_gen_grid(self):
        """Display all generated patches in a 4x4 grid with padding."""
        if not self.generated_patches:
            return
        
        # Update info
        self.lbl_gen_count.config(text=f"Patches: {len(self.generated_patches)}")
        
        # Update button text
        if self.show_gen_mask:
            self.btn_toggle_gen_view.config(text="Show: Masks")
        else:
            self.btn_toggle_gen_view.config(text="Show: Images")
        
        # Create 4x4 grid with padding
        grid_rows = 4
        grid_cols = 4
        patch_size = 128
        padding = 4  # Padding between patches
        
        # Calculate grid dimensions with padding
        grid_width = grid_cols * patch_size + (grid_cols + 1) * padding
        grid_height = grid_rows * patch_size + (grid_rows + 1) * padding
        
        # Create the grid image with dark background
        grid_img = np.full((grid_height, grid_width, 3), 32, dtype=np.uint8)
        
        for idx, patch in enumerate(self.generated_patches):
            if idx >= 16:
                break
            
            row = idx // grid_cols
            col = idx % grid_cols
            
            # Choose image or mask
            if self.show_gen_mask:
                patch_data = patch["mask"]
            else:
                patch_data = patch["image"]
            
            # Get patch dimensions
            ph, pw = patch_data.shape[:2]
            
            # Calculate position with padding
            y_start = padding + row * (patch_size + padding)
            x_start = padding + col * (patch_size + padding)
            grid_img[y_start:y_start+ph, x_start:x_start+pw] = patch_data
        
        # Display the grid
        self._draw_generated_image(grid_img)
    
    def _draw_generated_image(self, img_arr):
        """Draw a generated patch on the generation canvas."""
        img_h, img_w = img_arr.shape[:2]
        
        canvas_w = self.gen_canvas.winfo_width()
        canvas_h = self.gen_canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            self.root.after(100, lambda: self._draw_generated_image(img_arr))
            return
        
        # Scale to fit canvas while maintaining aspect ratio
        scale_w = canvas_w * 0.8 / img_w
        scale_h = canvas_h * 0.8 / img_h
        scale = min(scale_w, scale_h)
        
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        # Resize
        if scale > 1.5:
            interp = cv2.INTER_NEAREST
        else:
            interp = cv2.INTER_LINEAR
        
        resized = cv2.resize(img_arr, (new_w, new_h), interpolation=interp)
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(resized)
        self.gen_photo_image = ImageTk.PhotoImage(img_pil)
        
        # Center on canvas
        offset_x = (canvas_w - new_w) // 2
        offset_y = (canvas_h - new_h) // 2
        
        # Clear and draw
        self.gen_canvas.delete("all")
        self.gen_canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.gen_photo_image)
    
    def toggle_generation_view(self):
        """Toggle between images and masks view in data generation."""
        self.show_gen_mask = not self.show_gen_mask
        self.display_gen_grid()
    
    def on_regenerate_key(self):
        """Handle 'r' key press - regenerate if in data generation tab, otherwise do nothing."""
        # Check which tab is active
        current_tab = self.tab_control.index(self.tab_control.select())
        if current_tab == 1:  # Data Generation tab (index 1)
            self.generate_training_patches()
    
    def on_toggle_mask_key(self):
        """Handle 'm' key press - toggle mask based on active tab."""
        # Check which tab is active
        current_tab = self.tab_control.index(self.tab_control.select())
        if current_tab == 0:  # Labelling tab
            self.toggle_mask()
        elif current_tab == 1:  # Data Generation tab
            self.toggle_generation_view()
    
    # Training Methods
    
    def log_training(self, message):
        """Add message to training log."""
        self.train_log.config(state='normal')
        self.train_log.insert(tk.END, message + '\n')
        self.train_log.see(tk.END)
        self.train_log.config(state='disabled')
    
    def generate_training_samples(self, num_samples):
        """Generate augmented training samples (optimized)."""
        samples = []
        labeled_samples = [s for s in self.db.samples if len(s.lines) > 0]
        
        if not labeled_samples:
            return samples
        
        self.log_training(f"Generating {num_samples} training samples from {len(labeled_samples)} labeled regions...")
        
        # Pre-load all frames into memory for speed
        frame_cache = {}
        self.log_training("Pre-loading frames...")
        for sample in labeled_samples:
            if sample.video_path not in frame_cache:
                cap = cv2.VideoCapture(sample.video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_idx)
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    frame_cache[sample.video_path] = (sample.frame_idx, frame)
        
        self.log_training(f"Cached frames, generating {num_samples} samples...")
        
        buffer_factor = 1.8
        initial_size = int(128 * buffer_factor)
        
        for i in range(num_samples):
            if i % 500 == 0:
                progress = (i / num_samples) * 50
                self.progress_var.set(progress)
                self.root.update_idletasks()
            
            sample = random.choice(labeled_samples)
            
            # Get cached frame
            if sample.video_path not in frame_cache:
                continue
            
            _, frame = frame_cache[sample.video_path]
            
            x, y, w, h = sample.crop_rect
            crop = frame[y:y+h, x:x+w]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Random augmentation parameters
            zoom_factor = random.uniform(0.95, 1.05)
            rotation_angle = random.uniform(-180, 180)
            stretch_x = random.uniform(0.9, 1.1)
            stretch_y = random.uniform(0.9, 1.1)
            
            if w < initial_size or h < initial_size:
                patch_x, patch_y = 0, 0
                patch_w, patch_h = w, h
            else:
                patch_x = random.randint(0, w - initial_size)
                patch_y = random.randint(0, h - initial_size)
                patch_w, patch_h = initial_size, initial_size
            
            large_patch = crop_rgb[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
            
            # Build transformation matrix
            center_x, center_y = patch_w / 2, patch_h / 2
            
            rad = np.deg2rad(rotation_angle)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)
            
            scale_x = zoom_factor * stretch_x
            scale_y = zoom_factor * stretch_y
            
            # Combined transform: translate → rotate → scale → translate back
            M = np.array([
                [scale_x * cos_a, -scale_x * sin_a, center_x - scale_x * cos_a * center_x + scale_x * sin_a * center_y],
                [scale_y * sin_a, scale_y * cos_a, center_y - scale_y * sin_a * center_x - scale_y * cos_a * center_y]
            ], dtype=np.float32)
            
            transformed = cv2.warpAffine(large_patch, M, (patch_w, patch_h), 
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            
            crop_x = (patch_w - 128) // 2
            crop_y = (patch_h - 128) // 2
            patch_img_aug = transformed[crop_y:crop_y+128, crop_x:crop_x+128]
            
            # Create mask
            mask_large = np.zeros((patch_h, patch_w), dtype=np.uint8)
            
            for line in sample.lines:
                p1_x = line.start[0] - patch_x
                p1_y = line.start[1] - patch_y
                p2_x = line.end[0] - patch_x
                p2_y = line.end[1] - patch_y
                
                # Apply transformation directly to line points
                p1_t = np.array([M[0, 0] * p1_x + M[0, 1] * p1_y + M[0, 2], 
                                 M[1, 0] * p1_x + M[1, 1] * p1_y + M[1, 2]], dtype=np.int32)
                p2_t = np.array([M[0, 0] * p2_x + M[0, 1] * p2_y + M[0, 2],
                                 M[1, 0] * p2_x + M[1, 1] * p2_y + M[1, 2]], dtype=np.int32)
                
                cv2.line(mask_large, tuple(p1_t), tuple(p2_t), 255, thickness=3)
            
            mask = mask_large[crop_y:crop_y+128, crop_x:crop_x+128]
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            
            samples.append({'image': patch_img_aug, 'mask': mask_rgb})
        
        self.log_training(f"Generated {len(samples)} samples")
        return samples
    
    def train_model_thread(self, num_samples, epochs, batch_size, lr):
        """Training thread function."""
        try:
            # Generate samples
            all_samples = self.generate_training_samples(num_samples)
            
            if len(all_samples) < 10:
                self.log_training("Error: Not enough samples generated")
                self.is_training = False
                return
            
            # Split into train/val
            split_idx = int(len(all_samples) * 0.8)
            train_samples = all_samples[:split_idx]
            val_samples = all_samples[split_idx:]
            
            self.log_training(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
            
            # Create datasets
            train_dataset = LineDataset(train_samples)
            val_dataset = LineDataset(val_samples)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model with pretrained MobileNetV2 backbone
            self.log_training("Loading MobileNetV2 backbone (pretrained on ImageNet)...")
            self.model = MobileUNet(pretrained=True).to(self.device)
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            criterion = nn.BCELoss()
            
            self.log_training("Starting training...")
            
            best_val_loss = float('inf')
            
            for epoch in range(epochs):
                if not self.is_training:
                    self.log_training("Training cancelled")
                    break
                
                # Train
                self.model.train()
                train_loss = 0
                for images, masks in train_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validate
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for images, masks in val_loader:
                        images = images.to(self.device)
                        masks = masks.to(self.device)
                        outputs = self.model(images)
                        loss = criterion(outputs, masks)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # Update UI
                progress = 50 + (epoch / epochs) * 50
                self.progress_var.set(progress)
                self.lbl_train_loss.config(text=f"Train Loss: {train_loss:.4f}")
                self.lbl_val_loss.config(text=f"Val Loss: {val_loss:.4f}")
                
                log_msg = f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    log_msg += " (Best)"
                
                self.log_training(log_msg)
                self.root.update_idletasks()
            
            self.log_training("Training completed!")
            self.lbl_model_status.config(text=f"Model: Trained ({epochs} epochs)")
            
        except Exception as e:
            self.log_training(f"Error: {str(e)}")
        finally:
            self.is_training = False
            self.btn_train.config(text="Start Training", state='normal')
            self.lbl_train_status.config(text="Training finished")
    
    def start_training(self):
        """Start training in a separate thread."""
        if self.is_training:
            self.log_training("Already training...")
            return
        
        labeled_samples = [s for s in self.db.samples if len(s.lines) > 0]
        if not labeled_samples:
            messagebox.showwarning("No Data", "No labeled samples found. Please label some data first.")
            return
        
        self.is_training = True
        self.btn_train.config(text="Training...", state='disabled')
        self.lbl_train_status.config(text="Generating training data...")
        self.progress_var.set(0)
        
        num_samples = self.train_samples_var.get()
        epochs = self.epochs_var.get()
        batch_size = self.batch_size_var.get()
        lr = self.lr_var.get()
        
        # Start training in thread
        thread = threading.Thread(target=self.train_model_thread, 
                                  args=(num_samples, epochs, batch_size, lr))
        thread.daemon = True
        thread.start()
    
    def save_model(self):
        """Save the trained model."""
        if self.model is None:
            messagebox.showwarning("No Model", "No trained model to save.")
            return
        
        torch.save(self.model.state_dict(), "line_detector_unet.pth")
        self.log_training("Model saved to line_detector_unet.pth")
        messagebox.showinfo("Success", "Model saved successfully!")
    
    # Inference Methods
    
    def load_model_for_inference(self):
        """Load a trained model for inference."""
        try:
            if self.model is None:
                self.model = MobileUNet(pretrained=False).to(self.device)
            
            self.model.load_state_dict(torch.load("line_detector_unet.pth", map_location=self.device))
            self.model.eval()
            self.lbl_inference_model.config(text="Model: Loaded ✓", foreground="green")
            messagebox.showinfo("Success", "Model loaded successfully!")
        except FileNotFoundError:
            messagebox.showerror("Error", "Model file 'line_detector_unet.pth' not found.\nPlease train and save a model first.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def generate_inference_samples(self):
        """Generate random test samples from any video frames."""
        if self.model is None:
            messagebox.showwarning("No Model", "Please load a model first.")
            return
        
        num_samples = self.inference_samples_var.get()
        self.inference_samples = []
        
        print(f"Generating {num_samples} test samples...")
        
        for i in range(num_samples):
            # Get random video and frame
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
            
            # Random 128x128 crop
            if h < 128 or w < 128:
                continue
            
            x = random.randint(0, w - 128)
            y = random.randint(0, h - 128)
            
            patch = frame[y:y+128, x:x+128]
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            
            # Run inference
            with torch.no_grad():
                # Prepare input
                input_tensor = torch.from_numpy(patch_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                input_tensor = input_tensor.to(self.device)
                
                # Predict
                prediction = self.model(input_tensor)
                pred_mask = (prediction.squeeze().cpu().numpy() * 255).astype(np.uint8)
                pred_mask_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
            
            self.inference_samples.append({
                'image': patch_rgb,
                'prediction': pred_mask_rgb,
                'source_video': os.path.basename(video_path),
                'source_frame': frame_idx,
                'location': (x, y)
            })
        
        print(f"Generated {len(self.inference_samples)} test samples")
        
        if self.inference_samples:
            self.inference_idx = 0
            self.show_inference_prediction = False
            self.display_inference_sample()
        else:
            messagebox.showwarning("No Samples", "Failed to generate test samples.")
    
    def display_inference_sample(self):
        """Display all inference samples in a 4x4 grid."""
        if not self.inference_samples:
            return
        
        # Update info
        self.lbl_inference_idx.config(text=f"Samples: {len(self.inference_samples)}")
        
        # Update button text
        if self.show_inference_prediction:
            self.btn_toggle_inference.config(text="Show: Predictions")
        else:
            self.btn_toggle_inference.config(text="Show: Images")
        
        # Create 4x4 grid
        grid_rows = 4
        grid_cols = 4
        patch_size = 128
        padding = 4
        
        grid_width = grid_cols * patch_size + (grid_cols + 1) * padding
        grid_height = grid_rows * patch_size + (grid_rows + 1) * padding
        
        grid_img = np.full((grid_height, grid_width, 3), 32, dtype=np.uint8)
        
        for idx, sample in enumerate(self.inference_samples):
            if idx >= 16:
                break
            
            row = idx // grid_cols
            col = idx % grid_cols
            
            # Choose image or prediction
            if self.show_inference_prediction:
                patch_data = sample['prediction']
            else:
                patch_data = sample['image']
            
            ph, pw = patch_data.shape[:2]
            
            # Place in grid
            y_start = padding + row * (patch_size + padding)
            x_start = padding + col * (patch_size + padding)
            grid_img[y_start:y_start+ph, x_start:x_start+pw] = patch_data
        
        # Display the grid
        self._draw_inference_image(grid_img)
    
    def _draw_inference_image(self, img_arr):
        """Draw inference image on canvas."""
        img_h, img_w = img_arr.shape[:2]
        
        canvas_w = self.inference_canvas.winfo_width()
        canvas_h = self.inference_canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            self.root.after(100, lambda: self._draw_inference_image(img_arr))
            return
        
        # Scale to fit
        scale_w = canvas_w * 0.8 / img_w
        scale_h = canvas_h * 0.8 / img_h
        scale = min(scale_w, scale_h)
        
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        if scale > 1.5:
            interp = cv2.INTER_NEAREST
        else:
            interp = cv2.INTER_LINEAR
        
        resized = cv2.resize(img_arr, (new_w, new_h), interpolation=interp)
        
        img_pil = Image.fromarray(resized)
        self.inference_photo_image = ImageTk.PhotoImage(img_pil)
        
        offset_x = (canvas_w - new_w) // 2
        offset_y = (canvas_h - new_h) // 2
        
        self.inference_canvas.delete("all")
        self.inference_canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.inference_photo_image)
    
    def toggle_inference_view(self):
        """Toggle between images and predictions view."""
        self.show_inference_prediction = not self.show_inference_prediction
        self.display_inference_sample()
    
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
    
    def draw_point_on_canvas(self, point, color, size):
        """Draw a point on the canvas."""
        canvas_x, canvas_y = self.image_to_canvas_coords(point[0], point[1])
        self.canvas.create_oval(
            canvas_x - size, canvas_y - size,
            canvas_x + size, canvas_y + size,
            fill=color, outline=color
        )
    
    def draw_lines_on_canvas(self):
        """Draw all placed lines on the canvas."""
        for line_idx, line in enumerate(self.lines):
            p1, p2 = line
            canvas_x1, canvas_y1 = self.image_to_canvas_coords(p1[0], p1[1])
            canvas_x2, canvas_y2 = self.image_to_canvas_coords(p2[0], p2[1])
            
            # Check if this line is being edited or selected
            is_editing = (self.editing_point is not None and 
                         self.editing_point[0] == line_idx)
            is_selected = (self.selected_line_idx == line_idx)
            
            if is_editing:
                # Draw yellow dotted line while editing
                line_width = 2
                self.canvas.create_line(
                    canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                    fill="yellow", width=line_width, dash=(4, 4)
                )
                # Draw endpoints with same width as line
                self.draw_point_on_canvas(p1, "yellow", line_width)
                self.draw_point_on_canvas(p2, "yellow", line_width)
            elif is_selected:
                # Draw green line for selected
                line_width = 3
                self.canvas.create_line(
                    canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                    fill="lime", width=line_width
                )
                # Draw endpoints with same width as line
                self.draw_point_on_canvas(p1, "lime", line_width)
                self.draw_point_on_canvas(p2, "lime", line_width)
            else:
                # Draw solid cyan line normally
                line_width = 2
                self.canvas.create_line(
                    canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                    fill="cyan", width=line_width
                )
                # Draw endpoints with same width as line
                self.draw_point_on_canvas(p1, "cyan", line_width)
                self.draw_point_on_canvas(p2, "cyan", line_width)
    
    def on_canvas_press(self, event):
        """Handle mouse press on canvas - start dragging a new point or edit existing."""
        if not self.current_patch_info or self.show_mask_mode:
            return
        
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        
        # Check if clicking near an existing point (within 15px in image coords)
        click_threshold = 15 / self.display_scale
        
        for line_idx, line in enumerate(self.lines):
            for point_idx, point in enumerate(line):
                dist = ((point[0] - img_x)**2 + (point[1] - img_y)**2)**0.5
                if dist < click_threshold:
                    # Start editing this point
                    self.is_dragging = True
                    self.editing_point = (line_idx, point_idx)
                    self.draw_image()
                    return
        
        # Not clicking on existing point - start new point/line
        self.is_dragging = True
        self.current_point = (img_x, img_y)
        self.draw_image()
    
    def on_canvas_drag(self, event):
        """Handle mouse drag on canvas - update the point being dragged."""
        if not self.is_dragging or not self.current_patch_info:
            return
        
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        
        if self.editing_point is not None:
            # Update existing point
            line_idx, point_idx = self.editing_point
            self.lines[line_idx][point_idx] = (img_x, img_y)
        else:
            # Update the current point being dragged (no clamping - allow points outside image)
            self.current_point = (img_x, img_y)
        
        self.draw_image()
    
    def on_canvas_release(self, event):
        """Handle mouse release on canvas - lock in the point."""
        if not self.is_dragging or not self.current_patch_info:
            return
        
        self.is_dragging = False
        
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        
        if self.editing_point is not None:
            # Finished editing existing point
            line_idx, point_idx = self.editing_point
            self.lines[line_idx][point_idx] = (img_x, img_y)
            self.selected_line_idx = line_idx  # Select the edited line
            self.editing_point = None
            self._save_current_lines()  # Save after editing
            self.update_lines_list()
        else:
            # No clamping - allow points outside image bounds
            final_point = (img_x, img_y)
            
            if self.first_point is None:
                # First point is now locked in
                self.first_point = final_point
                self.current_point = None
            else:
                # Second point - create the line
                self.lines.append([self.first_point, final_point])
                self.selected_line_idx = len(self.lines) - 1  # Select the newly created line
                self.first_point = None
                self.current_point = None
                self._save_current_lines()  # Save after creating new line
                # Update the lines list
                self.update_lines_list()
        
        self.draw_image()

def find_videos(input_paths):
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
    parser = argparse.ArgumentParser(description="View random 400x400 patches from videos.")
    parser.add_argument("videos", nargs="*", help="Video files or directories")
    args = parser.parse_args()
    
    video_inputs = args.videos
    if not video_inputs:
        # Default to 'videos' directory if it exists
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
    root.state('zoomed')  # Fullscreen on Windows
    
    app = RandomPatchViewer(root, video_paths)
    
    root.mainloop()

if __name__ == "__main__":
    main()
