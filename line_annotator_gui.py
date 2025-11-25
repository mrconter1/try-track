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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import Intel Extension for PyTorch (IPEX)
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

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
        # ImageNet normalization statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image'].astype(np.float32) / 255.0
        mask = sample['mask'][:, :, 0].astype(np.float32) / 255.0
        
        # Convert to tensors (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Apply ImageNet normalization for pretrained MobileNetV2
        mean = torch.from_numpy(self.mean).reshape(3, 1, 1)
        std = torch.from_numpy(self.std).reshape(3, 1, 1)
        image = (image - mean) / std
        
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
    def __init__(self, root, video_paths, patch_size=400, test_graph=False):
        self.root = root
        self.video_paths = [os.path.abspath(p) for p in video_paths]
        self.patch_size = patch_size
        self.test_graph = test_graph
        
        # Pre-calculate frame counts for proportional sampling (parallel)
        self.video_frame_counts = {}
        print("Scanning videos (parallel)...")
        
        # Use ThreadPoolExecutor to scan videos in parallel
        max_workers = min(8, len(self.video_paths))  # Use up to 8 threads
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
        
        # Undo/Redo stack
        self.undo_stack = []  # Stack of states for undo
        self.current_sample_id = None  # To track sample changes
        
        # Data generation state
        self.generated_patches = []  # List of {image, mask, source_info}
        self.gen_patch_idx = -1
        self.show_gen_mask = False  # Toggle between image and mask in gen tab
        
        # Training state
        self.model = None
        # Device priority: CUDA > Intel XPU (IPEX) > CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif IPEX_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available():
            self.device = torch.device('xpu')
        else:
            self.device = torch.device('cpu')
        self.is_training = False
        self.training_samples = []
        self.loss_history = {'train': [], 'val': []}
        
        # Populate mock data if test_graph flag is set
        if self.test_graph:
            import math
            # Generate mock training loss data (decaying curve with noise)
            for i in range(500):
                epoch = i / 10.0  # 0 to 50 epochs
                base_loss = 0.8 * math.exp(-epoch / 15) + 0.05
                noise = random.uniform(-0.02, 0.02)
                self.loss_history['train'].append((epoch, max(0.01, base_loss + noise)))
            # Generate mock validation loss (sampled less frequently, slightly higher)
            for i in range(50):
                epoch = float(i + 1)
                base_loss = 0.85 * math.exp(-epoch / 15) + 0.08
                noise = random.uniform(-0.03, 0.03)
                self.loss_history['val'].append((epoch, max(0.01, base_loss + noise)))
        
        # Inference state
        self.inference_samples = []
        self.inference_idx = 0
        self.show_inference_prediction = False
        
        # UI Setup
        self.root.title("LineAnnotatorGUI")
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
        self.root.bind("<Control-z>", lambda e: self.undo_last_action())
        self.root.bind("<r>", lambda e: self.on_r_key())
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
        
        # Generate initial training patches for Data Generation tab
        self.root.after(500, self._generate_initial_training_patches)

    def _load_sample_image(self, sample):
        """Helper to load a single sample image."""
        cap = cv2.VideoCapture(sample.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            x, y, w, h = sample.crop_rect
            patch = frame[y:y+h, x:x+w]
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            
            return {
                "video_path": sample.video_path,
                "frame_idx": sample.frame_idx,
                "crop_rect": sample.crop_rect,
                "image": patch_rgb
            }
        return None

    def _load_history_from_db(self):
        """Reconstruct history from saved samples in database (parallel)."""
        if not self.db.samples:
            return

        print(f"Loading {len(self.db.samples)} history samples...")
        
        # Use ThreadPoolExecutor to load samples in parallel
        max_workers = min(8, len(self.db.samples))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._load_sample_image, sample) for sample in self.db.samples]
            
            # Collect results in order
            for future in futures:
                try:
                    patch_info = future.result()
                    if patch_info:
                        self.history.append(patch_info)
                except Exception as e:
                    print(f"Warning: Error loading sample: {e}")
        
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
        
        btn_resample = ttk.Button(nav_frame, text="Resample (R)", command=self.resample_current_patch)
        btn_resample.pack(fill=tk.X, pady=5)
        
        self.btn_toggle_mask = ttk.Button(nav_frame, text="Show Mask (M)", command=self.toggle_mask)
        self.btn_toggle_mask.pack(fill=tk.X, pady=5)
        
        # Sample management
        sample_frame = ttk.LabelFrame(sidebar, text="Sample Management", padding=10)
        sample_frame.pack(fill=tk.X, pady=(0, 20))
        
        btn_delete_sample = ttk.Button(sample_frame, text="Delete Sample", command=self.delete_current_sample)
        btn_delete_sample.pack(fill=tk.X, pady=5)

    
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
        
    
    def _build_training_tab(self):
        """Build the UI for the training tab."""
        # Main frame
        main_frame = ttk.Frame(self.training_tab, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top container for Controls + Graph
        top_container = ttk.Frame(main_frame)
        top_container.pack(fill=tk.X, pady=(0, 10))
        
        # Training controls (Left)
        controls_frame = ttk.LabelFrame(top_container, text="Training Configuration", padding=10)
        controls_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        # Number of samples
        ttk.Label(controls_frame, text="Training Samples:").grid(row=0, column=0, sticky="w", pady=5)
        self.train_samples_var = tk.IntVar(value=10000)
        samples_spinbox = ttk.Spinbox(controls_frame, from_=100, to=100000, increment=1000, 
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
        self.batch_size_var = tk.IntVar(value=32)
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
        
        # Graph (Right)
        graph_frame = ttk.LabelFrame(top_container, text="Validation Loss", padding=10)
        graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.loss_canvas = tk.Canvas(graph_frame, bg="white", height=200)
        self.loss_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind resize to redraw graph
        self.loss_canvas.bind("<Configure>", lambda e: self.draw_loss_graph())
        
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

    def draw_loss_graph(self):
        """Draw the validation loss graph."""
        if not hasattr(self, 'loss_canvas'):
            return
            
        canvas = self.loss_canvas
        canvas.delete("all")
        
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        
        if w < 10 or h < 10:
            return
            
        padding_left = 55  # Extra space for y-axis labels
        padding_right = 15
        padding_top = 25
        padding_bottom = 40  # Extra space for x-axis labels
        graph_w = w - padding_left - padding_right
        graph_h = h - padding_top - padding_bottom
        
        # Draw axes
        canvas.create_line(padding_left, h - padding_bottom, w - padding_right, h - padding_bottom, fill="black", width=2)  # X axis
        canvas.create_line(padding_left, h - padding_bottom, padding_left, padding_top, fill="black", width=2)  # Y axis
        
        val_loss = self.loss_history['val']
        
        if not val_loss:
            canvas.create_text(w/2, h/2, text="No validation data yet", fill="gray")
            return
            
        # Find max loss for scaling
        max_val = max([x[1] for x in val_loss]) if val_loss else 0
        max_loss = max_val
        
        if max_loss == 0: max_loss = 1.0
        max_loss = max_loss * 1.1  # Add some headroom
        
        # Determine the maximum epoch reached so far in the data
        max_epoch_reached = 0
        if val_loss:
            max_epoch_reached = max(max_epoch_reached, val_loss[-1][0])
            
        # Ensure we have at least some range to avoid division by zero
        if max_epoch_reached == 0:
            max_epoch_reached = 0.1

        def to_canvas(epoch_val, loss_val):
            x = padding_left + (epoch_val / max_epoch_reached) * graph_w
            y = h - padding_bottom - (loss_val / max_loss) * graph_h
            return x, y
        
        # Draw X-axis tick marks and labels (epochs)
        num_x_ticks = min(6, max(2, int(max_epoch_reached)))
        for i in range(num_x_ticks + 1):
            epoch_val = (i / num_x_ticks) * max_epoch_reached
            x, _ = to_canvas(epoch_val, 0)
            # Tick mark
            canvas.create_line(x, h - padding_bottom, x, h - padding_bottom + 5, fill="black", width=1)
            # Label
            label = f"{epoch_val:.0f}" if epoch_val >= 1 else f"{epoch_val:.1f}"
            canvas.create_text(x, h - padding_bottom + 12, text=label, fill="black", font=("TkDefaultFont", 8))
        
        # Draw Y-axis tick marks and labels (loss)
        num_y_ticks = 5
        for i in range(num_y_ticks + 1):
            loss_val = (i / num_y_ticks) * max_loss
            _, y = to_canvas(0, loss_val)
            # Tick mark
            canvas.create_line(padding_left - 4, y, padding_left, y, fill="black", width=1)
            # Label (use 2 decimals if loss is small)
            if max_loss < 1:
                label = f"{loss_val:.2f}"
            else:
                label = f"{loss_val:.1f}"
            canvas.create_text(padding_left - 6, y, text=label, fill="black", font=("TkDefaultFont", 8), anchor="e")
        
        # Draw val loss (Red)
        if val_loss:
            points_val = []
            for ep, loss in val_loss:
                x, y = to_canvas(ep, loss)
                points_val.append(x)
                points_val.append(y)
                # Draw point
                canvas.create_oval(x-2, y-2, x+2, y+2, fill="red", outline="red")
                
            if len(points_val) >= 4:
                canvas.create_line(points_val, fill="red", width=2, smooth=True)
        
        # Draw axis labels
        canvas.create_text((padding_left + w - padding_right) / 2, h - 8, text="Epoch", fill="black", font=("TkDefaultFont", 9))
        canvas.create_text(10, (padding_top + h - padding_bottom) / 2, text="Validation Loss", fill="black", font=("TkDefaultFont", 9), angle=90)
        
        # Show current values (latest validation loss and epoch)
        if val_loss:
            current_epoch = val_loss[-1][0]
            current_loss = val_loss[-1][1]
            status_text = f"Epoch: {current_epoch:.1f}  Val Loss: {current_loss:.4f}"
            canvas.create_text(w - padding_right, padding_top - 5, text=status_text, 
                             fill="black", font=("TkDefaultFont", 9, "bold"), anchor="ne")
    
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
        self.undo_stack = []  # Clear undo stack when switching samples
        
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
            # Draw each line as 1px wide white line
            for line in self.lines:
                p1, p2 = line
                x1, y1 = int(round(p1[0])), int(round(p1[1]))
                x2, y2 = int(round(p2[0])), int(round(p2[1]))
                cv2.line(display_arr, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
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
                self.draw_point_on_canvas(self.first_point, "lime", 2)
            
            # Draw the point being currently dragged
            if self.current_point is not None:
                if self.first_point is None:
                    # Dragging first point
                    self.draw_point_on_canvas(self.current_point, "yellow", 2)
                else:
                    # Dragging second point - also show preview line
                    self.draw_point_on_canvas(self.current_point, "red", 2)
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
        
        # Current sample number
        current_sample = self.history_idx + 1 if self.history_idx >= 0 else 0
        
        self.lbl_total_samples.config(text=f"Total samples: {current_sample} / {total_samples}")
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
            self._push_undo_state("Delete line")
            del self.lines[self.selected_line_idx]
            self.selected_line_idx = None
            self.save_annotations(show_message=False)  # Auto-save to disk
            self.update_statistics()  # Update stats after deletion
            self.update_lines_list()
            self.draw_image()
    
    def _push_undo_state(self, action_name=""):
        """Save current state to undo stack."""
        state = {
            'lines': [line.copy() for line in self.lines],  # Deep copy of lines
            'selected_line_idx': self.selected_line_idx,
            'action': action_name
        }
        self.undo_stack.append(state)
    
    def undo_last_action(self):
        """Undo the last action."""
        if not self.current_patch_info or not self.undo_stack:
            return
        
        # Pop the last state
        state = self.undo_stack.pop()
        
        # Restore state
        self.lines = [line.copy() for line in state['lines']]
        self.selected_line_idx = state['selected_line_idx']
        
        # Clear any in-progress drawing
        self.first_point = None
        self.current_point = None
        self.is_dragging = False
        self.editing_point = None
        
        self.save_annotations(show_message=False)  # Auto-save to disk
        self.update_statistics()
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
    
    def _generate_single_patch(self, sample, frame, include_visualization=False):
        """
        Generate a single augmented training patch + mask from a sample.
        
        Args:
            sample: Sample object with video_path, frame_idx, crop_rect, lines
            frame: The video frame (BGR format)
            include_visualization: If True, include extra data for visualization
            
        Returns:
            Dictionary with 'image' (128x128 RGB), 'mask' (128x128 RGB), 
            and optionally visualization data
        """
        # Extract the original crop
        x, y, w, h = sample.crop_rect
        crop = frame[y:y+h, x:x+w]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Random augmentation parameters
        zoom_factor = random.uniform(0.75, 1.25)
        rotation_angle = random.uniform(-180, 180)
        stretch_x = random.uniform(0.9, 1.1)
        stretch_y = random.uniform(0.9, 1.1)
        
        # Perspective augmentation - random corner displacement (as fraction of size)
        perspective_strength = 0.15  # Max displacement as fraction of patch size
        perspective_corners = [
            (random.uniform(-perspective_strength, perspective_strength),
             random.uniform(-perspective_strength, perspective_strength))
            for _ in range(4)
        ]
        
        # Flip augmentations
        flip_horizontal = random.random() < 0.5
        flip_vertical = random.random() < 0.5
        
        # Buffer factor to ensure 128x128 is fully filled after transformation
        # Increased to handle zoom + perspective + rotation
        buffer_factor = 2.5
        initial_size = int(128 * buffer_factor)
        
        # Random location for the larger initial patch
        if w < initial_size or h < initial_size:
            patch_x, patch_y = 0, 0
            patch_w, patch_h = w, h
        else:
            patch_x = random.randint(0, w - initial_size)
            patch_y = random.randint(0, h - initial_size)
            patch_w, patch_h = initial_size, initial_size
        
        # Extract larger patch from image
        large_patch = crop_rgb[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
        
        # Create mask for the large patch area (draw lines BEFORE transformation)
        mask_large = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
        for line in sample.lines:
            p1_x, p1_y = line.start
            p2_x, p2_y = line.end
            # Translate to large patch coordinates
            p1_patch = (int(p1_x - patch_x), int(p1_y - patch_y))
            p2_patch = (int(p2_x - patch_x), int(p2_y - patch_y))
            cv2.line(mask_large, p1_patch, p2_patch, (255, 255, 255), thickness=1)
        
        # Try to find a valid transformation (retry if source region goes out of bounds)
        max_attempts = 10
        for attempt in range(max_attempts):
            # Regenerate perspective for retries (keep other params)
            if attempt > 0:
                perspective_corners = [
                    (random.uniform(-perspective_strength, perspective_strength),
                     random.uniform(-perspective_strength, perspective_strength))
                    for _ in range(4)
                ]
            
            # Build transformation using homography (perspective) transform
            center_x, center_y = patch_w / 2, patch_h / 2
            
            rad = np.deg2rad(rotation_angle)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)
            
            scale_x = zoom_factor * stretch_x
            scale_y = zoom_factor * stretch_y
            
            # Define source corners (corners of the large patch)
            src_corners = np.array([
                [0, 0],
                [patch_w, 0],
                [patch_w, patch_h],
                [0, patch_h]
            ], dtype=np.float32)
            
            # Apply affine transform (rotation + scale) to get intermediate corners
            # Transform around center
            dst_corners = []
            for i, (sx, sy) in enumerate(src_corners):
                # Translate to center
                x = sx - center_x
                y = sy - center_y
                # Rotate
                xr = x * cos_a - y * sin_a
                yr = x * sin_a + y * cos_a
                # Scale
                xs = xr * scale_x
                ys = yr * scale_y
            # Translate back
                xf = xs + center_x
                yf = ys + center_y
                # Apply perspective displacement
                px, py = perspective_corners[i]
                xf += px * patch_w
                yf += py * patch_h
                dst_corners.append([xf, yf])
            
            dst_corners = np.array(dst_corners, dtype=np.float32)
            
            # Compute homography matrix from source to destination
            H = cv2.getPerspectiveTransform(src_corners, dst_corners)
            
            # Check if the center 128x128 output region maps to valid source pixels
            H_inverse = np.linalg.inv(H)
            crop_x_offset = (patch_w - 128) // 2
            crop_y_offset = (patch_h - 128) // 2
            
            # Check all 4 corners of the output region
            output_corners = np.array([
                [crop_x_offset, crop_y_offset],
                [crop_x_offset + 128, crop_y_offset],
                [crop_x_offset + 128, crop_y_offset + 128],
                [crop_x_offset, crop_y_offset + 128]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            source_corners_check = cv2.perspectiveTransform(output_corners, H_inverse).reshape(-1, 2)
            
            # Check if all source corners are within bounds (with small margin)
            margin = 2
            valid = True
            for sx, sy in source_corners_check:
                if sx < margin or sx > patch_w - margin or sy < margin or sy > patch_h - margin:
                    valid = False
                    break
            
            if valid:
                break
        
        # Apply the SAME perspective transformation to both image and mask
        transformed_img = cv2.warpPerspective(large_patch, H, (patch_w, patch_h), 
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        transformed_mask = cv2.warpPerspective(mask_large, H, (patch_w, patch_h), 
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            
        # Crop center 128x128 from both
        final_image = transformed_img[crop_y_offset:crop_y_offset+128, crop_x_offset:crop_x_offset+128]
        final_mask = transformed_mask[crop_y_offset:crop_y_offset+128, crop_x_offset:crop_x_offset+128]
        
        # Apply flip augmentations
        if flip_horizontal:
            final_image = cv2.flip(final_image, 1)  # 1 = horizontal flip
            final_mask = cv2.flip(final_mask, 1)
        if flip_vertical:
            final_image = cv2.flip(final_image, 0)  # 0 = vertical flip
            final_mask = cv2.flip(final_mask, 0)
        
        # Apply traditional image augmentations (to image only, not mask)
        final_image = final_image.astype(np.float32)
        
        # Brightness adjustment (±30%)
        brightness = random.uniform(-0.3, 0.3)
        final_image = final_image + brightness * 255
        
        # Contrast adjustment (0.7 to 1.3)
        contrast = random.uniform(0.7, 1.3)
        mean = np.mean(final_image)
        final_image = (final_image - mean) * contrast + mean
        
        # Gamma correction (0.7 to 1.5)
        gamma = random.uniform(0.7, 1.5)
        final_image = np.clip(final_image, 0, 255)
        final_image = 255.0 * np.power(final_image / 255.0, gamma)
        
        # Gaussian noise (σ = 0 to 25)
        noise_sigma = random.uniform(0, 25)
        if noise_sigma > 0:
            noise = np.random.normal(0, noise_sigma, final_image.shape)
            final_image = final_image + noise
        
        # Gaussian blur (σ = 0 to 1.5, apply with 50% probability)
        if random.random() < 0.5:
            blur_sigma = random.uniform(0.5, 1.5)
            final_image = cv2.GaussianBlur(final_image.astype(np.float32), (0, 0), blur_sigma)
        
        # Clip and convert back to uint8
        final_image = np.clip(final_image, 0, 255).astype(np.uint8)
        
        result = {
            'image': final_image,
            'mask': final_mask
        }
        
        if include_visualization:
            # Create source mask for visualization (full region, no transformation)
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
                'homography': H,  # Store the full homography matrix
                'step5_final': final_image
            })
        
        return result
    
    def _generate_initial_training_patches(self):
        """Generate initial training patches automatically on app start."""
        labeled_samples = [s for s in self.db.samples if len(s.lines) > 0]
        if not labeled_samples:
            return
        
        # Run generation in separate thread
        threading.Thread(target=self.generate_training_patches, daemon=True).start()
    
    def generate_training_patches(self):
        """Generate 16 random 128x128 patches from labeled samples for visualization."""
        # Update UI to show loading state (schedule on main thread)
        self.root.after(0, lambda: self.lbl_gen_count.config(text="Generating..."))
        
        self.generated_patches = []  # Clear existing
        new_patches = []
        
        # Only use samples with lines
        labeled_samples = [s for s in self.db.samples if len(s.lines) > 0]
        
        if not labeled_samples:
            self.root.after(0, lambda: messagebox.showwarning("No Labeled Data", "No labeled samples found. Please label some data first."))
            return
        
        print(f"Generating 16 patches from {len(labeled_samples)} labeled samples...")
        
        # Pre-load frames needed for generation
        # Key by (video_path, frame_idx) to handle multiple frames from same video
        frame_cache = {}
        
        # Select samples first to know which frames to load
        selected_samples = [random.choice(labeled_samples) for _ in range(16)]
        
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
            
            # Use unified function with visualization data
            patch_data = self._generate_single_patch(sample, frame, include_visualization=True)
            new_patches.append(patch_data)
        
        print(f"Generated {len(new_patches)} patches")
        
        # Update state and UI on main thread
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
        
        # Update info
        self.lbl_gen_count.config(text=f"Patches: {len(self.generated_patches)}")
        
        # Update button text
        if self.show_gen_mask:
            self.btn_toggle_gen_view.config(text="Show: Masks")
        else:
            self.btn_toggle_gen_view.config(text="Show: Step 5 Results")
        
        # Layout: 3x3 grid, each cell has [Full Region | Warped 128x128]
        num_rows = 3
        num_cols = 3
        region_size = 180  # Smaller to fit 3 columns
        patch_size = 128   # Training sample size
        padding = 6
        cell_spacing = 12  # Space between cells
        
        # Calculate cell dimensions
        cell_width = region_size + padding + patch_size
        cell_height = region_size
        
        # Calculate grid dimensions
        grid_width = num_cols * cell_width + (num_cols - 1) * cell_spacing + 2 * padding
        grid_height = num_rows * cell_height + (num_rows - 1) * cell_spacing + 2 * padding
        
        # Create the grid image with dark background
        grid_img = np.full((grid_height, grid_width, 3), 32, dtype=np.uint8)
        
        patch_idx = 0
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                if patch_idx >= len(self.generated_patches):
                    break
                
                patch = self.generated_patches[patch_idx]
                patch_idx += 1
                
                # Calculate cell position
                cell_x = padding + col_idx * (cell_width + cell_spacing)
                cell_y = padding + row_idx * (cell_height + cell_spacing)
                
                # ===== LEFT: Full source region with green highlight =====
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
                
                # Remember original size for scaling
                orig_h, orig_w = left_img.shape[:2]
                
                # Resize to region_size
                left_img = cv2.resize(left_img, (region_size, region_size), interpolation=cv2.INTER_LINEAR)
                
                # Draw green polygon or mask overlay depending on mode
                left_img_display = left_img.copy()
                
                if patch_offset and source_crop:
                    patch_x, patch_y, patch_w, patch_h = patch_offset
                    
                    # Get the homography matrix (or compute from params if not available)
                    H = patch.get("homography")
                    
                    if H is not None:
                        # Use inverse homography to map output corners back to source
                        H_inverse = np.linalg.inv(H)
                        
                        crop_x_offset = (patch_w - 128) // 2
                        crop_y_offset = (patch_h - 128) // 2
                        
                        # Output corners in homogeneous coordinates
                        output_corners = np.array([
                            [crop_x_offset, crop_y_offset],
                            [crop_x_offset + 128, crop_y_offset],
                            [crop_x_offset + 128, crop_y_offset + 128],
                            [crop_x_offset, crop_y_offset + 128]
                        ], dtype=np.float32).reshape(-1, 1, 2)
                        
                        # Apply inverse homography
                        source_corners = cv2.perspectiveTransform(output_corners, H_inverse).reshape(-1, 2)
                        
                        # Translate to full crop coordinates
                        source_corners[:, 0] += patch_x
                        source_corners[:, 1] += patch_y
                        
                        _, _, crop_w, crop_h = source_crop
                        scale_x_display = region_size / crop_w
                        scale_y_display = region_size / crop_h
                        
                        display_corners = source_corners.copy()
                        display_corners[:, 0] *= scale_x_display
                        display_corners[:, 1] *= scale_y_display
                        display_corners = display_corners.astype(np.int32)
                    else:
                        # Fallback if no homography available
                        display_corners = None
                    
                if self.show_gen_mask:
                    # Show source mask overlay on region
                    source_mask = patch.get("source_mask")
                    if source_mask is not None:
                        # Resize mask to match region_size
                        mask_resized = cv2.resize(source_mask, (region_size, region_size), interpolation=cv2.INTER_LINEAR)
                        # Blend mask with image (white lines on image)
                        mask_gray = cv2.cvtColor(mask_resized, cv2.COLOR_RGB2GRAY) if len(mask_resized.shape) == 3 else mask_resized
                        left_img_display[mask_gray > 128] = [255, 255, 255]
                    # Also draw the green polygon (now with perspective!)
                    if display_corners is not None:
                        cv2.polylines(left_img_display, [display_corners], isClosed=True, color=(0, 255, 0), thickness=2)
                else:
                    # Just draw green polygon (now with perspective!)
                    if display_corners is not None:
                        cv2.polylines(left_img_display, [display_corners], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Draw flip indicators
                step3_params = patch.get("step3_params", {})
                flip_h = step3_params.get("flip_h", False)
                flip_v = step3_params.get("flip_v", False)
                
                if flip_h or flip_v:
                    # Draw flip arrows/indicators at the center of the polygon
                    if display_corners is not None:
                        cx = int(np.mean(display_corners[:, 0]))
                        cy = int(np.mean(display_corners[:, 1]))
                        
                        if flip_h:
                            # Draw horizontal double arrow (↔)
                            cv2.arrowedLine(left_img_display, (cx - 15, cy - 10), (cx + 15, cy - 10), (255, 255, 0), 2, tipLength=0.3)
                            cv2.arrowedLine(left_img_display, (cx + 15, cy - 10), (cx - 15, cy - 10), (255, 255, 0), 2, tipLength=0.3)
                        
                        if flip_v:
                            # Draw vertical double arrow (↕)
                            cv2.arrowedLine(left_img_display, (cx, cy - 15 + (10 if flip_h else 0)), (cx, cy + 15 + (10 if flip_h else 0)), (255, 0, 255), 2, tipLength=0.3)
                            cv2.arrowedLine(left_img_display, (cx, cy + 15 + (10 if flip_h else 0)), (cx, cy - 15 + (10 if flip_h else 0)), (255, 0, 255), 2, tipLength=0.3)
                
                # Place region image
                grid_img[cell_y:cell_y+region_size, cell_x:cell_x+region_size] = left_img_display
                
                # ===== RIGHT: Final 128x128 training sample =====
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
                
                # Place patch image (centered vertically)
                right_x = cell_x + region_size + padding
                right_y_offset = (region_size - patch_size) // 2
                grid_img[cell_y + right_y_offset:cell_y + right_y_offset + patch_size,
                         right_x:right_x + patch_size] = right_img
        
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
    
    def on_r_key(self):
        """Handle 'r' key press - resample current in labelling tab, regenerate in data generation tab."""
        # Check which tab is active
        current_tab = self.tab_control.index(self.tab_control.select())
        if current_tab == 0:  # Labelling tab
            self.resample_current_patch()
        elif current_tab == 1:  # Data Generation tab (index 1)
            self.generate_training_patches()
    
    def delete_current_sample(self):
        """Delete the current sample from history and database."""
        if not self.current_patch_info or len(self.history) == 0:
            return
        
        # Remove from database if it exists there
        info = self.current_patch_info
        for i, sample in enumerate(self.db.samples):
            if (sample.video_path == info['video_path'] and 
                sample.frame_idx == info['frame_idx'] and 
                sample.crop_rect == info['crop_rect']):
                del self.db.samples[i]
                break
        
        # Remove from history
        if self.history_idx >= 0 and self.history_idx < len(self.history):
            del self.history[self.history_idx]
        
        # Navigate to appropriate sample
        if len(self.history) == 0:
            # No samples left, generate a new one
            self.history_idx = -1
            self.current_patch_info = None
            self.next_patch()
        elif self.history_idx >= len(self.history):
            # Was at end, go to new end
            self.history_idx = len(self.history) - 1
            self.current_patch_info = self.history[self.history_idx]
            self._load_current_lines()
            self.display_current_patch()
        else:
            # Stay at same index (now pointing to next sample)
            self.current_patch_info = self.history[self.history_idx]
            self._load_current_lines()
            self.display_current_patch()
        
        # Save changes
        self.save_annotations(show_message=False)
    
    def resample_current_patch(self):
        """Replace the current sample with a new random patch (to find difficult examples)."""
        if self.total_combined_frames == 0:
            return
        
        # Generate a new random patch
        new_info = self.generate_new_patch()
        if not new_info:
            return
        
        # Replace current position in history (or add if at end)
        if self.history_idx >= 0 and self.history_idx < len(self.history):
            # Replace the current sample
            self.history[self.history_idx] = new_info
        else:
            # Add new
            self.history.append(new_info)
            self.history_idx = len(self.history) - 1
        
        self.current_patch_info = new_info
        
        # Clear lines for new sample
        self.lines = []
        self.selected_line_idx = None
        self.first_point = None
        self.current_point = None
        self.editing_point = None
        self.undo_stack = []
        
        self.display_current_patch()
    
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
        """Generate augmented training samples using the unified generation function."""
        samples = []
        labeled_samples = [s for s in self.db.samples if len(s.lines) > 0]
        
        if not labeled_samples:
            return samples
        
        self.log_training(f"Generating {num_samples} training samples from {len(labeled_samples)} labeled regions...")
        
        # Pre-load all frames into memory for speed
        # Key by (video_path, frame_idx) to handle multiple frames from same video
        frame_cache = {}
        self.log_training("Pre-loading frames...")
        for sample in labeled_samples:
            cache_key = (sample.video_path, sample.frame_idx)
            if cache_key not in frame_cache:
                cap = cv2.VideoCapture(sample.video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_idx)
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    frame_cache[cache_key] = frame
        
        self.log_training(f"Cached {len(frame_cache)} frames, generating {num_samples} samples...")
        
        for i in range(num_samples):
            if i % 500 == 0:
                progress = (i / num_samples) * 50
                self.progress_var.set(progress)
                self.root.update_idletasks()
            
            sample = random.choice(labeled_samples)
            cache_key = (sample.video_path, sample.frame_idx)
            
            # Get cached frame
            if cache_key not in frame_cache:
                continue
            
            frame = frame_cache[cache_key]
            
            # Use unified function (without visualization data for speed)
            patch_data = self._generate_single_patch(sample, frame, include_visualization=False)
            samples.append(patch_data)
        
        self.log_training(f"Generated {len(samples)} samples")
        return samples
    
    def _dice_loss(self, pred, target, smooth=1e-8):
        """
        Compute Dice loss for thin line segmentation.
        Better than BCELoss alone for sparse targets.
        """
        intersection = (pred * target).sum()
        dice = 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return dice
    
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
            
            # Use parallel workers and pin_memory for faster data loading
            num_workers = 0 if self.device.type == 'cuda' else 0  # Windows compatibility
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)
            
            # Initialize model with pretrained MobileNetV2 backbone
            self.log_training("Loading MobileNetV2 backbone (pretrained on ImageNet)...")
            self.model = MobileUNet(pretrained=True).to(self.device)
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = nn.BCELoss()
            
            self.log_training("Using combined BCELoss (0.5) + DiceLoss (0.5)")
            self.log_training("Using CosineAnnealingLR scheduler + weight_decay=1e-4")
            
            self.log_training("Starting training...")
            
            best_val_loss = float('inf')
            
            for epoch in range(epochs):
                if not self.is_training:
                    self.log_training("Training cancelled")
                    break
                
                # Train
                self.model.train()
                train_loss = 0
                total_batches = len(train_loader)
                
                for batch_idx, (images, masks) in enumerate(train_loader):
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    
                    # Combined loss: BCELoss (0.5) + DiceLoss (0.5)
                    bce_loss = criterion(outputs, masks)
                    dice_loss = self._dice_loss(outputs, masks)
                    loss = 0.5 * bce_loss + 0.5 * dice_loss
                    
                    loss.backward()
                    
                    # Gradient clipping for stable training
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    loss_val = loss.item()
                    train_loss += loss_val
                    
                    # Update UI every 5 batches
                    if batch_idx % 5 == 0:
                        # Calculate progress: Base 50% + (current_epoch_progress / total_epochs) * 50%
                        # current_epoch_progress = epoch + (batch_idx / total_batches)
                        current_progress = 50 + ((epoch + (batch_idx / total_batches)) / epochs) * 50
                        self.progress_var.set(current_progress)
                        
                        # Show current batch loss
                        self.lbl_train_loss.config(text=f"Train Loss: {loss_val:.4f} (Batch {batch_idx}/{total_batches})")
                        
                        # Update status label with detailed progress
                        status_msg = f"Training Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{total_batches}"
                        self.lbl_train_status.config(text=status_msg)
                        
                        # Update graph with partial epoch data
                        # We store fractional epoch numbers for smooth plotting
                        current_epoch_frac = epoch + (batch_idx / total_batches)
                        self.loss_history['train'].append((current_epoch_frac, loss_val))
                        self.root.after(0, self.draw_loss_graph)
                        
                        self.root.update_idletasks()
                
                train_loss /= len(train_loader)
                
                # Validate
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for images, masks in val_loader:
                        images = images.to(self.device)
                        masks = masks.to(self.device)
                        outputs = self.model(images)
                        
                        # Same combined loss for validation
                        bce_loss = criterion(outputs, masks)
                        dice_loss = self._dice_loss(outputs, masks)
                        loss = 0.5 * bce_loss + 0.5 * dice_loss
                        
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # Step the learning rate scheduler
                scheduler.step()
                
                # Update UI
                progress = 50 + (epoch / epochs) * 50
                self.progress_var.set(progress)
                self.lbl_train_loss.config(text=f"Train Loss: {train_loss:.4f}")
                self.lbl_val_loss.config(text=f"Val Loss: {val_loss:.4f}")
                
                # Update graph history and redraw (Validation is plotted at integer epoch steps)
                # For training, we just ensure the final point of the epoch is added if not already
                # self.loss_history['train'].append((epoch + 1.0, train_loss)) 
                self.loss_history['val'].append((epoch + 1.0, val_loss))
                self.root.after(0, self.draw_loss_graph)
                
                log_msg = f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    log_msg += " (Best - Saved)"
                    # Auto-save best model
                    torch.save(self.model.state_dict(), "line_detector_unet_best.pth")
                
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
        
        # Reset graph
        self.loss_history = {'train': [], 'val': []}
        self.draw_loss_graph()
        
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
    
    def _fit_line_to_contour(self, contour):
        """
        Fit a line to a contour using cv2.fitLine.
        
        Args:
            contour: OpenCV contour
            
        Returns:
            Tuple (x1, y1, x2, y2) representing the fitted line
        """
        # Fit line to contour
        rows, cols = contour.shape[0], 2
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Extend line to image boundaries (0-128)
        leftmost = int(x - 64 * vx)
        topmost = int(y - 64 * vy)
        rightmost = int(x + 64 * vx)
        bottommost = int(y + 64 * vy)
        
        return (leftmost, topmost, rightmost, bottommost)
    
    def _lines_are_aligned(self, line1, line2, angle_threshold=5, distance_threshold=10):
        """
        Check if two lines are aligned (parallel/collinear) and close to each other.
        
        Args:
            line1, line2: Tuples (x1, y1, x2, y2)
            angle_threshold: Maximum angle difference in degrees
            distance_threshold: Maximum distance between lines in pixels
            
        Returns:
            True if lines are aligned and close
        """
        x1a, y1a, x2a, y2a = line1
        x1b, y1b, x2b, y2b = line2
        
        # Calculate angles
        angle1 = np.arctan2(y2a - y1a, x2a - x1a) * 180 / np.pi
        angle2 = np.arctan2(y2b - y1b, x2b - x1b) * 180 / np.pi
        
        # Normalize angles to [0, 180)
        angle1 = angle1 % 180
        angle2 = angle2 % 180
        
        # Check angle difference (considering 180° wrapping)
        angle_diff = min(abs(angle1 - angle2), 180 - abs(angle1 - angle2))
        
        if angle_diff > angle_threshold:
            return False
        
        # Check distance between line segments
        # Distance from point to line
        def point_to_line_distance(px, py, x1, y1, x2, y2):
            line_len_sq = (x2 - x1)**2 + (y2 - y1)**2
            if line_len_sq == 0:
                return np.sqrt((px - x1)**2 + (py - y1)**2)
            t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))
            proj_x = x1 + t * (x2 - x1)
            proj_y = y1 + t * (y2 - y1)
            return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
        
        # Check distance from line1 endpoints to line2
        d1 = point_to_line_distance(x1a, y1a, x1b, y1b, x2b, y2b)
        d2 = point_to_line_distance(x2a, y2a, x1b, y1b, x2b, y2b)
        
        return min(d1, d2) < distance_threshold
    
    def _merge_aligned_lines(self, lines):
        """
        Merge lines that are aligned and close to each other.
        
        Args:
            lines: List of lines as [(x1, y1, x2, y2), ...]
            
        Returns:
            Merged list of lines
        """
        if len(lines) <= 1:
            return lines
        
        merged = []
        used = set()
        
        for i, line in enumerate(lines):
            if i in used:
                continue
            
            # Start a new merged group
            group = [line]
            used.add(i)
            
            # Find all aligned lines and merge them
            for j in range(i + 1, len(lines)):
                if j in used:
                    continue
                
                if self._lines_are_aligned(line, lines[j]):
                    group.append(lines[j])
                    used.add(j)
            
            # Merge group into single line (if multiple)
            if len(group) > 1:
                # Concatenate all endpoints and fit a new line
                all_points = []
                for x1, y1, x2, y2 in group:
                    all_points.append([x1, y1])
                    all_points.append([x2, y2])
                
                all_points = np.array(all_points, dtype=np.float32)
                [vx, vy, x, y] = cv2.fitLine(all_points, cv2.DIST_L2, 0, 0.01, 0.01)
                
                # Extend to boundaries
                leftmost = int(x - 64 * vx)
                topmost = int(y - 64 * vy)
                rightmost = int(x + 64 * vx)
                bottommost = int(y + 64 * vy)
                
                merged.append((leftmost, topmost, rightmost, bottommost))
            else:
                merged.append(line)
        
        return merged
    
    def _detect_lines_lsd(self, mask):
        """
        Detect lines from a binary mask using Line Segment Detector (LSD).
        Fits lines directly to mask contours and merges aligned lines.
        
        Args:
            mask: Binary mask (single channel or 3-channel)
            
        Returns:
            List of lines as [(x1, y1, x2, y2), ...] (consolidated, max ~10 lines)
        """
        # Convert to grayscale if needed
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask
        
        # Threshold to binary
        _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Dilate slightly to connect broken lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=2)
        
        # Thin the mask to extract skeletons/centerlines (handles thick predicted regions)
        # This gives us the medial axis of thick white areas
        size = np.size(binary)
        skel = np.zeros(binary.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(binary, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(binary, temp)
            skel = cv2.bitwise_or(skel, temp)
            binary = eroded.copy()
            if cv2.countNonZero(binary) == 0:
                break
        
        # Use the skeleton for line fitting
        binary = skel
        
        # Find contours in the mask
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_lines = []
        
        # Fit line to each contour
        for contour in contours:
            if len(contour) >= 4:  # Need at least 4 points to fit a line
                try:
                    line = self._fit_line_to_contour(contour)
                    detected_lines.append(line)
                except:
                    continue
        
        # If no lines from contours, fall back to LSD edge detection
        if not detected_lines:
            edges = cv2.Canny(binary, 50, 150)
            lsd = cv2.createLineSegmentDetector(0)
            lines, _, _, _ = lsd.detect(edges)
            
            if lines is not None and len(lines) > 0:
                detected_lines = [tuple(line[0].astype(int)) for line in lines]
        
        # Merge aligned lines
        if detected_lines:
            detected_lines = self._merge_aligned_lines(detected_lines)
        
        # Sort by line length and keep top 10
        detected_lines.sort(
            key=lambda line: ((line[2] - line[0])**2 + (line[3] - line[1])**2)**0.5,
            reverse=True
        )
        
        return detected_lines[:10]
    
    def _draw_lines_on_image(self, img, lines, color=(0, 255, 255), thickness=2, endpoint_size=3):
        """
        Draw lines on an image with endpoints marked (similar to training labeler).
        
        Args:
            img: Image to draw on (will be copied)
            lines: List of lines as [(x1, y1, x2, y2), ...]
            color: RGB color tuple (default cyan)
            thickness: Line thickness
            endpoint_size: Radius of endpoint circles
            
        Returns:
            Image with lines drawn
        """
        result = img.copy()
        
        for x1, y1, x2, y2 in lines:
            # Draw line
            cv2.line(result, (x1, y1), (x2, y2), color, thickness)
            
            # Draw endpoints (start point in one shade, end point in another)
            cv2.circle(result, (x1, y1), endpoint_size, (0, 255, 0), -1)  # Green start
            cv2.circle(result, (x2, y2), endpoint_size, (255, 0, 0), -1)  # Blue end
            
            # Draw outer ring for better visibility
            cv2.circle(result, (x1, y1), endpoint_size, (255, 255, 255), 1)
            cv2.circle(result, (x2, y2), endpoint_size, (255, 255, 255), 1)
        
        return result
    
    def load_model_for_inference(self):
        """Load a trained model for inference."""
        try:
            if self.model is None:
                self.model = MobileUNet(pretrained=False).to(self.device)
            
            self.model.load_state_dict(torch.load("line_detector_unet_best.pth", map_location=self.device))
            self.model.eval()
            self.lbl_inference_model.config(text="Model: Loaded ✓", foreground="green")
            messagebox.showinfo("Success", "Model loaded successfully!")
        except FileNotFoundError:
            messagebox.showerror("Error", "Model file 'line_detector_unet_best.pth' not found.\nPlease train and save a model first.")
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
                # Prepare input with ImageNet normalization
                input_tensor = torch.from_numpy(patch_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                
                # Apply ImageNet normalization for pretrained MobileNetV2
                mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
                input_tensor = (input_tensor - mean) / std
                
                input_tensor = input_tensor.to(self.device)
                
                # Predict
                prediction = self.model(input_tensor)
                pred_mask = (prediction.squeeze().cpu().numpy() * 255).astype(np.uint8)
                pred_mask_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
            
            # Detect lines from the prediction mask using LSD (Line Segment Detector)
            detected_lines = self._detect_lines_lsd(pred_mask_rgb)
            
            self.inference_samples.append({
                'image': patch_rgb,
                'prediction': pred_mask_rgb,
                'detected_lines': detected_lines,
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
            self.btn_toggle_inference.config(text="Show: Predictions + Lines")
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
                patch_data = sample['prediction'].copy()
                
                # Overlay detected lines on the prediction
                detected_lines = sample.get('detected_lines', [])
                if detected_lines:
                    # Draw lines with endpoints marked
                    patch_data = self._draw_lines_on_image(
                        patch_data, 
                        detected_lines, 
                        color=(0, 255, 255),  # Cyan lines
                        thickness=2,
                        endpoint_size=3
                    )
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
                    # Start editing this point - save undo state BEFORE starting drag
                    self._push_undo_state("Edit point")
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
        
        # Save undo state BEFORE making changes (for new points/lines)
        if self.editing_point is None:
            self._push_undo_state()
        
        if self.editing_point is not None:
            # Finished editing existing point
            line_idx, point_idx = self.editing_point
            self.lines[line_idx][point_idx] = (img_x, img_y)
            self.selected_line_idx = line_idx  # Select the edited line
            self.editing_point = None
            self.save_annotations(show_message=False)  # Auto-save after editing
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
                self.save_annotations(show_message=False)  # Auto-save after creating new line
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
    parser.add_argument("--test-graph", action="store_true", help="Show training graph with mock data for testing")
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
    
    app = RandomPatchViewer(root, video_paths, test_graph=args.test_graph)
    
    root.mainloop()

if __name__ == "__main__":
    main()
