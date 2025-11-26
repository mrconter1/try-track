import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
import random
import bisect
import argparse
import sys
import math
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
        def to_relative_path(path):
            """Convert absolute path to relative path (videos/filename.mp4)."""
            basename = path.replace('\\', '/').split('/')[-1]
            return f"videos/{basename}"
        
        return {
            "samples": [
                {
                    "video_path": to_relative_path(sample.video_path),
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
    def __init__(self, root, video_paths, patch_size=400, test_graph=False, default_train_samples=25000):
        self.root = root
        self.video_paths = [os.path.abspath(p) for p in video_paths]
        self.patch_size = patch_size
        self.test_graph = test_graph
        self.default_train_samples = default_train_samples
        
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
        self.inference_samples = []  # List of {frame, prediction, source_info}
        self.inference_selected_indices = set()  # Set of selected indices
        self.inference_idx = 0
        self.show_inference_prediction = False
        self.inference_stride = 64  # Stride for sliding window (64 = 50% overlap)
        
        # Tile Detector state (initialized here, UI built later)
        self.tile_cap = None
        self.tile_total_frames = 0
        self.tile_fps = 30
        self.tile_current_frame = None
        self.tile_current_prediction = None
        self.tile_current_lines = []
        self.tile_slider_debounce = None
        self.tile_photo_image = None
        
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
        self.root.bind("<g>", lambda e: self.on_g_key())
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
        
        # Tile Detector Tab
        self.tile_detector_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tile_detector_tab, text="Tile Detector")
        
        self._build_labelling_tab()
        self._build_data_generation_tab()
        self._build_training_tab()
        self._build_inference_tab()
        self._build_tile_detector_tab()
        
        # Bind tab change event
        self.tab_control.bind("<<NotebookTabChanged>>", self._on_tab_changed)
    
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
        self.train_samples_var = tk.IntVar(value=self.default_train_samples)
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
        
        # Only use labeled samples toggle
        self.only_labeled_var = tk.BooleanVar(value=True)
        only_labeled_check = ttk.Checkbutton(controls_frame, text="Only samples with lines", 
                                              variable=self.only_labeled_var)
        only_labeled_check.grid(row=5, column=0, columnspan=2, sticky="w", pady=5)
        
        # Model info
        ttk.Label(controls_frame, text="Backbone: MobileNetV2 (ImageNet)", foreground="green").grid(row=6, column=0, columnspan=2, sticky="w", pady=2)
        
        # Train button
        self.btn_train = ttk.Button(controls_frame, text="Start Training", command=self.start_training)
        self.btn_train.grid(row=7, column=0, columnspan=2, sticky="ew", pady=10)
        
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
        
        # Mode selection (Region vs Full Frame)
        ttk.Label(sample_frame, text="Inference Mode:").pack(anchor="w", pady=2)
        self.inference_mode_var = tk.StringVar(value="Full Frame")
        mode_combo = ttk.Combobox(sample_frame, textvariable=self.inference_mode_var, 
                                  values=["Random Regions", "Full Frame"], state="readonly")
        mode_combo.pack(fill=tk.X, pady=5)
        mode_combo.bind("<<ComboboxSelected>>", self._update_inference_controls)
        
        # Number of samples/frames
        self.lbl_num_samples = ttk.Label(sample_frame, text="Number of frames:")
        self.lbl_num_samples.pack(anchor="w", pady=2)
        
        self.inference_samples_var = tk.IntVar(value=6)
        self.samples_spinbox = ttk.Spinbox(sample_frame, from_=1, to=100, increment=1, 
                                       textvariable=self.inference_samples_var, width=10)
        self.samples_spinbox.pack(anchor="w", pady=5)
        
        # Stride controls (container)
        self.stride_frame = ttk.Frame(sample_frame)
        self.stride_frame.pack(fill=tk.X)
        
        ttk.Label(self.stride_frame, text="Full Frame Stride:").pack(anchor="w", pady=2)
        self.inference_stride_var = tk.IntVar(value=64)
        stride_spinbox = ttk.Spinbox(self.stride_frame, from_=32, to=128, increment=16, 
                                      textvariable=self.inference_stride_var, width=10)
        stride_spinbox.pack(anchor="w", pady=5)
        
        # Full frame direct inference toggle
        self.direct_inference_frame = ttk.Frame(sample_frame)
        self.direct_inference_frame.pack(fill=tk.X)
        
        self.full_frame_mode_var = tk.BooleanVar(value=True)
        full_frame_check = ttk.Checkbutton(self.direct_inference_frame, text="Use Direct Inference (Fast)", 
                                           variable=self.full_frame_mode_var,
                                           command=self._toggle_stride_controls)
        full_frame_check.pack(anchor="w", pady=5)
        
        # Initialize UI state
        self._update_inference_controls()
        
        btn_generate_inference = ttk.Button(sample_frame, text="Generate Predictions", 
                                           command=self.generate_inference_samples)
        btn_generate_inference.pack(fill=tk.X, pady=5)
        
        # View controls
        view_frame = ttk.LabelFrame(sidebar, text="View", padding=10)
        view_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_toggle_inference = ttk.Button(view_frame, text="Show: Image", 
                                               command=self.toggle_inference_view)
        self.btn_toggle_inference.pack(fill=tk.X, pady=5)
        
        # Toggle to show/hide detected lines
        self.show_inference_lines_var = tk.BooleanVar(value=True)
        show_lines_check = ttk.Checkbutton(view_frame, text="Show detected lines", 
                                           variable=self.show_inference_lines_var,
                                           command=self.display_inference_sample)
        show_lines_check.pack(fill=tk.X, pady=5)
        
        # Line Detection Settings
        line_frame = ttk.LabelFrame(sidebar, text="Line Detection", padding=10)
        line_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Mode dropdown
        ttk.Label(line_frame, text="Mode:").pack(anchor="w")
        self.line_detect_mode_var = tk.StringVar(value="HoughLinesP + Cluster")
        line_mode_combo = ttk.Combobox(line_frame, textvariable=self.line_detect_mode_var,
                                        values=["Extended Lines", "Skeleton + Contour", "HoughLinesP", "HoughLinesP + Cluster", "LSD", "Simple Contour"],
                                        state="readonly", width=18)
        line_mode_combo.pack(fill=tk.X, pady=2)
        line_mode_combo.bind("<<ComboboxSelected>>", self._on_line_param_changed)
        
        # Parameters frame (dynamic based on mode)
        self.line_params_frame = ttk.Frame(line_frame)
        self.line_params_frame.pack(fill=tk.X, pady=5)
        
        # Common parameters
        self.line_min_length_var = tk.IntVar(value=20)
        self.line_merge_dist_var = tk.IntVar(value=15)
        self.line_merge_angle_var = tk.IntVar(value=10)
        
        # Extended Lines parameters
        self.line_cluster_angle_var = tk.IntVar(value=15)  # Angle tolerance for clustering
        self.line_min_pixels_var = tk.IntVar(value=50)     # Minimum pixels to form a line
        self.line_max_lines_var = tk.IntVar(value=28)
        
        # HoughLinesP specific
        self.hough_threshold_var = tk.IntVar(value=50)
        self.hough_max_gap_var = tk.IntVar(value=10)
        
        # Auto-update state
        self._line_update_pending = None
        self._last_line_params = None
        
        # Build initial params UI
        self._update_line_detect_params()
        
        # Re-detect button
        btn_redetect = ttk.Button(line_frame, text="Re-detect Lines", command=self.redetect_lines)
        btn_redetect.pack(fill=tk.X, pady=5)
        
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
        
        # Scale image to ~90% of screen height while maintaining aspect ratio
        target_height = int(canvas_h * 0.90)
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
    
    def on_g_key(self):
        """Handle 'g' key press - generate new samples in inference tab."""
        current_tab = self.tab_control.index(self.tab_control.select())
        if current_tab == 3:  # Inference tab
            self.generate_inference_samples()
    
    def delete_current_sample(self):
        """Delete the current sample from history and database."""
        if not self.current_patch_info or len(self.history) == 0:
            return
        
        # Confirmation dialog
        result = messagebox.askyesno(
            "Delete Sample", 
            "Are you sure you want to delete this sample?\n\nThis will remove the sample and all its line annotations.",
            icon='warning'
        )
        if not result:
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
        
        # Remove old sample from database if it exists
        if self.current_patch_info:
            old_info = self.current_patch_info
            for i, sample in enumerate(self.db.samples):
                if (sample.video_path == old_info['video_path'] and 
                    sample.frame_idx == old_info['frame_idx'] and 
                    sample.crop_rect == old_info['crop_rect']):
                    del self.db.samples[i]
                    break
        
        # Generate a new random patch
        new_info = self.generate_new_patch()
        if not new_info:
            return
        
        # Replace current position in history
        if self.history_idx >= 0 and self.history_idx < len(self.history):
            self.history[self.history_idx] = new_info
        else:
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
        
        # Save to disk (removes old sample)
        self.save_annotations(show_message=False)
        
        self.display_current_patch()
    
    def on_toggle_mask_key(self):
        """Handle 'm' key press - toggle mask based on active tab."""
        # Check which tab is active
        current_tab = self.tab_control.index(self.tab_control.select())
        if current_tab == 0:  # Labelling tab
            self.toggle_mask()
        elif current_tab == 1:  # Data Generation tab
            self.toggle_generation_view()
        elif current_tab == 3:  # Inference tab
            self.toggle_inference_view()
    
    def _on_tab_changed(self, event):
        """Handle tab change events."""
        current_tab = self.tab_control.index(self.tab_control.select())
        
        # Auto-load model when switching to Inference tab
        if current_tab == 3:  # Inference tab
            if self.model is None:
                self.load_model_for_inference()
        
        # Auto-load model when switching to Tile Detector tab
        if current_tab == 4:  # Tile Detector tab
            if self.model is None:
                self._load_tile_detector_model()

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
        
        # Filter samples based on toggle
        if self.only_labeled_var.get():
            labeled_samples = [s for s in self.db.samples if len(s.lines) > 0]
        else:
            labeled_samples = self.db.samples
        
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
        
        # Check for available samples based on toggle
        if self.only_labeled_var.get():
            available_samples = [s for s in self.db.samples if len(s.lines) > 0]
            if not available_samples:
                messagebox.showwarning("No Data", "No samples with lines found. Please label some data first.")
                return
        else:
            available_samples = self.db.samples
            if not available_samples:
                messagebox.showwarning("No Data", "No samples found. Please create some samples first.")
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
        
        # Extract scalar values from arrays (fixes NumPy deprecation warning)
        vx, vy, x, y = float(vx[0]), float(vy[0]), float(x[0]), float(y[0])
        
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
    
    def _merge_lines(self, lines, distance_threshold, angle_threshold):
        """
        Merge lines that are close and aligned.
        
        Args:
            lines: List of lines as [(x1, y1, x2, y2), ...]
            distance_threshold: Max distance between endpoints to merge
            angle_threshold: Max angle difference (radians) to merge
            
        Returns:
            Merged list of lines
        """
        if len(lines) <= 1:
            return lines
        
        def get_angle(line):
            x1, y1, x2, y2 = line
            return np.arctan2(y2 - y1, x2 - x1)
        
        def lines_similar(l1, l2):
            # Check angle difference
            a1, a2 = get_angle(l1), get_angle(l2)
            angle_diff = abs(a1 - a2)
            angle_diff = min(angle_diff, np.pi - angle_diff)  # Handle opposite directions
            if angle_diff > angle_threshold:
                return False
            
            # Check distance between endpoints
            x1a, y1a, x2a, y2a = l1
            x1b, y1b, x2b, y2b = l2
            
            # Find minimum distance between any pair of endpoints
            distances = [
                np.sqrt((x1a - x1b)**2 + (y1a - y1b)**2),
                np.sqrt((x1a - x2b)**2 + (y1a - y2b)**2),
                np.sqrt((x2a - x1b)**2 + (y2a - y1b)**2),
                np.sqrt((x2a - x2b)**2 + (y2a - y2b)**2),
            ]
            return min(distances) < distance_threshold
        
        merged = []
        used = set()
        
        for i, line in enumerate(lines):
            if i in used:
                continue
            
            # Find all lines that can be merged with this one
            group = [line]
            used.add(i)
            
            for j, other in enumerate(lines):
                if j in used:
                    continue
                if any(lines_similar(g, other) for g in group):
                    group.append(other)
                    used.add(j)
            
            # Merge group into single line
            if len(group) > 1:
                all_x = []
                all_y = []
                for x1, y1, x2, y2 in group:
                    all_x.extend([x1, x2])
                    all_y.extend([y1, y2])
                
                # Use extreme points
                min_x_idx = np.argmin(all_x)
                max_x_idx = np.argmax(all_x)
                
                if all_x[max_x_idx] - all_x[min_x_idx] > all_y[np.argmax(all_y)] - all_y[np.argmin(all_y)]:
                    # More horizontal - use x extremes
                    merged.append((all_x[min_x_idx], all_y[min_x_idx], all_x[max_x_idx], all_y[max_x_idx]))
                else:
                    # More vertical - use y extremes
                    min_y_idx = np.argmin(all_y)
                    max_y_idx = np.argmax(all_y)
                    merged.append((all_x[min_y_idx], all_y[min_y_idx], all_x[max_y_idx], all_y[max_y_idx]))
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
    
    def _find_line_intersections(self, lines, img_w, img_h):
        """
        Find all intersection points between lines.
        
        Args:
            lines: List of lines as [(x1, y1, x2, y2), ...]
            img_w, img_h: Image dimensions to filter out-of-bounds intersections
            
        Returns:
            List of intersection points as [(x, y), ...]
        """
        intersections = []
        
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines):
                if j <= i:
                    continue  # Avoid duplicate pairs
                
                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2
                
                # Line 1: P1 + t*(P2-P1)
                # Line 2: P3 + s*(P4-P3)
                # Solve for intersection
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                
                if abs(denom) < 1e-6:
                    continue  # Lines are parallel
                
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                
                # Intersection point
                ix = x1 + t * (x2 - x1)
                iy = y1 + t * (y2 - y1)
                
                # Check if within image bounds (with small margin)
                margin = 5
                if -margin <= ix <= img_w + margin and -margin <= iy <= img_h + margin:
                    intersections.append((ix, iy, i, j))  # Store point and line indices
        
        return intersections
    
    def _find_quadrilateral_regions(self, lines, intersections, img_w, img_h):
        """
        Find square-like quadrilateral regions formed by line intersections.
        
        A quadrilateral is formed when 4 intersections are connected by 4 lines,
        with each intersection being the meeting point of exactly 2 of those lines.
        
        Filters for:
        - Square-like shapes (aspect ratio close to 1)
        - Areas close to the median (removes outliers)
        
        Args:
            lines: List of lines
            intersections: List of (x, y, line_i, line_j) tuples
            img_w, img_h: Image dimensions
            
        Returns:
            List of quadrilaterals, each as [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        if len(intersections) < 4:
            return []
        
        # Build a mapping: line_index -> list of intersections on that line
        line_to_intersections = {}
        for idx, (x, y, li, lj) in enumerate(intersections):
            if li not in line_to_intersections:
                line_to_intersections[li] = []
            if lj not in line_to_intersections:
                line_to_intersections[lj] = []
            line_to_intersections[li].append(idx)
            line_to_intersections[lj].append(idx)
        
        # Find all candidate quadrilaterals
        candidates = []  # List of (corners_sorted, area, aspect_ratio)
        checked = set()
        
        # For each pair of intersecting lines (starting corner)
        for start_idx, (x0, y0, l0, l1) in enumerate(intersections):
            # Try to form a quad starting from this corner
            # We need to find 3 more corners that form a closed loop
            
            # From corner 0, we can go along line l0 or l1
            for first_line in [l0, l1]:
                other_first = l1 if first_line == l0 else l0
                
                # Find other intersections on first_line
                for idx1 in line_to_intersections.get(first_line, []):
                    if idx1 == start_idx:
                        continue
                    x1, y1, l1a, l1b = intersections[idx1]
                    
                    # The second line at corner 1 (not first_line)
                    second_line = l1b if l1a == first_line else l1a
                    
                    # Find intersections on second_line
                    for idx2 in line_to_intersections.get(second_line, []):
                        if idx2 in [start_idx, idx1]:
                            continue
                        x2, y2, l2a, l2b = intersections[idx2]
                        
                        # The third line at corner 2
                        third_line = l2b if l2a == second_line else l2a
                        
                        # Find intersections on third_line
                        for idx3 in line_to_intersections.get(third_line, []):
                            if idx3 in [start_idx, idx1, idx2]:
                                continue
                            x3, y3, l3a, l3b = intersections[idx3]
                            
                            # The fourth line should be other_first to close the loop
                            fourth_line = l3b if l3a == third_line else l3a
                            
                            if fourth_line == other_first:
                                # Check if this intersection is connected back to start
                                # via other_first line
                                if start_idx in line_to_intersections.get(other_first, []):
                                    # Found a quadrilateral!
                                    quad_key = tuple(sorted([start_idx, idx1, idx2, idx3]))
                                    if quad_key not in checked:
                                        checked.add(quad_key)
                                        
                                        # Order corners in a consistent way (clockwise/counterclockwise)
                                        corners = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
                                        
                                        # Compute centroid
                                        cx = sum(c[0] for c in corners) / 4
                                        cy = sum(c[1] for c in corners) / 4
                                        
                                        # Sort by angle from centroid
                                        corners_sorted = sorted(corners, 
                                            key=lambda c: math.atan2(c[1] - cy, c[0] - cx))
                                        
                                        # Calculate area using shoelace formula
                                        n = len(corners_sorted)
                                        area = 0
                                        for i in range(n):
                                            j = (i + 1) % n
                                            area += corners_sorted[i][0] * corners_sorted[j][1]
                                            area -= corners_sorted[j][0] * corners_sorted[i][1]
                                        area = abs(area) / 2
                                        
                                        if area < 100:  # Skip tiny quads
                                            continue
                                        
                                        # Calculate aspect ratio using side lengths
                                        # Measure all 4 side lengths
                                        side_lengths = []
                                        for i in range(4):
                                            j = (i + 1) % 4
                                            dx = corners_sorted[j][0] - corners_sorted[i][0]
                                            dy = corners_sorted[j][1] - corners_sorted[i][1]
                                            side_lengths.append(math.sqrt(dx*dx + dy*dy))
                                        
                                        # For a square, opposite sides should be equal
                                        # and all sides should be similar
                                        # Compare pairs of opposite sides
                                        side_a = (side_lengths[0] + side_lengths[2]) / 2  # avg of opposite
                                        side_b = (side_lengths[1] + side_lengths[3]) / 2  # avg of opposite
                                        
                                        if min(side_a, side_b) < 10:  # Avoid division by zero
                                            continue
                                        
                                        aspect_ratio = max(side_a, side_b) / min(side_a, side_b)
                                        
                                        # Only keep square-ish shapes (aspect ratio < 1.5)
                                        if aspect_ratio < 1.5:
                                            candidates.append((corners_sorted, area, aspect_ratio))
        
        if not candidates:
            return []
        
        # Step 1: Remove overlapping quads (keep the one with better aspect ratio)
        def polygon_iou(poly1, poly2):
            """Calculate intersection over union of two polygons."""
            # Convert to numpy arrays for cv2
            pts1 = np.array(poly1, dtype=np.float32).reshape(-1, 2)
            pts2 = np.array(poly2, dtype=np.float32).reshape(-1, 2)
            
            # Use cv2.intersectConvexConvex for convex polygons
            try:
                ret, intersection = cv2.intersectConvexConvex(pts1, pts2)
                if ret == 0 or intersection is None or len(intersection) < 3:
                    return 0.0
                
                inter_area = cv2.contourArea(intersection)
                area1 = cv2.contourArea(pts1)
                area2 = cv2.contourArea(pts2)
                
                union_area = area1 + area2 - inter_area
                if union_area < 1e-6:
                    return 0.0
                
                return inter_area / union_area
            except:
                return 0.0
        
        # Sort by area (smaller first - prefer smaller quads when overlapping)
        candidates_sorted = sorted(candidates, key=lambda c: c[1])
        
        non_overlapping = []
        for corners, area, aspect in candidates_sorted:
            is_overlapping = False
            for existing_corners, _, _ in non_overlapping:
                iou = polygon_iou(corners, existing_corners)
                if iou > 0.3:  # More than 30% overlap
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                non_overlapping.append((corners, area, aspect))
        
        if not non_overlapping:
            return []
        
        # Step 2: Filter by area - remove outliers (keep those close to median)
        areas = [c[1] for c in non_overlapping]
        median_area = sorted(areas)[len(areas) // 2]
        
        # Keep quads with area between 0.75x and 1.25x the median
        filtered = []
        for corners, area, aspect in non_overlapping:
            if 0.75 * median_area <= area <= 1.25 * median_area:
                filtered.append(corners)
        
        return filtered
    
    def _draw_regions_on_image(self, img, lines, regions=None):
        """
        Draw lines and numbered quadrilateral regions on an image.
        
        Args:
            img: Image to draw on
            lines: List of lines
            regions: Optional pre-computed regions, or None to compute them
            
        Returns:
            Image with regions drawn and numbered
        """
        result = img.copy()
        img_h, img_w = result.shape[:2]
        
        # Find intersections and regions if not provided
        if regions is None:
            intersections = self._find_line_intersections(lines, img_w, img_h)
            regions = self._find_quadrilateral_regions(lines, intersections, img_w, img_h)
        
        # Draw regions with semi-transparent fill and number
        for idx, corners in enumerate(regions):
            # Convert to numpy array for drawing
            pts = np.array(corners, dtype=np.int32)
            
            # Draw filled polygon with transparency
            overlay = result.copy()
            # Use different colors for different regions
            colors = [
                (255, 100, 100),  # Light red
                (100, 255, 100),  # Light green
                (100, 100, 255),  # Light blue
                (255, 255, 100),  # Yellow
                (255, 100, 255),  # Magenta
                (100, 255, 255),  # Cyan
                (200, 150, 100),  # Tan
                (150, 100, 200),  # Purple
            ]
            color = colors[idx % len(colors)]
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
            
            # Draw border
            cv2.polylines(result, [pts], isClosed=True, color=color, thickness=2)
            
            # Calculate centroid for label
            cx = int(sum(c[0] for c in corners) / 4)
            cy = int(sum(c[1] for c in corners) / 4)
            
            # Draw number label with background
            label = str(idx + 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.5, min(img_w, img_h) / 400)
            thickness = max(1, int(font_scale * 2))
            
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Background rectangle
            padding = 4
            cv2.rectangle(result, 
                         (cx - text_w//2 - padding, cy - text_h//2 - padding),
                         (cx + text_w//2 + padding, cy + text_h//2 + padding),
                         (0, 0, 0), -1)
            
            # Text
            cv2.putText(result, label, 
                       (cx - text_w//2, cy + text_h//2),
                       font, font_scale, (255, 255, 255), thickness)
        
        # Also draw intersection points
        intersections = self._find_line_intersections(lines, img_w, img_h)
        for x, y, _, _ in intersections:
            cv2.circle(result, (int(x), int(y)), 5, (255, 255, 0), -1)  # Yellow filled
            cv2.circle(result, (int(x), int(y)), 5, (0, 0, 0), 1)  # Black outline
        
        return result
    
    def _update_line_detect_params(self, event=None):
        """Update the line detection parameters UI based on selected mode."""
        # Clear existing widgets
        for widget in self.line_params_frame.winfo_children():
            widget.destroy()
        
        mode = self.line_detect_mode_var.get()
        
        def make_slider(parent, label, var, from_, to, row):
            """Create a labeled slider with value display."""
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
            
            # Value label
            val_label = ttk.Label(parent, text=str(var.get()), width=4)
            val_label.grid(row=row, column=2, sticky="e", pady=1)
            
            # Slider
            def on_slide(val):
                int_val = int(float(val))
                var.set(int_val)
                val_label.config(text=str(int_val))
                self._on_line_param_changed()
            
            slider = ttk.Scale(parent, from_=from_, to=to, variable=var, 
                              orient="horizontal", command=on_slide)
            slider.grid(row=row, column=1, sticky="ew", pady=1, padx=5)
            return slider
        
        row = 0
        
        # Min line length
        make_slider(self.line_params_frame, "Min length:", self.line_min_length_var, 5, 100, row)
        row += 1
        
        # Max lines
        make_slider(self.line_params_frame, "Max lines:", self.line_max_lines_var, 1, 30, row)
        row += 1
        
        if mode in ["Skeleton + Contour", "Simple Contour", "LSD"]:
            # Merge distance
            make_slider(self.line_params_frame, "Merge dist:", self.line_merge_dist_var, 0, 50, row)
            row += 1
            
            # Merge angle
            make_slider(self.line_params_frame, "Merge angle°:", self.line_merge_angle_var, 0, 45, row)
            row += 1
        
        if mode == "HoughLinesP":
            # Hough threshold
            make_slider(self.line_params_frame, "Threshold:", self.hough_threshold_var, 10, 200, row)
            row += 1
            
            # Max gap
            make_slider(self.line_params_frame, "Max gap:", self.hough_max_gap_var, 1, 50, row)
            row += 1
        
        if mode == "HoughLinesP + Cluster":
            # Hough threshold
            make_slider(self.line_params_frame, "Threshold:", self.hough_threshold_var, 10, 200, row)
            row += 1
            
            # Max gap
            make_slider(self.line_params_frame, "Max gap:", self.hough_max_gap_var, 1, 50, row)
            row += 1
            
            # Cluster angle tolerance
            make_slider(self.line_params_frame, "Cluster angle°:", self.line_cluster_angle_var, 5, 45, row)
            row += 1
            
            # Cluster distance tolerance
            make_slider(self.line_params_frame, "Cluster dist:", self.line_merge_dist_var, 5, 100, row)
            row += 1
        
        if mode == "Extended Lines":
            # Cluster angle tolerance
            make_slider(self.line_params_frame, "Cluster angle°:", self.line_cluster_angle_var, 5, 45, row)
            row += 1
            
            # Minimum pixels per line
            make_slider(self.line_params_frame, "Min pixels:", self.line_min_pixels_var, 10, 200, row)
            row += 1
        
        # Configure column weights
        self.line_params_frame.columnconfigure(0, weight=0)
        self.line_params_frame.columnconfigure(1, weight=1)
        self.line_params_frame.columnconfigure(2, weight=0)
    
    def _on_line_param_changed(self, event=None):
        """Called when line detection parameters change. Debounces updates."""
        # Cancel any pending update
        if self._line_update_pending:
            self.root.after_cancel(self._line_update_pending)
        
        # Schedule new update after 250ms
        self._line_update_pending = self.root.after(250, self._do_line_update)
    
    def _do_line_update(self):
        """Actually perform the line update after debounce."""
        self._line_update_pending = None
        
        # Check if we have samples and lines are visible
        if not self.inference_samples:
            return
        if not self.show_inference_lines_var.get():
            return
        
        # Get current params
        current_params = (
            self.line_detect_mode_var.get(),
            self.line_min_length_var.get(),
            self.line_max_lines_var.get(),
            self.line_merge_dist_var.get(),
            self.line_merge_angle_var.get(),
            self.hough_threshold_var.get(),
            self.hough_max_gap_var.get(),
            self.line_cluster_angle_var.get(),
            self.line_min_pixels_var.get()
        )
        
        # Only update if params actually changed
        if current_params == self._last_line_params:
            return
        
        self._last_line_params = current_params
        self.redetect_lines()
    
    def redetect_lines(self):
        """Re-run line detection on existing inference samples with current parameters."""
        if not self.inference_samples:
            return
        
        mode = self.line_detect_mode_var.get()
        self.lbl_inference_idx.config(text=f"Re-detecting lines ({mode})...")
        self.root.update()
        
        for sample in self.inference_samples:
            mask = sample['prediction']
            sample['detected_lines'] = self._detect_lines_with_mode(mask, mode)
        
        self.display_inference_sample()
    
    def _detect_lines_with_mode(self, mask, mode):
        """Detect lines using the specified mode and current parameters."""
        min_length = self.line_min_length_var.get()
        max_lines = self.line_max_lines_var.get()
        merge_dist = self.line_merge_dist_var.get()
        merge_angle = self.line_merge_angle_var.get()
        
        if mode == "Skeleton + Contour":
            return self._detect_lines_skeleton(mask, min_length, max_lines, merge_dist, merge_angle)
        elif mode == "HoughLinesP":
            threshold = self.hough_threshold_var.get()
            max_gap = self.hough_max_gap_var.get()
            return self._detect_lines_hough(mask, min_length, max_lines, threshold, max_gap)
        elif mode == "HoughLinesP + Cluster":
            threshold = self.hough_threshold_var.get()
            max_gap = self.hough_max_gap_var.get()
            cluster_angle = self.line_cluster_angle_var.get()
            cluster_dist = self.line_merge_dist_var.get()
            return self._detect_lines_hough_clustered(mask, min_length, max_lines, threshold, max_gap, cluster_angle, cluster_dist)
        elif mode == "LSD":
            return self._detect_lines_lsd_mode(mask, min_length, max_lines, merge_dist, merge_angle)
        elif mode == "Simple Contour":
            return self._detect_lines_contour(mask, min_length, max_lines, merge_dist, merge_angle)
        elif mode == "Extended Lines":
            cluster_angle = self.line_cluster_angle_var.get()
            min_pixels = self.line_min_pixels_var.get()
            return self._detect_lines_extended(mask, min_pixels, max_lines, cluster_angle)
        else:
            return []
    
    def _detect_lines_skeleton(self, mask, min_length, max_lines, merge_dist, merge_angle):
        """Detect lines using skeletonization + contour fitting."""
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask
        
        _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Skeletonize
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skel = np.zeros(binary.shape, np.uint8)
        temp_binary = binary.copy()
        while True:
            eroded = cv2.erode(temp_binary, kernel)
            temp = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(temp_binary, temp)
            skel = cv2.bitwise_or(skel, temp)
            temp_binary = eroded.copy()
            if cv2.countNonZero(temp_binary) == 0:
                break
        
        # Find contours and fit lines
        contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        lines = []
        for contour in contours:
            if len(contour) < 2:
                continue
            pts = contour.reshape(-1, 2)
            if len(pts) < 2:
                continue
            
            # Get endpoints
            p1 = tuple(pts[0])
            p2 = tuple(pts[-1])
            length = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            
            if length >= min_length:
                lines.append((p1[0], p1[1], p2[0], p2[1]))
        
        # Merge similar lines
        if merge_dist > 0 or merge_angle > 0:
            lines = self._merge_lines(lines, merge_dist, np.radians(merge_angle))
        
        # Sort by length and limit
        lines = sorted(lines, key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2, reverse=True)
        return lines[:max_lines]
    
    def _detect_lines_hough(self, mask, min_length, max_lines, threshold, max_gap):
        """Detect lines using HoughLinesP."""
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask
        
        _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(binary, 50, 150)
        
        # Detect lines
        hough_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold,
                                       minLineLength=min_length, maxLineGap=max_gap)
        
        if hough_lines is None:
            return []
        
        lines = []
        for line in hough_lines:
            x1, y1, x2, y2 = line[0]
            lines.append((x1, y1, x2, y2))
        
        # Sort by length and limit
        lines = sorted(lines, key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2, reverse=True)
        return lines[:max_lines]
    
    def _detect_lines_hough_clustered(self, mask, min_length, max_lines, threshold, max_gap, cluster_angle_deg, cluster_dist):
        """
        Detect lines using HoughLinesP, then cluster collinear lines and extend to frame boundaries.
        
        Lines are considered collinear if they have similar angles and their infinite line
        representations are close together.
        """
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask
        
        h, w = mask_gray.shape
        _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(binary, 50, 150)
        
        # Detect lines with HoughLinesP
        hough_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold,
                                       minLineLength=min_length, maxLineGap=max_gap)
        
        if hough_lines is None:
            return []
        
        # Convert to list of (x1, y1, x2, y2)
        raw_lines = [tuple(line[0]) for line in hough_lines]
        
        if len(raw_lines) == 0:
            return []
        
        cluster_angle_rad = np.radians(cluster_angle_deg)
        
        def get_line_params(line):
            """Get angle and perpendicular distance from origin for a line."""
            x1, y1, x2, y2 = line
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length < 1e-6:
                return 0, 0, (x1, y1)
            
            # Normalize angle to [0, pi)
            angle = np.arctan2(dy, dx)
            if angle < 0:
                angle += np.pi
            if angle >= np.pi:
                angle -= np.pi
            
            # Perpendicular distance from origin to the infinite line
            # Using formula: |ax + by + c| / sqrt(a^2 + b^2)
            # Line equation: dy*x - dx*y + (dx*y1 - dy*x1) = 0
            midx, midy = (x1 + x2) / 2, (y1 + y2) / 2
            perp_dist = abs(dy * 0 - dx * 0 + (dx * y1 - dy * x1)) / length
            
            return angle, perp_dist, (midx, midy)
        
        def lines_collinear(l1, l2, angle_thresh, dist_thresh):
            """Check if two lines are approximately collinear."""
            angle1, dist1, mid1 = get_line_params(l1)
            angle2, dist2, mid2 = get_line_params(l2)
            
            # Check angle similarity (handle wrap-around at 0/pi)
            angle_diff = abs(angle1 - angle2)
            angle_diff = min(angle_diff, np.pi - angle_diff)
            if angle_diff > angle_thresh:
                return False
            
            # For collinear lines, also check that points from one line
            # are close to the infinite extension of the other
            x1, y1, x2, y2 = l1
            x3, y3, x4, y4 = l2
            
            # Direction vector of line 1
            dx1, dy1 = x2 - x1, y2 - y1
            len1 = np.sqrt(dx1**2 + dy1**2)
            if len1 < 1e-6:
                return False
            dx1, dy1 = dx1 / len1, dy1 / len1
            
            # Distance from midpoint of line 2 to infinite line 1
            mid2x, mid2y = (x3 + x4) / 2, (y3 + y4) / 2
            # Point to line distance
            dist = abs((mid2x - x1) * dy1 - (mid2y - y1) * dx1)
            
            return dist < dist_thresh
        
        # Cluster collinear lines
        used = set()
        clusters = []
        
        for i, line in enumerate(raw_lines):
            if i in used:
                continue
            
            cluster = [line]
            used.add(i)
            
            for j, other in enumerate(raw_lines):
                if j in used:
                    continue
                if lines_collinear(line, other, cluster_angle_rad, cluster_dist):
                    cluster.append(other)
                    used.add(j)
            
            clusters.append(cluster)
        
        # For each cluster, fit a line and extend to frame boundaries
        result_lines = []
        
        for cluster in clusters:
            # Collect all endpoints from the cluster
            all_points = []
            for x1, y1, x2, y2 in cluster:
                all_points.append([x1, y1])
                all_points.append([x2, y2])
            
            all_points = np.array(all_points, dtype=np.float32)
            
            # Fit a line through all points using PCA
            mean = all_points.mean(axis=0)
            centered = all_points - mean
            cov = np.cov(centered.T)
            
            if cov.shape != (2, 2):
                continue
            
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            direction = eigenvectors[:, 1]  # Principal direction
            dx, dy = direction[0], direction[1]
            
            # Extend line to image boundaries
            t_values = []
            
            # Left edge (x = 0)
            if abs(dx) > 1e-6:
                t = -mean[0] / dx
                y_at_t = mean[1] + t * dy
                if 0 <= y_at_t <= h:
                    t_values.append(t)
            
            # Right edge (x = w-1)
            if abs(dx) > 1e-6:
                t = (w - 1 - mean[0]) / dx
                y_at_t = mean[1] + t * dy
                if 0 <= y_at_t <= h:
                    t_values.append(t)
            
            # Top edge (y = 0)
            if abs(dy) > 1e-6:
                t = -mean[1] / dy
                x_at_t = mean[0] + t * dx
                if 0 <= x_at_t <= w:
                    t_values.append(t)
            
            # Bottom edge (y = h-1)
            if abs(dy) > 1e-6:
                t = (h - 1 - mean[1]) / dy
                x_at_t = mean[0] + t * dx
                if 0 <= x_at_t <= w:
                    t_values.append(t)
            
            if len(t_values) < 2:
                continue
            
            # Get the two extreme intersection points
            t_min, t_max = min(t_values), max(t_values)
            x1 = int(np.clip(mean[0] + t_min * dx, 0, w - 1))
            y1 = int(np.clip(mean[1] + t_min * dy, 0, h - 1))
            x2 = int(np.clip(mean[0] + t_max * dx, 0, w - 1))
            y2 = int(np.clip(mean[1] + t_max * dy, 0, h - 1))
            
            # Check if line is long enough
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length >= min_length:
                result_lines.append((x1, y1, x2, y2))
        
        # Sort by length and limit
        result_lines = sorted(result_lines, key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2, reverse=True)
        return result_lines[:max_lines]
    
    def _detect_lines_lsd_mode(self, mask, min_length, max_lines, merge_dist, merge_angle):
        """Detect lines using OpenCV's LSD."""
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask
        
        _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Create LSD detector
        lsd = cv2.createLineSegmentDetector(0)
        detected, _, _, _ = lsd.detect(binary)
        
        if detected is None:
            return []
        
        lines = []
        for line in detected:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length >= min_length:
                lines.append((int(x1), int(y1), int(x2), int(y2)))
        
        # Merge similar lines
        if merge_dist > 0 or merge_angle > 0:
            lines = self._merge_lines(lines, merge_dist, np.radians(merge_angle))
        
        # Sort by length and limit
        lines = sorted(lines, key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2, reverse=True)
        return lines[:max_lines]
    
    def _detect_lines_contour(self, mask, min_length, max_lines, merge_dist, merge_angle):
        """Detect lines using simple contour fitting."""
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask
        
        _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Dilate to connect broken parts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=2)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        lines = []
        for contour in contours:
            if len(contour) < 5:
                continue
            
            # Fit line to contour
            [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Get extent
            pts = contour.reshape(-1, 2)
            projections = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
            min_proj, max_proj = projections.min(), projections.max()
            
            x1 = int(x0 + min_proj * vx)
            y1 = int(y0 + min_proj * vy)
            x2 = int(x0 + max_proj * vx)
            y2 = int(y0 + max_proj * vy)
            
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length >= min_length:
                lines.append((x1, y1, x2, y2))
        
        # Merge similar lines
        if merge_dist > 0 or merge_angle > 0:
            lines = self._merge_lines(lines, merge_dist, np.radians(merge_angle))
        
        # Sort by length and limit
        lines = sorted(lines, key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2, reverse=True)
        return lines[:max_lines]
    
    def _detect_lines_extended(self, mask, min_pixels, max_lines, cluster_angle_deg):
        """
        Detect lines and extend them to image boundaries.
        Uses RANSAC-style clustering to find dominant line directions,
        then fits and extends lines to span the entire image.
        """
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask
        
        h, w = mask_gray.shape
        _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Get all white pixel coordinates
        points = np.column_stack(np.where(binary > 0))  # (row, col) = (y, x)
        if len(points) < min_pixels:
            return []
        
        # Convert to (x, y) format
        points_xy = points[:, ::-1].astype(np.float32)  # Now (x, y)
        
        lines = []
        remaining_points = points_xy.copy()
        cluster_angle_rad = np.radians(cluster_angle_deg)
        
        # Iteratively find dominant lines using RANSAC-like approach
        for _ in range(max_lines):
            if len(remaining_points) < min_pixels:
                break
            
            best_inliers = []
            best_line = None
            
            # RANSAC iterations
            n_iterations = min(100, len(remaining_points) * 2)
            for _ in range(n_iterations):
                # Random sample of 2 points
                if len(remaining_points) < 2:
                    break
                indices = np.random.choice(len(remaining_points), 2, replace=False)
                p1, p2 = remaining_points[indices]
                
                # Compute line direction
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                length = np.sqrt(dx**2 + dy**2)
                if length < 1:
                    continue
                
                # Normalize direction
                dx, dy = dx / length, dy / length
                
                # Find inliers: points close to and aligned with this line
                # Distance from point to line: |cross product| / length
                diff = remaining_points - p1
                cross = np.abs(diff[:, 0] * dy - diff[:, 1] * dx)
                
                # Also check angle similarity (for thick masks)
                dist_threshold = 10  # pixels
                inlier_mask = cross < dist_threshold
                
                inliers = remaining_points[inlier_mask]
                
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    # Fit line to all inliers using PCA
                    if len(inliers) >= 2:
                        mean = inliers.mean(axis=0)
                        centered = inliers - mean
                        cov = np.cov(centered.T)
                        if cov.shape == (2, 2):
                            eigenvalues, eigenvectors = np.linalg.eigh(cov)
                            direction = eigenvectors[:, 1]  # Principal direction
                            best_line = (mean, direction)
            
            if best_line is None or len(best_inliers) < min_pixels:
                break
            
            mean, direction = best_line
            dx, dy = direction[0], direction[1]
            
            # Extend line to image boundaries
            # Line equation: point = mean + t * direction
            # Find t where line intersects image edges
            t_values = []
            
            # Left edge (x = 0)
            if abs(dx) > 1e-6:
                t = -mean[0] / dx
                y_at_t = mean[1] + t * dy
                if 0 <= y_at_t <= h:
                    t_values.append(t)
            
            # Right edge (x = w-1)
            if abs(dx) > 1e-6:
                t = (w - 1 - mean[0]) / dx
                y_at_t = mean[1] + t * dy
                if 0 <= y_at_t <= h:
                    t_values.append(t)
            
            # Top edge (y = 0)
            if abs(dy) > 1e-6:
                t = -mean[1] / dy
                x_at_t = mean[0] + t * dx
                if 0 <= x_at_t <= w:
                    t_values.append(t)
            
            # Bottom edge (y = h-1)
            if abs(dy) > 1e-6:
                t = (h - 1 - mean[1]) / dy
                x_at_t = mean[0] + t * dx
                if 0 <= x_at_t <= w:
                    t_values.append(t)
            
            if len(t_values) >= 2:
                t_min, t_max = min(t_values), max(t_values)
                x1 = int(np.clip(mean[0] + t_min * dx, 0, w - 1))
                y1 = int(np.clip(mean[1] + t_min * dy, 0, h - 1))
                x2 = int(np.clip(mean[0] + t_max * dx, 0, w - 1))
                y2 = int(np.clip(mean[1] + t_max * dy, 0, h - 1))
                
                lines.append((x1, y1, x2, y2))
            
            # Remove inliers from remaining points
            if len(best_inliers) > 0:
                # Create mask for points to keep
                inlier_set = set(map(tuple, best_inliers.astype(int)))
                keep_mask = np.array([tuple(p.astype(int)) not in inlier_set for p in remaining_points])
                remaining_points = remaining_points[keep_mask]
        
        return lines

    def _update_inference_controls(self, event=None):
        """Update UI based on selected inference mode."""
        mode = self.inference_mode_var.get()
        
        if mode == "Full Frame":
            self.lbl_num_samples.config(text="Number of frames:")
            self.samples_spinbox.config(to=10)
            if self.inference_samples_var.get() > 10:
                self.inference_samples_var.set(3)
            
            # Show full frame options
            self.direct_inference_frame.pack(fill=tk.X)
            self._toggle_stride_controls()
            
        else:  # Random Regions
            self.lbl_num_samples.config(text="Number of patches:")
            self.samples_spinbox.config(to=100)
            if self.inference_samples_var.get() < 25:
                self.inference_samples_var.set(25)
            
            # Hide full frame options
            self.direct_inference_frame.pack_forget()
            self.stride_frame.pack_forget()

    def _toggle_stride_controls(self):
        """Show/hide stride controls based on full frame mode."""
        if self.full_frame_mode_var.get():
            self.stride_frame.pack_forget()
        else:
            self.stride_frame.pack(fill=tk.X)
    
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
        """Generate predictions based on selected mode (Full Frame or Random Regions)."""
        if self.model is None:
            messagebox.showwarning("No Model", "Please load a model first.")
            return
        
        if hasattr(self, 'inference_running') and self.inference_running:
            return  # Already running
        
        self.inference_running = True
        self.lbl_inference_idx.config(text="Generating predictions...")
        
        # Run in background thread
        thread = threading.Thread(target=self._generate_inference_thread, daemon=True)
        thread.start()
    
    def _generate_inference_thread(self):
        """Background thread for generating inference samples."""
        try:
            mode = self.inference_mode_var.get()
            num_samples = self.inference_samples_var.get()
            samples = []
            
            self.model.eval()
            
            # Pre-compute normalization tensors
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).reshape(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).reshape(1, 3, 1, 1)
            
            if mode == "Random Regions":
                patch_size = self.patch_size
                
                # First, collect all patches (fast)
                patches_data = []
                for i in range(num_samples):
                    video_path, frame_idx = self.get_random_frame_location()
                    if not video_path: continue
                    
                    cap = cv2.VideoCapture(video_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if not ret or frame is None: continue
                    
                    h, w = frame.shape[:2]
                    if h < patch_size or w < patch_size: continue
                    
                    x = random.randint(0, w - patch_size)
                    y = random.randint(0, h - patch_size)
                    
                    patch = frame[y:y+patch_size, x:x+patch_size]
                    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                    
                    patches_data.append({
                        'patch_rgb': patch_rgb,
                        'video_path': video_path,
                        'frame_idx': frame_idx,
                        'x': x, 'y': y
                    })
                    
                    # Update progress
                    self.root.after(0, lambda i=i: self.lbl_inference_idx.config(
                        text=f"Loading patches: {i+1}/{num_samples}"))
                
                # Now batch inference
                batch_size = 4  # Process 4 at a time
                for batch_start in range(0, len(patches_data), batch_size):
                    batch_end = min(batch_start + batch_size, len(patches_data))
                    batch = patches_data[batch_start:batch_end]
                    
                    # Prepare batch tensors
                    batch_tensors = []
                    batch_info = []
                    
                    for item in batch:
                        patch_rgb = item['patch_rgb']
                        ph, pw = patch_rgb.shape[:2]
                        pad_h = (32 - ph % 32) % 32
                        pad_w = (32 - pw % 32) % 32
                        
                        if pad_h > 0 or pad_w > 0:
                            patch_padded = np.pad(patch_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                        else:
                            patch_padded = patch_rgb
                        
                        input_tensor = torch.from_numpy(patch_padded.astype(np.float32) / 255.0).permute(2, 0, 1)
                        batch_tensors.append(input_tensor)
                        batch_info.append({'item': item, 'ph': ph, 'pw': pw})
                    
                    # Stack and run batch inference
                    with torch.no_grad():
                        batch_input = torch.stack(batch_tensors).to(self.device)
                        batch_input = (batch_input - mean) / std
                        predictions = self.model(batch_input)
                    
                    # Process results
                    for idx, (pred, info) in enumerate(zip(predictions, batch_info)):
                        item = info['item']
                        ph, pw = info['ph'], info['pw']
                        
                        pred_np = pred.squeeze().cpu().numpy()
                        pred_mask = (pred_np[:ph, :pw] * 255).astype(np.uint8)
                        pred_mask_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
                        
                        line_mode = self.line_detect_mode_var.get()
                        detected_lines = self._detect_lines_with_mode(pred_mask_rgb, line_mode)
                        
                        samples.append({
                            'frame': item['patch_rgb'],
                            'prediction': pred_mask_rgb,
                            'detected_lines': detected_lines,
                            'source_video': os.path.basename(item['video_path']),
                            'source_frame': item['frame_idx'],
                            'location': (item['x'], item['y']),
                            'size': (patch_size, patch_size),
                            'type': 'patch'
                        })
                    
                    # Update progress
                    progress = min(batch_end, len(patches_data))
                    self.root.after(0, lambda p=progress, t=len(patches_data): self.lbl_inference_idx.config(
                        text=f"Inference: {p}/{t}"))
            
            else:  # Full Frame mode
                use_full_frame = self.full_frame_mode_var.get()
                
                for i in range(num_samples):
                    self.root.after(0, lambda i=i: self.lbl_inference_idx.config(
                        text=f"Processing frame {i+1}/{num_samples}..."))
                    
                    video_path, frame_idx = self.get_random_frame_location()
                    if not video_path: continue
                    
                    cap = cv2.VideoCapture(video_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if not ret or frame is None: continue
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = frame_rgb.shape[:2]
                    if h < 128 or w < 128: continue
                    
                    if use_full_frame:
                        pad_h = (32 - h % 32) % 32
                        pad_w = (32 - w % 32) % 32
                        
                        if pad_h > 0 or pad_w > 0:
                            frame_padded = np.pad(frame_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                        else:
                            frame_padded = frame_rgb
                        
                        with torch.no_grad():
                            input_tensor = torch.from_numpy(frame_padded.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                            input_tensor = (input_tensor.to(self.device) - mean) / std
                            prediction = self.model(input_tensor)
                            pred_full = prediction.squeeze().cpu().numpy()
                        
                        pred_mask = (pred_full[:h, :w] * 255).astype(np.uint8)
                        pred_mask_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
                    else:
                        stride = self.inference_stride_var.get()
                        batch_size = 32
                        pred_sum = np.zeros((h, w), dtype=np.float32)
                        pred_count = np.zeros((h, w), dtype=np.float32)
                        patch_size = 128
                        
                        y_positions = list(range(0, h - patch_size + 1, stride))
                        if y_positions[-1] + patch_size < h: y_positions.append(h - patch_size)
                        x_positions = list(range(0, w - patch_size + 1, stride))
                        if x_positions[-1] + patch_size < w: x_positions.append(w - patch_size)
                        
                        all_positions = [(y, x) for y in y_positions for x in x_positions]
                        
                        with torch.no_grad():
                            for batch_start in range(0, len(all_positions), batch_size):
                                batch_end = min(batch_start + batch_size, len(all_positions))
                                batch_positions = all_positions[batch_start:batch_end]
                                
                                batch_patches = []
                                for y, x in batch_positions:
                                    patch = frame_rgb[y:y+patch_size, x:x+patch_size]
                                    patch_tensor = torch.from_numpy(patch.astype(np.float32) / 255.0).permute(2, 0, 1)
                                    batch_patches.append(patch_tensor)
                                
                                batch_tensor = torch.stack(batch_patches).to(self.device)
                                batch_tensor = (batch_tensor - mean) / std
                                
                                predictions = self.model(batch_tensor)
                                pred_patches = predictions.squeeze(1).cpu().numpy()
                                
                                for idx, (y, x) in enumerate(batch_positions):
                                    pred_sum[y:y+patch_size, x:x+patch_size] += pred_patches[idx]
                                    pred_count[y:y+patch_size, x:x+patch_size] += 1
                        
                        pred_count[pred_count == 0] = 1
                        pred_avg = pred_sum / pred_count
                        pred_mask = (pred_avg * 255).astype(np.uint8)
                        pred_mask_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
                    
                    line_mode = self.line_detect_mode_var.get()
                    detected_lines = self._detect_lines_with_mode(pred_mask_rgb, line_mode)
                    
                    samples.append({
                        'frame': frame_rgb,
                        'prediction': pred_mask_rgb,
                        'detected_lines': detected_lines,
                        'source_video': os.path.basename(video_path),
                        'source_frame': frame_idx,
                        'size': (w, h),
                        'type': 'full'
                    })
            
            # Finish up - update UI on main thread
            self.inference_samples = samples
            self.inference_selected_indices = set()
            
            def finish_inference():
                self.inference_running = False
                if self.inference_samples:
                    self.inference_idx = 0
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
    
    def display_inference_sample(self):
        """Display predictions in appropriate layout."""
        if not self.inference_samples:
            return
        
        # Update info
        num_samples = len(self.inference_samples)
        mode = self.inference_mode_var.get()
        self.lbl_inference_idx.config(text=f"{mode}: {num_samples}")
        
        # Update button text
        if self.show_inference_prediction:
            self.btn_toggle_inference.config(text="Show: Predictions")
        else:
            self.btn_toggle_inference.config(text="Show: Images")
        
        # Layout configuration based on mode
        if mode == "Random Regions":
            # Calculate grid size to fit all samples (prefer square-ish grid)
            num_cols = max(1, int(math.ceil(math.sqrt(num_samples))))
            num_rows = max(1, int(math.ceil(num_samples / num_cols)))
            max_items = num_samples
        else:
            num_cols = min(3, num_samples)
            num_rows = (num_samples + num_cols - 1) // num_cols
            max_items = 100  # Show all frames
        
        # Get canvas size
        canvas_w = self.inference_canvas.winfo_width()
        canvas_h = self.inference_canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            self.root.after(100, self.display_inference_sample)
            return
        
        # Calculate cell size
        padding = 10
        available_w = canvas_w - padding * (num_cols + 1)
        available_h = canvas_h - padding * (num_rows + 1)
        cell_w = available_w // num_cols
        cell_h = available_h // num_rows
        
        # Store grid geometry for click detection
        self.inference_grid_geometry = {
            'num_cols': num_cols,
            'cell_w': cell_w,
            'cell_h': cell_h,
            'padding': padding,
            'max_items': max_items
        }
        
        grid_img = np.full((canvas_h, canvas_w, 3), 32, dtype=np.uint8)
        
        for idx, sample in enumerate(self.inference_samples):
            if idx >= max_items: break
            
            row = idx // num_cols
            col = idx % num_cols
            
            # Position
            x_start = padding + col * (cell_w + padding)
            y_start = padding + row * (cell_h + padding)
            
            # Draw selection highlight border if selected
            if idx in self.inference_selected_indices:
                # Draw thick green border
                cv2.rectangle(grid_img, 
                             (x_start - 4, y_start - 4), 
                             (x_start + cell_w + 4, y_start + cell_h + 4), 
                             (0, 255, 0), 4)  # Green border, 4px thick
            
            # Choose frame or prediction
            if self.show_inference_prediction:
                img_data = sample['prediction'].copy()
                
                if self.show_inference_lines_var.get():
                    detected_lines = sample.get('detected_lines', [])
                    if detected_lines:
                        img_h, img_w = img_data.shape[:2]
                        thickness = max(1, min(img_w, img_h) // 200)
                        endpoint_size = max(2, thickness + 1)
                        
                        # Check if using HoughLinesP + Cluster mode - show regions
                        line_mode = self.line_detect_mode_var.get()
                        if line_mode == "HoughLinesP + Cluster":
                            # Draw lines first
                            img_data = self._draw_lines_on_image(
                                img_data, 
                                detected_lines, 
                                color=(0, 255, 255),
                                thickness=thickness,
                                endpoint_size=endpoint_size
                            )
                            # Then draw regions with numbering
                            img_data = self._draw_regions_on_image(img_data, detected_lines)
                        else:
                            img_data = self._draw_lines_on_image(
                                img_data, 
                                detected_lines, 
                                color=(0, 255, 255),
                                thickness=thickness,
                                endpoint_size=endpoint_size
                            )
            else:
                img_data = sample['frame'].copy()
                
                # Also show regions on original frame if lines are visible
                if self.show_inference_lines_var.get():
                    detected_lines = sample.get('detected_lines', [])
                    line_mode = self.line_detect_mode_var.get()
                    if detected_lines and line_mode == "HoughLinesP + Cluster":
                        img_h, img_w = img_data.shape[:2]
                        thickness = max(1, min(img_w, img_h) // 200)
                        endpoint_size = max(2, thickness + 1)
                        
                        img_data = self._draw_lines_on_image(
                            img_data, 
                            detected_lines, 
                            color=(0, 255, 255),
                            thickness=thickness,
                            endpoint_size=endpoint_size
                        )
                        img_data = self._draw_regions_on_image(img_data, detected_lines)
            
            # Resize
            img_h, img_w = img_data.shape[:2]
            scale = min(cell_w / img_w, cell_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            
            resized = cv2.resize(img_data, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Centered within cell
            img_x = x_start + (cell_w - new_w) // 2
            img_y = y_start + (cell_h - new_h) // 2
            
            grid_img[img_y:img_y+new_h, img_x:img_x+new_w] = resized
        
        self._draw_inference_image(grid_img)
    
    def on_inference_click(self, event):
        """Handle click on inference grid to select/deselect items."""
        if not hasattr(self, 'inference_grid_geometry') or not self.inference_samples:
            return
        if not hasattr(self, 'inference_display_offset') or not hasattr(self, 'inference_display_scale'):
            return
            
        geom = self.inference_grid_geometry
        
        # Get display transform
        offset_x, offset_y = self.inference_display_offset
        scale = self.inference_display_scale
        
        # Convert canvas click to original image coordinates
        # First subtract display offset, then divide by scale
        x_in_scaled = event.x - offset_x
        y_in_scaled = event.y - offset_y
        
        # Check if click is within the displayed image
        if hasattr(self, 'inference_photo_image'):
            img_w = self.inference_photo_image.width()
            img_h = self.inference_photo_image.height()
            if x_in_scaled < 0 or x_in_scaled >= img_w or y_in_scaled < 0 or y_in_scaled >= img_h:
                return  # Click outside image
        
        # Convert to original (unscaled) image coordinates
        x_orig = x_in_scaled / scale
        y_orig = y_in_scaled / scale
        
        # Now convert to grid cell
        padding = geom['padding']
        cell_w = geom['cell_w']
        cell_h = geom['cell_h']
        num_cols = geom['num_cols']
        
        # Calculate which cell (accounting for padding between cells)
        # Each cell occupies: padding + cell_w, and starts at padding + col*(cell_w + padding)
        col = int((x_orig - padding) / (cell_w + padding))
        row = int((y_orig - padding) / (cell_h + padding))
        
        # Verify click is actually within a cell (not in padding)
        cell_x_start = padding + col * (cell_w + padding)
        cell_y_start = padding + row * (cell_h + padding)
        
        if x_orig < cell_x_start or x_orig >= cell_x_start + cell_w:
            return  # Click in horizontal padding
        if y_orig < cell_y_start or y_orig >= cell_y_start + cell_h:
            return  # Click in vertical padding
        
        if 0 <= col < num_cols and row >= 0:
            idx = int(row * num_cols + col)
            
            if 0 <= idx < len(self.inference_samples) and idx < geom['max_items']:
                # Toggle selection
                if idx in self.inference_selected_indices:
                    self.inference_selected_indices.remove(idx)
                else:
                    self.inference_selected_indices.add(idx)
                
                self.display_inference_sample()
    
    def on_inference_right_click(self, event):
        """Handle right click on inference grid - shows context menu if items are selected."""
        if not self.inference_samples:
            return
        
        # Only show context menu if we have selections (don't auto-select on right click)
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
                
                # Create a new sample object
                # Note: We need crop_rect (x, y, w, h)
                # For Random Regions mode, we have this implicitly via location and size
                # For Full Frame, we use the whole frame (0, 0, w, h)
                
                video_path = None
                # Find full path from basename
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
                    # Full frame (no location stored usually means 0,0)
                    crop_rect = (0, 0, w, h)
                
                # Add to database
                self.db.find_sample(video_path, frame_idx, crop_rect)
                
                # Also add to history so it shows up immediately in labelling tab
                # Use the frame data we already have
                frame_rgb = sample_data['frame']  # Already RGB
                
                history_entry = {
                    "video_path": video_path,
                    "frame_idx": frame_idx,
                    "crop_rect": crop_rect,
                    "image": frame_rgb
                }
                self.history.append(history_entry)
                
                count += 1
        
        # Save database
        self.db.save("line_annotations.json")
        
        # Update statistics in labelling tab
        self.update_statistics()
        
        # Clear selection
        self.inference_selected_indices.clear()
        self.display_inference_sample()
        
        # Notify user
        messagebox.showinfo("Success", f"Added {count} of {num_selected} selected samples to labeling queue.\n\nGo to Labelling tab to see them (they're at the end).")
        
        # Reload history to include new samples immediately
        # We can append them to history directly to avoid restart
        # But _load_history_from_db is better to be consistent
        # For now, just saving is enough, user can reload or we can append manually
        # Let's append to history so they show up
        
        # Actually, best to just reload history from DB
        # But that's heavy. Let's just let user know.
    
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
        
        # Store display transform for click detection
        self.inference_display_offset = (offset_x, offset_y)
        self.inference_display_scale = scale
        
        self.inference_canvas.delete("all")
        self.inference_canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.inference_photo_image)
    
    def toggle_inference_view(self):
        """Toggle between images and predictions view."""
        self.show_inference_prediction = not self.show_inference_prediction
        self.display_inference_sample()
    
    # ==================== Tile Detector Tab Methods ====================
    
    def _build_tile_detector_tab(self):
        """Build the UI for the tile detector tab."""
        # Main frame with canvas and sidebar
        main_frame = ttk.Frame(self.tile_detector_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas Area (Left)
        self.tile_canvas = tk.Canvas(main_frame, bg="#222222", highlightthickness=0)
        self.tile_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Sidebar (Right)
        sidebar = ttk.Frame(main_frame, width=300, padding=10)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        # Model status
        model_frame = ttk.LabelFrame(sidebar, text="Model", padding=10)
        model_frame.pack(fill=tk.X, pady=(10, 10))
        
        self.lbl_tile_model = ttk.Label(model_frame, text="Model: Loading...")
        self.lbl_tile_model.pack(anchor="w", pady=5)
        
        # Video selection
        video_frame = ttk.LabelFrame(sidebar, text="Video Selection", padding=10)
        video_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(video_frame, text="Video:").pack(anchor="w", pady=2)
        
        # Get video basenames for dropdown
        video_names = [os.path.basename(vp) for vp in self.video_paths]
        self.tile_video_var = tk.StringVar(value=video_names[0] if video_names else "")
        self.tile_video_combo = ttk.Combobox(video_frame, textvariable=self.tile_video_var,
                                              values=video_names, state="readonly", width=25)
        self.tile_video_combo.pack(fill=tk.X, pady=5)
        self.tile_video_combo.bind("<<ComboboxSelected>>", self._on_tile_video_changed)
        
        # Time slider
        time_frame = ttk.LabelFrame(sidebar, text="Frame Selection", padding=10)
        time_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.tile_frame_var = tk.IntVar(value=0)
        self.tile_frame_slider = ttk.Scale(time_frame, from_=0, to=100, 
                                           variable=self.tile_frame_var,
                                           orient="horizontal",
                                           command=self._on_tile_slider_changed)
        self.tile_frame_slider.pack(fill=tk.X, pady=5)
        
        self.lbl_tile_frame = ttk.Label(time_frame, text="Frame: 0 / 0")
        self.lbl_tile_frame.pack(anchor="w", pady=2)
        
        # Time display
        self.lbl_tile_time = ttk.Label(time_frame, text="Time: 0:00.0")
        self.lbl_tile_time.pack(anchor="w", pady=2)
        
        # View options
        view_frame = ttk.LabelFrame(sidebar, text="View Options", padding=10)
        view_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.tile_show_prediction_var = tk.BooleanVar(value=False)
        show_pred_check = ttk.Checkbutton(view_frame, text="Show prediction mask",
                                          variable=self.tile_show_prediction_var,
                                          command=self._display_tile_frame)
        show_pred_check.pack(fill=tk.X, pady=2)
        
        self.tile_show_lines_var = tk.BooleanVar(value=True)
        show_lines_check = ttk.Checkbutton(view_frame, text="Show detected lines",
                                           variable=self.tile_show_lines_var,
                                           command=self._display_tile_frame)
        show_lines_check.pack(fill=tk.X, pady=2)
        
        self.tile_show_regions_var = tk.BooleanVar(value=True)
        show_regions_check = ttk.Checkbutton(view_frame, text="Show numbered regions",
                                             variable=self.tile_show_regions_var,
                                             command=self._display_tile_frame)
        show_regions_check.pack(fill=tk.X, pady=2)
        
        # Detection info
        info_frame = ttk.LabelFrame(sidebar, text="Detection Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_tile_lines = ttk.Label(info_frame, text="Lines: 0")
        self.lbl_tile_lines.pack(anchor="w", pady=2)
        
        self.lbl_tile_regions = ttk.Label(info_frame, text="Regions: 0")
        self.lbl_tile_regions.pack(anchor="w", pady=2)
    
    def _load_tile_detector_model(self):
        """Load the model for tile detection (silent, no dialog)."""
        try:
            if self.model is None:
                self.model = MobileUNet(pretrained=False).to(self.device)
            
            model_path = "line_detector_unet_best.pth"
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.lbl_tile_model.config(text="Model: Loaded ✓", foreground="green")
            else:
                self.lbl_tile_model.config(text="Model: Not found", foreground="red")
        except Exception as e:
            self.lbl_tile_model.config(text=f"Model: Error - {str(e)[:20]}", foreground="red")
        
        # Initialize video after model loads
        if self.tile_video_var.get():
            self._on_tile_video_changed()
    
    def _on_tile_video_changed(self, event=None):
        """Handle video selection change."""
        video_name = self.tile_video_var.get()
        if not video_name:
            return
        
        # Find full path
        video_path = None
        for vp in self.video_paths:
            if os.path.basename(vp) == video_name:
                video_path = vp
                break
        
        if not video_path:
            return
        
        # Close previous capture
        if self.tile_cap is not None:
            self.tile_cap.release()
        
        # Open new video
        self.tile_cap = cv2.VideoCapture(video_path)
        self.tile_total_frames = int(self.tile_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.tile_fps = self.tile_cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Update slider range
        self.tile_frame_slider.config(to=max(1, self.tile_total_frames - 1))
        self.tile_frame_var.set(0)
        
        # Load first frame
        self._load_and_predict_frame(0)
    
    def _on_tile_slider_changed(self, value):
        """Handle slider change with debouncing."""
        # Cancel previous pending update
        if self.tile_slider_debounce is not None:
            self.root.after_cancel(self.tile_slider_debounce)
        
        # Schedule new update after 150ms
        self.tile_slider_debounce = self.root.after(150, self._do_tile_slider_update)
    
    def _do_tile_slider_update(self):
        """Actually load and predict the frame after debounce."""
        self.tile_slider_debounce = None
        frame_idx = int(self.tile_frame_var.get())
        self._load_and_predict_frame(frame_idx)
    
    def _load_and_predict_frame(self, frame_idx):
        """Load a frame and run prediction on it."""
        if self.tile_cap is None or self.model is None:
            return
        
        # Seek to frame
        self.tile_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.tile_cap.read()
        
        if not ret or frame is None:
            return
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.tile_current_frame = frame_rgb
        
        # Update frame label
        self.lbl_tile_frame.config(text=f"Frame: {frame_idx} / {self.tile_total_frames}")
        
        # Update time label
        time_sec = frame_idx / self.tile_fps
        minutes = int(time_sec // 60)
        seconds = time_sec % 60
        self.lbl_tile_time.config(text=f"Time: {minutes}:{seconds:05.2f}")
        
        # Run prediction
        self._predict_current_frame()
        
        # Display
        self._display_tile_frame()
    
    def _predict_current_frame(self):
        """Run line detection on current frame."""
        if self.tile_current_frame is None or self.model is None:
            return
        
        frame = self.tile_current_frame
        h, w = frame.shape[:2]
        
        # Prepare for model (pad to multiple of 32)
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        
        if pad_h > 0 or pad_w > 0:
            frame_padded = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            frame_padded = frame
        
        # Run inference
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).reshape(1, 3, 1, 1)
        
        with torch.no_grad():
            input_tensor = torch.from_numpy(frame_padded.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            input_tensor = (input_tensor.to(self.device) - mean) / std
            prediction = self.model(input_tensor)
            pred_np = prediction.squeeze().cpu().numpy()
        
        # Crop to original size
        pred_mask = (pred_np[:h, :w] * 255).astype(np.uint8)
        self.tile_current_prediction = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
        
        # Detect lines using HoughLinesP + Cluster
        self.tile_current_lines = self._detect_lines_hough_clustered(
            self.tile_current_prediction,
            min_length=self.line_min_length_var.get(),
            max_lines=self.line_max_lines_var.get(),
            threshold=self.hough_threshold_var.get(),
            max_gap=self.hough_max_gap_var.get(),
            cluster_angle_deg=self.line_cluster_angle_var.get(),
            cluster_dist=self.line_merge_dist_var.get()
        )
        
        # Update info
        self.lbl_tile_lines.config(text=f"Lines: {len(self.tile_current_lines)}")
    
    def _display_tile_frame(self):
        """Display the current frame with overlays."""
        if self.tile_current_frame is None:
            return
        
        # Choose base image
        if self.tile_show_prediction_var.get() and self.tile_current_prediction is not None:
            img_display = self.tile_current_prediction.copy()
        else:
            img_display = self.tile_current_frame.copy()
        
        img_h, img_w = img_display.shape[:2]
        
        # Draw lines if enabled
        if self.tile_show_lines_var.get() and self.tile_current_lines:
            thickness = max(1, min(img_w, img_h) // 300)
            endpoint_size = max(2, thickness + 1)
            img_display = self._draw_lines_on_image(
                img_display,
                self.tile_current_lines,
                color=(0, 255, 255),
                thickness=thickness,
                endpoint_size=endpoint_size
            )
        
        # Draw regions if enabled
        num_regions = 0
        if self.tile_show_regions_var.get() and self.tile_current_lines:
            # Find and draw regions
            intersections = self._find_line_intersections(self.tile_current_lines, img_w, img_h)
            regions = self._find_quadrilateral_regions(self.tile_current_lines, intersections, img_w, img_h)
            num_regions = len(regions)
            
            if regions:
                img_display = self._draw_regions_on_image(img_display, self.tile_current_lines, regions)
        
        self.lbl_tile_regions.config(text=f"Regions: {num_regions}")
        
        # Display on canvas
        self._draw_tile_image(img_display)
    
    def _draw_tile_image(self, img_arr):
        """Draw image on tile detector canvas."""
        img_h, img_w = img_arr.shape[:2]
        
        canvas_w = self.tile_canvas.winfo_width()
        canvas_h = self.tile_canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            self.root.after(100, lambda: self._draw_tile_image(img_arr))
            return
        
        # Scale to fit canvas
        scale = min(canvas_w / img_w, canvas_h / img_h) * 0.95
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        # Resize
        resized = cv2.resize(img_arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(resized)
        self.tile_photo_image = ImageTk.PhotoImage(img_pil)
        
        # Center on canvas
        offset_x = (canvas_w - new_w) // 2
        offset_y = (canvas_h - new_h) // 2
        
        self.tile_canvas.delete("all")
        self.tile_canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.tile_photo_image)
    
    # ==================== End Tile Detector Tab Methods ====================
    
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

def train_cli(video_paths, num_samples, batch_size, epochs, lr, model_name="line_detector_unet", use_amp=True):
    """
    CLI training mode - no GUI, verbose console output.
    """
    import time
    
    print("=" * 60)
    print("LINE DETECTOR - CLI TRAINING MODE")
    print("=" * 60)
    
    # Load annotation database
    db_path = "line_annotations.json"
    if os.path.exists(db_path):
        db = AnnotationDatabase.load(db_path)
        print(f"[INFO] Loaded annotation database: {len(db.samples)} samples")
    else:
        print(f"[ERROR] No annotation database found at '{db_path}'")
        print("[ERROR] Please run the GUI first to create annotations.")
        return False
    
    # Use all samples
    all_samples = db.samples
    print(f"[INFO] Using {len(all_samples)} samples")
    
    if not all_samples:
        print("[ERROR] No samples available for training!")
        return False
    
    # Build a mapping from video basenames to full paths
    # Handle both Windows and Unix path separators
    def get_basename(path):
        # Split on both \ and / to handle cross-platform paths
        return path.replace('\\', '/').split('/')[-1]
    
    video_path_map = {}
    for vp in video_paths:
        basename = get_basename(vp)
        video_path_map[basename] = vp
    
    print(f"[INFO] Found {len(video_path_map)} videos in folder")
    
    # Pre-load frames
    print(f"\n[STEP 1/4] Pre-loading frames...")
    frame_cache = {}
    path_warnings = set()
    
    for i, sample in enumerate(all_samples):
        # Resolve video path - try original first, then by basename
        video_path = sample.video_path
        if not os.path.exists(video_path):
            basename = get_basename(video_path)
            if basename in video_path_map:
                video_path = video_path_map[basename]
            else:
                if basename not in path_warnings:
                    print(f"  [WARN] Video not found: {basename}")
                    path_warnings.add(basename)
                continue
        
        cache_key = (video_path, sample.frame_idx)
        if cache_key not in frame_cache:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                frame_cache[cache_key] = frame
                # Store resolved path for later use
                sample._resolved_path = video_path
        else:
            sample._resolved_path = video_path
            
        if (i + 1) % 50 == 0 or i == len(all_samples) - 1:
            print(f"  Loaded {i+1}/{len(all_samples)} samples ({len(frame_cache)} unique frames)")
    
    print(f"[INFO] Cached {len(frame_cache)} unique frames")
    
    # Generate training samples
    print(f"\n[STEP 2/4] Generating {num_samples} augmented training samples...")
    samples = []
    patch_size = 128
    
    start_time = time.time()
    # Filter to only samples with resolved paths
    valid_samples = [s for s in all_samples if hasattr(s, '_resolved_path')]
    
    if not valid_samples:
        print("[ERROR] No valid samples with accessible video files!")
        return False
    
    print(f"[INFO] {len(valid_samples)} samples have accessible video files")
    
    for i in range(num_samples):
        sample = random.choice(valid_samples)
        cache_key = (sample._resolved_path, sample.frame_idx)
        
        if cache_key not in frame_cache:
            continue
        
        frame = frame_cache[cache_key]
        x, y, w, h = sample.crop_rect
        region = frame[y:y+h, x:x+w]
        
        # Apply augmentation (simplified version of _generate_single_patch)
        region_h, region_w = region.shape[:2]
        
        # Random zoom
        zoom = random.uniform(0.7, 1.3)
        new_size = max(patch_size, int(min(region_w, region_h) * zoom))
        
        # Random crop position
        if region_w > patch_size:
            crop_x = random.randint(0, region_w - patch_size)
        else:
            crop_x = 0
        if region_h > patch_size:
            crop_y = random.randint(0, region_h - patch_size)
        else:
            crop_y = 0
        
        patch = region[crop_y:crop_y+patch_size, crop_x:crop_x+patch_size]
        
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            patch = cv2.resize(patch, (patch_size, patch_size))
        
        # Create mask
        mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
        
        for line in sample.lines:
            p1 = (int(line.start[0] - crop_x), int(line.start[1] - crop_y))
            p2 = (int(line.end[0] - crop_x), int(line.end[1] - crop_y))
            cv2.line(mask, p1, p2, 255, thickness=1)
        
        # Random augmentations
        if random.random() < 0.5:
            patch = cv2.flip(patch, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < 0.5:
            patch = cv2.flip(patch, 0)
            mask = cv2.flip(mask, 0)
        
        # Brightness/contrast
        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-20, 20)
        patch = np.clip(alpha * patch + beta, 0, 255).astype(np.uint8)
        
        # Convert to RGB
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        samples.append({'image': patch_rgb, 'mask': mask_rgb})
        
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (num_samples - i - 1) / rate
            print(f"  Generated {i+1}/{num_samples} samples ({rate:.1f} samples/sec, ~{remaining:.1f}s remaining)")
    
    print(f"[INFO] Generated {len(samples)} training samples in {time.time() - start_time:.1f}s")
    
    # Split train/val
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    print(f"[INFO] Train: {len(train_samples)}, Validation: {len(val_samples)}")
    
    # Create datasets and loaders
    print(f"\n[STEP 3/4] Initializing model and data loaders...", flush=True)
    train_dataset = LineDataset(train_samples)
    val_dataset = LineDataset(val_samples)
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}", flush=True)
    
    # Check GPU info and print details
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[INFO] GPU: {gpu_name} ({gpu_mem:.1f} GB)", flush=True)
    
    # Mixed precision training for faster GPU performance
    use_amp = use_amp and device.type == 'cuda'
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print(f"[INFO] Mixed Precision (AMP): ENABLED ✓", flush=True)
    else:
        scaler = None
        if device.type == 'cuda':
            print(f"[INFO] Mixed Precision (AMP): DISABLED (--no-amp flag)", flush=True)
        else:
            print(f"[INFO] Mixed Precision (AMP): Disabled (CPU mode)", flush=True)
    
    # Use more workers on Linux/Colab for faster data loading
    num_workers = 4 if device.type == 'cuda' else 0
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True, persistent_workers=num_workers>0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True, persistent_workers=num_workers>0)
    
    print(f"[INFO] Batch size: {batch_size}", flush=True)
    print(f"[INFO] Num workers: {num_workers}", flush=True)
    print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)
    
    model = MobileUNet(pretrained=True).to(device)
    print("[INFO] Loaded MobileNetV2 backbone (pretrained on ImageNet)", flush=True)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"[INFO] Loss: BCE + Dice (combined)", flush=True)
    print(f"[INFO] Optimizer: Adam (lr={lr}, weight_decay=1e-4)", flush=True)
    print(f"[INFO] Scheduler: CosineAnnealingLR (T_max={epochs})", flush=True)
    print(f"[INFO] Output model: {model_name}.pth, {model_name}_best.pth", flush=True)
    
    # Dice loss helper
    def dice_loss(pred, target, smooth=1e-8):
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    # Training loop
    print(f"\n[STEP 4/4] Training for {epochs} epochs...")
    print("-" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                
                # Compute loss OUTSIDE autocast (BCELoss not safe with autocast)
                outputs_f32 = outputs.float()
                masks_f32 = masks.float()
                bce = criterion(outputs_f32, masks_f32)
                dice = dice_loss(outputs_f32, masks_f32)
                loss = 0.5 * bce + 0.5 * dice
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                bce = criterion(outputs, masks)
                dice = dice_loss(outputs, masks)
                loss = 0.5 * bce + 0.5 * dice
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                    # Compute loss outside autocast
                    outputs_f32 = outputs.float()
                    masks_f32 = masks.float()
                    bce = criterion(outputs_f32, masks_f32)
                    dice = dice_loss(outputs_f32, masks_f32)
                    loss = 0.5 * bce + 0.5 * dice
                else:
                    outputs = model(images)
                    bce = criterion(outputs, masks)
                    dice = dice_loss(outputs, masks)
                    loss = 0.5 * bce + 0.5 * dice
                    
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - epoch_start
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s | *** BEST - SAVED ***")
        else:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
    
    # Save final model
    torch.save(model.state_dict(), f"{model_name}.pth")
    
    print("-" * 60)
    print(f"[DONE] Training complete!")
    print(f"[INFO] Best validation loss: {best_val_loss:.4f}")
    print(f"[INFO] Models saved: {model_name}.pth, {model_name}_best.pth")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Line annotation and training tool.")
    parser.add_argument("videos", nargs="*", help="Video files or directories")
    parser.add_argument("--video-folder", type=str, help="Path to video folder (alternative to positional argument)")
    parser.add_argument("--test-graph", action="store_true", help="Show training graph with mock data for testing")
    parser.add_argument("--train-samples", type=int, default=25000, help="Number of training samples (default: 25000)")
    
    # CLI training mode arguments
    parser.add_argument("--train", action="store_true", help="Run training in CLI mode (no GUI)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--model-name", type=str, default="line_detector_unet", help="Output model name (default: line_detector_unet)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training (AMP)")
    
    args = parser.parse_args()
    
    # Handle video folder argument
    video_inputs = args.videos
    if args.video_folder:
        video_inputs = [args.video_folder]
    
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
    
    # CLI training mode
    if args.train:
        success = train_cli(
            video_paths=video_paths,
            num_samples=args.train_samples,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            model_name=args.model_name,
            use_amp=not args.no_amp
        )
        return
    
    # GUI mode
    root = tk.Tk()
    root.state('zoomed')  # Fullscreen on Windows
    
    app = RandomPatchViewer(root, video_paths, test_graph=args.test_graph, 
                           default_train_samples=args.train_samples)
    
    root.mainloop()

if __name__ == "__main__":
    main()
