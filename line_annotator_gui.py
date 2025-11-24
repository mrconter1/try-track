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
        
        # UI Setup
        self.root.title(f"Random Patch Viewer ({patch_size}x{patch_size})")
        self._build_ui()
        
        # Bindings
        self.root.bind("<Configure>", self.on_resize)
        self.root.bind("<a>", lambda e: self.prev_patch())
        self.root.bind("<d>", lambda e: self.next_patch())
        self.root.bind("<Left>", lambda e: self.prev_patch())
        self.root.bind("<Right>", lambda e: self.next_patch())
        self.root.bind("<m>", lambda e: self.toggle_mask())
        self.root.bind("<l>", lambda e: self.toggle_lines())
        self.root.bind("<Delete>", lambda e: self.delete_selected_line())
        self.root.bind("<Control-s>", lambda e: self.save_annotations())
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
        
        self._build_labelling_tab()
        self._build_data_generation_tab()
    
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
        ttk.Label(sidebar, text="• Generate 4x4 grid of patches from labeled data\n• Toggle between images and masks view").pack(anchor="w")

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
            
            # Random 128x128 location within the crop
            if w < 128 or h < 128:
                # If crop is smaller than 128x128, use the whole thing
                patch_x, patch_y = 0, 0
                patch_w, patch_h = min(w, 128), min(h, 128)
            else:
                patch_x = random.randint(0, w - 128)
                patch_y = random.randint(0, h - 128)
                patch_w, patch_h = 128, 128
            
            # Extract patch
            patch_img = crop[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
            patch_img_rgb = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)
            
            # Create mask
            mask = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
            
            # Draw lines that intersect with this patch
            for line in sample.lines:
                # Line coords are relative to the crop
                p1_x, p1_y = line.start
                p2_x, p2_y = line.end
                
                # Translate to patch coordinates
                p1_patch = (int(p1_x - patch_x), int(p1_y - patch_y))
                p2_patch = (int(p2_x - patch_x), int(p2_y - patch_y))
                
                # Draw line (even if it goes outside - OpenCV clips it)
                cv2.line(mask, p1_patch, p2_patch, (255, 255, 255), thickness=3)
            
            patch_data = {
                "image": patch_img_rgb,
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
                self.canvas.create_line(
                    canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                    fill="yellow", width=2, dash=(4, 4)
                )
            elif is_selected:
                # Draw green line for selected
                self.canvas.create_line(
                    canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                    fill="lime", width=3
                )
            else:
                # Draw solid cyan line normally
                self.canvas.create_line(
                    canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                    fill="cyan", width=2
                )
            
            # Draw endpoints - highlight if selected
            if is_selected:
                self.draw_point_on_canvas(p1, "lime", 6)
                self.draw_point_on_canvas(p2, "lime", 6)
            else:
                self.draw_point_on_canvas(p1, "lime", 5)
                self.draw_point_on_canvas(p2, "red", 5)
    
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
