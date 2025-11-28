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
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    def __init__(self, root, video_paths, annotations_file="cross_annotations.json", patch_size=400):
        self.root = root
        self.video_paths = [os.path.abspath(p) for p in video_paths]
        self.patch_size = patch_size
        self.annotations_file = annotations_file
        
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
        self.root.bind("<r>", lambda e: self.resample_current_patch())
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
    
    def _load_sample_image(self, sample):
        """Load a single sample image."""
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
        """Load history from saved samples."""
        if not self.db.samples:
            return
        
        print(f"Loading {len(self.db.samples)} history samples...")
        
        max_workers = min(8, len(self.db.samples))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._load_sample_image, sample) for sample in self.db.samples]
            
            for future in futures:
                try:
                    patch_info = future.result()
                    if patch_info:
                        self.history.append(patch_info)
                except Exception as e:
                    print(f"Warning: Error loading sample: {e}")
        
        print(f"Loaded {len(self.history)} samples")
    
    def _build_ui(self):
        """Build the UI."""
        main_frame = ttk.Frame(self.root)
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
        
        # Add new crossing
        self._push_undo_state("Add crossing")
        self.crossings.append((img_x, img_y))
        self.selected_crossing_idx = len(self.crossings) - 1
        self.dragging_idx = None
        self.save_annotations(show_message=False)
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
    parser = argparse.ArgumentParser(description="Crossing point annotation tool")
    parser.add_argument("videos", nargs="*", help="Video files or directories")
    parser.add_argument("--annotations", "-a", default="cross_annotations.json", help="Annotations file")
    parser.add_argument("--patch-size", type=int, default=400, help="Patch size (default: 400)")
    
    args = parser.parse_args()
    
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
    
    app = CrossingAnnotator(root, video_paths, annotations_file=args.annotations, patch_size=args.patch_size)
    
    root.mainloop()


if __name__ == "__main__":
    main()

