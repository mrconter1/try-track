import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import argparse
import os
import random
import json
import threading
import subprocess
import torch
import torch.nn as nn
from torchvision import models, transforms

class GridTool:
    def __init__(self, root, video_path):
        self.root = root
        self.video_path = video_path
        
        # Check if video_path is a directory
        if os.path.isdir(video_path):
            self.video_list = self._build_video_list(video_path)
            if not self.video_list:
                raise IOError(f"No video files found in directory: {video_path}")
            self.current_video_idx = 0
            self.cap = cv2.VideoCapture(self.video_list[0][0])
            if not self.cap.isOpened():
                raise IOError(f"Cannot open first video from directory: {video_path}")
        else:
            if not os.path.isfile(video_path):
                raise IOError(f"Video file not found: {video_path}")
            self.cap = cv2.VideoCapture(video_path)
            self.video_list = None
            if not self.cap.isOpened():
                raise IOError(f"Cannot open video file: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0
        
        self.grid_subdiv_x = tk.IntVar(value=1)
        self.grid_subdiv_y = tk.IntVar(value=1)
        self.low_res_preview_var = tk.BooleanVar(value=False)
        self.has_grid_var = tk.BooleanVar(value=False)  # Default: No Grid

        self.root.title("Grid Tool - Click to place 4 corner points")
        self.root.state('zoomed')
        self._build_ui()
    
    def _build_video_list(self, directory):
        """Build list of (video_path, frame_count) tuples from directory."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        video_list = []
        
        print(f"\n[Grid Tool] Scanning directory: {directory}")
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in video_extensions:
                    cap = cv2.VideoCapture(file_path)
                    if cap.isOpened():
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if frame_count > 0:
                            video_list.append((file_path, frame_count))
                            print(f"  [+] {filename}: {frame_count} frames")
                        cap.release()
        
        video_list = sorted(video_list)
        total_frames = sum(fc for _, fc in video_list)
        print(f"[Grid Tool] Found {len(video_list)} video(s), {total_frames} total frames\n")
        
        return video_list
    
    def _build_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(main_frame, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # Side panel for grid settings
        side_panel = ttk.Frame(main_frame, width=200, padding="5")
        side_panel.grid(row=0, column=1, sticky="ns", padx=5)
        
        ttk.Label(side_panel, text="Grid Settings", font=("Arial", 12, "bold")).pack(pady=10)
        
        ttk.Checkbutton(side_panel, text="Has Grid", variable=self.has_grid_var, command=self._on_has_grid_toggle).pack(anchor=tk.W, pady=5)
        ttk.Checkbutton(side_panel, text="256x256 Preview", variable=self.low_res_preview_var, command=self._display_frame).pack(anchor=tk.W, pady=5)
        
        ttk.Label(side_panel, text="Horizontal Subdivisions:").pack(anchor=tk.W, pady=(10, 0))
        subdiv_x_spinbox = ttk.Spinbox(side_panel, from_=1, to=20, textvariable=self.grid_subdiv_x, command=self._display_frame)
        subdiv_x_spinbox.pack(fill=tk.X, pady=5)
        subdiv_x_spinbox.bind("<FocusIn>", lambda e: side_panel.focus_set())
        
        ttk.Label(side_panel, text="Vertical Subdivisions:").pack(anchor=tk.W, pady=(10, 0))
        subdiv_y_spinbox = ttk.Spinbox(side_panel, from_=1, to=20, textvariable=self.grid_subdiv_y, command=self._display_frame)
        subdiv_y_spinbox.pack(fill=tk.X, pady=5)
        subdiv_y_spinbox.bind("<FocusIn>", lambda e: side_panel.focus_set())

        # Stats Display
        ttk.Label(side_panel, text="Statistics:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(15, 5))
        self.stats_label = ttk.Label(side_panel, text="", font=("Arial", 9), justify=tk.LEFT)
        self.stats_label.pack(anchor=tk.W, pady=5)
        
        # Normalized Coordinates Display
        ttk.Label(side_panel, text="Normalized Coordinates:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(15, 5))
        self.coord_text = tk.Text(side_panel, height=8, width=25, font=("Consolas", 9))
        self.coord_text.pack(fill=tk.X, pady=5)
        self.coord_text.insert("1.0", "Move points to see\ncoordinates...")
        self.coord_text.configure(state="disabled")

        # History List
        ttk.Label(side_panel, text="History:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(15, 5))
        
        list_frame = ttk.Frame(side_panel)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.history_listbox = tk.Listbox(list_frame, font=("Consolas", 8), height=10, yscrollcommand=scrollbar.set)
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.history_listbox.yview)
        
        self.history_listbox.bind('<<ListboxSelect>>', self._on_history_select)

        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=5, padx=5)
        
        self.info_label = ttk.Label(controls_frame, text="Click to place 4 corner points of grid cell")
        self.info_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.reset_button = ttk.Button(controls_frame, text="Reset Points", command=self.reset_points)
        self.reset_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.clear_data_button = ttk.Button(controls_frame, text="Clear All Data", command=self.clear_all_data)
        self.clear_data_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.train_button = ttk.Button(controls_frame, text="Train Model", command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.auto_label_button = ttk.Button(controls_frame, text="Auto Label (p)", command=self._predict_grid)
        self.auto_label_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Frame navigation
        nav_frame = ttk.Frame(controls_frame)
        nav_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Button(nav_frame, text="< Previous", command=self.prev_frame).pack(side=tk.LEFT, padx=5)
        
        self.frame_label = ttk.Label(nav_frame, text="Frame: 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(nav_frame, text="Next >", command=self.next_frame).pack(side=tk.LEFT, padx=5)
        
        self.root.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._canvas_click)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self.root.bind("r", self._random_frame)
        self.root.bind("R", self._random_frame)
        self.root.bind("a", self._prev_in_history)
        self.root.bind("d", self._next_or_random_frame)
        self.root.bind("1", self._decrease_subdiv_x)
        self.root.bind("2", self._increase_subdiv_x)
        self.root.bind("3", self._decrease_subdiv_y)
        self.root.bind("4", self._increase_subdiv_y)
        self.root.bind("g", self._toggle_has_grid)
        self.root.bind("G", self._toggle_has_grid)
        self.root.bind("p", self._predict_grid)
        self.root.bind("P", self._predict_grid)
        
        # Grid state
        self.grid_points = [None, None, None, None]
        self.dragging_point = None
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.canvas_scale = 1.0
        self.photo_image = None
        self.current_frame = None

        # Magnifier state
        self.magnifier_widget = None
        self.magnifier_size = 150  # Diameter in pixels
        self.magnifier_zoom = 4
        self.drag_start_pos = None
        self.drag_start_point_pos = None
    
        # Frame history for a/d navigation: [(video_path, frame_idx), ...]
        if self.video_list:
            initial_video = self.video_list[0][0]
        else:
            initial_video = self.video_path
        self.frame_history = [(initial_video, 0)]  # Start with frame 0
        self.history_index = 0
        
        # Store grid config per frame: {(video_path, frame_idx): (points, subdiv_x, subdiv_y)}
        self.frame_grid_config = {}
        
        # Inference model
        self.model = None
        self.device = torch.device("cpu") # Keep inference on CPU for simplicity/interactivity
        
        # Load persistent state
        self._load_persistent_state()
        
        # Load the initial frame (last from history if available, else 0)
        if self.frame_history:
            # Ensure we start at the end of the list as requested
            self.history_index = len(self.frame_history) - 1
            video_path, frame_idx = self.frame_history[self.history_index]
            self._switch_to_video_and_frame(video_path, frame_idx)
        else:
            self._load_frame(0)
            
        self._update_history_list()
    
    def _load_frame(self, frame_idx, auto_save_new=False):
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return
        
        self.current_frame_idx = frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame = frame
            # Try to load saved config; if not found, use default
            if not self._load_frame_config():
                h, w = frame.shape[:2]
                # Default cell in center of frame
                cell_w = w // 4
                cell_h = h // 4
                center_x = w // 2
                center_y = h // 2
                self.grid_points = [
                    (center_x - cell_w // 2, center_y - cell_h // 2),  # P1 (top-left)
                    (center_x + cell_w // 2, center_y - cell_h // 2),  # P2 (top-right)
                    (center_x - cell_w // 2, center_y + cell_h // 2),  # P3 (bottom-left)
                    (center_x + cell_w // 2, center_y + cell_h // 2),  # P4 (bottom-right)
                ]
                self.grid_subdiv_x.set(1)
                self.grid_subdiv_y.set(1)
                self.has_grid_var.set(False)  # Default: No Grid
                
                # Auto-save new random frames
                if auto_save_new:
                    self._save_frame_config()
            
            self._display_frame()
            self.frame_label.config(text=f"Frame: {frame_idx}/{self.total_frames-1}")
    
    def prev_frame(self):
        self._load_frame(self.current_frame_idx - 1)
    
    def next_frame(self):
        self._load_frame(self.current_frame_idx + 1)
    
    def reset_points(self):
        self.grid_points = [None, None, None, None]
        self.dragging_point = None
        self._display_frame()
    
    def clear_all_data(self):
        """Clear all saved frame configurations and history."""
        # Ask for confirmation
        import tkinter.messagebox as messagebox
        result = messagebox.askyesno(
            "Clear All Data",
            "This will clear all saved grid configurations and frame history.\n\nAre you sure you want to continue?",
            icon='warning'
        )
        
        if result:
            # Clear in-memory data
            self.frame_grid_config.clear()
            if self.video_list:
                initial_video = self.video_list[0][0]
            else:
                initial_video = self.video_path
            self.frame_history = [(initial_video, 0)]
            self.history_index = 0
            
            # Clear persistent state file
            state_file = self._get_state_file()
            if os.path.exists(state_file):
                os.remove(state_file)
            
            # Reset current frame to defaults
            h, w = self.current_frame.shape[:2]
            cell_w = w // 4
            cell_h = h // 4
            center_x = w // 2
            center_y = h // 2
            self.grid_points = [
                (center_x - cell_w // 2, center_y - cell_h // 2),
                (center_x + cell_w // 2, center_y - cell_h // 2),
                (center_x - cell_w // 2, center_y + cell_h // 2),
                (center_x + cell_w // 2, center_y + cell_h // 2),
            ]
            self.grid_subdiv_x.set(1)
            self.grid_subdiv_y.set(1)
            self.has_grid_var.set(False)
            
            self._display_frame()
            print("[Grid Tool] All data cleared.")
    
    def _random_frame(self, event=None):
        """Jump to a random frame in the video."""
        # If the user is typing in a widget, ignore the hotkey
        if event and isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox)):
            return

        # Reset grid points and subdivisions
        self.grid_points = [None, None, None, None]
        self.grid_subdiv_x.set(1)
        self.grid_subdiv_y.set(1)

        random_idx = random.randint(0, self.total_frames - 1)
        self._load_frame(random_idx, auto_save_new=True)
    
    def _prev_in_history(self, event=None):
        """Go to previous frame in history (a key)."""
        if event and isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox)):
            return
        
        if self.history_index > 0:
            self.history_index -= 1
            video_path, frame_idx = self.frame_history[self.history_index]
            self._switch_to_video_and_frame(video_path, frame_idx)
            self._save_persistent_state()
    
    def _next_or_random_frame(self, event=None):
        """Go to next frame in history, or pick a new random one if at end (d key)."""
        if event and isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox)):
            return
        
        if self.history_index < len(self.frame_history) - 1:
            # Navigate forward in existing history
            self.history_index += 1
            video_path, frame_idx = self.frame_history[self.history_index]
            self._switch_to_video_and_frame(video_path, frame_idx, auto_save_new=False)
        else:
            # At the end, pick a new random frame
            if self.video_list:
                # Sample from all videos proportionally to frame count
                video_path, frame_idx = self._sample_random_frame_proportional()
            else:
                video_path = self.video_path
                frame_idx = random.randint(0, self.total_frames - 1)
            self.frame_history.append((video_path, frame_idx))
            self.history_index += 1
            self._switch_to_video_and_frame(video_path, frame_idx, auto_save_new=True)
        
        self._save_persistent_state()
    
    def _sample_random_frame_proportional(self):
        """Sample a random frame from all videos, proportional to frame counts. Returns (video_path, frame_idx)."""
        # Use weighted sampling based on frame counts
        video_paths = [vp for vp, _ in self.video_list]
        frame_counts = [fc for _, fc in self.video_list]
        
        # Pick a random video weighted by frame count
        video_path = random.choices(video_paths, weights=frame_counts, k=1)[0]
        
        # Get frame count for selected video
        frame_count = next(fc for vp, fc in self.video_list if vp == video_path)
        
        # Pick random frame in that video
        frame_idx = random.randint(0, frame_count - 1)
        
        # Reset grid points and subdivisions
        self.grid_points = [None, None, None, None]
        self.grid_subdiv_x.set(1)
        self.grid_subdiv_y.set(1)
        
        print(f"[Grid Tool] Sampled: {os.path.basename(video_path)} (frame {frame_idx}/{frame_count})")
        
        return (video_path, frame_idx)
    
    def _switch_to_video_and_frame(self, video_path, frame_idx, auto_save_new=False):
        """Switch to a specific video and frame."""
        # Switch video if needed
        if self.video_list:
            current_video_path = self.video_list[self.current_video_idx][0]
            if video_path != current_video_path:
                print(f"[Grid Tool] Switching to: {os.path.basename(video_path)} (frame {frame_idx})")
                self.cap.release()
                self.cap = cv2.VideoCapture(video_path)
                self.current_video_idx = next(i for i, (vp, _) in enumerate(self.video_list) if vp == video_path)
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Load the frame
        self._load_frame(frame_idx, auto_save_new=auto_save_new)

    def _canvas_click(self, event):
        if self.current_frame is None:
            return
        
        # Convert canvas coordinates to image coordinates
        img_x = (event.x - self.canvas_offset_x) / self.canvas_scale
        img_y = (event.y - self.canvas_offset_y) / self.canvas_scale
        
        # Check if clicking near an existing point to start dragging
        for i, point in enumerate(self.grid_points):
            if point is not None:
                dist = np.sqrt((img_x - point[0])**2 + (img_y - point[1])**2)
                if dist < 20 / self.canvas_scale: # Use scaled tolerance
                    self.dragging_point = i
                    self.drag_start_pos = (event.x, event.y)
                    self.drag_start_point_pos = point
                    self._create_magnifier(event)
                    return
    
    def _canvas_drag(self, event):
        if self.dragging_point is None or self.current_frame is None:
            return
        
        # Calculate mouse movement delta
        dx = event.x - self.drag_start_pos[0]
        dy = event.y - self.drag_start_pos[1]
        
        # Apply movement (1:1 with mouse)
        # We divide by canvas_scale to convert screen pixel delta to image pixel delta
        move_dx = (dx / self.canvas_scale)
        move_dy = (dy / self.canvas_scale)
        
        new_x = self.drag_start_point_pos[0] + move_dx
        new_y = self.drag_start_point_pos[1] + move_dy
        
        # Validate new configuration
        new_points = list(self.grid_points)
        new_points[self.dragging_point] = (new_x, new_y)
        
        if self._is_convex(new_points):
            self.grid_points[self.dragging_point] = (new_x, new_y)
            self._save_frame_config()
        self._update_magnifier(event)
        self._display_frame()
    
    def _is_convex(self, points):
        """
        Check if the quadrilateral formed by P1(TL), P2(TR), P3(BL), P4(BR) is convex.
        Polygon order: P1 -> P2 -> P4 -> P3
        """
        if any(p is None for p in points):
            return False

        p1, p2, p3, p4 = points
        # Polygon vertices in order
        poly = [p1, p2, p4, p3]
        
        def cross_product(a, b, c):
            return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            
        cp1 = cross_product(poly[0], poly[1], poly[2])
        cp2 = cross_product(poly[1], poly[2], poly[3])
        cp3 = cross_product(poly[2], poly[3], poly[0])
        cp4 = cross_product(poly[3], poly[0], poly[1])
        
        # Check if all have the same sign (and are non-zero)
        # Allow small epsilon for collinearity if needed, but strictly > 0 prevents collapse
        return (cp1 > 0 and cp2 > 0 and cp3 > 0 and cp4 > 0) or \
               (cp1 < 0 and cp2 < 0 and cp3 < 0 and cp4 < 0)
    
    def _canvas_release(self, event):
        if self.dragging_point is not None:
            self._destroy_magnifier()
        self.dragging_point = None
        self.drag_start_pos = None
        self.drag_start_point_pos = None
    
    def _create_magnifier(self, event):
        """Create and place the magnifier widget on the canvas."""
        if self.magnifier_widget:
            self._destroy_magnifier()
        
        self.magnifier_widget = tk.Canvas(self.canvas, width=self.magnifier_size, height=self.magnifier_size,
                                          highlightthickness=2, highlightbackground="yellow")
        self.magnifier_widget.place(x=event.x, y=event.y, anchor='center')
        self._update_magnifier(event)

    def _update_magnifier(self, event):
        """Update the content and position of the magnifier."""
        if not self.magnifier_widget or self.current_frame is None:
            return

        # Move the magnifier widget
        self.magnifier_widget.place(x=event.x, y=event.y, anchor='center')
        
        # Calculate the region to capture from the original frame
        img_x, img_y = self.grid_points[self.dragging_point]
        
        patch_size_img_coords = self.magnifier_size / (self.magnifier_zoom * self.canvas_scale)
        patch_radius = patch_size_img_coords / 2
        
        # Desired crop coordinates
        x1 = int(img_x - patch_radius)
        y1 = int(img_y - patch_radius)
        x2 = int(img_x + patch_radius)
        y2 = int(img_y + patch_radius)
        
        desired_w = x2 - x1
        desired_h = y2 - y1
        
        if desired_w <= 0 or desired_h <= 0:
            return

        # Image bounds
        h, w = self.current_frame.shape[:2]
        
        # Intersection with image
        ix1 = max(0, x1)
        iy1 = max(0, y1)
        ix2 = min(w, x2)
        iy2 = min(h, y2)
        
        # Create blank patch (black background)
        patch = np.zeros((desired_h, desired_w, 3), dtype=np.uint8)
        
        # Check if we have any overlap
        if ix1 < ix2 and iy1 < iy2:
            # Extract valid region
            img_patch = self.current_frame[iy1:iy2, ix1:ix2]
            
            # Calculate placement in the blank patch
            px1 = ix1 - x1
            py1 = iy1 - y1
            px2 = px1 + (ix2 - ix1)
            py2 = py1 + (iy2 - iy1)
            
            # Place the valid image part into the patch
            patch[py1:py2, px1:px2] = img_patch
        
        # Resize patch to magnifier size for zoom effect
        zoomed_patch = cv2.resize(patch, (self.magnifier_size, self.magnifier_size), interpolation=cv2.INTER_NEAREST)
        zoomed_patch_rgb = cv2.cvtColor(zoomed_patch, cv2.COLOR_BGR2RGB)
        
        # Create PhotoImage and display it
        self.magnifier_photo = ImageTk.PhotoImage(Image.fromarray(zoomed_patch_rgb))
        self.magnifier_widget.create_image(0, 0, anchor='nw', image=self.magnifier_photo)
        
        # Draw the crosshair
        center = self.magnifier_size / 2
        self.magnifier_widget.delete("crosshair")
        self.magnifier_widget.create_line(center, 0, center, self.magnifier_size, fill='red', width=1, tags="crosshair")
        self.magnifier_widget.create_line(0, center, self.magnifier_size, center, fill='red', width=1, tags="crosshair")
        
    def _destroy_magnifier(self):
        """Destroy the magnifier widget."""
        if self.magnifier_widget:
            self.magnifier_widget.destroy()
            self.magnifier_widget = None

    def _display_frame(self):
        if self.current_frame is None:
            return
        
        # Update statistics
        self._update_stats()
        
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        
        # Handle low-res preview
        if self.low_res_preview_var.get():
             h, w = frame_rgb.shape[:2]
             # Downscale to 256x256 using linear interpolation (mimic network input)
             small = cv2.resize(frame_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
             # Upscale back to original size using nearest neighbor to show pixels clearly
             frame_rgb = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2:
            return
        
        img_h, img_w = frame_rgb.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        disp_w, disp_h = int(img_w * scale), int(img_h * scale)
        offset_x, offset_y = (canvas_w - disp_w) // 2, (canvas_h - disp_h) // 2
        
        self.canvas_offset_x = offset_x
        self.canvas_offset_y = offset_y
        self.canvas_scale = scale
        
        resized = cv2.resize(frame_rgb, (disp_w, disp_h))
        self.photo_image = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.photo_image)
        
        # Update coordinate text display
        if not self.has_grid_var.get():
            # No grid - show message
            self.coord_text.configure(state="normal")
            self.coord_text.delete("1.0", tk.END)
            self.coord_text.insert("1.0", "NO GRID\n\nThis frame does not contain\na visible grid.")
            self.coord_text.configure(state="disabled")
        elif all(p is not None for p in self.grid_points):
            try:
                # Get subdivisions
                try:
                    sub_x = float(self.grid_subdiv_x.get())
                except tk.TclError:
                    sub_x = 1.0
                try:
                    sub_y = float(self.grid_subdiv_y.get())
                except tk.TclError:
                    sub_y = 1.0

                # Calculate Homography for the user-defined macro cell
                # User points: P1(TL), P2(TR), P3(BL), P4(BR)
                p1, p2, p3, p4 = self.grid_points
                
                # Source: Unit square (0,0) to (1,1)
                # Destination: User points
                # Mapping: (0,0)->p1, (1,0)->p2, (0,1)->p3, (1,1)->p4
                src_pts = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
                dst_pts = np.float32([p1, p2, p3, p4])
                
                H = cv2.getPerspectiveTransform(src_pts, dst_pts)
                
                # Define the logical coordinates of the single top-left Unit Cell
                # It spans from (0,0) to (1/Sx, 1/Sy) in the macro space
                unit_w = 1.0 / max(1, sub_x)
                unit_h = 1.0 / max(1, sub_y)
                
                # Standard Polygon Order: TL, TR, BR, BL
                unit_logical = np.float32([
                    [0, 0],            # TL
                    [unit_w, 0],       # TR
                    [unit_w, unit_h],  # BR
                    [0, unit_h]        # BL
                ]).reshape(-1, 1, 2)
                
                # Transform logical unit points to pixel coordinates
                unit_pixels = cv2.perspectiveTransform(unit_logical, H)
                unit_pixels = unit_pixels.reshape(-1, 2)
                
                # --- NEW: Calculate Origin & Basis Vectors (Anchor) ---
                
                # 1. Find logical coordinate closest to image center
                # Center of image in pixel coords
                center_img = np.array([[[img_w / 2.0, img_h / 2.0]]], dtype=np.float32)
                
                # Transform image center to logical grid coords (inverse homography)
                H_inv = np.linalg.inv(H)
                center_logical = cv2.perspectiveTransform(center_img, H_inv)[0][0]
                
                # The logical grid is scaled by sub_x, sub_y in the definition above?
                # Actually H maps Unit Square (0,0)-(1,1) to the USER's QUAD.
                # User quad has dimensions sub_x, sub_y.
                # So true grid integer coordinates are: logical_coord * (sub_x, sub_y)
                
                grid_gx = center_logical[0] * sub_x
                grid_gy = center_logical[1] * sub_y
                
                # Find nearest integer intersection
                nearest_gx = round(grid_gx)
                nearest_gy = round(grid_gy)
                
                # 2. Define Origin and Basis points in Logical Space (0-1 normalized to User Quad)
                # We need to convert back from integer grid coords to H-space coords
                # H-space x = grid_x / sub_x
                
                origin_logical = np.array([[[nearest_gx / sub_x, nearest_gy / sub_y]]], dtype=np.float32)
                basis_u_logical = np.array([[[ (nearest_gx + 1) / sub_x, nearest_gy / sub_y ]]], dtype=np.float32)
                basis_v_logical = np.array([[[ nearest_gx / sub_x, (nearest_gy + 1) / sub_y ]]], dtype=np.float32)
                
                # 3. Project back to Pixel Space
                origin_px = cv2.perspectiveTransform(origin_logical, H)[0][0]
                basis_u_px = cv2.perspectiveTransform(basis_u_logical, H)[0][0]
                basis_v_px = cv2.perspectiveTransform(basis_v_logical, H)[0][0]
                
                # 4. Calculate Vectors
                vec_u = basis_u_px - origin_px
                vec_v = basis_v_px - origin_px
                
                # Normalize for display
                norm_origin = origin_px / [img_w, img_h]
                norm_vec_u = vec_u / [img_w, img_h]
                norm_vec_v = vec_v / [img_w, img_h]
                
                norm_text = "Grid Anchor (Normalized):\n"
                norm_text += f"Origin: {norm_origin[0]:.4f}, {norm_origin[1]:.4f}\n"
                norm_text += f"Vec U : {norm_vec_u[0]:.4f}, {norm_vec_u[1]:.4f}\n"
                norm_text += f"Vec V : {norm_vec_v[0]:.4f}, {norm_vec_v[1]:.4f}\n"
                
                self.coord_text.configure(state="normal")
                self.coord_text.delete("1.0", tk.END)
                self.coord_text.insert("1.0", norm_text)
                self.coord_text.configure(state="disabled")
                
                # Visualize this unit cell in lime green
                uc_canvas = []
                for px, py in unit_pixels:
                    cx = self.canvas_offset_x + px * self.canvas_scale
                    cy = self.canvas_offset_y + py * self.canvas_scale
                    uc_canvas.append((cx, cy))
                
                if self.has_grid_var.get():
                    # Draw User Unit Cell (Lime) - kept for reference
                    # self.canvas.create_line(uc_canvas[0][0], uc_canvas[0][1], uc_canvas[1][0], uc_canvas[1][1], fill='lime', width=3)
                    # self.canvas.create_line(uc_canvas[1][0], uc_canvas[1][1], uc_canvas[2][0], uc_canvas[2][1], fill='lime', width=3)
                    # self.canvas.create_line(uc_canvas[2][0], uc_canvas[2][1], uc_canvas[3][0], uc_canvas[3][1], fill='lime', width=3)
                    # self.canvas.create_line(uc_canvas[3][0], uc_canvas[3][1], uc_canvas[0][0], uc_canvas[0][1], fill='lime', width=3)
                    
                    # Draw Anchor Basis
                    # Origin
                    ox = self.canvas_offset_x + origin_px[0] * self.canvas_scale
                    oy = self.canvas_offset_y + origin_px[1] * self.canvas_scale
                    
                    # U Vector tip
                    ux = self.canvas_offset_x + basis_u_px[0] * self.canvas_scale
                    uy = self.canvas_offset_y + basis_u_px[1] * self.canvas_scale
                    
                    # V Vector tip
                    vx = self.canvas_offset_x + basis_v_px[0] * self.canvas_scale
                    vy = self.canvas_offset_y + basis_v_px[1] * self.canvas_scale
                    
                    # Draw Origin Dot
                    r = 6
                    self.canvas.create_oval(ox-r, oy-r, ox+r, oy+r, fill='magenta', outline='white', width=2)
                    
                    # Draw Vectors (Arrows)
                    self.canvas.create_line(ox, oy, ux, uy, fill='red', width=3, arrow=tk.LAST)
                    self.canvas.create_line(ox, oy, vx, vy, fill='lime', width=3, arrow=tk.LAST)
                    
                    # Label
                    self.canvas.create_text(ox, oy-15, text="Anchor", fill="magenta", font=("Arial", 10, "bold"))

            except Exception as e:
                print(f"[Grid Tool] Error calculating grid anchor: {e}")
                self.coord_text.configure(state="normal")
                self.coord_text.delete("1.0", tk.END)
                self.coord_text.insert("1.0", f"CALC ERROR\n{e}")
                self.coord_text.configure(state="disabled")

        # Draw grid if all 4 points are placed AND has_grid is checked
        if self.has_grid_var.get() and all(p is not None for p in self.grid_points):
            self._draw_grid_overlay()
    
    def _draw_grid_overlay(self):
        """Draw grid lines using a perspective transform from the four corner points."""
        p1, p2, p3, p4 = self.grid_points

        # Convert image points to canvas points for drawing markers
        canvas_points = []
        for point in self.grid_points:
            cx = self.canvas_offset_x + point[0] * self.canvas_scale
            cy = self.canvas_offset_y + point[1] * self.canvas_scale
            canvas_points.append((cx, cy))
        cp1, cp2, cp3, cp4 = canvas_points

        # Define logical and image coordinates for the transform
        # P1(TL)->(0,0), P2(TR)->(1,0), P3(BL)->(0,1), P4(BR)->(1,1)
        src_pts = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
        dst_pts = np.float32([p1, p2, p3, p4])

        try:
            # Calculate the perspective transformation matrix
            H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            grid_range = 5
            
            try:
                sub_x = self.grid_subdiv_x.get()
            except tk.TclError:
                sub_x = 1
                
            try:
                sub_y = self.grid_subdiv_y.get()
            except tk.TclError:
                sub_y = 1
            
            def transform_point(px, py):
                """Apply H to (px, py) and return screen (cx, cy) if w > 0."""
                # Homogeneous multiply: H * [px, py, 1]
                # We do it manually to check 'w'
                x = H[0,0]*px + H[0,1]*py + H[0,2]
                y = H[1,0]*px + H[1,1]*py + H[1,2]
                w = H[2,0]*px + H[2,1]*py + H[2,2]
                
                if w <= 1e-5:
                    return None
                
                img_x = x / w
                img_y = y / w
                
                cx = self.canvas_offset_x + img_x * self.canvas_scale
                cy = self.canvas_offset_y + img_y * self.canvas_scale
                return cx, cy

            def draw_segment(x1, y1, x2, y2, **kwargs):
                pt1 = transform_point(x1, y1)
                pt2 = transform_point(x2, y2)
                if pt1 and pt2:
                    self.canvas.create_line(pt1[0], pt1[1], pt2[0], pt2[1], **kwargs)

            # Draw Vertical lines
            # Iterate x from -grid_range to grid_range + 1
            # But we also need subdivisions between i and i+1
            
            # Total vertical lines to consider:
            #Integers from -grid_range to grid_range + 1
            
            for i in range(-grid_range, grid_range + 1):
                # For this integer X, draw the line from Y=-grid_range to Y=grid_range+1
                # We segment it by integer Y steps to handle horizon clipping
                
                # Main line at X=i
                for j in range(-grid_range, grid_range + 1):
                    draw_segment(i, j, i, j + 1, fill="cyan", width=2)

                # Subdivisions between i and i+1 (if not the last column)
                if i < grid_range:
                    for k in range(1, sub_x):
                        frac = k / sub_x
                        val = i + frac
                        for j in range(-grid_range, grid_range + 1):
                            draw_segment(val, j, val, j + 1, fill="cyan", width=1, dash=(2, 2))

            # Draw Horizontal lines
            for i in range(-grid_range, grid_range + 1):
                # Main line at Y=i
                for j in range(-grid_range, grid_range + 1):
                    draw_segment(j, i, j + 1, i, fill="cyan", width=2)

                # Subdivisions
                if i < grid_range:
                    for k in range(1, sub_y):
                        frac = k / sub_y
                        val = i + frac
                        for j in range(-grid_range, grid_range + 1):
                            draw_segment(j, val, j + 1, val, fill="cyan", width=1, dash=(2, 2))

        except cv2.error:
            # If transform fails (e.g., collinear points), just draw the quad
            pass
        
        # Draw the quadrilateral boundary of the main cell
        self.canvas.create_line(cp1[0], cp1[1], cp2[0], cp2[1], fill='yellow', width=2)
        self.canvas.create_line(cp2[0], cp2[1], cp4[0], cp4[1], fill='yellow', width=2)
        self.canvas.create_line(cp4[0], cp4[1], cp3[0], cp3[1], fill='yellow', width=2)
        self.canvas.create_line(cp3[0], cp3[1], cp1[0], cp1[1], fill='yellow', width=2)

        # Draw the four corner points with labels
        point_radius = 8
        labels = ["P1 (TL)", "P2 (TR)", "P3 (BL)", "P4 (BR)"]
        for i, (cx, cy) in enumerate(canvas_points):
            self.canvas.create_oval(cx - point_radius, cy - point_radius, cx + point_radius, cy + point_radius, fill="red", outline="yellow", width=2)
            self.canvas.create_text(cx, cy - 20, text=labels[i], fill="white", font=("Arial", 10, "bold"))
    
    def _on_resize(self, event):
        self._display_frame()
    
    def _decrease_subdiv_x(self, event=None):
        """Decrease horizontal subdivisions (1 key)."""
        if event and isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox)):
            return
        current = self.grid_subdiv_x.get()
        if current > 1:
            self.grid_subdiv_x.set(current - 1)
            self._save_frame_config()
            self._display_frame()
    
    def _increase_subdiv_x(self, event=None):
        """Increase horizontal subdivisions (2 key)."""
        if event and isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox)):
            return
        current = self.grid_subdiv_x.get()
        if current < 20:
            self.grid_subdiv_x.set(current + 1)
            self._save_frame_config()
            self._display_frame()
    
    def _decrease_subdiv_y(self, event=None):
        """Decrease vertical subdivisions (3 key)."""
        if event and isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox)):
            return
        current = self.grid_subdiv_y.get()
        if current > 1:
            self.grid_subdiv_y.set(current - 1)
            self._save_frame_config()
            self._display_frame()
    
    def _increase_subdiv_y(self, event=None):
        """Increase vertical subdivisions (4 key)."""
        if event and isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox)):
            return
        current = self.grid_subdiv_y.get()
        if current < 20:
            self.grid_subdiv_y.set(current + 1)
            self._save_frame_config()
            self._display_frame()
    
    def _on_has_grid_toggle(self):
        """Handle toggling of Has Grid checkbox."""
        self._save_frame_config()
        self._display_frame()
    
    def _toggle_has_grid(self, event=None):
        """Toggle Has Grid with 'g' key."""
        if event and isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox)):
            return
        self.has_grid_var.set(not self.has_grid_var.get())
        self._save_frame_config()
        self._display_frame()
    
    def _update_stats(self):
        """Update the statistics label with frame counts."""
        total_labeled = len(self.frame_grid_config)
        
        # Count frames with and without grids
        grid_frames = 0
        no_grid_frames = 0
        
        for config in self.frame_grid_config.values():
            has_grid = config[3] if len(config) > 3 else False
            if has_grid:
                grid_frames += 1
            else:
                no_grid_frames += 1
        
        stats_text = f"Total Labeled: {total_labeled}\n"
        stats_text += f"Has Grid: {grid_frames}\n"
        stats_text += f"No Grid: {no_grid_frames}"
        
        self.stats_label.config(text=stats_text)
    
    def _save_frame_config(self):
        """Save the current frame's grid configuration."""
        if self.video_list:
            video_path = self.video_list[self.current_video_idx][0]
        else:
            video_path = self.video_path
        
        frame_key = (video_path, self.current_frame_idx)
        self.frame_grid_config[frame_key] = (
            list(self.grid_points),
            self.grid_subdiv_x.get(),
            self.grid_subdiv_y.get(),
            self.has_grid_var.get()
        )
        self._save_persistent_state()
    
    def _load_frame_config(self):
        """Load the saved grid configuration for the current frame if it exists."""
        if self.video_list:
            video_path = self.video_list[self.current_video_idx][0]
        else:
            video_path = self.video_path
        
        frame_key = (video_path, self.current_frame_idx)
        if frame_key in self.frame_grid_config:
            config = self.frame_grid_config[frame_key]
            points = config[0]
            subdiv_x = config[1]
            subdiv_y = config[2]
            has_grid = config[3] if len(config) > 3 else False  # Backward compatibility
            
            self.grid_points = list(points)
            self.grid_subdiv_x.set(subdiv_x)
            self.grid_subdiv_y.set(subdiv_y)
            self.has_grid_var.set(has_grid)
            return True
        return False
    
    def _get_state_file(self):
        """Get the path to the persistent state file."""
        return os.path.join(os.path.expanduser("~"), ".grid_tool_state.json")
    
    def _load_persistent_state(self):
        """Load frame history and grid configs from disk."""
        state_file = self._get_state_file()
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    # Load frame history - convert from list to list of tuples
                    if 'frame_history' in state:
                        self.frame_history = [tuple(item) if isinstance(item, list) else item for item in state['frame_history']]
                        # Ensure old format compatibility (if history was just frame indices)
                        if self.frame_history and not isinstance(self.frame_history[0], tuple):
                            if self.video_list:
                                video_path = self.video_list[0][0]
                            else:
                                video_path = self.video_path
                            self.frame_history = [(video_path, idx) for idx in self.frame_history]
                        self.history_index = min(state.get('history_index', 0), len(self.frame_history) - 1)
                    # Load grid configs - convert string keys back to tuples
                    if 'frame_grid_config' in state:
                        for key_str, value in state['frame_grid_config'].items():
                            # key_str is like "path/to/video.mp4,123"
                            parts = key_str.rsplit(',', 1)
                            if len(parts) == 2:
                                video_path, frame_idx_str = parts
                                frame_idx = int(frame_idx_str)
                                points = [tuple(p) if p else None for p in value[0]]
                                has_grid = value[3] if len(value) > 3 else False
                                self.frame_grid_config[(video_path, frame_idx)] = (points, value[1], value[2], has_grid)
            except Exception as e:
                print(f"[Grid Tool] Warning: Could not load persistent state: {e}")
    
    def _predict_grid(self, event=None):
        """Predict grid using trained model."""
        if event and isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox)):
            return
            
        if self.current_frame is None:
            return

        model_path = "grid_model.pth"
        if not os.path.exists(model_path):
            print("[Grid Tool] No model found. Train one first!")
            return

        # Reload model if file changed (simple check could be added, but for now just lazy load once or force reload if trained)
        # Actually, if we just trained, we want to reload. 
        # Simplest: Always reload if we triggered training, but here let's just load if None.
        # TODO: Add flag to force reload after training finishes.
        
        if self.model is None:
            try:
                print("[Grid Tool] Loading model...")
                self.model = models.mobilenet_v3_small(weights=None)
                in_features = self.model.classifier[3].in_features
                self.model.classifier[3] = nn.Linear(in_features, 7)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"[Grid Tool] Error loading model: {e}")
                self.model = None
                return

        # Preprocess frame
        h_orig, w_orig = self.current_frame.shape[:2]
        img = cv2.resize(self.current_frame, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Inference
        try:
            with torch.no_grad():
                outputs = self.model(img_tensor)
                # Output is logits. 
                # Conf is index 0
                conf_logit = outputs[0, 0]
                conf = torch.sigmoid(conf_logit).item()
                coords = outputs[0, 1:].cpu().numpy()
        except Exception as e:
             print(f"[Grid Tool] Inference error: {e}")
             return
            
        print(f"[Grid Tool] Prediction Confidence: {conf:.4f}")
        
        if conf < 0.5:
            print("[Grid Tool] Low confidence. Assuming No Grid.")
            self.has_grid_var.set(False)
            self._save_frame_config()
            self._display_frame()
            return
            
        # High confidence - reconstruction
        self.has_grid_var.set(True)
        
        # coords: [Ox, Oy, Ux, Uy, Vx, Vy] (Normalized)
        ox, oy = coords[0] * w_orig, coords[1] * h_orig
        ux, uy = coords[2] * w_orig, coords[3] * h_orig
        vx, vy = coords[4] * w_orig, coords[5] * h_orig
        
        # Origin Point
        origin = np.array([ox, oy])
        vec_u = np.array([ux, uy])
        vec_v = np.array([vx, vy])
        
        # Current subdivisions
        try:
            sub_x = self.grid_subdiv_x.get()
            sub_y = self.grid_subdiv_y.get()
        except:
            sub_x = 1
            sub_y = 1
            
        # Construct 4 corners centered on Origin
        # Center offset in grid units
        off_x = sub_x / 2.0
        off_y = sub_y / 2.0
        
        # P1 (Top-Left) = Origin - off_x * U - off_y * V
        p1 = origin - off_x * vec_u - off_y * vec_v
        
        # P2 (Top-Right) = Origin + off_x * U - off_y * V
        p2 = origin + off_x * vec_u - off_y * vec_v
        
        # P3 (Bottom-Left) = Origin - off_x * U + off_y * V
        p3 = origin - off_x * vec_u + off_y * vec_v
        
        # P4 (Bottom-Right) = Origin + off_x * U + off_y * V
        p4 = origin + off_x * vec_u + off_y * vec_v
        
        # Update points
        self.grid_points = [
            (float(p1[0]), float(p1[1])),
            (float(p2[0]), float(p2[1])),
            (float(p3[0]), float(p3[1])),
            (float(p4[0]), float(p4[1]))
        ]
        
        self._save_frame_config()
        self._display_frame()

    def _update_history_list(self):
        """Update the history listbox content."""
        self.history_listbox.delete(0, tk.END)
        
        for i, (video_path, frame_idx) in enumerate(self.frame_history):
            filename = os.path.basename(video_path)
            
            # Check if labeled
            frame_key = (video_path, frame_idx)
            status = "[?]"
            if frame_key in self.frame_grid_config:
                config = self.frame_grid_config[frame_key]
                has_grid = config[3] if len(config) > 3 else False
                status = "[G]" if has_grid else "[N]"
            
            item_text = f"{i+1}. {status} {filename} #{frame_idx}"
            self.history_listbox.insert(tk.END, item_text)
            
        # Select current
        if 0 <= self.history_index < self.history_listbox.size():
            self.history_listbox.selection_clear(0, tk.END)
            self.history_listbox.selection_set(self.history_index)
            self.history_listbox.see(self.history_index)

    def _on_history_select(self, event):
        """Handle click on history list item."""
        selection = self.history_listbox.curselection()
        if not selection:
            return
            
        index = selection[0]
        if index != self.history_index:
            self.history_index = index
            video_path, frame_idx = self.frame_history[self.history_index]
            self._switch_to_video_and_frame(video_path, frame_idx)

    def _save_persistent_state(self):
        """Save frame history and grid configs to disk."""
        self._update_history_list()
        state_file = self._get_state_file()
        try:
            # Convert frame_grid_config keys to strings for JSON serialization
            config_serializable = {}
            for (video_path, frame_idx), config in self.frame_grid_config.items():
                key_str = f"{video_path},{frame_idx}"
                # config is (points, subdiv_x, subdiv_y, has_grid)
                config_serializable[key_str] = config
            
            state = {
                'frame_history': self.frame_history,
                'history_index': self.history_index,
                'frame_grid_config': config_serializable
            }
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[Grid Tool] Warning: Could not save persistent state: {e}")

    def start_training(self):
        """Start the training process in a separate thread."""
        def run_train():
            self.train_button.config(state="disabled", text="Training...")
            try:
                # Run grid_train.py using the same python interpreter
                import sys
                
                # On Windows, use CREATE_NO_WINDOW to avoid popping up a terminal
                creation_flags = 0
                if os.name == 'nt':
                    creation_flags = 0x08000000  # CREATE_NO_WINDOW
                
                process = subprocess.Popen(
                    [sys.executable, "grid_train.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    creationflags=creation_flags
                )
                
                print("\n[Grid Tool] Training started...")
                
                # Read output line by line and print to console
                for line in process.stdout:
                    print(f"[Train] {line.strip()}")
                
                process.wait()
                print("[Grid Tool] Training finished.")
                
                # Force reload of model
                self.model = None
                
            except Exception as e:
                print(f"[Grid Tool] Error starting training: {e}")
            finally:
                # Schedule UI update on main thread
                self.root.after(0, lambda: self.train_button.config(state="normal", text="Train Model"))
        
        threading.Thread(target=run_train, daemon=True).start()

def main():
    parser = argparse.ArgumentParser(description="Interactive grid tool for video frames")
    parser.add_argument("--input", type=str, required=True, help="Path to video file or directory of videos")
    args = parser.parse_args()
    
    if not os.path.isfile(args.input) and not os.path.isdir(args.input):
        print(f"Error: Video file or directory not found: {args.input}")
        return
    
    root = tk.Tk()
    root.geometry("1200x800")
    
    try:
        app = GridTool(root, args.input)
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

