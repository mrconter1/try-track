import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import argparse
import os
import random

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
        self.show_grid_var = tk.BooleanVar(value=True)
        self.low_res_preview_var = tk.BooleanVar(value=False)

        self.root.title("Grid Tool - Click to place 4 corner points")
        self.root.state('zoomed')
        self._build_ui()
        self._load_frame(0)
    
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
        
        ttk.Checkbutton(side_panel, text="Show Grid", variable=self.show_grid_var, command=self._display_frame).pack(anchor=tk.W, pady=5)
        ttk.Checkbutton(side_panel, text="256x256 Preview", variable=self.low_res_preview_var, command=self._display_frame).pack(anchor=tk.W, pady=5)
        
        ttk.Label(side_panel, text="Horizontal Subdivisions:").pack(anchor=tk.W, pady=(10, 0))
        subdiv_x_spinbox = ttk.Spinbox(side_panel, from_=1, to=20, textvariable=self.grid_subdiv_x, command=self._display_frame)
        subdiv_x_spinbox.pack(fill=tk.X, pady=5)
        subdiv_x_spinbox.bind("<FocusIn>", lambda e: side_panel.focus_set())
        
        ttk.Label(side_panel, text="Vertical Subdivisions:").pack(anchor=tk.W, pady=(10, 0))
        subdiv_y_spinbox = ttk.Spinbox(side_panel, from_=1, to=20, textvariable=self.grid_subdiv_y, command=self._display_frame)
        subdiv_y_spinbox.pack(fill=tk.X, pady=5)
        subdiv_y_spinbox.bind("<FocusIn>", lambda e: side_panel.focus_set())

        # Normalized Coordinates Display
        ttk.Label(side_panel, text="Normalized Coordinates:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(15, 5))
        self.coord_text = tk.Text(side_panel, height=8, width=25, font=("Consolas", 9))
        self.coord_text.pack(fill=tk.X, pady=5)
        self.coord_text.insert("1.0", "Move points to see\ncoordinates...")
        self.coord_text.configure(state="disabled")

        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=5, padx=5)
        
        self.info_label = ttk.Label(controls_frame, text="Click to place 4 corner points of grid cell")
        self.info_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.reset_button = ttk.Button(controls_frame, text="Reset Points", command=self.reset_points)
        self.reset_button.pack(side=tk.LEFT, padx=10, pady=5)
        
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
        
        # Frame history for a/d navigation
        self.frame_history = [0]  # Start with frame 0
        self.history_index = 0
    
    def _load_frame(self, frame_idx):
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return
        
        self.current_frame_idx = frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame = frame
            # Initialize grid points with default positions if not already set
            if self.grid_points[0] is None:
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
        self._load_frame(random_idx)
    
    def _prev_in_history(self, event=None):
        """Go to previous frame in history (a key)."""
        if event and isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox)):
            return
        
        if self.history_index > 0:
            self.history_index -= 1
            frame_idx = self.frame_history[self.history_index]
            self._load_frame(frame_idx)
    
    def _next_or_random_frame(self, event=None):
        """Go to next frame in history, or pick a new random one if at end (d key)."""
        if event and isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox)):
            return
        
        if self.history_index < len(self.frame_history) - 1:
            # Navigate forward in existing history
            self.history_index += 1
            frame_idx = self.frame_history[self.history_index]
            self._load_frame(frame_idx)
        else:
            # At the end, pick a new random frame
            if self.video_list:
                # Sample from all videos proportionally to frame count
                random_idx = self._sample_random_frame_proportional()
            else:
                random_idx = random.randint(0, self.total_frames - 1)
            self.frame_history.append(random_idx)
            self.history_index += 1
            self._load_frame(random_idx)
    
    def _sample_random_frame_proportional(self):
        """Sample a random frame from all videos, proportional to frame counts."""
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
        
        # Switch to that video if needed
        current_video_path = self.video_list[self.current_video_idx][0]
        if video_path != current_video_path:
            print(f"[Grid Tool] Switching to: {os.path.basename(video_path)} (frame {frame_idx}/{frame_count})")
            self.cap.release()
            self.cap = cv2.VideoCapture(video_path)
            self.current_video_idx = next(i for i, (vp, _) in enumerate(self.video_list) if vp == video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            print(f"[Grid Tool] Frame {frame_idx}/{frame_count}")
        
        return frame_idx

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
        
        x1 = int(img_x - patch_size_img_coords / 2)
        y1 = int(img_y - patch_size_img_coords / 2)
        x2 = int(img_x + patch_size_img_coords / 2)
        y2 = int(img_y + patch_size_img_coords / 2)

        # Ensure coordinates are within frame bounds
        h, w = self.current_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x1 >= x2 or y1 >= y2: return # Avoid invalid crop size
        
        patch = self.current_frame[y1:y2, x1:x2]
        
        # Resize patch to magnifier size for zoom effect
        zoomed_patch = cv2.resize(patch, (self.magnifier_size, self.magnifier_size), interpolation=cv2.INTER_NEAREST)
        zoomed_patch_rgb = cv2.cvtColor(zoomed_patch, cv2.COLOR_BGR2RGB)
        
        # Create PhotoImage and display it
        self.magnifier_photo = ImageTk.PhotoImage(Image.fromarray(zoomed_patch_rgb))
        self.magnifier_widget.create_image(0, 0, anchor='nw', image=self.magnifier_photo)
        
        # Draw the crosshair
        center = self.magnifier_size / 2
        self.magnifier_widget.create_line(center, 0, center, self.magnifier_size, fill='red', width=1)
        self.magnifier_widget.create_line(0, center, self.magnifier_size, center, fill='red', width=1)
        
    def _destroy_magnifier(self):
        """Destroy the magnifier widget."""
        if self.magnifier_widget:
            self.magnifier_widget.destroy()
            self.magnifier_widget = None

    def _display_frame(self):
        if self.current_frame is None:
            return
        
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
        
        # Update coordinate text display with Unit Cell info
        if all(p is not None for p in self.grid_points):
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
                
                norm_text = "NN Unit Cell (Normalized):\n"
                display_labels = ["TL", "TR", "BR", "BL"]
                
                for idx, label in enumerate(display_labels):
                    px, py = unit_pixels[idx]
                    nx = px / img_w
                    ny = py / img_h
                    norm_text += f"{label}: {nx:.4f}, {ny:.4f}\n"
                
                self.coord_text.configure(state="normal")
                self.coord_text.delete("1.0", tk.END)
                self.coord_text.insert("1.0", norm_text)
                self.coord_text.configure(state="disabled")
                
                # Optionally visualize this unit cell in a different color (e.g. Green)
                # Convert to canvas coords
                uc_canvas = []
                for px, py in unit_pixels:
                    cx = self.canvas_offset_x + px * self.canvas_scale
                    cy = self.canvas_offset_y + py * self.canvas_scale
                    uc_canvas.append((cx, cy))
                
                if self.show_grid_var.get():
                    self.canvas.create_line(uc_canvas[0][0], uc_canvas[0][1], uc_canvas[1][0], uc_canvas[1][1], fill='lime', width=3)
                    self.canvas.create_line(uc_canvas[1][0], uc_canvas[1][1], uc_canvas[2][0], uc_canvas[2][1], fill='lime', width=3)
                    self.canvas.create_line(uc_canvas[2][0], uc_canvas[2][1], uc_canvas[3][0], uc_canvas[3][1], fill='lime', width=3)
                    self.canvas.create_line(uc_canvas[3][0], uc_canvas[3][1], uc_canvas[0][0], uc_canvas[0][1], fill='lime', width=3)

            except Exception:
                pass

        # Draw grid if all 4 points are placed and grid is enabled
        if self.show_grid_var.get() and all(p is not None for p in self.grid_points):
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

