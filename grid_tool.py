import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import argparse
import os

class GridTool:
    def __init__(self, root, video_path):
        self.root = root
        self.video_path = video_path
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0
        
        self.grid_subdiv_x = tk.IntVar(value=1)
        self.grid_subdiv_y = tk.IntVar(value=1)

        self.root.title("Grid Tool - Click to place 4 corner points")
        self.root.state('zoomed')
        self._build_ui()
        self._load_frame(0)
    
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
        
        ttk.Label(side_panel, text="Horizontal Subdivisions:").pack(anchor=tk.W, pady=(10, 0))
        ttk.Spinbox(side_panel, from_=1, to=20, textvariable=self.grid_subdiv_x, command=self._display_frame).pack(fill=tk.X, pady=5)
        
        ttk.Label(side_panel, text="Vertical Subdivisions:").pack(anchor=tk.W, pady=(10, 0))
        ttk.Spinbox(side_panel, from_=1, to=20, textvariable=self.grid_subdiv_y, command=self._display_frame).pack(fill=tk.X, pady=5)

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
    
    def _load_frame(self, frame_idx):
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return
        
        self.current_frame_idx = frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame = frame
            # Initialize grid points with default positions on first load
            if frame_idx == 0 and self.grid_points[0] is None:
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
        
        self.grid_points[self.dragging_point] = (new_x, new_y)
        
        self._display_frame()
    
    def _canvas_release(self, event):
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
        
        # Draw grid if all 4 points are placed
        if all(p is not None for p in self.grid_points):
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
            
            # Draw subdivisions
            sub_x = self.grid_subdiv_x.get()
            sub_y = self.grid_subdiv_y.get()
            
            # Draw grid lines with subdivisions
            # Vertical lines
            for i in range(-grid_range, grid_range + 1):
                # Main grid lines
                line_src = np.float32([[[i, -grid_range]], [[i, grid_range + 1]]])
                line_dst = cv2.perspectiveTransform(line_src, H)
                pt1, pt2 = line_dst[0][0], line_dst[1][0]
                self.canvas.create_line(
                    self.canvas_offset_x + pt1[0] * self.canvas_scale, self.canvas_offset_y + pt1[1] * self.canvas_scale,
                    self.canvas_offset_x + pt2[0] * self.canvas_scale, self.canvas_offset_y + pt2[1] * self.canvas_scale,
                    fill="cyan", width=2)
                
                # Subdivisions
                if i < grid_range:
                    for j in range(1, sub_x):
                        frac = j / sub_x
                        val = i + frac
                        line_src = np.float32([[[val, -grid_range]], [[val, grid_range + 1]]])
                        line_dst = cv2.perspectiveTransform(line_src, H)
                        pt1, pt2 = line_dst[0][0], line_dst[1][0]
                        self.canvas.create_line(
                            self.canvas_offset_x + pt1[0] * self.canvas_scale, self.canvas_offset_y + pt1[1] * self.canvas_scale,
                            self.canvas_offset_x + pt2[0] * self.canvas_scale, self.canvas_offset_y + pt2[1] * self.canvas_scale,
                            fill="cyan", width=1, dash=(2, 2))

            # Horizontal lines
            for i in range(-grid_range, grid_range + 1):
                # Main grid lines
                line_src = np.float32([[[-grid_range, i]], [[grid_range + 1, i]]])
                line_dst = cv2.perspectiveTransform(line_src, H)
                pt1, pt2 = line_dst[0][0], line_dst[1][0]
                self.canvas.create_line(
                    self.canvas_offset_x + pt1[0] * self.canvas_scale, self.canvas_offset_y + pt1[1] * self.canvas_scale,
                    self.canvas_offset_x + pt2[0] * self.canvas_scale, self.canvas_offset_y + pt2[1] * self.canvas_scale,
                    fill="cyan", width=2)

                # Subdivisions
                if i < grid_range:
                    for j in range(1, sub_y):
                        frac = j / sub_y
                        val = i + frac
                        line_src = np.float32([[[-grid_range, val]], [[grid_range + 1, val]]])
                        line_dst = cv2.perspectiveTransform(line_src, H)
                        pt1, pt2 = line_dst[0][0], line_dst[1][0]
                        self.canvas.create_line(
                            self.canvas_offset_x + pt1[0] * self.canvas_scale, self.canvas_offset_y + pt1[1] * self.canvas_scale,
                            self.canvas_offset_x + pt2[0] * self.canvas_scale, self.canvas_offset_y + pt2[1] * self.canvas_scale,
                            fill="cyan", width=1, dash=(2, 2))
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
    parser.add_argument("--input", type=str, required=True, help="Path to video file")
    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"Error: Video file not found: {args.input}")
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

