import cv2
import numpy as np
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

MIN_ANGLE_DIFF = 50.0

# --- Tkinter GUI Application ---

class App:
    def __init__(self, root, args, initial_frame, video_path, total_frames):
        self.root = root
        self.args = args
        self.original_frame = initial_frame
        self.video_path = video_path
        self.total_frames = total_frames
        self.current_frame_num = args.frame
        
        self.root.title("Crosshair Tool")
        
        # --- Variables ---
        h, w = self.original_frame.shape[:2]
        initial_cx = self.args.center_x if self.args.center_x is not None else w / 2
        initial_cy = self.args.center_y if self.args.center_y is not None else h / 2

        self.cx_var = tk.DoubleVar(value=initial_cx)
        self.cy_var = tk.DoubleVar(value=initial_cy)
        self.v_angle_var = tk.DoubleVar(value=0.0)   # First line angle
        self.h_angle_var = tk.DoubleVar(value=90.0)  # Second line angle (perpendicular)
        self.line_length_var = tk.IntVar(value=100)  # Line length for both
        self.num_samples_var = tk.IntVar(value=15)   # Number of sample points per line
        self.inner_exclusion_var = tk.DoubleVar(value=30.0)  # Inner radius for adjacent lines
        self.clahe_enabled = tk.BooleanVar(value=True)
        self.clahe_clip_limit = tk.DoubleVar(value=10.0)
        self.clahe_tile_size = tk.IntVar(value=8)
        self.frame_num_var = tk.IntVar(value=self.current_frame_num)
        self._optimizing = False
        self.search_rect = None
        self.processed_frame = self.original_frame.copy()
        
        # --- GUI Layout ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # --- Controls Frame ---
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

        # Controls
        h, w = self.original_frame.shape[:2]
        
        # Frame Number Controls
        ttk.Label(control_frame, text="Frame:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_frame(-10)).grid(row=0, column=1)
        self.frame_slider = tk.Scale(control_frame, from_=0, to=self.total_frames-1, orient=tk.HORIZONTAL, variable=self.frame_num_var, command=self.load_frame, resolution=1, showvalue=0)
        self.frame_slider.grid(row=0, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_frame(10)).grid(row=0, column=3)
        self.frame_label = ttk.Label(control_frame, text=f"{self.frame_num_var.get()}", width=7)
        self.frame_label.grid(row=0, column=4, padx=5)

        # Center X Controls
        ttk.Label(control_frame, text="Center X:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_cx(-1)).grid(row=1, column=1)
        self.cx_slider = tk.Scale(control_frame, from_=0, to=w, orient=tk.HORIZONTAL, variable=self.cx_var, command=self.update_image, resolution=0.1, showvalue=0)
        self.cx_slider.grid(row=1, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_cx(1)).grid(row=1, column=3)
        self.cx_label = ttk.Label(control_frame, text=f"{self.cx_var.get():.1f}", width=7)
        self.cx_label.grid(row=1, column=4, padx=5)

        # Center Y Controls
        ttk.Label(control_frame, text="Center Y:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_cy(-1)).grid(row=2, column=1)
        self.cy_slider = tk.Scale(control_frame, from_=0, to=h, orient=tk.HORIZONTAL, variable=self.cy_var, command=self.update_image, resolution=0.1, showvalue=0)
        self.cy_slider.grid(row=2, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_cy(1)).grid(row=2, column=3)
        self.cy_label = ttk.Label(control_frame, text=f"{self.cy_var.get():.1f}", width=7)
        self.cy_label.grid(row=2, column=4, padx=5)

        # Vertical Line Angle Controls
        ttk.Label(control_frame, text="V-Angle:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_v_angle(-1)).grid(row=3, column=1)
        self.v_angle_slider = tk.Scale(control_frame, from_=0, to=180, orient=tk.HORIZONTAL, variable=self.v_angle_var, command=self.update_image, resolution=0.1, showvalue=0)
        self.v_angle_slider.grid(row=3, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_v_angle(1)).grid(row=3, column=3)
        self.v_angle_label = ttk.Label(control_frame, text=f"{self.v_angle_var.get():.1f}", width=7)
        self.v_angle_label.grid(row=3, column=4, padx=5)

        # Horizontal Line Angle Controls
        ttk.Label(control_frame, text="H-Angle:").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_h_angle(-1)).grid(row=4, column=1)
        self.h_angle_slider = tk.Scale(control_frame, from_=0, to=180, orient=tk.HORIZONTAL, variable=self.h_angle_var, command=self.update_image, resolution=0.1, showvalue=0)
        self.h_angle_slider.grid(row=4, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_h_angle(1)).grid(row=4, column=3)
        self.h_angle_label = ttk.Label(control_frame, text=f"{self.h_angle_var.get():.1f}", width=7)
        self.h_angle_label.grid(row=4, column=4, padx=5)

        # Line Length Controls
        ttk.Label(control_frame, text="Length:").grid(row=5, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_line_length(-5)).grid(row=5, column=1)
        self.line_length_slider = tk.Scale(control_frame, from_=10, to=200, orient=tk.HORIZONTAL, variable=self.line_length_var, command=self.update_image, resolution=1, showvalue=0)
        self.line_length_slider.grid(row=5, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_line_length(5)).grid(row=5, column=3)
        self.line_length_label = ttk.Label(control_frame, text=f"{self.line_length_var.get()}", width=7)
        self.line_length_label.grid(row=5, column=4, padx=5)

        # Number of Samples Controls
        ttk.Label(control_frame, text="Samples:").grid(row=6, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_num_samples(-1)).grid(row=6, column=1)
        self.num_samples_slider = tk.Scale(control_frame, from_=3, to=50, orient=tk.HORIZONTAL, variable=self.num_samples_var, command=self.update_image, resolution=1, showvalue=0)
        self.num_samples_slider.grid(row=6, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_num_samples(1)).grid(row=6, column=3)
        self.num_samples_label = ttk.Label(control_frame, text=f"{self.num_samples_var.get()}", width=7)
        self.num_samples_label.grid(row=6, column=4, padx=5)

        # Inner Gap Controls
        ttk.Label(control_frame, text="Inner Gap:").grid(row=7, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_inner_gap(-1)).grid(row=7, column=1)
        self.inner_gap_slider = tk.Scale(control_frame, from_=0, to=150, orient=tk.HORIZONTAL, variable=self.inner_exclusion_var, command=self.update_image, resolution=1, showvalue=0)
        self.inner_gap_slider.grid(row=7, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_inner_gap(1)).grid(row=7, column=3)
        self.inner_gap_label = ttk.Label(control_frame, text=f"{self.inner_exclusion_var.get():.0f}", width=7)
        self.inner_gap_label.grid(row=7, column=4, padx=5)

        # CLAHE Enable/Disable
        ttk.Label(control_frame, text="CLAHE:").grid(row=8, column=0, sticky=tk.W, pady=2)
        self.clahe_checkbox = ttk.Checkbutton(control_frame, variable=self.clahe_enabled, command=self.update_image)
        self.clahe_checkbox.grid(row=8, column=1, sticky=tk.W)
        
        # CLAHE Clip Limit
        ttk.Label(control_frame, text="Clip:").grid(row=8, column=2, sticky=tk.W, padx=(10, 0))
        self.clahe_clip_slider = tk.Scale(control_frame, from_=1.0, to=10.0, orient=tk.HORIZONTAL, variable=self.clahe_clip_limit, command=self.update_image, resolution=0.5, showvalue=0)
        self.clahe_clip_slider.grid(row=8, column=3, sticky="ew")
        self.clahe_clip_label = ttk.Label(control_frame, text=f"{self.clahe_clip_limit.get():.1f}", width=5)
        self.clahe_clip_label.grid(row=8, column=4, padx=5)

        control_frame.columnconfigure(2, weight=1) # Make slider stretch

        # --- Force update to calculate control frame's actual size ---
        self.root.update_idletasks()
        control_height = control_frame.winfo_reqheight()

        # --- Now, dynamically set Display Size based on remaining space ---
        max_width = root.winfo_screenwidth() - 40   # Padding for window borders
        max_height = root.winfo_screenheight() - control_height - 150 # Padding for borders, taskbar, and margins
        
        h_orig, w_orig = self.original_frame.shape[:2]
        ratio = min(max_width / w_orig, max_height / h_orig)
        
        self.display_w = int(w_orig * ratio)
        self.display_h = int(h_orig * ratio)

        # --- Image and Plot Layout ---
        # Use a Canvas for the image to capture mouse events properly
        self.image_canvas = tk.Canvas(main_frame, bg="black", highlightthickness=0)
        self.image_canvas.grid(row=0, column=0, sticky="nsew")
        self.image_canvas.config(width=self.display_w, height=self.display_h)
        
        # Pixel Plot Canvas
        self.plot_canvas = tk.Canvas(main_frame, bg="white", width=200, height=self.display_h)
        self.plot_canvas.grid(row=0, column=1, sticky="nsew")

        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3) # Image gets 3/4 of the space
        main_frame.columnconfigure(1, weight=1) # Plot gets 1/4 of the space
        
        # Store display scaling factor for mouse interaction
        self.scale_factor = 1.0
        self.dragging_endpoint = None  # Track which endpoint is being dragged
        self.image_on_canvas = None  # Store the PhotoImage to prevent garbage collection
        
        # Bind mouse events to image canvas
        self.image_canvas.bind("<Button-1>", self.on_image_click)
        self.image_canvas.bind("<B1-Motion>", self.on_image_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_image_release)
        self.image_canvas.bind("<Motion>", self.on_mouse_move)
        self.image_canvas.bind("<Leave>", self.on_mouse_leave)

        # Optimize button
        self.optimize_button = ttk.Button(control_frame, text="Optimize Contrast", command=self.start_optimization)
        self.optimize_button.grid(row=0, column=5, rowspan=9, sticky="ns", padx=10)

        self.update_image()
    
    def apply_clahe(self, frame):
        """Apply CLAHE to even out shadows and lighting."""
        if not self.clahe_enabled.get():
            return frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit.get(), tileGridSize=(self.clahe_tile_size.get(), self.clahe_tile_size.get()))
        gray_clahe = clahe.apply(gray)
        
        # Merge the CLAHE-processed gray channel back to BGR
        return cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)
    
    def get_line_samples(self, cx, cy, angle, line_length, num_samples=10):
        """Sample pixel values along a line and return them."""
        return self.get_line_samples_with_radius(cx, cy, angle, line_length, num_samples)[0]

    def compute_sample_points(self, cx, cy, angle, line_length, num_samples, inner_radius):
        """Compute discrete sample points along a line excluding inner radius."""
        h, w = self.original_frame.shape[:2]
        angle_rad = np.deg2rad(angle)
        dx = line_length * np.cos(angle_rad)
        dy = line_length * np.sin(angle_rad)
        t_vals = np.linspace(-1, 1, num_samples)
        x_vals = cx + t_vals * dx
        y_vals = cy + t_vals * dy
        distances = np.sqrt((x_vals - cx)**2 + (y_vals - cy)**2)
        mask = distances >= inner_radius
        if not np.any(mask):
            # Ensure at least first and last sample
            mask[0] = True
            mask[-1] = True
        x_vals = np.clip(x_vals[mask], 0, w - 1).astype(int)
        y_vals = np.clip(y_vals[mask], 0, h - 1).astype(int)
        return list(zip(x_vals, y_vals))

    def get_line_samples_with_radius(self, cx, cy, angle, line_length, num_samples=10, inner_radius=0, frame=None):
        """Return (pixel_values, points) for a line respecting inner radius."""
        h, w = self.original_frame.shape[:2]
        points = self.compute_sample_points(cx, cy, angle, line_length, num_samples, inner_radius)
        if not points:
            return np.array([]), []
        xs = np.array([pt[0] for pt in points])
        ys = np.array([pt[1] for pt in points])
        source_frame = frame if frame is not None else (self.processed_frame if self.processed_frame is not None else self.original_frame)
        gray_frame = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
        pixel_values = gray_frame[ys, xs]
        return pixel_values, points
    
    def draw_pixel_plot(self, v_pixels, h_pixels, mid1_pixels, mid2_pixels):
        """Draw a plot of pixel values for all four lines."""
        self.plot_canvas.delete("all")
        
        canvas_w = self.plot_canvas.winfo_width()
        canvas_h = self.plot_canvas.winfo_height()
        
        if canvas_w < 2 or canvas_h < 2:
            self.root.after(50, lambda: self.draw_pixel_plot(v_pixels, h_pixels, mid1_pixels, mid2_pixels))
            return
        
        # Helper to draw line on canvas with sample points
        def draw_line(pixels, color):
            if pixels is None or len(pixels) == 0:
                return
            points = []
            num_samples = len(pixels)
            x_step = canvas_w / max(1, num_samples - 1)
            for i, value in enumerate(pixels):
                x = i * x_step
                y = canvas_h - (value / 255.0) * canvas_h
                points.extend([x, y])
                
                # Draw sample point as small circle
                self.plot_canvas.create_oval(x-2, y-2, x+2, y+2, fill=color, outline=color)
            
            if len(points) > 2:
                self.plot_canvas.create_line(points, fill=color, width=2)
        
        # Draw all four lines with matching frame colors
        draw_line(v_pixels, "red")     # Red line (V-angle)
        draw_line(h_pixels, "blue")    # Blue line (H-angle)
        draw_line(mid1_pixels, "green")  # Green line (Mid1)
        draw_line(mid2_pixels, "cyan")   # Cyan line (Mid2)

        # Draw combined median lines
        combined_mid = []
        if mid1_pixels is not None:
            combined_mid.extend(mid1_pixels.tolist())
        if mid2_pixels is not None:
            combined_mid.extend(mid2_pixels.tolist())
        if combined_mid:
            perc_mid = np.percentile(combined_mid, 95)
            y_perc_mid = canvas_h - (perc_mid / 255.0) * canvas_h
            self.plot_canvas.create_line(0, y_perc_mid, canvas_w, y_perc_mid, fill="magenta", dash=(4, 4), width=2)
            self.plot_canvas.create_text(5, y_perc_mid - 12, anchor="nw", text="Mid 95%", fill="magenta", font=("Arial", 8))

        combined_main = []
        if v_pixels is not None:
            combined_main.extend(v_pixels.tolist())
        if h_pixels is not None:
            combined_main.extend(h_pixels.tolist())
        if combined_main:
            perc_main = np.percentile(combined_main, 95)
            y_perc_main = canvas_h - (perc_main / 255.0) * canvas_h
            self.plot_canvas.create_line(0, y_perc_main, canvas_w, y_perc_main, fill="orange", dash=(4, 4), width=2)
            self.plot_canvas.create_text(5, y_perc_main - 24, anchor="nw", text="Main 95%", fill="orange", font=("Arial", 8))
        
        # Draw legend
        legend_y = 10
        colors = [
            ("red", "V-Line"),
            ("blue", "H-Line"),
            ("green", "Mid1"),
            ("cyan", "Mid2"),
            ("magenta", "Mid 95%"),
            ("orange", "Main 95%")
        ]
        for color, label in colors:
            self.plot_canvas.create_line(canvas_w - 90, legend_y, canvas_w - 70, legend_y, fill=color, width=2)
            self.plot_canvas.create_text(canvas_w - 65, legend_y, anchor="w", text=label, font=("Arial", 8))
            legend_y += 15

    def on_image_click(self, event):
        """Handle mouse click on the image."""
        h_orig, w_orig = self.original_frame.shape[:2]
        self.scale_factor = self.display_w / w_orig
        
        # Convert display coordinates to frame coordinates
        frame_x = event.x / self.scale_factor
        frame_y = event.y / self.scale_factor
        
        cx = self.cx_var.get()
        cy = self.cy_var.get()
        v_angle = self.v_angle_var.get()
        h_angle = self.h_angle_var.get()
        line_length = self.line_length_var.get()
        
        # Calculate both endpoints for red and blue lines
        v_angle_rad = np.deg2rad(v_angle)
        v_dx = line_length * np.cos(v_angle_rad)
        v_dy = line_length * np.sin(v_angle_rad)
        v_x1 = cx - v_dx
        v_y1 = cy - v_dy
        v_x2 = cx + v_dx
        v_y2 = cy + v_dy
        
        h_angle_rad = np.deg2rad(h_angle)
        h_dx = line_length * np.cos(h_angle_rad)
        h_dy = line_length * np.sin(h_angle_rad)
        h_x1 = cx - h_dx
        h_y1 = cy - h_dy
        h_x2 = cx + h_dx
        h_y2 = cy + h_dy
        
        # Check if click is near an endpoint (within 15 pixels in frame space)
        threshold = 15
        
        v_dist1 = np.sqrt((frame_x - v_x1)**2 + (frame_y - v_y1)**2)
        v_dist2 = np.sqrt((frame_x - v_x2)**2 + (frame_y - v_y2)**2)
        h_dist1 = np.sqrt((frame_x - h_x1)**2 + (frame_y - h_y1)**2)
        h_dist2 = np.sqrt((frame_x - h_x2)**2 + (frame_y - h_y2)**2)
        
        min_v_dist = min(v_dist1, v_dist2)
        min_h_dist = min(h_dist1, h_dist2)
        
        if min_v_dist < threshold and min_v_dist < min_h_dist:
            self.dragging_endpoint = "v"
        elif min_h_dist < threshold:
            self.dragging_endpoint = "h"
        else:
            self.dragging_endpoint = None
        
        # Store the initial position for dragging
        self.last_x = event.x
        self.last_y = event.y
    
    def on_image_drag(self, event):
        """Handle mouse drag on the image to move center or rotate endpoints."""
        h_orig, w_orig = self.original_frame.shape[:2]
        self.scale_factor = self.display_w / w_orig
        
        # Convert display coordinates to frame coordinates
        frame_x = event.x / self.scale_factor
        frame_y = event.y / self.scale_factor
        
        cx = self.cx_var.get()
        cy = self.cy_var.get()
        
        if self.dragging_endpoint == "v":
            # Dragging red line endpoint - change v_angle
            dx = frame_x - cx
            dy = frame_y - cy
            new_angle = np.degrees(np.arctan2(dy, dx))
            self.v_angle_var.set(round(new_angle % 180.0, 1))
        elif self.dragging_endpoint == "h":
            # Dragging blue line endpoint - change h_angle
            dx = frame_x - cx
            dy = frame_y - cy
            new_angle = np.degrees(np.arctan2(dy, dx))
            self.h_angle_var.set(round(new_angle % 180.0, 1))
        else:
            # Dragging center point
            delta_x = event.x - self.last_x
            delta_y = event.y - self.last_y
            
            frame_delta_x = delta_x / self.scale_factor
            frame_delta_y = delta_y / self.scale_factor
            
            new_cx = cx + frame_delta_x
            new_cy = cy + frame_delta_y
            
            new_cx = max(0, min(w_orig - 1, new_cx))
            new_cy = max(0, min(h_orig - 1, new_cy))
            
            self.cx_var.set(round(new_cx, 1))
            self.cy_var.set(round(new_cy, 1))
            
            self.last_x = event.x
            self.last_y = event.y
        
        self.update_image()
    
    def on_image_release(self, event):
        """Handle mouse release."""
        self.dragging_endpoint = None

    def on_mouse_move(self, event):
        """Display pixel color under the cursor when not dragging."""
        # Ignore updates while dragging (mouse button held down)
        if event.state & 0x0100:  # Left mouse button bitmask
            return

        h_orig, w_orig = self.original_frame.shape[:2]
        self.scale_factor = self.display_w / w_orig

        frame_x = int(event.x / self.scale_factor)
        frame_y = int(event.y / self.scale_factor)

        frame_x = np.clip(frame_x, 0, w_orig - 1)
        frame_y = np.clip(frame_y, 0, h_orig - 1)

        source_frame = self.processed_frame if self.processed_frame is not None else self.original_frame
        b, g, r = source_frame[frame_y, frame_x]
        brightness = int(0.299 * r + 0.587 * g + 0.114 * b)
        info_text = f"Pos: ({frame_x}, {frame_y})  Brightness: {brightness}"

        self.image_canvas.delete("pixel_info")
        self.image_canvas.delete("pixel_info_bg")
        self.image_canvas.create_rectangle(
            5, 45, 5 + 260, 70, fill="black", stipple="gray25", outline="", tags="pixel_info_bg"
        )
        self.image_canvas.create_text(
            10, 50, anchor="nw", text=info_text, fill="white",
            font=("Arial", 10), tags="pixel_info"
        )

    def on_mouse_leave(self, event):
        """Clear pixel info when cursor leaves the image."""
        self.image_canvas.delete("pixel_info")
        self.image_canvas.delete("pixel_info_bg")

    def start_optimization(self):
        """Kick off contrast optimization in background."""
        if getattr(self, "_optimizing", False):
            return
        self._optimizing = True
        self.optimize_button.config(state="disabled", text="Optimizing...")
        thread = threading.Thread(target=self._optimize_worker, daemon=True)
        thread.start()

    def _optimize_worker(self):
        """Search parameter space to maximize mid95 - main95."""
        h, w = self.original_frame.shape[:2]
        current_params = {
            "cx": self.cx_var.get(),
            "cy": self.cy_var.get(),
            "v_angle": self.v_angle_var.get(),
            "h_angle": self.h_angle_var.get(),
        }
        num_samples = self.num_samples_var.get()
        line_length = self.line_length_var.get()
        inner_gap = self.inner_exclusion_var.get()
        best_params = current_params.copy()
        best_score = self.evaluate_contrast_metric(
            line_length=line_length,
            inner_gap=inner_gap,
            num_samples=num_samples,
            **best_params,
        )

        diag = max(w, h)
        search_configs = [
            (diag / 2, 90, 2500),
            (diag / 4, 60, 2000),
            (diag / 8, 30, 1500),
            (diag / 16, 15, 1000),
            (diag / 32, 8, 700),
        ]
        rng = np.random.default_rng()
        final_rect_range = max(5, search_configs[-1][0])

        for pos_range, angle_range, samples in search_configs:
            pos_range = max(5, pos_range)
            angle_range = max(5, angle_range)

            stage_best = best_score
            stage_params = best_params.copy()

            for _ in range(int(samples)):
                cx_min = max(0, best_params["cx"] - pos_range)
                cx_max = min(w, best_params["cx"] + pos_range)
                cy_min = max(0, best_params["cy"] - pos_range)
                cy_max = min(h, best_params["cy"] + pos_range)

                candidate = {
                    "cx": rng.uniform(cx_min, cx_max),
                    "cy": rng.uniform(cy_min, cy_max),
                    "v_angle": (best_params["v_angle"] + rng.uniform(-angle_range, angle_range)) % 180,
                    "h_angle": (best_params["h_angle"] + rng.uniform(-angle_range, angle_range)) % 180,
                }
                if not self.angle_gap_ok(candidate["v_angle"], candidate["h_angle"]):
                    continue

                score = self.evaluate_contrast_metric(
                    line_length=line_length,
                    inner_gap=inner_gap,
                    num_samples=num_samples,
                    **candidate,
                )
                if score > stage_best:
                    stage_best = score
                    stage_params = candidate

            best_score = stage_best
            best_params = stage_params

        self.root.after(0, self._finish_optimization, best_params, final_rect_range)

    def _finish_optimization(self, params, search_range):
        """Apply best parameters and re-enable button."""
        if not self.angle_gap_ok(params["v_angle"], params["h_angle"]):
            params["h_angle"] = (params["v_angle"] + MIN_ANGLE_DIFF) % 180
        self.cx_var.set(params["cx"])
        self.cy_var.set(params["cy"])
        self.v_angle_var.set(params["v_angle"] % 180)
        self.h_angle_var.set(params["h_angle"] % 180)
        self.search_rect = (params["cx"], params["cy"], search_range)
        self.update_image()
        self.optimize_button.config(state="normal", text="Optimize Contrast")
        self._optimizing = False

    def evaluate_contrast_metric(self, cx, cy, v_angle, h_angle, line_length, inner_gap, num_samples):
        """Return contrast metric (mid95 - main95) for given parameters."""
        if not self.angle_gap_ok(v_angle, h_angle):
            return -np.inf
        candidate_frame = self.apply_clahe(self.original_frame.copy())
        v_pixels, _ = self.get_line_samples_with_radius(cx, cy, v_angle, line_length, num_samples=num_samples, inner_radius=0, frame=candidate_frame)
        h_pixels, _ = self.get_line_samples_with_radius(cx, cy, h_angle, line_length, num_samples=num_samples, inner_radius=0, frame=candidate_frame)
        mid_angle = (v_angle + h_angle) / 2.0
        mid2_angle = (mid_angle + 90) % 180
        mid1_pixels, _ = self.get_line_samples_with_radius(cx, cy, mid_angle, line_length, num_samples=num_samples, inner_radius=inner_gap, frame=candidate_frame)
        mid2_pixels, _ = self.get_line_samples_with_radius(cx, cy, mid2_angle, line_length, num_samples=num_samples, inner_radius=inner_gap, frame=candidate_frame)

        combined_mid = []
        if mid1_pixels is not None:
            combined_mid.extend(mid1_pixels.tolist())
        if mid2_pixels is not None:
            combined_mid.extend(mid2_pixels.tolist())

        combined_main = []
        if v_pixels is not None:
            combined_main.extend(v_pixels.tolist())
        if h_pixels is not None:
            combined_main.extend(h_pixels.tolist())

        if not combined_mid or not combined_main:
            return -np.inf

        mid95 = np.percentile(combined_mid, 95)
        main95 = np.percentile(combined_main, 95)
        return mid95 - main95

    @staticmethod
    def angle_gap_ok(v_angle, h_angle):
        """Ensure smallest angle difference exceeds required minimum."""
        diff = abs(((v_angle - h_angle + 90) % 180) - 90)
        return diff >= MIN_ANGLE_DIFF

    def adjust_cx(self, amount):
        current_val = self.cx_var.get()
        self.cx_var.set(round(current_val + amount, 1))
        self.update_image()

    def adjust_cy(self, amount):
        current_val = self.cy_var.get()
        self.cy_var.set(round(current_val + amount, 1))
        self.update_image()

    def adjust_v_angle(self, amount):
        current_val = self.v_angle_var.get()
        new_val = (current_val + amount) % 180.0
        self.v_angle_var.set(round(new_val, 1))
        self.update_image()

    def adjust_h_angle(self, amount):
        current_val = self.h_angle_var.get()
        new_val = (current_val + amount) % 180.0
        self.h_angle_var.set(round(new_val, 1))
        self.update_image()

    def adjust_line_length(self, amount):
        current_val = self.line_length_var.get()
        new_val = max(10, min(200, current_val + amount))
        self.line_length_var.set(new_val)
        self.update_image()

    def adjust_num_samples(self, amount):
        current_val = self.num_samples_var.get()
        new_val = max(3, min(50, current_val + amount))
        self.num_samples_var.set(new_val)
        self.update_image()

    def adjust_inner_gap(self, amount):
        current_val = self.inner_exclusion_var.get()
        new_val = max(0, min(150, current_val + amount))
        self.inner_exclusion_var.set(new_val)
        self.update_image()

    def adjust_frame(self, amount):
        """Adjust frame number by a given amount."""
        current_frame = self.frame_num_var.get()
        new_frame = max(0, min(self.total_frames - 1, current_frame + amount))
        self.frame_num_var.set(new_frame)
        self.load_frame(str(new_frame))

    def load_frame(self, frame_num_str):
        """Load a specific frame from the video."""
        frame_num = int(float(frame_num_str))
        self.current_frame_num = frame_num
        
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            self.original_frame = frame
            self.frame_label.config(text=f"{frame_num}")
            self.update_image()
        
    def update_image(self, *args):
        frame_copy = self.original_frame.copy()
        
        cx = self.cx_var.get()
        cy = self.cy_var.get()
        v_angle = self.v_angle_var.get()
        h_angle = self.h_angle_var.get()
        line_length = self.line_length_var.get()

        # Apply CLAHE and store processed frame
        frame_copy = self.apply_clahe(frame_copy)
        self.processed_frame = frame_copy.copy()

        num_samples = self.num_samples_var.get()
        inner_gap = self.inner_exclusion_var.get()

        # Draw crosshair with angled lines
        h, w = frame_copy.shape[:2]
        cx_int = int(cx)
        cy_int = int(cy)
        
        # First line with v_angle
        v_angle_rad = np.deg2rad(v_angle)
        v_dx = line_length * np.cos(v_angle_rad)
        v_dy = line_length * np.sin(v_angle_rad)
        v_x1 = int(cx_int - v_dx)
        v_y1 = int(cy_int - v_dy)
        v_x2 = int(cx_int + v_dx)
        v_y2 = int(cy_int + v_dy)
        v_x1 = np.clip(v_x1, 0, w - 1)
        v_y1 = np.clip(v_y1, 0, h - 1)
        v_x2 = np.clip(v_x2, 0, w - 1)
        v_y2 = np.clip(v_y2, 0, h - 1)
        cv2.line(frame_copy, (v_x1, v_y1), (v_x2, v_y2), (0, 0, 255), 2)  # Red
        
        # Second line perpendicular to first (90° offset)
        h_angle_rad = np.deg2rad(h_angle)
        h_dx = line_length * np.cos(h_angle_rad)
        h_dy = line_length * np.sin(h_angle_rad)
        h_x1 = int(cx_int - h_dx)
        h_y1 = int(cy_int - h_dy)
        h_x2 = int(cx_int + h_dx)
        h_y2 = int(cy_int + h_dy)
        h_x1 = np.clip(h_x1, 0, w - 1)
        h_y1 = np.clip(h_y1, 0, h - 1)
        h_x2 = np.clip(h_x2, 0, w - 1)
        h_y2 = np.clip(h_y2, 0, h - 1)
        cv2.line(frame_copy, (h_x1, h_y1), (h_x2, h_y2), (255, 0, 0), 2)  # Blue
        
        # Calculate in-between angles (bisector lines)
        mid_angle_1 = (v_angle + h_angle) / 2.0
        mid_angle_2 = mid_angle_1 + 90.0
        
        # First in-between line
        mid1_angle_rad = np.deg2rad(mid_angle_1)
        mid1_dx = line_length * np.cos(mid1_angle_rad)
        mid1_dy = line_length * np.sin(mid1_angle_rad)
        mid1_x1 = int(cx_int - mid1_dx)
        mid1_y1 = int(cy_int - mid1_dy)
        mid1_x2 = int(cx_int + mid1_dx)
        mid1_y2 = int(cy_int + mid1_dy)
        mid1_x1 = np.clip(mid1_x1, 0, w - 1)
        mid1_y1 = np.clip(mid1_y1, 0, h - 1)
        mid1_x2 = np.clip(mid1_x2, 0, w - 1)
        mid1_y2 = np.clip(mid1_y2, 0, h - 1)
        cv2.line(frame_copy, (mid1_x1, mid1_y1), (mid1_x2, mid1_y2), (0, 255, 0), 2)  # Green
        
        # Second in-between line (perpendicular to first in-between)
        mid2_angle_rad = np.deg2rad(mid_angle_2)
        mid2_dx = line_length * np.cos(mid2_angle_rad)
        mid2_dy = line_length * np.sin(mid2_angle_rad)
        mid2_x1 = int(cx_int - mid2_dx)
        mid2_y1 = int(cy_int - mid2_dy)
        mid2_x2 = int(cx_int + mid2_dx)
        mid2_y2 = int(cy_int + mid2_dy)
        mid2_x1 = np.clip(mid2_x1, 0, w - 1)
        mid2_y1 = np.clip(mid2_y1, 0, h - 1)
        mid2_x2 = np.clip(mid2_x2, 0, w - 1)
        mid2_y2 = np.clip(mid2_y2, 0, h - 1)
        cv2.line(frame_copy, (mid2_x1, mid2_y1), (mid2_x2, mid2_y2), (255, 255, 0), 2)  # Cyan
        
        # Draw a red dot at center
        cv2.circle(frame_copy, (cx_int, cy_int), 5, (0, 0, 255), -1)

        # Visualize inner exclusion circle if enabled
        if inner_gap > 0:
            gap_radius = int(inner_gap)
            cv2.circle(
                frame_copy,
                (cx_int, cy_int),
                gap_radius,
                (150, 150, 150),
                1,
                lineType=cv2.LINE_AA
            )
        
        # Visualize last search rectangle if available
        if self.search_rect:
            rect_cx, rect_cy, rect_range = self.search_rect
            x1 = int(np.clip(rect_cx - rect_range, 0, w - 1))
            x2 = int(np.clip(rect_cx + rect_range, 0, w - 1))
            y1 = int(np.clip(rect_cy - rect_range, 0, h - 1))
            y2 = int(np.clip(rect_cy + rect_range, 0, h - 1))
            cv2.rectangle(
                frame_copy,
                (x1, y1),
                (x2, y2),
                (200, 200, 50),
                1,
                cv2.LINE_AA
            )
        
        # Draw endpoint circles for red and blue lines (draggable)
        cv2.circle(frame_copy, (v_x2, v_y2), 7, (0, 0, 255), 2)  # Red line endpoint
        cv2.circle(frame_copy, (h_x2, h_y2), 7, (255, 0, 0), 2)  # Blue line endpoint
        
        # Sample pixels along all 4 lines (for dots & plot)
        v_pixels, v_points = self.get_line_samples_with_radius(cx, cy, v_angle, line_length, num_samples=num_samples, inner_radius=0)
        h_pixels, h_points = self.get_line_samples_with_radius(cx, cy, h_angle, line_length, num_samples=num_samples, inner_radius=0)
        mid1_angle = (v_angle + h_angle) / 2.0
        mid2_angle = mid1_angle + 90.0
        mid1_pixels, mid1_points = self.get_line_samples_with_radius(cx, cy, mid1_angle, line_length, num_samples=num_samples, inner_radius=inner_gap)
        mid2_pixels, mid2_points = self.get_line_samples_with_radius(cx, cy, mid2_angle, line_length, num_samples=num_samples, inner_radius=inner_gap)
        
        # Draw sample points on all four lines
        for px, py in v_points:
            cv2.circle(frame_copy, (px, py), 4, (0, 0, 255), -1)  # Red dots
        for px, py in h_points:
            cv2.circle(frame_copy, (px, py), 4, (255, 0, 0), -1)  # Blue dots
        for px, py in mid1_points:
            cv2.circle(frame_copy, (px, py), 4, (0, 255, 0), -1)  # Green dots
        for px, py in mid2_points:
            cv2.circle(frame_copy, (px, py), 4, (255, 255, 0), -1)  # Cyan dots
        
        # Display text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Center: ({cx:.0f}, {cy:.0f}) | V: {v_angle:.1f}° H: {h_angle:.1f}° | Length: {line_length}px"
        cv2.putText(frame_copy, text, (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Update labels
        self.cx_label.config(text=f"{cx:.1f}")
        self.cy_label.config(text=f"{cy:.1f}")
        self.v_angle_label.config(text=f"{v_angle:.1f}")
        self.h_angle_label.config(text=f"{h_angle:.1f}")
        self.line_length_label.config(text=f"{line_length}")
        self.num_samples_label.config(text=f"{num_samples}")
        self.inner_gap_label.config(text=f"{inner_gap:.0f}")
        self.frame_label.config(text=f"{self.frame_num_var.get()}")
        self.clahe_clip_label.config(text=f"{self.clahe_clip_limit.get():.1f}")
        
        # Draw pixel plot
        self.draw_pixel_plot(v_pixels, h_pixels, mid1_pixels, mid2_pixels)
        
        # --- Scale frame for display ---
        display_frame = cv2.resize(frame_copy, (self.display_w, self.display_h), interpolation=cv2.INTER_AREA)

        # Convert for Tkinter
        img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # Update canvas with image
        self.image_canvas.delete("image")
        self.image_canvas.create_image(0, 0, image=img_tk, anchor="nw", tags="image")
        self.image_on_canvas = img_tk  # Keep a reference to prevent garbage collection

def main(args):
    """Main function to load frame and launch the GUI."""
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frame >= total_frames:
        print(f"Error: Frame {args.frame} is out of bounds. Video has {total_frames} frames.")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Error: Could not read frame {args.frame}.")
        return

    root = tk.Tk()
    root.state('zoomed') # Maximize the window
    app = App(root, args, frame, args.video, total_frames)
    root.mainloop()

def parse_args():
    parser = argparse.ArgumentParser(description="Crosshair tool for marking positions on video frames.")
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to the video file.")
    parser.add_argument("--frame", type=int, default=2000, help="Frame number to load.")
    parser.add_argument("--center-x", type=int, default=None, help="Initial X coordinate of the crosshair center. Defaults to frame center.")
    parser.add_argument("--center-y", type=int, default=None, help="Initial Y coordinate of the crosshair center. Defaults to frame center.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

