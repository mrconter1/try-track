import cv2
import numpy as np
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

# --- Core OpenCV Functions (Largely Unchanged) ---

def draw_hough_line(frame, rho, theta_deg, color=(0, 0, 255), thickness=2):
    """Draws a line on a frame given rho and theta (in degrees)."""
    h, w = frame.shape[:2]
    theta_rad = np.deg2rad(theta_deg)
    
    a = np.cos(theta_rad)
    b = np.sin(theta_rad)
    x0 = a * rho
    y0 = b * rho
    
    x1 = int(x0 + 2000 * (-b))
    y1 = int(y0 + 2000 * (a))
    x2 = int(x0 - 2000 * (-b))
    y2 = int(y0 - 2000 * (a))
    
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    return frame

def get_line_metrics(frame, rho, theta_deg, num_samples):
    """
    Calculates metrics for a line.
    Returns (std_dev, pixel_sum, pixel_values)
    """
    if num_samples <= 0:
        return None, None, None
        
    h, w = frame.shape[:2]
    theta_rad = np.deg2rad(theta_deg)
    
    a = np.cos(theta_rad)
    b = np.sin(theta_rad)
    x0 = a * rho
    y0 = b * rho

    x1 = int(x0 + 2000 * (-b))
    y1 = int(y0 + 2000 * (a))
    x2 = int(x0 - 2000 * (-b))
    y2 = int(y0 - 2000 * (a))

    rect = (0, 0, w, h)
    inside, p1, p2 = cv2.clipLine(rect, (x1, y1), (x2, y2))

    if inside:
        x_coords = np.linspace(p1[0], p2[0], num_samples, dtype=int)
        y_coords = np.linspace(p1[1], p2[1], num_samples, dtype=int)
        x_coords = np.clip(x_coords, 0, w - 1)
        y_coords = np.clip(y_coords, 0, h - 1)
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pixel_values = gray_frame[y_coords, x_coords]
        return np.std(pixel_values), np.sum(pixel_values), pixel_values
    else:
        return None, None, None

def get_parametric_line_samples(rho, theta_deg, num_samples, frame_shape):
    """
    Returns parametric sample points along a line using Parametric Line with Normal Offset.
    Each sample includes (x, y) on the line and a parameter t in [0, 1].
    Returns list of (x, y, t) tuples or None if line is out of bounds.
    """
    if num_samples <= 0:
        return None
    
    h, w = frame_shape[:2]
    theta_rad = np.deg2rad(theta_deg)
    
    # Normal vector (perpendicular to line, pointing in +rho direction)
    normal_x = np.cos(theta_rad)
    normal_y = np.sin(theta_rad)
    
    # Direction vector along the line (perpendicular to normal)
    dir_x = -np.sin(theta_rad)
    dir_y = np.cos(theta_rad)
    
    # Point on the line at distance rho from origin
    start_x = rho * normal_x
    start_y = rho * normal_y
    
    # Extend in both directions to find endpoints
    x1 = int(start_x + 2000 * dir_x)
    y1 = int(start_y + 2000 * dir_y)
    x2 = int(start_x - 2000 * dir_x)
    y2 = int(start_y - 2000 * dir_y)
    
    rect = (0, 0, w, h)
    inside, p1, p2 = cv2.clipLine(rect, (x1, y1), (x2, y2))

    if inside:
        samples = []
        for i in range(num_samples):
            t = i / max(1, num_samples - 1)
            x = int(p1[0] + t * (p2[0] - p1[0]))
            y = int(p1[1] + t * (p2[1] - p1[1]))
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)
            samples.append((x, y, t))
        return samples
    else:
        return None

def get_perpendicular_offset_point(x, y, theta_deg, offset_distance):
    """
    Given a point (x, y) on a line, return the corresponding point offset
    perpendicular to the line by offset_distance.
    The offset is in the direction of the normal vector.
    """
    theta_rad = np.deg2rad(theta_deg)
    normal_x = np.cos(theta_rad)
    normal_y = np.sin(theta_rad)
    
    offset_x = int(x + offset_distance * normal_x)
    offset_y = int(y + offset_distance * normal_y)
    
    return (offset_x, offset_y)

# --- New Tkinter GUI Application ---

class App:
    def __init__(self, root, args, initial_frame, video_path, total_frames):
        self.root = root
        self.args = args
        self.original_frame = initial_frame
        self.video_path = video_path
        self.total_frames = total_frames
        self.current_frame_num = args.frame
        self.display_mode = args.display_mode
        
        self.root.title("Hough Line Control")
        
        # --- Variables ---
        h, w = self.original_frame.shape[:2]
        initial_cx = self.args.center_x if self.args.center_x is not None else w / 2
        initial_cy = self.args.center_y if self.args.center_y is not None else h / 2

        self.angle_var = tk.DoubleVar(value=self.args.angle)
        self.cx_var = tk.DoubleVar(value=initial_cx)
        self.cy_var = tk.DoubleVar(value=initial_cy)
        self.num_samples_var = tk.IntVar(value=self.args.num_samples)
        self.clahe_enabled = tk.BooleanVar(value=True)
        self.clahe_clip_limit = tk.DoubleVar(value=2.0)
        self.clahe_tile_size = tk.IntVar(value=8)
        self.frame_num_var = tk.IntVar(value=self.current_frame_num)
        self.probe_distance_var = tk.IntVar(value=15)
        
        # --- GUI Layout ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # --- Controls Frame (Create this first to measure its height) ---
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

        # Angle Controls
        ttk.Label(control_frame, text="Angle:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_angle(-0.1)).grid(row=1, column=1)
        self.angle_slider = tk.Scale(control_frame, from_=0, to=180, orient=tk.HORIZONTAL, variable=self.angle_var, command=self.update_image, resolution=0.01, showvalue=0)
        self.angle_slider.grid(row=1, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_angle(0.1)).grid(row=1, column=3)
        self.angle_label = ttk.Label(control_frame, text=f"{self.angle_var.get():.2f}", width=7)
        self.angle_label.grid(row=1, column=4, padx=5)

        # Center X Controls
        ttk.Label(control_frame, text="Center X:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_cx(-1)).grid(row=2, column=1)
        self.cx_slider = tk.Scale(control_frame, from_=0, to=w, orient=tk.HORIZONTAL, variable=self.cx_var, command=self.update_image, resolution=0.1, showvalue=0)
        self.cx_slider.grid(row=2, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_cx(1)).grid(row=2, column=3)
        self.cx_label = ttk.Label(control_frame, text=f"{self.cx_var.get():.1f}", width=7)
        self.cx_label.grid(row=2, column=4, padx=5)

        # Center Y Controls
        ttk.Label(control_frame, text="Center Y:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_cy(-1)).grid(row=3, column=1)
        self.cy_slider = tk.Scale(control_frame, from_=0, to=h, orient=tk.HORIZONTAL, variable=self.cy_var, command=self.update_image, resolution=0.1, showvalue=0)
        self.cy_slider.grid(row=3, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_cy(1)).grid(row=3, column=3)
        self.cy_label = ttk.Label(control_frame, text=f"{self.cy_var.get():.1f}", width=7)
        self.cy_label.grid(row=3, column=4, padx=5)

        # Num Samples Controls
        ttk.Label(control_frame, text="Samples:").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_num_samples(-5)).grid(row=4, column=1)
        self.num_samples_slider = tk.Scale(control_frame, from_=10, to=500, orient=tk.HORIZONTAL, variable=self.num_samples_var, command=self.update_image, resolution=1, showvalue=0)
        self.num_samples_slider.grid(row=4, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_num_samples(5)).grid(row=4, column=3)
        self.num_samples_label = ttk.Label(control_frame, text=f"{self.num_samples_var.get()}", width=7)
        self.num_samples_label.grid(row=4, column=4, padx=5)

        # CLAHE Enable/Disable
        ttk.Label(control_frame, text="CLAHE:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.clahe_checkbox = ttk.Checkbutton(control_frame, variable=self.clahe_enabled, command=self.update_image)
        self.clahe_checkbox.grid(row=5, column=1, sticky=tk.W)
        
        # CLAHE Clip Limit
        ttk.Label(control_frame, text="Clip:").grid(row=5, column=2, sticky=tk.W, padx=(10, 0))
        self.clahe_clip_slider = tk.Scale(control_frame, from_=1.0, to=10.0, orient=tk.HORIZONTAL, variable=self.clahe_clip_limit, command=self.update_image, resolution=0.5, showvalue=0)
        self.clahe_clip_slider.grid(row=5, column=3, sticky="ew")
        self.clahe_clip_label = ttk.Label(control_frame, text=f"{self.clahe_clip_limit.get():.1f}", width=5)
        self.clahe_clip_label.grid(row=5, column=4, padx=5)

        # Probe Distance Controls
        ttk.Label(control_frame, text="Probe Dist:").grid(row=6, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_probe_distance(-1)).grid(row=6, column=1)
        self.probe_distance_slider = tk.Scale(control_frame, from_=1, to=50, orient=tk.HORIZONTAL, variable=self.probe_distance_var, command=self.update_image, resolution=1, showvalue=0)
        self.probe_distance_slider.grid(row=6, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_probe_distance(1)).grid(row=6, column=3)
        self.probe_distance_label = ttk.Label(control_frame, text=f"{self.probe_distance_var.get()}", width=5)
        self.probe_distance_label.grid(row=6, column=4, padx=5)
        
        # --- Search Button ---
        self.search_button = ttk.Button(control_frame, text="Find Best Fit", command=self.start_best_fit_search)
        self.search_button.grid(row=0, column=5, rowspan=7, padx=10, sticky="ns")

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
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew")
        
        # Pixel Plot Canvas
        self.plot_canvas = tk.Canvas(main_frame, bg="white", width=200, height=self.display_h)
        self.plot_canvas.grid(row=0, column=1, sticky="nsew")

        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3) # Image gets 3/4 of the space
        main_frame.columnconfigure(1, weight=1) # Plot gets 1/4 of the space

        self.update_image()
    
    def start_best_fit_search(self):
        """Starts the grid search in a new thread to avoid freezing the GUI."""
        self.search_button.config(state="disabled", text="Searching...")
        
        # Get the metrics for the current line to set a baseline
        angle_deg = self.angle_var.get()
        cx = self.cx_var.get()
        cy = self.cy_var.get()
        
        theta_deg = (angle_deg + 90) % 180
        theta_rad = np.deg2rad(theta_deg)
        rho = cx * np.cos(theta_rad) + cy * np.sin(theta_rad)
        
        # Calculate initial symmetric difference metric
        probe_distance = self.probe_distance_var.get()
        rho_probe = rho + probe_distance
        rho_probe2 = rho - probe_distance
        _, _, p_main = get_line_metrics(self.original_frame, rho, theta_deg, self.args.num_samples)
        _, _, p_probe1 = get_line_metrics(self.original_frame, rho_probe, theta_deg, self.args.num_samples)
        _, _, p_probe2 = get_line_metrics(self.original_frame, rho_probe2, theta_deg, self.args.num_samples)

        initial_metric = float('inf')
        if p_main is not None and p_probe1 is not None and p_probe2 is not None and len(p_main) == len(p_probe1) and len(p_main) == len(p_probe2):
            diff_values = (p_main.astype(float) - (p_probe1.astype(float) + p_probe2.astype(float)) / 2.0)
            if diff_values.size > 0:
                initial_metric = np.percentile(diff_values, 5)

        # Run the actual search in a worker thread
        search_thread = threading.Thread(target=self._grid_search_worker, args=(initial_metric,))
        search_thread.daemon = True # Allows main program to exit even if thread is running
        search_thread.start()

    def _grid_search_worker(self, initial_metric):
        """The long-running grid search task."""
        center_cx = self.cx_var.get()
        center_cy = self.cy_var.get()
        center_angle = self.angle_var.get()

        best_cx = center_cx
        best_cy = center_cy
        best_angle = center_angle
        min_metric = initial_metric if initial_metric is not None else float('inf')

        # Get probe distance once for the search loop
        probe_distance = self.probe_distance_var.get()

        iterations = zip(self.args.position_ranges, self.args.angle_ranges, self.args.num_search_samples)
        total_iterations = len(self.args.position_ranges)

        for iter_num, (pos_range, angle_range, num_samples) in enumerate(iterations):
            print(f"--- Iteration {iter_num + 1}/{total_iterations}: Pos Range={pos_range}, Angle Range={angle_range}, Samples={num_samples} ---")
            
            # Define the search space for the current iteration
            cx_min, cx_max = center_cx - pos_range, center_cx + pos_range
            cy_min, cy_max = center_cy - pos_range, center_cy + pos_range
            angle_min, angle_max = center_angle - angle_range, center_angle + angle_range

            for i in range(num_samples):
                cx = np.random.uniform(cx_min, cx_max)
                cy = np.random.uniform(cy_min, cy_max)
                angle_raw = np.random.uniform(angle_min, angle_max)
                angle = angle_raw % 180.0

                # Convert to rho/theta for metrics calculation
                theta_deg = (angle + 90) % 180
                theta_rad = np.deg2rad(theta_deg)
                rho = cx * np.cos(theta_rad) + cy * np.sin(theta_rad)
                
                # Calculate symmetric difference metric
                rho_probe = rho + probe_distance
                rho_probe2 = rho - probe_distance
                
                _, _, p_main = get_line_metrics(self.original_frame, rho, theta_deg, self.args.num_samples)
                _, _, p_probe1 = get_line_metrics(self.original_frame, rho_probe, theta_deg, self.args.num_samples)
                _, _, p_probe2 = get_line_metrics(self.original_frame, rho_probe2, theta_deg, self.args.num_samples)

                if p_main is not None and p_probe1 is not None and p_probe2 is not None and len(p_main) == len(p_probe1) and len(p_main) == len(p_probe2):
                    diff_values = (p_main.astype(float) - (p_probe1.astype(float) + p_probe2.astype(float)) / 2.0)
                    
                    if diff_values.size > 0:
                        current_metric = np.percentile(diff_values, 5)
                        if current_metric < min_metric:
                            min_metric = current_metric
                            best_cx = cx
                            best_cy = cy
                            best_angle = angle

                # Print progress to the console periodically
                if (i + 1) % 200 == 0 or (i + 1) == num_samples:
                    progress = ((i + 1) / num_samples) * 100
                    print(f"  Search progress: {progress:.1f}%")
            
            # Update the center for the next iteration to be the best point found so far
            center_cx, center_cy, center_angle = best_cx, best_cy, best_angle
        
        # When done, schedule an update on the main GUI thread
        self.root.after(0, self.finish_best_fit_search, best_cx, best_cy, best_angle)

    def finish_best_fit_search(self, best_cx, best_cy, best_angle):
        """Updates the GUI with the results from the search."""
        self.cx_var.set(best_cx)
        self.cy_var.set(best_cy)
        self.angle_var.set(best_angle)
        
        self.search_button.config(state="normal", text="Find Best Fit")
        
        self.update_image()


    def draw_pixel_plot(self, pixel_values, probe_pixel_values=None, probe2_pixel_values=None, show_only_max=False):
        self.plot_canvas.delete("all") # Clear previous plot

        if pixel_values is None or len(pixel_values) == 0:
            self.plot_canvas.create_text(10, 10, anchor="nw", text="Line is out of bounds.")
            return

        canvas_w = self.plot_canvas.winfo_width()
        canvas_h = self.plot_canvas.winfo_height()

        if canvas_w < 2 or canvas_h < 2: # Canvas not ready on first draw
             self.root.after(50, lambda: self.draw_pixel_plot(pixel_values, probe_pixel_values))
             return

        # Create points for the main line graph
        points = []
        num_samples = len(pixel_values)
        x_step = canvas_w / max(1, num_samples - 1)

        for i, value in enumerate(pixel_values):
            x = i * x_step
            y = canvas_h - (value / 255.0) * canvas_h # Invert Y-axis for drawing
            points.extend([x, y])

        if len(points) > 2:
            self.plot_canvas.create_line(points, fill="red", width=2)

        if show_only_max:
            if pixel_values is not None and len(pixel_values) > 0:
                max_value = np.max(pixel_values)
                max_y = canvas_h - (max_value / 255.0) * canvas_h
                self.plot_canvas.create_line(0, max_y, canvas_w, max_y, fill="orange", width=2, dash=(4, 4))
                self.plot_canvas.create_text(canvas_w - 60, max_y - 10, anchor="w", text=f"Max: {int(max_value)}", font=("Arial", 8))

            # Draw Y-axis labels for context
            for val in range(0, 256, 25):
                y = canvas_h - (val / 255.0) * canvas_h
                self.plot_canvas.create_text(15, y, anchor="w", text=str(val), font=("Arial", 9))
                self.plot_canvas.create_line(0, y, 10, y)
            return

        # Draw probe line if available
        if probe_pixel_values is not None and len(probe_pixel_values) > 0:
            probe_points = []
            for i, value in enumerate(probe_pixel_values):
                x = i * x_step
                y = canvas_h - (value / 255.0) * canvas_h
                probe_points.extend([x, y])
            
            if len(probe_points) > 2:
                self.plot_canvas.create_line(probe_points, fill="green", width=2)

        # Draw second probe line if available
        if probe2_pixel_values is not None and len(probe2_pixel_values) > 0:
            probe2_points = []
            for i, value in enumerate(probe2_pixel_values):
                x = i * x_step
                y = canvas_h - (value / 255.0) * canvas_h
                probe2_points.extend([x, y])
            
            if len(probe2_points) > 2:
                self.plot_canvas.create_line(probe2_points, fill="yellow", width=2)

        # Draw Symmetric Difference line (main - avg(probe1, probe2))
        if probe_pixel_values is not None and probe2_pixel_values is not None and \
           len(pixel_values) == len(probe_pixel_values) and len(pixel_values) == len(probe2_pixel_values):
            
            diff_points = []
            diff_values = []
            for i, (main_val, probe1_val, probe2_val) in enumerate(zip(pixel_values, probe_pixel_values, probe2_pixel_values)):
                avg_probe = (float(probe1_val) + float(probe2_val)) / 2.0
                diff = float(main_val) - avg_probe
                diff_values.append(diff)
                
                # Center the difference at canvas_h/2 and scale it
                x = i * x_step
                y = canvas_h / 2 - (diff / 255.0) * (canvas_h / 2)
                diff_points.extend([x, y])
            
            if len(diff_points) > 2:
                self.plot_canvas.create_line(diff_points, fill="purple", width=2)
            
            # Draw zero difference line (center)
            center_y = canvas_h / 2
            self.plot_canvas.create_line(0, center_y, canvas_w, center_y, fill="purple", width=1, dash=(2, 2))
            
            # --- Calculate and draw stats for the Symmetric Difference ---
            
            # Median
            median_diff = np.median(diff_values)
            median_y = canvas_h / 2 - (median_diff / 255.0) * (canvas_h / 2)
            self.plot_canvas.create_line(0, median_y, canvas_w, median_y, fill="orange", width=2, dash=(4, 4))

            # Min
            min_diff = np.min(diff_values)
            min_y = canvas_h / 2 - (min_diff / 255.0) * (canvas_h / 2)
            self.plot_canvas.create_line(0, min_y, canvas_w, min_y, fill="cyan", width=2, dash=(2, 2))

            # 5th Percentile
            percentile_5_diff = np.percentile(diff_values, 5)
            percentile_5_y = canvas_h / 2 - (percentile_5_diff / 255.0) * (canvas_h / 2)
            self.plot_canvas.create_line(0, percentile_5_y, canvas_w, percentile_5_y, fill="magenta", width=2, dash=(2, 4))
            
        else:
            # --- Fallback to drawing stats for the main line if no probe data ---
            
            # Calculate and draw median line (for main line only)
            median_value = np.median(pixel_values)
            median_y = canvas_h - (median_value / 255.0) * canvas_h
            self.plot_canvas.create_line(0, median_y, canvas_w, median_y, fill="orange", width=2, dash=(4, 4))

            # Calculate and draw min line (for main line only)
            min_value = np.min(pixel_values)
            min_y = canvas_h - (min_value / 255.0) * canvas_h
            self.plot_canvas.create_line(0, min_y, canvas_w, min_y, fill="cyan", width=2, dash=(2, 2))

            # Calculate and draw 5th percentile line (for main line only)
            percentile_5 = np.percentile(pixel_values, 5)
            percentile_5_y = canvas_h - (percentile_5 / 255.0) * canvas_h
            self.plot_canvas.create_line(0, percentile_5_y, canvas_w, percentile_5_y, fill="magenta", width=2, dash=(2, 4))


        # Draw Y-axis labels for context
        for val in range(0, 256, 25):
            y = canvas_h - (val / 255.0) * canvas_h
            self.plot_canvas.create_text(15, y, anchor="w", text=str(val), font=("Arial", 9))
            self.plot_canvas.create_line(0, y, 10, y)

        # Draw legend
        legend_y = 10
        self.plot_canvas.create_line(canvas_w - 90, legend_y, canvas_w - 70, legend_y, fill="red", width=2)
        self.plot_canvas.create_text(canvas_w - 65, legend_y, anchor="w", text="Main", font=("Arial", 8))
        
        legend_y += 15
        self.plot_canvas.create_line(canvas_w - 90, legend_y, canvas_w - 70, legend_y, fill="green", width=2)
        self.plot_canvas.create_text(canvas_w - 65, legend_y, anchor="w", text="Probe", font=("Arial", 8))
        
        legend_y += 15
        self.plot_canvas.create_line(canvas_w - 90, legend_y, canvas_w - 70, legend_y, fill="yellow", width=2)
        self.plot_canvas.create_text(canvas_w - 65, legend_y, anchor="w", text="Probe 2", font=("Arial", 8))
        
        legend_y += 15
        self.plot_canvas.create_line(canvas_w - 90, legend_y, canvas_w - 70, legend_y, fill="purple", width=2)
        self.plot_canvas.create_text(canvas_w - 65, legend_y, anchor="w", text="Diff", font=("Arial", 8))
        
        legend_y += 15
        self.plot_canvas.create_line(canvas_w - 90, legend_y, canvas_w - 70, legend_y, fill="orange", width=2, dash=(4, 4))
        self.plot_canvas.create_text(canvas_w - 65, legend_y, anchor="w", text="Median", font=("Arial", 8))
        
        legend_y += 15
        self.plot_canvas.create_line(canvas_w - 90, legend_y, canvas_w - 70, legend_y, fill="cyan", width=2, dash=(2, 2))
        self.plot_canvas.create_text(canvas_w - 65, legend_y, anchor="w", text="Min", font=("Arial", 8))
        
        legend_y += 15
        self.plot_canvas.create_line(canvas_w - 90, legend_y, canvas_w - 70, legend_y, fill="magenta", width=2, dash=(2, 4))
        self.plot_canvas.create_text(canvas_w - 65, legend_y, anchor="w", text="5th %ile", font=("Arial", 8))


    def adjust_angle(self, amount):
        current_val = self.angle_var.get()
        self.angle_var.set(round(current_val + amount, 2))
        self.update_image()

    def adjust_cx(self, amount):
        current_val = self.cx_var.get()
        self.cx_var.set(round(current_val + amount, 1))
        self.update_image()

    def adjust_cy(self, amount):
        current_val = self.cy_var.get()
        self.cy_var.set(round(current_val + amount, 1))
        self.update_image()

    def adjust_num_samples(self, amount):
        current_val = self.num_samples_var.get()
        self.num_samples_var.set(max(10, current_val + amount))
        self.update_image()

    def adjust_probe_distance(self, amount):
        current_val = self.probe_distance_var.get()
        self.probe_distance_var.set(max(1, min(50, current_val + amount)))
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
        # Get user-friendly parameters from the GUI
        angle_deg = self.angle_var.get()
        cx = self.cx_var.get()
        cy = self.cy_var.get()

        # --- Convert (cx, cy, angle) to (rho, theta) ---
        # Theta is the angle of the normal, so it's 90 degrees offset from the line's angle.
        # We constrain it to [0, 180) as required by OpenCV's line representation.
        theta_deg = (angle_deg + 90) % 180
        theta_rad = np.deg2rad(theta_deg)
        
        # Rho is the distance from the origin to the line, calculated by projecting
        # the center point onto the normal vector.
        rho = cx * np.cos(theta_rad) + cy * np.sin(theta_rad)

        frame_copy = self.original_frame.copy()
        
        # Apply CLAHE if enabled
        frame_copy = self.apply_clahe(frame_copy)
        
        # Draw line
        frame_with_line = draw_hough_line(frame_copy, rho, theta_deg)
        
        show_only_main = self.display_mode == "minimal"

        probe_distance = self.probe_distance_var.get()
        num_samples = self.num_samples_var.get()

        # Calculate Std Dev for main line
        std_dev, _, pixel_values = get_line_metrics(self.original_frame, rho, theta_deg, num_samples)

        probe_pixel_values = None
        probe2_pixel_values = None

        if not show_only_main:
            # Draw parallel probe line at configurable distance
            rho_probe = rho + probe_distance
            frame_with_line = draw_hough_line(frame_with_line, rho_probe, theta_deg, color=(0, 255, 0), thickness=2)

            # Draw second probe line on the other side
            rho_probe2 = rho - probe_distance
            frame_with_line = draw_hough_line(frame_with_line, rho_probe2, theta_deg, color=(0, 255, 255), thickness=2)

            # Get pixel values for probe lines
            _, _, probe_pixel_values = get_line_metrics(self.original_frame, rho_probe, theta_deg, num_samples)
            _, _, probe2_pixel_values = get_line_metrics(self.original_frame, rho_probe2, theta_deg, num_samples)

            # Get parametric sample points for main line
            main_sample_points = get_parametric_line_samples(rho, theta_deg, num_samples, self.original_frame.shape)

            # Draw sample points on the frame with perpendicular correspondences
            if main_sample_points:
                for x, y, t in main_sample_points:
                    cv2.circle(frame_with_line, (x, y), 5, (0, 0, 255), -1)  # Red circles for main

                    # Get corresponding probe point by perpendicular offset
                    probe_x, probe_y = get_perpendicular_offset_point(x, y, theta_deg, probe_distance)
                    cv2.circle(frame_with_line, (probe_x, probe_y), 5, (0, 255, 0), -1)  # Green circles for probe

                    # Get corresponding second probe point by perpendicular offset
                    probe2_x, probe2_y = get_perpendicular_offset_point(x, y, theta_deg, -probe_distance)
                    cv2.circle(frame_with_line, (probe2_x, probe2_y), 5, (0, 255, 255), -1) # Yellow circles for probe 2
        else:
            # In minimal mode, ensure only the main line is visible
            pass

        # Draw a blue dot at the center point last so it remains visible (only in full mode)
        if not show_only_main:
            cv2.circle(frame_with_line, (int(cx), int(cy)), 5, (255, 0, 0), -1)

        # Display text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Angle: {angle_deg:.1f}, Center: ({cx:.0f}, {cy:.0f})"
        cv2.putText(frame_with_line, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        if std_dev is not None:
            std_dev_text = f"Std Dev: {std_dev:.2f}"
            cv2.putText(frame_with_line, std_dev_text, (10, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # --- Update the pixel plot ---
        self.draw_pixel_plot(pixel_values, probe_pixel_values, probe2_pixel_values, show_only_max=show_only_main)

        # --- Scale frame for display ---
        display_frame = cv2.resize(frame_with_line, (self.display_w, self.display_h), interpolation=cv2.INTER_AREA)

        # Convert for Tkinter
        img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        self.image_label.imgtk = img_tk
        self.image_label.configure(image=img_tk)

        # Update labels
        self.angle_label.config(text=f"{angle_deg:.2f}")
        self.cx_label.config(text=f"{cx:.1f}")
        self.cy_label.config(text=f"{cy:.1f}")
        self.num_samples_label.config(text=f"{num_samples}")
        self.clahe_clip_label.config(text=f"{self.clahe_clip_limit.get():.1f}")
        self.probe_distance_label.config(text=f"{self.probe_distance_var.get()}")

def main(args):
    """Main function to load frame and launch the GUI."""
    num_pos_ranges = len(args.position_ranges)
    num_angle_ranges = len(args.angle_ranges)
    num_samples_list = len(args.num_search_samples)

    if num_pos_ranges != num_angle_ranges:
        print("Error: --position-ranges and --angle-ranges must have the same number of arguments.")
        return

    if num_samples_list == 1:
        # If one sample count is given, repeat it for all iterations
        args.num_search_samples = args.num_search_samples * num_pos_ranges
    elif num_samples_list != num_pos_ranges:
        print("Error: --num-search-samples must have either 1 argument, or the same number of arguments as the range lists.")
        return

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
    parser = argparse.ArgumentParser(description="Draw a single Hough line on a video frame interactively.")
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to the video file.")
    parser.add_argument("--frame", type=int, default=2000, help="Frame number to load.")
    parser.add_argument("--angle", type=float, default=45.0, help="The initial angle of the line (in degrees).")
    parser.add_argument("--center-x", type=int, default=None, help="Initial X coordinate of the line's center point. Defaults to frame center.")
    parser.add_argument("--center-y", type=int, default=None, help="Initial Y coordinate of the line's center point. Defaults to frame center.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples for color std deviation.")
    parser.add_argument("--position-ranges", type=float, nargs='+', default=[20.0, 10.0, 5.0], help="Search ranges for cx and cy for each iteration.")
    parser.add_argument("--angle-ranges", type=float, nargs='+', default=[180.0, 20.0, 5.0], help="Search ranges for the angle for each iteration.")
    parser.add_argument("--num-search-samples", type=int, nargs='+', default=[1000, 1000, 500], help="Number of random samples for each iteration of the darkest line search.")
    parser.add_argument("--display-mode", type=str, choices=["full", "minimal"], default="full", help="Display either the full set of overlays or only the main line and its max intensity line.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
