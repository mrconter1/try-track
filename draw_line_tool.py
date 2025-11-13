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

def get_parallel_line_pixel_values(frame, cx, cy, angle_deg, probe_offset, num_samples):
    """
    Gets pixel values for two parallel lines, ensuring samples correspond spatially.
    Returns (main_pixel_values, probe_pixel_values) where values can be np.nan
    if the sample point is outside the frame.
    """
    h, w = frame.shape[:2]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    angle_rad = np.deg2rad(angle_deg)
    v_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])       # Vector along the line
    v_norm = np.array([-np.sin(angle_rad), np.cos(angle_rad)])    # Vector perpendicular to the line

    center_point = np.array([cx, cy])
    
    # Define a sampling axis much longer than the frame diagonal to ensure full coverage
    max_dist = np.sqrt(h**2 + w**2)
    distances = np.linspace(-max_dist, max_dist, num_samples)

    main_values = []
    probe_values = []

    for s in distances:
        # Calculate the point on the main line
        main_point = center_point + s * v_dir
        # Calculate the corresponding point on the probe line
        probe_point = main_point + probe_offset * v_norm

        # Check main point
        mx, my = int(main_point[0]), int(main_point[1])
        if 0 <= mx < w and 0 <= my < h:
            main_values.append(gray_frame[my, mx])
        else:
            main_values.append(np.nan)

        # Check probe point
        px, py = int(probe_point[0]), int(probe_point[1])
        if 0 <= px < w and 0 <= py < h:
            probe_values.append(gray_frame[py, px])
        else:
            probe_values.append(np.nan)

    return np.array(main_values), np.array(probe_values)


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

# --- New Tkinter GUI Application ---

class App:
    def __init__(self, root, args, initial_frame):
        self.root = root
        self.args = args
        self.original_frame = initial_frame
        
        self.root.title("Hough Line Control")
        
        # --- Variables ---
        h, w = self.original_frame.shape[:2]
        initial_cx = self.args.center_x if self.args.center_x is not None else w / 2
        initial_cy = self.args.center_y if self.args.center_y is not None else h / 2

        self.angle_var = tk.DoubleVar(value=self.args.angle)
        self.cx_var = tk.DoubleVar(value=initial_cx)
        self.cy_var = tk.DoubleVar(value=initial_cy)
        self.probe_offset_var = tk.DoubleVar(value=self.args.probe_offset)
        
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
        
        # Angle Controls
        ttk.Label(control_frame, text="Angle:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_angle(-0.1)).grid(row=0, column=1)
        self.angle_slider = tk.Scale(control_frame, from_=0, to=180, orient=tk.HORIZONTAL, variable=self.angle_var, command=self.update_image, resolution=0.01, showvalue=0)
        self.angle_slider.grid(row=0, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_angle(0.1)).grid(row=0, column=3)
        self.angle_label = ttk.Label(control_frame, text=f"{self.angle_var.get():.2f}", width=7)
        self.angle_label.grid(row=0, column=4, padx=5)

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
        
        # Probe Offset Controls
        ttk.Label(control_frame, text="Probe Offset:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_probe_offset(-1)).grid(row=3, column=1)
        self.probe_offset_slider = tk.Scale(control_frame, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.probe_offset_var, command=self.update_image, resolution=0.1, showvalue=0)
        self.probe_offset_slider.grid(row=3, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_probe_offset(1)).grid(row=3, column=3)
        self.probe_offset_label = ttk.Label(control_frame, text=f"{self.probe_offset_var.get():.1f}", width=7)
        self.probe_offset_label.grid(row=3, column=4, padx=5)

        # --- Search Button ---
        self.search_button = ttk.Button(control_frame, text="Find Max Difference", command=self.start_line_search)
        self.search_button.grid(row=0, column=5, rowspan=4, padx=10, sticky="ns")

        control_frame.columnconfigure(2, weight=1) # Make slider stretch

        # --- Force update to calculate control frame's actual size ---
        self.root.update_idletasks()
        control_height = control_frame.winfo_reqheight()

        # --- Now, dynamically set Display Size based on remaining space ---
        max_width = root.winfo_screenwidth() - 40   # Padding for window borders
        max_height = root.winfo_screenheight() - control_height - 100 # Padding for borders and taskbar
        
        h_orig, w_orig = self.original_frame.shape[:2]
        ratio = min(max_width / w_orig, max_height / h_orig)
        
        self.display_w = int(w_orig * ratio)
        self.display_h = int(h_orig * ratio)

        # --- Image and Plot Layout ---
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew")
        
        # Pixel Plot Canvas
        self.plot_canvas = tk.Canvas(main_frame, bg="white", width=200)
        self.plot_canvas.grid(row=0, column=1, sticky="nsew")

        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3) # Image gets 3/4 of the space
        main_frame.columnconfigure(1, weight=1) # Plot gets 1/4 of the space

        self.update_image()
    
    def start_line_search(self):
        """Starts the grid search in a new thread to avoid freezing the GUI."""
        self.search_button.config(state="disabled", text="Searching...")
        
        # Get the current parameters
        angle_deg = self.angle_var.get()
        cx = self.cx_var.get()
        cy = self.cy_var.get()
        probe_offset = self.probe_offset_var.get()
        
        # Get the metrics for the current line to set a baseline
        main_vals, probe_vals = get_parallel_line_pixel_values(
            self.original_frame, cx, cy, angle_deg, probe_offset, self.args.num_samples
        )
        initial_diff_sum = np.nansum(probe_vals - main_vals)

        # Run the actual search in a worker thread
        search_thread = threading.Thread(target=self._grid_search_worker, args=(initial_diff_sum, probe_offset))
        search_thread.daemon = True # Allows main program to exit even if thread is running
        search_thread.start()

    def _grid_search_worker(self, initial_diff_sum, probe_offset):
        """The long-running grid search task."""
        center_cx = self.cx_var.get()
        center_cy = self.cy_var.get()
        center_angle = self.angle_var.get()

        best_cx = center_cx
        best_cy = center_cy
        best_angle = center_angle
        max_diff_sum = initial_diff_sum if initial_diff_sum is not None else float('-inf')

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

                main_vals, probe_vals = get_parallel_line_pixel_values(
                    self.original_frame, cx, cy, angle, probe_offset, self.args.num_samples
                )
                
                current_diff_sum = np.nansum(probe_vals - main_vals)
                
                if current_diff_sum > max_diff_sum:
                    max_diff_sum = current_diff_sum
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
        self.root.after(0, self.finish_line_search, best_cx, best_cy, best_angle)

    def finish_line_search(self, best_cx, best_cy, best_angle):
        """Updates the GUI with the results from the search."""
        self.cx_var.set(best_cx)
        self.cy_var.set(best_cy)
        self.angle_var.set(best_angle)
        
        self.search_button.config(state="normal", text="Find Max Difference")
        
        self.update_image()


    def draw_pixel_plot(self, main_pixel_values, probe_pixel_values, diff_pixel_values):
        self.plot_canvas.delete("all") # Clear previous plot

        # --- Helper to draw an intensity plot (0-255 range) ---
        def _draw_intensity_plot(pixel_values, color):
            if pixel_values is None or len(pixel_values) == 0:
                return

            canvas_w = self.plot_canvas.winfo_width()
            canvas_h = self.plot_canvas.winfo_height()

            # Create points, handling NaNs
            num_samples = len(pixel_values)
            x_step = canvas_w / max(1, num_samples - 1)

            current_segment = []
            for i, value in enumerate(pixel_values):
                if not np.isnan(value):
                    x = i * x_step
                    y = canvas_h - (value / 255.0) * canvas_h
                    current_segment.extend([x, y])
                else:
                    if len(current_segment) > 2:
                        self.plot_canvas.create_line(current_segment, fill=color, width=2)
                    current_segment = []
            if len(current_segment) > 2:
                self.plot_canvas.create_line(current_segment, fill=color, width=2)

        # --- Helper to draw a difference plot (-255 to 255 range) ---
        def _draw_difference_plot(pixel_values, color):
            if pixel_values is None or len(pixel_values) == 0:
                return

            canvas_w = self.plot_canvas.winfo_width()
            canvas_h = self.plot_canvas.winfo_height()

            # Create points, handling NaNs
            num_samples = len(pixel_values)
            x_step = canvas_w / max(1, num_samples - 1)

            current_segment = []
            for i, value in enumerate(pixel_values):
                if not np.isnan(value):
                    x = i * x_step
                    # Scale from -255 to 255, with 0 in the middle
                    y = (canvas_h / 2) - (value / 255.0) * (canvas_h / 2)
                    current_segment.extend([x, y])
                else:
                    if len(current_segment) > 2:
                        self.plot_canvas.create_line(current_segment, fill=color, width=1)
                    current_segment = []
            if len(current_segment) > 2:
                self.plot_canvas.create_line(current_segment, fill=color, width=1)

        # --- Check canvas readiness and draw plots ---
        canvas_w = self.plot_canvas.winfo_width()
        canvas_h = self.plot_canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2:
             self.root.after(50, lambda: self.draw_pixel_plot(main_pixel_values, probe_pixel_values, diff_pixel_values))
             return

        # Draw the plots
        _draw_intensity_plot(main_pixel_values, "red")
        _draw_intensity_plot(probe_pixel_values, "green")
        _draw_difference_plot(diff_pixel_values, "blue")

        # Draw axis labels and lines for context
        has_data = not (np.all(np.isnan(main_pixel_values)) and
                        np.all(np.isnan(probe_pixel_values)))

        if has_data:
            # Intensity scale (left)
            self.plot_canvas.create_text(15, 10, anchor="nw", text="255", font=("Arial", 10), fill="black")
            self.plot_canvas.create_line(0, 10, 10, 10)
            self.plot_canvas.create_text(15, canvas_h - 10, anchor="sw", text="0", font=("Arial", 10), fill="black")
            self.plot_canvas.create_line(0, canvas_h - 10, 10, canvas_h - 10)

            # Difference scale (right)
            self.plot_canvas.create_line(0, canvas_h/2, canvas_w, canvas_h/2, fill="lightblue", dash=(2, 4))
            self.plot_canvas.create_text(canvas_w - 15, canvas_h/2, anchor="e", text="0", font=("Arial", 10), fill="blue")

        else:
            self.plot_canvas.create_text(10, 10, anchor="nw", text="Lines are out of bounds.")


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
        
    def adjust_probe_offset(self, amount):
        current_val = self.probe_offset_var.get()
        self.probe_offset_var.set(round(current_val + amount, 1))
        self.update_image()

    def update_image(self, *args):
        # Get user-friendly parameters from the GUI
        angle_deg = self.angle_var.get()
        cx = self.cx_var.get()
        cy = self.cy_var.get()
        probe_offset = self.probe_offset_var.get()

        # --- Convert (cx, cy, angle) to (rho, theta) ---
        # Theta is the angle of the normal, so it's 90 degrees offset from the line's angle.
        # We constrain it to [0, 180) as required by OpenCV's line representation.
        theta_deg = (angle_deg + 90) % 180
        theta_rad = np.deg2rad(theta_deg)
        
        # Rho is the distance from the origin to the line, calculated by projecting
        # the center point onto the normal vector.
        rho = cx * np.cos(theta_rad) + cy * np.sin(theta_rad)

        frame_copy = self.original_frame.copy()
        
        # Draw main line
        frame_with_line = draw_hough_line(frame_copy, rho, theta_deg, color=(0, 0, 255), thickness=2)
        
        # Draw probe line
        probe_rho = rho + probe_offset
        frame_with_line = draw_hough_line(frame_with_line, probe_rho, theta_deg, color=(0, 255, 0), thickness=2)

        # Draw a green dot at the center point
        cv2.circle(frame_with_line, (int(cx), int(cy)), 5, (0, 255, 0), -1)

        # Get metrics for both lines using the new corresponding sample method
        main_pixel_values, probe_pixel_values = get_parallel_line_pixel_values(
            self.original_frame, cx, cy, angle_deg, probe_offset, self.args.num_samples
        )
        
        # Calculate the difference
        diff_pixel_values = probe_pixel_values - main_pixel_values # Green - Red

        # Calculate Std Dev only on the valid (non-NaN) pixels of the main line
        std_dev = np.nanstd(main_pixel_values) if not np.all(np.isnan(main_pixel_values)) else None


        # Display text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Angle: {angle_deg:.1f}, Center: ({cx:.0f}, {cy:.0f})"
        cv2.putText(frame_with_line, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        if std_dev is not None:
            std_dev_text = f"Std Dev: {std_dev:.2f}"
            cv2.putText(frame_with_line, std_dev_text, (10, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # --- Update the pixel plot ---
        self.draw_pixel_plot(main_pixel_values, probe_pixel_values, diff_pixel_values)

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
        self.probe_offset_label.config(text=f"{probe_offset:.1f}")

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
    app = App(root, args, frame)
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
    parser.add_argument("--probe-offset", type=float, default=20.0, help="Distance between the main line and the probe line.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
