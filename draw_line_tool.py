import cv2
import numpy as np
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

# --- Core OpenCV Functions (Largely Unchanged) ---

def draw_hough_line(frame, rho, theta_deg):
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
    
    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
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

# --- New Tkinter GUI Application ---

class App:
    def __init__(self, root, args, initial_frame):
        self.root = root
        self.args = args
        self.original_frame = initial_frame
        
        self.root.title("Hough Line Control")
        
        # --- Set Display Size based on screen size for maximized window ---
        max_width = root.winfo_screenwidth() - 40   # Padding for window borders
        max_height = root.winfo_screenheight() - 180 # Padding for borders, taskbar, and controls
        
        h, w = self.original_frame.shape[:2]
        
        # Calculate the best fit for the screen
        ratio = min(max_width / w, max_height / h)
        
        self.display_w = int(w * ratio)
        self.display_h = int(h * ratio)

        # --- Variables ---
        self.theta_var = tk.DoubleVar(value=self.args.theta)
        self.rho_var = tk.DoubleVar(value=self.args.rho)
        
        # --- GUI Layout ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Image display
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew")
        
        # Pixel Plot Canvas
        self.plot_canvas = tk.Canvas(main_frame, bg="white", width=200)
        self.plot_canvas.grid(row=0, column=1, sticky="nsew")

        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3) # Image gets 3/4 of the space
        main_frame.columnconfigure(1, weight=1) # Plot gets 1/4 of the space

        # --- Controls Frame ---
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

        # Controls
        h, w = self.original_frame.shape[:2]
        self.max_rho = int(np.sqrt(h**2 + w**2))

        # Theta Controls
        ttk.Label(control_frame, text="Theta:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_theta(-0.01)).grid(row=0, column=1)
        self.theta_slider = tk.Scale(control_frame, from_=0, to=180, orient=tk.HORIZONTAL, variable=self.theta_var, command=self.update_image, resolution=0.01, showvalue=0)
        self.theta_slider.grid(row=0, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_theta(0.01)).grid(row=0, column=3)
        self.theta_label = ttk.Label(control_frame, text=f"{self.theta_var.get():.2f}", width=6)
        self.theta_label.grid(row=0, column=4, padx=5)

        # Rho Controls
        ttk.Label(control_frame, text="Rho:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_rho(-0.01)).grid(row=1, column=1)
        self.rho_slider = tk.Scale(control_frame, from_=-self.max_rho, to=self.max_rho, orient=tk.HORIZONTAL, variable=self.rho_var, command=self.update_image, resolution=0.01, showvalue=0)
        self.rho_slider.grid(row=1, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_rho(0.01)).grid(row=1, column=3)
        self.rho_label = ttk.Label(control_frame, text=f"{self.rho_var.get():.2f}", width=6)
        self.rho_label.grid(row=1, column=4, padx=5)
        
        # --- Search Button ---
        self.search_button = ttk.Button(control_frame, text="Find Darkest Line", command=self.start_darkest_line_search)
        self.search_button.grid(row=0, column=5, rowspan=2, padx=10)

        control_frame.columnconfigure(2, weight=1) # Make slider stretch

        self.update_image()
    
    def start_darkest_line_search(self):
        """Starts the grid search in a new thread to avoid freezing the GUI."""
        self.search_button.config(state="disabled", text="Searching...")
        
        # Run the actual search in a worker thread
        search_thread = threading.Thread(target=self._grid_search_worker)
        search_thread.daemon = True # Allows main program to exit even if thread is running
        search_thread.start()

    def _grid_search_worker(self):
        """The long-running grid search task."""
        current_rho = self.rho_var.get()
        current_theta = self.theta_var.get()

        best_rho = current_rho
        best_theta = current_theta
        min_pixel_sum = float('inf')

        # Define the search space
        rho_min = current_rho - self.args.search_range
        rho_max = current_rho + self.args.search_range
        theta_min = current_theta - self.args.search_range
        theta_max = current_theta + self.args.search_range

        print("Starting darkest line search with random sampling...")

        for i in range(self.args.num_search_samples):
            rho = np.random.uniform(rho_min, rho_max)
            theta = np.random.uniform(theta_min, theta_max)

            _, pixel_sum, _ = get_line_metrics(self.original_frame, rho, theta, self.args.num_samples)
            
            if pixel_sum is not None and pixel_sum < min_pixel_sum:
                min_pixel_sum = pixel_sum
                best_rho = rho
                best_theta = theta

            # Print progress to the console periodically
            if (i + 1) % 100 == 0 or (i + 1) == self.args.num_search_samples:
                progress = ((i + 1) / self.args.num_search_samples) * 100
                print(f"Search progress: {progress:.1f}%")
        
        # When done, schedule an update on the main GUI thread
        self.root.after(0, self.finish_darkest_line_search, best_rho, best_theta)

    def finish_darkest_line_search(self, best_rho, best_theta):
        """Updates the GUI with the results from the search."""
        self.rho_var.set(best_rho)
        self.theta_var.set(best_theta)
        
        self.search_button.config(state="normal", text="Find Darkest Line")
        
        self.update_image()


    def draw_pixel_plot(self, pixel_values):
        self.plot_canvas.delete("all") # Clear previous plot

        if pixel_values is None or len(pixel_values) == 0:
            self.plot_canvas.create_text(10, 10, anchor="nw", text="Line is out of bounds.")
            return

        canvas_w = self.plot_canvas.winfo_width()
        canvas_h = self.plot_canvas.winfo_height()

        if canvas_w < 2 or canvas_h < 2: # Canvas not ready on first draw
             self.root.after(50, lambda: self.draw_pixel_plot(pixel_values))
             return

        # Create points for the line graph
        points = []
        num_samples = len(pixel_values)
        x_step = canvas_w / max(1, num_samples - 1)

        for i, value in enumerate(pixel_values):
            x = i * x_step
            y = canvas_h - (value / 255.0) * canvas_h # Invert Y-axis for drawing
            points.extend([x, y])

        if len(points) > 2:
            self.plot_canvas.create_line(points, fill="blue", width=2)

        # Draw Y-axis labels for context
        self.plot_canvas.create_text(15, 10, anchor="nw", text="255", font=("Arial", 10))
        self.plot_canvas.create_line(0, 10, 10, 10)
        self.plot_canvas.create_text(15, canvas_h - 10, anchor="sw", text="0", font=("Arial", 10))
        self.plot_canvas.create_line(0, canvas_h-10, 10, canvas_h-10)


    def adjust_theta(self, amount):
        current_val = self.theta_var.get()
        self.theta_var.set(round(current_val + amount, 2))
        self.update_image()

    def adjust_rho(self, amount):
        current_val = self.rho_var.get()
        self.rho_var.set(round(current_val + amount, 2))
        self.update_image()
        
    def update_image(self, *args):
        theta_deg = self.theta_var.get()
        rho = self.rho_var.get()

        frame_copy = self.original_frame.copy()
        
        # Draw line
        frame_with_line = draw_hough_line(frame_copy, rho, theta_deg)
        
        # Calculate Std Dev
        std_dev, _, pixel_values = get_line_metrics(self.original_frame, rho, theta_deg, self.args.num_samples)

        # Display text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Rho: {rho:.1f}, Theta: {theta_deg:.1f}"
        cv2.putText(frame_with_line, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        if std_dev is not None:
            std_dev_text = f"Std Dev: {std_dev:.2f}"
            cv2.putText(frame_with_line, std_dev_text, (10, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # --- Update the pixel plot ---
        self.draw_pixel_plot(pixel_values)

        # --- Scale frame for display ---
        display_frame = cv2.resize(frame_with_line, (self.display_w, self.display_h), interpolation=cv2.INTER_AREA)

        # Convert for Tkinter
        img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        self.image_label.imgtk = img_tk
        self.image_label.configure(image=img_tk)

        # Update labels
        self.theta_label.config(text=f"{theta_deg:.2f}")
        self.rho_label.config(text=f"{rho:.2f}")

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
    app = App(root, args, frame)
    root.mainloop()

def parse_args():
    parser = argparse.ArgumentParser(description="Draw a single Hough line on a video frame interactively.")
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to the video file.")
    parser.add_argument("--frame", type=int, default=2000, help="Frame number to load.")
    parser.add_argument("--rho", type=float, default=100.0, help="The initial 'rho' parameter of the line.")
    parser.add_argument("--theta", type=float, default=45.0, help="The initial 'theta' parameter of the line (in degrees).")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples for color std deviation.")
    parser.add_argument("--search-range", type=float, default=5.0, help="Search range (+-) for rho and theta for darkest line search.")
    parser.add_argument("--num-search-samples", type=int, default=1000, help="Number of random samples for the darkest line search.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
