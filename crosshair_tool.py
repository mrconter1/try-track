import cv2
import numpy as np
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

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
        self.line_length_var = tk.IntVar(value=50)   # Line length for both
        self.clahe_enabled = tk.BooleanVar(value=True)
        self.clahe_clip_limit = tk.DoubleVar(value=2.0)
        self.clahe_tile_size = tk.IntVar(value=8)
        self.frame_num_var = tk.IntVar(value=self.current_frame_num)
        
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

        # CLAHE Enable/Disable
        ttk.Label(control_frame, text="CLAHE:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.clahe_checkbox = ttk.Checkbutton(control_frame, variable=self.clahe_enabled, command=self.update_image)
        self.clahe_checkbox.grid(row=6, column=1, sticky=tk.W)
        
        # CLAHE Clip Limit
        ttk.Label(control_frame, text="Clip:").grid(row=6, column=2, sticky=tk.W, padx=(10, 0))
        self.clahe_clip_slider = tk.Scale(control_frame, from_=1.0, to=10.0, orient=tk.HORIZONTAL, variable=self.clahe_clip_limit, command=self.update_image, resolution=0.5, showvalue=0)
        self.clahe_clip_slider.grid(row=6, column=3, sticky="ew")
        self.clahe_clip_label = ttk.Label(control_frame, text=f"{self.clahe_clip_limit.get():.1f}", width=5)
        self.clahe_clip_label.grid(row=6, column=4, padx=5)

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

        # --- Image Display ---
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew")

        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

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

        # Apply CLAHE
        frame_copy = self.apply_clahe(frame_copy)

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
        cv2.line(frame_copy, (v_x1, v_y1), (v_x2, v_y2), (0, 0, 255), 2)
        
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
        cv2.line(frame_copy, (h_x1, h_y1), (h_x2, h_y2), (255, 0, 0), 2)
        
        # Draw a red dot at center
        cv2.circle(frame_copy, (cx_int, cy_int), 5, (0, 0, 255), -1)
        
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
        self.frame_label.config(text=f"{self.frame_num_var.get()}")
        self.clahe_clip_label.config(text=f"{self.clahe_clip_limit.get():.1f}")
        
        # --- Scale frame for display ---
        display_frame = cv2.resize(frame_copy, (self.display_w, self.display_h), interpolation=cv2.INTER_AREA)

        # Convert for Tkinter
        img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        self.image_label.imgtk = img_tk
        self.image_label.configure(image=img_tk)

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

