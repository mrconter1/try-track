import cv2
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

def draw_grid(
    frame,
    grid_size=50,
    grid_orientation=0.0,
    camera_height=200.0,
    pitch_deg=0.0,
    roll_deg=0.0,
    grid_offset_x=0.0,
    grid_offset_y=0.0,
):
    """Draws a perspective grid on the frame using simple camera-plane parameters."""
    if grid_size <= 0:
        return frame

    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    focal_length = max(w, h)
    fx = fy = focal_length

    orientation_rad = np.deg2rad(grid_orientation)
    cos_o = np.cos(orientation_rad)
    sin_o = np.sin(orientation_rad)

    pitch_rad = np.deg2rad(pitch_deg)
    roll_rad = np.deg2rad(roll_deg)

    R_base = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )

    R_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0.0, np.sin(pitch_rad), np.cos(pitch_rad)],
        ],
        dtype=np.float64,
    )

    R_roll = np.array(
        [
            [np.cos(roll_rad), -np.sin(roll_rad), 0.0],
            [np.sin(roll_rad), np.cos(roll_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    rotation = R_roll @ R_pitch @ R_base
    camera_center = np.array([0.0, camera_height, 0.0], dtype=np.float64)

    grid_half_width = grid_size * 20
    depth_near = max(grid_size * 0.5, 1.0)
    depth_far = grid_size * 40
    num_width_lines = 20
    num_depth_lines = 40

    def plane_to_world(u, v):
        x_local = u * cos_o - v * sin_o
        z_local = u * sin_o + v * cos_o
        x = x_local + grid_offset_x
        z = z_local + grid_offset_y
        return np.array([x, 0.0, z], dtype=np.float64)

    def project_point(point_world):
        vec = point_world - camera_center
        point_cam = rotation @ vec
        if point_cam[2] <= 1e-3:
            return None
        u = fx * (point_cam[0] / point_cam[2]) + cx
        v = fy * (point_cam[1] / point_cam[2]) + cy
        if not np.isfinite(u) or not np.isfinite(v):
            return None
        return int(round(u)), int(round(v))

    color = (0, 255, 0)
    thickness = 1

    for i in range(-num_width_lines, num_width_lines + 1):
        u = i * grid_size
        p_near = plane_to_world(u, depth_near)
        p_far = plane_to_world(u, depth_far)
        pt1 = project_point(p_near)
        pt2 = project_point(p_far)
        if pt1 and pt2:
            cv2.line(frame, pt1, pt2, color, thickness)

    for j in range(num_depth_lines + 1):
        v = depth_near + j * grid_size
        p_left = plane_to_world(-grid_half_width, v)
        p_right = plane_to_world(grid_half_width, v)
        pt1 = project_point(p_left)
        pt2 = project_point(p_right)
        if pt1 and pt2:
            cv2.line(frame, pt1, pt2, color, thickness)

    return frame

class App:
    def __init__(self, root, args, initial_frame, video_path, total_frames):
        self.root = root
        self.args = args
        self.original_frame = initial_frame
        self.video_path = video_path
        self.total_frames = total_frames
        self.current_frame_num = args.frame
        
        self.root.title("Grid Overlay Control")
        
        # --- Variables ---
        self.frame_num_var = tk.IntVar(value=self.current_frame_num)
        self.grid_size_var = tk.IntVar(value=args.grid_size)
        self.camera_height_var = tk.DoubleVar(value=args.camera_height)
        self.pitch_var = tk.DoubleVar(value=args.pitch)
        self.roll_var = tk.DoubleVar(value=args.roll)
        self.grid_orientation_var = tk.DoubleVar(value=args.grid_orientation)
        self.grid_offset_x_var = tk.DoubleVar(value=args.grid_offset_x)
        self.grid_offset_y_var = tk.DoubleVar(value=args.grid_offset_y)
        
        # --- GUI Layout ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.grid(row=1, column=0, sticky="ew")

        # Frame Number Controls
        ttk.Label(control_frame, text="Frame:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_frame(-10)).grid(row=0, column=1)
        self.frame_slider = tk.Scale(control_frame, from_=0, to=self.total_frames-1, orient=tk.HORIZONTAL, variable=self.frame_num_var, command=self.load_frame, resolution=1, showvalue=0)
        self.frame_slider.grid(row=0, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_frame(10)).grid(row=0, column=3)
        self.frame_label = ttk.Label(control_frame, text=f"{self.frame_num_var.get()}", width=7)
        self.frame_label.grid(row=0, column=4, padx=5)

        # Grid Size Controls
        ttk.Label(control_frame, text="Grid Size:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_grid_size(-1)).grid(row=1, column=1)
        self.grid_size_slider = tk.Scale(control_frame, from_=5, to=200, orient=tk.HORIZONTAL, variable=self.grid_size_var, command=self.update_image, resolution=1, showvalue=0)
        self.grid_size_slider.grid(row=1, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_grid_size(1)).grid(row=1, column=3)
        self.grid_size_label = ttk.Label(control_frame, text=f"{self.grid_size_var.get()}", width=7)
        self.grid_size_label.grid(row=1, column=4, padx=5)

        # Camera Height Controls
        ttk.Label(control_frame, text="Camera Height:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_camera_height(-5)).grid(row=2, column=1)
        self.camera_height_slider = tk.Scale(
            control_frame,
            from_=20,
            to=1000,
            orient=tk.HORIZONTAL,
            variable=self.camera_height_var,
            command=self.update_image,
            resolution=5,
            showvalue=0,
        )
        self.camera_height_slider.grid(row=2, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_camera_height(5)).grid(row=2, column=3)
        self.camera_height_label = ttk.Label(control_frame, text=f"{self.camera_height_var.get():.0f}", width=7)
        self.camera_height_label.grid(row=2, column=4, padx=5)

        # Pitch Controls
        ttk.Label(control_frame, text="Pitch (deg):").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_pitch(-1)).grid(row=3, column=1)
        self.pitch_slider = tk.Scale(
            control_frame,
            from_=-60,
            to=60,
            orient=tk.HORIZONTAL,
            variable=self.pitch_var,
            command=self.update_image,
            resolution=0.1,
            showvalue=0,
        )
        self.pitch_slider.grid(row=3, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_pitch(1)).grid(row=3, column=3)
        self.pitch_label = ttk.Label(control_frame, text=f"{self.pitch_var.get():.1f}", width=7)
        self.pitch_label.grid(row=3, column=4, padx=5)

        # Roll Controls
        ttk.Label(control_frame, text="Roll (deg):").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_roll(-1)).grid(row=4, column=1)
        self.roll_slider = tk.Scale(
            control_frame,
            from_=-45,
            to=45,
            orient=tk.HORIZONTAL,
            variable=self.roll_var,
            command=self.update_image,
            resolution=0.1,
            showvalue=0,
        )
        self.roll_slider.grid(row=4, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_roll(1)).grid(row=4, column=3)
        self.roll_label = ttk.Label(control_frame, text=f"{self.roll_var.get():.1f}", width=7)
        self.roll_label.grid(row=4, column=4, padx=5)

        # Grid Orientation Controls
        ttk.Label(control_frame, text="Grid Orientation:").grid(row=5, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_grid_orientation(-1)).grid(row=5, column=1)
        self.grid_orientation_slider = tk.Scale(
            control_frame,
            from_=-180,
            to=180,
            orient=tk.HORIZONTAL,
            variable=self.grid_orientation_var,
            command=self.update_image,
            resolution=0.5,
            showvalue=0,
        )
        self.grid_orientation_slider.grid(row=5, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_grid_orientation(1)).grid(row=5, column=3)
        self.grid_orientation_label = ttk.Label(control_frame, text=f"{self.grid_orientation_var.get():.1f}", width=7)
        self.grid_orientation_label.grid(row=5, column=4, padx=5)

        # Grid Offset X Controls
        ttk.Label(control_frame, text="Grid Offset X:").grid(row=6, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_grid_offset_x(-10)).grid(row=6, column=1)
        self.grid_offset_x_slider = tk.Scale(
            control_frame,
            from_=-2000,
            to=2000,
            orient=tk.HORIZONTAL,
            variable=self.grid_offset_x_var,
            command=self.update_image,
            resolution=5,
            showvalue=0,
        )
        self.grid_offset_x_slider.grid(row=6, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_grid_offset_x(10)).grid(row=6, column=3)
        self.grid_offset_x_label = ttk.Label(control_frame, text=f"{self.grid_offset_x_var.get():.0f}", width=7)
        self.grid_offset_x_label.grid(row=6, column=4, padx=5)

        # Grid Offset Y Controls
        ttk.Label(control_frame, text="Grid Offset Y:").grid(row=7, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_grid_offset_y(-10)).grid(row=7, column=1)
        self.grid_offset_y_slider = tk.Scale(
            control_frame,
            from_=-2000,
            to=2000,
            orient=tk.HORIZONTAL,
            variable=self.grid_offset_y_var,
            command=self.update_image,
            resolution=5,
            showvalue=0,
        )
        self.grid_offset_y_slider.grid(row=7, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_grid_offset_y(10)).grid(row=7, column=3)
        self.grid_offset_y_label = ttk.Label(control_frame, text=f"{self.grid_offset_y_var.get():.0f}", width=7)
        self.grid_offset_y_label.grid(row=7, column=4, padx=5)

        control_frame.columnconfigure(2, weight=1)

        self.root.update_idletasks()
        control_height = control_frame.winfo_reqheight()

        max_width = root.winfo_screenwidth() - 40
        max_height = root.winfo_screenheight() - control_height - 150
        
        h_orig, w_orig = self.original_frame.shape[:2]
        ratio = min(max_width / w_orig, max_height / h_orig)
        
        self.display_w = int(w_orig * ratio)
        self.display_h = int(h_orig * ratio)

        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew")
        
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        self.update_image()

    def adjust_grid_size(self, amount):
        current_val = self.grid_size_var.get()
        self.grid_size_var.set(max(5, current_val + amount))
        self.update_image()

    def adjust_camera_height(self, amount):
        current_val = self.camera_height_var.get()
        new_val = max(20.0, min(2000.0, current_val + amount))
        self.camera_height_var.set(new_val)
        self.update_image()

    def adjust_pitch(self, amount):
        current_val = self.pitch_var.get()
        new_val = max(-90.0, min(90.0, current_val + amount))
        self.pitch_var.set(round(new_val, 1))
        self.update_image()

    def adjust_roll(self, amount):
        current_val = self.roll_var.get()
        new_val = max(-90.0, min(90.0, current_val + amount))
        self.roll_var.set(round(new_val, 1))
        self.update_image()

    def adjust_grid_orientation(self, amount):
        current_val = self.grid_orientation_var.get()
        new_val = current_val + amount
        self.grid_orientation_var.set(round(new_val, 1))
        self.update_image()

    def adjust_grid_offset_x(self, amount):
        current_val = self.grid_offset_x_var.get()
        new_val = current_val + amount
        self.grid_offset_x_var.set(new_val)
        self.update_image()

    def adjust_grid_offset_y(self, amount):
        current_val = self.grid_offset_y_var.get()
        new_val = current_val + amount
        self.grid_offset_y_var.set(new_val)
        self.update_image()

    def adjust_frame(self, amount):
        current_frame = self.frame_num_var.get()
        new_frame = max(0, min(self.total_frames - 1, current_frame + amount))
        self.frame_num_var.set(new_frame)
        self.load_frame(str(new_frame))

    def load_frame(self, frame_num_str):
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
        grid_size = self.grid_size_var.get()
        
        frame_with_grid = draw_grid(
            frame_copy,
            grid_size=grid_size,
            grid_orientation=self.grid_orientation_var.get(),
            camera_height=self.camera_height_var.get(),
            pitch_deg=self.pitch_var.get(),
            roll_deg=self.roll_var.get(),
            grid_offset_x=self.grid_offset_x_var.get(),
            grid_offset_y=self.grid_offset_y_var.get(),
        )
        
        display_frame = cv2.resize(frame_with_grid, (self.display_w, self.display_h), interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        self.image_label.imgtk = img_tk
        self.image_label.configure(image=img_tk)

        self.grid_size_label.config(text=f"{grid_size}")
        self.camera_height_label.config(text=f"{self.camera_height_var.get():.0f}")
        self.pitch_label.config(text=f"{self.pitch_var.get():.1f}")
        self.roll_label.config(text=f"{self.roll_var.get():.1f}")
        self.grid_orientation_label.config(text=f"{self.grid_orientation_var.get():.1f}")
        self.grid_offset_x_label.config(text=f"{self.grid_offset_x_var.get():.0f}")
        self.grid_offset_y_label.config(text=f"{self.grid_offset_y_var.get():.0f}")
        self.frame_label.config(text=f"{self.current_frame_num}")

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
    root.state('zoomed')
    app = App(root, args, frame, args.video, total_frames)
    root.mainloop()

def parse_args():
    parser = argparse.ArgumentParser(description="Display a video frame with an interactive grid overlay.")
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to the video file.")
    parser.add_argument("--frame", type=int, default=4113, help="Frame number to load.")
    parser.add_argument("--grid-size", type=int, default=50, help="Initial size of the grid cells in pixels.")
    parser.add_argument("--camera-height", type=float, default=200.0, help="Initial camera height above the floor plane.")
    parser.add_argument("--pitch", type=float, default=0.0, help="Initial pitch angle in degrees.")
    parser.add_argument("--roll", type=float, default=0.0, help="Initial roll angle in degrees.")
    parser.add_argument("--grid-orientation", type=float, default=0.0, help="Initial grid orientation on the floor plane.")
    parser.add_argument("--grid-offset-x", type=float, default=0.0, help="Initial lateral offset of the grid on the floor plane.")
    parser.add_argument("--grid-offset-y", type=float, default=0.0, help="Initial depth offset of the grid on the floor plane.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
