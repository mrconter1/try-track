import argparse
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively project a planar grid using four draggable points."
    )
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to the video file.")
    parser.add_argument("--frame", type=int, default=4113, help="Frame number to load.")
    parser.add_argument(
        "--points",
        type=float,
        nargs=8,
        metavar=("x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"),
        default=[200, 200, 400, 200, 400, 400, 200, 400],
        help="Image coordinates (top-left, top-right, bottom-right, bottom-left).",
    )
    parser.add_argument(
        "--tile-radius",
        type=float,
        default=20.0,
        help="How many tiles to extend outward from the reference tile (each direction).",
    )
    parser.add_argument("--grid-color", type=str, default="0,255,0", help="Grid line color as B,G,R.")
    parser.add_argument("--thickness", type=int, default=1, help="Grid line thickness.")
    return parser.parse_args()


def parse_color(color_str: str):
    parts = color_str.split(",")
    if len(parts) != 3:
        raise ValueError("Color must be B,G,R with three components.")
    return tuple(int(max(0, min(255, float(p)))) for p in parts)


def load_frame(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx >= total_frames:
        cap.release()
        raise ValueError(f"Frame {frame_idx} out of bounds. Video has {total_frames} frames.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx}")
    return frame


def build_homography(points):
    src = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dst = np.array(points, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(src, dst)


def draw_reference_points(frame, points, colors):
    pts = np.array(points, dtype=np.int32).reshape(4, 2)
    if len(pts) == 4:
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        for (x, y), color in zip(pts, colors):
            cv2.circle(frame, (x, y), 7, color, -1, cv2.LINE_AA)


def draw_infinite_grid(frame, homography, tile_radius, color, thickness):
    step = 1.0
    extent = tile_radius + 1.0

    def project_line(pt_a, pt_b):
        pts = np.array([pt_a, pt_b], dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
        return projected.astype(int)

    x_values = np.arange(-tile_radius, tile_radius + step, step)
    y_min = -extent
    y_max = extent
    for x in x_values:
        p1 = [x, y_min]
        p2 = [x, y_max]
        img_pts = project_line(p1, p2)
        cv2.line(frame, tuple(img_pts[0]), tuple(img_pts[1]), color, thickness, cv2.LINE_AA)

    y_values = np.arange(-tile_radius, tile_radius + step, step)
    x_min = -extent
    x_max = extent
    for y in y_values:
        p1 = [x_min, y]
        p2 = [x_max, y]
        img_pts = project_line(p1, p2)
        cv2.line(frame, tuple(img_pts[0]), tuple(img_pts[1]), color, thickness, cv2.LINE_AA)


def bgr_to_hex(color):
    b, g, r = [max(0, min(255, int(c))) for c in color]
    return f"#{r:02x}{g:02x}{b:02x}"


class App:
    POINT_NAMES = ["Top Left", "Top Right", "Bottom Right", "Bottom Left"]
    POINT_COLORS = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]

    def __init__(self, root, args, initial_frame, video_path, total_frames):
        self.root = root
        self.args = args
        self.original_frame = initial_frame
        self.video_path = video_path
        self.total_frames = total_frames
        self.current_frame_num = args.frame
        self.grid_color = parse_color(args.grid_color)
        self.tile_radius = args.tile_radius
        self.thickness = args.thickness

        self.root.title("Four-Point Grid Viewer")

        h, w = self.original_frame.shape[:2]

        self.frame_num_var = tk.IntVar(value=self.current_frame_num)
        self.point_vars = []
        for i in range(4):
            x_val = float(args.points[2 * i])
            y_val = float(args.points[2 * i + 1])
            x_var = tk.DoubleVar(value=x_val)
            y_var = tk.DoubleVar(value=y_val)
            self.point_vars.append((x_var, y_var))

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.grid(row=1, column=0, sticky="ew")

        ttk.Label(control_frame, text="Frame:").grid(row=0, column=0, sticky=tk.W, pady=3)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_frame(-10)).grid(row=0, column=1)
        self.frame_slider = tk.Scale(
            control_frame,
            from_=0,
            to=self.total_frames - 1,
            orient=tk.HORIZONTAL,
            variable=self.frame_num_var,
            command=self.load_frame,
            resolution=1,
            showvalue=0,
        )
        self.frame_slider.grid(row=0, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_frame(10)).grid(row=0, column=3)
        self.frame_label = ttk.Label(control_frame, text=f"{self.frame_num_var.get()}", width=7)
        self.frame_label.grid(row=0, column=4, padx=5)

        control_row = 1
        self.point_value_labels = []
        for idx, (name, color) in enumerate(zip(self.POINT_NAMES, self.POINT_COLORS)):
            hex_color = bgr_to_hex(color)
            x_var, y_var = self.point_vars[idx]

            ttk.Label(control_frame, text=f"{name} X:", foreground=hex_color).grid(
                row=control_row, column=0, sticky=tk.W, pady=2
            )
            x_slider = tk.Scale(
                control_frame,
                from_=0,
                to=w,
                orient=tk.HORIZONTAL,
                variable=x_var,
                command=lambda val: self.update_image(),
                resolution=1,
                showvalue=0,
            )
            x_slider.grid(row=control_row, column=2, sticky="ew")
            ttk.Button(
                control_frame, text="-", width=3, command=lambda idx=idx: self.nudge_point(idx, dx=-1, dy=0)
            ).grid(row=control_row, column=1)
            ttk.Button(
                control_frame, text="+", width=3, command=lambda idx=idx: self.nudge_point(idx, dx=1, dy=0)
            ).grid(row=control_row, column=3)
            x_value_label = ttk.Label(control_frame, text=f"{x_var.get():.0f}", width=7, foreground=hex_color)
            x_value_label.grid(row=control_row, column=4, padx=5)

            control_row += 1
            ttk.Label(control_frame, text=f"{name} Y:", foreground=hex_color).grid(
                row=control_row, column=0, sticky=tk.W, pady=2
            )
            y_slider = tk.Scale(
                control_frame,
                from_=0,
                to=h,
                orient=tk.HORIZONTAL,
                variable=y_var,
                command=lambda val: self.update_image(),
                resolution=1,
                showvalue=0,
            )
            y_slider.grid(row=control_row, column=2, sticky="ew")
            ttk.Button(
                control_frame, text="-", width=3, command=lambda idx=idx: self.nudge_point(idx, dx=0, dy=-1)
            ).grid(row=control_row, column=1)
            ttk.Button(
                control_frame, text="+", width=3, command=lambda idx=idx: self.nudge_point(idx, dx=0, dy=1)
            ).grid(row=control_row, column=3)
            y_value_label = ttk.Label(control_frame, text=f"{y_var.get():.0f}", width=7, foreground=hex_color)
            y_value_label.grid(row=control_row, column=4, padx=5)

            self.point_value_labels.append((x_value_label, y_value_label))
            control_row += 1

        control_frame.columnconfigure(2, weight=1)

        self.root.update_idletasks()
        control_height = control_frame.winfo_reqheight()
        max_width = root.winfo_screenwidth() - 40
        max_height = root.winfo_screenheight() - control_height - 150

        ratio = min(max_width / w, max_height / h)
        self.display_w = int(w * ratio)
        self.display_h = int(h * ratio)

        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew")
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        self.error_var = tk.StringVar(value="")
        self.error_label = ttk.Label(main_frame, textvariable=self.error_var, foreground="red")
        self.error_label.grid(row=2, column=0, pady=5)

        self.update_image()

    def nudge_point(self, idx, dx=0, dy=0):
        x_var, y_var = self.point_vars[idx]
        x_var.set(x_var.get() + dx)
        y_var.set(y_var.get() + dy)
        self.update_image()

    def adjust_frame(self, amount):
        current = self.frame_num_var.get()
        new_frame = max(0, min(self.total_frames - 1, current + amount))
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

    def get_points(self):
        points = []
        for x_var, y_var in self.point_vars:
            points.extend([x_var.get(), y_var.get()])
        return points

    def update_image(self, *args):
        frame_copy = self.original_frame.copy()
        points = self.get_points()

        try:
            H = build_homography(points)
            draw_infinite_grid(
                frame_copy,
                homography=H,
                tile_radius=self.tile_radius,
                color=self.grid_color,
                thickness=self.thickness,
            )
            self.error_var.set("")
        except cv2.error as err:
            self.error_var.set(f"Homography error: {err}")

        draw_reference_points(frame_copy, points, self.POINT_COLORS)

        display_frame = cv2.resize(frame_copy, (self.display_w, self.display_h), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.image_label.imgtk = img_tk
        self.image_label.configure(image=img_tk)
        self.frame_label.config(text=f"{self.current_frame_num}")

        for (x_label, y_label), (x_var, y_var) in zip(self.point_value_labels, self.point_vars):
            x_label.config(text=f"{x_var.get():.0f}")
            y_label.config(text=f"{y_var.get():.0f}")


def main():
    args = parse_args()
    frame = load_frame(args.video, args.frame)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    root = tk.Tk()
    root.state("zoomed")
    App(root, args, frame, args.video, total_frames)
    root.mainloop()


if __name__ == "__main__":
    main()

