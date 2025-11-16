import cv2
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np


class DotViewer:
    DIRECTIONS = [
        ("right", 1.0, 0.0),
        ("left", -1.0, 0.0),
        ("up", 0.0, -1.0),
        ("down", 0.0, 1.0),
        ("up_right", np.sqrt(0.5), -np.sqrt(0.5)),
        ("up_left", -np.sqrt(0.5), -np.sqrt(0.5)),
        ("down_right", np.sqrt(0.5), np.sqrt(0.5)),
        ("down_left", -np.sqrt(0.5), np.sqrt(0.5)),
    ]
    def __init__(self, root, video_path, initial_frame_idx, total_frames):
        self.root = root
        self.video_path = video_path
        self.total_frames = total_frames
        self.original_frame = load_frame(video_path, initial_frame_idx)
        self.h, self.w = self.original_frame.shape[:2]

        self.root.title("Dot Viewer")

        self.x_var = tk.DoubleVar(value=self.w / 2)
        self.y_var = tk.DoubleVar(value=self.h / 2)
        self.frame_var = tk.IntVar(value=initial_frame_idx)

        main = ttk.Frame(root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        control = ttk.Frame(main, padding=5)
        control.grid(row=1, column=0, sticky="ew")

        ttk.Label(control, text="Frame:").grid(row=0, column=0, sticky=tk.W)
        self.frame_slider = tk.Scale(
            control, from_=0, to=self.total_frames - 1, orient=tk.HORIZONTAL,
            variable=self.frame_var, command=self.on_frame_change,
            showvalue=0, resolution=1
        )
        self.frame_slider.grid(row=0, column=1, sticky="ew")
        self.frame_label = ttk.Label(control, text=f"{initial_frame_idx}")
        self.frame_label.grid(row=0, column=2, padx=5)

        ttk.Label(control, text="X:").grid(row=1, column=0, sticky=tk.W)
        self.x_slider = tk.Scale(
            control, from_=0, to=self.w - 1, orient=tk.HORIZONTAL,
            variable=self.x_var, command=lambda *_: self.on_position_slider_change(), showvalue=0, resolution=1
        )
        self.x_slider.grid(row=1, column=1, sticky="ew")
        self.x_label = ttk.Label(control, text=f"{self.x_var.get():.0f}")
        self.x_label.grid(row=1, column=2, padx=5)

        ttk.Label(control, text="Y:").grid(row=2, column=0, sticky=tk.W)
        self.y_slider = tk.Scale(
            control, from_=0, to=self.h - 1, orient=tk.HORIZONTAL,
            variable=self.y_var, command=lambda *_: self.on_position_slider_change(), showvalue=0, resolution=1
        )
        self.y_slider.grid(row=2, column=1, sticky="ew")
        self.y_label = ttk.Label(control, text=f"{self.y_var.get():.0f}")
        self.y_label.grid(row=2, column=2, padx=5)

        ttk.Label(control, text="History:").grid(row=3, column=0, sticky=tk.W)
        self.history_var = tk.IntVar(value=30)
        self.history_slider = tk.Scale(
            control, from_=5, to=30, orient=tk.HORIZONTAL,
            variable=self.history_var, command=lambda *_: self.on_param_change(), showvalue=0, resolution=1
        )
        self.history_slider.grid(row=3, column=1, sticky="ew")
        self.history_label = ttk.Label(control, text=f"{self.history_var.get()}")
        self.history_label.grid(row=3, column=2, padx=5)

        ttk.Label(control, text="Z Threshold:").grid(row=4, column=0, sticky=tk.W)
        self.threshold_var = tk.DoubleVar(value=5.0)
        self.threshold_slider = tk.Scale(
            control, from_=2.0, to=20.0, orient=tk.HORIZONTAL,
            variable=self.threshold_var, command=lambda *_: self.on_param_change(), showvalue=0, resolution=0.5
        )
        self.threshold_slider.grid(row=4, column=1, sticky="ew")
        self.threshold_label = ttk.Label(control, text=f"{self.threshold_var.get():.1f}")
        self.threshold_label.grid(row=4, column=2, padx=5)

        ttk.Label(control, text="Grid Count:").grid(row=5, column=0, sticky=tk.W)
        self.grid_count_var = tk.IntVar(value=5)
        self.grid_count_slider = tk.Scale(
            control, from_=1, to=9, orient=tk.HORIZONTAL,
            variable=self.grid_count_var, command=lambda *_: self.on_param_change(),
            showvalue=0, resolution=2
        )
        self.grid_count_slider.grid(row=5, column=1, sticky="ew")
        self.grid_count_label = ttk.Label(control, text=f"{self.grid_count_var.get()}")
        self.grid_count_label.grid(row=5, column=2, padx=5)

        ttk.Label(control, text="Grid Extent:").grid(row=6, column=0, sticky=tk.W)
        self.grid_extent_var = tk.DoubleVar(value=40.0)
        self.grid_extent_slider = tk.Scale(
            control, from_=5, to=200, orient=tk.HORIZONTAL,
            variable=self.grid_extent_var, command=lambda *_: self.on_param_change(),
            showvalue=0, resolution=5
        )
        self.grid_extent_slider.grid(row=6, column=1, sticky="ew")
        self.grid_extent_label = ttk.Label(control, text=f"{self.grid_extent_var.get():.0f}")
        self.grid_extent_label.grid(row=6, column=2, padx=5)

        self.search_button = ttk.Button(control, text="Start", command=self.toggle_scan)
        self.search_button.grid(row=7, column=0, columnspan=3, pady=5, sticky="ew")

        control.columnconfigure(1, weight=1)

        self.image_label = tk.Canvas(main, highlightthickness=0, bg="black")
        self.image_label.grid(row=0, column=0, sticky="nsew")
        self.image_label.bind("<Button-1>", self.on_canvas_click)
        self.image_label.bind("<B1-Motion>", self.on_canvas_drag)

        main.rowconfigure(0, weight=1)

        self.display_w = min(1200, self.w)
        self.display_h = int(self.display_w * (self.h / self.w))

        self.scanning = False
        self.scan_origins = None
        self.pending_scan_job = None
        self.update_image()
        self.schedule_scan_restart()

    def update_image(self, *args):
        frame_copy = self.original_frame.copy()
        if self.scan_origins:
            color = (0, 255, 0)
            for entry in self.scan_origins:
                base_x, base_y = entry["origin"]
                for state in entry["states"].values():
                    length = state["length"]
                    if length <= 0:
                        continue
                    end_x = int(np.clip(base_x + state["dx"] * length, 0, self.w - 1))
                    end_y = int(np.clip(base_y + state["dy"] * length, 0, self.h - 1))
                    cv2.line(frame_copy, (base_x, base_y), (end_x, end_y), color, 2)
                    cv2.circle(frame_copy, (end_x, end_y), 4, color, -1)

        cursor_x = int(self.x_var.get())
        cursor_y = int(self.y_var.get())

        grid_points = self.compute_grid_points()
        for px, py in grid_points:
            cv2.circle(frame_copy, (px, py), 2, (0, 0, 255), -1)

        cv2.circle(frame_copy, (cursor_x, cursor_y), 6, (0, 0, 255), -1)

        display = cv2.resize(frame_copy, (self.display_w, self.display_h), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img))
        self.image_label.delete("all")
        self.display_image = img_tk
        self.image_label.create_image(0, 0, anchor="nw", image=img_tk, tags="frame")

        self.x_label.config(text=f"{self.x_var.get():.0f}")
        self.y_label.config(text=f"{self.y_var.get():.0f}")
        self.frame_label.config(text=f"{self.frame_var.get()}")
        self.history_label.config(text=f"{self.history_var.get()}")
        self.threshold_label.config(text=f"{self.threshold_var.get():.1f}")
        self.grid_count_label.config(text=f"{self.grid_count_var.get()}")
        self.grid_extent_label.config(text=f"{self.grid_extent_var.get():.0f}")

    def compute_grid_points(self):
        center_x = int(self.x_var.get())
        center_y = int(self.y_var.get())
        count = max(1, self.grid_count_var.get())
        if count % 2 == 0:
            count += 1
        radius = count // 2
        points = []
        if radius == 0:
            points.append((center_x, center_y))
            return points
        extent = max(1.0, self.grid_extent_var.get())
        spacing = extent / radius
        for gx in range(-radius, radius + 1):
            for gy in range(-radius, radius + 1):
                px = int(np.clip(center_x + gx * spacing, 0, self.w - 1))
                py = int(np.clip(center_y + gy * spacing, 0, self.h - 1))
                points.append((px, py))
        return points

    def on_frame_change(self, value):
        frame_idx = int(float(value))
        if self.scanning or self.scan_origins:
            self.finish_scan(clear_line=True)
        frame = load_frame(self.video_path, frame_idx)
        self.original_frame = frame
        self.update_image()
        self.schedule_scan_restart()

    def on_canvas_click(self, event):
        self.move_dot_to_canvas(event.x, event.y)

    def on_canvas_drag(self, event):
        self.move_dot_to_canvas(event.x, event.y)

    def move_dot_to_canvas(self, canvas_x, canvas_y):
        if self.scanning or self.scan_origins:
            self.finish_scan(clear_line=True)
        scale_x = self.w / self.display_w
        scale_y = self.h / self.display_h
        new_x = np.clip(canvas_x * scale_x, 0, self.w - 1)
        new_y = np.clip(canvas_y * scale_y, 0, self.h - 1)
        self.x_var.set(round(new_x, 1))
        self.y_var.set(round(new_y, 1))
        self.update_image()
        self.schedule_scan_restart()

    def on_position_slider_change(self):
        if self.scanning or self.scan_origins:
            self.finish_scan(clear_line=True)
        self.update_image()
        self.schedule_scan_restart()

    def on_param_change(self):
        if self.scanning or self.scan_origins:
            self.finish_scan(clear_line=True)
        self.update_image()
        self.schedule_scan_restart()

    def schedule_scan_restart(self):
        if self.pending_scan_job:
            self.root.after_cancel(self.pending_scan_job)
        self.pending_scan_job = self.root.after(100, self._delayed_start)

    def _delayed_start(self):
        self.pending_scan_job = None
        self.start_line_scan()

    def toggle_scan(self):
        if self.scanning:
            self.finish_scan()
        else:
            self.start_line_scan()

    def start_line_scan(self):
        if self.pending_scan_job:
            self.root.after_cancel(self.pending_scan_job)
            self.pending_scan_job = None
        if self.scanning:
            self.finish_scan(clear_line=True)
        self.scanning = True
        grid_points = self.compute_grid_points()
        self.scan_origins = []
        for origin in grid_points:
            states = {
                name: {"dx": dx, "dy": dy, "length": 0.0, "active": True}
                for name, dx, dy in self.DIRECTIONS
            }
            self.scan_origins.append({"origin": origin, "states": states})
        self.search_button.config(text="Stop")
        self.schedule_scan_step()

    def schedule_scan_step(self):
        if not self.scanning or not self.scan_origins:
            return
        any_active = False
        for entry in self.scan_origins:
            base_x, base_y = entry["origin"]
            for name, state in entry["states"].items():
                if not state["active"]:
                    continue
                next_len = state["length"] + 1
                end_x = base_x + state["dx"] * next_len
                end_y = base_y + state["dy"] * next_len
                if not (0 <= end_x < self.w and 0 <= end_y < self.h):
                    state["active"] = False
                    continue
                state["length"] = next_len
                any_active = True
                drop = self.log_scan_brightness(base_x, base_y, name, state)
                if drop:
                    state["active"] = False

        self.update_image()
        if not any_active:
            self.finish_scan()
        else:
            self.root.after(10, self.schedule_scan_step)

    def finish_scan(self, clear_line=False):
        self.scanning = False
        if clear_line:
            self.scan_origins = None
        else:
            if self.scan_origins:
                for entry in self.scan_origins:
                    for state in entry["states"].values():
                        state["active"] = False
        self.search_button.config(text="Start")
        self.update_image()

    def log_scan_brightness(self, base_x, base_y, direction, state):
        dx, dy = state["dx"], state["dy"]
        length = state["length"]
        history = self.history_var.get()
        brightness_list = []
        start_step = max(1, int(length) - history + 1)
        for step in range(start_step, int(length) + 1):
            px = int(np.clip(base_x + dx * step, 0, self.w - 1))
            py = int(np.clip(base_y + dy * step, 0, self.h - 1))
            b, g, r = self.original_frame[py, px]
            brightness = int(0.299 * r + 0.587 * g + 0.114 * b)
            brightness_list.append(brightness)

        prev_values = brightness_list[:-1]
        current_value = brightness_list[-1]
        if len(prev_values) >= max(5, history // 2):
            mean_prev = np.mean(prev_values)
            std_prev = np.std(prev_values)
            z_score = (mean_prev - current_value) / std_prev if std_prev > 0 else 0
            threshold = self.threshold_var.get()
            if std_prev > 0 and z_score > threshold:
                end_x = int(np.clip(base_x + dx * length, 0, self.w - 1))
                end_y = int(np.clip(base_y + dy * length, 0, self.h - 1))
                print(f"[stop][{direction}] x={end_x} y={end_y} z={z_score:.2f}")
                return True
        return False


def load_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame {frame_idx}")
    return frame


def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def parse_args():
    parser = argparse.ArgumentParser(description="Simple dot viewer on a video frame.")
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to video file.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to display.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    root.state("zoomed")
    total_frames = get_total_frames(args.video)
    viewer = DotViewer(root, args.video, args.frame, total_frames)
    root.mainloop()


if __name__ == "__main__":
    main()

