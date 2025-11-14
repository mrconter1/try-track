from __future__ import annotations

import argparse
import threading
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

WINDOW_TITLE = "Square Grid (6-DOF)"


STEP_BUTTONS_NEG = [
    ("-1", -1.0),
    ("-0.1", -0.1),
    ("-0.01", -0.01),
    ("-0.001", -0.001),
]

STEP_BUTTONS_POS = [
    ("+0.001", 0.001),
    ("+0.01", 0.01),
    ("+0.1", 0.1),
    ("+1", 1.0),
]

TOTAL_CONTROL_COLUMNS = len(STEP_BUTTONS_NEG) + len(STEP_BUTTONS_POS) + 2  # slider + value label

SLIDERS = [
    {"name": "Grid X", "min": -20.0, "max": 20.0, "initial": 0.0, "unit": "tiles"},
    {"name": "Grid Y", "min": -20.0, "max": 20.0, "initial": 0.0, "unit": "tiles"},
    {"name": "Grid Rot", "min": -90.0, "max": 90.0, "initial": 0.0, "unit": "deg"},
    {"name": "Scale", "min": 0.10, "max": 5.0, "initial": 1.0, "unit": "x"},
    {"name": "Pitch", "min": -60.0, "max": 60.0, "initial": 0.0, "unit": "deg"},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively adjust a planar square grid using six intuitive sliders."
    )
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to the video file.")
    parser.add_argument("--frame", type=int, default=4113, help="Frame number to display.")
    parser.add_argument("--tile-radius", type=int, default=8, help="Grid radius in tiles from the origin.")
    parser.add_argument(
        "--samples-per-line",
        type=int,
        default=150,
        help="Number of points sampled per grid line for drawing.",
    )
    parser.add_argument("--color", type=str, default="0,255,0", help="Grid color as B,G,R integers.")
    parser.add_argument("--thickness", type=int, default=1, help="Line thickness.")
    return parser.parse_args()


def load_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0 or frame_idx >= total_frames:
        cap.release()
        raise ValueError(f"Frame {frame_idx} is outside the valid range 0-{total_frames-1}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx}")
    return frame


def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def build_camera_rotation(pitch_rad: float) -> np.ndarray:
    r_base = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    r_pitch = rotation_matrix_x(pitch_rad)
    return r_pitch @ r_base


def generate_plane_lines(tile_radius: int, samples: int) -> list[np.ndarray]:
    axis = np.linspace(-tile_radius, tile_radius, samples, dtype=np.float64)
    lines = []
    for x in range(-tile_radius, tile_radius + 1):
        xs = np.full_like(axis, float(x))
        lines.append(np.stack([xs, axis], axis=1))
    for y in range(-tile_radius, tile_radius + 1):
        ys = np.full_like(axis, float(y))
        lines.append(np.stack([axis, ys], axis=1))
    return lines


def plane_points_to_world(
    plane_points: np.ndarray,
    grid_x: float,
    grid_y: float,
    rotation_matrix: np.ndarray,
    scale: float,
) -> np.ndarray:
    zeros = np.zeros((plane_points.shape[0], 1), dtype=np.float64)
    pts3 = np.concatenate([plane_points[:, [0]], zeros, plane_points[:, [1]]], axis=1)
    rotated = (rotation_matrix @ pts3.T).T
    translated = rotated + np.array([grid_x, 0.0, grid_y], dtype=np.float64)
    return translated * scale


def project_world_points(
    world_points: np.ndarray,
    camera_pos: np.ndarray,
    camera_rotation: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z_epsilon: float = 1e-4,
) -> list[np.ndarray]:
    rel = world_points - camera_pos
    cam_pts = (camera_rotation @ rel.T).T
    zs = cam_pts[:, 2]
    xs = cam_pts[:, 0]
    ys = cam_pts[:, 1]

    valid_mask = zs > z_epsilon
    segments = []
    current = []

    for is_valid, x_cam, y_cam, z_cam in zip(valid_mask, xs, ys, zs):
        if is_valid:
            x_img = fx * (x_cam / z_cam) + cx
            y_img = fy * (y_cam / z_cam) + cy
            current.append([x_img, y_img])
        else:
            if len(current) >= 2:
                segments.append(np.array(current, dtype=np.float32))
            current = []

    if len(current) >= 2:
        segments.append(np.array(current, dtype=np.float32))

    return segments


def parse_color(color_str: str) -> tuple[int, int, int]:
    parts = [int(float(c.strip())) for c in color_str.split(",")]
    if len(parts) != 3:
        raise ValueError("Color must be provided as B,G,R")
    return tuple(max(0, min(255, c)) for c in parts)


def draw_grid_overlay(
    frame: np.ndarray,
    params: dict[str, float],
    lines: list[np.ndarray],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    color: tuple[int, int, int],
    thickness: int,
):
    camera_height = 1.0
    camera_pos = np.array([0.0, camera_height, 0.0], dtype=np.float64)
    camera_rotation = build_camera_rotation(np.deg2rad(params["Pitch"]))
    grid_rotation = rotation_matrix_y(np.deg2rad(params["Grid Rot"]))

    for plane_line in lines:
        world_pts = plane_points_to_world(
            plane_line,
            grid_x=params["Grid X"],
            grid_y=params["Grid Y"],
            rotation_matrix=grid_rotation,
            scale=params["Scale"],
        )
        projected_segments = project_world_points(world_pts, camera_pos, camera_rotation, fx, fy, cx, cy)
        for segment in projected_segments:
            pts_int = np.round(segment).astype(np.int32).reshape(-1, 1, 2)
            if pts_int.shape[0] >= 2:
                cv2.polylines(frame, [pts_int], False, color, thickness, lineType=cv2.LINE_AA)


class App:
    def __init__(self, root: tk.Tk, args, frame: np.ndarray):
        self.root = root
        self.args = args
        self.base_frame = frame.copy()
        self.gray_frame = cv2.cvtColor(self.base_frame, cv2.COLOR_BGR2GRAY)
        self.lines = generate_plane_lines(args.tile_radius, args.samples_per_line)
        self.search_lines = generate_plane_lines(args.tile_radius, 50)
        self.height, self.width = self.base_frame.shape[:2]
        self.fx = self.fy = max(self.width, self.height)
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.color = parse_color(args.color)
        self._pending_job: str | None = None
        self.searching = False
        self.status_var = tk.StringVar(value="")
        self.camera_height = 1.0
        self.camera_pos = np.array([0.0, self.camera_height, 0.0], dtype=np.float64)

        max_w = 960
        max_h = 720
        scale = min(max_w / self.width, max_h / self.height, 1.0)
        self.display_w = int(round(self.width * scale))
        self.display_h = int(round(self.height * scale))

        self.root.title(WINDOW_TITLE)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(main)
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        control_frame = ttk.Frame(main)
        control_frame.grid(row=0, column=1, sticky="ns")
        control_frame.columnconfigure(0, weight=1)

        self.slider_vars: dict[str, tk.DoubleVar] = {}
        self.value_labels: dict[str, ttk.Label] = {}
        self.slider_configs = {slider["name"]: slider for slider in SLIDERS}

        for idx, slider in enumerate(SLIDERS):
            var = tk.DoubleVar(value=slider["initial"])
            self.slider_vars[slider["name"]] = var
            row_offset = idx * 3
            ttk.Label(control_frame, text=slider["name"]).grid(
                row=row_offset, column=0, columnspan=TOTAL_CONTROL_COLUMNS, sticky="w"
            )

            for col, (label, delta) in enumerate(STEP_BUTTONS_NEG):
                ttk.Button(
                    control_frame,
                    text=label,
                    width=max(4, len(label)),
                    command=lambda name=slider["name"], step=delta: self.adjust_slider(name, step),
                ).grid(row=row_offset + 1, column=col, sticky="ew", padx=(0, 2))

            scale = ttk.Scale(
                control_frame,
                from_=slider["min"],
                to=slider["max"],
                orient=tk.HORIZONTAL,
                variable=var,
                command=self._on_slider_changed,
            )
            slider_col = len(STEP_BUTTONS_NEG)
            scale.grid(row=row_offset + 1, column=slider_col, sticky="ew", padx=4)

            for idx_pos, (label, delta) in enumerate(STEP_BUTTONS_POS):
                col = slider_col + 1 + idx_pos
                ttk.Button(
                    control_frame,
                    text=label,
                    width=max(5, len(label)),
                    command=lambda name=slider["name"], step=delta: self.adjust_slider(name, step),
                ).grid(row=row_offset + 1, column=col, sticky="ew", padx=(2, 0))

            val_label = ttk.Label(control_frame, text=self._format_value(slider["name"], slider["unit"], var.get()))
            val_label.grid(
                row=row_offset + 1,
                column=slider_col + 1 + len(STEP_BUTTONS_POS),
                sticky="w",
                padx=(6, 0),
            )
            self.value_labels[slider["name"]] = val_label
            control_frame.grid_rowconfigure(row_offset + 2, minsize=6)

        control_frame.columnconfigure(len(STEP_BUTTONS_NEG), weight=1)

        button_frame = ttk.Frame(control_frame, padding=(0, 10))
        button_frame.grid(row=len(SLIDERS) * 3, column=0, columnspan=TOTAL_CONTROL_COLUMNS, sticky="ew")
        ttk.Button(button_frame, text="Reset", command=self.reset_sliders).grid(row=0, column=0, sticky="ew")
        ttk.Button(button_frame, text="Close", command=self.root.destroy).grid(row=0, column=1, sticky="ew", padx=(6, 4))
        self.find_button = ttk.Button(button_frame, text="Find Best Grid", command=self.start_bruteforce)
        self.find_button.grid(row=0, column=2, sticky="ew")
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=2)

        status_label = ttk.Label(button_frame, textvariable=self.status_var, anchor="w")
        status_label.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        self.update_image()

    def _format_value(self, name: str, unit: str, value: float) -> str:
        if unit == "x":
            return f"{value:.2f} {unit}"
        if unit == "tiles":
            return f"{value:.2f} {unit}"
        return f"{value:.2f} {unit}"

    def _on_slider_changed(self, _value: str):
        self._update_value_labels()
        self._schedule_update()

    def reset_sliders(self):
        for slider in SLIDERS:
            self.slider_vars[slider["name"]].set(slider["initial"])
        self._update_value_labels()
        self.update_image()

    def adjust_slider(self, slider_name: str, delta: float):
        config = self.slider_configs[slider_name]
        var = self.slider_vars[slider_name]
        new_value = np.clip(var.get() + delta, config["min"], config["max"])
        var.set(float(new_value))
        self._update_value_label(slider_name)
        self._schedule_update()

    def read_params(self) -> dict[str, float]:
        return {name: var.get() for name, var in self.slider_vars.items()}

    def _update_value_labels(self):
        for slider in SLIDERS:
            self._update_value_label(slider["name"])

    def _update_value_label(self, slider_name: str):
        slider = self.slider_configs[slider_name]
        current = self.slider_vars[slider_name].get()
        self.value_labels[slider_name].config(
            text=self._format_value(slider_name, slider["unit"], current)
        )

    def _schedule_update(self):
        if self._pending_job is not None:
            self.root.after_cancel(self._pending_job)
        self._pending_job = self.root.after(10, self.update_image)

    def update_image(self):
        self._pending_job = None
        params = self.read_params()
        frame = self.base_frame.copy()
        draw_grid_overlay(
            frame,
            params=params,
            lines=self.lines,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            color=self.color,
            thickness=self.args.thickness,
        )
        display_frame = cv2.resize(frame, (self.display_w, self.display_h), interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image=image_pil)
        self.image_label.configure(image=image_tk)
        self.image_label.image = image_tk

    def start_bruteforce(self):
        if self.searching:
            return
        self.searching = True
        self.find_button.config(state=tk.DISABLED)
        self.status_var.set("Searching for best grid...")
        print("[BruteForce] Search started", flush=True)
        thread = threading.Thread(target=self._bruteforce_search, daemon=True)
        thread.start()

    def _bruteforce_search(self):
        ranges = {
            "Grid X": self._frange(-1.0, 1.0, 0.2),
            "Grid Y": self._frange(-1.0, 1.0, 0.2),
            "Grid Rot": self._frange(-5.0, 5.0, 1.0),
            "Scale": self._frange(0.15, 0.25, 0.01),
            "Pitch": self._frange(-20.0, -10.0, 1.0),
        }

        best_score = float("inf")
        best_params = None
        total = (
            len(ranges["Grid X"])
            * len(ranges["Grid Y"])
            * len(ranges["Grid Rot"])
            * len(ranges["Scale"])
            * len(ranges["Pitch"])
        )
        checked = 0

        for grid_x in ranges["Grid X"]:
            for grid_y in ranges["Grid Y"]:
                for grid_rot in ranges["Grid Rot"]:
                    for scale in ranges["Scale"]:
                        for pitch in ranges["Pitch"]:
                            params = {
                                "Grid X": grid_x,
                                "Grid Y": grid_y,
                                "Grid Rot": grid_rot,
                                "Scale": scale,
                                "Pitch": pitch,
                            }
                            score = self.compute_grid_brightness(params)
                            if score < best_score:
                                best_score = score
                                best_params = params.copy()
                            checked += 1
                            if checked % 500 == 0:
                                progress_text = f"Searching... {checked:,}/{total:,} (best={best_score:.2f})"
                                self._queue_status(progress_text)
                                print(f"[BruteForce] {progress_text}", flush=True)

        self.root.after(0, lambda: self._apply_search_result(best_params, best_score))

    def _apply_search_result(self, best_params: dict[str, float] | None, best_score: float):
        self.searching = False
        self.find_button.config(state=tk.NORMAL)
        if not best_params:
            final_text = "Search finished: no valid grid found."
            self.status_var.set(final_text)
            print(f"[BruteForce] {final_text}", flush=True)
            return

        for name, value in best_params.items():
            self.slider_vars[name].set(value)
        self._update_value_labels()
        self.update_image()
        final_text = f"Best brightness: {best_score:.2f}"
        self.status_var.set(final_text)
        print(f"[BruteForce] {final_text}", flush=True)

    def compute_grid_brightness(self, params: dict[str, float]) -> float:
        grid_rotation = rotation_matrix_y(np.deg2rad(params["Grid Rot"]))
        camera_rotation = build_camera_rotation(np.deg2rad(params["Pitch"]))
        total = 0.0
        for plane_line in self.search_lines:
            world_pts = plane_points_to_world(
                plane_line,
                grid_x=params["Grid X"],
                grid_y=params["Grid Y"],
                rotation_matrix=grid_rotation,
                scale=params["Scale"],
            )
            xs, ys, valid = self._project_world_points(world_pts, camera_rotation)
            total += self._sample_brightness(xs, ys, valid)
        return total

    def _project_world_points(
        self,
        world_points: np.ndarray,
        camera_rotation: np.ndarray,
        z_epsilon: float = 1e-4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rel = world_points - self.camera_pos
        cam_pts = (camera_rotation @ rel.T).T
        zs = cam_pts[:, 2]
        xs = np.zeros_like(zs)
        ys = np.zeros_like(zs)
        valid = zs > z_epsilon
        if np.any(valid):
            xs[valid] = self.fx * (cam_pts[valid, 0] / zs[valid]) + self.cx
            ys[valid] = self.fy * (cam_pts[valid, 1] / zs[valid]) + self.cy
        return xs, ys, valid

    def _sample_brightness(self, xs: np.ndarray, ys: np.ndarray, valid: np.ndarray) -> float:
        h, w = self.gray_frame.shape
        inside = valid & (xs >= 0.0) & (xs < w) & (ys >= 0.0) & (ys < h)
        if not np.any(inside):
            return 0.0
        xs_idx = np.clip(np.round(xs[inside]).astype(int), 0, w - 1)
        ys_idx = np.clip(np.round(ys[inside]).astype(int), 0, h - 1)
        return float(np.sum(self.gray_frame[ys_idx, xs_idx]))

    @staticmethod
    def _frange(start: float, end: float, step: float) -> list[float]:
        count = int(round((end - start) / step))
        values = [round(start + i * step, 6) for i in range(count + 1)]
        return values

    def _queue_status(self, text: str):
        self.root.after(0, lambda: self.status_var.set(text))


def main():
    args = parse_args()
    frame = load_frame(args.video, args.frame)
    root = tk.Tk()
    App(root, args, frame)
    root.mainloop()


if __name__ == "__main__":
    main()

