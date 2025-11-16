import argparse
import random
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw

class RandomPatchViewer:
    def __init__(self, root, video_path, total_frames, patch_size):
        self.root = root
        self.video_path = video_path
        self.total_frames = total_frames
        self.patch_size = patch_size
        self.history = []
        self.history_idx = -1
        self.current_entry = None
        self.photo_image = None
        self.max_display_width = 1100
        self.max_display_height = 750
        self.scale_x = 1.0
        self.scale_y = 1.0

        self.root.title("Frame Annotation Viewer")
        self._build_ui()
        self._append_random_frame()

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=20)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_columnconfigure(0, weight=1)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.minsize(1000, 900)
        try:
            self.root.state("zoomed")
        except tk.TclError:
            self.root.geometry("1200x900")
        self.root.bind("<d>", self._on_key_next)
        self.root.bind("<D>", self._on_key_next)
        self.root.bind("<a>", self._on_key_previous)
        self.root.bind("<A>", self._on_key_previous)

        content = ttk.Frame(container)
        content.grid(row=0, column=0, sticky="n")
        content.grid_columnconfigure(0, weight=1)

        self.info_label = ttk.Label(content, justify="center", anchor="center")
        self.info_label.grid(row=0, column=0, sticky="ew")

        sample_label_frame = ttk.Frame(content)
        sample_label_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        sample_label_frame.grid_columnconfigure(0, weight=1)
        self.state_label = ttk.Label(
            sample_label_frame,
            justify="center",
            anchor="center",
            font=("Segoe UI", 12, "bold"),
        )
        self.state_label.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        self.coords_label = ttk.Label(
            sample_label_frame,
            justify="center",
            anchor="center",
            font=("Segoe UI", 12, "bold"),
        )
        self.coords_label.grid(row=1, column=0, sticky="ew")

        self.canvas = tk.Canvas(
            content,
            width=self.max_display_width,
            height=self.max_display_height,
            highlightthickness=0,
            borderwidth=0,
            bg="black",
        )
        self.canvas.grid(row=2, column=0, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)

        button_row = ttk.Frame(content)
        button_row.grid(row=3, column=0, sticky="ew", pady=(5, 0))
        button_row.columnconfigure(0, weight=1)

        self.random_button = ttk.Button(
            button_row,
            text="Next random frame",
            command=self.load_next_frame,
        )
        self.random_button.grid(row=0, column=0, sticky="ew")

        self.stats_label = ttk.Label(content, justify="center", anchor="center")
        self.stats_label.grid(row=4, column=0, pady=(10, 0), sticky="ew")

    def _append_random_frame(self):
        frame_idx, frame, total_frames = choose_random_frame(self.video_path)
        self.total_frames = total_frames
        entry = {"frame_idx": frame_idx, "frame": frame, "annotations": []}
        if self.history_idx < len(self.history) - 1:
            self.history = self.history[: self.history_idx + 1]
        self.history.append(entry)
        self.history_idx = len(self.history) - 1
        self._set_current_entry(entry)

    def _set_current_entry(self, entry):
        self.current_entry = entry
        self._update_info_label()
        self._display_current_frame()
        self._update_stats_label()

    def _on_key_next(self, event):
        self.load_next_frame()

    def _on_key_previous(self, event):
        self.load_previous_frame()

    def load_next_frame(self):
        if self.history_idx < len(self.history) - 1:
            self.history_idx += 1
            self._set_current_entry(self.history[self.history_idx])
        else:
            self._append_random_frame()

    def load_previous_frame(self):
        if self.history_idx <= 0:
            return
        self.history_idx -= 1
        self._set_current_entry(self.history[self.history_idx])

    def _update_info_label(self):
        if not self.current_entry:
            self.info_label.config(text="–")
            return
        entry = self.current_entry
        info_text = (
            f"Video: {self.video_path}\n"
            f"Frame: {entry['frame_idx'] + 1} / {self.total_frames}\n"
            f"Annotations on this frame: {len(entry['annotations'])}"
        )
        self.info_label.config(text=info_text)
        self._update_annotation_label()

    def _update_annotation_label(self):
        if not self.current_entry or not self.current_entry["annotations"]:
            self.state_label.config(text="Annotations on frame: 0")
            self.coords_label.config(text="Last point: –")
            return
        count = len(self.current_entry["annotations"])
        last = self.current_entry["annotations"][-1]
        self.state_label.config(text=f"Annotations on frame: {count}")
        self.coords_label.config(
            text=f"Last point: ({last['x']:.1f}, {last['y']:.1f})"
        )

    def _update_stats_label(self):
        total_frames = len(self.history)
        annotated_frames = sum(1 for e in self.history if e["annotations"])
        total_points = sum(len(e["annotations"]) for e in self.history)
        self.stats_label.config(
            text=f"Frames visited: {total_frames} | Frames with crosses: {annotated_frames} | Total crosses: {total_points}"
        )

    def _display_current_frame(self):
        if not self.current_entry:
            return
        frame = self.current_entry["frame"]
        h, w = frame.shape[:2]
        scale = min(
            self.max_display_width / max(1, w),
            self.max_display_height / max(1, h),
            1.0,
        )
        disp_w = int(w * scale)
        disp_h = int(h * scale)
        self.scale_x = scale
        self.scale_y = scale
        resized = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        for ann in self.current_entry["annotations"]:
            dx = ann["x"] * scale
            dy = ann["y"] * scale
            half = 10
            draw.line((dx - half, dy, dx + half, dy), fill="red", width=2)
            draw.line((dx, dy - half, dx, dy + half), fill="red", width=2)
        self.photo_image = ImageTk.PhotoImage(image)
        self.canvas.configure(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)

    def on_canvas_click(self, event):
        if not self.current_entry:
            return
        frame_x = event.x / max(1e-6, self.scale_x)
        frame_y = event.y / max(1e-6, self.scale_y)
        frame = self.current_entry["frame"]
        h, w = frame.shape[:2]
        frame_x = float(np.clip(frame_x, 0.0, w - 1e-6))
        frame_y = float(np.clip(frame_y, 0.0, h - 1e-6))
        self.current_entry["annotations"].append({"x": frame_x, "y": frame_y})
        self._update_annotation_label()
        self._update_stats_label()
        self._display_current_frame()

    def on_canvas_right_click(self, event):
        if not self.current_entry or not self.current_entry["annotations"]:
            return
        frame_x = event.x / max(1e-6, self.scale_x)
        frame_y = event.y / max(1e-6, self.scale_y)
        annotations = self.current_entry["annotations"]
        distances = [
            ((a["x"] - frame_x) ** 2 + (a["y"] - frame_y) ** 2, idx)
            for idx, a in enumerate(annotations)
        ]
        distances.sort()
        _, idx = distances[0]
        annotations.pop(idx)
        self._update_annotation_label()
        self._update_stats_label()
        self._display_current_frame()



def choose_random_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video '{video_path}'")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video '{video_path}' does not contain any frames")

    frame_idx = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise ValueError(f"Failed to read frame {frame_idx} from '{video_path}'")

    return frame_idx, frame, total_frames


def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video '{video_path}'")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    return total


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate frames and train a simple cross detector."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="video.mp4",
        help="Path to the video file to sample from.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=100,
        help="Square patch size in pixels (default: 100).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    total_frames = get_total_frames(args.video)
    viewer = RandomPatchViewer(root, args.video, total_frames, args.patch_size)
    root.mainloop()


if __name__ == "__main__":
    main()

