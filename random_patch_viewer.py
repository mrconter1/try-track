import argparse
import random
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk, ImageDraw


class RandomPatchViewer:
    def __init__(self, root, video_path, total_frames, frame_idx, coords, patch, patch_size):
        self.root = root
        self.video_path = video_path
        self.patch_size = patch_size
        self.display_size = 400
        self.photo_image = None
        self.total_frames = total_frames
        self.frame_idx = frame_idx
        self.coords = coords
        self.patch = patch
        self.history = []
        self.history_idx = -1
        self.current_sample = None

        self.root.title("Random Patch Viewer")
        self._build_ui()
        self._store_sample(frame_idx, total_frames, coords, patch)

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=20)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_columnconfigure(0, weight=1)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.minsize(720, 780)
        self.root.geometry("760x820")
        self.root.bind("<d>", self._on_key_randomize)
        self.root.bind("<D>", self._on_key_randomize)
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
        self._update_info_label()

        self.canvas = tk.Canvas(
            content,
            width=self.display_size,
            height=self.display_size,
            highlightthickness=0,
            borderwidth=0,
            bg="black",
        )
        self.canvas.grid(row=2, column=0, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)

        self.random_button = ttk.Button(
            content,
            text="Randomize patch",
            command=self.load_next_patch,
        )
        self.random_button.grid(row=3, column=0, sticky="ew")

    def _display_patch(self):
        rgb_patch = cv2.cvtColor(self.patch, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_patch).resize(
            (self.display_size, self.display_size),
            resample=Image.NEAREST,
        )
        if self.current_sample and self.current_sample.get("annotation"):
            ann_x, ann_y = self.current_sample["annotation"]
            patch_h, patch_w = self.patch.shape[:2]
            scale_x = self.display_size / max(1, patch_w)
            scale_y = self.display_size / max(1, patch_h)
            draw = ImageDraw.Draw(image)
            disp_x = int(ann_x * scale_x)
            disp_y = int(ann_y * scale_y)
            r = 6
            draw.ellipse(
                (disp_x - r, disp_y - r, disp_x + r, disp_y + r),
                fill="red",
                outline="black",
            )
        self.photo_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)

    def _update_info_label(self):
        info_text = (
            f"Video: {self.video_path}\n"
            f"Frame: {self.frame_idx + 1} / {self.total_frames}\n"
            f"Top-left pixel: ({self.coords[0]}, {self.coords[1]})"
        )
        self.info_label.config(text=info_text)
        self._update_sample_label()

    def _update_sample_label(self):
        state_color = "#cc0000"
        coords_color = "#cc0000"
        if not self.current_sample:
            state_text = "State: –"
            coord_text = "Patch coords: –"
        else:
            state = self.current_sample.get("state", "No cross")
            annotation = self.current_sample.get("annotation")
            if annotation:
                coord_text = f"Patch coords: ({annotation[0]}, {annotation[1]})"
                coords_color = "#003399"
            else:
                coord_text = "Patch coords: –"
            if state == "Has cross" and annotation:
                state_color = "#003399"
                coords_color = "#003399"
            else:
                state_color = "#cc0000"
                coords_color = "#cc0000" if annotation is None else coords_color
            state_text = f"State: {state}"
        self.state_label.config(text=state_text, foreground=state_color)
        self.coords_label.config(text=coord_text, foreground=coords_color)

    def load_next_patch(self):
        if self.history_idx < len(self.history) - 1:
            self.history_idx += 1
            sample = self.history[self.history_idx]
            self._apply_sample(sample)
        else:
            self._append_random_sample()

    def _append_random_sample(self):
        frame_idx, frame, total_frames = choose_random_frame(self.video_path)
        coords, patch = choose_random_patch(frame, self.patch_size)
        self._store_sample(frame_idx, total_frames, coords, patch)

    def _on_key_randomize(self, event):
        self.load_next_patch()

    def load_previous_patch(self):
        if self.history_idx <= 0:
            return
        self.history_idx -= 1
        sample = self.history[self.history_idx]
        self._apply_sample(sample)

    def _store_sample(self, frame_idx, total_frames, coords, patch):
        if self.history_idx < len(self.history) - 1:
            self.history = self.history[: self.history_idx + 1]
        sample = {
            "frame_idx": frame_idx,
            "total_frames": total_frames,
            "coords": coords,
            "patch": patch,
            "state": "No cross",
            "annotation": None,
        }
        self.history.append(sample)
        self.history_idx += 1
        self._apply_sample(sample)

    def _apply_sample(self, sample):
        self.current_sample = sample
        self.frame_idx = sample["frame_idx"]
        self.total_frames = sample["total_frames"]
        self.coords = sample["coords"]
        self.patch = sample["patch"]
        self._update_info_label()
        self._display_patch()

    def _on_key_previous(self, event):
        self.load_previous_patch()

    def on_canvas_click(self, event):
        self._update_annotation_from_event(event)

    def on_canvas_drag(self, event):
        self._update_annotation_from_event(event)

    def _update_annotation_from_event(self, event):
        if not self.current_sample:
            return
        patch_h, patch_w = self.patch.shape[:2]
        if patch_h == 0 or patch_w == 0:
            return
        scale_x = patch_w / self.display_size
        scale_y = patch_h / self.display_size
        patch_x = int(min(max(event.x * scale_x, 0), patch_w - 1))
        patch_y = int(min(max(event.y * scale_y, 0), patch_h - 1))
        self.set_annotation(patch_x, patch_y)

    def set_annotation(self, patch_x, patch_y):
        if not self.current_sample:
            return
        self.current_sample["annotation"] = (patch_x, patch_y)
        self.current_sample["state"] = "Has cross"
        self._update_sample_label()
        self._display_patch()


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


def choose_random_patch(frame, patch_size):
    if patch_size <= 0:
        raise ValueError("Patch size must be positive")

    height, width = frame.shape[:2]
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"Patch size {patch_size} exceeds frame dimensions {width}x{height}"
        )

    max_x = width - patch_size
    max_y = height - patch_size
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    patch = frame[y : y + patch_size, x : x + patch_size].copy()
    return (x, y), patch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display a random 100x100 region from a random frame in a video."
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
    frame_idx, frame, total_frames = choose_random_frame(args.video)
    coords, patch = choose_random_patch(frame, args.patch_size)

    root = tk.Tk()
    viewer = RandomPatchViewer(
        root,
        args.video,
        total_frames,
        frame_idx,
        coords,
        patch,
        args.patch_size,
    )
    root.mainloop()


if __name__ == "__main__":
    main()

