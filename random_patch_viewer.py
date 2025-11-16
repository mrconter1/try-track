import argparse
import random
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk


class RandomPatchViewer:
    def __init__(self, root, video_path, total_frames, frame_idx, coords, patch, patch_size):
        self.root = root
        self.video_path = video_path
        self.total_frames = total_frames
        self.frame_idx = frame_idx
        self.coords = coords
        self.patch = patch
        self.patch_size = patch_size
        self.display_size = 250
        self.photo_image = None

        self.root.title("Random Patch Viewer")
        self._build_ui()
        self._display_patch()

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=20)
        container.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.info_label = ttk.Label(container, justify="left")
        self.info_label.grid(row=0, column=0, sticky="w")
        self._update_info_label()

        self.canvas = tk.Canvas(
            container,
            width=self.display_size,
            height=self.display_size,
            highlightthickness=0,
            borderwidth=0,
            bg="black",
        )
        self.canvas.grid(row=1, column=0, pady=10)

        self.random_button = ttk.Button(
            container,
            text="Randomize patch",
            command=self.load_random_patch,
        )
        self.random_button.grid(row=2, column=0, sticky="ew")

    def _display_patch(self):
        rgb_patch = cv2.cvtColor(self.patch, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_patch).resize(
            (self.display_size, self.display_size),
            resample=Image.NEAREST,
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

    def load_random_patch(self):
        frame_idx, frame, total_frames = choose_random_frame(self.video_path)
        coords, patch = choose_random_patch(frame, self.patch_size)
        self.frame_idx = frame_idx
        self.total_frames = total_frames
        self.coords = coords
        self.patch = patch
        self._update_info_label()
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

