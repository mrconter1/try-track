import cv2
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class DotViewer:
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
            variable=self.x_var, command=self.update_image, showvalue=0, resolution=1
        )
        self.x_slider.grid(row=1, column=1, sticky="ew")
        self.x_label = ttk.Label(control, text=f"{self.x_var.get():.0f}")
        self.x_label.grid(row=1, column=2, padx=5)

        ttk.Label(control, text="Y:").grid(row=2, column=0, sticky=tk.W)
        self.y_slider = tk.Scale(
            control, from_=0, to=self.h - 1, orient=tk.HORIZONTAL,
            variable=self.y_var, command=self.update_image, showvalue=0, resolution=1
        )
        self.y_slider.grid(row=2, column=1, sticky="ew")
        self.y_label = ttk.Label(control, text=f"{self.y_var.get():.0f}")
        self.y_label.grid(row=2, column=2, padx=5)

        self.search_button = ttk.Button(control, text="Search", command=self.toggle_scan)
        self.search_button.grid(row=3, column=0, columnspan=3, pady=5, sticky="ew")

        control.columnconfigure(1, weight=1)

        self.image_label = ttk.Label(main)
        self.image_label.grid(row=0, column=0, sticky="nsew")

        main.rowconfigure(0, weight=1)

        self.display_w = min(1200, self.w)
        self.display_h = int(self.display_w * (self.h / self.w))

        self.scanning = False
        self.scan_length = 0
        self.update_image()

    def update_image(self, *args):
        frame_copy = self.original_frame.copy()
        x = int(self.x_var.get())
        y = int(self.y_var.get())
        cv2.circle(frame_copy, (x, y), 6, (0, 0, 255), -1)

        if self.scanning and self.scan_length > 0:
            x_end = min(x + self.scan_length, self.w - 1)
            cv2.line(frame_copy, (x, y), (x_end, y), (0, 255, 0), 2)
            cv2.circle(frame_copy, (x_end, y), 4, (0, 255, 0), -1)

        display = cv2.resize(frame_copy, (self.display_w, self.display_h), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img))
        self.image_label.imgtk = img_tk
        self.image_label.configure(image=img_tk)

        self.x_label.config(text=f"{self.x_var.get():.0f}")
        self.y_label.config(text=f"{self.y_var.get():.0f}")
        self.frame_label.config(text=f"{self.frame_var.get()}")

    def on_frame_change(self, value):
        frame_idx = int(float(value))
        frame = load_frame(self.video_path, frame_idx)
        self.original_frame = frame
        self.update_image()

    def toggle_scan(self):
        if self.scanning:
            self.finish_scan()
        else:
            self.start_line_scan()

    def start_line_scan(self):
        if self.scanning:
            return
        self.scanning = True
        self.scan_length = 0
        self.search_button.config(text="Stop")
        self.schedule_scan_step()

    def schedule_scan_step(self):
        if not self.scanning:
            return
        self.scan_length += 1
        self.log_scan_brightness()
        self.update_image()
        max_len = self.w - int(self.x_var.get()) - 1
        if self.scan_length >= max_len:
            self.finish_scan()
        else:
            self.root.after(100, self.schedule_scan_step)

    def finish_scan(self):
        self.scanning = False
        self.scan_length = 0
        self.search_button.config(text="Search")
        self.update_image()

    def log_scan_brightness(self):
        x = int(self.x_var.get() + self.scan_length)
        y = int(self.y_var.get())
        x = min(x, self.w - 1)
        b, g, r = self.original_frame[y, x]
        brightness = int(0.299 * r + 0.587 * g + 0.114 * b)
        print(f"[scan] x={x} y={y} brightness={brightness}")


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

