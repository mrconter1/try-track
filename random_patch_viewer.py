import argparse
import json
import os
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw

class RandomPatchViewer:
    def __init__(self, root, video_paths, patch_size):
        self.root = root
        self.video_paths = [os.path.abspath(p) for p in video_paths]
        self.patch_size = patch_size
        self.history = []
        self.history_idx = -1
        self.current_entry = None
        self.photo_image = None
        self.max_display_width = 1100
        self.max_display_height = 750
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.dragging = False
        self.preview_point = None
        self.magnifier_window = None
        self.magnifier_image = None
        self.magnifier_size = 160
        self.magnifier_zoom = 4
        self.drawing_rect = False
        self.rect_start = None
        self.rect_preview = None

        self.root.title("Frame Annotation Viewer")
        self._build_ui()
        self._load_existing_annotations()
        if not self.history:
            self._append_random_frame()

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=20)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_columnconfigure(0, weight=3)
        container.grid_columnconfigure(1, weight=1)
        container.grid_rowconfigure(0, weight=1)

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
        self.root.bind("<Control-z>", self._on_undo)
        self.root.bind("<Control-Z>", self._on_undo)

        canvas_frame = ttk.Frame(container)
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.bind("<Configure>", self._on_canvas_frame_resize)

        self.canvas = tk.Canvas(
            canvas_frame,
            highlightthickness=0,
            borderwidth=0,
            bg="black",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<ButtonPress-3>", self.on_right_press)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)

        control_frame = ttk.Frame(container)
        control_frame.grid(row=0, column=1, sticky="nsew")
        control_frame.grid_columnconfigure(0, weight=1)

        self.info_label = ttk.Label(
            control_frame, justify="left", anchor="w", font=("Segoe UI", 11, "bold")
        )
        self.info_label.grid(row=0, column=0, sticky="ew")

        sample_label_frame = ttk.Frame(control_frame)
        sample_label_frame.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        sample_label_frame.grid_columnconfigure(0, weight=1)
        self.state_label = ttk.Label(
            sample_label_frame,
            justify="left",
            anchor="w",
            font=("Segoe UI", 12, "bold"),
        )
        self.state_label.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        self.coords_label = ttk.Label(
            sample_label_frame,
            justify="left",
            anchor="w",
            font=("Segoe UI", 12, "bold"),
        )
        self.coords_label.grid(row=1, column=0, sticky="ew")

        self.stats_label = ttk.Label(
            control_frame, justify="left", anchor="w", wraplength=260
        )
        self.stats_label.grid(row=2, column=0, sticky="ew", pady=(0, 15))

        self.random_button = ttk.Button(
            control_frame,
            text="Next random frame",
            command=self.load_next_frame,
        )
        self.random_button.grid(row=3, column=0, sticky="ew")

        self.export_button = ttk.Button(
            control_frame,
            text="Export annotations",
            command=self.export_annotations,
        )
        self.export_button.grid(row=4, column=0, sticky="ew", pady=(10, 0))

    def _append_random_frame(self):
        video_path, frame_idx, frame, total_frames = choose_random_frame_multi(self.video_paths)
        entry = {
            "video_path": video_path,
            "frame_idx": frame_idx,
            "frame": frame,
            "total_frames": total_frames,
            "annotations": [],
            "negative_rects": []
        }
        if self.history_idx < len(self.history) - 1:
            self.history = self.history[: self.history_idx + 1]
        self.history.append(entry)
        self.history_idx = len(self.history) - 1
        self._set_current_entry(entry)

    def _set_current_entry(self, entry):
        self.current_entry = entry
        self.preview_point = None
        self.rect_start = None
        self.rect_preview = None
        self._hide_magnifier()
        self._update_info_label()
        self._display_current_frame()
        self._update_stats_label()

    def _on_key_next(self, event):
        self.load_next_frame()

    def _on_key_previous(self, event):
        self.load_previous_frame()

    def _on_undo(self, event):
        if not self.current_entry:
            return
        if self.current_entry["annotations"]:
            self.current_entry["annotations"].pop()
            self._update_annotation_label()
            self._update_stats_label()
            self._display_current_frame()
        elif self.current_entry.get("negative_rects"):
            self.current_entry["negative_rects"].pop()
            self._update_annotation_label()
            self._update_stats_label()
            self._display_current_frame()

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
        video_name = os.path.basename(entry.get("video_path", "unknown"))
        total_frames = entry.get("total_frames", "?")
        info_text = (
            f"Video: {video_name}\n"
            f"Frame: {entry['frame_idx'] + 1} / {total_frames}\n"
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
        )
        scale = max(scale, 0.01)
        disp_w = int(w * scale)
        disp_h = int(h * scale)
        self.scale_x = scale
        self.scale_y = scale
        resized = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        for rect in self.current_entry.get("negative_rects", []):
            x0 = rect["x0"] * scale
            y0 = rect["y0"] * scale
            x1 = rect["x1"] * scale
            y1 = rect["y1"] * scale
            draw.rectangle([x0, y0, x1, y1], outline="blue", width=2)
            rx = rect["x0"]
            ry = rect["y0"]
            rw = rect["x1"] - rect["x0"]
            rh = rect["y1"] - rect["y0"]
            label = f"({rx:.0f},{ry:.0f}) {rw:.0f}x{rh:.0f}"
            draw.text((x0 + 4, y0 + 4), label, fill="blue")
        if self.rect_preview:
            x0 = self.rect_preview["x0"] * scale
            y0 = self.rect_preview["y0"] * scale
            x1 = self.rect_preview["x1"] * scale
            y1 = self.rect_preview["y1"] * scale
            draw.rectangle([x0, y0, x1, y1], outline="cyan", width=2)
            rx = self.rect_preview["x0"]
            ry = self.rect_preview["y0"]
            rw = self.rect_preview["x1"] - self.rect_preview["x0"]
            rh = self.rect_preview["y1"] - self.rect_preview["y0"]
            label = f"({rx:.0f},{ry:.0f}) {rw:.0f}x{rh:.0f}"
            draw.text((x0 + 4, y0 + 4), label, fill="cyan")
        for ann in self.current_entry["annotations"]:
            dx = ann["x"] * scale
            dy = ann["y"] * scale
            half = 10
            draw.line((dx - half, dy, dx + half, dy), fill="red", width=2)
            draw.line((dx, dy - half, dx, dy + half), fill="red", width=2)
        if self.preview_point:
            dx = self.preview_point["x"] * scale
            dy = self.preview_point["y"] * scale
            half = 12
            draw.line((dx - half, dy, dx + half, dy), fill="lime", width=2)
            draw.line((dx, dy - half, dx, dy + half), fill="lime", width=2)
        self.photo_image = ImageTk.PhotoImage(image)
        self.canvas.configure(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)

    def _on_canvas_frame_resize(self, event):
        new_w = max(100, event.width)
        new_h = max(100, event.height)
        if (
            abs(new_w - self.max_display_width) > 2
            or abs(new_h - self.max_display_height) > 2
        ):
            self.max_display_width = new_w
            self.max_display_height = new_h
            self._display_current_frame()

    def on_left_press(self, event):
        self.dragging = True
        self._update_preview(event)

    def on_left_drag(self, event):
        if not self.dragging:
            return
        self._update_preview(event)

    def on_left_release(self, event):
        if not self.dragging:
            return
        self.dragging = False
        self._hide_magnifier()
        if not self.preview_point:
            return
        if not self.current_entry:
            self.preview_point = None
            return
        self.current_entry["annotations"].append(
            {"x": self.preview_point["x"], "y": self.preview_point["y"]}
        )
        self.preview_point = None
        self._update_annotation_label()
        self._update_stats_label()
        self._display_current_frame()

    def _update_preview(self, event):
        if not self.current_entry:
            return
        frame_x = event.x / max(1e-6, self.scale_x)
        frame_y = event.y / max(1e-6, self.scale_y)
        frame = self.current_entry["frame"]
        h, w = frame.shape[:2]
        frame_x = float(np.clip(frame_x, 0.0, w - 1e-6))
        frame_y = float(np.clip(frame_y, 0.0, h - 1e-6))
        self.preview_point = {"x": frame_x, "y": frame_y}
        self._display_current_frame()
        self._show_magnifier(frame_x, frame_y, event.x_root, event.y_root)

    def on_right_press(self, event):
        if not self.current_entry:
            return
        self.drawing_rect = True
        frame_x = event.x / max(1e-6, self.scale_x)
        frame_y = event.y / max(1e-6, self.scale_y)
        frame = self.current_entry["frame"]
        h, w = frame.shape[:2]
        frame_x = float(np.clip(frame_x, 0.0, w - 1e-6))
        frame_y = float(np.clip(frame_y, 0.0, h - 1e-6))
        self.rect_start = {"x": frame_x, "y": frame_y}
        self.rect_preview = {"x0": frame_x, "y0": frame_y, "x1": frame_x, "y1": frame_y}
        self._display_current_frame()

    def on_right_drag(self, event):
        if not self.drawing_rect or not self.rect_start:
            return
        frame_x = event.x / max(1e-6, self.scale_x)
        frame_y = event.y / max(1e-6, self.scale_y)
        frame = self.current_entry["frame"]
        h, w = frame.shape[:2]
        frame_x = float(np.clip(frame_x, 0.0, w - 1e-6))
        frame_y = float(np.clip(frame_y, 0.0, h - 1e-6))
        x0 = min(self.rect_start["x"], frame_x)
        y0 = min(self.rect_start["y"], frame_y)
        x1 = max(self.rect_start["x"], frame_x)
        y1 = max(self.rect_start["y"], frame_y)
        self.rect_preview = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
        self._display_current_frame()

    def on_right_release(self, event):
        if not self.drawing_rect or not self.rect_preview:
            return
        self.drawing_rect = False
        if not self.current_entry:
            self.rect_start = None
            self.rect_preview = None
            return
        rect = {
            "x0": self.rect_preview["x0"],
            "y0": self.rect_preview["y0"],
            "x1": self.rect_preview["x1"],
            "y1": self.rect_preview["y1"],
        }
        if "negative_rects" not in self.current_entry:
            self.current_entry["negative_rects"] = []
        self.current_entry["negative_rects"].append(rect)
        self.rect_start = None
        self.rect_preview = None
        self._update_annotation_label()
        self._update_stats_label()
        self._display_current_frame()

    def _show_magnifier(self, frame_x, frame_y, root_x, root_y):
        if self.magnifier_window is None:
            self.magnifier_window = tk.Toplevel(self.root)
            self.magnifier_window.overrideredirect(True)
            self.magnifier_window.attributes("-topmost", True)
            self.magnifier_label = ttk.Label(
                self.magnifier_window, borderwidth=1, relief="solid"
            )
            self.magnifier_label.pack()
        size = max(4, self.magnifier_size // self.magnifier_zoom)
        patch, _ = self._extract_magnifier_patch(frame_x, frame_y, size)
        if patch is None:
            self._hide_magnifier()
            return
        enlarged = cv2.resize(
            patch,
            (self.magnifier_size, self.magnifier_size),
            interpolation=cv2.INTER_NEAREST,
        )
        rgb = cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        half = self.magnifier_size // 2
        draw.line((half - 10, half, half + 10, half), fill="lime", width=2)
        draw.line((half, half - 10, half, half + 10), fill="lime", width=2)
        self.magnifier_image = ImageTk.PhotoImage(image)
        self.magnifier_label.config(image=self.magnifier_image)
        offset = 20
        self.magnifier_window.geometry(
            f"+{root_x + offset}+{root_y + offset}"
        )
        self.magnifier_window.deiconify()

    def _hide_magnifier(self):
        if self.magnifier_window:
            self.magnifier_window.withdraw()

    def _extract_magnifier_patch(self, frame_x, frame_y, size):
        if not self.current_entry:
            return None, None
        frame = self.current_entry["frame"]
        h, w = frame.shape[:2]
        pad = size * 2
        padded = cv2.copyMakeBorder(
            frame, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT_101
        )
        cx = frame_x + pad
        cy = frame_y + pad
        half = size // 2
        x0 = int(round(cx - half))
        y0 = int(round(cy - half))
        patch = padded[y0 : y0 + size, x0 : x0 + size]
        return patch, None

    def _load_existing_annotations(self):
        output_path = "annotations.json"
        if not os.path.exists(output_path):
            return
        
        try:
            with open(output_path, "r") as f:
                data = json.load(f)
            
            # Load frames from all videos in our video list
            for video in data.get("videos", []):
                video_path = video.get("video_path", "")
                abs_video_path = os.path.abspath(video_path)
                
                # Check if this video is in our list
                if abs_video_path not in self.video_paths:
                    continue
                
                # Get total frames for this video
                try:
                    total_frames = get_total_frames(abs_video_path)
                except:
                    total_frames = 0
                
                for frame_data in video.get("frames", []):
                    frame_idx = frame_data["frame_idx"]
                    try:
                        frame = load_frame(abs_video_path, frame_idx)
                    except:
                        continue
                    
                    h, w = frame.shape[:2]
                    
                    annotations = []
                    for cross in frame_data.get("crosses", []):
                        annotations.append({
                            "x": float(cross["x"] * w),
                            "y": float(cross["y"] * h)
                        })
                    
                    negative_rects = []
                    for rect in frame_data.get("negative_rects", []):
                        negative_rects.append({
                            "x0": float(rect["x0"] * w),
                            "y0": float(rect["y0"] * h),
                            "x1": float(rect["x1"] * w),
                            "y1": float(rect["y1"] * h)
                        })
                    
                    entry = {
                        "video_path": abs_video_path,
                        "frame_idx": frame_idx,
                        "frame": frame,
                        "total_frames": total_frames,
                        "annotations": annotations,
                        "negative_rects": negative_rects
                    }
                    self.history.append(entry)
            
            if self.history:
                self.history_idx = 0
                self._set_current_entry(self.history[0])
                
        except Exception as e:
            print(f"[Warning] Could not load annotations: {e}")

    def export_annotations(self):
        # Group frames by video
        video_frames = {}
        for entry in self.history:
            video_path = entry["video_path"]
            if video_path not in video_frames:
                video_frames[video_path] = []
            
            frame = entry["frame"]
            h, w = frame.shape[:2]
            
            normalized_crosses = []
            for ann in entry["annotations"]:
                normalized_crosses.append({
                    "x": float(ann["x"] / w),
                    "y": float(ann["y"] / h)
                })
            
            normalized_rects = []
            for rect in entry.get("negative_rects", []):
                normalized_rects.append({
                    "x0": float(rect["x0"] / w),
                    "y0": float(rect["y0"] / h),
                    "x1": float(rect["x1"] / w),
                    "y1": float(rect["y1"] / h)
                })
            
            frame_data = {
                "frame_idx": entry["frame_idx"],
                "crosses": normalized_crosses,
                "negative_rects": normalized_rects
            }
            video_frames[video_path].append(frame_data)
        
        # Load existing JSON if present
        output_path = "annotations.json"
        existing_data = {"videos": []}
        if os.path.exists(output_path):
            try:
                with open(output_path, "r") as f:
                    existing_data = json.load(f)
            except:
                pass
        
        # Merge: update existing videos or add new ones
        existing_videos = {v["video_path"]: v for v in existing_data.get("videos", [])}
        for video_path, frames in video_frames.items():
            existing_videos[video_path] = {
                "video_path": video_path,
                "frames": frames
            }
        
        export_data = {
            "videos": list(existing_videos.values())
        }
        
        try:
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)
            messagebox.showinfo("Export successful", f"Annotations saved to {output_path}")
        except Exception as e:
            messagebox.showerror("Export failed", f"Could not save file: {e}")



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


def choose_random_frame_multi(video_paths):
    """Choose a random video and a random frame from it."""
    if not video_paths:
        raise ValueError("No video paths provided")
    
    video_path = random.choice(video_paths)
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

    return video_path, frame_idx, frame, total_frames


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
        "--videos",
        nargs="+",
        default=["video.mp4"],
        help="Paths to one or more video files to sample from.",
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
    viewer = RandomPatchViewer(root, args.videos, args.patch_size)
    root.mainloop()


if __name__ == "__main__":
    main()

