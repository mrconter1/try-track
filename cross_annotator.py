import argparse
import os
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import random
import json

class CrossAnnotator:
    def __init__(self, root, video_path):
        self.root = root
        self.video_path = os.path.abspath(video_path)

        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open video file: {self.video_path}")
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.root.destroy()
            return

        self.current_frame_idx = -1
        self.current_frame = None
        self.active_region_frame = None
        self.photo_image = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.annotations = {}
        self.current_region_idx = -1

        self.is_dragging = False
        self.current_drag_point = None
        self.drag_start_pos = None

        self.magnifier_window = None
        self.magnifier_canvas = None
        self.magnifier_size = 160
        self.magnifier_zoom = 4

        self.root.title("Cross Annotator")
        self._build_ui()
        self._load_existing_annotations()
        
        self.load_frame_and_region(0, new_region=True)

    def _load_existing_annotations(self):
        self.annotations_path = "cross_annotations.json"
        if not os.path.exists(self.annotations_path):
            return

        try:
            with open(self.annotations_path, 'r') as f:
                data = json.load(f)
            
            video_data = next((v for v in data.get("videos", []) if os.path.abspath(v["video_path"]) == self.video_path), None)
            
            if video_data:
                for frame_info in video_data.get("frames", []):
                    frame_idx = frame_info["frame_idx"]
                    loaded_regions = frame_info.get("regions", [])
                    
                    processed_regions = []
                    for region_data in loaded_regions:
                        rect = region_data.get("rect")
                        if not rect: continue
                        
                        _, _, w, h = rect
                        
                        denormalized_crosses = []
                        for cross in region_data.get("crosses", []):
                            px = cross["x"] * w
                            py = cross["y"] * h
                            denormalized_crosses.append([px, py])

                        processed_regions.append({
                            "rect": rect,
                            "crosses": denormalized_crosses
                        })

                    if processed_regions:
                        self.annotations[frame_idx] = {"regions": processed_regions}
            
            print(f"[Info] Loaded annotations for {len(self.annotations)} frames.")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Warning] Could not parse annotations file: {e}")

    def _build_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(main_frame, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        sidebar_frame = ttk.Frame(main_frame)
        sidebar_frame.grid(row=0, column=1, sticky="ns", pady=5, padx=5)

        # --- Navigation Controls ---
        nav_frame = ttk.LabelFrame(sidebar_frame, text="Navigation")
        nav_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10), anchor="n")

        self.prev_button = ttk.Button(nav_frame, text="<< Prev (A)", command=self.prev_frame)
        self.prev_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.next_button = ttk.Button(nav_frame, text="Next (D) >>", command=self.next_frame)
        self.next_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))

        self.new_region_button = ttk.Button(nav_frame, text="New Random Region (N)", command=lambda: self.load_frame_and_region(self.current_frame_idx, new_region=True))
        self.new_region_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))

        # --- Info Labels ---
        info_frame = ttk.LabelFrame(sidebar_frame, text="Info")
        info_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10), anchor="n")

        self.frame_label = ttk.Label(info_frame, text="Frame: 0 / 0")
        self.frame_label.pack(side=tk.TOP, anchor="w", padx=5, pady=2)
        
        self.crosses_label = ttk.Label(info_frame, text="Crosses in region: 0")
        self.crosses_label.pack(side=tk.TOP, anchor="w", padx=5, pady=2)

        # --- Action Controls ---
        actions_frame = ttk.LabelFrame(sidebar_frame, text="Actions")
        actions_frame.pack(side=tk.TOP, fill=tk.X, anchor="n")
        
        self.undo_button = ttk.Button(actions_frame, text="Undo (Ctrl+Z)", command=self.undo_last_cross)
        self.undo_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.export_button = ttk.Button(actions_frame, text="Export Annotations", command=self.export_annotations)
        self.export_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))

        self.root.bind("<Configure>", self.on_resize)
        self.root.bind("<Control-z>", lambda e: self.undo_last_cross())
        self.root.bind("<a>", lambda e: self.prev_frame())
        self.root.bind("<d>", lambda e: self.next_frame())
        self.root.bind("<n>", lambda e: self.load_frame_and_region(self.current_frame_idx, new_region=True))
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def load_frame_and_region(self, frame_idx, region_idx=None, new_region=False):
        if not (0 <= frame_idx < self.total_frames):
            return

        self.current_frame_idx = frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            self.current_frame = None
            return
            
        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if new_region:
            self._select_new_random_region()
        else:
            if self.current_frame_idx not in self.annotations or not self.annotations[self.current_frame_idx]["regions"]:
                self._select_new_random_region()
            else:
                self.current_region_idx = region_idx if region_idx is not None else 0
                self.display_region()

    def _select_new_random_region(self):
        if self.current_frame is None:
            return

        h, w = self.current_frame.shape[:2]
        region_size = 500
        if w < region_size or h < region_size:
            x, y, rw, rh = 0, 0, w, h
        else:
            x = random.randint(0, w - region_size)
            y = random.randint(0, h - region_size)
            rw, rh = region_size, region_size
        
        new_region = {"rect": [x, y, rw, rh], "crosses": []}

        if self.current_frame_idx not in self.annotations:
            self.annotations[self.current_frame_idx] = {"regions": []}
        
        self.annotations[self.current_frame_idx]["regions"].append(new_region)
        self.current_region_idx = len(self.annotations[self.current_frame_idx]["regions"]) - 1
        
        self.display_region()

    def display_region(self):
        if self.current_frame is None or self.current_region_idx == -1:
            return

        try:
            region_data = self.annotations[self.current_frame_idx]["regions"][self.current_region_idx]
            x, y, w, h = region_data["rect"]
            self.active_region_frame = self.current_frame[y:y+h, x:x+w]
        except (KeyError, IndexError):
            self.canvas.delete("all")
            self.update_info_labels()
            return
        
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2: return

        img_h, img_w = self.active_region_frame.shape[:2]
        
        scale = min(canvas_w / img_w, canvas_h / img_h)
        disp_w, disp_h = int(img_w * scale), int(img_h * scale)
        
        self.scale, self.offset_x, self.offset_y = scale, (canvas_w - disp_w) // 2, (canvas_h - disp_h) // 2

        resized = cv2.resize(self.active_region_frame, (disp_w, disp_h))
        self.photo_image = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.photo_image)

        self.draw_annotations()
        self.update_info_labels()

    def draw_annotations(self):
        if self.current_frame_idx in self.annotations and self.current_region_idx != -1:
            try:
                region = self.annotations[self.current_frame_idx]["regions"][self.current_region_idx]
                crosses = region.get("crosses", [])
                for cross in crosses:
                    self.draw_cross(cross, "red")
            except IndexError:
                pass
    
    def draw_cross(self, point, color):
        canvas_p = self.frame_to_canvas(point)
        if canvas_p:
            x, y = canvas_p
            size = 8
            self.canvas.create_line(x - size, y, x + size, y, fill=color, width=2)
            self.canvas.create_line(x, y - size, x, y + size, fill=color, width=2)

    def on_press(self, event):
        frame_coords = self.canvas_to_frame((event.x, event.y))
        if frame_coords is None: return

        self.is_dragging = True
        self.drag_start_pos = frame_coords
        self.current_drag_point = list(frame_coords)
        self.canvas.config(cursor="none")
        self._show_magnifier(event)
        self.canvas.delete("drag_marker")
        canvas_coords = (event.x, event.y)
        x, y = canvas_coords
        size = 8
        self.canvas.create_line(x - size, y, x + size, y, fill="yellow", width=2, tags="drag_marker")
        self.canvas.create_line(x, y - size, x, y + size, fill="yellow", width=2, tags="drag_marker")

    def on_drag(self, event):
        if not self.is_dragging: return
        
        self._update_magnifier(event)
        
        frame_coords = self.canvas_to_frame((event.x, event.y))
        if frame_coords:
            self.current_drag_point = list(frame_coords)

        self.canvas.delete("drag_marker")
        canvas_coords = (event.x, event.y)
        x, y = canvas_coords
        size = 8
        self.canvas.create_line(x - size, y, x + size, y, fill="yellow", width=2, tags="drag_marker")
        self.canvas.create_line(x, y - size, x, y + size, fill="yellow", width=2, tags="drag_marker")

    def on_release(self, event):
        if not self.is_dragging: return
        
        self._hide_magnifier()
        self.canvas.config(cursor="")
        self.is_dragging = False
        self.canvas.delete("drag_marker")

        final_coords = self.current_drag_point
        if final_coords:
            if self.current_frame_idx not in self.annotations:
                self._select_new_random_region()
            
            self.annotations[self.current_frame_idx]["regions"][self.current_region_idx]["crosses"].append(final_coords)

        self.display_region()

    def prev_frame(self):
        self._hide_magnifier()
        self.load_frame_and_region(self.current_frame_idx - 1)

    def next_frame(self):
        self._hide_magnifier()
        self.load_frame_and_region(self.current_frame_idx + 1)

    def undo_last_cross(self):
        if self.current_frame_idx in self.annotations and self.current_region_idx != -1:
            try:
                crosses = self.annotations[self.current_frame_idx]["regions"][self.current_region_idx].get("crosses", [])
                if crosses:
                    crosses.pop()
                    self.display_region()
            except IndexError:
                pass
    
    def update_info_labels(self):
        self.frame_label.config(text=f"Frame: {self.current_frame_idx} / {self.total_frames - 1}")
        num_crosses = 0
        if self.current_frame_idx in self.annotations and self.current_region_idx != -1:
            try:
                num_crosses = len(self.annotations[self.current_frame_idx]["regions"][self.current_region_idx].get("crosses", []))
            except IndexError:
                pass
        self.crosses_label.config(text=f"Crosses in region: {num_crosses}")

    def on_resize(self, event):
        self._hide_magnifier()
        self.display_region()

    def _show_magnifier(self, event):
        if self.magnifier_window: self._hide_magnifier()
        self.magnifier_window = tk.Toplevel(self.root)
        self.magnifier_window.overrideredirect(True)
        self.magnifier_canvas = tk.Canvas(self.magnifier_window, width=self.magnifier_size, height=self.magnifier_size)
        self.magnifier_canvas.pack()
        self._update_magnifier(event)

    def _hide_magnifier(self):
        if self.magnifier_window:
            self.magnifier_window.destroy()
            self.magnifier_window = None

    def _update_magnifier(self, event):
        if not self.magnifier_window or self.active_region_frame is None: return

        half_mag_size = self.magnifier_size // 2
        self.magnifier_window.geometry(f"+{event.x_root - half_mag_size}+{event.y_root - half_mag_size}")
        
        frame_coords = self.canvas_to_frame((event.x, event.y))
        
        zoomed_patch = np.zeros((self.magnifier_size, self.magnifier_size, 3), dtype=np.uint8)

        if frame_coords is not None:
            fx, fy = frame_coords
            
            # Create an affine transformation matrix for sub-pixel zoom and pan.
            # This maps the cursor's precise float location (fx, fy) to the magnifier's center.
            M = np.float32([
                [self.magnifier_zoom, 0, self.magnifier_size/2 - self.magnifier_zoom * fx],
                [0, self.magnifier_zoom, self.magnifier_size/2 - self.magnifier_zoom * fy]
            ])
            
            # Apply the transformation. INTER_NEAREST preserves the sharp pixel look while the
            # matrix provides the smooth sub-pixel panning. BORDER_CONSTANT creates the black
            # background for out-of-bounds areas.
            zoomed_patch = cv2.warpAffine(
                self.active_region_frame,
                M,
                (self.magnifier_size, self.magnifier_size),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

        # Draw a central crosshair on the magnifier
        m_center = self.magnifier_size // 2
        size = 8
        cv2.line(zoomed_patch, (m_center - size, m_center), (m_center + size, m_center), (255, 0, 255), 1)
        cv2.line(zoomed_patch, (m_center, m_center - size), (m_center, m_center + size), (255, 0, 255), 1)
        
        self.magnifier_photo = ImageTk.PhotoImage(Image.fromarray(zoomed_patch))
        self.magnifier_canvas.create_image(0, 0, anchor="nw", image=self.magnifier_photo)

    def frame_to_canvas(self, frame_coords):
        x, y = frame_coords
        canvas_x = self.offset_x + x * self.scale
        canvas_y = self.offset_y + y * self.scale
        return canvas_x, canvas_y

    def canvas_to_frame(self, canvas_coords):
        canvas_x, canvas_y = canvas_coords
        if self.active_region_frame is None: return None
        
        img_h, img_w = self.active_region_frame.shape[:2]
        disp_w, disp_h = int(img_w * self.scale), int(img_h * self.scale)

        if not (self.offset_x <= canvas_x <= self.offset_x + disp_w and
                self.offset_y <= canvas_y <= self.offset_y + disp_h):
            return None

        frame_x = (canvas_x - self.offset_x) / self.scale
        frame_y = (canvas_y - self.offset_y) / self.scale
        return frame_x, frame_y

    def export_annotations(self):
        output_path = "cross_annotations.json"
        existing_data = {"videos": []}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                pass
        
        video_entry = next((v for v in existing_data["videos"] if os.path.abspath(v["video_path"]) == self.video_path), None)
        if not video_entry:
            video_entry = {"video_path": self.video_path, "frames": []}
            existing_data["videos"].append(video_entry)

        existing_frames = {f["frame_idx"]: f for f in video_entry["frames"]}

        for frame_idx, data in self.annotations.items():
            if data.get("regions"):
                valid_regions = []
                for r in data["regions"]:
                    if r.get("crosses"):
                        _, _, w, h = r["rect"]
                        normalized_crosses = [{"x": c[0]/w, "y": c[1]/h} for c in r["crosses"]]
                        valid_regions.append({"rect": r["rect"], "crosses": normalized_crosses})
                
                if valid_regions:
                    existing_frames[frame_idx] = {"frame_idx": frame_idx, "regions": valid_regions}
        
        video_entry["frames"] = sorted(existing_frames.values(), key=lambda x: x["frame_idx"])

        try:
            with open(output_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
            messagebox.showinfo("Export successful", f"Annotations saved to {output_path}")
        except Exception as e:
            messagebox.showerror("Export failed", f"Could not save annotations: {e}")

    def on_close(self):
        if messagebox.askokcancel("Save on exit", "Do you want to export annotations before closing?"):
            self.export_annotations()
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser(description="Annotate crosses on video frames.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file.")
    args = parser.parse_args()

    root = tk.Tk()
    try:
        root.state('zoomed')
    except tk.TclError:
        root.geometry("1200x800")
        
    app = CrossAnnotator(root, args.video)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
