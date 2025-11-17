import argparse
import os
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import random
import json
import bisect

def get_video_props(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

class CrossAnnotator:
    def __init__(self, root, video_paths):
        self.root = root
        self.video_paths = [os.path.abspath(p) for p in video_paths]
        self.video_frame_counts = {path: get_video_props(path) for path in self.video_paths}
        
        # --- Proportional Sampling Setup ---
        self.cumulative_frames = []
        self.total_combined_frames = 0
        current_total = 0
        for path in self.video_paths:
            count = self.video_frame_counts.get(path, 0)
            current_total += count
            self.cumulative_frames.append(current_total)
        self.total_combined_frames = current_total
        # --- End Proportional Sampling ---

        self.current_video_path = None
        self.current_frame_idx = -1
        self.current_frame = None
        self.active_region_frame = None
        self.photo_image = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.annotations = {}
        self.history = [] # To store (video_path, frame_idx, region_idx) tuples
        self.history_idx = -1
        self.current_region_idx = -1

        self.is_dragging = False
        self.current_drag_point = None
        self.drag_start_pos = None

        self.magnifier_window = None
        self.magnifier_canvas = None
        self.magnifier_size = 200
        self.magnifier_zoom = 4

        self.root.title("Cross Annotator")
        self._build_ui()
        self._load_and_build_history()
        
        # The initial load_frame_and_region call is now handled by _load_and_build_history

    def _load_and_build_history(self):
        self._load_existing_annotations()
        
        # Build history from loaded annotations
        if self.annotations:
            sorted_videos = sorted(self.annotations.keys())
            for video_path in sorted_videos:
                sorted_frames = sorted(self.annotations[video_path].keys())
                for frame_idx in sorted_frames:
                    for region_idx in range(len(self.annotations[video_path][frame_idx]["regions"])):
                        self.history.append((video_path, frame_idx, region_idx))
        
        if self.history:
            self.history_idx = 0
            self._load_history_entry(self.history_idx)
        else:
            self._append_new_random_frame_entry()

    def _load_existing_annotations(self):
        self.annotations_path = "cross_annotations.json"
        if not os.path.exists(self.annotations_path):
            return

        try:
            with open(self.annotations_path, 'r') as f:
                data = json.load(f)
            
            for video_data in data.get("videos", []):
                video_path = os.path.abspath(video_data.get("video_path", ""))
                if video_path not in self.video_paths:
                    continue

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
                        if video_path not in self.annotations:
                            self.annotations[video_path] = {}
                        self.annotations[video_path][frame_idx] = {"regions": processed_regions}
            
            print(f"[Info] Loaded annotations for {sum(len(v) for v in self.annotations.values())} frames across {len(self.annotations)} videos.")

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

        self.prev_button = ttk.Button(nav_frame, text="<< Prev (A)", command=self.prev_entry)
        self.prev_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.next_button = ttk.Button(nav_frame, text="Next (D) >>", command=self.next_entry)
        self.next_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))

        self.new_region_button = ttk.Button(nav_frame, text="New Random Region (N)", command=self.next_entry)
        self.new_region_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))

        # --- Info Labels ---
        info_frame = ttk.LabelFrame(sidebar_frame, text="Info")
        info_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10), anchor="n")

        self.frame_label = ttk.Label(info_frame, text="Frame: ...\nRegion: ...", justify=tk.LEFT)
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
        self.root.bind("<a>", lambda e: self.prev_entry())
        self.root.bind("<d>", lambda e: self.next_entry())
        self.root.bind("<n>", lambda e: self.next_entry())
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def _load_history_entry(self, history_idx):
        if not (0 <= history_idx < len(self.history)):
            return

        self.history_idx = history_idx
        video_path, frame_idx, region_idx = self.history[self.history_idx]

        if self.current_video_path != video_path or self.current_frame_idx != frame_idx:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                self.current_frame = None
                return
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_video_path = video_path
            self.current_frame_idx = frame_idx
        
        self.current_region_idx = region_idx
        self.display_region()

    def _append_new_random_frame_entry(self):
        if self.total_combined_frames <= 0:
            messagebox.showerror("Error", "No frames found in any of the provided videos.")
            return

        # --- Proportional Sampling Logic ---
        # 1. Pick a random frame index from the combined total
        global_frame_idx = random.randint(0, self.total_combined_frames - 1)

        # 2. Find which video this global index falls into
        video_idx = bisect.bisect_left(self.cumulative_frames, global_frame_idx)
        video_path = self.video_paths[video_idx]

        # 3. Calculate the local frame index within that video
        previous_cumulative = self.cumulative_frames[video_idx - 1] if video_idx > 0 else 0
        frame_idx = global_frame_idx - previous_cumulative
        # --- End Proportional Sampling Logic ---
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"[Warning] Failed to read frame {frame_idx} from '{video_path}'.")
            return

        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_video_path = video_path
        self.current_frame_idx = frame_idx

        # Now, select a random region within this new frame
        h, w = self.current_frame.shape[:2]
        region_size = 500
        if w < region_size or h < region_size:
            x, y, rw, rh = 0, 0, w, h
        else:
            x = random.randint(0, w - region_size)
            y = random.randint(0, h - region_size)
            rw, rh = region_size, region_size
        
        new_region = {"rect": [x, y, rw, rh], "crosses": []}

        if self.current_video_path not in self.annotations:
            self.annotations[self.current_video_path] = {}
        if frame_idx not in self.annotations[self.current_video_path]:
            self.annotations[self.current_video_path][frame_idx] = {"regions": []}
        
        self.annotations[self.current_video_path][frame_idx]["regions"].append(new_region)
        new_region_idx = len(self.annotations[self.current_video_path][frame_idx]["regions"]) - 1
        
        # This handles the case where you go back, then forward in a new direction
        # It removes any "future" history that we are now branching away from.
        self.history = self.history[:self.history_idx + 1]

        self.history.append((self.current_video_path, frame_idx, new_region_idx))
        self.history_idx = len(self.history) - 1
        
        # We have all the info for the new state, so just display it directly
        self.current_region_idx = new_region_idx
        self.display_region()

    def display_region(self):
        if self.current_frame is None or self.current_region_idx == -1:
            return

        try:
            region_data = self.annotations[self.current_video_path][self.current_frame_idx]["regions"][self.current_region_idx]
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
        if self.current_video_path in self.annotations and self.current_frame_idx in self.annotations[self.current_video_path] and self.current_region_idx != -1:
            try:
                region = self.annotations[self.current_video_path][self.current_frame_idx]["regions"][self.current_region_idx]
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
            self.canvas.create_line(x - size, y, x + size, y, fill=color, width=1)
            self.canvas.create_line(x, y - size, x, y + size, fill=color, width=1)

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
        self.canvas.create_line(x - size, y, x + size, y, fill="yellow", width=1, tags="drag_marker")
        self.canvas.create_line(x, y - size, x, y + size, fill="yellow", width=1, tags="drag_marker")

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
        self.canvas.create_line(x - size, y, x + size, y, fill="yellow", width=1, tags="drag_marker")
        self.canvas.create_line(x, y - size, x, y + size, fill="yellow", width=1, tags="drag_marker")

    def on_release(self, event):
        if not self.is_dragging: return
        
        self._hide_magnifier()
        self.canvas.config(cursor="")
        self.is_dragging = False
        self.canvas.delete("drag_marker")

        final_coords = self.current_drag_point
        if final_coords:
            if self.current_video_path not in self.annotations:
                self.annotations[self.current_video_path] = {}
            if self.current_frame_idx not in self.annotations[self.current_video_path]:
                self.annotations[self.current_video_path][self.current_frame_idx] = {"regions": []}
                # Create a placeholder region if it somehow doesn't exist
                if not self.annotations[self.current_video_path][self.current_frame_idx]["regions"]:
                     self.annotations[self.current_video_path][self.current_frame_idx]["regions"].append({"rect": [0,0,500,500], "crosses":[]})
                self.current_region_idx = 0
            
            self.annotations[self.current_video_path][self.current_frame_idx]["regions"][self.current_region_idx]["crosses"].append(final_coords)

        self.display_region()

    def prev_entry(self):
        self._hide_magnifier()
        if self.history_idx > 0:
            self._load_history_entry(self.history_idx - 1)

    def next_entry(self):
        self._hide_magnifier()
        if self.history_idx < len(self.history) - 1:
            # We are in the middle of history, just move forward
            self._load_history_entry(self.history_idx + 1)
        else:
            # We are at the end, append a new random entry
            self._append_new_random_frame_entry()

    def prev_frame(self):
        self._hide_magnifier()
        self.prev_entry()

    def next_frame(self):
        self._hide_magnifier()
        self.next_entry()

    def undo_last_cross(self):
        if self.current_video_path in self.annotations and self.current_frame_idx in self.annotations[self.current_video_path] and self.current_region_idx != -1:
            try:
                crosses = self.annotations[self.current_video_path][self.current_frame_idx]["regions"][self.current_region_idx].get("crosses", [])
                if crosses:
                    crosses.pop()
                    self.display_region()
            except IndexError:
                pass
    
    def update_info_labels(self):
        total_regions_on_frame = 0
        if self.current_video_path in self.annotations and self.current_frame_idx in self.annotations[self.current_video_path]:
            total_regions_on_frame = len(self.annotations[self.current_video_path][self.current_frame_idx].get("regions", []))
        
        video_name = os.path.basename(self.current_video_path) if self.current_video_path else "N/A"
        total_frames = self.video_frame_counts.get(self.current_video_path, 0)

        video_text = f"Video: {video_name}"
        frame_text = f"Frame: {self.current_frame_idx} / {total_frames - 1}"
        region_text = f"Region: {self.current_region_idx + 1} / {total_regions_on_frame}"

        self.frame_label.config(text=f"{video_text}\n{frame_text}\n{region_text}")

        num_crosses = 0
        if self.current_video_path in self.annotations and self.current_frame_idx in self.annotations[self.current_video_path] and self.current_region_idx != -1:
            try:
                num_crosses = len(self.annotations[self.current_video_path][self.current_frame_idx]["regions"][self.current_region_idx].get("crosses", []))
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
        
        # Create a dictionary of video entries for easy lookup
        existing_videos_map = {os.path.abspath(v["video_path"]): v for v in existing_data["videos"]}

        # Iterate over all videos we've annotated in this session
        for video_path, frames_data in self.annotations.items():
            if video_path not in existing_videos_map:
                existing_videos_map[video_path] = {"video_path": video_path, "frames": []}
            
            video_entry = existing_videos_map[video_path]
            existing_frames = {f["frame_idx"]: f for f in video_entry["frames"]}

            for frame_idx, data in frames_data.items():
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

        existing_data["videos"] = list(existing_videos_map.values())

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
    parser.add_argument("--input", nargs='+', required=True, help="One or more paths to video files or directories containing videos.")
    args = parser.parse_args()

    video_paths = find_videos_in_paths(args.input)
    if not video_paths:
        messagebox.showerror("Error", "No video files found in the specified paths.")
        return

    root = tk.Tk()
    try:
        root.state('zoomed')
    except tk.TTclError:
        root.geometry("1200x800")
        
    app = CrossAnnotator(root, video_paths)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

def find_videos_in_paths(paths):
    video_files = []
    supported_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    for path in paths:
        path = os.path.abspath(path)
        if os.path.isfile(path):
            if os.path.splitext(path)[1].lower() in supported_extensions:
                video_files.append(path)
        elif os.path.isdir(path):
            for item in os.listdir(path):
                full_path = os.path.join(path, item)
                if os.path.isfile(full_path) and os.path.splitext(full_path)[1].lower() in supported_extensions:
                    video_files.append(full_path)
    
    print(f"Found {len(video_files)} videos to process.")
    return sorted(list(set(video_files)))


if __name__ == "__main__":
    main()
