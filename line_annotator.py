import argparse
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np # Added for padding in magnifier
import random # Added for random region selection

class LineAnnotator:
    def __init__(self, root, video_path):
        self.root = root
        self.video_path = os.path.abspath(video_path)
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open video file: {self.video_path}")
            self.root.destroy()
            return
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0
        self.current_frame = None
        self.photo_image = None

        # Data structure for annotations
        self.annotations = {} # {frame_idx: {"regions": [{"rect": [x,y,w,h], "lines": [...]}]}}
        self.current_region_idx = -1

        # State for drawing a new line
        self.first_point = None
        self.is_dragging = False
        self.current_drag_item = None # Holds the ID of the circle or line being dragged

        # Magnifier state
        self.magnifier_window = None
        self.magnifier_canvas = None
        self.magnifier_size = 160 # Display size of magnifier
        self.magnifier_zoom = 4 # Zoom level

        self.root.title("Line Annotator")
        self._build_ui()
        self._load_existing_annotations()
        
        self.load_frame_and_region(0, new_region=True)

    def _load_existing_annotations(self):
        output_path = "line_annotations.json"
        if not os.path.exists(output_path):
            return

        try:
            with open(output_path, "r") as f:
                data = json.load(f)

            # Find data for the current video
            video_data = None
            for v in data.get("videos", []):
                if os.path.abspath(v.get("video_path", "")) == self.video_path:
                    video_data = v
                    break
            
            if video_data:
                for frame_info in video_data.get("frames", []):
                    frame_idx = frame_info["frame_idx"]
                    regions = frame_info.get("regions", [])
                    if regions:
                        self.annotations[frame_idx] = {"regions": regions}
            
            print(f"[Info] Loaded annotations for {len(self.annotations)} frames.")

        except Exception as e:
            messagebox.showwarning("Load Warning", f"Could not parse annotations file: {e}")

    def _build_ui(self):
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        # Main canvas for video frame
        self.canvas = tk.Canvas(container, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=10, pady=10)

        # Control panel on the right
        controls_frame = ttk.Frame(container, padding=10)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Frame navigation
        nav_frame = ttk.LabelFrame(controls_frame, text="Navigation", padding=10)
        nav_frame.pack(fill=tk.X, pady=5)

        self.prev_button = ttk.Button(nav_frame, text="<< Prev (A)", command=self.prev_frame)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(nav_frame, text="Next (D) >>", command=self.next_frame)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.new_region_button = ttk.Button(nav_frame, text="New Random Region (N)", command=self.select_new_random_region)
        self.new_region_button.pack(side=tk.LEFT, padx=5)

        self.frame_label = ttk.Label(nav_frame, text="Frame: 0 / 0", width=20)
        self.frame_label.pack(side=tk.LEFT, padx=5)

        # Annotation controls
        annot_frame = ttk.LabelFrame(controls_frame, text="Annotation", padding=10)
        annot_frame.pack(fill=tk.X, pady=5)

        self.undo_button = ttk.Button(annot_frame, text="Undo Last Line (Ctrl+Z)", command=self.undo_last_line)
        self.undo_button.pack(fill=tk.X, pady=2)

        self.clear_button = ttk.Button(annot_frame, text="Clear All on Frame", command=self.clear_frame_annotations)
        self.clear_button.pack(fill=tk.X, pady=2)

        self.lines_label = ttk.Label(annot_frame, text="Lines on frame: 0")
        self.lines_label.pack(pady=5)

        # Export
        export_frame = ttk.LabelFrame(controls_frame, text="Data", padding=10)
        export_frame.pack(fill=tk.X, pady=5)

        self.export_button = ttk.Button(export_frame, text="Export Annotations", command=self.export_annotations)
        self.export_button.pack(fill=tk.X)

        # Bindings
        self.root.bind("<Configure>", self.on_resize)
        self.root.bind("<Control-z>", lambda e: self.undo_last_line())
        self.root.bind("<a>", lambda e: self.prev_frame())
        self.root.bind("<d>", lambda e: self.next_frame())
        self.root.bind("<n>", lambda e: self.select_new_random_region())
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
            self.select_new_random_region(add_to_history=False)
        else:
            if self.current_frame_idx not in self.annotations or not self.annotations[self.current_frame_idx]["regions"]:
                self.select_new_random_region(add_to_history=False)
            else:
                self.current_region_idx = region_idx if region_idx is not None else 0
                self.display_region()

    def select_new_random_region(self, add_to_history=True):
        if self.current_frame is None:
            return

        h, w = self.current_frame.shape[:2]
        region_size = 300
        if w < region_size or h < region_size:
            messagebox.showwarning("Warning", f"Frame is smaller than {region_size}x{region_size}, using full frame.")
            x, y = 0, 0
            rw, rh = w, h
        else:
            x = random.randint(0, w - region_size)
            y = random.randint(0, h - region_size)
            rw, rh = region_size, region_size
        
        new_region = {"rect": [x, y, rw, rh], "lines": []}

        if add_to_history:
            if self.current_frame_idx not in self.annotations:
                self.annotations[self.current_frame_idx] = {"regions": []}
            self.annotations[self.current_frame_idx]["regions"].append(new_region)
            self.current_region_idx = len(self.annotations[self.current_frame_idx]["regions"]) - 1
        else:
            # Temporarily use this region without saving it yet
            if self.current_frame_idx not in self.annotations:
                self.annotations[self.current_frame_idx] = {"regions": [new_region]}
            else:
                 self.annotations[self.current_frame_idx]["regions"].insert(0, new_region)
            self.current_region_idx = 0

        self.display_region()

    def display_region(self):
        if self.current_frame is None or self.current_region_idx == -1:
            return

        # Extract the region
        try:
            region_data = self.annotations[self.current_frame_idx]["regions"][self.current_region_idx]
            x, y, w, h = region_data["rect"]
            self.active_region_frame = self.current_frame[y:y+h, x:x+w]
        except (KeyError, IndexError):
            self.canvas.delete("all")
            self.update_info_labels()
            return
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 2 or canvas_h < 2:
            return

        img_h, img_w = self.active_region_frame.shape[:2]
        
        scale = min(canvas_w / img_w, canvas_h / img_h)
        disp_w, disp_h = int(img_w * scale), int(img_h * scale)
        
        self.scale = scale
        self.offset_x = (canvas_w - disp_w) // 2
        self.offset_y = (canvas_h - disp_h) // 2

        resized = cv2.resize(self.active_region_frame, (disp_w, disp_h))
        
        self.photo_image = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.photo_image)

        self.draw_annotations()
        
        if self.first_point:
            canvas_p1 = self.frame_to_canvas(self.first_point)
            x, y = canvas_p1
            size = 6
            self.canvas.create_line(x - size, y, x + size, y, fill="lime", width=2)
            self.canvas.create_line(x, y - size, x, y + size, fill="lime", width=2)

        self.update_info_labels()

    def draw_annotations(self):
        # First, draw the originally placed line segments faintly
        if self.current_frame_idx in self.annotations and self.current_region_idx != -1:
            try:
                lines = self.annotations[self.current_frame_idx]["regions"][self.current_region_idx].get("lines", [])
                for line in lines:
                    p1, p2 = line
                    canvas_p1 = self.frame_to_canvas(p1)
                    canvas_p2 = self.frame_to_canvas(p2)
                    self.canvas.create_line(canvas_p1, canvas_p2, fill="#444444", width=1, dash=(2,4))
            except IndexError:
                pass # Region might not exist yet

        if self.current_frame_idx in self.annotations and self.current_region_idx != -1:
            try:
                region_data = self.annotations[self.current_frame_idx]["regions"][self.current_region_idx]
                lines = region_data.get("lines", [])
                if not lines:
                    return
                
                _, _, w, h = region_data["rect"]

                # 1. Draw clipped, extended lines
                for line in lines:
                    p1, p2 = np.array(line[0]), np.array(line[1])
                    
                    direction = p2 - p1
                    if np.linalg.norm(direction) < 1e-6: continue
                    
                    p_start = p1 - 10000 * direction
                    p_end = p1 + 10000 * direction

                    clipped_points = self.clip_line_to_frame(p_start, p_end, (0, 0, w, h))
                    if clipped_points:
                        start_point, end_point = clipped_points
                        canvas_p1 = self.frame_to_canvas(start_point)
                        canvas_p2 = self.frame_to_canvas(end_point)
                        self.canvas.create_line(canvas_p1, canvas_p2, fill="cyan", width=2)
                
                # 2. Calculate and draw intersections
                intersections = []
                for i in range(len(lines)):
                    for j in range(i + 1, len(lines)):
                        p1, p2 = lines[i]
                        p3, p4 = lines[j]
                        x1, y1 = p1; x2, y2 = p2
                        x3, y3 = p3; x4, y4 = p4

                        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                        if abs(den) < 1e-6: continue

                        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
                        ix = x1 + t * (x2 - x1)
                        iy = y1 + t * (y2 - y1)
                        
                        if 0 <= ix <= w and 0 <= iy <= h:
                            intersections.append([ix, iy])

                for point in intersections:
                    canvas_p = self.frame_to_canvas(point)
                    x, y = canvas_p
                    self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="red", outline="red")
            except IndexError:
                pass

    def on_press(self, event):
        frame_coords = self.canvas_to_frame((event.x, event.y))
        if frame_coords is None: return

        self.is_dragging = True
        self.canvas.config(cursor="none")
        self._show_magnifier(event)

        if self.first_point is None:
            # Starting to place the first point
            canvas_coords = (event.x, event.y)
            x, y = canvas_coords
            size = 6
            self.current_drag_item = [
                self.canvas.create_line(x - size, y, x + size, y, fill="yellow", width=2, tags="drag_marker"),
                self.canvas.create_line(x, y - size, x, y + size, fill="yellow", width=2, tags="drag_marker")
            ]
        else:
            # Starting to place the second point
            canvas_p1 = self.frame_to_canvas(self.first_point)
            self.current_drag_item = self.canvas.create_line(canvas_p1, (event.x, event.y), fill="lime", dash=(4, 2), tags="drag_marker")

    def on_drag(self, event):
        if not self.is_dragging or not self.current_drag_item:
            return

        self._update_magnifier(event)

        if self.first_point is None:
            # Dragging the first point's marker (a cross made of two lines)
            x, y = event.x, event.y
            size = 6
            self.canvas.coords(self.current_drag_item[0], x - size, y, x + size, y)
            self.canvas.coords(self.current_drag_item[1], x, y - size, x, y + size)
        else:
            # Dragging the end of the preview line
            canvas_p1 = self.frame_to_canvas(self.first_point)
            self.canvas.coords(self.current_drag_item, canvas_p1[0], canvas_p1[1], event.x, event.y)

    def on_release(self, event):
        if not self.is_dragging:
            return
        
        self._hide_magnifier()
        self.canvas.config(cursor="")
        self.is_dragging = False
        frame_coords = self.canvas_to_frame((event.x, event.y))
        if frame_coords is None:
            self.canvas.delete("drag_marker")
            self.current_drag_item = None
            return

        if self.first_point is None:
            # Finalized the first point
            self.first_point = frame_coords
            self.canvas.delete("drag_marker") # remove yellow preview
            self.display_region() # Redraw to show permanent green marker
        else:
            # Finalized the second point
            second_point = frame_coords
            
            if self.current_frame_idx not in self.annotations:
                self.annotations[self.current_frame_idx] = {"regions": []}
            if not self.annotations[self.current_frame_idx]["regions"]:
                # This must be the first region for this frame
                self.select_new_random_region()

            self.annotations[self.current_frame_idx]["regions"][self.current_region_idx]["lines"].append([self.first_point, second_point])
            
            self.first_point = None
            self.canvas.delete("drag_marker")
            self.display_region()

    def on_mouse_move(self, event):
        pass # No longer needed for rubber-band, handled by on_drag

    def _show_magnifier(self, event):
        if self.magnifier_window:
            self._hide_magnifier()

        self.magnifier_window = tk.Toplevel(self.root)
        self.magnifier_window.overrideredirect(True) # Borderless
        
        self.magnifier_canvas = tk.Canvas(self.magnifier_window, width=self.magnifier_size, height=self.magnifier_size)
        self.magnifier_canvas.pack()
        
        self._update_magnifier(event)

    def _hide_magnifier(self):
        if self.magnifier_window:
            self.magnifier_window.destroy()
            self.magnifier_window = None
            self.magnifier_canvas = None
    
    def _update_magnifier(self, event):
        if not self.magnifier_window or self.current_frame is None:
            return

        # Center window on cursor
        half_mag_size = self.magnifier_size // 2
        self.magnifier_window.geometry(f"+{event.x_root - half_mag_size}+{event.y_root - half_mag_size}")
        
        # Get coordinates in original frame
        frame_coords = self.canvas_to_frame((event.x, event.y))
        
        # --- New logic for handling out-of-bounds ---
        patch_size = self.magnifier_size // self.magnifier_zoom
        half_patch = patch_size // 2
        
        # Create a black background for our patch
        patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)

        if frame_coords is not None:
            fx, fy = frame_coords
            h, w = self.current_frame.shape[:2]
            
            # Define the source rectangle in frame coordinates (can be out of bounds)
            src_x0, src_y0 = int(fx - half_patch), int(fy - half_patch)
            src_x1, src_y1 = src_x0 + patch_size, src_y0 + patch_size
            
            # Find the valid intersection of the source rect and the frame
            valid_src_x0 = max(0, src_x0)
            valid_src_y0 = max(0, src_y0)
            valid_src_x1 = min(w, src_x1)
            valid_src_y1 = min(h, src_y1)
            
            # If there is an intersection, copy the valid part
            if valid_src_x0 < valid_src_x1 and valid_src_y0 < valid_src_y1:
                # Region to copy from the source frame
                frame_part = self.current_frame[valid_src_y0:valid_src_y1, valid_src_x0:valid_src_x1]
                
                # Define where to paste this part onto our black background patch
                dest_x0 = valid_src_x0 - src_x0
                dest_y0 = valid_src_y0 - src_y0
                dest_x1 = dest_x0 + (valid_src_x1 - valid_src_x0)
                dest_y1 = dest_y0 + (valid_src_y1 - valid_src_y0)
                
                patch[dest_y0:dest_y1, dest_x0:dest_x1] = frame_part

        # Zoom the patch (which may be all or partially black)
        zoomed_patch = cv2.resize(patch, (self.magnifier_size, self.magnifier_size), interpolation=cv2.INTER_NEAREST)
        
        # Draw a central crosshair on the magnifier
        m_center = self.magnifier_size // 2
        size = 8
        cv2.line(zoomed_patch, (m_center - size, m_center), (m_center + size, m_center), (255, 0, 255), 1)
        cv2.line(zoomed_patch, (m_center, m_center - size), (m_center, m_center + size), (255, 0, 255), 1)

        # Display on magnifier canvas
        self.magnifier_photo = ImageTk.PhotoImage(Image.fromarray(zoomed_patch))
        self.magnifier_canvas.create_image(0, 0, anchor="nw", image=self.magnifier_photo)

    def prev_frame(self):
        self._hide_magnifier()
        self.first_point = None
        self.load_frame_and_region(self.current_frame_idx - 1)

    def next_frame(self):
        self._hide_magnifier()
        self.first_point = None
        self.load_frame_and_region(self.current_frame_idx + 1)

    def undo_last_line(self):
        # If currently drawing a line, undo just the first point
        if self.first_point:
            self.first_point = None
            self.canvas.delete("drag_marker")
            self.display_region()
            return

        if self.current_frame_idx in self.annotations and self.current_region_idx != -1:
            try:
                lines = self.annotations[self.current_frame_idx]["regions"][self.current_region_idx].get("lines", [])
                if lines:
                    lines.pop()
                    self.display_region()
            except IndexError:
                pass

    def clear_frame_annotations(self):
        self.first_point = None
        if self.current_frame_idx in self.annotations and self.current_region_idx != -1:
            try:
                self.annotations[self.current_frame_idx]["regions"][self.current_region_idx]["lines"] = []
                self.display_region()
            except IndexError:
                pass
    
    def update_info_labels(self):
        self.frame_label.config(text=f"Frame: {self.current_frame_idx} / {self.total_frames - 1}")
        num_lines = 0
        if self.current_frame_idx in self.annotations and self.current_region_idx != -1:
            try:
                num_lines = len(self.annotations[self.current_frame_idx]["regions"][self.current_region_idx].get("lines", []))
            except IndexError:
                pass
        self.lines_label.config(text=f"Lines on frame: {num_lines}")

    def on_resize(self, event):
        self._hide_magnifier()
        self.display_region()

    def clip_line_to_frame(self, p1, p2, frame_rect):
        """Clips a line segment to a rectangular area."""
        x1, y1 = p1
        x2, y2 = p2
        xmin, ymin, xmax, ymax = frame_rect

        # Using Cohen-Sutherland outcodes
        INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

        def compute_outcode(x, y):
            code = INSIDE
            if x < xmin: code |= LEFT
            elif x > xmax: code |= RIGHT
            if y < ymin: code |= BOTTOM
            elif y > ymax: code |= TOP
            return code

        outcode1 = compute_outcode(x1, y1)
        outcode2 = compute_outcode(x2, y2)
        
        while True:
            if not (outcode1 | outcode2): # Both points inside
                return [(x1, y1), (x2, y2)]
            elif outcode1 & outcode2: # Both points outside on the same side
                return None
            else:
                x, y = 0, 0
                outcode_out = outcode1 if outcode1 else outcode2

                if outcode_out & TOP:
                    x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
                    y = ymax
                elif outcode_out & BOTTOM:
                    x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
                    y = ymin
                elif outcode_out & RIGHT:
                    y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
                    x = xmax
                elif outcode_out & LEFT:
                    y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
                    x = xmin

                if outcode_out == outcode1:
                    x1, y1 = x, y
                    outcode1 = compute_outcode(x1, y1)
                else:
                    x2, y2 = x, y
                    outcode2 = compute_outcode(x2, y2)

    def frame_to_canvas(self, point):
        x, y = point
        canvas_x = self.offset_x + x * self.scale
        canvas_y = self.offset_y + y * self.scale
        return int(canvas_x), int(canvas_y)

    def canvas_to_frame(self, point):
        canvas_x, canvas_y = point
        
        # Check if click is inside the displayed image area
        if not (self.offset_x <= canvas_x <= self.offset_x + (self.active_region_frame.shape[1] * self.scale) and
                self.offset_y <= canvas_y <= self.offset_y + (self.active_region_frame.shape[0] * self.scale)):
            return None

        frame_x = (canvas_x - self.offset_x) / self.scale
        frame_y = (canvas_y - self.offset_y) / self.scale
        return [frame_x, frame_y]

    def export_annotations(self):
        output_path = "line_annotations.json"
        
        # Load existing data to merge
        all_data = {"videos": []}
        if os.path.exists(output_path):
            try:
                with open(output_path, "r") as f:
                    all_data = json.load(f)
            except json.JSONDecodeError:
                pass # Overwrite if file is corrupt
        
        # Find entry for current video or create it
        video_entry = None
        for v in all_data["videos"]:
            if os.path.abspath(v.get("video_path", "")) == self.video_path:
                video_entry = v
                break
        
        if not video_entry:
            video_entry = {"video_path": self.video_path, "frames": []}
            all_data["videos"].append(video_entry)
        
        # Create a map of existing frames for quick lookup
        existing_frames = {f["frame_idx"]: f for f in video_entry["frames"]}

        # Update or add frames from current session
        for frame_idx, data in self.annotations.items():
            if data.get("regions"):
                # Filter out regions with no lines
                valid_regions = [r for r in data["regions"] if r.get("lines")]
                if valid_regions:
                    existing_frames[frame_idx] = {"frame_idx": frame_idx, "regions": valid_regions}
        
        # Sort frames by index and update video entry
        video_entry["frames"] = sorted(existing_frames.values(), key=lambda f: f["frame_idx"])

        try:
            with open(output_path, "w") as f:
                json.dump(all_data, f, indent=2)
            messagebox.showinfo("Export Successful", f"Annotations saved to {output_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save annotations: {e}")
            
    def on_close(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser(description="Annotate lines on video frames.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file.")
    args = parser.parse_args()

    root = tk.Tk()
    try:
        root.state('zoomed')
    except tk.TclError:
        # Fallback for other OSes
        root.geometry("1200x800")
        
    app = LineAnnotator(root, args.video)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
