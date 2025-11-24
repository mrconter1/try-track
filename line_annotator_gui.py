import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
import random
import bisect
import argparse
import sys
import numpy as np

def get_video_props(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

class RandomPatchViewer:
    def __init__(self, root, video_paths, patch_size=400):
        self.root = root
        self.video_paths = [os.path.abspath(p) for p in video_paths]
        self.patch_size = patch_size
        
        # Pre-calculate frame counts for proportional sampling
        self.video_frame_counts = {}
        print("Scanning videos...")
        for path in self.video_paths:
            count = get_video_props(path)
            if count > 0:
                self.video_frame_counts[path] = count
                print(f"  {os.path.basename(path)}: {count} frames")
        
        # Setup proportional sampling (Cumulative Distribution)
        self.cumulative_frames = []
        self.active_video_paths = [] # Only videos with >0 frames
        self.total_combined_frames = 0
        current_total = 0
        
        for path in self.video_paths:
            if path in self.video_frame_counts:
                count = self.video_frame_counts[path]
                current_total += count
                self.cumulative_frames.append(current_total)
                self.active_video_paths.append(path)
                
        self.total_combined_frames = current_total
        print(f"Total frames across {len(self.active_video_paths)} videos: {self.total_combined_frames}")

        # State
        self.current_patch_info = None # {video_path, frame_idx, crop_rect: (x,y,w,h), image}
        self.history = [] # List of patch_info dicts
        self.history_idx = -1
        
        self.photo_image = None
        self.scale = 1.0
        
        # Line drawing state
        self.lines = [] # List of [(x1, y1), (x2, y2)] in image coords
        self.first_point = None # The locked-in first point
        self.current_point = None # The point being dragged right now
        self.is_dragging = False
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.display_scale = 1.0
        self.show_mask_mode = False  # Toggle between normal and mask view
        self.show_lines = True  # Toggle line visibility
        self.selected_line_idx = None  # Index of currently selected line
        
        # State for editing existing points
        self.editing_point = None  # (line_index, point_index) being edited, or None
        
        # UI Setup
        self.root.title(f"Random Patch Viewer ({patch_size}x{patch_size})")
        self._build_ui()
        
        # Bindings
        self.root.bind("<Configure>", self.on_resize)
        self.root.bind("<a>", lambda e: self.prev_patch())
        self.root.bind("<d>", lambda e: self.next_patch())
        self.root.bind("<Left>", lambda e: self.prev_patch())
        self.root.bind("<Right>", lambda e: self.next_patch())
        self.root.bind("<m>", lambda e: self.toggle_mask())
        self.root.bind("<l>", lambda e: self.toggle_lines())
        self.canvas.bind("<Button-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Initial patch
        if self.total_combined_frames > 0:
            self.next_patch()
        else:
            messagebox.showerror("Error", "No valid frames found in the provided videos.")

    def _build_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas Area (Left)
        self.canvas = tk.Canvas(main_frame, bg="#222222", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Sidebar (Right)
        sidebar = ttk.Frame(main_frame, width=300, padding=10)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False) # Force width
        
        # Title
        ttk.Label(sidebar, text="Patch Viewer", font=("Arial", 14, "bold")).pack(pady=(0, 20), anchor="w")
        
        # Info Panel
        info_frame = ttk.LabelFrame(sidebar, text="Current Sample", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.lbl_video = ttk.Label(info_frame, text="Video: -", wraplength=260)
        self.lbl_video.pack(anchor="w", pady=2)
        
        self.lbl_frame = ttk.Label(info_frame, text="Frame: -")
        self.lbl_frame.pack(anchor="w", pady=2)
        
        self.lbl_coords = ttk.Label(info_frame, text="Crop: -")
        self.lbl_coords.pack(anchor="w", pady=2)
        
        # Lines List Panel
        lines_frame = ttk.LabelFrame(sidebar, text="Lines", padding=10)
        lines_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Create scrollable list for lines
        list_container = ttk.Frame(lines_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.lines_listbox = tk.Listbox(list_container, yscrollcommand=scrollbar.set, height=10)
        self.lines_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.lines_listbox.yview)
        self.lines_listbox.bind('<<ListboxSelect>>', self.on_line_select)
        
        # Navigation Panel
        nav_frame = ttk.LabelFrame(sidebar, text="Navigation", padding=10)
        nav_frame.pack(fill=tk.X, pady=(0, 20))
        
        btn_prev = ttk.Button(nav_frame, text="<< Previous (A)", command=self.prev_patch)
        btn_prev.pack(fill=tk.X, pady=5)
        
        btn_next = ttk.Button(nav_frame, text="Next Random (D) >>", command=self.next_patch)
        btn_next.pack(fill=tk.X, pady=5)
        
        self.btn_toggle_mask = ttk.Button(nav_frame, text="Show Mask (M)", command=self.toggle_mask)
        self.btn_toggle_mask.pack(fill=tk.X, pady=5)

        # Instructions
        ttk.Label(sidebar, text="Instructions:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(20, 5))
        ttk.Label(sidebar, text="• Press 'D' or Right Arrow for a new random sample\n• Press 'A' or Left Arrow to go back\n• Resizing window scales the patch").pack(anchor="w")

    def get_random_frame_location(self):
        """Select a video and frame index proportional to frame count."""
        if self.total_combined_frames == 0:
            return None, None
            
        global_idx = random.randint(0, self.total_combined_frames - 1)
        video_idx = bisect.bisect_left(self.cumulative_frames, global_idx)
        
        # Safety check
        if video_idx >= len(self.active_video_paths):
            video_idx = len(self.active_video_paths) - 1
            
        video_path = self.active_video_paths[video_idx]
        
        prev_cumulative = self.cumulative_frames[video_idx - 1] if video_idx > 0 else 0
        frame_idx = global_idx - prev_cumulative # Local frame index
        
        return video_path, frame_idx

    def generate_new_patch(self):
        if self.total_combined_frames == 0:
            return None

        # Try up to 10 times to get a valid frame/patch (in case of read errors)
        for _ in range(10):
            video_path, frame_idx = self.get_random_frame_location()
            if not video_path:
                        continue
                    
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                print(f"Warning: Could not read frame {frame_idx} from {video_path}")
                continue
                
            h, w = frame.shape[:2]
            
            # Ensure frame is large enough
            if h < self.patch_size or w < self.patch_size:
                # If frame is too small, just take center crop or resize? 
                # User asked for 250x250. If smaller, let's skip or take whole.
                # Let's just pad it if smaller, or skip. Skipping is safer for "random 250x250 area"
                if h < 10 or w < 10: # Extremely small
                    continue
                
                # If slightly smaller, just use 0,0 and min size
                x, y = 0, 0
                cw, ch = min(w, self.patch_size), min(h, self.patch_size)
            else:
                x = random.randint(0, w - self.patch_size)
                y = random.randint(0, h - self.patch_size)
                cw, ch = self.patch_size, self.patch_size
            
            # Extract patch
            patch = frame[y:y+ch, x:x+cw]
            
            # Convert BGR to RGB
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            
            return {
                "video_path": video_path,
                "frame_idx": frame_idx,
                "crop_rect": (x, y, cw, ch),
                "image": patch_rgb
            }
            
        return None

    def next_patch(self):
        if self.history_idx < len(self.history) - 1:
            # Move forward in history
            self.history_idx += 1
            self.current_patch_info = self.history[self.history_idx]
        else:
            # Generate new
            new_info = self.generate_new_patch()
            if new_info:
                self.history.append(new_info)
                self.history_idx = len(self.history) - 1
                self.current_patch_info = new_info
        
        self.display_current_patch()

    def prev_patch(self):
        if self.history_idx > 0:
            self.history_idx -= 1
            self.current_patch_info = self.history[self.history_idx]
            self.display_current_patch()

    def display_current_patch(self):
        if not self.current_patch_info:
            return
            
        info = self.current_patch_info
        
        # Update Info Labels
        self.lbl_video.config(text=f"Video: {os.path.basename(info['video_path'])}")
        self.lbl_frame.config(text=f"Frame: {info['frame_idx']}")
        x, y, w, h = info['crop_rect']
        self.lbl_coords.config(text=f"Crop: x={x}, y={y} ({w}x{h})")
        
        # Update Lines List
        self.update_lines_list()
        
        # Display Image on Canvas
        self.draw_image()

    def draw_image(self):
        if not self.current_patch_info:
            return
            
        img_arr = self.current_patch_info['image']
        img_h, img_w = img_arr.shape[:2]
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            # Wait for layout
            self.root.after(100, self.draw_image)
            return
        
        # Choose what to display based on mode
        if self.show_mask_mode:
            # Create mask view
            display_arr = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            # Draw each line as 3px wide white line
            for line in self.lines:
                p1, p2 = line
                x1, y1 = int(round(p1[0])), int(round(p1[1]))
                x2, y2 = int(round(p2[0])), int(round(p2[1]))
                cv2.line(display_arr, (x1, y1), (x2, y2), (255, 255, 255), thickness=3)
        else:
            # Normal view - show image
            display_arr = img_arr
        
        # Scale image to ~33% of screen height while maintaining aspect ratio
        target_height = int(canvas_h * 0.33)
        scale = target_height / img_h
        
        # Use Nearest Neighbor for sharp upscaling, Linear for downscaling
        interp = cv2.INTER_NEAREST if scale > 1.5 else cv2.INTER_LINEAR
        
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized = cv2.resize(display_arr, (new_w, new_h), interpolation=interp)
        
        self.photo_image = ImageTk.PhotoImage(Image.fromarray(resized))
        
        # Center horizontally and vertically
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        
        self.canvas_offset_x = x_offset
        self.canvas_offset_y = y_offset
        self.display_scale = scale
        
        self.canvas.delete("all")
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.photo_image)
        
        # Only draw UI overlays in normal mode
        if not self.show_mask_mode:
            # Draw existing lines (if visible)
            if self.show_lines:
                self.draw_lines_on_canvas()
            
            # Draw locked first point if it exists
            if self.first_point is not None:
                self.draw_point_on_canvas(self.first_point, "lime", 8)
            
            # Draw the point being currently dragged
            if self.current_point is not None:
                if self.first_point is None:
                    # Dragging first point
                    self.draw_point_on_canvas(self.current_point, "yellow", 8)
                else:
                    # Dragging second point - also show preview line
                    self.draw_point_on_canvas(self.current_point, "red", 8)
                    canvas_x1, canvas_y1 = self.image_to_canvas_coords(self.first_point[0], self.first_point[1])
                    canvas_x2, canvas_y2 = self.image_to_canvas_coords(self.current_point[0], self.current_point[1])
                    self.canvas.create_line(
                        canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                        fill="yellow", width=2, dash=(4, 4)
                    )

    def on_resize(self, event):
        # Debounce or just redraw
        if self.current_patch_info:
            self.draw_image()
    
    def update_lines_list(self):
        """Update the lines listbox with current lines."""
        self.lines_listbox.delete(0, tk.END)
        
        if not self.lines:
            self.lines_listbox.insert(tk.END, "No lines yet")
        else:
            for i, line in enumerate(self.lines):
                p1, p2 = line
                text = f"Line {i+1}: ({p1[0]:.1f}, {p1[1]:.1f}) → ({p2[0]:.1f}, {p2[1]:.1f})"
                self.lines_listbox.insert(tk.END, text)
            
            # Select the currently selected line in the listbox
            if self.selected_line_idx is not None and self.selected_line_idx < len(self.lines):
                self.lines_listbox.selection_clear(0, tk.END)
                self.lines_listbox.selection_set(self.selected_line_idx)
                self.lines_listbox.see(self.selected_line_idx)
    
    def toggle_mask(self):
        """Toggle between normal view and mask view."""
        self.show_mask_mode = not self.show_mask_mode
        
        # Update button text
        if self.show_mask_mode:
            self.btn_toggle_mask.config(text="Show Image (M)")
        else:
            self.btn_toggle_mask.config(text="Show Mask (M)")
        
        # Redraw
        self.draw_image()
    
    def toggle_lines(self):
        """Toggle visibility of lines on the canvas."""
        self.show_lines = not self.show_lines
        self.draw_image()
    
    def on_line_select(self, event):
        """Handle selection of a line from the listbox."""
        selection = self.lines_listbox.curselection()
        if selection:
            # selection is a tuple of indices
            self.selected_line_idx = selection[0]
            self.draw_image()
    
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to image coordinates."""
        img_x = (canvas_x - self.canvas_offset_x) / self.display_scale
        img_y = (canvas_y - self.canvas_offset_y) / self.display_scale
        return img_x, img_y
    
    def image_to_canvas_coords(self, img_x, img_y):
        """Convert image coordinates to canvas coordinates."""
        canvas_x = self.canvas_offset_x + img_x * self.display_scale
        canvas_y = self.canvas_offset_y + img_y * self.display_scale
        return canvas_x, canvas_y
    
    def draw_point_on_canvas(self, point, color, size):
        """Draw a point on the canvas."""
        canvas_x, canvas_y = self.image_to_canvas_coords(point[0], point[1])
        self.canvas.create_oval(
            canvas_x - size, canvas_y - size,
            canvas_x + size, canvas_y + size,
            fill=color, outline=color
        )
    
    def draw_lines_on_canvas(self):
        """Draw all placed lines on the canvas."""
        for line_idx, line in enumerate(self.lines):
            p1, p2 = line
            canvas_x1, canvas_y1 = self.image_to_canvas_coords(p1[0], p1[1])
            canvas_x2, canvas_y2 = self.image_to_canvas_coords(p2[0], p2[1])
            
            # Check if this line is being edited or selected
            is_editing = (self.editing_point is not None and 
                         self.editing_point[0] == line_idx)
            is_selected = (self.selected_line_idx == line_idx)
            
            if is_editing:
                # Draw yellow dotted line while editing
                self.canvas.create_line(
                    canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                    fill="yellow", width=2, dash=(4, 4)
                )
            elif is_selected:
                # Draw green line for selected
                self.canvas.create_line(
                    canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                    fill="lime", width=3
                )
            else:
                # Draw solid cyan line normally
                self.canvas.create_line(
                    canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                    fill="cyan", width=2
                )
            
            # Draw endpoints - highlight if selected
            if is_selected:
                self.draw_point_on_canvas(p1, "lime", 6)
                self.draw_point_on_canvas(p2, "lime", 6)
            else:
                self.draw_point_on_canvas(p1, "lime", 5)
                self.draw_point_on_canvas(p2, "red", 5)
    
    def on_canvas_press(self, event):
        """Handle mouse press on canvas - start dragging a new point or edit existing."""
        if not self.current_patch_info or self.show_mask_mode:
            return
        
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        
        # Check if clicking near an existing point (within 15px in image coords)
        click_threshold = 15 / self.display_scale
        
        for line_idx, line in enumerate(self.lines):
            for point_idx, point in enumerate(line):
                dist = ((point[0] - img_x)**2 + (point[1] - img_y)**2)**0.5
                if dist < click_threshold:
                    # Start editing this point
                    self.is_dragging = True
                    self.editing_point = (line_idx, point_idx)
                    self.draw_image()
                    return
        
        # Not clicking on existing point - start new point/line
        self.is_dragging = True
        self.current_point = (img_x, img_y)
        self.draw_image()
    
    def on_canvas_drag(self, event):
        """Handle mouse drag on canvas - update the point being dragged."""
        if not self.is_dragging or not self.current_patch_info:
            return
        
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        
        if self.editing_point is not None:
            # Update existing point
            line_idx, point_idx = self.editing_point
            self.lines[line_idx][point_idx] = (img_x, img_y)
        else:
            # Update the current point being dragged (no clamping - allow points outside image)
            self.current_point = (img_x, img_y)
        
        self.draw_image()
    
    def on_canvas_release(self, event):
        """Handle mouse release on canvas - lock in the point."""
        if not self.is_dragging or not self.current_patch_info:
            return
        
        self.is_dragging = False
        
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        
        if self.editing_point is not None:
            # Finished editing existing point
            line_idx, point_idx = self.editing_point
            self.lines[line_idx][point_idx] = (img_x, img_y)
            self.selected_line_idx = line_idx  # Select the edited line
            self.editing_point = None
            self.update_lines_list()
        else:
            # No clamping - allow points outside image bounds
            final_point = (img_x, img_y)
            
            if self.first_point is None:
                # First point is now locked in
                self.first_point = final_point
                self.current_point = None
            else:
                # Second point - create the line
                self.lines.append([self.first_point, final_point])
                self.selected_line_idx = len(self.lines) - 1  # Select the newly created line
                self.first_point = None
                self.current_point = None
                # Update the lines list
                self.update_lines_list()
        
        self.draw_image()

def find_videos(input_paths):
    video_files = []
    supported_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    
    for path in input_paths:
        if os.path.isfile(path):
            if os.path.splitext(path)[1].lower() in supported_extensions:
                video_files.append(path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if os.path.splitext(file)[1].lower() in supported_extensions:
                        video_files.append(os.path.join(root, file))
    
    return sorted(list(set(video_files)))

def main():
    parser = argparse.ArgumentParser(description="View random 400x400 patches from videos.")
    parser.add_argument("videos", nargs="*", help="Video files or directories")
    args = parser.parse_args()
    
    video_inputs = args.videos
    if not video_inputs:
        # Default to 'videos' directory if it exists
        if os.path.exists("videos"):
            video_inputs = ["videos"]
        else:
            print("No video paths provided and 'videos' folder not found.")
            return

    video_paths = find_videos(video_inputs)
    
    if not video_paths:
        print("No video files found.")
        return
        
    print(f"Found {len(video_paths)} videos.")
    
    root = tk.Tk()
    root.state('zoomed')  # Fullscreen on Windows
    
    app = RandomPatchViewer(root, video_paths)
    
    root.mainloop()

if __name__ == "__main__":
    main()
