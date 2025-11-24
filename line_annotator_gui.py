import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
import random
import bisect
import argparse
import sys

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
        
        # UI Setup
        self.root.title(f"Random Patch Viewer ({patch_size}x{patch_size})")
        self._build_ui()
        
        # Bindings
        self.root.bind("<Configure>", self.on_resize)
        self.root.bind("<a>", lambda e: self.prev_patch())
        self.root.bind("<d>", lambda e: self.next_patch())
        self.root.bind("<Left>", lambda e: self.prev_patch())
        self.root.bind("<Right>", lambda e: self.next_patch())
        
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
        
        # Navigation Panel
        nav_frame = ttk.LabelFrame(sidebar, text="Navigation", padding=10)
        nav_frame.pack(fill=tk.X, pady=(0, 20))
        
        btn_prev = ttk.Button(nav_frame, text="<< Previous (A)", command=self.prev_patch)
        btn_prev.pack(fill=tk.X, pady=5)
        
        btn_next = ttk.Button(nav_frame, text="Next Random (D) >>", command=self.next_patch)
        btn_next.pack(fill=tk.X, pady=5)

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
        
        # Scale image to ~33% of screen height while maintaining aspect ratio
        target_height = int(canvas_h * 0.33)
        scale = target_height / img_h
        
        # Use Nearest Neighbor for sharp upscaling, Linear for downscaling
        interp = cv2.INTER_NEAREST if scale > 1.5 else cv2.INTER_LINEAR
        
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized = cv2.resize(img_arr, (new_w, new_h), interpolation=interp)
        
        self.photo_image = ImageTk.PhotoImage(Image.fromarray(resized))
        
        # Center horizontally and vertically
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        
        self.canvas.delete("all")
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.photo_image)

    def on_resize(self, event):
        # Debounce or just redraw
        if self.current_patch_info:
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
