import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import Optional, Dict, Tuple

# Assuming frame_pair_overlay_viewer.py is in the same directory
from frame_pair_overlay_viewer import FrameCache, compose_display, extract_crossings, TILE_DISPLAY_SIZE, calculate_diff_for_offset

class VideoPlayer:
    def __init__(self, root, video_path: str, max_width: int, max_height: int, start_frame: int = 0):
        self.root = root
        self.root.title("Frame Pair Overlay Viewer")

        self.frame_cache = FrameCache(video_path)
        self.total_frames = self.frame_cache.get_total_frames()
        self.max_width = max_width
        self.max_height = max_height
        self.x_offset = 0
        self.y_offset = 0

        if self.total_frames < 2:
            raise RuntimeError("Video must contain at least two frames.")

        self.current_pair_index = start_frame if start_frame < self.total_frames - 1 else 0
        self.previous_pair_index = -1
        self.last_printed_index = -1  # Track last printed frame index

        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a label to display the video frames
        self.image_label = ttk.Label(main_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Create a slider to navigate through the frames
        self.slider = ttk.Scale(
            main_frame,
            from_=0,
            to=self.total_frames - 2,
            orient=tk.HORIZONTAL,
            command=self.on_slider_move,
        )
        self.slider.set(self.current_pair_index)
        self.slider.pack(fill=tk.X, padx=10, pady=5)

        # Add a label for pixel distance
        self.dist_label = ttk.Label(main_frame, text="Pixel distance: N/A", font=("Helvetica", 12))
        self.dist_label.pack(pady=5)

        self.root.bind_all("<Left>", self.prev_frame)
        self.root.bind_all("<Right>", self.next_frame)
        self.root.bind_all("<Shift-Left>", self.shift_left)
        self.root.bind_all("<Shift-Right>", self.shift_right)
        self.root.bind_all("<Shift-Up>", self.shift_up)
        self.root.bind_all("<Shift-Down>", self.shift_down)
        
        # Maximize the window
        self.root.state('zoomed')

        self.resize_timer = None
        self.root.bind("<Configure>", self.on_resize)

        self.update_frame()

    def on_slider_move(self, value):
        new_index = int(float(value))
        if new_index != self.current_pair_index:
            self.current_pair_index = new_index
            self.update_frame()

    def on_resize(self, event):
        if self.resize_timer:
            self.root.after_cancel(self.resize_timer)
        self.resize_timer = self.root.after(100, self.perform_resize)

    def perform_resize(self):
        new_width = self.image_label.winfo_width()
        new_height = self.image_label.winfo_height()

        if new_width > 1 and new_height > 1:
            self.max_width = new_width
            self.max_height = new_height
            self.update_frame()

    def next_frame(self, event=None):
        if self.current_pair_index < self.total_frames - 2:
            self.current_pair_index += 1
            self.slider.set(self.current_pair_index)
            self.update_frame()

    def prev_frame(self, event=None):
        if self.current_pair_index > 0:
            self.current_pair_index -= 1
            self.slider.set(self.current_pair_index)
            self.update_frame()

    def shift_left(self, event=None):
        self.x_offset -= TILE_DISPLAY_SIZE
        self.update_frame()

    def shift_right(self, event=None):
        self.x_offset += TILE_DISPLAY_SIZE
        self.update_frame()

    def shift_up(self, event=None):
        self.y_offset -= TILE_DISPLAY_SIZE
        self.update_frame()

    def shift_down(self, event=None):
        self.y_offset += TILE_DISPLAY_SIZE
        self.update_frame()

    def update_frame(self):
        # Close the previous frame's external windows if the index has changed
        if self.previous_pair_index != -1 and self.previous_pair_index != self.current_pair_index:
            try:
                cv2.destroyWindow(f"Source Frame: {self.previous_pair_index}")
                cv2.destroyWindow(f"Target Frame: {self.previous_pair_index + 1}")
            except cv2.error:
                # This can happen if the user manually closes a window. It's safe to ignore.
                pass

        first = self.frame_cache.get_frame_data(self.current_pair_index)
        second = self.frame_cache.get_frame_data(self.current_pair_index + 1)

        if first is None or second is None:
            return

        # Create and show individual frame windows with grids
        frame1_grid = self.draw_grid(first["frame"].copy(), first["grid_map"])
        frame2_grid = self.draw_grid(second["frame"].copy(), second["grid_map"])
        cv2.imshow(f"Source Frame: {self.current_pair_index}", frame1_grid)
        cv2.imshow(f"Target Frame: {self.current_pair_index + 1}", frame2_grid)
        cv2.waitKey(1) # Allow cv2 windows to process events

        # Only print crossings if the frame index has changed
        if self.current_pair_index != self.last_printed_index:
            crossings_first = extract_crossings(first["grid_map"], first["frame"].shape[:2])
            crossings_second = extract_crossings(second["grid_map"], second["frame"].shape[:2])
            
            print(f"\n--- Frame {self.current_pair_index} ---")
            print(f"Crossings ({len(crossings_first)}): {crossings_first}")
            print(f"\n--- Frame {self.current_pair_index + 1} ---")
            print(f"Crossings ({len(crossings_second)}): {crossings_second}")
            
            print("\n--- Pixel Distances for Offsets (+-3 tiles) ---")
            results = []
            for y_tile in range(-3, 4):
                for x_tile in range(-3, 4):
                    x_offset_px = x_tile * TILE_DISPLAY_SIZE
                    y_offset_px = y_tile * TILE_DISPLAY_SIZE
                    total_dist, norm_dist, area = calculate_diff_for_offset(
                        first, second, x_offset_px, y_offset_px
                    )
                    if area > 0:
                        results.append(((x_tile, y_tile), total_dist, norm_dist))
            
            # Sort results by normalized distance (ascending)
            results.sort(key=lambda item: item[2])
            
            # Automatically apply the best offset
            if results:
                best_offset_tile, _, _ = results[0]
                self.x_offset = best_offset_tile[0] * TILE_DISPLAY_SIZE
                self.y_offset = best_offset_tile[1] * TILE_DISPLAY_SIZE

            for (x_tile, y_tile), total_dist, norm_dist in results:
                print(f"Tile Offset ({x_tile:2d}, {y_tile:2d}): Sum={total_dist:12,.0f}, Avg={norm_dist:8,.2f}")

            self.last_printed_index = self.current_pair_index

        display_image, pixel_dist, norm_dist = compose_display(
            first, second, self.current_pair_index, self.current_pair_index + 1, self.x_offset, self.y_offset
        )
        self.dist_label.config(
            text=f"Pixel Diff Sum: {pixel_dist:,.0f} | Per-Pixel Avg: {norm_dist:,.2f}"
        )

        # Resize for display
        h, w = display_image.shape[:2]
        scale = min(self.max_width / max(w, 1), self.max_height / max(h, 1))
        
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w > 0 and new_h > 0:
            display_image = cv2.resize(display_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Convert the image to a format Tkinter can use
        img = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk)

        # Update the previous index tracker for the next call
        self.previous_pair_index = self.current_pair_index

    def draw_grid(self, frame: np.ndarray, grid_map: Dict[Tuple[int, int], np.ndarray]) -> np.ndarray:
        """Draws the grid lines on a frame using blue lines."""
        if not grid_map:
            return frame
        
        for square_coords in grid_map.values():
            contour = np.array(square_coords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [contour], isClosed=True, color=(255, 0, 0), thickness=2)
            
        return frame

    def on_closing(self):
        self.frame_cache.release()
        self.root.destroy()
        cv2.destroyAllWindows()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View consecutive frame pairs with overlay and unwarped tiles using Tkinter.")
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to input video file")
    parser.add_argument(
        "--max-window-width",
        type=int,
        default=2400,
        help="Maximum display window width in pixels (display is resized if wider)",
    )
    parser.add_argument(
        "--max-window-height",
        type=int,
        default=1350,
        help="Maximum display window height in pixels (display is resized if taller)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Frame index to start the viewer on",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    root = tk.Tk()
    app = VideoPlayer(root, args.video, args.max_window_width, args.max_window_height, args.start_frame)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
