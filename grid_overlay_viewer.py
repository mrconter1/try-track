import cv2
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def draw_grid(frame, grid_size=50):
    """Draws a grid on the frame."""
    if grid_size <= 0:
        return frame
    h, w = frame.shape[:2]
    color = (0, 255, 0)
    thickness = 1
    for y in range(0, h, grid_size):
        cv2.line(frame, (0, y), (w, y), color, thickness)
    for x in range(0, w, grid_size):
        cv2.line(frame, (x, 0), (x, h), color, thickness)
    return frame

class App:
    def __init__(self, root, args, initial_frame, video_path, total_frames):
        self.root = root
        self.args = args
        self.original_frame = initial_frame
        self.video_path = video_path
        self.total_frames = total_frames
        self.current_frame_num = args.frame
        
        self.root.title("Grid Overlay Control")
        
        # --- Variables ---
        self.frame_num_var = tk.IntVar(value=self.current_frame_num)
        self.grid_size_var = tk.IntVar(value=args.grid_size)
        
        # --- GUI Layout ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.grid(row=1, column=0, sticky="ew")

        # Frame Number Controls
        ttk.Label(control_frame, text="Frame:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_frame(-10)).grid(row=0, column=1)
        self.frame_slider = tk.Scale(control_frame, from_=0, to=self.total_frames-1, orient=tk.HORIZONTAL, variable=self.frame_num_var, command=self.load_frame, resolution=1, showvalue=0)
        self.frame_slider.grid(row=0, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_frame(10)).grid(row=0, column=3)
        self.frame_label = ttk.Label(control_frame, text=f"{self.frame_num_var.get()}", width=7)
        self.frame_label.grid(row=0, column=4, padx=5)

        # Grid Size Controls
        ttk.Label(control_frame, text="Grid Size:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="-", width=3, command=lambda: self.adjust_grid_size(-1)).grid(row=1, column=1)
        self.grid_size_slider = tk.Scale(control_frame, from_=5, to=200, orient=tk.HORIZONTAL, variable=self.grid_size_var, command=self.update_image, resolution=1, showvalue=0)
        self.grid_size_slider.grid(row=1, column=2, sticky="ew")
        ttk.Button(control_frame, text="+", width=3, command=lambda: self.adjust_grid_size(1)).grid(row=1, column=3)
        self.grid_size_label = ttk.Label(control_frame, text=f"{self.grid_size_var.get()}", width=7)
        self.grid_size_label.grid(row=1, column=4, padx=5)

        control_frame.columnconfigure(2, weight=1)

        self.root.update_idletasks()
        control_height = control_frame.winfo_reqheight()

        max_width = root.winfo_screenwidth() - 40
        max_height = root.winfo_screenheight() - control_height - 150
        
        h_orig, w_orig = self.original_frame.shape[:2]
        ratio = min(max_width / w_orig, max_height / h_orig)
        
        self.display_w = int(w_orig * ratio)
        self.display_h = int(h_orig * ratio)

        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew")
        
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        self.update_image()

    def adjust_grid_size(self, amount):
        current_val = self.grid_size_var.get()
        self.grid_size_var.set(max(5, current_val + amount))
        self.update_image()

    def adjust_frame(self, amount):
        current_frame = self.frame_num_var.get()
        new_frame = max(0, min(self.total_frames - 1, current_frame + amount))
        self.frame_num_var.set(new_frame)
        self.load_frame(str(new_frame))

    def load_frame(self, frame_num_str):
        frame_num = int(float(frame_num_str))
        self.current_frame_num = frame_num
        
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            self.original_frame = frame
            self.frame_label.config(text=f"{frame_num}")
            self.update_image()
        
    def update_image(self, *args):
        frame_copy = self.original_frame.copy()
        grid_size = self.grid_size_var.get()
        
        frame_with_grid = draw_grid(frame_copy, grid_size)
        
        display_frame = cv2.resize(frame_with_grid, (self.display_w, self.display_h), interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        self.image_label.imgtk = img_tk
        self.image_label.configure(image=img_tk)

        self.grid_size_label.config(text=f"{grid_size}")
        self.frame_label.config(text=f"{self.current_frame_num}")

def main(args):
    """Main function to load frame and launch the GUI."""
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frame >= total_frames:
        print(f"Error: Frame {args.frame} is out of bounds. Video has {total_frames} frames.")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Error: Could not read frame {args.frame}.")
        return

    root = tk.Tk()
    root.state('zoomed')
    app = App(root, args, frame, args.video, total_frames)
    root.mainloop()

def parse_args():
    parser = argparse.ArgumentParser(description="Display a video frame with an interactive grid overlay.")
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to the video file.")
    parser.add_argument("--frame", type=int, default=4113, help="Frame number to load.")
    parser.add_argument("--grid-size", type=int, default=50, help="Initial size of the grid cells in pixels.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
