import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

class VideoFrameViewer:
    def __init__(self, root, video_path):
        self.root = root
        self.root.title("Video Frame Viewer")
        self.root.geometry("800x700")
        
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.slider_updating = False
        
        # Top frame for controls
        control_frame = ttk.Frame(root)
        control_frame.pack(pady=10)
        
        ttk.Button(control_frame, text="◀ Previous", command=self.prev_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next ▶", command=self.next_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="⏮ First", command=self.first_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="⏭ Last", command=self.last_frame).pack(side=tk.LEFT, padx=5)
        
        # Frame info
        self.info_label = ttk.Label(root, text="", font=("Arial", 10))
        self.info_label.pack()
        
        # Slider
        slider_frame = ttk.Frame(root)
        slider_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.slider = ttk.Scale(slider_frame, from_=0, to=self.total_frames-1, orient=tk.HORIZONTAL, command=self.slider_changed)
        self.slider.pack(fill=tk.X)
        
        # Image display
        self.image_label = ttk.Label(root)
        self.image_label.pack(pady=10)
        
        self.display_frame()
    
    def display_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            scale = min(640 / w, 480 / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
            
            self.image_label.config(image=photo)
            self.image_label.image = photo
            
            time_seconds = self.current_frame / self.fps
            self.info_label.config(text=f"Frame {self.current_frame + 1} / {self.total_frames} | Time: {time_seconds:.2f}s")
            self.slider_updating = True
            self.slider.set(self.current_frame)
            self.slider_updating = False
    
    def next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.display_frame()
    
    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.display_frame()
    
    def first_frame(self):
        self.current_frame = 0
        self.display_frame()
    
    def last_frame(self):
        self.current_frame = self.total_frames - 1
        self.display_frame()
    
    def slider_changed(self, value):
        if not self.slider_updating:
            self.current_frame = int(float(value))
            self.display_frame()

if __name__ == "__main__":
    root = tk.Tk()
    viewer = VideoFrameViewer(root, "video.mp4")
    root.mainloop()

