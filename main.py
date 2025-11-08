import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from contour_detector import ContourDetector

class VideoFrameViewer:
    def __init__(self, root, video_path):
        self.root = root
        self.root.title("Video Frame Viewer")
        self.root.state('zoomed')
        self.is_fullscreen = True
        
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.slider_updating = False
        self.contour_detector = ContourDetector()
        self.show_tiles = True
        
        # Main container with sidebar and video area
        main_container = ttk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left sidebar for controls
        sidebar = ttk.Frame(main_container, width=200)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        sidebar.pack_propagate(False)
        
        # Control buttons
        ttk.Button(sidebar, text="◀ Previous", command=self.prev_frame).pack(fill=tk.X, pady=5)
        ttk.Button(sidebar, text="Next ▶", command=self.next_frame).pack(fill=tk.X, pady=5)
        ttk.Button(sidebar, text="⏮ First", command=self.first_frame).pack(fill=tk.X, pady=5)
        ttk.Button(sidebar, text="⏭ Last", command=self.last_frame).pack(fill=tk.X, pady=5)
        ttk.Button(sidebar, text="🔲 Toggle Tiles", command=self.toggle_tiles).pack(fill=tk.X, pady=5)
        
        # Frame info
        self.info_label = ttk.Label(sidebar, text="", font=("Arial", 9), wraplength=180, justify=tk.LEFT)
        self.info_label.pack(fill=tk.X, pady=10)
        
        # Slider
        slider_label = ttk.Label(sidebar, text="Timeline:")
        slider_label.pack(fill=tk.X, pady=(10, 5))
        self.slider = ttk.Scale(sidebar, from_=0, to=self.total_frames-1, orient=tk.VERTICAL, command=self.slider_changed)
        self.slider.pack(fill=tk.Y, expand=True)
        
        # Right side for video display
        video_container = ttk.Frame(main_container)
        video_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.image_label = ttk.Label(video_container, background="black")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Bind arrow keys
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        self.root.bind('<F11>', self.toggle_fullscreen)
        self.root.bind('<Escape>', self.exit_fullscreen)
        
        self.root.after(100, self.display_frame)
    
    def display_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        
        if ret:
            # Apply tile detection if enabled
            if self.show_tiles:
                frame, _ = self.contour_detector.detect_tiles(frame)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            label_width = self.image_label.winfo_width()
            label_height = self.image_label.winfo_height()
            if label_width > 1 and label_height > 1:
                scale = min(label_width / w, label_height / h)
            else:
                scale = min(640 / w, 480 / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
            
            self.image_label.config(image=photo)
            self.image_label.image = photo
            
            time_seconds = self.current_frame / self.fps
            tiles_status = "ON" if self.show_tiles else "OFF"
            self.info_label.config(text=f"Frame {self.current_frame + 1} / {self.total_frames}\nTime: {time_seconds:.2f}s\nTiles: {tiles_status}")
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
    
    def toggle_fullscreen(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        self.root.state('zoomed' if self.is_fullscreen else 'normal')
    
    def exit_fullscreen(self, event=None):
        if self.is_fullscreen:
            self.is_fullscreen = False
            self.root.state('normal')
    
    def toggle_tiles(self):
        self.show_tiles = not self.show_tiles
        print(f"\n[GUI] Tile detection: {'ENABLED' if self.show_tiles else 'DISABLED'}")
        self.display_frame()

if __name__ == "__main__":
    root = tk.Tk()
    viewer = VideoFrameViewer(root, "video.mp4")
    root.mainloop()

