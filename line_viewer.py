import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from line_detector import LineDetector

class LineViewerApp:
    def __init__(self, root, video_path):
        self.root = root
        self.root.title("Line Viewer")
        self.root.state('zoomed')
        self.is_fullscreen = True
        
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.slider_updating = False
        self.line_detector = LineDetector()
        self.show_lines = True
        self.current_lines = []
        self.grid_squares = []
        self.square_ids = {}  # Maps (frame_number, square_index) to assigned ID
        self.frame_scale = 1.0  # Scale factor for display frame
        self.original_frame_size = None
        
        # Main container with sidebar and video
        main_container = ttk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left sidebar with controls
        sidebar = ttk.Frame(main_container, width=200)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        sidebar.pack_propagate(False)
        
        # Control buttons
        ttk.Button(sidebar, text="◀ Previous", command=self.prev_frame).pack(fill=tk.X, pady=5)
        ttk.Button(sidebar, text="Next ▶", command=self.next_frame).pack(fill=tk.X, pady=5)
        ttk.Button(sidebar, text="⏮ First", command=self.first_frame).pack(fill=tk.X, pady=5)
        ttk.Button(sidebar, text="⏭ Last", command=self.last_frame).pack(fill=tk.X, pady=5)
        ttk.Button(sidebar, text="🔲 Toggle Lines", command=self.toggle_lines).pack(fill=tk.X, pady=5)
        
        # Frame info
        self.info_label = ttk.Label(sidebar, text="", font=("Arial", 9), wraplength=180, justify=tk.LEFT)
        self.info_label.pack(fill=tk.X, pady=10)
        
        # Slider
        slider_label = ttk.Label(sidebar, text="Timeline:")
        slider_label.pack(fill=tk.X, pady=(10, 5))
        self.slider = ttk.Scale(sidebar, from_=0, to=self.total_frames-1, orient=tk.VERTICAL, command=self.slider_changed)
        self.slider.pack(fill=tk.Y, expand=True)
        
        # Center for video display
        video_container = ttk.Frame(main_container)
        video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.image_label = ttk.Label(video_container, background="black")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.image_label.bind("<Button-1>", self.on_frame_click)
        
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
            self.original_frame_size = (frame.shape[1], frame.shape[0])
            
            # Apply line detection if enabled
            if self.show_lines:
                frame, self.current_lines = self.line_detector.detect_lines(frame)
                num_lines = len(self.current_lines)
                self.grid_squares = self._extract_grid_squares(self.current_lines, frame.shape[:2])
            else:
                num_lines = 0
                self.current_lines = []
                self.grid_squares = []
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            label_width = self.image_label.winfo_width()
            label_height = self.image_label.winfo_height()
            if label_width > 1 and label_height > 1:
                self.frame_scale = min(label_width / w, label_height / h)
            else:
                self.frame_scale = min(640 / w, 480 / h)
            new_w, new_h = int(w * self.frame_scale), int(h * self.frame_scale)
            frame = cv2.resize(frame, (new_w, new_h))
            
            # Draw square IDs on frame
            frame = self._draw_square_ids(frame)
            
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
            
            self.image_label.config(image=photo)
            self.image_label.image = photo
            
            time_seconds = self.current_frame / self.fps
            lines_status = "ON" if self.show_lines else "OFF"
            squares_status = len(self.grid_squares)
            self.info_label.config(text=f"Frame {self.current_frame + 1} / {self.total_frames}\nTime: {time_seconds:.2f}s\nLines: {lines_status}\nDetected: {num_lines}\nSquares: {squares_status}")
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
    
    def toggle_lines(self):
        self.show_lines = not self.show_lines
        print(f"\n[GUI] Line detection: {'ENABLED' if self.show_lines else 'DISABLED'}")
        self.display_frame()
    
    def _extract_grid_squares(self, lines, frame_shape):
        """Extract grid squares formed by intersecting lines"""
        if len(lines) < 2:
            return []
        
        h, w = frame_shape[:2]
        squares = []
        
        # Group lines into horizontal and vertical
        horizontal = []
        vertical = []
        
        for rho, theta in lines:
            theta_deg = theta * 180 / np.pi
            # Normalize theta to 0-180 range
            theta_deg = theta_deg % 180
            
            # Lines close to 0° or 180° are horizontal
            # Lines close to 90° are vertical
            if theta_deg < 45 or theta_deg > 135:
                horizontal.append((rho, theta))
            else:
                vertical.append((rho, theta))
        
        # Sort lines by rho
        horizontal.sort(key=lambda x: x[0])
        vertical.sort(key=lambda x: x[0])
        
        # Find intersections between adjacent horizontal and vertical lines
        for i in range(len(horizontal) - 1):
            for j in range(len(vertical) - 1):
                rho_h1, theta_h1 = horizontal[i]
                rho_h2, theta_h2 = horizontal[i + 1]
                rho_v1, theta_v1 = vertical[j]
                rho_v2, theta_v2 = vertical[j + 1]
                
                # Find four corner points
                p1 = self._line_intersection(rho_h1, theta_h1, rho_v1, theta_v1)
                p2 = self._line_intersection(rho_h1, theta_h1, rho_v2, theta_v2)
                p3 = self._line_intersection(rho_h2, theta_h2, rho_v2, theta_v2)
                p4 = self._line_intersection(rho_h2, theta_h2, rho_v1, theta_v1)
                
                if all(p is not None for p in [p1, p2, p3, p4]):
                    # Check if all points are within frame bounds
                    if all(0 <= p[0] <= w and 0 <= p[1] <= h for p in [p1, p2, p3, p4]):
                        squares.append((p1, p2, p3, p4))
        
        return squares
    
    def _line_intersection(self, rho1, theta1, rho2, theta2):
        """Find intersection of two lines defined by (rho, theta)"""
        a1 = np.cos(theta1)
        b1 = np.sin(theta1)
        a2 = np.cos(theta2)
        b2 = np.sin(theta2)
        
        denom = a1 * b2 - a2 * b1
        if abs(denom) < 1e-6:
            return None
        
        x = (rho1 * b2 - rho2 * b1) / denom
        y = (a1 * rho2 - a2 * rho1) / denom
        
        return (int(x), int(y))
    
    def _draw_square_ids(self, frame_rgb):
        """Draw assigned IDs on squares"""
        frame_array = np.array(frame_rgb)
        
        for square_idx, square in enumerate(self.grid_squares):
            square_key = (self.current_frame, square_idx)
            if square_key in self.square_ids:
                p1, p2, p3, p4 = square
                center_x = int((p1[0] + p2[0] + p3[0] + p4[0]) / 4)
                center_y = int((p1[1] + p2[1] + p3[1] + p4[1]) / 4)
                
                # Scale for display
                center_x = int(center_x * self.frame_scale)
                center_y = int(center_y * self.frame_scale)
                
                square_id = self.square_ids[square_key]
                cv2.putText(frame_array, str(square_id), (center_x - 10, center_y + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame_array
    
    def on_frame_click(self, event):
        """Handle mouse click on frame to assign ID to square"""
        if not self.grid_squares:
            messagebox.showwarning("No Squares", "No grid squares detected on this frame")
            return
        
        # Convert click position to original frame coordinates
        click_x = int(event.x / self.frame_scale)
        click_y = int(event.y / self.frame_scale)
        
        # Find which square was clicked
        clicked_square = None
        for square_idx, square in enumerate(self.grid_squares):
            p1, p2, p3, p4 = square
            # Simple bounding box check
            xs = [p1[0], p2[0], p3[0], p4[0]]
            ys = [p1[1], p2[1], p3[1], p4[1]]
            
            if min(xs) <= click_x <= max(xs) and min(ys) <= click_y <= max(ys):
                clicked_square = square_idx
                break
        
        if clicked_square is None:
            messagebox.showinfo("No Hit", "Click was not inside any grid square")
            return
        
        # Ask for ID
        square_key = (self.current_frame, clicked_square)
        current_id = self.square_ids.get(square_key, "")
        new_id = simpledialog.askstring("Assign ID", f"Enter ID for square {clicked_square}:", initialvalue=str(current_id))
        
        if new_id is not None and new_id.strip():
            self.square_ids[square_key] = new_id.strip()
            print(f"[GUI] Assigned ID '{new_id.strip()}' to square {clicked_square} on frame {self.current_frame}")
            self.display_frame()

if __name__ == "__main__":
    root = tk.Tk()
    app = LineViewerApp(root, "video.mp4")
    root.mainloop()

