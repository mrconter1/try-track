"""
Client for cloud-based crossing detection overlay.
Sends screen captures to a remote GPU server for inference.
"""

import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk
import mss
import argparse
import threading
import queue
import time
import requests
import io

class CrossingOverlayClient:
    def __init__(self, server_url, width=480, height=640, opacity=1.0, scale=1.0):
        self.server_url = server_url.rstrip('/') + "/predict"
        self.width = width
        self.height = height
        self.opacity = opacity
        self.scale = scale
        self.running = True
        
        print(f"Server URL: {self.server_url}")
        print(f"Capture: {width}x{height}")
        
        # Screen capture (created per-thread)
        self.sct = None
        
        # Persistent Session for Keep-Alive
        self.session = requests.Session()
        
        # Async processing
        self.result_queue = queue.Queue(maxsize=1)
        self.capture_lock = threading.Lock()
        self.last_capture_pos = (0, 0)
        
        # Stats
        self.fps = 0
        self.latency = 0
        
        # Setup GUI
        self._setup_gui()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
    def _setup_gui(self):
        """Create capture frame and result window."""
        self.root = tk.Tk()
        self.root.withdraw()
        
        # Capture frame window - transparent with colored border
        self.capture_frame = tk.Toplevel(self.root)
        self.capture_frame.title("Capture Region (drag me)")
        
        # Center on screen
        screen_w = self.capture_frame.winfo_screenwidth()
        screen_h = self.capture_frame.winfo_screenheight()
        center_x = (screen_w - self.width) // 2
        center_y = (screen_h - self.height) // 2
        self.capture_frame.geometry(f"{self.width}x{self.height}+{center_x}+{center_y}")
        
        self.capture_frame.attributes('-topmost', True)
        self.capture_frame.resizable(False, False)
        self.capture_frame.attributes('-transparentcolor', 'magenta')
        self.capture_frame.configure(bg='magenta')
        
        # Create border
        border = 1
        tk.Frame(self.capture_frame, bg='lime', height=border).place(x=0, y=0, relwidth=1)
        tk.Frame(self.capture_frame, bg='lime', height=border).place(x=0, rely=1, y=-border, relwidth=1)
        tk.Frame(self.capture_frame, bg='lime', width=border).place(x=0, y=0, relheight=1)
        tk.Frame(self.capture_frame, bg='lime', width=border).place(relx=1, x=-border, y=0, relheight=1)
        
        # Result window - vertically centered, at left edge
        self.result_window = tk.Toplevel(self.root)
        self.result_window.title("Cloud Detection Result")
        result_y = (screen_h - self.height) // 2
        self.result_window.geometry(f"{self.width}x{self.height}+0+{result_y}")
        self.result_window.attributes('-topmost', True)
        self.result_window.resizable(False, False)
        
        self.result_canvas = tk.Canvas(self.result_window, width=self.width, height=self.height, bg='black')
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        self.result_image = None
        
        # Status
        self.status_var = tk.StringVar(value="Connecting...")
        self.status_label = tk.Label(self.result_window, textvariable=self.status_var,
                                      bg='black', fg='lime', font=('Consolas', 9))
        self.status_label.place(x=5, y=5)
        
        # Bindings
        self.capture_frame.bind('<Escape>', lambda e: self._quit())
        self.result_window.bind('<Escape>', lambda e: self._quit())
        self.capture_frame.bind('<Up>', lambda e: self._adjust_opacity(0.1))
        self.capture_frame.bind('<Down>', lambda e: self._adjust_opacity(-0.1))
        self.result_window.bind('<Up>', lambda e: self._adjust_opacity(0.1))
        self.result_window.bind('<Down>', lambda e: self._adjust_opacity(-0.1))
        
        # Handle window close
        self.capture_frame.protocol("WM_DELETE_WINDOW", self._quit)
        self.result_window.protocol("WM_DELETE_WINDOW", self._quit)
        
        self.root.after(100, self._update_loop)
    
    def _adjust_opacity(self, delta):
        self.opacity = max(0.1, min(1.0, self.opacity + delta))
        self._update_status()
    
    def _update_status(self):
        self.status_var.set(f"FPS: {self.fps:.1f} | Latency: {self.latency:.0f}ms | Opacity: {self.opacity:.1f}")
    
    def _quit(self):
        self.running = False
        self.root.destroy()
    
    def _create_overlay_image(self, img, heatmap):
        """Blend original image with green heatmap overlay."""
        h, w = img.shape[:2]
        
        if heatmap.shape != (h, w):
            heatmap = cv2.resize(heatmap, (w, h))
        
        # Heatmap is now uint8 (0-255)
        intensity = heatmap.astype(float) / 255.0
        
        overlay = np.zeros_like(img)
        overlay[:, :, 1] = 255  # Green channel
        
        blend_factor = (intensity * self.opacity)[:, :, np.newaxis]
        result = img * (1 - blend_factor) + overlay * blend_factor
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result

    def _process_loop(self):
        """Background thread for capture and server request."""
        sct = mss.mss()
        frame_count = 0
        
        # Pre-allocate monitoring dict to avoid dict creation overhead
        monitor = {"left": 0, "top": 0, "width": self.width - 2, "height": self.height - 2}
        
        while self.running:
            try:
                t0 = time.time()
                
                # Get position
                with self.capture_lock:
                    x, y = self.last_capture_pos
                
                # Update monitor position
                monitor["left"] = x + 1
                monitor["top"] = y + 30
                
                # Capture
                screenshot = sct.grab(monitor)
                img_bgra = np.array(screenshot)
                img_rgb = cv2.cvtColor(img_bgra[:, :, :3], cv2.COLOR_BGR2RGB)
                
                # Encode to JPG for sending (Compress input too!)
                # Quality 75 is a good tradeoff
                _, img_encoded = cv2.imencode('.jpg', img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 75])
                img_bytes = img_encoded.tobytes()
                
                t1 = time.time()
                
                # Send to server using Session for Keep-Alive
                files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
                try:
                    response = self.session.post(self.server_url, files=files, timeout=2.0)
                    
                    if response.status_code == 200:
                        t2 = time.time()
                        
                        # Decode response (heatmap is now a JPG image, not npy bytes)
                        heatmap_arr = np.frombuffer(response.content, np.uint8)
                        heatmap = cv2.imdecode(heatmap_arr, cv2.IMREAD_GRAYSCALE)
                        
                        if heatmap is None:
                            print("Error decoding heatmap")
                            continue

                        # Create overlay
                        result = self._create_overlay_image(img_rgb, heatmap)
                        
                        t3 = time.time()
                        
                        # Metrics
                        total_ms = (t3 - t0) * 1000
                        net_ms = (t2 - t1) * 1000
                        fps = 1000.0 / total_ms if total_ms > 0 else 0
                        
                        # Log occasionally
                        frame_count += 1
                        if frame_count % 30 == 0:
                            print(f"Net: {net_ms:.0f}ms | Total: {total_ms:.0f}ms | FPS: {fps:.1f}")
                        
                        try:
                            self.result_queue.put_nowait((result, fps, total_ms))
                        except queue.Full:
                            pass
                            
                    else:
                        print(f"Server error: {response.status_code}")
                        time.sleep(0.5)
                        
                except requests.exceptions.RequestException as e:
                    print(f"Connection error: {e}")
                    time.sleep(1.0)
                    
            except Exception as e:
                print(f"Client error: {e}")
                time.sleep(0.5)
    
    def _update_loop(self):
        """Main thread UI update."""
        if not self.running:
            return
        
        # Update capture position
        with self.capture_lock:
            self.last_capture_pos = (self.capture_frame.winfo_x(), self.capture_frame.winfo_y())
        
        # Check queue
        try:
            result, fps, latency = self.result_queue.get_nowait()
            
            pil_img = Image.fromarray(result)
            # Ensure it fits result window
            pil_img = pil_img.resize((self.width, self.height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(pil_img)
            
            if self.result_image is None:
                self.result_image = self.result_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            else:
                self.result_canvas.itemconfig(self.result_image, image=self.photo)
            
            self.fps = fps
            self.latency = latency
            self._update_status()
            
        except queue.Empty:
            pass
        
        self.root.after(10, self._update_loop)
    
    def run(self):
        print("\nControls:")
        print("  Up/Down   - Adjust opacity")
        print("  Escape    - Quit")
        self.root.mainloop()

def main():
    parser = argparse.ArgumentParser(description="Crossing detector cloud client")
    parser.add_argument('--server', required=True, help='Server URL (e.g. http://1.2.3.4:12345)')
    parser.add_argument('--width', type=int, default=480, help='Capture width (default: 480)')
    parser.add_argument('--height', type=int, default=640, help='Capture height (default: 640)')
    parser.add_argument('--opacity', type=float, default=1.0, help='Overlay opacity')
    args = parser.parse_args()
    
    client = CrossingOverlayClient(
        server_url=args.server,
        width=args.width,
        height=args.height,
        opacity=args.opacity
    )
    client.run()

if __name__ == "__main__":
    main()
