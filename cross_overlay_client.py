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
import itertools

class CrossingOverlayClient:
    def __init__(self, server_url, width=480, height=640, opacity=1.0, scale=1.0):
        self.server_url = server_url.rstrip('/') + "/predict"
        self.width = width
        self.height = height
        self.opacity = opacity
        self.scale = scale
        self.running = True
        
        # New settings
        self.mode = "heatmap"  # "heatmap", "points", "quads"
        self.threshold = 0.5
        
        print(f"Server URL: {self.server_url}")
        print(f"Capture: {width}x{height}")
        
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
        
        # Opacity
        self.capture_frame.bind('<Up>', lambda e: self._adjust_opacity(0.1))
        self.capture_frame.bind('<Down>', lambda e: self._adjust_opacity(-0.1))
        self.result_window.bind('<Up>', lambda e: self._adjust_opacity(0.1))
        self.result_window.bind('<Down>', lambda e: self._adjust_opacity(-0.1))
        
        # Threshold (Left/Right)
        self.capture_frame.bind('<Left>', lambda e: self._adjust_threshold(-0.05))
        self.capture_frame.bind('<Right>', lambda e: self._adjust_threshold(0.05))
        self.result_window.bind('<Left>', lambda e: self._adjust_threshold(-0.05))
        self.result_window.bind('<Right>', lambda e: self._adjust_threshold(0.05))
        
        # Mode switch (M)
        self.capture_frame.bind('<m>', lambda e: self._toggle_mode())
        self.result_window.bind('<m>', lambda e: self._toggle_mode())
        
        # Handle window close
        self.capture_frame.protocol("WM_DELETE_WINDOW", self._quit)
        self.result_window.protocol("WM_DELETE_WINDOW", self._quit)
        
        self.root.after(100, self._update_loop)
    
    def _adjust_opacity(self, delta):
        self.opacity = max(0.1, min(1.0, self.opacity + delta))
        self._update_status()
        
    def _adjust_threshold(self, delta):
        self.threshold = max(0.0, min(1.0, self.threshold + delta))
        self._update_status()
        
    def _toggle_mode(self):
        modes = ["heatmap", "points", "quads"]
        current_idx = modes.index(self.mode)
        self.mode = modes[(current_idx + 1) % len(modes)]
        self._update_status()
    
    def _update_status(self):
        self.status_var.set(f"FPS: {self.fps:.1f} | Lat: {self.latency:.0f}ms | Mode: {self.mode} | Thresh: {self.threshold:.2f}")
    
    def _quit(self):
        self.running = False
        self.root.destroy()
        
    def _find_peaks(self, heatmap, threshold):
        """Find local maxima above threshold."""
        thresh_val = int(threshold * 255)
        _, binary = cv2.threshold(heatmap, thresh_val, 255, cv2.THRESH_BINARY)
        
        points = []
        if cv2.countNonZero(binary) > 0:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points.append((cx, cy))
        return points

    def _is_convex(self, pts):
        """Check if 4 points form a convex polygon."""
        # Must be ordered first (e.g., clockwise)
        # 1. Compute centroid
        center = np.mean(pts, axis=0)
        # 2. Sort by angle from centroid
        sorted_pts = sorted(pts, key=lambda p: np.arctan2(p[1]-center[1], p[0]-center[0]))
        
        # 3. Check cross products
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        cp_signs = []
        for i in range(4):
            p1 = sorted_pts[i]
            p2 = sorted_pts[(i + 1) % 4]
            p3 = sorted_pts[(i + 2) % 4]
            cp = cross_product(p1, p2, p3)
            cp_signs.append(np.sign(cp))
            
        # If all cross products have same sign (and not 0), it's convex
        return all(s > 0 for s in cp_signs) or all(s < 0 for s in cp_signs), sorted_pts

    def _check_quad_constraints(self, quad_pts, all_points, margin=5):
        """
        Check:
        1. Convexity
        2. Corner angles >= 60 degrees
        3. No other points inside/near border
        """
        is_conv, ordered_pts = self._is_convex(quad_pts)
        if not is_conv:
            return False, []
            
        # Check angles
        for i in range(4):
            p1 = np.array(ordered_pts[i-1])
            p2 = np.array(ordered_pts[i])
            p3 = np.array(ordered_pts[(i+1)%4])
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Normalize
            l1 = np.linalg.norm(v1)
            l2 = np.linalg.norm(v2)
            if l1 == 0 or l2 == 0: return False, []
            
            angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)))
            if angle < 75 or angle > 105: # Strict "squarish" check
                return False, []

        # Check for points inside
        # Create a slightly smaller polygon for "inside" check to allow points ON corners
        poly_contour = np.array(ordered_pts, dtype=np.int32)
        
        # Margin check: Points shouldn't be too close to edges unless they are the corners
        # This is expensive, so simplified: Check if any other point is inside
        
        for p in all_points:
            # Skip if point is one of the corners
            if any(np.array_equal(p, c) for c in quad_pts):
                continue
                
            dist = cv2.pointPolygonTest(poly_contour, (float(p[0]), float(p[1])), True)
            if dist > -margin: # Inside or within margin distance outside
                return False, []
                
        return True, ordered_pts

    def _find_quads(self, points):
        """Find valid quads from points."""
        if len(points) < 4:
            return []
            
        # Limit points to avoid explosion (max 20 strongest/detected)
        # Since we don't have strength here easily without re-parsing heatmap, 
        # we just take first 25 found.
        search_points = points[:25] 
        
        valid_quads = []
        
        for quad_combo in itertools.combinations(search_points, 4):
            is_valid, ordered_pts = self._check_quad_constraints(quad_combo, points)
            if is_valid:
                valid_quads.append(ordered_pts)
                
        return valid_quads

    def _create_overlay_image(self, img, heatmap):
        """Blend original image with overlay based on mode."""
        h, w = img.shape[:2]
        if heatmap.shape != (h, w):
            heatmap = cv2.resize(heatmap, (w, h))
        
        result = img.copy()
        
        if self.mode == "heatmap":
            intensity = heatmap.astype(float) / 255.0
            overlay = np.zeros_like(img)
            overlay[:, :, 1] = 255
            blend_factor = (intensity * self.opacity)[:, :, np.newaxis]
            result = img * (1 - blend_factor) + overlay * blend_factor
            result = np.clip(result, 0, 255).astype(np.uint8)
            
        elif self.mode == "points":
            points = self._find_peaks(heatmap, self.threshold)
            result = cv2.addWeighted(result, 0.7, np.zeros_like(result), 0.3, 0)
            for (cx, cy) in points:
                cv2.circle(result, (cx, cy), 8, (0, 0, 255), 2)
                cv2.drawMarker(result, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 8, 2)
                
        elif self.mode == "quads":
            points = self._find_peaks(heatmap, self.threshold)
            quads = self._find_quads(points)
            
            # Draw quads
            overlay = result.copy()
            for q in quads:
                pts = np.array(q, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], (255, 255, 0)) # Cyan filled
                cv2.polylines(result, [pts], True, (0, 255, 255), 2) # Yellow border
                
            # Blend fill
            cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
            
            # Draw corner points too for reference
            for (cx, cy) in points:
                cv2.circle(result, (cx, cy), 4, (0, 0, 255), -1)

        return result

    def _process_loop(self):
        """Background thread for capture and server request."""
        sct = mss.mss()
        frame_count = 0
        monitor = {"left": 0, "top": 0, "width": self.width - 2, "height": self.height - 2}
        
        while self.running:
            try:
                t0 = time.time()
                
                with self.capture_lock:
                    x, y = self.last_capture_pos
                monitor["left"] = x + 1
                monitor["top"] = y + 30
                
                screenshot = sct.grab(monitor)
                img_bgra = np.array(screenshot)
                img_rgb = cv2.cvtColor(img_bgra[:, :, :3], cv2.COLOR_BGR2RGB)
                
                _, img_encoded = cv2.imencode('.jpg', img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 75])
                img_bytes = img_encoded.tobytes()
                
                t1 = time.time()
                
                files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
                try:
                    response = self.session.post(self.server_url, files=files, timeout=2.0)
                    
                    if response.status_code == 200:
                        t2 = time.time()
                        
                        heatmap_arr = np.frombuffer(response.content, np.uint8)
                        heatmap = cv2.imdecode(heatmap_arr, cv2.IMREAD_GRAYSCALE)
                        
                        if heatmap is None: continue

                        result = self._create_overlay_image(img_rgb, heatmap)
                        
                        t3 = time.time()
                        total_ms = (t3 - t0) * 1000
                        net_ms = (t2 - t1) * 1000
                        fps = 1000.0 / total_ms if total_ms > 0 else 0
                        
                        if frame_count % 30 == 0:
                            print(f"Net: {net_ms:.0f}ms | Total: {total_ms:.0f}ms | FPS: {fps:.1f}")
                        frame_count += 1
                        
                        try:
                            self.result_queue.put_nowait((result, fps, total_ms))
                        except queue.Full:
                            pass
                    else:
                        time.sleep(0.5)
                except requests.exceptions.RequestException:
                    time.sleep(1.0)
            except Exception:
                time.sleep(0.5)
    
    def _update_loop(self):
        """Main thread UI update."""
        if not self.running: return
        with self.capture_lock:
            self.last_capture_pos = (self.capture_frame.winfo_x(), self.capture_frame.winfo_y())
        try:
            result, fps, latency = self.result_queue.get_nowait()
            pil_img = Image.fromarray(result)
            pil_img = pil_img.resize((self.width, self.height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(pil_img)
            if self.result_image is None:
                self.result_image = self.result_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            else:
                self.result_canvas.itemconfig(self.result_image, image=self.photo)
            self.fps = fps
            self.latency = latency
            self._update_status()
        except queue.Empty: pass
        self.root.after(10, self._update_loop)
    
    def run(self):
        print("\nControls:")
        print("  Up/Down   - Adjust opacity")
        print("  Left/Right- Adjust threshold")
        print("  M         - Toggle mode (Heatmap/Points/Quads)")
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
