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
import json

class CrossingOverlayClient:
    def __init__(self, server_url, width=480, height=640, opacity=1.0, scale=1.0):
        self.server_url = server_url.rstrip('/') + "/predict"
        self.width = width
        self.height = height
        self.opacity = opacity
        self.scale = scale
        self.running = True
        
        # Settings
        self.mode = "quads"  # "heatmap", "points", "quads", "topdown"
        self.threshold = 0.1
        
        print(f"Server URL: {self.server_url}")
        print(f"Capture: {width}x{height}")
        
        self.session = requests.Session()
        self.result_queue = queue.Queue(maxsize=1)
        self.capture_lock = threading.Lock()
        self.last_capture_pos = (0, 0)
        
        self.fps = 0
        self.latency = 0
        
        self._setup_gui()
        
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
    def _setup_gui(self):
        self.root = tk.Tk()
        self.root.withdraw()
        
        self.capture_frame = tk.Toplevel(self.root)
        self.capture_frame.title("Capture Region")
        
        screen_w = self.capture_frame.winfo_screenwidth()
        screen_h = self.capture_frame.winfo_screenheight()
        center_x = (screen_w - self.width) // 2
        center_y = (screen_h - self.height) // 2
        self.capture_frame.geometry(f"{self.width}x{self.height}+{center_x}+{center_y}")
        
        self.capture_frame.attributes('-topmost', True)
        self.capture_frame.resizable(False, False)
        self.capture_frame.attributes('-transparentcolor', 'magenta')
        self.capture_frame.configure(bg='magenta')
        
        border = 1
        tk.Frame(self.capture_frame, bg='lime', height=border).place(x=0, y=0, relwidth=1)
        tk.Frame(self.capture_frame, bg='lime', height=border).place(x=0, rely=1, y=-border, relwidth=1)
        tk.Frame(self.capture_frame, bg='lime', width=border).place(x=0, y=0, relheight=1)
        tk.Frame(self.capture_frame, bg='lime', width=border).place(relx=1, x=-border, y=0, relheight=1)
        
        self.result_window = tk.Toplevel(self.root)
        self.result_window.title("Cloud Detection Result")
        result_y = (screen_h - self.height) // 2
        self.result_window.geometry(f"{self.width}x{self.height}+0+{result_y}")
        self.result_window.attributes('-topmost', True)
        self.result_window.resizable(False, False)
        
        self.result_canvas = tk.Canvas(self.result_window, width=self.width, height=self.height, bg='black')
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        self.result_image = None
        
        self.status_var = tk.StringVar(value="Connecting...")
        self.status_label = tk.Label(self.result_window, textvariable=self.status_var,
                                      bg='black', fg='lime', font=('Consolas', 9))
        self.status_label.place(x=5, y=5)
        
        self.capture_frame.bind('<Escape>', lambda e: self._quit())
        self.result_window.bind('<Escape>', lambda e: self._quit())
        
        self.capture_frame.bind('<Up>', lambda e: self._adjust_opacity(0.1))
        self.capture_frame.bind('<Down>', lambda e: self._adjust_opacity(-0.1))
        self.result_window.bind('<Up>', lambda e: self._adjust_opacity(0.1))
        self.result_window.bind('<Down>', lambda e: self._adjust_opacity(-0.1))
        
        self.capture_frame.bind('<Left>', lambda e: self._adjust_threshold(-0.05))
        self.capture_frame.bind('<Right>', lambda e: self._adjust_threshold(0.05))
        self.result_window.bind('<Left>', lambda e: self._adjust_threshold(-0.05))
        self.result_window.bind('<Right>', lambda e: self._adjust_threshold(0.05))
        
        self.capture_frame.bind('<m>', lambda e: self._toggle_mode())
        self.result_window.bind('<m>', lambda e: self._toggle_mode())
        
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
        modes = ["heatmap", "points", "quads", "topdown"]
        current_idx = modes.index(self.mode)
        self.mode = modes[(current_idx + 1) % len(modes)]
        self._update_status()
    
    def _update_status(self):
        self.status_var.set(f"FPS: {self.fps:.1f} | Lat: {self.latency:.0f}ms | Mode: {self.mode} | Thresh: {self.threshold:.2f}")
    
    def _quit(self):
        self.running = False
        self.root.destroy()

    def _compute_robust_homography(self, quads, h, w):
        """Compute H using aggregated vanishing points from all quads."""
        if not quads: return None
        
        lines_h = [] # Horizontal-ish lines
        lines_v = [] # Vertical-ish lines
        
        # 1. Extract lines from all quads
        for q in quads:
            pts = np.array(q, dtype=np.float32)
            # Quad order: 0-1 (Top), 1-2 (Right), 2-3 (Bottom), 3-0 (Left)
            # Horizontal pair: (0,1) and (3,2)
            lines_h.append((pts[0], pts[1]))
            lines_h.append((pts[3], pts[2]))
            # Vertical pair: (0,3) and (1,2)
            lines_v.append((pts[0], pts[3]))
            lines_v.append((pts[1], pts[2]))
            
        # 2. Find Vanishing Points (Intersection of line pairs)
        def intersect(l1, l2):
            p1, p2 = l1
            p3, p4 = l2
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6: return None # Parallel lines
            px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
            return np.array([px, py], dtype=np.float32)

        def get_vp_center(lines):
            intersections = []
            # Sample random pairs to find VPs
            # Since all 'h' lines should converge to VP_h, any pair gives a candidate
            # We take pairs that are far apart (from different quads) for stability
            if len(lines) < 2: return None
            
            # Try a few pairs
            for _ in range(20): 
                idx1, idx2 = np.random.choice(len(lines), 2, replace=False)
                vp = intersect(lines[idx1], lines[idx2])
                if vp is not None:
                    # Check if point is reasonable (not at infinity inside image)
                    # VP is usually far outside image
                    if np.linalg.norm(vp) > 1e6: continue # Too far/parallel
                    intersections.append(vp)
            
            if not intersections: return None
            
            # Filter outliers (Median geometric filter)
            pts = np.array(intersections)
            median = np.median(pts, axis=0)
            dists = np.linalg.norm(pts - median, axis=1)
            # Keep points within reasonable distance of median
            valid_pts = pts[dists < np.median(dists) * 2 + 100] 
            
            if len(valid_pts) == 0: return median
            return np.mean(valid_pts, axis=0)

        vp_h = get_vp_center(lines_h)
        vp_v = get_vp_center(lines_v)
        
        if vp_h is None or vp_v is None:
            # Fallback: Use single best quad logic if robust fails
            return None

        # 3. Construct Homography from VPs
        # The Horizon Line connects vp_h and vp_v.
        # H = K * R * K_inv. We don't know K.
        # Simpler: Use the VPs to define the rectification.
        
        # Projective transform part [h31, h32, h33] comes from the line at infinity.
        # The line at infinity l_inf in the image connects VP_h and VP_v.
        # l_inf = VP_h x VP_v (cross product)
        
        vh = np.append(vp_h, 1)
        vv = np.append(vp_v, 1)
        l_inf = np.cross(vh, vv)
        l_inf = l_inf / l_inf[2] # Normalize
        
        # H = [[1, 0, 0], [0, 1, 0], [l1, l2, 1]] roughly rectifies projective distortion
        H_rect = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [l_inf[0], l_inf[1], l_inf[2]]
        ], dtype=np.float32)
        
        # Note: This H rectifies metric properties but might shear/scale/rotate heavily.
        # We need to align it to a square.
        # Apply H_rect to the Best Quad, measure distortion, and correct affine.
        
        # Find best quad (largest)
        best_quad = None
        max_area = 0
        for q in quads:
            pts = np.array(q, np.float32)
            area = cv2.contourArea(pts)
            if area > max_area:
                max_area = area
                best_quad = pts
                
        if best_quad is None: return None
        
        # Transform best quad by H_rect
        rect_quad = cv2.perspectiveTransform(best_quad.reshape(1, -1, 2), H_rect).reshape(-1, 2)
        
        # Now we have a parallelogram (affine distorted). We want to map it to a square.
        # Source: rect_quad
        # Dest: Perfect Square with same Area/Centroid?
        # Simpler: Just solve H_affine from rect_quad -> Unit Square
        dst_sq = np.array([[0,0], [100,0], [100,100], [0,100]], dtype=np.float32)
        
        # H_affine: Affine map (3 points is enough, 4 is robust)
        # We can use findHomography because it generalizes
        H_affine, _ = cv2.findHomography(rect_quad, dst_sq)
        
        # Full H = H_affine * H_rect
        H_full = np.dot(H_affine, H_rect)
        
        return H_full

    def _create_overlay_image(self, img, response_data, request_mode):
        """Blend original image with overlay based on mode and server response."""
        result = img.copy()
        
        # Auto-detect if response is image
        is_image = response_data.startswith(b'\xff\xd8')
        
        if is_image:
            heatmap_arr = np.frombuffer(response_data, np.uint8)
            heatmap = cv2.imdecode(heatmap_arr, cv2.IMREAD_GRAYSCALE)
            
            if heatmap is not None:
                h, w = img.shape[:2]
                if heatmap.shape != (h, w):
                    heatmap = cv2.resize(heatmap, (w, h))
                    
                intensity = heatmap.astype(float) / 255.0
                overlay = np.zeros_like(img)
                overlay[:, :, 1] = 255
                blend_factor = (intensity * self.opacity)[:, :, np.newaxis]
                result = img * (1 - blend_factor) + overlay * blend_factor
                result = np.clip(result, 0, 255).astype(np.uint8)
                
        elif request_mode in ["points", "quads", "topdown"]:
            try:
                data = json.loads(response_data)
                points = data.get("points", [])
                
                if request_mode != "topdown":
                    # Draw points in overlay modes
                    result = cv2.addWeighted(result, 0.7, np.zeros_like(result), 0.3, 0)
                    for p in points:
                        cx, cy = p[0], p[1]
                        cv2.circle(result, (cx, cy), 8, (0, 0, 255), 2)
                        cv2.drawMarker(result, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 8, 2)
                    
                if request_mode == "quads":
                    quads = data.get("quads", [])
                    overlay = result.copy()
                    for q in quads:
                        pts = np.array(q, np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(overlay, [pts], (255, 255, 0)) 
                        cv2.polylines(result, [pts], True, (0, 255, 255), 2) 
                    
                    cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
                
                elif request_mode == "topdown":
                    quads = data.get("quads", [])
                    h, w = img.shape[:2]
                    
                    # Try Robust Averaging First
                    H = self._compute_robust_homography(quads, h, w)
                    
                    # Fallback to Single Best Quad if Robust fails (e.g. < 2 lines)
                    if H is None and quads:
                        best_quad = max(quads, key=lambda q: cv2.contourArea(np.array(q, np.float32)))
                        src_pts = np.array(best_quad, np.float32)
                        dst_pts = np.array([[0,0], [100,0], [100,100], [0,100]], dtype=np.float32)
                        H, _ = cv2.findHomography(src_pts, dst_pts)

                    if H is not None:
                        # 3. Center the view based on SCREEN center
                        screen_center = np.array([[[w/2, h/2]]], dtype=np.float32)
                        center_transformed = cv2.perspectiveTransform(screen_center, H)
                        cx, cy = center_transformed[0][0]
                        
                        tx = w/2 - cx
                        ty = h/2 - cy
                        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
                        H_final = np.dot(T, H)
                        
                        result = cv2.warpPerspective(img, H_final, (w, h))
                        cv2.circle(result, (w//2, h//2), 5, (0, 255, 0), -1)
                    
            except Exception as e:
                print(f"Err: {e}")
                pass

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
                
                request_mode = self.mode
                # Map 'topdown' to 'quads' for server
                server_mode = "quads" if request_mode == "topdown" else request_mode
                
                files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
                data = {'mode': server_mode, 'threshold': self.threshold}
                
                try:
                    response = self.session.post(self.server_url, files=files, data=data, timeout=2.0)
                    
                    if response.status_code == 200:
                        t2 = time.time()
                        
                        result = self._create_overlay_image(img_rgb, response.content, request_mode)
                        
                        t3 = time.time()
                        total_ms = (t3 - t0) * 1000
                        net_ms = (t2 - t1) * 1000
                        fps = 1000.0 / total_ms if total_ms > 0 else 0
                        
                        if frame_count % 60 == 0:
                            print(f"FPS: {fps:.1f} | Net: {net_ms:.0f}ms")
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
        print("  M         - Toggle mode (Heatmap/Points/Quads/TopDown)")
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
