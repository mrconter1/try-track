"""
Video frame sampler - samples random frames and sends to server for prediction.
Press R for new random frames, ESC to quit.
"""

import cv2
import numpy as np
import argparse
import requests
import random
import json
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor

class FrameSampler:
    def __init__(self, video_path, server_url, scale=0.5, threshold=0.1, count=4):
        self.video_path = video_path
        self.server_url = server_url
        self.scale = scale
        self.threshold = threshold
        self.count = count
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.current_frames = [] # List of (frame_idx, original_image, result_dict or None)
        self.loading = False
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        
        # Use a session for connection pooling
        self.session = requests.Session()
        
    def close(self):
        self.cap.release()
        self.session.close()

    def start_sampling(self):
        if self.loading: return
        self.loading = True
        
        # Pick random frames immediately so UI can show them
        indices = sorted(random.sample(range(self.total_frames), min(self.count, self.total_frames)))
        new_frames = []
        for idx in indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                new_frames.append({'idx': idx, 'img': frame, 'result': None})
        
        with self.lock:
            self.current_frames = new_frames
            
        # Start background processing for ALL frames in parallel
        threading.Thread(target=self._process_frames_parallel, args=(new_frames,), daemon=True).start()

    def _process_single_frame(self, item):
        idx = item['idx']
        frame = item['img']
        
        # Encode
        _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_bytes = img_encoded.tobytes()
        
        files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
        data = {'mode': 'quads', 'threshold': self.threshold}
        
        try:
            # print(f"Requesting frame {idx}...")
            t0 = time.time()
            response = self.session.post(self.server_url, files=files, data=data, timeout=60.0)
            dt = (time.time() - t0) * 1000
            
            if response.status_code == 200:
                result = json.loads(response.content)
                quads = len(result.get('quads', []))
                points = len(result.get('points', []))
                print(f"  -> Frame {idx}: {quads} quads, {points} points ({dt:.0f}ms)")
                
                with self.lock:
                    # Update result
                    for cf in self.current_frames:
                        if cf['idx'] == idx:
                            cf['result'] = result
                            break
            else:
                print(f"  -> Frame {idx}: Failed with status {response.status_code}")
                
        except Exception as e:
            print(f"Request error for frame {idx}: {e}")
        
        # Notify UI to redraw
        self.queue.put("update")

    def _process_frames_parallel(self, frames):
        # Use a ThreadPoolExecutor to send all requests at once
        with ThreadPoolExecutor(max_workers=len(frames)) as executor:
            executor.map(self._process_single_frame, frames)
            
        self.loading = False
        self.queue.put("done")

    def get_display_image(self):
        with self.lock:
            frames_to_draw = list(self.current_frames)
            
        if not frames_to_draw:
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        display_imgs = []
        for item in frames_to_draw:
            frame = item['img']
            result = item['result']
            idx = item['idx']
            
            display = frame.copy()
            
            if result:
                quads = result.get("quads", [])
                points = result.get("points", [])
                
                for q in quads:
                    pts = np.array(q, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display, [pts], True, (0, 255, 255), 2)
                
                for p in points:
                    cx, cy = int(p[0]), int(p[1])
                    cv2.drawMarker(display, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
            elif self.loading and result is None:
                 cv2.putText(display, "Loading...", (display.shape[1]//2 - 50, display.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

            h, w = display.shape[:2]
            new_w, new_h = int(w * self.scale), int(h * self.scale)
            resized = cv2.resize(display, (new_w, new_h))
            
            cv2.putText(resized, f"Frame {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            display_imgs.append(resized)
            
        # Stack horizontally
        return np.hstack(display_imgs)

def main():
    parser = argparse.ArgumentParser(description="Video frame sampler with server prediction")
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--server', default='http://localhost:8000', help='Server URL')
    parser.add_argument('--scale', type=float, default=0.5, help='Display scale')
    parser.add_argument('--threshold', type=float, default=0.1, help='Detection threshold')
    args = parser.parse_args()
    
    server_url = args.server.rstrip('/') + "/predict"
    print(f"Server: {server_url}")
    print(f"Video: {args.video}")
    
    try:
        sampler = FrameSampler(args.video, server_url, args.scale, args.threshold)
    except Exception as e:
        print(f"Error: {e}")
        return

    cv2.namedWindow("Video Sampler", cv2.WINDOW_NORMAL)
    # Start initial sample
    sampler.start_sampling()
    
    print("\nControls: R = new random frames, ESC = quit")
    
    while True:
        # Check if window is still open
        if cv2.getWindowProperty("Video Sampler", cv2.WND_PROP_VISIBLE) < 1:
            break

        # Check for updates
        try:
            while True: # Drain queue
                sampler.queue.get_nowait()
        except queue.Empty:
            pass
            
        grid = sampler.get_display_image()
        cv2.imshow("Video Sampler", grid)
        
        key = cv2.waitKey(50) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('r') or key == ord('R'):
            sampler.start_sampling()
            
    sampler.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
