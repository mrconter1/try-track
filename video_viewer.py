"""
Video viewer with perspective unwarp.
Left: Original frame with detected quads
Right: Top-down unwarped view with grid overlay
Use slider or arrow keys to navigate frames.
"""

import cv2
import numpy as np
import argparse
import requests
import json
import threading

class VideoViewer:
    def __init__(self, video_path, server_url, threshold=0.1):
        self.video_path = video_path
        self.server_url = server_url
        self.threshold = threshold
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.current_frame_idx = 0
        self.current_frame = None
        self.current_result = None
        self.loading = False
        self.session = requests.Session()
        
        # Output size for unwarped view
        self.unwarp_size = 600
        
        # Alignment comparison mode
        self.previous_unwarped = None
        self.current_unwarped = None
        self.offset_x = 0
        self.offset_y = 0
        self.rotation = 0.0  # degrees
        self.compare_mode = False
        
    def close(self):
        self.cap.release()
        self.session.close()
        
    def get_frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def request_prediction(self, frame):
        """Send frame to server and get quads."""
        _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_bytes = img_encoded.tobytes()
        
        files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
        data = {'mode': 'quads', 'threshold': self.threshold}
        
        try:
            response = self.session.post(self.server_url, files=files, data=data, timeout=30.0)
            if response.status_code == 200:
                return json.loads(response.content)
        except Exception as e:
            print(f"Request error: {e}")
        return None
    
    def compute_homography(self, quads):
        """Compute homography from quads to create top-down view of entire image."""
        if not quads:
            return None
            
        # Use the largest quad to define the perspective
        best_quad = max(quads, key=lambda q: cv2.contourArea(np.array(q, np.float32)))
        src_pts = np.array(best_quad, dtype=np.float32)
        
        # Compute the side lengths of the quad to estimate tile size
        side1 = np.linalg.norm(src_pts[1] - src_pts[0])
        side2 = np.linalg.norm(src_pts[2] - src_pts[1])
        tile_size = (side1 + side2) / 2  # Average side length
        
        # Scale factor: how many pixels per tile in output
        output_tile_size = 80
        
        # Destination: square tile centered in output
        cx, cy = self.unwarp_size // 2, self.unwarp_size // 2
        half = output_tile_size // 2
        
        dst_pts = np.array([
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half]
        ], dtype=np.float32)
        
        H, _ = cv2.findHomography(src_pts, dst_pts)
        return H
    
    def draw_original(self, frame, result):
        """Draw quads and points on original frame."""
        display = frame.copy()
        
        if result:
            quads = result.get("quads", [])
            points = result.get("points", [])
            
            # Draw quads
            for q in quads:
                pts = np.array(q, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts], True, (0, 255, 255), 2)
            
            # Draw points
            for p in points:
                cx, cy = int(p[0]), int(p[1])
                cv2.drawMarker(display, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        
        return display
    
    def draw_unwarped(self, frame, result, with_alpha=False):
        """Create top-down unwarped view of entire image with transformed quads."""
        if not result or not result.get("quads"):
            # No quads - show placeholder
            placeholder = np.zeros((self.unwarp_size, self.unwarp_size, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No quads", (self.unwarp_size//2 - 50, self.unwarp_size//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            if with_alpha:
                alpha = np.zeros((self.unwarp_size, self.unwarp_size), dtype=np.uint8)
                return cv2.merge([placeholder, alpha])
            return placeholder
        
        quads = result.get("quads", [])
        H = self.compute_homography(quads)
        
        if H is None:
            placeholder = np.zeros((self.unwarp_size, self.unwarp_size, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No H", (self.unwarp_size//2 - 30, self.unwarp_size//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            if with_alpha:
                alpha = np.zeros((self.unwarp_size, self.unwarp_size), dtype=np.uint8)
                return cv2.merge([placeholder, alpha])
            return placeholder
        
        # Warp the ENTIRE image
        unwarped = cv2.warpPerspective(frame, H, (self.unwarp_size, self.unwarp_size))
        
        # Create alpha mask by warping a white image
        if with_alpha:
            mask_src = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
            alpha = cv2.warpPerspective(mask_src, H, (self.unwarp_size, self.unwarp_size))
        
        # Transform and draw all quads on unwarped view
        for q in quads:
            src_pts = np.array(q, dtype=np.float32).reshape(-1, 1, 2)
            dst_pts = cv2.perspectiveTransform(src_pts, H)
            dst_pts = dst_pts.reshape(-1, 2).astype(np.int32)
            cv2.polylines(unwarped, [dst_pts], True, (0, 255, 255), 2)
        
        if with_alpha:
            return cv2.merge([unwarped, alpha])
        return unwarped
    
    def process_frame(self, idx):
        """Load frame and get prediction."""
        self.loading = True
        frame = self.get_frame(idx)
        if frame is None:
            self.loading = False
            return
        
        # Get prediction BEFORE updating display
        result = self.request_prediction(frame)
        
        # Only save previous if current is a valid unwarped (not placeholder)
        if self.current_unwarped is not None and self.current_result and self.current_result.get("quads"):
            self.previous_unwarped = self.current_unwarped.copy()
        
        # Update both together atomically
        self.current_frame = frame
        self.current_frame_idx = idx
        self.current_result = result
        
        # Only update current_unwarped if we have valid quads (with alpha for comparison)
        if result and result.get("quads"):
            self.current_unwarped = self.draw_unwarped(frame, result, with_alpha=True)
        
        if result:
            quads = len(result.get('quads', []))
            points = len(result.get('points', []))
            print(f"Frame {idx}: {quads} quads, {points} points")
        
        self.loading = False
    
    def create_display(self):
        """Create combined display with original and unwarped views."""
        if self.current_frame is None:
            return np.zeros((self.frame_height, self.frame_width * 2, 3), dtype=np.uint8)
        
        # Original with annotations
        original = self.draw_original(self.current_frame, self.current_result)
        
        # Unwarped view
        pixel_diff = None
        overlap_pct = None
        if self.compare_mode and self.previous_unwarped is not None and self.current_unwarped is not None:
            # Compare mode: blend previous and current with offset
            # Images are BGRA (4 channels)
            pad = max(abs(self.offset_x), abs(self.offset_y)) + 50
            h, w = self.previous_unwarped.shape[:2]
            canvas_h, canvas_w = h + 2 * pad, w + 2 * pad
            
            # Separate BGR and Alpha
            prev_bgr = self.previous_unwarped[:, :, :3]
            prev_alpha = self.previous_unwarped[:, :, 3]
            curr_bgr = self.current_unwarped[:, :, :3]
            curr_alpha = self.current_unwarped[:, :, 3]
            
            # Apply rotation to current frame if needed
            if self.rotation != 0:
                center = (w // 2, h // 2)
                M_rot = cv2.getRotationMatrix2D(center, -self.rotation, 1.0)
                curr_bgr = cv2.warpAffine(curr_bgr, M_rot, (w, h))
                curr_alpha = cv2.warpAffine(curr_alpha, M_rot, (w, h))
            
            # Place previous frame centered on canvas
            prev_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            prev_canvas[pad:pad+h, pad:pad+w] = prev_bgr
            prev_alpha_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            prev_alpha_canvas[pad:pad+h, pad:pad+w] = prev_alpha
            
            # Place current frame with offset on canvas
            curr_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            curr_alpha_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            y1, x1 = pad + self.offset_y, pad + self.offset_x
            curr_canvas[y1:y1+h, x1:x1+w] = curr_bgr
            curr_alpha_canvas[y1:y1+h, x1:x1+w] = curr_alpha
            
            # Calculate overlap region and pixel difference (only where both have valid alpha)
            prev_y1, prev_y2 = pad, pad + h
            prev_x1, prev_x2 = pad, pad + w
            curr_y1, curr_y2 = y1, y1 + h
            curr_x1, curr_x2 = x1, x1 + w
            ov_y1 = max(prev_y1, curr_y1)
            ov_y2 = min(prev_y2, curr_y2)
            ov_x1 = max(prev_x1, curr_x1)
            ov_x2 = min(prev_x2, curr_x2)
            
            # Count total valid pixels in each frame
            prev_total_valid = np.sum(prev_alpha_canvas > 0)
            curr_total_valid = np.sum(curr_alpha_canvas > 0)
            
            if ov_y2 > ov_y1 and ov_x2 > ov_x1:
                prev_region = prev_canvas[ov_y1:ov_y2, ov_x1:ov_x2].astype(np.int32)
                curr_region = curr_canvas[ov_y1:ov_y2, ov_x1:ov_x2].astype(np.int32)
                prev_alpha_region = prev_alpha_canvas[ov_y1:ov_y2, ov_x1:ov_x2]
                curr_alpha_region = curr_alpha_canvas[ov_y1:ov_y2, ov_x1:ov_x2]
                # Mask: only pixels where BOTH have valid alpha (> 0)
                valid_mask = (prev_alpha_region > 0) & (curr_alpha_region > 0)
                num_valid = np.sum(valid_mask)
                
                # Calculate overlap percentage
                min_valid = min(prev_total_valid, curr_total_valid)
                if min_valid > 0:
                    overlap_pct = (num_valid / min_valid) * 100
                
                if num_valid > 0:
                    diff = np.abs(prev_region - curr_region)
                    # Apply mask to each channel
                    masked_diff = diff[valid_mask]
                    pixel_diff = np.sum(masked_diff) / num_valid
            
            # Blend at 50% opacity
            blended = cv2.addWeighted(prev_canvas, 0.5, curr_canvas, 0.5, 0)
            
            # Crop back to fixed display size centered on the previous frame
            crop_pad = 50
            unwarped = blended[pad-crop_pad:pad+h+crop_pad, pad-crop_pad:pad+w+crop_pad]
        else:
            # Normal mode: show current frame's unwarped (with placeholder if no quads)
            unwarped = self.draw_unwarped(self.current_frame, self.current_result)
        
        # Resize original to match height with unwarped
        scale = self.unwarp_size / original.shape[0]
        original_resized = cv2.resize(original, None, fx=scale, fy=scale)
        
        # Pad unwarped to match height if needed
        if original_resized.shape[0] != unwarped.shape[0]:
            diff = original_resized.shape[0] - unwarped.shape[0]
            if diff > 0:
                unwarped = cv2.copyMakeBorder(unwarped, 0, diff, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            else:
                original_resized = cv2.copyMakeBorder(original_resized, 0, -diff, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        
        # Combine side by side
        combined = np.hstack([original_resized, unwarped])
        
        # Add frame info
        cv2.putText(combined, f"Frame {self.current_frame_idx}/{self.total_frames}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.loading:
            cv2.putText(combined, "Loading...", (combined.shape[1]//2, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Show compare mode status
        if self.compare_mode:
            cv2.putText(combined, f"COMPARE: offset ({self.offset_x}, {self.offset_y}) rot {self.rotation:.1f}deg", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            # Show pixel difference on top right of right canvas
            if pixel_diff is not None:
                diff_text = f"Diff/px: {pixel_diff:.2f}"
                text_size = cv2.getTextSize(diff_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = combined.shape[1] - text_size[0] - 10
                cv2.putText(combined, diff_text, (text_x, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # Show overlap percentage
            if overlap_pct is not None:
                overlap_text = f"Overlap: {overlap_pct:.1f}%"
                text_size = cv2.getTextSize(overlap_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = combined.shape[1] - text_size[0] - 10
                cv2.putText(combined, overlap_text, (text_x, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return combined


def on_trackbar(val):
    """Trackbar callback - will be handled in main loop."""
    pass


def main():
    parser = argparse.ArgumentParser(description="Video viewer with perspective unwarp")
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--server', default='http://localhost:8000', help='Server URL')
    parser.add_argument('--threshold', type=float, default=0.1, help='Detection threshold')
    parser.add_argument('--start', type=int, default=0, help='Start frame')
    parser.add_argument('--compare', action='store_true', help='Start with compare mode on')
    args = parser.parse_args()
    
    server_url = args.server.rstrip('/') + "/predict"
    print(f"Server: {server_url}")
    print(f"Video: {args.video}")
    
    try:
        viewer = VideoViewer(args.video, server_url, args.threshold)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    print(f"Total frames: {viewer.total_frames}")
    print("\nControls:")
    print("  Slider: Jump to frame")
    print("  A/D: Step -1/+1 frame")
    print("  W/S: Step -10/+10 frames")
    print("  Q/E: First/Last frame")
    print("  Space: Toggle auto-play")
    print("  C: Toggle compare mode (overlay prev/current)")
    print("  Arrow keys: Move current frame offset (in compare mode)")
    print("  R/T: Rotate current frame clockwise/counter-clockwise")
    print("  Z: Reset offset and rotation")
    print("  ESC: Quit")
    
    window_name = "Video Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Frame", window_name, 0, viewer.total_frames - 1, on_trackbar)
    
    # Set compare mode if requested
    if args.compare:
        viewer.compare_mode = True
        print("Compare mode: ON")
    
    # Initial frame - load start frame and previous for compare mode
    start_frame = min(args.start, viewer.total_frames - 1)
    if start_frame > 0:
        viewer.process_frame(start_frame - 1)
    viewer.process_frame(start_frame)
    cv2.setTrackbarPos("Frame", window_name, start_frame)
    
    last_trackbar_pos = start_frame
    auto_play = False
    
    while True:
        # Check if window closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        
        # Check trackbar
        trackbar_pos = cv2.getTrackbarPos("Frame", window_name)
        if trackbar_pos != last_trackbar_pos and trackbar_pos != viewer.current_frame_idx:
            if not viewer.loading:
                threading.Thread(target=viewer.process_frame, args=(trackbar_pos,), daemon=True).start()
            last_trackbar_pos = trackbar_pos
        
        # Update trackbar to match current frame
        if viewer.current_frame_idx != trackbar_pos:
            cv2.setTrackbarPos("Frame", window_name, viewer.current_frame_idx)
            last_trackbar_pos = viewer.current_frame_idx
        
        # Display
        display = viewer.create_display()
        cv2.imshow(window_name, display)
        
        # Auto-play
        if auto_play and not viewer.loading:
            next_idx = (viewer.current_frame_idx + 1) % viewer.total_frames
            threading.Thread(target=viewer.process_frame, args=(next_idx,), daemon=True).start()
        
        # Handle keys (use waitKeyEx for arrow key support)
        key = cv2.waitKeyEx(50)
        
        if key == 27:  # ESC
            break
        elif key == 32:  # Space
            auto_play = not auto_play
            print(f"Auto-play: {'ON' if auto_play else 'OFF'}")
        elif key == ord('a') or key == ord('A'):  # A - step back
            if not viewer.loading:
                new_idx = max(0, viewer.current_frame_idx - 1)
                threading.Thread(target=viewer.process_frame, args=(new_idx,), daemon=True).start()
        elif key == ord('d') or key == ord('D'):  # D - step forward
            if not viewer.loading:
                new_idx = min(viewer.total_frames - 1, viewer.current_frame_idx + 1)
                threading.Thread(target=viewer.process_frame, args=(new_idx,), daemon=True).start()
        elif key == ord('w') or key == ord('W'):  # W - step back 10
            if not viewer.loading:
                new_idx = max(0, viewer.current_frame_idx - 10)
                threading.Thread(target=viewer.process_frame, args=(new_idx,), daemon=True).start()
        elif key == ord('s') or key == ord('S'):  # S - step forward 10
            if not viewer.loading:
                new_idx = min(viewer.total_frames - 1, viewer.current_frame_idx + 10)
                threading.Thread(target=viewer.process_frame, args=(new_idx,), daemon=True).start()
        elif key == ord('q') or key == ord('Q'):  # Q - first frame
            if not viewer.loading:
                threading.Thread(target=viewer.process_frame, args=(0,), daemon=True).start()
        elif key == ord('e') or key == ord('E'):  # E - last frame
            if not viewer.loading:
                threading.Thread(target=viewer.process_frame, args=(viewer.total_frames - 1,), daemon=True).start()
        elif key == ord('c') or key == ord('C'):  # C - toggle compare mode
            viewer.compare_mode = not viewer.compare_mode
            print(f"Compare mode: {'ON' if viewer.compare_mode else 'OFF'}")
        elif key == ord('r') or key == ord('R'):  # R - rotate clockwise
            viewer.rotation = (viewer.rotation + 90.0) % 360.0
        elif key == ord('t') or key == ord('T'):  # T - rotate counter-clockwise
            viewer.rotation = (viewer.rotation - 90.0) % 360.0
        elif key == ord('z') or key == ord('Z'):  # Z - reset offset and rotation
            viewer.offset_x = 0
            viewer.offset_y = 0
            viewer.rotation = 0.0
            print("Reset: offset (0, 0), rotation 0.0")
        elif key == 2490368:  # Up arrow (Windows)
            viewer.offset_y -= 80  # Move one tile
        elif key == 2621440:  # Down arrow (Windows)
            viewer.offset_y += 80  # Move one tile
        elif key == 2424832:  # Left arrow (Windows)
            viewer.offset_x -= 80  # Move one tile
        elif key == 2555904:  # Right arrow (Windows)
            viewer.offset_x += 80  # Move one tile
    
    viewer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

