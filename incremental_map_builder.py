import argparse
import cv2
import numpy as np

from line_detector import LineDetector


def line_intersection(rho1, theta1, rho2, theta2):
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


def find_line_crossings(lines, frame_shape):
    """Find all intersections between horizontal and vertical lines with grid indices."""
    if len(lines) < 2:
        return [], []

    h, w = frame_shape[:2]
    crossings = []

    # Group lines into horizontal and vertical
    horizontal = []
    vertical = []

    for rho, theta in lines:
        theta_deg = theta * 180 / np.pi
        theta_deg = theta_deg % 180

        # Angles close to 90 degrees are horizontal
        if 45 < theta_deg < 135:
            horizontal.append((rho, theta))
        # Angles close to 0 or 180 degrees are vertical
        else:
            vertical.append((rho, theta))

    # Sort horizontal lines top-to-bottom
    def get_y_intercept(line, width):
        rho, theta = line
        if np.sin(theta) != 0:
            return (rho - (width / 2) * np.cos(theta)) / np.sin(theta)
        return float('inf')

    horizontal.sort(key=lambda line: get_y_intercept(line, w))

    # Sort vertical lines left-to-right
    def get_x_intercept(line, height):
        rho, theta = line
        if np.cos(theta) != 0:
            return (rho - (height / 2) * np.sin(theta)) / np.cos(theta)
        return float('inf')

    vertical.sort(key=lambda line: get_x_intercept(line, h))

    # Find all intersections with grid indices
    grid_crossings = []
    for i, (rho_h, theta_h) in enumerate(horizontal):
        for j, (rho_v, theta_v) in enumerate(vertical):
            intersection = line_intersection(rho_h, theta_h, rho_v, theta_v)
            if intersection is not None:
                x, y = intersection
                # Only keep intersections within frame bounds
                if 0 <= x < w and 0 <= y < h:
                    crossings.append((x, y))
                    grid_crossings.append(((x, y), (i, j)))

    return crossings, grid_crossings


def compute_unwarp_homography(grid_crossings, tile_size=100):
    """Compute homography to unwarp perspective to top-down view."""
    if len(grid_crossings) < 4:
        return None

    src_points = []
    dst_points = []

    for (x, y), (i, j) in grid_crossings:
        src_points.append([x, y])
        dst_points.append([j * tile_size, i * tile_size])

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return H


def build_incremental_map(
    video_path: str,
    scale: float = 0.5,
    tile_size: int = 100,
    max_frames: int = None,
    frame_step: int = 1,
):
    """Build global map incrementally: detect -> unwarp -> find offset -> composite."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return

    detector = LineDetector(scale=scale)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    print(f"Processing up to {total_frames} frames (step={frame_step})...")
    
    global_map = None
    prev_unwarped = None
    global_offset_x = 0
    global_offset_y = 0
    frame_index = 0
    
    window_name = "Incremental Map Builder"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while frame_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        # Detect lines
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        _, lines = detector.detect_lines(frame)
        sys.stdout = old_stdout
        
        # Find crossings and compute homography
        crossings, grid_crossings = find_line_crossings(lines, frame.shape[:2])
        H = compute_unwarp_homography(grid_crossings, tile_size=tile_size)
        
        if H is not None and len(grid_crossings) > 0:
            # Compute output size
            max_i = max(i for (x, y), (i, j) in grid_crossings)
            max_j = max(j for (x, y), (i, j) in grid_crossings)
            output_width = (max_j + 1) * tile_size
            output_height = (max_i + 1) * tile_size
            
            # Unwarp
            unwarped = cv2.warpPerspective(frame, H, (output_width, output_height))
            
            if global_map is None:
                # First frame: initialize
                global_map = unwarped.copy()
                prev_unwarped = unwarped.copy()
                global_offset_x = 0
                global_offset_y = 0
                print(f"Frame {frame_index}: initialized map {output_width}x{output_height}")
            else:
                # Find translation offset using phase correlation
                prev_gray = cv2.cvtColor(prev_unwarped, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(unwarped, cv2.COLOR_BGR2GRAY)
                
                if prev_gray.shape == curr_gray.shape:
                    shift, response = cv2.phaseCorrelate(prev_gray.astype(np.float32), curr_gray.astype(np.float32))
                    dx, dy = int(shift[0]), int(shift[1])
                else:
                    # Fallback: use ORB if sizes differ
                    orb = cv2.ORB_create(500)
                    kp1, des1 = orb.detectAndCompute(prev_gray, None)
                    kp2, des2 = orb.detectAndCompute(curr_gray, None)
                    
                    if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        matches = bf.match(des1, des2)
                        matches = sorted(matches, key=lambda x: x.distance)
                        
                        if len(matches) > 10:
                            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:10]])
                            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:10]])
                            offsets = pts1 - pts2
                            dx, dy = int(np.median(offsets[:, 0])), int(np.median(offsets[:, 1]))
                        else:
                            dx, dy = 0, 0
                    else:
                        dx, dy = 0, 0
                
                # Calculate where current frame should be placed in global coordinates
                paste_x = global_offset_x + dx
                paste_y = global_offset_y + dy
                
                # Expand global map if needed
                h_global, w_global = global_map.shape[:2]
                h_curr, w_curr = unwarped.shape[:2]
                
                new_left = min(0, paste_x)
                new_top = min(0, paste_y)
                new_right = max(w_global, paste_x + w_curr)
                new_bottom = max(h_global, paste_y + h_curr)
                
                new_width = new_right - new_left
                new_height = new_bottom - new_top
                
                if new_width != w_global or new_height != h_global or new_left < 0 or new_top < 0:
                    # Need to expand canvas
                    expanded_map = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                    
                    # Copy existing global map to expanded canvas
                    offset_x_in_expanded = -new_left
                    offset_y_in_expanded = -new_top
                    expanded_map[offset_y_in_expanded:offset_y_in_expanded + h_global,
                               offset_x_in_expanded:offset_x_in_expanded + w_global] = global_map
                    
                    global_map = expanded_map
                    global_offset_x += offset_x_in_expanded
                    global_offset_y += offset_y_in_expanded
                    paste_x += offset_x_in_expanded
                    paste_y += offset_y_in_expanded
                
                # Create mask for areas that are already filled in global map
                global_gray = cv2.cvtColor(global_map, cv2.COLOR_BGR2GRAY)
                filled_mask = (global_gray > 0).astype(np.uint8)
                
                # Paste current frame, only overwriting black (unfilled) areas
                for y in range(h_curr):
                    for x in range(w_curr):
                        global_y = paste_y + y
                        global_x = paste_x + x
                        if 0 <= global_y < global_map.shape[0] and 0 <= global_x < global_map.shape[1]:
                            if filled_mask[global_y, global_x] == 0:
                                global_map[global_y, global_x] = unwarped[y, x]
                
                # Update cumulative offset for next frame
                global_offset_x = paste_x
                global_offset_y = paste_y
                
                print(f"Frame {frame_index}: offset=({dx},{dy}), map size={global_map.shape[1]}x{global_map.shape[0]}")
                
                prev_unwarped = unwarped.copy()
            
            # Display current global map
            if global_map is not None:
                h, w = global_map.shape[:2]
                max_display = 1200
                if w > max_display or h > max_display:
                    scale_factor = min(max_display / w, max_display / h)
                    display = cv2.resize(global_map, (int(w * scale_factor), int(h * scale_factor)))
                else:
                    display = global_map
                
                cv2.imshow(window_name, display)
                cv2.waitKey(1)
        else:
            print(f"Frame {frame_index}: insufficient crossings, skipping")
        
        frame_index += frame_step
    
    cap.release()
    
    if global_map is None:
        print("No valid frames to build map.")
        cv2.destroyAllWindows()
        return
    
    print(f"\nFinal map size: {global_map.shape[1]}x{global_map.shape[0]}")
    cv2.imwrite("incremental_map.png", global_map)
    print("Saved map to incremental_map.png")
    
    # Display final result
    h, w = global_map.shape[:2]
    max_display = 1600
    if w > max_display or h > max_display:
        scale_factor = min(max_display / w, max_display / h)
        display = cv2.resize(global_map, (int(w * scale_factor), int(h * scale_factor)))
    else:
        display = global_map
    
    cv2.imshow(window_name, display)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build incremental map: detect grid -> unwarp -> find translation -> composite"
    )
    parser.add_argument("--video", type=str, default="video.mp4", help="Video file to process")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Downscale factor for line detection (default=0.5)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=100,
        help="Size of each tile in pixels for unwarped view (default=100)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default=all)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame (default=1, process all frames)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_incremental_map(
        video_path=args.video,
        scale=args.scale,
        tile_size=args.tile_size,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
    )

