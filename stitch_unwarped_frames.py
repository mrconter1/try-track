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


def stitch_unwarped_frames(
    video_path: str,
    scale: float = 0.5,
    tile_size: int = 100,
    max_frames: int = None,
    frame_step: int = 1,
    batch_size: int = 3,
):
    """Process video frames: detect grid -> unwarp -> incrementally stitch in batches."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return

    detector = LineDetector(scale=scale)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    print(f"Processing up to {total_frames} frames (step={frame_step}, batch_size={batch_size})...")
    
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    stitched_result = None
    frame_index = 0
    recent_frames = []  # Keep last few unwarped frames for overlap stitching
    
    window_name = "Incremental Stitching"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while frame_index < total_frames:
        batch_frames = []
        
        # Collect a batch of unwarped frames
        for _ in range(batch_size):
            if frame_index >= total_frames:
                break
            
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
                batch_frames.append(unwarped)
                
                print(f"Frame {frame_index}: unwarped {len(grid_crossings)} crossings -> {output_width}x{output_height}")
            else:
                print(f"Frame {frame_index}: insufficient crossings, skipping")
            
            frame_index += frame_step
        
        # Stitch the batch
        if len(batch_frames) == 0:
            continue
        
        if stitched_result is None:
            # First batch: initialize with first frame or stitch the batch
            if len(batch_frames) == 1:
                stitched_result = batch_frames[0]
                recent_frames = batch_frames.copy()
                print(f"Initialized with first frame")
            else:
                print(f"Stitching initial batch of {len(batch_frames)} frames...", flush=True)
                status, stitched = stitcher.stitch(batch_frames)
                if status == cv2.Stitcher_OK:
                    stitched_result = stitched
                    recent_frames = batch_frames.copy()
                    print(f"  -> Success! Result size: {stitched_result.shape[1]}x{stitched_result.shape[0]}")
                else:
                    print(f"  -> Failed (status={status}), using first frame only")
                    stitched_result = batch_frames[0]
                    recent_frames = batch_frames.copy()
        else:
            # Stitch new batch with recent frames (not entire result)
            print(f"Stitching {len(batch_frames)} new frames with {len(recent_frames)} recent frames...", flush=True)
            frames_to_stitch = recent_frames + batch_frames
            
            try:
                status, stitched_batch = stitcher.stitch(frames_to_stitch)
                
                if status == cv2.Stitcher_OK:
                    # Successfully stitched the overlap - now composite onto global result
                    # For simplicity, just replace the entire result (proper compositing would preserve edges)
                    stitched_result = stitched_batch
                    recent_frames = batch_frames.copy()
                    print(f"  -> Success! Result size: {stitched_result.shape[1]}x{stitched_result.shape[0]}")
                else:
                    print(f"  -> Failed (status={status}), keeping previous result")
                    # Still update recent frames for next attempt
                    recent_frames = batch_frames.copy()
            except Exception as e:
                print(f"  -> Exception during stitching: {e}, keeping previous result")
                recent_frames = batch_frames.copy()
        
        # Display current result
        if stitched_result is not None:
            h, w = stitched_result.shape[:2]
            max_display = 1200
            if w > max_display or h > max_display:
                scale_factor = min(max_display / w, max_display / h)
                display = cv2.resize(stitched_result, (int(w * scale_factor), int(h * scale_factor)))
            else:
                display = stitched_result
            
            cv2.imshow(window_name, display)
            cv2.waitKey(1)
    
    cap.release()
    
    if stitched_result is None:
        print("No valid frames to stitch.")
        cv2.destroyAllWindows()
        return
    
    print(f"\nFinal stitched result size: {stitched_result.shape[1]}x{stitched_result.shape[0]}")
    cv2.imwrite("stitched_output.png", stitched_result)
    print("Saved stitched result to stitched_output.png")
    
    # Display final result
    h, w = stitched_result.shape[:2]
    max_display = 1600
    if w > max_display or h > max_display:
        scale_factor = min(max_display / w, max_display / h)
        display = cv2.resize(stitched_result, (int(w * scale_factor), int(h * scale_factor)))
    else:
        display = stitched_result
    
    cv2.imshow(window_name, display)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch video frames: detect grid -> unwarp -> stitch with OpenCV Stitcher"
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Number of frames to add at a time during incremental stitching (default=3)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stitch_unwarped_frames(
        video_path=args.video,
        scale=args.scale,
        tile_size=args.tile_size,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        batch_size=args.batch_size,
    )

