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


def rotate_image_90(image, rotation):
    """Rotate image by 0, 90, 180, or 270 degrees."""
    if rotation == 0:
        return image
    elif rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def compute_overlap_score(prev_gray, curr_gray, dx, dy):
    """Compute overlap quality score using normalized cross-correlation."""
    h1, w1 = prev_gray.shape
    h2, w2 = curr_gray.shape
    
    # Determine overlap region
    x1_start = max(0, -dx)
    y1_start = max(0, -dy)
    x1_end = min(w1, w2 - dx)
    y1_end = min(h1, h2 - dy)
    
    x2_start = max(0, dx)
    y2_start = max(0, dy)
    x2_end = min(w2, w1 + dx)
    y2_end = min(h2, h1 + dy)
    
    # Check if there's valid overlap
    if x1_end <= x1_start or y1_end <= y1_start or x2_end <= x2_start or y2_end <= y2_start:
        return 0.0
    
    # Extract overlap regions
    overlap1 = prev_gray[y1_start:y1_end, x1_start:x1_end]
    overlap2 = curr_gray[y2_start:y2_end, x2_start:x2_end]
    
    if overlap1.size == 0 or overlap2.size == 0:
        return 0.0
    
    # Compute normalized cross-correlation
    overlap1_norm = (overlap1 - overlap1.mean()) / (overlap1.std() + 1e-6)
    overlap2_norm = (overlap2 - overlap2.mean()) / (overlap2.std() + 1e-6)
    
    correlation = np.mean(overlap1_norm * overlap2_norm)
    overlap_area = overlap1.size
    
    # Score is correlation weighted by overlap area
    return correlation * np.sqrt(overlap_area)


def find_best_alignment(prev_unwarp, curr_unwarp):
    """Find translation to align current frame to previous frame (no rotation).
    Returns: (rotation, dx, dy, score, aligned_image)
    """
    if prev_unwarp is None or curr_unwarp is None:
        return 0, 0, 0, 0.0, curr_unwarp

    prev_gray = cv2.cvtColor(prev_unwarp, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_unwarp, cv2.COLOR_BGR2GRAY)

    # Find translation using phase correlation
    if prev_gray.shape == curr_gray.shape:
        shift, response = cv2.phaseCorrelate(prev_gray.astype(np.float32), curr_gray.astype(np.float32))
        dx, dy = int(shift[0]), int(shift[1])
    else:
        # If shapes don't match, try feature matching
        orb = cv2.ORB_create(500)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)

        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
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
    
    # Compute overlap quality score
    score = compute_overlap_score(prev_gray, curr_gray, dx, dy)

    return 0, dx, dy, score, curr_unwarp


def create_alignment_visualization(prev_unwarp, curr_aligned, dx, dy, rotation, score):
    """Create a visualization showing the alignment between previous and current frames."""
    if prev_unwarp is None or curr_aligned is None:
        placeholder = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Waiting for frames...",
            (50, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return placeholder

    # Create overlay: previous in red channel, current in green channel
    h1, w1 = prev_unwarp.shape[:2]
    h2, w2 = curr_aligned.shape[:2]
    
    # Make canvas large enough for both with offset
    canvas_h = max(h1, h2 + abs(dy)) + 100
    canvas_w = max(w1, w2 + abs(dx)) + 100
    
    overlay = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Place previous frame (in cyan/blue tint)
    prev_gray = cv2.cvtColor(prev_unwarp, cv2.COLOR_BGR2GRAY)
    overlay[50:50+h1, 50:50+w1, 0] = prev_gray  # Blue channel
    overlay[50:50+h1, 50:50+w1, 1] = prev_gray  # Green channel
    
    # Place current frame with offset (in yellow/red tint)
    curr_gray = cv2.cvtColor(curr_aligned, cv2.COLOR_BGR2GRAY)
    y_offset = 50 + dy
    x_offset = 50 + dx
    
    if y_offset >= 0 and x_offset >= 0:
        y_end = min(y_offset + h2, canvas_h)
        x_end = min(x_offset + w2, canvas_w)
        h_crop = y_end - y_offset
        w_crop = x_end - x_offset
        overlay[y_offset:y_end, x_offset:x_end, 2] = curr_gray[:h_crop, :w_crop]  # Red channel
        overlay[y_offset:y_end, x_offset:x_end, 1] = curr_gray[:h_crop, :w_crop]  # Green channel (overlap = white)
    
    # Add alignment info
    cv2.putText(
        overlay,
        f"Translation: ({dx}, {dy})  Score: {score:.2f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    
    return overlay


def show_first_frame_with_grid(
    video_path: str,
    scale: float = 0.5,
    max_width: int = 1600,
    max_height: int = 900,
    tile_size: int = 100,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return

    detector = LineDetector(scale=scale)
    window_name = "Grid Detection & Unwarp"
    window_name_global = "Global Map"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_global, cv2.WINDOW_NORMAL)

    print("Controls: Right arrow → next frame, Left arrow → previous frame, q / Esc → quit")

    frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_unwarped = None
    global_map = None
    global_offset_x = 0
    global_offset_y = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Cannot read frame {frame_index}.")
            break

        frame_with_grid, lines = detector.detect_lines(frame)

        # Find and draw crossings
        crossings, grid_crossings = find_line_crossings(lines, frame.shape[:2])
        for x, y in crossings:
            cv2.circle(frame_with_grid, (x, y), 8, (0, 255, 0), 2)

        # Compute and apply homography for unwarp
        H = compute_unwarp_homography(grid_crossings, tile_size=tile_size)
        if H is not None and len(grid_crossings) > 0:
            # Compute output size based on grid extent
            max_i = max(i for (x, y), (i, j) in grid_crossings)
            max_j = max(j for (x, y), (i, j) in grid_crossings)
            output_width = (max_j + 1) * tile_size
            output_height = (max_i + 1) * tile_size
            unwarped = cv2.warpPerspective(frame, H, (output_width, output_height))
            
            # Draw grid on unwarped image for reference
            for i in range(max_i + 2):
                y_line = i * tile_size
                cv2.line(unwarped, (0, y_line), (output_width, y_line), (255, 0, 0), 1)
            for j in range(max_j + 2):
                x_line = j * tile_size
                cv2.line(unwarped, (x_line, 0), (x_line, output_height), (255, 0, 0), 1)
        else:
            unwarped = np.zeros((400, 400, 3), dtype=np.uint8)
            cv2.putText(
                unwarped,
                "Not enough crossings",
                (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        # Find alignment between previous and current unwarped frames
        rotation, dx, dy, score, curr_aligned = find_best_alignment(prev_unwarped, unwarped)
        alignment_vis = create_alignment_visualization(prev_unwarped, curr_aligned, dx, dy, rotation, score)

        # Update global map with current unwarped frame
        if global_map is None:
            # First frame: initialize global map
            global_map = unwarped.copy()
            global_offset_x = 0
            global_offset_y = 0
        else:
            # Subsequent frames: composite only non-overlapping regions
            h_curr, w_curr = unwarped.shape[:2]
            
            # Calculate where current frame should be placed in global coordinates
            paste_x = global_offset_x + dx
            paste_y = global_offset_y + dy
            
            # Expand global map if needed
            h_global, w_global = global_map.shape[:2]
            
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

        # Update previous frame
        prev_unwarped = unwarped.copy()

        # Match heights for side-by-side display by resizing (now three views)
        orig_h, orig_w = frame_with_grid.shape[:2]
        unwarp_h, unwarp_w = unwarped.shape[:2]
        align_h, align_w = alignment_vis.shape[:2]
        
        target_height = max(orig_h, unwarp_h, align_h)
        
        # Resize original to target height while preserving aspect ratio
        if orig_h != target_height and orig_h > 0:
            scale = target_height / orig_h
            new_width = int(round(orig_w * scale))
            frame_with_grid = cv2.resize(frame_with_grid, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        # Resize unwarped to target height while preserving aspect ratio
        if unwarp_h != target_height and unwarp_h > 0:
            scale = target_height / unwarp_h
            new_width = int(round(unwarp_w * scale))
            unwarped = cv2.resize(unwarped, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        # Resize alignment visualization to target height while preserving aspect ratio
        if align_h != target_height and align_h > 0:
            scale = target_height / align_h
            new_width = int(round(align_w * scale))
            alignment_vis = cv2.resize(alignment_vis, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        # Stack all three side by side
        combined = np.hstack([frame_with_grid, unwarped, alignment_vis])
        
        # Scale to fit display
        combined_h, combined_w = combined.shape[:2]
        _max_width = max_width if max_width > 0 else combined_w
        _max_height = max_height if max_height > 0 else combined_h
        
        scale_factor = min(_max_width / combined_w, _max_height / combined_h)
        if scale_factor <= 0:
            scale_factor = 1.0
        
        display_width = int(round(combined_w * scale_factor))
        display_height = int(round(combined_h * scale_factor))
        
        if display_width > 0 and display_height > 0 and scale_factor != 1.0:
            display_image = cv2.resize(combined, (display_width, display_height), interpolation=cv2.INTER_CUBIC)
        else:
            display_image = combined
        
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.resizeWindow(window_name, display_width, display_height)
        cv2.imshow(window_name, display_image)

        # Display global map
        if global_map is not None:
            global_h, global_w = global_map.shape[:2]
            global_scale = min(max_width / global_w, max_height / global_h) if global_w > 0 and global_h > 0 else 1.0
            if global_scale <= 0:
                global_scale = 1.0
            global_display_w = int(round(global_w * global_scale))
            global_display_h = int(round(global_h * global_scale))
            
            if global_display_w > 0 and global_display_h > 0 and global_scale != 1.0:
                global_display = cv2.resize(global_map, (global_display_w, global_display_h), interpolation=cv2.INTER_CUBIC)
            else:
                global_display = global_map
            
            if cv2.getWindowProperty(window_name_global, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.resizeWindow(window_name_global, global_display_w, global_display_h)
            cv2.imshow(window_name_global, global_display)

        key = cv2.waitKeyEx(0)
        if key == -1:
            continue

        if key in (27, ord("q")):
            break
        if key in (2555904, 65363):  # Right arrow (Windows, Linux)
            frame_index = min(frame_index + 1, total_frames - 1)
            continue
        if key in (2424832, 65361):  # Left arrow (Windows, Linux)
            frame_index = max(frame_index - 1, 0)
            continue

    cv2.destroyWindow(window_name)
    cv2.destroyWindow(window_name_global)
    cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display the first video frame with detected grid lines overlaid."
    )
    parser.add_argument("--video", type=str, default="video.mp4", help="Video file to open")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Downscale factor used during line detection (default=0.5)",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1600,
        help="Maximum display width while keeping aspect ratio (default=1600)",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=900,
        help="Maximum display height while keeping aspect ratio (default=900)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=100,
        help="Size of each tile in pixels for unwarped view (default=100)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    show_first_frame_with_grid(
        video_path=args.video,
        scale=args.scale,
        max_width=args.max_width,
        max_height=args.max_height,
        tile_size=args.tile_size,
    )

