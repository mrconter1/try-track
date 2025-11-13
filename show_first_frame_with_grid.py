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
    window_name_original = "Original with Grid"
    window_name_unwarp = "Unwarped Top-Down"
    cv2.namedWindow(window_name_original, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_unwarp, cv2.WINDOW_NORMAL)

    print("Controls: Right arrow → next frame, Left arrow → previous frame, q / Esc → quit")

    frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

        frame_height, frame_width = frame_with_grid.shape[:2]
        _max_width = max_width if max_width > 0 else frame_width
        _max_height = max_height if max_height > 0 else frame_height

        scale_factor = min(_max_width / frame_width, _max_height / frame_height)
        if scale_factor <= 0:
            scale_factor = 1.0

        display_width = int(round(frame_width * scale_factor))
        display_height = int(round(frame_height * scale_factor))

        if display_width <= 0 or display_height <= 0:
            display_image = frame_with_grid
            display_width, display_height = frame_width, frame_height
        elif scale_factor != 1.0:
            display_image = cv2.resize(
                frame_with_grid, (display_width, display_height), interpolation=cv2.INTER_CUBIC
            )
        else:
            display_image = frame_with_grid

        # Display original with grid
        if cv2.getWindowProperty(window_name_original, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.resizeWindow(window_name_original, display_width, display_height)
        cv2.imshow(window_name_original, display_image)

        # Display unwarped view
        unwarp_height, unwarp_width = unwarped.shape[:2]
        unwarp_scale = min(max_width / unwarp_width, max_height / unwarp_height) if unwarp_width > 0 and unwarp_height > 0 else 1.0
        if unwarp_scale <= 0:
            unwarp_scale = 1.0
        unwarp_display_width = int(round(unwarp_width * unwarp_scale))
        unwarp_display_height = int(round(unwarp_height * unwarp_scale))
        
        if unwarp_display_width > 0 and unwarp_display_height > 0 and unwarp_scale != 1.0:
            unwarped_display = cv2.resize(unwarped, (unwarp_display_width, unwarp_display_height), interpolation=cv2.INTER_CUBIC)
        else:
            unwarped_display = unwarped

        if cv2.getWindowProperty(window_name_unwarp, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.resizeWindow(window_name_unwarp, unwarp_display_width, unwarp_display_height)
        cv2.imshow(window_name_unwarp, unwarped_display)

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

    cv2.destroyWindow(window_name_original)
    cv2.destroyWindow(window_name_unwarp)
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

