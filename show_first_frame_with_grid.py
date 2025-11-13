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
    """Find all intersections between horizontal and vertical lines."""
    if len(lines) < 2:
        return []

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

    # Find all intersections between each horizontal and vertical line
    for rho_h, theta_h in horizontal:
        for rho_v, theta_v in vertical:
            intersection = line_intersection(rho_h, theta_h, rho_v, theta_v)
            if intersection is not None:
                x, y = intersection
                # Only keep intersections within frame bounds
                if 0 <= x < w and 0 <= y < h:
                    crossings.append((x, y))

    return crossings


def show_first_frame_with_grid(
    video_path: str,
    scale: float = 0.5,
    max_width: int = 1600,
    max_height: int = 900,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return

    detector = LineDetector(scale=scale)
    window_name = "Frame with Grid Lines"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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
        crossings = find_line_crossings(lines, frame.shape[:2])
        for x, y in crossings:
            cv2.circle(frame_with_grid, (x, y), 8, (0, 255, 0), 2)

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

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.resizeWindow(window_name, display_width, display_height)
        cv2.imshow(window_name, display_image)

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    show_first_frame_with_grid(
        video_path=args.video,
        scale=args.scale,
        max_width=args.max_width,
        max_height=args.max_height,
    )

