import argparse
import cv2

from line_detector import LineDetector


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

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("Error: Could not read the first frame from the video")
        return

    detector = LineDetector(scale=scale)
    frame_with_grid, _ = detector.detect_lines(frame)

    frame_height, frame_width = frame_with_grid.shape[:2]
    if max_width <= 0:
        max_width = frame_width
    if max_height <= 0:
        max_height = frame_height

    scale_factor = min(max_width / frame_width, max_height / frame_height)
    if scale_factor <= 0:
        scale_factor = 1.0

    display_width = int(round(frame_width * scale_factor))
    display_height = int(round(frame_height * scale_factor))

    if display_width <= 0 or display_height <= 0:
        display_width, display_height = frame_width, frame_height
        display_image = frame_with_grid
    elif scale_factor != 1.0:
        display_image = cv2.resize(
            frame_with_grid, (display_width, display_height), interpolation=cv2.INTER_CUBIC
        )
    else:
        display_image = frame_with_grid

    window_name = "First Frame with Grid Lines"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)
    cv2.imshow(window_name, display_image)
    print("Press any key in the display window to close.")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


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

