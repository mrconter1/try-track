import argparse
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Project a planar grid through four reference points.")
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to the video file.")
    parser.add_argument("--frame", type=int, default=4113, help="Frame number to load.")
    parser.add_argument("--square-size", type=float, default=1000.0, help="Size of the reference square in plane units.")
    parser.add_argument("--grid-size", type=float, default=50.0, help="Spacing between grid lines.")
    parser.add_argument(
        "--points",
        type=float,
        nargs=8,
        metavar=("x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"),
        default=[200, 200, 800, 180, 820, 600, 220, 620],
        help="Four non-collinear image points (top-left, top-right, bottom-right, bottom-left).",
    )
    parser.add_argument("--color", type=str, default="0,255,0", help="Grid color as B,G,R (0-255).")
    parser.add_argument("--thickness", type=int, default=1, help="Line thickness for the grid.")
    return parser.parse_args()


def load_frame(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx >= total_frames:
        cap.release()
        raise ValueError(f"Frame {frame_idx} out of bounds. Video has {total_frames} frames.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx}")
    return frame


def parse_color(color_str: str):
    parts = color_str.split(",")
    if len(parts) != 3:
        raise ValueError("Color must be in B,G,R format")
    return tuple(int(max(0, min(255, float(v)))) for v in parts)


def build_homography(square_size: float, image_points):
    src_points = np.array(
        [
            [0.0, 0.0],
            [square_size, 0.0],
            [square_size, square_size],
            [0.0, square_size],
        ],
        dtype=np.float32,
    )
    dst_points = np.array(image_points, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(src_points, dst_points)


def draw_grid_with_homography(frame, H, square_size, grid_size, color, thickness):
    """Draw grid lines by sampling the square plane and projecting via homography."""
    max_coord = square_size
    step = max(1.0, grid_size)
    plane_points = []

    def project_points(points):
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pts, H)
        return projected.reshape(-1, 2)

    # Vertical lines (constant x)
    x_values = np.arange(0.0, max_coord + 0.1, step)
    for x in x_values:
        line = [[x, 0.0], [x, max_coord]]
        plane_points.extend(line)
        pts_img = project_points(line)
        p1, p2 = pts_img.astype(int)
        cv2.line(frame, tuple(p1), tuple(p2), color, thickness, cv2.LINE_AA)

    # Horizontal lines (constant y)
    y_values = np.arange(0.0, max_coord + 0.1, step)
    for y in y_values:
        line = [[0.0, y], [max_coord, y]]
        pts_img = project_points(line)
        p1, p2 = pts_img.astype(int)
        cv2.line(frame, tuple(p1), tuple(p2), color, thickness, cv2.LINE_AA)

    return frame


def main():
    args = parse_args()
    frame = load_frame(args.video, args.frame)
    color = parse_color(args.color)

    H = build_homography(args.square_size, args.points)
    frame_with_grid = frame.copy()
    draw_grid_with_homography(
        frame_with_grid,
        H,
        square_size=args.square_size,
        grid_size=args.grid_size,
        color=color,
        thickness=args.thickness,
    )

    cv2.imshow("Grid Homography Viewer", frame_with_grid)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

