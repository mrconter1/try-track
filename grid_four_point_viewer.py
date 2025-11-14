import argparse
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project an extended planar grid onto a frame using four corner points."
    )
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to the video file.")
    parser.add_argument("--frame", type=int, default=4113, help="Frame number to load.")
    parser.add_argument(
        "--points",
        type=float,
        nargs=8,
        metavar=("x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"),
        default=[220, 240, 820, 210, 840, 640, 240, 660],
        help="Image coordinates (top-left, top-right, bottom-right, bottom-left).",
    )
    parser.add_argument(
        "--tile-radius",
        type=float,
        default=20.0,
        help="How many tiles to extend outward from the reference tile (each direction).",
    )
    parser.add_argument("--grid-color", type=str, default="0,255,0", help="Grid line color as B,G,R.")
    parser.add_argument("--point-color", type=str, default="255,0,0", help="Marker color for the four points.")
    parser.add_argument("--thickness", type=int, default=1, help="Grid line thickness.")
    return parser.parse_args()


def parse_color(color_str: str):
    parts = color_str.split(",")
    if len(parts) != 3:
        raise ValueError("Color must be B,G,R with three components.")
    return tuple(int(max(0, min(255, float(p)))) for p in parts)


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


def build_homography(points):
    src = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dst = np.array(points, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(src, dst)


def draw_reference_points(frame, points, color):
    pts = np.array(points, dtype=np.int32).reshape(4, 2)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 6, color, -1, cv2.LINE_AA)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)


def draw_infinite_grid(frame, homography, tile_radius, color, thickness):
    step = 1.0
    extent = tile_radius + 1.0

    def project_line(pt_a, pt_b):
        pts = np.array([pt_a, pt_b], dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
        return projected.astype(int)

    # vertical lines (constant x)
    x_values = np.arange(-tile_radius, tile_radius + step, step)
    y_min = -extent
    y_max = extent
    for x in x_values:
        p1 = [x, y_min]
        p2 = [x, y_max]
        img_pts = project_line(p1, p2)
        cv2.line(frame, tuple(img_pts[0]), tuple(img_pts[1]), color, thickness, cv2.LINE_AA)

    # horizontal lines (constant y)
    y_values = np.arange(-tile_radius, tile_radius + step, step)
    x_min = -extent
    x_max = extent
    for y in y_values:
        p1 = [x_min, y]
        p2 = [x_max, y]
        img_pts = project_line(p1, p2)
        cv2.line(frame, tuple(img_pts[0]), tuple(img_pts[1]), color, thickness, cv2.LINE_AA)


def main():
    args = parse_args()
    frame = load_frame(args.video, args.frame)

    grid_color = parse_color(args.grid_color)
    point_color = parse_color(args.point_color)

    H = build_homography(args.points)
    output = frame.copy()

    draw_infinite_grid(
        output,
        homography=H,
        tile_radius=args.tile_radius,
        color=grid_color,
        thickness=args.thickness,
    )
    draw_reference_points(output, args.points, point_color)

    cv2.imshow("Four-Point Grid Projection", output)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

