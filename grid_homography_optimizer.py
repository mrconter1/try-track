import argparse
import cv2
import numpy as np
from scipy.optimize import minimize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize a planar grid homography for a single video frame with tile-size constraints."
    )
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to the video file.")
    parser.add_argument("--frame", type=int, default=4113, help="Frame number to analyze.")
    parser.add_argument(
        "--points",
        type=float,
        nargs=8,
        metavar=("x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"),
        default=[200, 200, 400, 200, 400, 400, 200, 400],
        help="Initial image-space points (top-left, top-right, bottom-right, bottom-left).",
    )
    parser.add_argument("--tile-radius", type=float, default=20.0, help="Grid extent in tiles from the origin.")
    parser.add_argument(
        "--samples-per-line",
        type=int,
        default=100,
        help="Number of sample points per grid line when computing the brightness score.",
    )
    parser.add_argument("--tile-min-pixels", type=float, default=100.0, help="Minimum allowable tile size in pixels.")
    parser.add_argument("--tile-max-pixels", type=float, default=400.0, help="Maximum allowable tile size in pixels.")
    parser.add_argument("--grid-color", type=str, default="0,255,0", help="Overlay color as B,G,R values.")
    parser.add_argument("--thickness", type=int, default=1, help="Grid line thickness.")
    parser.add_argument("--maxiter", type=int, default=200, help="Maximum optimizer iterations.")
    return parser.parse_args()


def load_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx >= total_frames:
        cap.release()
        raise ValueError(f"Frame {frame_idx} exceeds total frame count {total_frames}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx}")
    return frame


def points_to_homography(points):
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


def homography_to_params(H):
    return np.array(
        [H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2], H[2, 0], H[2, 1]],
        dtype=np.float64,
    )


def params_to_homography(params):
    return np.array(
        [
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [params[6], params[7], 1.0],
        ],
        dtype=np.float64,
    )


def measure_tile_size_in_pixels(H, frame_shape):
    h, w = frame_shape[:2]
    center = np.array([[[w / 2.0, h / 2.0]]], dtype=np.float64)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    plane_center = cv2.perspectiveTransform(center, H_inv).reshape(2)
    plane_tile_x = np.round(plane_center[0])
    plane_tile_y = np.round(plane_center[1])

    plane_corners = np.array(
        [
            [plane_tile_x, plane_tile_y],
            [plane_tile_x + 1.0, plane_tile_y],
            [plane_tile_x + 1.0, plane_tile_y + 1.0],
            [plane_tile_x, plane_tile_y + 1.0],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    image_corners = cv2.perspectiveTransform(plane_corners, H).reshape(4, 2)
    if np.any(~np.isfinite(image_corners)):
        return None

    edges = []
    for i in range(4):
        p1 = image_corners[i]
        p2 = image_corners[(i + 1) % 4]
        edges.append(np.linalg.norm(p1 - p2))
    return float(np.mean(edges))


def project_plane_points(plane_points, H):
    pts = np.array(plane_points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return projected


def sample_line_brightness(image_gray, line_points):
    h, w = image_gray.shape
    xs = np.clip(np.round(line_points[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(line_points[:, 1]).astype(int), 0, h - 1)

    inside_mask = (line_points[:, 0] >= 0) & (line_points[:, 0] < w) & (line_points[:, 1] >= 0) & (
        line_points[:, 1] < h
    )
    if not np.any(inside_mask):
        return 0.0

    xs_inside = xs[inside_mask]
    ys_inside = ys[inside_mask]
    return float(np.sum(image_gray[ys_inside, xs_inside]))


def grid_alignment_score(
    h_params,
    image_gray,
    tile_radius,
    num_samples_per_line,
    tile_min_pixels,
    tile_max_pixels,
    penalty_value,
):
    H = params_to_homography(h_params)
    tile_size = measure_tile_size_in_pixels(H, image_gray.shape)
    if tile_size is None or tile_size < tile_min_pixels or tile_size > tile_max_pixels:
        return penalty_value

    total_brightness = 0.0
    lin_samples = np.linspace(-tile_radius, tile_radius, num_samples_per_line, dtype=np.float32)

    for x in np.arange(-tile_radius, tile_radius + 1, 1.0, dtype=np.float32):
        plane_points = np.column_stack((np.full_like(lin_samples, x), lin_samples))
        image_points = project_plane_points(plane_points, H)
        total_brightness += sample_line_brightness(image_gray, image_points)

    for y in np.arange(-tile_radius, tile_radius + 1, 1.0, dtype=np.float32):
        plane_points = np.column_stack((lin_samples, np.full_like(lin_samples, y)))
        image_points = project_plane_points(plane_points, H)
        total_brightness += sample_line_brightness(image_gray, image_points)

    return total_brightness


def draw_grid(frame, H, tile_radius, color, thickness):
    color_bgr = tuple(int(max(0, min(255, c))) for c in color)

    def line_points(start, end, samples=200):
        return np.linspace(start, end, samples, dtype=np.float32)

    for x in range(int(-tile_radius), int(tile_radius) + 1):
        plane_line = line_points([x, -tile_radius], [x, tile_radius])
        img_pts = project_plane_points(plane_line, H).astype(int)
        cv2.polylines(frame, [img_pts.reshape(-1, 1, 2)], False, color_bgr, thickness, cv2.LINE_AA)

    for y in range(int(-tile_radius), int(tile_radius) + 1):
        plane_line = line_points([-tile_radius, y], [tile_radius, y])
        img_pts = project_plane_points(plane_line, H).astype(int)
        cv2.polylines(frame, [img_pts.reshape(-1, 1, 2)], False, color_bgr, thickness, cv2.LINE_AA)


def main():
    args = parse_args()
    frame = load_frame(args.video, args.frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    penalty_value = 1e12
    grid_color = tuple(int(max(0, min(255, float(c)))) for c in map(float, args.grid_color.split(",")))

    initial_H = points_to_homography(args.points)
    initial_params = homography_to_params(initial_H)

    objective = lambda hp: grid_alignment_score(
        hp,
        image_gray=gray,
        tile_radius=args.tile_radius,
        num_samples_per_line=args.samples_per_line,
        tile_min_pixels=args.tile_min_pixels,
        tile_max_pixels=args.tile_max_pixels,
        penalty_value=penalty_value,
    )

    print("Starting optimization...")
    result = minimize(
        objective,
        initial_params,
        method="Powell",
        options={"maxiter": args.maxiter, "disp": True},
    )
    optimal_params = result.x
    optimal_H = params_to_homography(optimal_params)

    print("Optimization finished.")
    print(f"Final score: {result.fun:.2f}")
    tile_size = measure_tile_size_in_pixels(optimal_H, gray.shape)
    if tile_size:
        print(f"Estimated tile size: {tile_size:.2f} px")

    output = frame.copy()
    draw_grid(output, optimal_H, args.tile_radius, color=grid_color, thickness=args.thickness)

    cv2.imshow("Optimized Grid", output)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

