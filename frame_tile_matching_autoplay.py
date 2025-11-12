import argparse
import cProfile
import io
import auto_tile_detector
import cv2
import numpy as np
import pstats

from line_detector import LineDetector
from auto_tile_detector import (
    extract_grid_squares,
    extract_and_warp_square,
    generate_tile_hash,
    euclidean_distance,
)
from frame_tile_matching_viz import rotate_coordinates, analyze_frame_pair


# NOTE: we use Optional for Python 3.10 compatibility instead of PEP 604 union syntax
from typing import Optional


def compute_dhash_byte_fast(block: np.ndarray) -> int:
    """Compute dHash byte using vectorized operations for performance."""
    h, w = block.shape
    h_mid = h // 2
    w_mid = w // 2

    if h_mid == 0 or w_mid == 0:
        block_mean = float(block.mean(dtype=np.float32))
        return int(block_mean > 128) << 7

    h_even = h_mid * 2
    w_even = w_mid * 2
    trimmed = block[:h_even, :w_even]

    reshaped = trimmed.reshape(2, h_mid, 2, w_mid)
    quadrant_sums = reshaped.sum(axis=(1, 3), dtype=np.int64)
    quadrant_area = h_mid * w_mid

    q1 = float(quadrant_sums[0, 0]) / quadrant_area
    q2 = float(quadrant_sums[0, 1]) / quadrant_area
    q3 = float(quadrant_sums[1, 0]) / quadrant_area
    q4 = float(quadrant_sums[1, 1]) / quadrant_area

    total_sum = float(quadrant_sums.sum())
    block_mean = total_sum / (quadrant_area * 4)

    byte_val = (
        ((q1 > q2) << 0)
        | ((q3 > q4) << 1)
        | ((q1 > q3) << 2)
        | ((q2 > q4) << 3)
        | ((q1 > q4) << 4)
        | ((q2 > q3) << 5)
        | (((q1 + q4) > (q2 + q3)) << 6)
        | ((block_mean > 128) << 7)
    )
    return int(byte_val)


auto_tile_detector.compute_dhash_byte = compute_dhash_byte_fast


def frame_tile_matching_autoplay(
    video_path: str,
    end_frame: Optional[int] = None,
    max_frames: Optional[int] = None,
):
    """Iterate through frames automatically and visualize global tile stitching."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames
    end_frame = min(end_frame, total_frames)

    if max_frames is not None:
        max_frames = max(0, max_frames)
        frame_limit = min(end_frame, max_frames)
    else:
        frame_limit = end_frame

    print(f"[Autoplay] Video: {video_path}")
    if frame_limit > 0:
        print(f"[Autoplay] Frames: 0 → {frame_limit - 1} (total {frame_limit})\n")
    else:
        if max_frames == 0:
            print("[Autoplay] No frames scheduled (max_frames=0)\n")
        else:
            print("[Autoplay] No frames scheduled\n")

    line_detector = LineDetector()
    frames: dict[int, dict] = {}
    global_positions: dict[tuple[int, int], dict] = {}
    global_tiles: list[dict] = []
    integrated_frames: set[int] = set()
    sequential_frame_index = 0
    capture_positioned = False

    def read_next_frame():
        nonlocal sequential_frame_index, capture_positioned
        if not capture_positioned:
            cap.set(cv2.CAP_PROP_POS_FRAMES, sequential_frame_index)
            capture_positioned = True
        ret, frame = cap.read()
        if not ret:
            return None, None
        current_index = sequential_frame_index
        sequential_frame_index += 1
        return current_index, frame

    def extract_frame_data(frame_num: int):
        nonlocal sequential_frame_index
        if frame_num in frames:
            return frames[frame_num]

        if frame_num < 0 or frame_num >= total_frames:
            return None

        while sequential_frame_index <= frame_num:
            idx, next_frame = read_next_frame()
            if next_frame is None:
                break
            store_frame_data(idx, next_frame)

        return frames.get(frame_num)

    def store_frame_data(frame_index: int, frame: np.ndarray | None):
        if frame is None:
            return

        # Detect grid (suppress verbose output from detector)
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        _, lines = line_detector.detect_lines(frame)
        sys.stdout = old_stdout

        grid_map = extract_grid_squares(lines, frame.shape[:2])
        if not grid_map:
            frames[frame_index] = {
                "frame_number": frame_index,
                "tiles": [],
                "frame": frame,
                "grid_map": {},
                "num_rows": 0,
                "num_cols": 0,
                "signatures": {},
                "centers": {},
            }
            return

        grid_rows = [coord[0] for coord in grid_map.keys()]
        grid_cols = [coord[1] for coord in grid_map.keys()]
        num_rows = (max(grid_rows) + 1) if grid_rows else 0
        num_cols = (max(grid_cols) + 1) if grid_cols else 0

        tile_list = []
        signatures_by_coord = {}
        centers_by_coord = {}

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        for (row, col), square_polygon in grid_map.items():
            try:
                warped = extract_and_warp_square(frame, square_polygon)

                gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                if float(np.std(gray_warped)) > 20:
                    continue

                enhanced_gray = clahe.apply(gray_warped)

                p1, _, p3, _ = square_polygon
                cx = int((p1[0] + p3[0]) / 2)
                cy = int((p1[1] + p3[1]) / 2)
                centers_by_coord[(row, col)] = (cx, cy)

                signatures = {}
                for rotation in [0, 90, 180, 270]:
                    if rotation == 0:
                        rotated_gray = enhanced_gray
                    elif rotation == 90:
                        rotated_gray = cv2.rotate(enhanced_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif rotation == 180:
                        rotated_gray = cv2.rotate(enhanced_gray, cv2.ROTATE_180)
                    else:
                        rotated_gray = cv2.rotate(enhanced_gray, cv2.ROTATE_90_CLOCKWISE)

                    rotated_bgr = cv2.cvtColor(rotated_gray, cv2.COLOR_GRAY2BGR)
                    hash_sig = generate_tile_hash(rotated_bgr, blocks_per_side=8, use_clahe=False)
                    if hash_sig:
                        signatures[rotation] = hash_sig

                if signatures:
                    tile_obj = {
                        "signatures": signatures,
                        "image": warped,
                        "frame_row": row,
                        "frame_col": col,
                    }
                    tile_list.append(tile_obj)
                    signatures_by_coord[(row, col)] = signatures

            except Exception:
                continue

        frame_data = {
            "frame_number": frame_index,
            "tiles": tile_list,
            "frame": frame,
            "grid_map": grid_map,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "signatures": signatures_by_coord,
            "centers": centers_by_coord,
        }

        frames[frame_index] = frame_data
        return frame_data

    def add_global_tile(frame_index: int, tile_obj: dict, global_row: int, global_col: int) -> bool:
        coord_key = (global_row, global_col)
        if coord_key in global_positions:
            return False

        entry = {
            "frame_index": frame_index,
            "frame_row": tile_obj["frame_row"],
            "frame_col": tile_obj["frame_col"],
            "global_row": global_row,
            "global_col": global_col,
            "image": tile_obj["image"],
            "signatures": tile_obj["signatures"],
        }
        global_positions[coord_key] = entry
        global_tiles.append(entry)
        return True

    def integrate_frame(frame_num: int):
        if frame_num in integrated_frames:
            return

        frame_data = extract_frame_data(frame_num)
        if not frame_data or not frame_data["tiles"]:
            integrated_frames.add(frame_num)
            return

        if not global_tiles:
            for tile_obj in frame_data["tiles"]:
                add_global_tile(
                    frame_index=frame_num,
                    tile_obj=tile_obj,
                    global_row=tile_obj["frame_row"],
                    global_col=tile_obj["frame_col"],
                )
            integrated_frames.add(frame_num)
            return

        best_anchor = None
        best_distance = float("inf")

        for tile_obj in frame_data["tiles"]:
            curr_coord = (tile_obj["frame_row"], tile_obj["frame_col"])
            curr_sigs = tile_obj["signatures"]
            if not curr_sigs:
                continue

            for global_tile in global_tiles:
                global_sigs = global_tile["signatures"]
                for curr_rot, curr_sig in curr_sigs.items():
                    for global_rot, global_sig in global_sigs.items():
                        dist = euclidean_distance(curr_sig, global_sig)
                        if dist < best_distance:
                            best_distance = dist
                            best_anchor = {
                                "curr_tile": tile_obj,
                                "curr_coord": curr_coord,
                                "curr_rot": curr_rot,
                                "global_tile": global_tile,
                                "global_rot": global_rot,
                                "rotation_offset": (global_rot - curr_rot) % 360,
                                "distance": dist,
                            }

        if not best_anchor:
            integrated_frames.add(frame_num)
            return

        rotation_offset = best_anchor["rotation_offset"]
        anchor_global_row = best_anchor["global_tile"]["global_row"]
        anchor_global_col = best_anchor["global_tile"]["global_col"]
        num_rows = frame_data["num_rows"] or 1
        num_cols = frame_data["num_cols"] or 1

        rotated_anchor_row, rotated_anchor_col = rotate_coordinates(
            best_anchor["curr_tile"]["frame_row"],
            best_anchor["curr_tile"]["frame_col"],
            num_rows,
            num_cols,
            rotation_offset,
        )

        offset_row = anchor_global_row - rotated_anchor_row
        offset_col = anchor_global_col - rotated_anchor_col

        added_tiles = 0

        for tile_obj in frame_data["tiles"]:
            rotated_row, rotated_col = rotate_coordinates(
                tile_obj["frame_row"],
                tile_obj["frame_col"],
                num_rows,
                num_cols,
                rotation_offset,
            )
            global_row = rotated_row + offset_row
            global_col = rotated_col + offset_col
            if add_global_tile(frame_num, tile_obj, global_row, global_col):
                added_tiles += 1

        print(
            f"[Autoplay] Integrated frame {frame_num}: anchor dist={best_anchor['distance']:.0f}, "
            f"rotation={rotation_offset}°, added={added_tiles} tiles"
        )

        integrated_frames.add(frame_num)

    def pad_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
        if image.shape[0] == target_height:
            return image
        diff = target_height - image.shape[0]
        top = diff // 2
        bottom = diff - top
        return cv2.copyMakeBorder(
            image,
            top,
            bottom,
            0,
            0,
            borderType=cv2.BORDER_CONSTANT,
            value=(30, 30, 30),
        )

    def pad_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
        if image.shape[1] == target_width:
            return image
        diff = target_width - image.shape[1]
        left = diff // 2
        right = diff - left
        return cv2.copyMakeBorder(
            image,
            0,
            0,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=(30, 30, 30),
        )

    def render_frame(frame_num: int):
        curr_data = extract_frame_data(frame_num)
        if not curr_data:
            print(f"[Autoplay] Frame {frame_num}: no data, skipping")
            return None

        integrate_frame(frame_num)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_small_scale = 0.6
        font_small_thickness = 1

        if frame_num > 0 and (frame_num - 1) in frames:
            prev_data = frames[frame_num - 1]
            h_prev, w_prev = prev_data["frame"].shape[:2]
            h_curr, w_curr = curr_data["frame"].shape[:2]

            if (h_prev, w_prev) != (h_curr, w_curr):
                curr_resized = cv2.resize(curr_data["frame"], (w_prev, h_prev))
            else:
                curr_resized = curr_data["frame"]

            left_display = cv2.addWeighted(prev_data["frame"], 0.5, curr_resized, 0.5, 0)
            analysis = analyze_frame_pair(prev_data, curr_data)
            scaled_curr_centers = analysis["scaled_curr_centers"]
            best_match = analysis["best_match"]

            for (prev_row, prev_col), (prev_cx, prev_cy) in prev_data["centers"].items():
                cv2.circle(left_display, (prev_cx, prev_cy), 5, (0, 255, 255), -1)
                cv2.putText(
                    left_display,
                    f"({prev_row},{prev_col})",
                    (prev_cx - 40, prev_cy - 10),
                    font,
                    font_small_scale,
                    (0, 255, 255),
                    font_small_thickness + 1,
                    cv2.LINE_AA,
                )

            for (curr_row, curr_col), (curr_cx, curr_cy) in scaled_curr_centers.items():
                cv2.circle(left_display, (curr_cx, curr_cy), 5, (0, 255, 0), -1)
                cv2.putText(
                    left_display,
                    f"({curr_row},{curr_col})",
                    (curr_cx - 40, curr_cy + 20),
                    font,
                    font_small_scale,
                    (0, 255, 0),
                    font_small_thickness + 1,
                    cv2.LINE_AA,
                )

            if best_match:
                prev_cx, prev_cy = best_match["prev_center"]
                curr_cx, curr_cy = best_match["curr_center"]
                cv2.line(left_display, (prev_cx, prev_cy), (curr_cx, curr_cy), (255, 0, 0), 2)

                if best_match["rotations"]:
                    curr_rot, prev_rot = best_match["rotations"]
                    mid_x = (prev_cx + curr_cx) // 2
                    mid_y = (prev_cy + curr_cy) // 2
                    cv2.putText(
                        left_display,
                        f"{curr_rot}°→{prev_rot}°",
                        (mid_x - 30, mid_y - 5),
                        font,
                        font_small_scale,
                        (0, 255, 255),
                        font_small_thickness + 1,
                        cv2.LINE_AA,
                    )

                header_text = (
                    f"Frame {frame_num - 1}→{frame_num} | "
                    f"match ({best_match['curr_coord'][0]},{best_match['curr_coord'][1]})→"
                    f"({best_match['prev_coord'][0]},{best_match['prev_coord'][1]}) "
                    f"dist={best_match['distance']:.0f}"
                )
            else:
                header_text = f"Frame {frame_num - 1}→{frame_num} | no valid match"

            cv2.putText(
                left_display,
                header_text,
                (20, 40),
                font,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            left_display = curr_data["frame"].copy()
            cv2.putText(
                left_display,
                f"Frame {frame_num} (anchor)",
                (20, 40),
                font,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if global_tiles:
            tile_display_size = 30
            padding = 2

            rows = [tile["global_row"] for tile in global_tiles]
            cols = [tile["global_col"] for tile in global_tiles]
            min_row, max_row = min(rows), max(rows)
            min_col, max_col = min(cols), max(cols)

            grid_rows = (max_row - min_row + 1) + padding * 2
            grid_cols = (max_col - min_col + 1) + padding * 2
            composite_height = max(grid_rows, 1) * tile_display_size
            composite_width = max(grid_cols, 1) * tile_display_size
            composite_image = np.full((composite_height, composite_width, 3), 40, dtype=np.uint8)

            origin_in_bounds = (min_row <= 0 <= max_row) and (min_col <= 0 <= max_col)
            if origin_in_bounds:
                origin_y = (0 - min_row + padding) * tile_display_size
                origin_x = (0 - min_col + padding) * tile_display_size
                cv2.line(composite_image, (origin_x, origin_y - 10), (origin_x, origin_y + 10), (255, 0, 0), 2)
                cv2.line(composite_image, (origin_x - 10, origin_y), (origin_x + 10, origin_y), (255, 0, 0), 2)

            for tile in global_tiles:
                grid_row = (tile["global_row"] - min_row) + padding
                grid_col = (tile["global_col"] - min_col) + padding
                y_start = grid_row * tile_display_size
                x_start = grid_col * tile_display_size

                warped_resized = cv2.resize(tile["image"], (tile_display_size, tile_display_size))
                cv2.putText(
                    warped_resized,
                    f"G({tile['global_row']},{tile['global_col']})",
                    (2, 12),
                    font,
                    0.3,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

                composite_image[
                    y_start : y_start + tile_display_size, x_start : x_start + tile_display_size
                ] = warped_resized

            for i in range(0, grid_rows + 1):
                y = min(i * tile_display_size, composite_height - 1)
                thickness = 2 if i % 5 == 0 else 1
                cv2.line(composite_image, (0, y), (composite_width, y), (80, 80, 80), thickness)
            for j in range(0, grid_cols + 1):
                x = min(j * tile_display_size, composite_width - 1)
                thickness = 2 if j % 5 == 0 else 1
                cv2.line(composite_image, (x, 0), (x, composite_height), (80, 80, 80), thickness)

            cv2.putText(
                composite_image,
                f"Global map ≤ frame {frame_num} ({len(global_tiles)} tiles)",
                (20, 30),
                font,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            right_display = composite_image
        else:
            right_height = left_display.shape[0]
            right_width = max(left_display.shape[1] // 2, 400)
            right_display = np.full((right_height, right_width, 3), 40, dtype=np.uint8)
            cv2.putText(
                right_display,
                "Global map empty",
                (20, right_height // 2),
                font,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        target_height = max(left_display.shape[0], right_display.shape[0])
        left_display = pad_to_height(left_display, target_height)
        right_display = pad_to_height(right_display, target_height)

        target_width = max(left_display.shape[1], right_display.shape[1])
        left_display = pad_to_width(left_display, target_width)
        right_display = pad_to_width(right_display, target_width)

        combined_display = np.hstack([left_display, right_display])
        return combined_display

    window_name = "Tile Matching Autoplay"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    try:
        import ctypes

        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)
    except Exception:
        screen_width = 1920
        screen_height = 1080

    max_display_width = int(screen_width * 0.9)
    max_display_height = int(screen_height * 0.9)

    window_created = False
    try:
        for frame_num in range(frame_limit):
            display = render_frame(frame_num)
            if display is None:
                continue

            disp_h, disp_w = display.shape[:2]
            scale = min(max_display_width / disp_w, max_display_height / disp_h)
            if scale <= 0:
                scale = 1.0

            if scale != 1.0:
                resized_display = cv2.resize(
                    display,
                    (int(disp_w * scale), int(disp_h * scale)),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                resized_display = display

            cv2.resizeWindow(window_name, resized_display.shape[1], resized_display.shape[0])
            cv2.imshow(window_name, resized_display)
            window_created = True

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("[Autoplay] Quit requested, stopping")
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[Autoplay] Window closed by user, stopping")
                break
    finally:
        if window_created:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
        cap.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-play tile matching visualization")
    parser.add_argument("--video", type=str, default="video.mp4", help="Video file path")
    parser.add_argument(
        "--end-frame", type=int, default=None, help="End frame index (exclusive, default=video end)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process from the start",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        frame_tile_matching_autoplay(
            video_path=args.video,
            end_frame=args.end_frame,
            max_frames=args.max_frames,
        )
    finally:
        profiler.disable()
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.strip_dirs().sort_stats("cumulative").print_stats(20)
        print(stats_stream.getvalue())

