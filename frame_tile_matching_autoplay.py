import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

from line_detector import LineDetector
from auto_tile_detector import (
    extract_grid_squares,
    extract_and_warp_square,
    generate_tile_hash,
)
from frame_tile_matching_viz import rotate_coordinates, analyze_frame_pair


# NOTE: we use Optional for Python 3.10 compatibility instead of PEP 604 union syntax
from typing import Optional


def frame_tile_matching_autoplay(
    video_path: str,
    end_frame: Optional[int] = None,
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

    print(f"[Autoplay] Video: {video_path}")
    print(f"[Autoplay] Frames: 0 → {end_frame - 1} (total {end_frame})\n")

    line_detector = LineDetector()
    frames: dict[int, dict] = {}

    def extract_frame_data(frame_num: int):
        if frame_num in frames:
            return frames[frame_num]

        if frame_num < 0 or frame_num >= total_frames:
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            return None

        # Detect grid (suppress verbose output from detector)
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        _, lines = line_detector.detect_lines(frame)
        sys.stdout = old_stdout

        grid_map = extract_grid_squares(lines, frame.shape[:2])
        if not grid_map:
            return None

        grid_rows = [coord[0] for coord in grid_map.keys()]
        grid_cols = [coord[1] for coord in grid_map.keys()]
        num_rows = (max(grid_rows) + 1) if grid_rows else 0
        num_cols = (max(grid_cols) + 1) if grid_cols else 0

        tile_list = []
        signatures_by_coord = {}
        centers_by_coord = {}

        for (row, col), square_polygon in grid_map.items():
            try:
                warped = extract_and_warp_square(frame, square_polygon)

                gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                if float(np.std(gray_warped)) > 20:
                    continue

                p1, _, p3, _ = square_polygon
                cx = int((p1[0] + p3[0]) / 2)
                cy = int((p1[1] + p3[1]) / 2)
                centers_by_coord[(row, col)] = (cx, cy)

                signatures = {}
                for rotation in [0, 90, 180, 270]:
                    if rotation == 0:
                        rotated = warped
                    elif rotation == 90:
                        rotated = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif rotation == 180:
                        rotated = cv2.rotate(warped, cv2.ROTATE_180)
                    else:
                        rotated = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

                    hash_sig = generate_tile_hash(rotated, blocks_per_side=8, use_clahe=True)
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
            "frame_number": frame_num,
            "tiles": tile_list,
            "frame": frame,
            "grid_map": grid_map,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "signatures": signatures_by_coord,
            "centers": centers_by_coord,
        }

        frames[frame_num] = frame_data
        return frame_data

    def build_global_map_tiles(target_frame_num: int):
        global_positions: dict[tuple[int, int, int], tuple[int, int]] = {}
        global_tiles: list[dict] = []

        for idx in range(target_frame_num + 1):
            frame_data = extract_frame_data(idx)
            if not frame_data:
                continue

            if idx == 0 or not global_positions:
                for tile_obj in frame_data["tiles"]:
                    key = (idx, tile_obj["frame_row"], tile_obj["frame_col"])
                    pos = (tile_obj["frame_row"], tile_obj["frame_col"])
                    global_positions[key] = pos
                    global_tiles.append(
                        {
                            "frame_index": idx,
                            "frame_row": tile_obj["frame_row"],
                            "frame_col": tile_obj["frame_col"],
                            "global_row": pos[0],
                            "global_col": pos[1],
                            "image": tile_obj["image"],
                        }
                    )
                continue

            prev_data = extract_frame_data(idx - 1)
            if not prev_data:
                continue

            analysis = analyze_frame_pair(prev_data, frame_data)
            best_match = analysis["best_match"]
            if not best_match:
                continue

            rotations = best_match["rotations"]
            rotation_offset = ((rotations[1] - rotations[0]) % 360) if rotations else 0
            prev_row, prev_col = best_match["prev_coord"]
            prev_key = (idx - 1, prev_row, prev_col)
            if prev_key not in global_positions:
                continue

            prev_global_pos = global_positions[prev_key]
            curr_row, curr_col = best_match["curr_coord"]
            rotated_curr_row, rotated_curr_col = rotate_coordinates(
                curr_row,
                curr_col,
                frame_data["num_rows"],
                frame_data["num_cols"],
                rotation_offset,
            )
            offset_row = prev_global_pos[0] - rotated_curr_row
            offset_col = prev_global_pos[1] - rotated_curr_col

            for tile_obj in frame_data["tiles"]:
                local_row = tile_obj["frame_row"]
                local_col = tile_obj["frame_col"]
                rotated_row, rotated_col = rotate_coordinates(
                    local_row,
                    local_col,
                    frame_data["num_rows"],
                    frame_data["num_cols"],
                    rotation_offset,
                )

                global_row = rotated_row + offset_row
                global_col = rotated_col + offset_col
                key = (idx, local_row, local_col)
                if key in global_positions:
                    continue

                global_positions[key] = (global_row, global_col)
                global_tiles.append(
                    {
                        "frame_index": idx,
                        "frame_row": local_row,
                        "frame_col": local_col,
                        "global_row": global_row,
                        "global_col": global_col,
                        "image": tile_obj["image"],
                    }
                )

        return global_tiles

    def render_frame(ax_main, ax_grid, frame_num: int):
        curr_data = extract_frame_data(frame_num)
        if not curr_data:
            print(f"[Autoplay] Frame {frame_num}: no data, skipping")
            return False

        ax_main.cla()
        ax_grid.cla()

        if frame_num > 0 and (frame_num - 1) in frames:
            prev_data = frames[frame_num - 1]
            h_prev, w_prev = prev_data["frame"].shape[:2]
            h_curr, w_curr = curr_data["frame"].shape[:2]

            if (h_prev, w_prev) != (h_curr, w_curr):
                curr_resized = cv2.resize(curr_data["frame"], (w_prev, h_prev))
            else:
                curr_resized = curr_data["frame"]

            combined = cv2.addWeighted(prev_data["frame"], 0.5, curr_resized, 0.5, 0)
            analysis = analyze_frame_pair(prev_data, curr_data)
            scaled_curr_centers = analysis["scaled_curr_centers"]
            best_match = analysis["best_match"]

            for (_, _), (prev_cx, prev_cy) in prev_data["centers"].items():
                cv2.circle(combined, (prev_cx, prev_cy), 5, (0, 255, 255), -1)

            for (_, _), (curr_cx, curr_cy) in scaled_curr_centers.items():
                cv2.circle(combined, (curr_cx, curr_cy), 5, (0, 255, 0), -1)

            if best_match:
                prev_cx, prev_cy = best_match["prev_center"]
                curr_cx, curr_cy = best_match["curr_center"]
                cv2.line(combined, (prev_cx, prev_cy), (curr_cx, curr_cy), (255, 0, 0), 2)

            from PIL import Image, ImageDraw, ImageFont

            pil_combined = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_combined)
            try:
                font_large = ImageFont.truetype("arial.ttf", 36)
                font_small = ImageFont.truetype("arial.ttf", 24)
            except OSError:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()

            for (prev_row, prev_col), (prev_cx, prev_cy) in prev_data["centers"].items():
                draw.text(
                    (prev_cx - 40, prev_cy - 45),
                    f"({prev_row},{prev_col})",
                    fill=(255, 255, 0),
                    font=font_small,
                )

            for (curr_row, curr_col), (curr_cx, curr_cy) in scaled_curr_centers.items():
                draw.text(
                    (curr_cx - 40, curr_cy + 10),
                    f"({curr_row},{curr_col})",
                    fill=(0, 255, 0),
                    font=font_small,
                )

            if best_match and best_match["rotations"]:
                curr_rot, prev_rot = best_match["rotations"]
                prev_cx, prev_cy = best_match["prev_center"]
                curr_cx, curr_cy = best_match["curr_center"]
                mid_x = (prev_cx + curr_cx) // 2
                mid_y = (prev_cy + curr_cy) // 2
                draw.text(
                    (mid_x - 20, mid_y - 5),
                    f"{curr_rot}°→{prev_rot}°",
                    fill=(255, 255, 0),
                    font=font_small,
                )

            combined = cv2.cvtColor(np.array(pil_combined), cv2.COLOR_RGB2BGR)
            ax_main.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
            ax_main.set_title(f"Frame {frame_num - 1} → Frame {frame_num}")
        else:
            ax_main.imshow(cv2.cvtColor(curr_data["frame"], cv2.COLOR_BGR2RGB))
            ax_main.set_title(f"Frame {frame_num} (anchor)")

        global_tiles = build_global_map_tiles(frame_num)
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

            from PIL import Image, ImageDraw, ImageFont

            try:
                tile_font = ImageFont.truetype("arial.ttf", 10)
            except OSError:
                tile_font = ImageFont.load_default()

            for tile in global_tiles:
                grid_row = (tile["global_row"] - min_row) + padding
                grid_col = (tile["global_col"] - min_col) + padding
                y_start = grid_row * tile_display_size
                x_start = grid_col * tile_display_size

                warped_resized = cv2.resize(tile["image"], (tile_display_size, tile_display_size))
                pil_tile = Image.fromarray(cv2.cvtColor(warped_resized, cv2.COLOR_BGR2RGB))
                draw_tile = ImageDraw.Draw(pil_tile)
                draw_tile.text(
                    (2, 2),
                    f"G({tile['global_row']},{tile['global_col']})",
                    fill=(255, 0, 0),
                    font=tile_font,
                )
                warped_resized = cv2.cvtColor(np.array(pil_tile), cv2.COLOR_RGB2BGR)

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

            ax_grid.imshow(cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB))
            ax_grid.set_title(f"Global map ≤ frame {frame_num} ({len(global_tiles)} tiles)")
        else:
            ax_grid.text(0.5, 0.5, "No tiles", ha="center", va="center", transform=ax_grid.transAxes)
            ax_grid.set_title("Global map")

        ax_main.axis("off")
        ax_grid.axis("off")
        return True

    plt.ion()
    fig = plt.figure(figsize=(20, 7))
    ax_main = fig.add_subplot(1, 2, 1)
    ax_grid = fig.add_subplot(1, 2, 2)

    for frame_num in range(end_frame):
        if not plt.fignum_exists(fig.number):
            print("[Autoplay] Figure closed by user, stopping")
            break

        success = render_frame(ax_main, ax_grid, frame_num)
        if not success:
            continue

        plt.pause(0.015)

    plt.ioff()
    plt.show(block=True)
    cap.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-play tile matching visualization")
    parser.add_argument("--video", type=str, default="video.mp4", help="Video file path")
    parser.add_argument(
        "--end-frame", type=int, default=None, help="End frame index (exclusive, default=video end)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frame_tile_matching_autoplay(
        video_path=args.video,
        end_frame=args.end_frame,
    )

