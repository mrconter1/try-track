import argparse
import cv2
import numpy as np
from typing import Optional, Dict, Tuple

from line_detector import LineDetector
from auto_tile_detector import extract_grid_squares, extract_and_warp_square

TILE_DISPLAY_SIZE = 80


def pad_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image
    diff = target_height - image.shape[0]
    top = diff // 2
    bottom = diff - top
    return cv2.copyMakeBorder(image, top, bottom, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(30, 30, 30))


def pad_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    if image.shape[1] == target_width:
        return image
    diff = target_width - image.shape[1]
    left = diff // 2
    right = diff - left
    return cv2.copyMakeBorder(image, 0, 0, left, right, borderType=cv2.BORDER_CONSTANT, value=(30, 30, 30))


def build_tile_mosaic(frame: np.ndarray, grid_map: Dict[Tuple[int, int], Tuple]) -> Optional[np.ndarray]:
    if not grid_map:
        return None

    grid_rows = [coord[0] for coord in grid_map.keys()]
    grid_cols = [coord[1] for coord in grid_map.keys()]
    num_rows = (max(grid_rows) + 1) if grid_rows else 0
    num_cols = (max(grid_cols) + 1) if grid_cols else 0
    if num_rows == 0 or num_cols == 0:
        return None

    mosaic = np.full((num_rows * TILE_DISPLAY_SIZE, num_cols * TILE_DISPLAY_SIZE, 3), 40, dtype=np.uint8)

    for (row, col), square in grid_map.items():
        try:
            warped = extract_and_warp_square(frame, square)
            if warped is None or warped.size == 0:
                continue
            resized = cv2.resize(warped, (TILE_DISPLAY_SIZE, TILE_DISPLAY_SIZE))
            y_start = row * TILE_DISPLAY_SIZE
            x_start = col * TILE_DISPLAY_SIZE
            mosaic[y_start : y_start + TILE_DISPLAY_SIZE, x_start : x_start + TILE_DISPLAY_SIZE] = resized
        except Exception:
            continue

    return mosaic


class FrameCache:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")
        self.cache: Dict[int, Dict] = {}
        self.line_detector = LineDetector()
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_total_frames(self) -> int:
        return self.total_frames

    def get_frame_data(self, index: int) -> Optional[Dict]:
        if index < 0 or index >= self.total_frames:
            return None

        if index in self.cache:
            return self.cache[index]

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if not ret:
            return None

        try:
            _, lines = self.line_detector.detect_lines(frame)
        except Exception:
            lines = []

        grid_map = extract_grid_squares(lines, frame.shape[:2]) if lines is not None else {}
        mosaic = build_tile_mosaic(frame, grid_map)

        data = {
            "frame": frame,
            "grid_map": grid_map,
            "mosaic": mosaic,
        }
        self.cache[index] = data
        return data

    def release(self) -> None:
        self.cap.release()


def extract_crossings(grid_map: Dict[Tuple[int, int], Tuple], frame_shape: Tuple[int, int]) -> list:
    """Extract crossing points (tile corners) from grid_map in frame coordinates."""
    if not grid_map:
        return []
    
    crossings = set()
    for (row, col), square in grid_map.items():
        if square is None or len(square) < 4:
            continue
        pts = square
        # Each square has 4 corner points
        for pt in pts:
            x, y = int(pt[0]), int(pt[1])
            crossings.add((x, y))
    
    return sorted(list(crossings))


def compose_display(
    first: Dict, second: Dict, frame_idx: int, next_idx: int, x_offset: int = 0, y_offset: int = 0
) -> Tuple[np.ndarray, float, float]:
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    mosaics = []

    # --- NEW LOGIC: Create a common canvas for both frames ---
    all_coords = set()
    if first.get("grid_map"):
        all_coords.update(first["grid_map"].keys())
    if second.get("grid_map"):
        all_coords.update(second["grid_map"].keys())

    if not all_coords:
        # If no grid is found in either frame, display "No grid" message for all views
        for i in range(3):
            blank = np.full((200, 200, 3), 60, dtype=np.uint8)
            text = f"Frame {frame_idx}" if i == 0 else f"Frame {next_idx}" if i == 1 else f"Overlay {frame_idx}+{next_idx}"
            color = (0, 255, 255) if i == 0 else (0, 255, 0) if i == 1 else (255, 255, 255)
            cv2.putText(blank, text, (20, 40), label_font, 0.7, color, 2, cv2.LINE_AA)
            cv2.putText(blank, "No grid", (30, 110), label_font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            mosaics.append(blank)
    else:
        min_row = min(c[0] for c in all_coords)
        max_row = max(c[0] for c in all_coords)
        min_col = min(c[1] for c in all_coords)
        max_col = max(c[1] for c in all_coords)

        num_rows = max_row - min_row + 1
        num_cols = max_col - min_col + 1
        canvas_h = num_rows * TILE_DISPLAY_SIZE
        canvas_w = num_cols * TILE_DISPLAY_SIZE

        # --- Re-create mosaics on the shared canvas ---
        mosaic_a = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)
        mosaic_b = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)

        # Populate mosaic for the first frame
        if first.get("grid_map"):
            for (row, col), square in first["grid_map"].items():
                try:
                    warped = extract_and_warp_square(first["frame"], square)
                    if warped is not None and warped.size > 0:
                        resized = cv2.resize(warped, (TILE_DISPLAY_SIZE, TILE_DISPLAY_SIZE))
                        y = (row - min_row) * TILE_DISPLAY_SIZE
                        x = (col - min_col) * TILE_DISPLAY_SIZE
                        mosaic_a[y:y + TILE_DISPLAY_SIZE, x:x + TILE_DISPLAY_SIZE] = resized
                except Exception:
                    continue

        # Populate mosaic for the second frame
        if second.get("grid_map"):
            for (row, col), square in second["grid_map"].items():
                try:
                    warped = extract_and_warp_square(second["frame"], square)
                    if warped is not None and warped.size > 0:
                        resized = cv2.resize(warped, (TILE_DISPLAY_SIZE, TILE_DISPLAY_SIZE))
                        y = (row - min_row) * TILE_DISPLAY_SIZE
                        x = (col - min_col) * TILE_DISPLAY_SIZE
                        mosaic_b[y:y + TILE_DISPLAY_SIZE, x:x + TILE_DISPLAY_SIZE] = resized
                except Exception:
                    continue

        # --- Add individual mosaics to the display list ---
        # Add frame A
        mosaic_a_display = mosaic_a.copy()
        cv2.putText(mosaic_a_display, f"Frame {frame_idx}", (20, 40), label_font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        mosaics.append(mosaic_a_display)

        # Add frame B
        mosaic_b_display = mosaic_b.copy()
        cv2.putText(mosaic_b_display, f"Frame {next_idx}", (20, 40), label_font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        mosaics.append(mosaic_b_display)

        # --- Create and add the overlay ---
        # Apply offset to the second mosaic
        mosaic_b_shifted = mosaic_b
        if x_offset != 0 or y_offset != 0:
            M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
            mosaic_b_shifted = cv2.warpAffine(mosaic_b, M, (mosaic_b.shape[1], mosaic_b.shape[0]), borderValue=(40, 40, 40))

        # Calculate pixel distance
        background_color = np.array([40, 40, 40], dtype=np.uint8)
        mask_a = np.any(mosaic_a != background_color, axis=-1)
        mask_b_shifted = np.any(mosaic_b_shifted != background_color, axis=-1)
        overlap_mask = mask_a & mask_b_shifted

        pixel_dist = 0.0
        if np.any(overlap_mask):
            diff = np.abs(mosaic_a.astype(np.float32) - mosaic_b_shifted.astype(np.float32))
            pixel_dist = np.sum(diff[overlap_mask])
        
        overlap_area = np.sum(overlap_mask)
        normalized_dist = pixel_dist / overlap_area if overlap_area > 0 else 0.0

        overlay = cv2.addWeighted(mosaic_a, 0.5, mosaic_b_shifted, 0.5, 0)
        cv2.putText(
            overlay, f"Overlay {frame_idx} + {next_idx}", (20, 40), label_font, 0.9, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            overlay, f"Offset: ({x_offset}, {y_offset})", (20, 80), label_font, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        mosaics.append(overlay)

    # --- Final composition of the views ---
    # Match heights and stack horizontally
    max_height = max(img.shape[0] for img in mosaics) if mosaics else 0
    mosaics_padded = [pad_to_height(img, max_height) for img in mosaics]
    
    # Create blue vertical separators between views
    separator_width = 1
    separator = np.full((max_height, separator_width, 3), (255, 0, 0), dtype=np.uint8)  # Blue in BGR
    
    # Interleave mosaics with separators
    mosaics_with_separators = []
    for i, mosaic in enumerate(mosaics_padded):
        mosaics_with_separators.append(mosaic)
        if i < len(mosaics_padded) - 1:
            mosaics_with_separators.append(separator)
    
    if not mosaics_with_separators:
        return np.full((400, 800, 3), 30, dtype=np.uint8), 0.0, 0.0 # Return a blank image if something goes wrong

    combined = np.hstack(mosaics_with_separators)

    return combined, pixel_dist, normalized_dist


def run_viewer(args: argparse.Namespace) -> None:
    cache = FrameCache(args.video)
    total_frames = cache.get_total_frames()
    if total_frames < 2:
        cache.release()
        raise RuntimeError("Video must contain at least two frames.")

    window_name = "Frame Pair Overlay Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    pair_index = max(0, args.start_frame)
    if pair_index >= total_frames - 1:
        pair_index = total_frames - 2

    try:
        while True:
            first = cache.get_frame_data(pair_index)
            second = cache.get_frame_data(pair_index + 1)
            if first is None or second is None:
                print(f"Could not load frame pair {pair_index}, {pair_index + 1}")
                break

            display, _, _ = compose_display(first, second, pair_index, pair_index + 1)

            max_width = int(args.max_window_width)
            max_height = int(args.max_window_height)
            h, w = display.shape[:2]
            scale = min(max_width / max(w, 1), max_height / max(h, 1))
            if scale < 1.0:
                display = cv2.resize(display, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("d"):
                if pair_index < total_frames - 2:
                    pair_index += 1
                else:
                    print("Reached final frame pair")
            elif key == ord("a"):
                if pair_index > 0:
                    pair_index -= 1
                else:
                    print("Already at first frame pair")
    finally:
        cache.release()
        cv2.destroyWindow(window_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View consecutive frame pairs with overlay and unwarped tiles.")
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to input video file")
    parser.add_argument("--start-frame", type=int, default=0, help="Index of the first frame in the initial pair")
    parser.add_argument(
        "--max-window-width",
        type=float,
        default=2400,
        help="Maximum display window width in pixels (display is resized if wider)",
    )
    parser.add_argument(
        "--max-window-height",
        type=float,
        default=1350,
        help="Maximum display window height in pixels (display is resized if taller)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_viewer(parse_args())

