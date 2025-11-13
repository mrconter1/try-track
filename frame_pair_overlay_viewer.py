import argparse
import cv2
import numpy as np
from typing import Optional, Dict, Tuple

from line_detector import LineDetector
from auto_tile_detector import extract_grid_squares, extract_and_warp_square


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

    tile_display_size = 80
    mosaic = np.full((num_rows * tile_display_size, num_cols * tile_display_size, 3), 40, dtype=np.uint8)

    for (row, col), square in grid_map.items():
        try:
            warped = extract_and_warp_square(frame, square)
            if warped is None or warped.size == 0:
                continue
            resized = cv2.resize(warped, (tile_display_size, tile_display_size))
            y_start = row * tile_display_size
            x_start = col * tile_display_size
            mosaic[y_start : y_start + tile_display_size, x_start : x_start + tile_display_size] = resized
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


def compose_display(first: Dict, second: Dict, frame_idx: int, next_idx: int) -> np.ndarray:
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Determine max dimensions if both mosaics exist
    max_h = 0
    max_w = 0
    if first["mosaic"] is not None and second["mosaic"] is not None:
        max_h = max(first["mosaic"].shape[0], second["mosaic"].shape[0])
        max_w = max(first["mosaic"].shape[1], second["mosaic"].shape[1])
    
    mosaics = []
    
    # First unwarped mosaic
    if first["mosaic"] is not None:
        mosaic_a = first["mosaic"].copy()
        if max_h > 0 and max_w > 0:
            mosaic_a = pad_to_height(mosaic_a, max_h)
            mosaic_a = pad_to_width(mosaic_a, max_w)
        cv2.putText(mosaic_a, f"Frame {frame_idx}", (20, 40), label_font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        mosaics.append(mosaic_a)
    else:
        blank = np.full((200, 200, 3), 60, dtype=np.uint8)
        cv2.putText(blank, f"Frame {frame_idx}", (20, 40), label_font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(blank, "No grid", (30, 110), label_font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        mosaics.append(blank)
    
    # Second unwarped mosaic
    if second["mosaic"] is not None:
        mosaic_b = second["mosaic"].copy()
        if max_h > 0 and max_w > 0:
            mosaic_b = pad_to_height(mosaic_b, max_h)
            mosaic_b = pad_to_width(mosaic_b, max_w)
        cv2.putText(mosaic_b, f"Frame {next_idx}", (20, 40), label_font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        mosaics.append(mosaic_b)
    else:
        blank = np.full((200, 200, 3), 60, dtype=np.uint8)
        cv2.putText(blank, f"Frame {next_idx}", (20, 40), label_font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(blank, "No grid", (30, 110), label_font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        mosaics.append(blank)
    
    # Overlay unwarped mosaic
    if first["mosaic"] is not None and second["mosaic"] is not None:
        mosaic_a = first["mosaic"].copy()
        mosaic_b = second["mosaic"].copy()
        
        if max_h > 0 and max_w > 0:
            mosaic_a = pad_to_height(mosaic_a, max_h)
            mosaic_a = pad_to_width(mosaic_a, max_w)
            mosaic_b = pad_to_height(mosaic_b, max_h)
            mosaic_b = pad_to_width(mosaic_b, max_w)
        
        overlay = cv2.addWeighted(mosaic_a, 0.5, mosaic_b, 0.5, 0)
        cv2.putText(
            overlay,
            f"Overlay {frame_idx} + {next_idx}",
            (20, 40),
            label_font,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        mosaics.append(overlay)
    else:
        blank = np.full((200, 200, 3), 60, dtype=np.uint8)
        cv2.putText(blank, f"Overlay {frame_idx} + {next_idx}", (20, 40), label_font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(blank, "No grid", (30, 110), label_font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        mosaics.append(blank)
    
    # Match heights and stack horizontally
    max_height = max(img.shape[0] for img in mosaics)
    mosaics_padded = [pad_to_height(img, max_height) for img in mosaics]
    combined = np.hstack(mosaics_padded)
    
    return combined


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

            display = compose_display(first, second, pair_index, pair_index + 1)

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

