import cv2
import numpy as np
import argparse
from typing import Dict, Tuple, Optional, List

from line_detector import LineDetector
from auto_tile_detector import extract_grid_squares, extract_and_warp_square

TILE_DISPLAY_SIZE = 80

class GlobalMap:
    """Manages the state of the incrementally built global tile map."""
    def __init__(self):
        self.tiles: Dict[Tuple[int, int], np.ndarray] = {}
        self.current_pos = (0, 0)

    def add_tiles_from_frame(self, frame_tiles: Dict[Tuple[int, int], np.ndarray], frame_offset: Tuple[int, int]):
        """Adds new, unseen tiles from a frame to the global map."""
        for (r, c), tile_img in frame_tiles.items():
            global_r = self.current_pos[0] + r + frame_offset[0]
            global_c = self.current_pos[1] + c + frame_offset[1]
            if (global_r, global_c) not in self.tiles:
                self.tiles[(global_r, global_c)] = tile_img

    def render_map(self) -> np.ndarray:
        """Renders the current state of the global map into a single image."""
        if not self.tiles:
            return np.full((400, 400, 3), 60, dtype=np.uint8)

        min_r = min(r for r, c in self.tiles.keys())
        max_r = max(r for r, c in self.tiles.keys())
        min_c = min(c for r, c in self.tiles.keys())
        max_c = max(c for r, c in self.tiles.keys())

        map_h = (max_r - min_r + 1) * TILE_DISPLAY_SIZE
        map_w = (max_c - min_c + 1) * TILE_DISPLAY_SIZE
        
        vis_map = np.full((map_h, map_w, 3), 40, dtype=np.uint8)

        for (r, c), tile_img in self.tiles.items():
            y = (r - min_r) * TILE_DISPLAY_SIZE
            x = (c - min_c) * TILE_DISPLAY_SIZE
            vis_map[y:y + TILE_DISPLAY_SIZE, x:x + TILE_DISPLAY_SIZE] = tile_img
        
        # Draw a dot for the current position
        pos_r, pos_c = self.current_pos
        dot_y = (pos_r - min_r) * TILE_DISPLAY_SIZE + TILE_DISPLAY_SIZE // 2
        dot_x = (pos_c - min_c) * TILE_DISPLAY_SIZE + TILE_DISPLAY_SIZE // 2
        cv2.circle(vis_map, (dot_x, dot_y), radius=10, color=(255, 0, 0), thickness=-1) # Blue dot

        return vis_map

class FrameProcessor:
    """Handles video loading and processing of individual frames."""
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")
        self.line_detector = LineDetector()
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        if index < 0 or index >= self.total_frames:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        return frame if ret else None

    def extract_tiles(self, frame: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
        """Extracts warped tiles from a single frame."""
        tiles = {}
        try:
            _, lines = self.line_detector.detect_lines(frame)
            grid_map = extract_grid_squares(lines, frame.shape[:2]) if lines is not None else {}
            for (r, c), square in grid_map.items():
                warped = extract_and_warp_square(frame, square)
                if warped is not None and warped.size > 0:
                    resized = cv2.resize(warped, (TILE_DISPLAY_SIZE, TILE_DISPLAY_SIZE))
                    tiles[(r, c)] = resized
        except Exception:
            pass # Ignore frames where detection fails
        return tiles

    def release(self):
        self.cap.release()

def find_best_offset(
    global_map: GlobalMap,
    frame_tiles: Dict[Tuple[int, int], np.ndarray],
    search_range: int = 3
) -> Tuple[int, int]:
    """Finds the best offset for a new frame against the global map."""
    best_offset = (0, 0)
    min_avg_diff = float('inf')

    for r_offset in range(-search_range, search_range + 1):
        for c_offset in range(-search_range, search_range + 1):
            
            total_diff = 0
            overlap_count = 0

            for (r, c), frame_tile in frame_tiles.items():
                global_r = global_map.current_pos[0] + r + r_offset
                global_c = global_map.current_pos[1] + c + c_offset

                if (global_r, global_c) in global_map.tiles:
                    global_tile = global_map.tiles[(global_r, global_c)]
                    diff = np.sum(np.abs(frame_tile.astype(np.float32) - global_tile.astype(np.float32)))
                    total_diff += diff
                    overlap_count += 1
            
            if overlap_count > 0:
                avg_diff = total_diff / (overlap_count * TILE_DISPLAY_SIZE * TILE_DISPLAY_SIZE * 3)
                if avg_diff < min_avg_diff:
                    min_avg_diff = avg_diff
                    best_offset = (r_offset, c_offset)

    return best_offset

def main(args):
    processor = FrameProcessor(args.video)
    global_map = GlobalMap()
    
    window_name = "Incremental Map Builder"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    while frame_idx < processor.total_frames:
        frame = processor.get_frame(frame_idx)
        if frame is None:
            break

        frame_tiles = processor.extract_tiles(frame)

        if not frame_tiles:
            print(f"Frame {frame_idx}: No tiles found, skipping.")
            frame_idx += 1
            continue

        if not global_map.tiles: # First frame
            best_offset = (0, 0)
        else:
            best_offset = find_best_offset(global_map, frame_tiles)

        global_map.add_tiles_from_frame(frame_tiles, best_offset)
        global_map.current_pos = (
            global_map.current_pos[0] + best_offset[0],
            global_map.current_pos[1] + best_offset[1]
        )

        map_vis = global_map.render_map()
        cv2.putText(map_vis, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(map_vis, f"Current Offset: {global_map.current_pos}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow(window_name, map_vis)
        
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):
            break
        frame_idx += 1

    processor.release()
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="Incrementally build a map from video frames.")
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to input video file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

