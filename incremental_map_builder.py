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

    def render_map(self, 
                   highlight_tiles: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
                   frame_offset: Optional[Tuple[int, int]] = (0,0)
                  ) -> np.ndarray:
        """Renders the current state of the global map into a single image."""

        all_keys = list(self.tiles.keys())
        if highlight_tiles:
            for r, c in highlight_tiles.keys():
                # Calculate the prospective global position for highlighting
                global_r = self.current_pos[0] + r + frame_offset[0]
                global_c = self.current_pos[1] + c + frame_offset[1]
                all_keys.append((global_r, global_c))

        if not all_keys:
            return np.full((400, 400, 3), 60, dtype=np.uint8)

        min_r = min(r for r, c in all_keys)
        max_r = max(r for r, c in all_keys)
        min_c = min(c for r, c in all_keys)
        max_c = max(c for r, c in all_keys)

        map_h = (max_r - min_r + 1) * TILE_DISPLAY_SIZE
        map_w = (max_c - min_c + 1) * TILE_DISPLAY_SIZE
        
        vis_map = np.full((map_h, map_w, 3), 40, dtype=np.uint8)

        for (r, c), tile_img in self.tiles.items():
            y = (r - min_r) * TILE_DISPLAY_SIZE
            x = (c - min_c) * TILE_DISPLAY_SIZE
            vis_map[y:y + TILE_DISPLAY_SIZE, x:x + TILE_DISPLAY_SIZE] = tile_img
        
        # Draw highlighted tiles (the current frame's match)
        if highlight_tiles and frame_offset:
            for (r, c), tile_img in highlight_tiles.items():
                global_r = self.current_pos[0] + r + frame_offset[0]
                global_c = self.current_pos[1] + c + frame_offset[1]
                
                y = (global_r - min_r) * TILE_DISPLAY_SIZE
                x = (global_c - min_c) * TILE_DISPLAY_SIZE

                # Blend the new tile with the background for visualization
                if (global_r, global_c) in self.tiles:
                    # If it overlaps, blend with existing tile
                    existing_tile = vis_map[y:y + TILE_DISPLAY_SIZE, x:x + TILE_DISPLAY_SIZE]
                    vis_map[y:y + TILE_DISPLAY_SIZE, x:x + TILE_DISPLAY_SIZE] = cv2.addWeighted(existing_tile, 0.5, tile_img, 0.5, 0)
                else:
                    # Otherwise, just place it
                    vis_map[y:y + TILE_DISPLAY_SIZE, x:x + TILE_DISPLAY_SIZE] = tile_img

                # Draw a bright green rectangle to highlight it
                cv2.rectangle(vis_map, (x, y), (x + TILE_DISPLAY_SIZE - 1, y + TILE_DISPLAY_SIZE - 1), (0, 255, 0), 2)

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
    search_range: int = 3,
    min_overlap_tiles: int = 3,
) -> List[Tuple[Tuple[int, int], float, int]]:
    """
    Finds all valid offsets for a new frame against the global map,
    sorted by the best match (lowest average pixel difference).
    """
    valid_offsets = []

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
            
            if overlap_count >= min_overlap_tiles:
                # Normalize by total pixels in overlap area to get a per-pixel average
                avg_diff = total_diff / (overlap_count * TILE_DISPLAY_SIZE * TILE_DISPLAY_SIZE * 3)
                valid_offsets.append(((r_offset, c_offset), avg_diff, overlap_count))

    # Sort by the average difference (second element) and return the full list
    valid_offsets.sort(key=lambda item: item[1])
    return valid_offsets

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
            print(f"Frame {frame_idx}: Initializing map.")
            map_vis = global_map.render_map(highlight_tiles=frame_tiles, frame_offset=best_offset)
            global_map.add_tiles_from_frame(frame_tiles, best_offset)
        else:
            sorted_offsets = find_best_offset(global_map, frame_tiles)

            if sorted_offsets:
                # A confident match was found
                best_offset, _, _ = sorted_offsets[0]
                
                # Render the map WITH highlights before updating the map state
                map_vis = global_map.render_map(highlight_tiles=frame_tiles, frame_offset=best_offset)

                global_map.add_tiles_from_frame(frame_tiles, best_offset)
                global_map.current_pos = (
                    global_map.current_pos[0] + best_offset[0],
                    global_map.current_pos[1] + best_offset[1]
                )
                print(f"Frame {frame_idx}: Match found. New position: {global_map.current_pos}")
                print("--- Top 5 Matches ---")
                for (offset, avg_diff, overlap) in sorted_offsets[:5]:
                    print(f"  Offset: {str(offset):>8s}, Overlap: {overlap:2d} tiles, Avg Diff: {avg_diff:.2f}")

            else:
                # No confident match found, do not update position or map
                print(f"Frame {frame_idx}: No confident match found (overlap < 3 tiles). Position held at {global_map.current_pos}")
                map_vis = global_map.render_map() # Render without highlights


        cv2.putText(map_vis, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(map_vis, f"Current Position: {global_map.current_pos}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

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

