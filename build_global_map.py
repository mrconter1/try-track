import cv2
import numpy as np
import argparse
import json
import os
from pathlib import Path
from line_detector import LineDetector
from auto_tile_detector import (
    generate_tile_hash, 
    extract_grid_squares, 
    extract_and_warp_square,
    euclidean_distance
)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

class GlobalMapBuilder:
    def __init__(self, video_path, blocks=16, use_clahe=True, tile_max_std=None, clahe_clip_limit=2.0, num_frames=None):
        """Initialize the global map builder"""
        self.video_path = video_path
        self.blocks = blocks
        self.use_clahe = use_clahe
        self.tile_max_std = tile_max_std
        self.clahe_clip_limit = clahe_clip_limit
        self.num_frames = num_frames  # None = process all frames
        
        self.line_detector = LineDetector()
        
        # Global state
        self.tiles = []  # Array of tile objects: {tile_id, signatures, image, frame_number, row, col}
        self.frames = []  # Array of frame grids: {frame_number, grid: dict[(row,col) -> tile_id]}
        self.next_tile_id = 0
        
        self.tile_global_positions = {}  # tile_id -> (global_row, global_col)
        
        print(f"[GlobalMapBuilder] Initialized with:")
        print(f"  Video: {video_path}")
        print(f"  Blocks per side: {blocks}")
        print(f"  CLAHE: {use_clahe} (clip_limit={clahe_clip_limit})")
        print(f"  Max std dev threshold: {tile_max_std}")
        print(f"  Frames to process: {num_frames if num_frames else 'All'}")
    
    def process_video(self):
        """Process all frames in video"""
        print("\n" + "="*70)
        print("[STEP 1] FRAME PROCESSING PIPELINE")
        print("="*70)
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_process = self.num_frames if self.num_frames else total_frames
        print(f"\nProcessing {frames_to_process} frames (out of {total_frames} total)...")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Stop if we've reached the limit
            if self.num_frames and frame_idx >= self.num_frames:
                break
            
            if frame_idx % 10 == 0:
                print(f"\nFrame {frame_idx}/{frames_to_process}...", end=" ")
            
            # Detect lines
            _, lines = self.line_detector.detect_lines(frame)
            
            # Extract grid squares
            squares = extract_grid_squares(lines, frame.shape[:2])
            
            # Organize squares into 2D grid
            grid_dict, num_rows, num_cols = self._organize_squares_into_grid(squares)
            
            # Process frame
            if grid_dict:
                frame_grid = self._process_frame_grid(frame, squares, grid_dict, frame_idx)
                if frame_grid:  # Only store if we got tiles
                    self.frames.append({
                        'frame_number': frame_idx,
                        'grid': frame_grid,
                        'num_rows': num_rows,
                        'num_cols': num_cols
                    })
                    if frame_idx % 10 == 0:
                        print(f"({len(frame_grid)} tiles in {num_rows}x{num_cols} grid)")
            
            frame_idx += 1
        
        cap.release()
        
        print(f"\n✓ Processed {len(self.frames)} frames with tiles")
        print(f"✓ Total tiles extracted: {len(self.tiles)}")
    
    def _organize_squares_into_grid(self, squares):
        """Convert flat list of squares into 2D grid structure by position"""
        if not squares:
            return {}, 0, 0
        
        # Get center of each square
        centers = []
        for idx, square in enumerate(squares):
            p1, p2, p3, p4 = square
            cx = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
            cy = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
            centers.append((cx, cy, idx))
        
        # Sort by y (top to bottom), then x (left to right)
        centers_sorted = sorted(centers, key=lambda p: (p[1], p[0]))
        
        # Group into rows based on y-coordinate similarity
        rows = []
        current_row = []
        row_threshold = 50  # pixels - adjust based on tile size
        
        for cx, cy, idx in centers_sorted:
            if not current_row:
                current_row.append((cx, cy, idx))
            else:
                # Check if in same row
                if abs(cy - current_row[0][1]) < row_threshold:
                    current_row.append((cx, cy, idx))
                else:
                    # Start new row
                    rows.append(current_row)
                    current_row = [(cx, cy, idx)]
        
        if current_row:
            rows.append(current_row)
        
        # Create 2D grid mapping (row, col) -> square_index
        grid_dict = {}
        max_cols = 0
        for row_idx, row in enumerate(rows):
            row_sorted = sorted(row, key=lambda p: p[0])  # Sort by x within row
            max_cols = max(max_cols, len(row_sorted))
            for col_idx, (cx, cy, square_idx) in enumerate(row_sorted):
                grid_dict[(row_idx, col_idx)] = square_idx
        
        return grid_dict, len(rows), max_cols
    
    def _process_frame_grid(self, frame, squares, grid_dict, frame_number):
        """Process squares in a frame and return 2D grid of tile_ids"""
        frame_grid = {}
        
        for (row, col), square_idx in grid_dict.items():
            try:
                square = squares[square_idx]
                warped = extract_and_warp_square(frame, square)
                
                # Check std dev threshold (quality filter)
                if self.tile_max_std is not None:
                    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    std_dev = float(np.std(gray_warped))
                    if std_dev > self.tile_max_std:
                        continue  # Skip this tile
                
                # Generate signatures for all 4 rotations
                signatures = {}
                for rotation in [0, 90, 180, 270]:
                    if rotation == 0:
                        rotated = warped
                    elif rotation == 90:
                        rotated = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif rotation == 180:
                        rotated = cv2.rotate(warped, cv2.ROTATE_180)
                    else:  # 270
                        rotated = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                    
                    hash_sig = generate_tile_hash(rotated, self.blocks, self.use_clahe)
                    if hash_sig:
                        signatures[rotation] = hash_sig
                
                if signatures:
                    # Create tile object
                    tile_obj = {
                        'tile_id': self.next_tile_id,
                        'signatures': signatures,
                        'image': warped,
                        'frame_number': frame_number,
                        'frame_row': row,
                        'frame_col': col,
                    }
                    self.tiles.append(tile_obj)
                    
                    # Add to frame grid with (row, col) key
                    frame_grid[(row, col)] = self.next_tile_id
                    self.next_tile_id += 1
            
            except Exception as e:
                print(f"  Error processing tile at ({row},{col}): {e}")
                continue
        
        return frame_grid if frame_grid else None
    
    def build_global_map(self):
        """Build global map from processed frames using temporal frame-to-frame matching"""
        print("\n" + "="*70)
        print("[STEP 2] GLOBAL MAP BUILDING (Temporal Frame-to-Frame)")
        print("="*70)
        
        if not self.frames:
            print("✗ No frames to process")
            return
        
        print(f"\nStitching {len(self.frames)} frames into global map...")
        
        # Frame 0: Anchor at origin
        frame_0_data = self.frames[0]
        frame_0_grid = frame_0_data['grid']
        print(f"\n  Frame 0: {len(frame_0_grid)} tiles - Anchoring at origin")
        for (local_row, local_col), tile_id in frame_0_grid.items():
            if tile_id is not None:
                self.tile_global_positions[tile_id] = (local_row, local_col)
        
        # Process remaining frames sequentially
        for frame_idx in range(1, len(self.frames)):
            frame_data = self.frames[frame_idx]
            frame_grid = frame_data['grid']
            num_rows = frame_data['num_rows']
            num_cols = frame_data['num_cols']
            
            prev_frame_data = self.frames[frame_idx - 1]
            prev_frame_grid = prev_frame_data['grid']
            
            print(f"\n  Frame {frame_idx}: {len(frame_grid)} tiles ({num_rows}x{num_cols} local grid)", end=" - ")
            
            # Build similarity matrix between current frame and previous frame
            similarity_matrix, rotation_matrix = self._build_frame_similarity_matrix(frame_grid, prev_frame_grid)
            
            # Find the best match (smallest distance) in entire similarity matrix
            best_overall_distance = float('inf')
            first_match = None
            
            for (curr_tile_id, prev_tile_id), distance in similarity_matrix.items():
                if distance < best_overall_distance:
                    # Find positions for these tiles
                    curr_pos = None
                    for (row, col), tid in frame_grid.items():
                        if tid == curr_tile_id:
                            curr_pos = (row, col)
                            break
                    
                    prev_pos = None
                    for (row, col), tid in prev_frame_grid.items():
                        if tid == prev_tile_id:
                            prev_pos = (row, col)
                            break
                    
                    if curr_pos and prev_pos and prev_tile_id in self.tile_global_positions:
                        best_overall_distance = distance
                        rotation_offset = rotation_matrix[(curr_tile_id, prev_tile_id)]
                        first_match = (curr_pos[0], curr_pos[1], curr_tile_id, prev_tile_id, prev_pos, distance, rotation_offset)
            
            if first_match:
                curr_row, curr_col, curr_tile_id, prev_tile_id, prev_pos, distance, rotation_offset = first_match
                
                # Get global position of previous frame tile
                prev_global_pos = self.tile_global_positions[prev_tile_id]
                
                # If rotation detected, apply coordinate transform to match orientation
                if rotation_offset != 0:
                    curr_row, curr_col = self._rotate_coordinates(curr_row, curr_col, num_rows, num_cols, rotation_offset)
                
                # Calculate offset between current and previous frame positions
                offset_row = prev_global_pos[0] - curr_row
                offset_col = prev_global_pos[1] - curr_col
                
                print(f"Matched to frame {frame_idx-1} (distance: {distance:.0f}, rotation: {rotation_offset}°)")
                print(f"  Anchor: tile {curr_tile_id} at local ({curr_row},{curr_col}) → tile {prev_tile_id} at global {prev_global_pos}")
                print(f"  Offset: row={offset_row}, col={offset_col}")
                
                # Add all tiles from current frame EXCEPT the matched one (which is already in global map)
                tiles_added = 0
                for (local_row, local_col), tile_id in frame_grid.items():
                    if tile_id is None:
                        continue
                    
                    # Skip the matched tile (already in global map from previous frame)
                    if tile_id == curr_tile_id:
                        print(f"  Skipping matched tile {curr_tile_id} (already in global map)")
                        continue
                    
                    # Apply rotation to local coordinates if needed
                    if rotation_offset != 0:
                        rotated_row, rotated_col = self._rotate_coordinates(local_row, local_col, num_rows, num_cols, rotation_offset)
                    else:
                        rotated_row, rotated_col = local_row, local_col
                    
                    global_row = rotated_row + offset_row
                    global_col = rotated_col + offset_col
                    self.tile_global_positions[tile_id] = (global_row, global_col)
                    tiles_added += 1
                
                print(f"  Added {tiles_added} new tiles to global map")
            else:
                print(f"No match to previous frame, skipping")
        
        print(f"\n✓ Built global map with {len(self.tile_global_positions)} positioned tiles")
    
    def _build_frame_similarity_matrix(self, frame_grid, prev_frame_grid):
        """Build similarity matrix between tiles in current and previous frame"""
        similarity_matrix = {}
        rotation_matrix = {}  # Store rotation offsets
        
        # Get sorted lists of tile IDs for matrix
        curr_tile_ids = sorted([tid for _, tid in frame_grid.items() if tid is not None])
        prev_tile_ids = sorted([tid for _, tid in prev_frame_grid.items() if tid is not None])
        
        # Build matrix
        matrix_2d = np.zeros((len(curr_tile_ids), len(prev_tile_ids)))
        
        for i, curr_tile_id in enumerate(curr_tile_ids):
            curr_tile = next((t for t in self.tiles if t['tile_id'] == curr_tile_id), None)
            if not curr_tile:
                continue
            
            for j, prev_tile_id in enumerate(prev_tile_ids):
                prev_tile = next((t for t in self.tiles if t['tile_id'] == prev_tile_id), None)
                if not prev_tile:
                    continue
                
                # Calculate distance between tiles (with rotation info)
                distance, rotation_offset = self._tile_distance(curr_tile, prev_tile)
                similarity_matrix[(curr_tile_id, prev_tile_id)] = distance
                rotation_matrix[(curr_tile_id, prev_tile_id)] = rotation_offset
                matrix_2d[i, j] = distance
        
        # Print 2D matrix
        self._print_similarity_matrix(curr_tile_ids, prev_tile_ids, matrix_2d)
        
        return similarity_matrix, rotation_matrix
    
    def _print_similarity_matrix(self, curr_tile_ids, prev_tile_ids, matrix_2d):
        """Print 2D similarity matrix in readable format"""
        print("\n    ┌─ SIMILARITY MATRIX (distances) ─┐")
        
        # Header
        print("    Curr \\ Prev", end="")
        for prev_id in prev_tile_ids:
            print(f" {prev_id:4d}", end="")
        print()
        
        # Separator
        print("    " + "─" * (12 + len(prev_tile_ids) * 5))
        
        # Rows
        for i, curr_id in enumerate(curr_tile_ids):
            print(f"    {curr_id:4d}    |", end="")
            for j, _ in enumerate(prev_tile_ids):
                dist = matrix_2d[i, j]
                if dist < 300:
                    print(f" {dist:4.0f}", end="")
                else:
                    print(f" {dist:4.0f}", end="")
            print()
        
        # Find best matches
        print("\n    Best matches per current tile:")
        for i, curr_id in enumerate(curr_tile_ids):
            best_j = np.argmin(matrix_2d[i, :])
            best_dist = matrix_2d[i, best_j]
            prev_id = prev_tile_ids[best_j]
            marker = "✓" if best_dist < 650 else "✗"
            print(f"      {marker} Tile {curr_id} → Tile {prev_id}: {best_dist:.0f}")
    
    def _tile_distance(self, tile1, tile2):
        """Calculate distance between two tiles (checking all rotations)
        Returns: (distance, rotation_offset) where rotation_offset is how much tile1 needs to rotate to match tile2
        """
        sigs1 = tile1['signatures']
        sigs2 = tile2['signatures']
        
        best_distance = float('inf')
        best_rotation_offset = 0
        
        for rot1 in [0, 90, 180, 270]:
            for rot2 in [0, 90, 180, 270]:
                sig1 = sigs1.get(rot1)
                sig2 = sigs2.get(rot2)
                
                if sig1 and sig2:
                    distance = euclidean_distance(sig1, sig2)
                    if distance < best_distance:
                        best_distance = distance
                        # Calculate rotation offset: how much to rotate tile1 to match tile2's orientation
                        best_rotation_offset = (rot2 - rot1) % 360
        
        return best_distance, best_rotation_offset
    
    def _rotate_coordinates(self, row, col, height, width, rotation_offset):
        """Transform (row, col) based on rotation offset"""
        if rotation_offset == 0:
            return row, col
        elif rotation_offset == 90:
            return col, height - 1 - row
        elif rotation_offset == 180:
            return height - 1 - row, width - 1 - col
        elif rotation_offset == 270:
            return width - 1 - col, row
        else:
            return row, col
    
    def _find_match_in_global_map(self, tile, threshold=100, verbose=False):
        """Find if tile matches any tile in global map (checks all rotations)"""
        if not self.tile_global_positions:
            return None, None
        
        tile_id = tile['tile_id']
        tile_sigs = tile['signatures']
        
        best_match_id = None
        best_match_rotation = None
        best_distance = float('inf')
        
        # Try to match against all tiles in global map
        for global_tile_id in self.tile_global_positions.keys():
            global_tile = next((t for t in self.tiles if t['tile_id'] == global_tile_id), None)
            if not global_tile:
                continue
            
            global_sigs = global_tile['signatures']
            
            # Try all rotation combinations
            for curr_rot in [0, 90, 180, 270]:
                for global_rot in [0, 90, 180, 270]:
                    curr_sig = tile_sigs.get(curr_rot)
                    global_sig = global_sigs.get(global_rot)
                    
                    if curr_sig and global_sig:
                        distance = euclidean_distance(curr_sig, global_sig)
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_match_id = global_tile_id
                            best_match_rotation = (global_rot - curr_rot) % 360
        
        # Consider it a match if distance is low enough
        if best_distance < threshold:
            if verbose:
                print(f"      [MATCH] Tile {tile_id} → Global {best_match_id}, distance: {best_distance:.0f}")
            return best_match_id, best_match_rotation
        else:
            if verbose:
                print(f"      [BEST] Tile {tile_id}: distance {best_distance:.0f} (threshold: {threshold})")
        
        return None, None
    
    def visualize_global_map(self, max_display_size=2000, save_path=None, fixed_grid_size=1000):
        """Visualize the global map as a 2D grid of tiles
        
        Args:
            max_display_size: Maximum dimension in pixels for display (prevents memory issues)
            save_path: If provided, save to PNG instead of displaying
            fixed_grid_size: Use a fixed NxN grid (default 1000x1000) instead of calculating bounds
        """
        print("\n" + "="*70)
        print("[STEP 3] VISUALIZATION")
        print("="*70)
        
        if not self.tile_global_positions:
            print("✗ No tiles in global map to visualize")
            return
        
        print("\nBuilding visualization...")
        
        # Get actual data bounds
        positions = list(self.tile_global_positions.values())
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        
        data_min_row, data_max_row = int(min(rows)), int(max(rows))
        data_min_col, data_max_col = int(min(cols)), int(max(cols))
        data_height = data_max_row - data_min_row + 1
        data_width = data_max_col - data_min_col + 1
        
        print(f"  Data bounds: Rows({data_min_row}, {data_max_row}) Cols({data_min_col}, {data_max_col})")
        print(f"  Data size: {data_width}x{data_height}")
        
        # Use fixed grid canvas but add padding around data
        if fixed_grid_size:
            # Add 10% padding around data
            padding = int(max(data_height, data_width) * 0.1)
            min_row = max(0, data_min_row - padding)
            max_row = min(fixed_grid_size - 1, data_max_row + padding)
            min_col = max(0, data_min_col - padding)
            max_col = min(fixed_grid_size - 1, data_max_col + padding)
            
            grid_height = max_row - min_row + 1
            grid_width = max_col - min_col + 1
            
            print(f"  Using fixed {fixed_grid_size}x{fixed_grid_size} canvas, viewing: Rows({min_row}, {max_row}) Cols({min_col}, {max_col})")
        else:
            # Use data bounds
            min_row, max_row = data_min_row, data_max_row
            min_col, max_col = data_min_col, data_max_col
            grid_height = data_height
            grid_width = data_width
            print(f"  Using auto-zoom to data bounds")
        
        print(f"  Display size: {grid_width} x {grid_height}")
        print(f"  Total tiles: {len(self.tile_global_positions)}")
        
        # Dynamically scale tile size based on grid size
        # For large maps, use smaller tiles to fit in memory
        max_dim = max(grid_height, grid_width)
        
        if max_dim > 100:
            tile_size_display = max(10, min(80, max_display_size // max_dim))
            print(f"  Large map detected ({max_dim}x): scaling tiles to {tile_size_display}px")
        else:
            tile_size_display = 80
        
        # Calculate figure size (cap at reasonable limits)
        fig_width_px = grid_width * tile_size_display
        fig_height_px = grid_height * tile_size_display
        
        # Cap at max_display_size
        if fig_width_px > max_display_size or fig_height_px > max_display_size:
            scale_factor = max_display_size / max(fig_width_px, fig_height_px)
            fig_width_px = int(fig_width_px * scale_factor)
            fig_height_px = int(fig_height_px * scale_factor)
            tile_size_display = int(tile_size_display * scale_factor)
            print(f"  Scaling down to fit display: {fig_width_px}x{fig_height_px}px")
        
        # Convert pixels to inches for matplotlib (DPI=100)
        dpi = 100
        fig_width_in = fig_width_px / dpi
        fig_height_in = fig_height_px / dpi
        
        print(f"  Creating figure: {fig_width_px}x{fig_height_px}px ({fig_width_in:.1f}x{fig_height_in:.1f}in)")
        
        fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
        
        # Draw each tile thumbnail at its global position
        displayed_tiles = 0
        for tile_id, (global_row, global_col) in self.tile_global_positions.items():
            tile_obj = next((t for t in self.tiles if t['tile_id'] == tile_id), None)
            if not tile_obj:
                continue
            
            tile_image = tile_obj['image']
            
            # Resize thumbnail
            thumb = cv2.resize(tile_image, (tile_size_display, tile_size_display))
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            
            # Position on grid (normalize to start at 0,0)
            x_pos = (global_col - min_col) * tile_size_display
            y_pos = (global_row - min_row) * tile_size_display
            
            # Add thumbnail to plot
            extent = [x_pos, x_pos + tile_size_display, y_pos, y_pos + tile_size_display]
            ax.imshow(thumb_rgb, extent=extent, aspect='auto', origin='upper')
            
            # Add tile ID label only if tiles are large enough to read
            if tile_size_display > 20:
                ax.text(x_pos + tile_size_display/2, y_pos + tile_size_display/2, 
                       str(tile_id), ha='center', va='center', 
                       fontsize=max(4, int(tile_size_display/10)), color='white', weight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            displayed_tiles += 1
        
        # Set axis
        ax.set_xlim(0, grid_width * tile_size_display)
        ax.set_ylim(grid_height * tile_size_display, 0)  # Flip Y axis
        ax.set_aspect('equal')
        
        # Calculate sparsity info
        total_positions = grid_height * grid_width
        empty_positions = total_positions - len(self.tile_global_positions)
        sparsity = (empty_positions / total_positions * 100) if total_positions > 0 else 0
        
        # Labels
        ax.set_xlabel(f'Column Position (0 to {grid_width-1})', fontsize=10)
        ax.set_ylabel(f'Row Position (0 to {grid_height-1})', fontsize=10)
        
        if fixed_grid_size:
            title = f'Global Map: {len(self.tile_global_positions)} tiles on {grid_width}x{grid_height} canvas\n'
            title += f'From {len(self.frames)} frames | Occupancy: {(len(self.tile_global_positions)/total_positions*100):.1f}%'
        else:
            title = f'Global Map: {len(self.tile_global_positions)} tiles in {grid_height}x{grid_width} bounding box\n'
            title += f'From {len(self.frames)} frames | Sparsity: {sparsity:.1f}%'
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        ax.grid(True, alpha=0.2, linewidth=0.5)
        plt.tight_layout()
        
        if save_path:
            # Save to file
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
            plt.close(fig)
        else:
            # Display
            print(f"✓ Displaying {displayed_tiles} tiles")
            plt.show()
    
    def save_map_data(self, output_dir='export'):
        """Save map data to files"""
        print(f"\nSaving map data to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save map structure
        map_data = {
            'tile_global_positions': {str(k): v for k, v in self.tile_global_positions.items()},
            'num_tiles': len(self.tiles),
            'num_frames': len(self.frames),
        }
        
        with open(os.path.join(output_dir, 'global_map.json'), 'w') as f:
            json.dump(map_data, f, indent=2)
        
        print(f"✓ Saved map data to {output_dir}/global_map.json")

def main():
    parser = argparse.ArgumentParser(description='Build global map from video using frame-by-frame processing')
    parser.add_argument('--video', type=str, default='video.mp4', help='Video file path')
    parser.add_argument('--num-frames', type=int, default=None, help='Number of frames to process (default: all)')
    parser.add_argument('--blocks', type=int, default=16, help='Blocks per side for dhash')
    parser.add_argument('--clahe', action='store_true', default=True, help='Apply CLAHE preprocessing')
    parser.add_argument('--no-clahe', dest='clahe', action='store_false', help='Disable CLAHE')
    parser.add_argument('--tile-max-std', type=float, default=None, help='Maximum std dev for tiles')
    parser.add_argument('--clahe-clip', type=float, default=2.0, help='CLAHE clip limit')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    parser.add_argument('--save-viz', type=str, default=None, help='Save visualization to PNG file instead of displaying')
    parser.add_argument('--max-display', type=int, default=2000, help='Max display size in pixels (default: 2000)')
    
    args = parser.parse_args()
    
    try:
        builder = GlobalMapBuilder(
            args.video,
            blocks=args.blocks,
            use_clahe=args.clahe,
            tile_max_std=args.tile_max_std,
            clahe_clip_limit=args.clahe_clip,
            num_frames=args.num_frames
        )
        
        # Run pipeline
        start_time = time.time()
        builder.process_video()
        builder.build_global_map()
        
        if not args.no_viz:
            if args.save_viz:
                builder.visualize_global_map(max_display_size=args.max_display, save_path=args.save_viz)
            else:
                builder.visualize_global_map(max_display_size=args.max_display)
        
        builder.save_map_data()
        
        elapsed = time.time() - start_time
        print(f"\n✓ Pipeline completed in {elapsed:.1f} seconds")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
