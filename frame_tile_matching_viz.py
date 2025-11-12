import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from line_detector import LineDetector
from auto_tile_detector import extract_grid_squares, extract_and_warp_square, generate_tile_hash, euclidean_distance


def rotate_coordinates(row, col, height, width, rotation_offset):
    """Rotate (row, col) within a grid of size height x width by rotation_offset degrees."""
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


def analyze_frame_pair(prev_data, curr_data):
    """
    Analyze matching between two consecutive frames.
    Returns dictionary with accepted matches, best match, scaled centers, and scale factors.
    """
    h_prev, w_prev = prev_data['frame'].shape[:2]
    h_curr, w_curr = curr_data['frame'].shape[:2]

    scale_x = w_prev / w_curr if w_prev != w_curr else 1.0
    scale_y = h_prev / h_curr if h_prev != h_curr else 1.0

    distance_matrix = {}
    for curr_coord, curr_sigs in curr_data['signatures'].items():
        for prev_coord, prev_sigs in prev_data['signatures'].items():
            best_dist = float('inf')
            best_rots = None

            for curr_rot in [0, 90, 180, 270]:
                if curr_rot not in curr_sigs:
                    continue
                for prev_rot in [0, 90, 180, 270]:
                    if prev_rot not in prev_sigs:
                        continue
                    dist = euclidean_distance(curr_sigs[curr_rot], prev_sigs[prev_rot])
                    if dist < best_dist:
                        best_dist = dist
                        best_rots = (curr_rot, prev_rot)

            if best_rots:
                distance_matrix[(curr_coord, prev_coord)] = (best_dist, best_rots)

    sorted_pairs = sorted(distance_matrix.items(), key=lambda x: x[1][0])
    matched_prev = set()
    matched_curr = set()
    matches_list = []

    for (curr_coord, prev_coord), (dist, rotations) in sorted_pairs:
        if curr_coord in matched_curr or prev_coord in matched_prev:
            continue
        matches_list.append((curr_coord, prev_coord, rotations, dist))
        matched_curr.add(curr_coord)
        matched_prev.add(prev_coord)

    matched_details = []
    pixel_distances = []
    curr_centers = curr_data['centers']
    prev_centers = prev_data['centers']

    for curr_coord, prev_coord, rotations, distance in matches_list:
        if prev_coord not in prev_centers or curr_coord not in curr_centers:
            continue
        prev_cx, prev_cy = prev_centers[prev_coord]
        curr_cx_orig, curr_cy_orig = curr_centers[curr_coord]
        curr_cx = int(curr_cx_orig * scale_x)
        curr_cy = int(curr_cy_orig * scale_y)
        pixel_dist = np.sqrt((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2)
        pixel_distances.append(pixel_dist)
        matched_details.append({
            'curr_coord': curr_coord,
            'prev_coord': prev_coord,
            'rotations': rotations,
            'distance': distance,
            'prev_center': (prev_cx, prev_cy),
            'curr_center': (curr_cx, curr_cy),
            'pixel_distance': pixel_dist
        })

    if pixel_distances:
        avg_distance = np.mean(pixel_distances)
        std_distance = np.std(pixel_distances)
        threshold = avg_distance + 1 * std_distance
        accepted_matches = [m for m in matched_details if m['pixel_distance'] <= threshold]
    else:
        accepted_matches = matched_details

    best_match = min(accepted_matches, key=lambda m: m['distance']) if accepted_matches else None

    scaled_curr_centers = {}
    for coord, (cx, cy) in curr_centers.items():
        scaled_curr_centers[coord] = (int(cx * scale_x), int(cy * scale_y))

    return {
        'accepted_matches': accepted_matches,
        'best_match': best_match,
        'scaled_curr_centers': scaled_curr_centers,
        'scale_x': scale_x,
        'scale_y': scale_y
    }

def frame_tile_matching_viz(video_path, frame_number=0):
    """
    Visualize tile matching between consecutive frames with connection lines.
    
    Shows both frames side-by-side with lines connecting tiles that best match
    across all 4 rotations.
    
    Navigation: RIGHT arrow = next frame, LEFT arrow = previous frame, Q = quit
    """
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # State for navigation
    state = {'current_frame': frame_number, 'cap': cap, 'total_frames': total_frames, 'fig': None}
    
    # Frame cache - stores processed frame objects
    frames = {}  # frame_number -> frame_obj
    
    def extract_frame_data(frame_num):
        """Extract grid and tile data from a frame, return frame object with new structure"""
        # Check cache first
        if frame_num in frames:
            return frames[frame_num]
        
        if frame_num < 0 or frame_num >= total_frames:
            return None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            return None
        
        # Detect grid (suppress output)
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        line_detector = LineDetector()
        frame_with_lines, lines = line_detector.detect_lines(frame)
        sys.stdout = old_stdout
        grid_map = extract_grid_squares(lines, frame.shape[:2])
        
        if not grid_map:
            return None
        
        grid_rows = [coord[0] for coord in grid_map.keys()]
        grid_cols = [coord[1] for coord in grid_map.keys()]
        num_rows = (max(grid_rows) + 1) if grid_rows else 0
        num_cols = (max(grid_cols) + 1) if grid_cols else 0
        
        # Extract tiles - new structure: list of tile objects
        tile_list = []
        
        # Also keep dictionaries for matching logic (keyed by coordinates)
        signatures_by_coord = {}
        tiles_by_coord = {}
        centers_by_coord = {}
        filtered = set()
        
        for (row, col), square_polygon in grid_map.items():
            try:
                warped = extract_and_warp_square(frame, square_polygon)
                
                # Filter by std dev
                gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                std_dev = float(np.std(gray_warped))
                if std_dev > 20:
                    filtered.add((row, col))
                    continue
                
                # Calculate tile center in original frame
                p1, p2, p3, p4 = square_polygon
                cx = int((p1[0] + p3[0]) / 2)
                cy = int((p1[1] + p3[1]) / 2)
                centers_by_coord[(row, col)] = (cx, cy)
                
                # Generate signatures for all rotations
                sig_dict = {}
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
                        sig_dict[rotation] = hash_sig
                
                # Create tile object (new structure)
                tile_obj = {
                    'signatures': sig_dict,
                    'image': warped,
                    'frame_row': row,
                    'frame_col': col
                }
                tile_list.append(tile_obj)
                
                # Keep coord-based lookups for matching logic
                signatures_by_coord[(row, col)] = sig_dict
                tiles_by_coord[(row, col)] = warped
                
            except Exception as e:
                pass
        
        # Create frame object (new structure)
        frame_obj = {
            'frame_number': frame_num,
            'tiles': tile_list,
            'frame': frame,
            'grid_map': grid_map,
            'num_rows': num_rows,
            'num_cols': num_cols,
            # Keep these for backward compatibility with matching logic
            'signatures': signatures_by_coord,
            'tiles_by_coord': tiles_by_coord,
            'centers': centers_by_coord,
            'filtered': filtered
        }
        
        # Cache it
        frames[frame_num] = frame_obj
        return frame_obj
    
    def build_global_map_tiles(target_frame_num):
        """Build global tile placements up to the target frame using best anchors."""
        global_positions = {}
        global_tiles = []
        
        for idx in range(target_frame_num + 1):
            frame_data = extract_frame_data(idx)
            if not frame_data:
                continue
            
            if idx == 0 or not global_positions:
                # Anchor first available frame at origin using local coordinates
                for tile_obj in frame_data['tiles']:
                    key = (idx, tile_obj['frame_row'], tile_obj['frame_col'])
                    pos = (tile_obj['frame_row'], tile_obj['frame_col'])
                    global_positions[key] = pos
                    global_tiles.append({
                        'frame_index': idx,
                        'frame_row': tile_obj['frame_row'],
                        'frame_col': tile_obj['frame_col'],
                        'global_row': pos[0],
                        'global_col': pos[1],
                        'image': tile_obj['image']
                    })
                continue
            
            prev_data = extract_frame_data(idx - 1)
            if not prev_data:
                continue
            
            analysis = analyze_frame_pair(prev_data, frame_data)
            best_match = analysis['best_match']
            if not best_match:
                continue
            
            rotations = best_match['rotations']
            rotation_offset = ((rotations[1] - rotations[0]) % 360) if rotations else 0
            prev_row, prev_col = best_match['prev_coord']
            prev_key = (idx - 1, prev_row, prev_col)
            if prev_key not in global_positions:
                continue
            
            prev_global_pos = global_positions[prev_key]
            curr_row, curr_col = best_match['curr_coord']
            rotated_curr_row, rotated_curr_col = rotate_coordinates(
                curr_row, curr_col,
                frame_data['num_rows'], frame_data['num_cols'],
                rotation_offset
            )
            offset_row = prev_global_pos[0] - rotated_curr_row
            offset_col = prev_global_pos[1] - rotated_curr_col
            
            for tile_obj in frame_data['tiles']:
                local_row = tile_obj['frame_row']
                local_col = tile_obj['frame_col']
                rotated_row, rotated_col = rotate_coordinates(
                    local_row, local_col,
                    frame_data['num_rows'], frame_data['num_cols'],
                    rotation_offset
                )
                global_row = rotated_row + offset_row
                global_col = rotated_col + offset_col
                key = (idx, local_row, local_col)
                if key in global_positions:
                    continue
                
                global_positions[key] = (global_row, global_col)
                global_tiles.append({
                    'frame_index': idx,
                    'frame_row': local_row,
                    'frame_col': local_col,
                    'global_row': global_row,
                    'global_col': global_col,
                    'image': tile_obj['image']
                })
        
        return global_tiles
    
    def update_visualization(fig, frame_num):
        """Update figure with frames and matching lines"""
        curr_data = extract_frame_data(frame_num)
        if not curr_data:
            print(f"Frame {frame_num}: Could not extract data")
            return False
        
        # Clear axes
        for ax in fig.get_axes():
            ax.remove()
        
        ax_main = fig.add_subplot(1, 2, 1)
        ax_grid = fig.add_subplot(1, 2, 2)
        
        if frame_num > 0 and frame_num - 1 in frames:
            prev_data = frames[frame_num - 1]
            
            # Overlay frames - assume same size or resize
            h_prev, w_prev = prev_data['frame'].shape[:2]
            h_curr, w_curr = curr_data['frame'].shape[:2]
            
            # Resize current frame to match previous frame size
            if (h_prev, w_prev) != (h_curr, w_curr):
                curr_resized = cv2.resize(curr_data['frame'], (w_prev, h_prev))
            else:
                curr_resized = curr_data['frame']
            
            # Blend frames: 50% previous + 50% current
            combined = cv2.addWeighted(prev_data['frame'], 0.5, curr_resized, 0.5, 0)
            
            analysis = analyze_frame_pair(prev_data, curr_data)
            scaled_curr_centers = analysis['scaled_curr_centers']
            best_match = analysis['best_match']
            
            # Draw centers for all detected tiles
            for (prev_row, prev_col), (prev_cx, prev_cy) in prev_data['centers'].items():
                cv2.circle(combined, (prev_cx, prev_cy), 5, (0, 255, 255), -1)
            
            for (curr_row, curr_col), (curr_cx, curr_cy) in scaled_curr_centers.items():
                cv2.circle(combined, (curr_cx, curr_cy), 5, (0, 255, 0), -1)
            
            if best_match:
                prev_cx, prev_cy = best_match['prev_center']
                curr_cx, curr_cy = best_match['curr_center']
                cv2.line(combined, (prev_cx, prev_cy), (curr_cx, curr_cy), (255, 0, 0), 2)
            
            # Convert to PIL for text rendering
            from PIL import Image, ImageDraw, ImageFont
            pil_combined = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_combined)
            
            # Load font
            try:
                font_large = ImageFont.truetype("arial.ttf", 36)
                font_small = ImageFont.truetype("arial.ttf", 24)
            except:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
                font_coords = ImageFont.load_default()
            else:
                font_coords = font_small
            
            # Annotate coordinates for previous frame tiles
            for (prev_row, prev_col), (prev_cx, prev_cy) in prev_data['centers'].items():
                prev_label = f"({prev_row},{prev_col})"
                draw.text((prev_cx - 40, prev_cy - 45), prev_label, fill=(255, 255, 0), font=font_coords)
            
            # Annotate coordinates for current frame tiles
            for (curr_row, curr_col), (curr_cx, curr_cy) in scaled_curr_centers.items():
                curr_label = f"({curr_row},{curr_col})"
                draw.text((curr_cx - 40, curr_cy + 10), curr_label, fill=(0, 255, 0), font=font_coords)
            
            # Add rotation info near the line midpoint for best match
            if best_match and best_match['rotations']:
                curr_rot, prev_rot = best_match['rotations']
                prev_cx, prev_cy = best_match['prev_center']
                curr_cx, curr_cy = best_match['curr_center']
                mid_x = (prev_cx + curr_cx) // 2
                mid_y = (prev_cy + curr_cy) // 2
                rot_text = f"{curr_rot}°→{prev_rot}°"
                draw.text((mid_x - 20, mid_y - 5), rot_text, fill=(255, 255, 0), font=font_small)
            
            # Convert back to numpy array
            combined = cv2.cvtColor(np.array(pil_combined), cv2.COLOR_RGB2BGR)
            
            matches = 1 if best_match else 0
            
            # Print only best match details
            if best_match:
                curr_row, curr_col = best_match['curr_coord']
                prev_row, prev_col = best_match['prev_coord']
                distance = best_match['distance']
                print(f"Frame {frame_num}: best match ({curr_row},{curr_col}) → ({prev_row},{prev_col}) | dist={distance:.0f}")
            
            ax_main.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
            ax_main.set_title(f'Frame {frame_num-1} (Prev) → Frame {frame_num} (Curr) | {matches} matches')
            
        else:
            ax_main.imshow(cv2.cvtColor(curr_data['frame'], cv2.COLOR_BGR2RGB))
            ax_main.set_title(f'Frame {frame_num} (No previous frame - Anchor)')
            
            # For first frame, ensure it is cached for subsequent steps
            frames[frame_num] = curr_data
        
        # Build and render global map tiles up to the current frame
        global_tiles = build_global_map_tiles(frame_num)
        if global_tiles:
            tile_display_size = 30
            padding = 2
            global_rows = [tile['global_row'] for tile in global_tiles]
            global_cols = [tile['global_col'] for tile in global_tiles]
            min_row = min(global_rows)
            max_row = max(global_rows)
            min_col = min(global_cols)
            max_col = max(global_cols)
            
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
            except:
                tile_font = ImageFont.load_default()
            
            for tile in global_tiles:
                grid_row = (tile['global_row'] - min_row) + padding
                grid_col = (tile['global_col'] - min_col) + padding
                
                if grid_row < 0 or grid_col < 0:
                    continue
                
                y_start = grid_row * tile_display_size
                x_start = grid_col * tile_display_size
                
                warped = tile['image']
                warped_resized = cv2.resize(warped, (tile_display_size, tile_display_size))
                
                pil_tile = Image.fromarray(cv2.cvtColor(warped_resized, cv2.COLOR_BGR2RGB))
                draw_tile = ImageDraw.Draw(pil_tile)
                label = f"G({tile['global_row']},{tile['global_col']})"
                draw_tile.text((2, 2), label, fill=(255, 0, 0), font=tile_font)
                warped_resized = cv2.cvtColor(np.array(pil_tile), cv2.COLOR_RGB2BGR)
                
                composite_image[y_start:y_start + tile_display_size, x_start:x_start + tile_display_size] = warped_resized
            
            for i in range(0, grid_rows + 1):
                y = min(i * tile_display_size, composite_height - 1)
                thickness = 2 if i % 5 == 0 else 1
                cv2.line(composite_image, (0, y), (composite_width, y), (80, 80, 80), thickness)
            for j in range(0, grid_cols + 1):
                x = min(j * tile_display_size, composite_width - 1)
                thickness = 2 if j % 5 == 0 else 1
                cv2.line(composite_image, (x, 0), (x, composite_height), (80, 80, 80), thickness)
            
            ax_grid.imshow(cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB))
            ax_grid.set_title(f'Global Map (frames ≤ {frame_num}) | {len(global_tiles)} tiles')
        else:
            ax_grid.text(0.5, 0.5, 'No tiles placed yet', ha='center', va='center', transform=ax_grid.transAxes)
            ax_grid.set_title('Global Map')
        
        ax_grid.axis('off')
        ax_main.axis('off')
        
        fig.canvas.draw_idle()
        return True
    
    def on_key(event):
        """Handle keyboard events"""
        if event.key == 'right':
            next_frame = state['current_frame'] + 1
            if next_frame < state['total_frames']:
                state['current_frame'] = next_frame
                print(f"→ ", end="")
                update_visualization(state['fig'], next_frame)
        elif event.key == 'left':
            prev_frame = state['current_frame'] - 1
            if prev_frame >= 0:
                state['current_frame'] = prev_frame
                print(f"← ", end="")
                update_visualization(state['fig'], prev_frame)
        elif event.key == 'q':
            print("Closing...")
            plt.close('all')
    
    # Initial setup
    print(f"Loading video: {video_path}")
    print(f"Starting at frame {frame_number}/{total_frames-1}")
    print("Use RIGHT arrow to go to next frame, LEFT arrow for previous, Q to quit\n")
    
    # Create figure (wider to accommodate grid on the right)
    fig = plt.figure(figsize=(20, 7))
    state['fig'] = fig
    
    # Initial update
    if update_visualization(fig, frame_number):
        fig.canvas.mpl_connect('key_press_event', on_key)
        manager = fig.canvas.manager
        manager.window.showMaximized()
        plt.show()
    
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize tile matching between consecutive frames.')
    parser.add_argument('--video', type=str, default='video.mp4', help='Video file path')
    parser.add_argument('--frame', type=int, default=1, help='Frame number to start (default: 1)')
    
    args = parser.parse_args()
    
    frame_tile_matching_viz(args.video, args.frame)

