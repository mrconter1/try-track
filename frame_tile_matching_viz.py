import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from line_detector import LineDetector
from auto_tile_detector import extract_grid_squares, extract_and_warp_square, generate_tile_hash, euclidean_distance

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
            # Keep these for backward compatibility with matching logic
            'signatures': signatures_by_coord,
            'tiles_by_coord': tiles_by_coord,
            'centers': centers_by_coord,
            'filtered': filtered
        }
        
        # Cache it
        frames[frame_num] = frame_obj
        return frame_obj
    
    def find_best_match(curr_coord, curr_sigs, prev_signatures):
        """Find best matching tile in previous frame across all rotations"""
        best_dist = float('inf')
        best_prev_coord = None
        best_rotations = None
        
        for prev_coord, prev_sigs in prev_signatures.items():
            # Compare across all rotation combinations
            for curr_rot in [0, 90, 180, 270]:
                for prev_rot in [0, 90, 180, 270]:
                    if curr_rot in curr_sigs and prev_rot in prev_sigs:
                        dist = euclidean_distance(curr_sigs[curr_rot], prev_sigs[prev_rot])
                        if dist < best_dist:
                            best_dist = dist
                            best_prev_coord = prev_coord
                            best_rotations = (curr_rot, prev_rot)
        
        return best_prev_coord, best_rotations, best_dist
    
    def update_visualization(fig, frame_num):
        """Update figure with frames and matching lines"""
        curr_data = extract_frame_data(frame_num)
        if not curr_data:
            print(f"Frame {frame_num}: Could not extract data")
            return False
        
        kept_tiles = len(curr_data['tiles'])
        filtered_tiles = len(curr_data['filtered'])
        
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
            
            # Adjust current tile centers to match the resized dimensions
            scale_x = w_prev / w_curr if w_prev != w_curr else 1.0
            scale_y = h_prev / h_curr if h_prev != h_curr else 1.0
            
            # Find all pairwise distances and create one-to-one matching
            # Calculate distances for all current-prev tile pairs
            distance_matrix = {}
            for curr_coord, curr_sigs in curr_data['signatures'].items():
                for prev_coord, prev_sigs in prev_data['signatures'].items():
                    best_dist = float('inf')
                    best_rots = None
                    
                    for curr_rot in [0, 90, 180, 270]:
                        for prev_rot in [0, 90, 180, 270]:
                            if curr_rot in curr_sigs and prev_rot in prev_sigs:
                                dist = euclidean_distance(curr_sigs[curr_rot], prev_sigs[prev_rot])
                                if dist < best_dist:
                                    best_dist = dist
                                    best_rots = (curr_rot, prev_rot)
                    
                    if best_rots:
                        distance_matrix[(curr_coord, prev_coord)] = (best_dist, best_rots)
            
            # Greedy one-to-one matching: match lowest distances first
            sorted_pairs = sorted(distance_matrix.items(), key=lambda x: x[1][0])
            matched_prev = set()
            matched_curr = set()
            matches_list = []
            
            for (curr_coord, prev_coord), (dist, rotations) in sorted_pairs:
                # Only match if neither has been matched yet
                if curr_coord not in matched_curr and prev_coord not in matched_prev:
                    matches_list.append((curr_coord, prev_coord, rotations, dist))
                    matched_curr.add(curr_coord)
                    matched_prev.add(prev_coord)
            
            # Calculate physical pixel distances for all matches
            pixel_distances = []
            match_details = []  # Store full details for filtering
            
            for curr_coord, prev_coord, rotations, distance in matches_list:
                # Get previous tile center
                prev_cx, prev_cy = prev_data['centers'][prev_coord]
                
                # Get current tile center and scale if needed
                curr_cx_orig, curr_cy_orig = curr_data['centers'][curr_coord]
                curr_cx = int(curr_cx_orig * scale_x)
                curr_cy = int(curr_cy_orig * scale_y)
                
                # Calculate physical pixel distance
                pixel_dist = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
                pixel_distances.append(pixel_dist)
                match_details.append((curr_coord, prev_coord, rotations, distance, prev_cx, prev_cy, curr_cx, curr_cy, pixel_dist))
            
            # Calculate average and standard deviation
            if pixel_distances:
                avg_distance = np.mean(pixel_distances)
                std_distance = np.std(pixel_distances)
                threshold = avg_distance + 1 * std_distance  # Stricter outlier threshold (1 std dev)
                
                # Filter out outliers
                filtered_matches = [m for m in match_details if m[8] <= threshold]
            else:
                filtered_matches = match_details
            
            # Store only matched tiles in the frame cache
            matched_tile_list = []
            for curr_coord, prev_coord, _, _, _, _, _, _, _ in filtered_matches:
                row, col = curr_coord
                for tile_obj in curr_data['tiles']:
                    if tile_obj['frame_row'] == row and tile_obj['frame_col'] == col:
                        matched_tile_list.append(tile_obj)
                        break
            
            # Update cached frame object to contain only matched tiles
            curr_data['tiles'] = matched_tile_list
            frames[frame_num] = curr_data
            
            # Draw filtered matched pairs
            for curr_coord, prev_coord, rotations, distance, prev_cx, prev_cy, curr_cx, curr_cy, pixel_dist in filtered_matches:
                # Draw previous tile center (yellow)
                cv2.circle(combined, (prev_cx, prev_cy), 5, (0, 255, 255), -1)
                
                # Draw current tile center (green)
                cv2.circle(combined, (curr_cx, curr_cy), 5, (0, 255, 0), -1)
                
                # Draw line from prev to curr
                cv2.line(combined, (prev_cx, prev_cy), (curr_cx, curr_cy), (0, 255, 255), 2)
                
                # Add grid coordinates for previous frame tile (yellow text)
                prev_row, prev_col = prev_coord
                prev_label = f"({prev_row},{prev_col})"
                cv2.putText(combined, prev_label, (prev_cx - 25, prev_cy - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Add grid coordinates for current frame tile (green text)
                curr_row, curr_col = curr_coord
                curr_label = f"({curr_row},{curr_col})"
                cv2.putText(combined, curr_label, (curr_cx - 25, curr_cy - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Add rotation info near the line midpoint
                if rotations:
                    curr_rot, prev_rot = rotations
                    mid_x = (prev_cx + curr_cx) // 2
                    mid_y = (prev_cy + curr_cy) // 2
                    rot_text = f"{curr_rot}°→{prev_rot}°"
                    cv2.putText(combined, rot_text, (mid_x - 20, mid_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            matches = len(filtered_matches)
            
            # Print only matched tiles
            if matches > 0:
                matched_coords = [coord for coord, _, _, _, _, _, _, _, _ in filtered_matches]
                coord_str = ', '.join([f"({r},{c})" for r, c in matched_coords])
                print(f"Frame {frame_num}: {matches} matched tiles - {coord_str}")
            
            ax_main.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
            ax_main.set_title(f'Frame {frame_num-1} (Prev) → Frame {frame_num} (Curr) | {matches} matches')
            
            # Build grid visualization on 100x100 canvas (-50,-50 to 50,50)
            if filtered_matches or curr_data['tiles']:
                # Create 100x100 grid centered at origin
                tile_display_size = 30
                grid_size = 100
                composite_height = grid_size * tile_display_size
                composite_width = grid_size * tile_display_size
                composite_image = np.full((composite_height, composite_width, 3), 40, dtype=np.uint8)
                
                # Draw center crosshair at (0,0)
                center_pos = 50 * tile_display_size
                cv2.line(composite_image, (center_pos, center_pos - 10), (center_pos, center_pos + 10), (255, 0, 0), 2)
                cv2.line(composite_image, (center_pos - 10, center_pos), (center_pos + 10, center_pos), (255, 0, 0), 2)
                
                # Place tiles on grid
                from PIL import Image, ImageDraw, ImageFont
                for tile_obj in curr_data['tiles']:
                    row = tile_obj['frame_row']
                    col = tile_obj['frame_col']
                    
                    # Convert frame coordinates to grid coordinates (center at 0,0)
                    # Grid goes from -50,-50 to 50,50 (or 49,49)
                    grid_row = row + 50
                    grid_col = col + 50
                    
                    # Check if within bounds
                    if 0 <= grid_row < grid_size and 0 <= grid_col < grid_size:
                        warped = tile_obj['image']
                        warped_resized = cv2.resize(warped, (tile_display_size, tile_display_size))
                        
                        # Convert to RGB for PIL drawing
                        pil_image = Image.fromarray(cv2.cvtColor(warped_resized, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_image)
                        
                        # Add grid coordinates
                        try:
                            font = ImageFont.truetype("arial.ttf", 10)
                        except:
                            font = ImageFont.load_default()
                        
                        label = f"({row},{col})"
                        draw.text((2, 2), label, fill=(255, 0, 0), font=font)
                        
                        # Convert back to numpy array
                        warped_resized = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                        
                        y_start = grid_row * tile_display_size
                        x_start = grid_col * tile_display_size
                        composite_image[y_start:y_start+tile_display_size, x_start:x_start+tile_display_size] = warped_resized
                
                ax_grid.imshow(composite_image)
                ax_grid.set_title(f'100x100 Grid (-50,-50 to 50,50) | Center at (0,0)')
                
                # Add grid lines every 10 units
                for i in range(0, grid_size + 1, 10):
                    ax_grid.axhline(y=i * tile_display_size, color='gray', linewidth=0.5, alpha=0.5)
                    ax_grid.axvline(x=i * tile_display_size, color='gray', linewidth=0.5, alpha=0.5)
            else:
                ax_grid.text(0.5, 0.5, 'No tiles', ha='center', va='center', transform=ax_grid.transAxes)
                ax_grid.set_title('100x100 Grid')
            
            ax_grid.axis('off')
        else:
            ax_main.imshow(cv2.cvtColor(curr_data['frame'], cv2.COLOR_BGR2RGB))
            ax_main.set_title(f'Frame {frame_num} (No previous frame - Anchor)')
            
            # Still show 100x100 grid for first frame
            tile_display_size = 30
            grid_size = 100
            composite_height = grid_size * tile_display_size
            composite_width = grid_size * tile_display_size
            composite_image = np.full((composite_height, composite_width, 3), 40, dtype=np.uint8)
            
            # Draw center crosshair
            center_pos = 50 * tile_display_size
            cv2.line(composite_image, (center_pos, center_pos - 10), (center_pos, center_pos + 10), (255, 0, 0), 2)
            cv2.line(composite_image, (center_pos - 10, center_pos), (center_pos + 10, center_pos), (255, 0, 0), 2)
            
            # For first frame, just cache all tiles without filtering
            frames[frame_num] = curr_data
            
            ax_grid.imshow(composite_image)
            ax_grid.set_title(f'100x100 Grid (-50,-50 to 50,50) | Center at (0,0)')
            
            # Add grid lines every 10 units
            for i in range(0, grid_size + 1, 10):
                ax_grid.axhline(y=i * tile_display_size, color='gray', linewidth=0.5, alpha=0.5)
                ax_grid.axvline(x=i * tile_display_size, color='gray', linewidth=0.5, alpha=0.5)
            
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

