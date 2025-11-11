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
    
    # Cache for previous frame
    prev_frame_data = {}
    
    def extract_frame_data(frame_num):
        """Extract grid and tile data from a frame"""
        if frame_num < 0 or frame_num >= total_frames:
            return None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            return None
        
        # Detect grid
        line_detector = LineDetector()
        frame_with_lines, lines = line_detector.detect_lines(frame)
        grid_map = extract_grid_squares(lines, frame.shape[:2])
        
        if not grid_map:
            return None
        
        # Extract tiles and signatures
        tiles = {}
        signatures = {}
        filtered = set()
        centers = {}  # Store tile centers
        
        for (row, col), square_polygon in grid_map.items():
            try:
                warped = extract_and_warp_square(frame, square_polygon)
                
                # Filter by std dev
                gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                std_dev = float(np.std(gray_warped))
                if std_dev > 20:
                    filtered.add((row, col))
                    continue
                
                tiles[(row, col)] = warped
                
                # Calculate tile center in original frame
                p1, p2, p3, p4 = square_polygon
                cx = int((p1[0] + p3[0]) / 2)
                cy = int((p1[1] + p3[1]) / 2)
                centers[(row, col)] = (cx, cy)
                
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
                
                signatures[(row, col)] = sig_dict
            except Exception as e:
                pass
        
        return {
            'frame': frame,
            'grid_map': grid_map,
            'tiles': tiles,
            'signatures': signatures,
            'filtered': filtered,
            'centers': centers
        }
    
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
        print(f"Frame {frame_num}: {len(curr_data['grid_map'])} detected, {kept_tiles} kept, {filtered_tiles} filtered", end="")
        
        # Clear axes
        for ax in fig.get_axes():
            ax.remove()
        
        ax = fig.add_subplot(1, 1, 1)
        
        if frame_num > 0 and prev_frame_data and 'frame' in prev_frame_data:
            prev_data = prev_frame_data
            
            # Create side-by-side image
            h, w = curr_data['frame'].shape[:2]
            h_prev, w_prev = prev_data['frame'].shape[:2]
            
            # Pad to same height if needed
            max_h = max(h, h_prev)
            
            prev_frame_padded = np.zeros((max_h, w_prev, 3), dtype=np.uint8)
            prev_frame_padded[:h_prev] = prev_data['frame']
            
            curr_frame_padded = np.zeros((max_h, w, 3), dtype=np.uint8)
            curr_frame_padded[:h] = curr_data['frame']
            
            # Concatenate horizontally
            combined = np.concatenate([prev_frame_padded, curr_frame_padded], axis=1)
            
            # Draw tile centers on previous frame
            for (row, col), (cx, cy) in prev_data['centers'].items():
                cv2.circle(combined, (cx, cy), 5, (0, 255, 255), -1)
            
            # Draw tile centers on current frame and connection lines
            matches = 0
            for curr_coord, curr_sigs in curr_data['signatures'].items():
                curr_cx, curr_cy = curr_data['centers'][curr_coord]
                
                # Find best match in previous frame
                prev_coord, rotations, distance = find_best_match(curr_coord, curr_sigs, prev_data['signatures'])
                
                if prev_coord is not None:
                    matches += 1
                    prev_cx, prev_cy = prev_data['centers'][prev_coord]
                    
                    # Draw current tile center
                    cv2.circle(combined, (w_prev + curr_cx, curr_cy), 5, (0, 255, 0), -1)
                    
                    # Draw line from prev to curr
                    cv2.line(combined, (prev_cx, prev_cy), (w_prev + curr_cx, curr_cy), (0, 255, 255), 2)
                    
                    # Add rotation info near the line midpoint
                    if rotations:
                        curr_rot, prev_rot = rotations
                        mid_x = (prev_cx + w_prev + curr_cx) // 2
                        mid_y = (prev_cy + curr_cy) // 2
                        rot_text = f"{curr_rot}°→{prev_rot}°"
                        cv2.putText(combined, rot_text, (mid_x - 20, mid_y - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            ax.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
            ax.set_title(f'Frame {frame_num-1} (Prev) → Frame {frame_num} (Curr) | {matches} matches')
            print(f" | {matches}/{len(curr_data['signatures'])} tiles matched")
        else:
            ax.imshow(cv2.cvtColor(curr_data['frame'], cv2.COLOR_BGR2RGB))
            ax.set_title(f'Frame {frame_num} (No previous frame)')
            print()
        
        ax.axis('off')
        
        # Store current data as previous for next frame
        prev_frame_data.clear()
        prev_frame_data.update(curr_data)
        
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
    
    # Create figure
    fig = plt.figure(figsize=(16, 7))
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

