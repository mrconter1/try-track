import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
from line_detector import LineDetector
from auto_tile_detector import extract_grid_squares, extract_and_warp_square, generate_tile_hash, euclidean_distance

def frame_similarity_viz(video_path, frame_number=0):
    """
    Visualize frame with grid overlay and tile similarity matrix between consecutive frames.
    
    Left panel: Frame with grid lines and tile coordinates
    Right panel: Similarity matrix heatmap (current frame tiles vs previous frame tiles)
    
    Navigation: RIGHT arrow = next frame, LEFT arrow = previous frame, Q = quit
    """
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # State for navigation
    state = {'current_frame': frame_number, 'cap': cap, 'total_frames': total_frames, 'fig': None}
    
    # Cache for tiles and signatures from previous frame
    prev_frame_data = {'tiles': {}, 'signatures': {}, 'filtered': set()}
    
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
            'frame_with_lines': frame_with_lines,
            'grid_map': grid_map,
            'tiles': tiles,
            'signatures': signatures,
            'filtered': filtered,
            'lines': lines
        }
    
    def compute_similarity_matrix(curr_data, prev_data):
        """Compute similarity matrix between current and previous frame tiles"""
        if not curr_data or not prev_data:
            return None, None, None, None
        
        curr_tiles = list(curr_data['signatures'].keys())
        prev_tiles = list(prev_data['signatures'].keys())
        
        if not curr_tiles or not prev_tiles:
            return None, None, None, None
        
        # Build matrix
        matrix = np.zeros((len(curr_tiles), len(prev_tiles)))
        best_rotations = {}  # Store best rotation for each pair
        
        for i, curr_coord in enumerate(curr_tiles):
            curr_sigs = curr_data['signatures'][curr_coord]
            for j, prev_coord in enumerate(prev_tiles):
                prev_sigs = prev_data['signatures'][prev_coord]
                
                # Find best match across all rotations
                best_dist = float('inf')
                best_rot = None
                
                for curr_rot in [0, 90, 180, 270]:
                    for prev_rot in [0, 90, 180, 270]:
                        if curr_rot in curr_sigs and prev_rot in prev_sigs:
                            dist = euclidean_distance(curr_sigs[curr_rot], prev_sigs[prev_rot])
                            if dist < best_dist:
                                best_dist = dist
                                best_rot = (curr_rot, prev_rot)
                
                matrix[i, j] = best_dist
                best_rotations[(i, j)] = best_rot
        
        return matrix, curr_tiles, prev_tiles, best_rotations
    
    def update_visualization(fig, frame_num):
        """Update figure with frame and similarity matrix"""
        curr_data = extract_frame_data(frame_num)
        if not curr_data:
            print(f"Frame {frame_num}: Could not extract data")
            return False
        
        kept_tiles = len(curr_data['tiles'])
        filtered_tiles = len(curr_data['filtered'])
        print(f"Frame {frame_num}: {len(curr_data['grid_map'])} detected, {kept_tiles} kept, {filtered_tiles} filtered", end="")
        
        # Clear all axes and recreate
        for ax in fig.get_axes():
            ax.remove()
        
        ax_prev = fig.add_subplot(1, 2, 1)
        ax_curr = fig.add_subplot(1, 2, 2)
        
        # ===== LEFT PANEL: Previous Frame with Grid Overlay =====
        if frame_num > 0 and prev_frame_data and 'frame' in prev_frame_data:
            prev_data = prev_frame_data
            frame_with_overlay_prev = prev_data['frame'].copy()
            
            # Draw grid lines
            if prev_data['lines'] is not None:
                for rho, theta in prev_data['lines']:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(frame_with_overlay_prev, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Draw tile coordinates
            for (row, col) in prev_data['tiles'].keys():
                square_polygon = prev_data['grid_map'][(row, col)]
                p1, p2, p3, p4 = square_polygon
                cx = int((p1[0] + p3[0]) / 2)
                cy = int((p1[1] + p3[1]) / 2)
                
                label = f"({row},{col})"
                cv2.putText(frame_with_overlay_prev, label, (cx - 25, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ax_prev.imshow(cv2.cvtColor(frame_with_overlay_prev, cv2.COLOR_BGR2RGB))
            ax_prev.set_title(f'Frame {frame_num-1} (Previous)')
        else:
            ax_prev.text(0.5, 0.5, 'No previous frame', ha='center', va='center', transform=ax_prev.transAxes)
            ax_prev.set_title('Previous Frame')
        ax_prev.axis('off')
        
        # ===== MIDDLE PANEL: Current Frame with Grid Overlay =====
        frame_with_overlay = curr_data['frame'].copy()
        
        # Draw grid lines
        if curr_data['lines'] is not None:
            for rho, theta in curr_data['lines']:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(frame_with_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw tile coordinates
        for (row, col) in curr_data['tiles'].keys():
            square_polygon = curr_data['grid_map'][(row, col)]
            p1, p2, p3, p4 = square_polygon
            cx = int((p1[0] + p3[0]) / 2)
            cy = int((p1[1] + p3[1]) / 2)
            
            label = f"({row},{col})"
            cv2.putText(frame_with_overlay, label, (cx - 25, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        ax_curr.imshow(cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB))
        ax_curr.set_title(f'Frame {frame_num} (Current)')
        ax_curr.axis('off')
        
        print()
        
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
    
    # Create figure with 2 panels
    fig = plt.figure(figsize=(14, 6))
    state['fig'] = fig
    
    # Initial update
    if update_visualization(fig, frame_number):
        fig.canvas.mpl_connect('key_press_event', on_key)
        manager = fig.canvas.manager
        manager.window.showMaximized()
        plt.show()
    
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Frame similarity matrix visualization with grid overlay.')
    parser.add_argument('--video', type=str, default='video.mp4', help='Video file path')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to start (default: 0)')
    
    args = parser.parse_args()
    
    frame_similarity_viz(args.video, args.frame)

