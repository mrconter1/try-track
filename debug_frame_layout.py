import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from line_detector import LineDetector
from auto_tile_detector import extract_grid_squares, extract_and_warp_square, generate_tile_hash

def debug_frame_layout(video_path, frame_number=0):
    """
    Generates a 2-panel debug visualization with keyboard navigation.
    Panel 1: Frame with detected lines and grid coordinates.
    Panel 2: Warped tiles in 2D grid with signatures.
    
    Navigation: RIGHT arrow = next frame, LEFT arrow = previous frame, Q = quit
    """
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # State for navigation
    state = {'current_frame': frame_number, 'cap': cap, 'total_frames': total_frames, 'fig': None, 'should_update': True}
    
    def update_frame(fig, frame_num):
        """Update existing figure with new frame data"""
        if frame_num < 0 or frame_num >= total_frames:
            return False
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Error: Could not read frame {frame_num}")
            return False
        
        # --- Detect Lines and Extract Grid ---
        line_detector = LineDetector()
        frame_with_lines, lines = line_detector.detect_lines(frame)
        grid_map = extract_grid_squares(lines, frame.shape[:2])
        
        if not grid_map:
            print(f"Frame {frame_num}: No tiles detected")
            return False
        
        print(f"Frame {frame_num}: {len(grid_map)} tiles detected", end="")
        
        # --- Calculate Tile Signatures ---
        tile_signatures = {}
        filtered_coords = set()
        filtered_tiles = 0
        for (row, col), square_polygon in grid_map.items():
            try:
                warped = extract_and_warp_square(frame, square_polygon)
                
                # Filter by std dev (quality check)
                gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                std_dev = float(np.std(gray_warped))
                if std_dev > 20:
                    filtered_tiles += 1
                    filtered_coords.add((row, col))
                    continue
                
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
                    
                    hash_sig = generate_tile_hash(rotated, blocks_per_side=8, use_clahe=True)
                    if hash_sig:
                        signatures[rotation] = hash_sig
                
                tile_signatures[(row, col)] = signatures
            except Exception as e:
                pass
        
        if filtered_tiles > 0:
            print(f" → {filtered_tiles} filtered (std > 20), {len(tile_signatures)} kept")
        else:
            print()
        
        # --- Determine Grid Dimensions (only from kept tiles) ---
        kept_coords = [(r, c) for r, c in grid_map.keys() if (r, c) not in filtered_coords]
        all_rows = [r for r, c in kept_coords]
        all_cols = [c for r, c in kept_coords]
        num_rows = max(all_rows) + 1 if all_rows else 0
        num_cols = max(all_cols) + 1 if all_cols else 0
        
        # Clear axes
        ax1, ax2 = fig.get_axes()
        ax1.clear()
        ax2.clear()
        
        # Panel 1: Frame with Lines and Grid Coordinates
        frame_with_coords = frame_with_lines.copy()
        for (row, col), square_polygon in grid_map.items():
            # Skip filtered tiles
            if (row, col) in filtered_coords:
                continue
            
            p1, p2, p3, p4 = square_polygon
            cx = int((p1[0] + p3[0]) / 2)
            cy = int((p1[1] + p3[1]) / 2)
            
            label = f"({row},{col})"
            cv2.putText(frame_with_coords, label, (cx - 25, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        ax1.imshow(cv2.cvtColor(frame_with_coords, cv2.COLOR_BGR2RGB))
        ax1.set_title('Lines & Grid Coordinates')
        ax1.axis('off')
        
        # Panel 2: 2D Grid of Warped Images with Signatures
        tile_display_size = 120
        composite_height = max(1, num_rows) * tile_display_size
        composite_width = max(1, num_cols) * tile_display_size
        composite_image = np.full((composite_height, composite_width, 3), 40, dtype=np.uint8)
        
        for (row, col), square_polygon in grid_map.items():
            # Skip filtered tiles
            if (row, col) in filtered_coords:
                continue
            
            try:
                warped = extract_and_warp_square(frame, square_polygon)
                warped_resized = cv2.resize(warped, (tile_display_size, tile_display_size))
                
                pil_image = Image.fromarray(cv2.cvtColor(warped_resized, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                
                try:
                    font_small = ImageFont.truetype("arial.ttf", 10)
                except:
                    font_small = ImageFont.load_default()
                
                label = f"({row},{col})"
                draw.text((5, 5), label, fill=(255, 0, 0), font=font_small)
                
                if (row, col) in tile_signatures:
                    sigs = tile_signatures[(row, col)]
                    y_offset = 25
                    for rot in [0, 90, 180, 270]:
                        if rot in sigs:
                            sig = sigs[rot]
                            sig_val = sig[:8] if isinstance(sig, str) else f"{sig[0]:.1f}"
                            sig_line = f"{rot}deg={sig_val}"
                            draw.text((3, y_offset), sig_line, fill=(255, 0, 0), font=font_small)
                            y_offset += 12
                
                warped_resized = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                y_start = row * tile_display_size
                x_start = col * tile_display_size
                composite_image[y_start:y_start+tile_display_size, x_start:x_start+tile_display_size] = warped_resized
            except Exception as e:
                pass
        
        ax2.imshow(composite_image)
        ax2.set_title(f'Frame {frame_num} - Warped Tiles in {num_rows}x{num_cols} Grid | RIGHT=next, LEFT=prev, Q=quit')
        ax2.set_xticks(np.arange(-.5, num_cols, 1), minor=True)
        ax2.set_yticks(np.arange(-.5, num_rows, 1), minor=True)
        ax2.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        ax2.tick_params(which="minor", size=0)
        ax2.set_xticks(np.arange(0, num_cols, 1))
        ax2.set_yticks(np.arange(0, num_rows, 1))
        
        fig.canvas.draw_idle()
        return True
    
    def on_key(event):
        """Handle keyboard events"""
        if event.key == 'right':
            next_frame = state['current_frame'] + 1
            if next_frame < state['total_frames']:
                state['current_frame'] = next_frame
                print(f"→ Frame {next_frame}")
                update_frame(state['fig'], next_frame)
        elif event.key == 'left':
            prev_frame = state['current_frame'] - 1
            if prev_frame >= 0:
                state['current_frame'] = prev_frame
                print(f"← Frame {prev_frame}")
                update_frame(state['fig'], prev_frame)
        elif event.key == 'q':
            print("Closing...")
            plt.close('all')
    
    # Initial visualization
    print(f"Loading video: {video_path}")
    print(f"Starting at frame {frame_number}/{total_frames-1}")
    print("Use RIGHT arrow to go to next frame, LEFT arrow for previous frame, Q to quit")
    
    # Create figure once
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    state['fig'] = fig
    
    # Initial update
    if update_frame(fig, frame_number):
        fig.canvas.mpl_connect('key_press_event', on_key)
        manager = fig.canvas.manager
        manager.window.showMaximized()
        plt.show()
    
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debug frame processing with keyboard navigation.')
    parser.add_argument('--video', type=str, default='video.mp4', help='Video file path')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to start (default: 0)')
    
    args = parser.parse_args()
    
    debug_frame_layout(args.video, args.frame)
