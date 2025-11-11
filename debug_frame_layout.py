import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from line_detector import LineDetector
from auto_tile_detector import extract_grid_squares, extract_and_warp_square

def debug_frame_layout(video_path, frame_number=0):
    """
    Generates a 3-panel debug visualization for a single video frame:
    1. Original Frame
    2. Frame with detected lines and (row, col) grid coordinates overlaid.
    3. A composite image showing the warped tile images arranged in their detected 2D grid layout, with empty cells for missing tiles.
    """
    
    # --- 1. Load Frame ---
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        return
    
    print(f"\nProcessing frame {frame_number} (Shape: {frame.shape})")

    # --- 2. Detect Lines and Squares ---
    line_detector = LineDetector()
    frame_with_lines, lines = line_detector.detect_lines(frame)
    squares = extract_grid_squares(lines, frame.shape[:2])
    print(f"Detected {len(lines)} lines, which formed {len(squares)} potential tiles.")

    if not squares:
        print("No tiles were detected. Cannot generate grid layout.")
        # Display just the original and lines
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Frame {frame_number}: Original')
        ax1.axis('off')
        ax2.imshow(cv2.cvtColor(frame_with_lines, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Detected Lines ({len(lines)})')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()
        return

    # --- 3. Organize Squares into a 2D Grid (NEW, more robust logic) ---
    centers = []
    for idx, square in enumerate(squares):
        p1, p2, p3, p4 = square
        cx = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
        cy = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
        centers.append({'cx': cx, 'cy': cy, 'square_idx': idx, 'assigned': False})

    rows = []
    # Increased threshold to handle perspective skew
    row_threshold = 75  

    # Keep finding rows until all centers are assigned
    while any(not c['assigned'] for c in centers):
        # Find the topmost unassigned center to start a new row
        try:
            first_in_row = min((c for c in centers if not c['assigned']), key=lambda p: p['cy'])
        except ValueError:
            break # No unassigned centers left

        current_row_centers = []
        # Find all other unassigned centers that are vertically close to this one
        for center in centers:
            if not center['assigned']:
                if abs(center['cy'] - first_in_row['cy']) < row_threshold:
                    current_row_centers.append(center)
                    center['assigned'] = True
        
        # Sort the centers in the current row by their x-coordinate
        if current_row_centers:
            sorted_row = sorted(current_row_centers, key=lambda p: p['cx'])
            rows.append(sorted_row)

    grid_map = {}
    max_col_count = 0
    for row_idx, row_items in enumerate(rows):
        max_col_count = max(max_col_count, len(row_items))
        for col_idx, item in enumerate(row_items):
            grid_map[(row_idx, col_idx)] = item
    
    num_rows = len(rows)
    num_cols = max_col_count
    print(f"Organized tiles into a {num_rows}x{num_cols} logical grid.")

    # --- 4. Create Visualizations ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7))
    
    # Panel 1: Original Frame
    ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Frame {frame_number}: Original')
    ax1.axis('off')

    # Panel 2: Frame with Lines and Grid Coordinates
    frame_with_coords = frame_with_lines.copy()
    for row_idx, row_items in enumerate(rows):
        for col_idx, item in enumerate(row_items):
            label = f"({row_idx},{col_idx})"
            cv2.putText(frame_with_coords, label, (int(item['cx']) - 25, int(item['cy'])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Add the tile index number
            id_text = f"#{item['square_idx']}"
            cv2.putText(frame_with_coords, id_text, (int(item['cx']) - 25, int(item['cy']) + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    ax2.imshow(cv2.cvtColor(frame_with_coords, cv2.COLOR_BGR2RGB))
    ax2.set_title('Lines & Grid Coordinates')
    ax2.axis('off')
    
    # Panel 3: 2D Grid of Warped Images
    tile_display_size = 100
    # Ensure dimensions are at least 1, even if 0
    composite_height = max(1, num_rows) * tile_display_size
    composite_width = max(1, num_cols) * tile_display_size
    composite_image = np.full((composite_height, composite_width, 3), 40, dtype=np.uint8) # Dark gray background

    print("\nConstructing Warped Tile Grid:")
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            # Check if a tile exists at this grid coordinate
            if (row_idx, col_idx) in grid_map:
                item = grid_map[(row_idx, col_idx)]
                square_idx = item['square_idx']
                
                try:
                    warped = extract_and_warp_square(frame, squares[square_idx])
                    warped_resized = cv2.resize(warped, (tile_display_size, tile_display_size))
                    
                    y_start = row_idx * tile_display_size
                    x_start = col_idx * tile_display_size
                    composite_image[y_start:y_start+tile_display_size, x_start:x_start+tile_display_size] = warped_resized
                    
                    # Add tile ID text to the warped image
                    id_text = f"#{square_idx}"
                    cv2.putText(composite_image, id_text, 
                               (x_start + 5, y_start + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                except Exception as e:
                    print(f"  Error warping tile at ({row_idx},{col_idx}): {e}")
            # If no tile, the dark gray background will show through as an empty cell

    ax3.imshow(composite_image)
    ax3.set_title(f'Warped Tiles in {num_rows}x{num_cols} Grid')
    ax3.set_xticks(np.arange(-.5, num_cols, 1), minor=True)
    ax3.set_yticks(np.arange(-.5, num_rows, 1), minor=True)
    ax3.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax3.tick_params(which="minor", size=0)
    ax3.set_xticks(np.arange(0, num_cols, 1))
    ax3.set_yticks(np.arange(0, num_rows, 1))
    ax3.set_xticklabels(np.arange(0, num_cols, 1))
    ax3.set_yticklabels(np.arange(0, num_rows, 1))
    
    plt.tight_layout()
    
    # Maximize window
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    
    print("\n✓ Displaying debug visualization.")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debug frame processing with a 3-panel visualization.')
    parser.add_argument('--video', type=str, default='video.mp4', help='Video file path')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to display (default: 0)')
    
    args = parser.parse_args()
    
    debug_frame_layout(args.video, args.frame)
