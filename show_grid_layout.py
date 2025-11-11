import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from line_detector import LineDetector
from auto_tile_detector import extract_grid_squares, extract_and_warp_square

def show_grid_layout(video_path, frame_number=0):
    """
    Generates a 2-panel visualization:
    1. Frame with detected lines and (row, col) grid coordinates.
    2. A composite image showing warped tiles in their correct 2D layout.
    """
    
    # --- Load Frame ---
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
    
    print(f"Processing frame {frame_number}...")

    # --- Detect Lines and Extract Grid ---
    line_detector = LineDetector()
    frame_with_lines, lines = line_detector.detect_lines(frame)
    grid_map = extract_grid_squares(lines, frame.shape[:2])
    
    print(f"Detected {len(lines)} lines, which formed {len(grid_map)} tiles.")

    if not grid_map:
        print("No tiles detected.")
        return

    # --- Determine Grid Dimensions ---
    all_rows = [r for r, c in grid_map.keys()]
    all_cols = [c for r, c in grid_map.keys()]
    num_rows = max(all_rows) + 1 if all_rows else 0
    num_cols = max(all_cols) + 1 if all_cols else 0
    print(f"Grid layout: {num_rows}x{num_cols}")

    # --- Create 2-Panel Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Panel 1: Frame with Lines and Grid Coordinates
    frame_with_coords = frame_with_lines.copy()
    for (row, col), square_polygon in grid_map.items():
        p1, p2, p3, p4 = square_polygon
        cx = int((p1[0] + p3[0]) / 2)
        cy = int((p1[1] + p3[1]) / 2)
        
        label = f"({row},{col})"
        cv2.putText(frame_with_coords, label, (cx - 25, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                   
    ax1.imshow(cv2.cvtColor(frame_with_coords, cv2.COLOR_BGR2RGB))
    ax1.set_title('Lines & Grid Coordinates')
    ax1.axis('off')
    
    # Panel 2: 2D Grid of Warped Images
    tile_display_size = 100
    composite_height = max(1, num_rows) * tile_display_size
    composite_width = max(1, num_cols) * tile_display_size
    composite_image = np.full((composite_height, composite_width, 3), 40, dtype=np.uint8)

    for (row, col), square_polygon in grid_map.items():
        try:
            warped = extract_and_warp_square(frame, square_polygon)
            warped_resized = cv2.resize(warped, (tile_display_size, tile_display_size))
            
            # Add coordinate label
            label = f"({row},{col})"
            cv2.putText(warped_resized, label, (5, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            y_start = row * tile_display_size
            x_start = col * tile_display_size
            composite_image[y_start:y_start+tile_display_size, x_start:x_start+tile_display_size] = warped_resized
        except Exception as e:
            print(f"  Error warping tile at ({row},{col}): {e}")

    ax2.imshow(composite_image)
    ax2.set_title(f'Warped Tiles in {num_rows}x{num_cols} Grid')
    ax2.set_xticks(np.arange(-.5, num_cols, 1), minor=True)
    ax2.set_yticks(np.arange(-.5, num_rows, 1), minor=True)
    ax2.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax2.tick_params(which="minor", size=0)
    ax2.set_xticks(np.arange(0, num_cols, 1))
    ax2.set_yticks(np.arange(0, num_rows, 1))
    
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    
    print("✓ Displaying grid layout.")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show grid layout with 2 panels.')
    parser.add_argument('--video', type=str, default='video.mp4', help='Video file path')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to display (default: 0)')
    
    args = parser.parse_args()
    show_grid_layout(args.video, args.frame)

