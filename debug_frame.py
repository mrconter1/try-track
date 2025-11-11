import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from line_detector import LineDetector
from auto_tile_detector import extract_grid_squares, extract_and_warp_square

def debug_frame(video_path, frame_number=0):
    """Debug visualization showing frame, lines, and extracted tiles"""
    
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Go to specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        return
    
    print(f"\nProcessing frame {frame_number}...")
    print(f"Frame shape: {frame.shape}")
    
    # Detect lines
    line_detector = LineDetector()
    frame_with_lines, lines = line_detector.detect_lines(frame)
    
    print(f"\nDetected {len(lines)} lines")
    
    # Separate horizontal and vertical lines
    horizontal = []
    vertical = []
    
    for rho, theta in lines:
        theta_deg = theta * 180 / np.pi
        theta_deg = theta_deg % 180
        
        if theta_deg < 45 or theta_deg > 135:
            horizontal.append((rho, theta))
        else:
            vertical.append((rho, theta))
    
    print(f"  - {len(horizontal)} horizontal lines")
    print(f"  - {len(vertical)} vertical lines")
    
    # Extract grid squares
    squares = extract_grid_squares(lines, frame.shape[:2])
    print(f"\nExtracted {len(squares)} tiles from grid intersections")
    
    # Organize into 2D grid and print coordinates
    if squares:
        # Get centers
        centers = []
        for idx, square in enumerate(squares):
            p1, p2, p3, p4 = square
            cx = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
            cy = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
            centers.append((cx, cy, idx))
        
        # Sort by y, then x
        centers_sorted = sorted(centers, key=lambda p: (p[1], p[0]))
        
        # Group into rows
        rows = []
        current_row = []
        row_threshold = 50
        
        for cx, cy, idx in centers_sorted:
            if not current_row:
                current_row.append((cx, cy, idx))
            else:
                if abs(cy - current_row[0][1]) < row_threshold:
                    current_row.append((cx, cy, idx))
                else:
                    rows.append(current_row)
                    current_row = [(cx, cy, idx)]
        if current_row:
            rows.append(current_row)
        
        # Create grid and print
        print(f"\nGrid layout ({len(rows)} rows):")
        for row_idx, row in enumerate(rows):
            row_sorted = sorted(row, key=lambda p: p[0])
            print(f"  Row {row_idx}:", end=" ")
            for col_idx, (cx, cy, square_idx) in enumerate(row_sorted):
                print(f"[{row_idx},{col_idx}]=tile{square_idx}", end=" ")
            print()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 6))
    
    # 1. Original frame
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Frame {frame_number}: Original', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. Frame with detected lines AND grid coordinates
    ax2 = fig.add_subplot(1, 3, 2)
    frame_with_coords = frame_with_lines.copy()
    
    # Draw grid coordinates on tiles
    if squares:
        # This logic is duplicated from the print section, can be refactored later
        centers = []
        for idx, square in enumerate(squares):
            p1, p2, p3, p4 = square
            cx = int((p1[0] + p2[0] + p3[0] + p4[0]) / 4)
            cy = int((p1[1] + p2[1] + p3[1] + p4[1]) / 4)
            centers.append((cx, cy, idx))
        
        centers_sorted = sorted(centers, key=lambda p: (p[1], [0]))
        
        rows_for_drawing = []
        current_row_for_drawing = []
        row_threshold = 50
        
        for cx, cy, idx in centers_sorted:
            if not current_row_for_drawing:
                current_row_for_drawing.append((cx, cy, idx))
            else:
                if abs(cy - current_row_for_drawing[0][1]) < row_threshold:
                    current_row_for_drawing.append((cx, cy, idx))
                else:
                    rows_for_drawing.append(current_row_for_drawing)
                    current_row_for_drawing = [(cx, cy, idx)]
        if current_row_for_drawing:
            rows_for_drawing.append(current_row_for_drawing)
            
        # Draw coordinates and collect for plot
        plot_rows = []
        plot_cols = []
        plot_labels = []

        for row_idx, row in enumerate(rows_for_drawing):
            row_sorted = sorted(row, key=lambda p: p[0])
            for col_idx, (cx, cy, square_idx) in enumerate(row_sorted):
                # For plotting
                plot_rows.append(row_idx)
                plot_cols.append(col_idx)
                plot_labels.append(f"({row_idx},{col_idx})\n#{square_idx}")
                
                # Draw on image
                label = f"({row_idx},{col_idx})"
                cv2.putText(frame_with_coords, label, (cx - 30, cy + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame_with_coords, f"#{square_idx}", (cx - 20, cy + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    ax2.imshow(cv2.cvtColor(frame_with_coords, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Detected Lines + Grid Coords\n{len(lines)} total ({len(horizontal)}H, {len(vertical)}V)', 
                 fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 3. Grid layout plot
    ax3 = fig.add_subplot(1, 3, 3)
    
    if squares and 'rows_for_drawing' in locals() and rows_for_drawing:
        num_rows = len(rows_for_drawing)
        num_cols = max(len(r) for r in rows_for_drawing) if rows_for_drawing else 0
        
        ax3.scatter(plot_cols, plot_rows, c='green', s=300, alpha=0.7)
        for r, c, label in zip(plot_rows, plot_cols, plot_labels):
            ax3.text(c, r, label, ha='center', va='center', color='white', weight='bold')
        
        ax3.set_title(f'Grid Layout ({num_rows}x{num_cols})', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Column')
        ax3.set_ylabel('Row')
        if num_cols > 0:
            ax3.set_xticks(range(num_cols))
        if num_rows > 0:
            ax3.set_yticks(range(num_rows))
        ax3.invert_yaxis()
        ax3.grid(True)
    else:
        ax3.text(0.5, 0.5, 'No grid detected', ha='center', va='center')
    
    ax3.axis('on')  # Keep axis for grid plot
    
    plt.tight_layout()
    
    # Maximize window
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    
    print("\n✓ Displaying debug visualization")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debug frame processing with visualization')
    parser.add_argument('--video', type=str, default='video.mp4', help='Video file path')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to display (default: 0)')
    
    args = parser.parse_args()
    
    debug_frame(args.video, args.frame)

