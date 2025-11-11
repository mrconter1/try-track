import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from line_detector import LineDetector
from auto_tile_detector import extract_grid_squares, extract_and_warp_square

def show_frame_grid(video_path, frame_number=0):
    """Display the grid for a single frame"""
    
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
    
    # Detect lines
    line_detector = LineDetector()
    frame_with_lines, lines = line_detector.detect_lines(frame)
    
    # Extract grid squares
    squares = extract_grid_squares(lines, frame.shape[:2])
    
    print(f"Detected {len(squares)} tiles")
    
    if not squares:
        print("No tiles detected!")
        return
    
    # Organize squares into 2D grid by position
    centers = []
    for idx, square in enumerate(squares):
        p1, p2, p3, p4 = square
        cx = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
        cy = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
        centers.append((cx, cy, idx))
    
    # Sort by y (top to bottom), then x (left to right)
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
    
    # Create grid dictionary
    grid = {}
    for row_idx, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda p: p[0])
        for col_idx, (cx, cy, square_idx) in enumerate(row_sorted):
            grid[(row_idx, col_idx)] = square_idx
    
    num_rows = len(rows)
    num_cols = max(len(r) for r in rows) if rows else 0
    
    print(f"Grid size: {num_rows} rows x {num_cols} columns")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot tiles
    plot_rows = []
    plot_cols = []
    tile_indices = []
    
    for (row, col), square_idx in grid.items():
        plot_rows.append(row)
        plot_cols.append(col)
        tile_indices.append(square_idx)
    
    # Scatter plot
    ax.scatter(plot_cols, plot_rows, c='green', s=300, alpha=0.7, edgecolors='darkgreen', linewidth=2)
    
    # Add tile index labels
    for row, col, tile_idx in zip(plot_rows, plot_cols, tile_indices):
        ax.text(col, row, str(tile_idx), ha='center', va='center', 
               fontsize=12, color='white', weight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.invert_yaxis()
    
    # Labels
    ax.set_xlabel(f'Column (0 to {num_cols-1})', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Row (0 to {num_rows-1})', fontsize=14, fontweight='bold')
    ax.set_title(f'Frame {frame_number}: Grid Layout\n{len(grid)} tiles in {num_rows}x{num_cols} grid', 
                fontsize=16, fontweight='bold')
    
    # Set limits with padding
    ax.set_xlim(-0.5, num_cols - 0.5)
    ax.set_ylim(num_rows - 0.5, -0.5)
    
    plt.tight_layout()
    
    # Maximize window
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    
    print("\n✓ Displaying grid visualization")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show grid layout for a single frame')
    parser.add_argument('--video', type=str, default='video.mp4', help='Video file path')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to display (default: 0)')
    
    args = parser.parse_args()
    
    show_frame_grid(args.video, args.frame)

