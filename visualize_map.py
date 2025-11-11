import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def visualize_map(json_path, grid_size=100, use_scatter=True):
    """Load global map JSON and visualize tile positions"""
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    positions = data['tile_global_positions']
    num_tiles = len(positions)
    
    print(f"Loaded {num_tiles} tiles from {json_path}")
    
    # Extract coordinates
    rows = []
    cols = []
    tile_ids = []
    for tile_id_str, (row, col) in positions.items():
        rows.append(int(row))
        cols.append(int(col))
        tile_ids.append(int(tile_id_str))
    
    if not rows:
        print("No tiles found!")
        return
    
    # Get bounds
    min_row, max_row = int(min(rows)), int(max(rows))
    min_col, max_col = int(min(cols)), int(max(cols))
    
    print(f"\nTile position statistics:")
    print(f"  Rows: {min_row} to {max_row} (span: {max_row - min_row + 1})")
    print(f"  Cols: {min_col} to {max_col} (span: {max_col - min_col + 1})")
    print(f"  Total tiles: {num_tiles}")
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Scatter plot - all green dots
    ax.scatter(cols, rows, c='green', s=100, alpha=0.7, edgecolors='darkgreen', linewidth=1, label='Tiles')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Labels and title
    ax.set_xlabel(f'Column Position (X) [{min_col} to {max_col}]', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Row Position (Y) [{min_row} to {max_row}]', fontsize=12, fontweight='bold')
    ax.set_title(f'Global Map - Tile Positions (Scatter Plot)\n{num_tiles} tiles: Rows [{min_row}, {max_row}] Cols [{min_col}, {max_col}]', 
                fontsize=14, fontweight='bold')
    
    # Invert Y axis so row 0 is at top
    ax.invert_yaxis()
    
    # Add some padding
    row_padding = (max_row - min_row) * 0.1 if (max_row - min_row) > 0 else 1
    col_padding = (max_col - min_col) * 0.1 if (max_col - min_col) > 0 else 1
    ax.set_ylim(max_row + row_padding, min_row - row_padding)
    ax.set_xlim(min_col - col_padding, max_col + col_padding)
    
    # Add major ticks
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Maximize window on launch
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize global map as green cells on a grid')
    parser.add_argument('--json', type=str, default='export/global_map.json', help='Path to global_map.json')
    parser.add_argument('--grid-size', type=int, default=100, help='Grid size (default: 100x100)')
    parser.add_argument('--save', type=str, default=None, help='Save to PNG instead of displaying')
    
    args = parser.parse_args()
    
    visualize_map(args.json, grid_size=args.grid_size)

