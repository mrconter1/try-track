import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import binascii
from scipy.fftpack import dct
from line_detector import LineDetector
import os

def compute_dhash_byte(block):
    """Compute dHash for a block"""
    h, w = block.shape
    
    # Split block into quadrants
    h_mid = h // 2
    w_mid = w // 2
    
    q1 = block[:h_mid, :w_mid].mean()
    q2 = block[:h_mid, w_mid:].mean()
    q3 = block[h_mid:, :w_mid].mean()
    q4 = block[h_mid:, w_mid:].mean()
    
    # Compute 8 bits from different comparisons
    bits = []
    bits.append(1 if q1 > q2 else 0)
    bits.append(1 if q3 > q4 else 0)
    bits.append(1 if q1 > q3 else 0)
    bits.append(1 if q2 > q4 else 0)
    bits.append(1 if q1 > q4 else 0)
    bits.append(1 if q2 > q3 else 0)
    bits.append(1 if (q1 + q4) > (q2 + q3) else 0)
    bits.append(1 if block.mean() > 128 else 0)
    
    byte_val = 0
    for i, bit in enumerate(bits):
        byte_val += bit * (2 ** i)
    
    return byte_val

def generate_tile_hash(tile_image, blocks_per_side=16, use_clahe=True):
    """Generate dhash for a tile image"""
    if tile_image is None or tile_image.size == 0:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
    
    # Resize to 256x256
    resized = cv2.resize(gray, (256, 256))
    
    # Apply CLAHE preprocessing
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(resized)
    else:
        processed = resized
    
    # Compute dhash blocks
    block_size = 256 // blocks_per_side
    signature = []
    
    for row in range(blocks_per_side):
        for col in range(blocks_per_side):
            y_start = row * block_size
            y_end = (row + 1) * block_size
            x_start = col * block_size
            x_end = (col + 1) * block_size
            
            block = processed[y_start:y_end, x_start:x_end]
            descriptor = compute_dhash_byte(block)
            signature.append(descriptor)
    
    # Convert to hex hash
    signature_bytes = bytes(signature)
    hash_str = binascii.hexlify(signature_bytes).decode('ascii')
    
    return hash_str

def hex_to_bytes(hex_str):
    """Convert hex string to array of byte values"""
    return [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]

def euclidean_distance(hash1, hash2):
    """Compute Euclidean distance between two hashes"""
    bytes1 = hex_to_bytes(hash1)
    bytes2 = hex_to_bytes(hash2)
    return np.sqrt(sum((b1 - b2)**2 for b1, b2 in zip(bytes1, bytes2)))

def extract_grid_squares(lines, frame_shape):
    """Extract grid squares formed by intersecting lines"""
    if len(lines) < 2:
        return []
    
    h, w = frame_shape[:2]
    squares = []
    
    # Group lines into horizontal and vertical
    horizontal = []
    vertical = []
    
    for rho, theta in lines:
        theta_deg = theta * 180 / np.pi
        theta_deg = theta_deg % 180
        
        if theta_deg < 45 or theta_deg > 135:
            horizontal.append((rho, theta))
        else:
            vertical.append((rho, theta))
    
    # Sort lines by rho
    horizontal.sort(key=lambda x: x[0])
    vertical.sort(key=lambda x: x[0])
    
    # Find intersections between adjacent horizontal and vertical lines
    for i in range(len(horizontal) - 1):
        for j in range(len(vertical) - 1):
            rho_h1, theta_h1 = horizontal[i]
            rho_h2, theta_h2 = horizontal[i + 1]
            rho_v1, theta_v1 = vertical[j]
            rho_v2, theta_v2 = vertical[j + 1]
            
            # Find four corner points
            p1 = line_intersection(rho_h1, theta_h1, rho_v1, theta_v1)
            p2 = line_intersection(rho_h1, theta_h1, rho_v2, theta_v2)
            p3 = line_intersection(rho_h2, theta_h2, rho_v2, theta_v2)
            p4 = line_intersection(rho_h2, theta_h2, rho_v1, theta_v1)
            
            if all(p is not None for p in [p1, p2, p3, p4]):
                # Check if all points are within frame bounds
                if all(0 <= p[0] <= w and 0 <= p[1] <= h for p in [p1, p2, p3, p4]):
                    squares.append((p1, p2, p3, p4))
    
    return squares

def line_intersection(rho1, theta1, rho2, theta2):
    """Find intersection of two lines defined by (rho, theta)"""
    a1 = np.cos(theta1)
    b1 = np.sin(theta1)
    a2 = np.cos(theta2)
    b2 = np.sin(theta2)
    
    denom = a1 * b2 - a2 * b1
    if abs(denom) < 1e-6:
        return None
    
    x = (rho1 * b2 - rho2 * b1) / denom
    y = (a1 * rho2 - a2 * rho1) / denom
    
    return (int(x), int(y))

def extract_and_warp_square(frame, square):
    """Extract and warp square to perfect square perspective"""
    p1, p2, p3, p4 = square
    
    # Calculate appropriate size
    side_length = max(
        int(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)),
        int(np.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)),
        int(np.sqrt((p3[0] - p4[0])**2 + (p3[1] - p4[1])**2)),
        int(np.sqrt((p4[0] - p1[0])**2 + (p4[1] - p1[1])**2))
    )
    side_length = max(side_length, 50)
    
    # Source points
    src_points = np.float32([p1, p2, p3, p4])
    
    # Destination points
    dst_points = np.float32([
        [0, 0],
        [side_length, 0],
        [side_length, side_length],
        [0, side_length]
    ])
    
    # Perspective transform
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(frame, matrix, (side_length, side_length))
    
    return warped

def main(video_path, start_frame, num_frames, blocks, use_clahe):
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    line_detector = LineDetector()
    tile_hashes = []
    tile_names = []
    tile_images = {}  # Store warped images for hover display
    
    print(f"Processing frames {start_frame} to {start_frame + num_frames}")
    
    for frame_idx in range(start_frame, start_frame + num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Could not read frame {frame_idx}")
            continue
        
        print(f"Frame {frame_idx}...", end=" ")
        
        # Detect lines
        frame_with_lines, lines = line_detector.detect_lines(frame)
        
        # Extract grid squares
        squares = extract_grid_squares(lines, frame.shape[:2])
        
        # Extract tiles
        for square_idx, square in enumerate(squares):
            try:
                warped = extract_and_warp_square(frame, square)
                hash_str = generate_tile_hash(warped, blocks, use_clahe)
                
                if hash_str:
                    tile_name = f"F{frame_idx}_S{square_idx}"
                    tile_hashes.append(hash_str)
                    tile_names.append(tile_name)
                    tile_images[tile_name] = warped  # Store for hover display
            except Exception as e:
                print(f"Error processing square: {e}")
                continue
        
        print(f"({len(squares)} tiles)")
    
    cap.release()
    
    if len(tile_hashes) < 2:
        print("Not enough tiles detected")
        return
    
    print(f"\nTotal tiles extracted: {len(tile_hashes)}")
    
    # Compute distance matrix
    print("Computing distance matrix...")
    num_tiles = len(tile_hashes)
    distance_matrix = np.zeros((num_tiles, num_tiles))
    
    for i in range(num_tiles):
        for j in range(num_tiles):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                distance_matrix[i][j] = euclidean_distance(tile_hashes[i], tile_hashes[j])
    
    # Normalize
    min_dist = np.min(distance_matrix[distance_matrix > 0])
    max_dist = np.max(distance_matrix)
    normalized_matrix = (distance_matrix - min_dist) / (max_dist - min_dist)
    
    print(f"Distance range: {min_dist:.0f} - {max_dist:.0f}")
    
    # Display heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(normalized_matrix, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(range(num_tiles))
    ax.set_yticks(range(num_tiles))
    ax.set_xticklabels(tile_names, fontsize=8)
    ax.set_yticklabels(tile_names, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Distance (0=similar, 1=different)', rotation=270, labelpad=20)
    
    ax.set_xlabel('Tile', fontsize=10)
    ax.set_ylabel('Tile', fontsize=10)
    ax.set_title(f'Tile Similarity Matrix (dhash + euclidean, {blocks}x{blocks} blocks, clahe={use_clahe})', fontsize=12, fontweight='bold')
    
    ax.set_xticks(np.arange(num_tiles)-.5, minor=True)
    ax.set_yticks(np.arange(num_tiles)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # Store data for hover interaction
    hover_data = {
        'tile_names': tile_names,
        'distance_matrix': distance_matrix,
        'normalized_matrix': normalized_matrix,
        'tile_images': tile_images,
        'text_box': None,
        'rect': None,
        'image_display': None
    }
    
    # Define hover event handler
    def on_hover(event):
        if event.inaxes != ax:
            if hover_data['text_box']:
                hover_data['text_box'].set_visible(False)
            if hover_data['rect']:
                hover_data['rect'].set_visible(False)
            if hover_data['image_display']:
                hover_data['image_display'].set_visible(False)
            fig.canvas.draw_idle()
            return
        
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        
        if 0 <= x < num_tiles and 0 <= y < num_tiles:
            tile_y = tile_names[y]
            tile_x = tile_names[x]
            
            raw_dist = distance_matrix[y, x]
            norm_dist = normalized_matrix[y, x]
            
            info_text = f"Y: {tile_y}\nX: {tile_x}\nDist: {raw_dist:.0f}\nNorm: {norm_dist:.3f}"
            
            # Create or update text box
            if hover_data['text_box'] is None:
                hover_data['text_box'] = ax.annotate('', xy=(event.xdata, event.ydata),
                                                      xytext=(10, 10), textcoords='offset points',
                                                      fontsize=9,
                                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                                                      ha='left', va='bottom')
            else:
                hover_data['text_box'].xy = (event.xdata, event.ydata)
            
            hover_data['text_box'].set_text(info_text)
            hover_data['text_box'].set_visible(True)
            
            # Load and display tile images
            try:
                if tile_y in hover_data['tile_images'] and tile_x in hover_data['tile_images']:
                    img_y = hover_data['tile_images'][tile_y]
                    img_x = hover_data['tile_images'][tile_x]
                    
                    if img_y is not None and img_x is not None:
                        # Resize to thumbnails (80x80)
                        thumb_size = 80
                        img_y_thumb = cv2.resize(img_y, (thumb_size, thumb_size))
                        img_x_thumb = cv2.resize(img_x, (thumb_size, thumb_size))
                        
                        # Convert to RGB
                        img_y_rgb = cv2.cvtColor(img_y_thumb, cv2.COLOR_BGR2RGB)
                        img_x_rgb = cv2.cvtColor(img_x_thumb, cv2.COLOR_BGR2RGB)
                        
                        # Combine side by side
                        combined = np.hstack([img_y_rgb, img_x_rgb])
                        
                        # Create or update image display
                        if hover_data['image_display'] is None:
                            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
                            imagebox = OffsetImage(combined, zoom=0.8)
                            hover_data['image_display'] = AnnotationBbox(imagebox, 
                                                                          xy=(event.xdata, event.ydata),
                                                                          xybox=(10, -50),
                                                                          boxcoords='offset points',
                                                                          pad=0.5,
                                                                          frameon=True)
                            ax.add_artist(hover_data['image_display'])
                        else:
                            hover_data['image_display'].offsetbox.set_data(combined)
                            hover_data['image_display'].xy = (event.xdata, event.ydata)
                        
                        hover_data['image_display'].set_visible(True)
            except Exception as e:
                print(f"Error loading images for display: {e}")
            
            # Draw rectangle around cell
            if hover_data['rect'] is None:
                from matplotlib.patches import Rectangle
                hover_data['rect'] = Rectangle((x-0.5, y-0.5), 1, 1, 
                                               fill=False, edgecolor='black', linewidth=2)
                ax.add_patch(hover_data['rect'])
            else:
                hover_data['rect'].set_xy((x-0.5, y-0.5))
            
            hover_data['rect'].set_visible(True)
            fig.canvas.draw_idle()
        else:
            if hover_data['text_box']:
                hover_data['text_box'].set_visible(False)
            if hover_data['rect']:
                hover_data['rect'].set_visible(False)
            if hover_data['image_display']:
                hover_data['image_display'].set_visible(False)
            fig.canvas.draw_idle()
    
    # Connect hover event
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    
    # Create sorted similarity ranking window
    fig2, ax2 = plt.subplots(figsize=(16, 12))
    
    # Create sorted ranking matrix - for each column, sort rows by distance
    sorted_rankings = np.zeros((num_tiles, num_tiles))
    for col in range(num_tiles):
        # Get column and sort indices
        col_distances = normalized_matrix[:, col]
        sorted_indices = np.argsort(col_distances)  # Ascending order (best matches first)
        sorted_rankings[:, col] = sorted_indices
    
    # Create a new normalized matrix showing ranks colored by similarity
    rank_colored = np.zeros((num_tiles, num_tiles))
    for col in range(num_tiles):
        col_distances = normalized_matrix[:, col]
        sorted_indices = np.argsort(col_distances)
        for rank, idx in enumerate(sorted_indices):
            rank_colored[rank, col] = col_distances[idx]
    
    im2 = ax2.imshow(rank_colored, cmap='RdYlGn_r', aspect='auto')
    
    ax2.set_xlabel('Tile', fontsize=10)
    ax2.set_ylabel('Similarity Rank (0=most similar)', fontsize=10)
    ax2.set_title('Sorted Similarity Rankings per Tile', fontsize=12, fontweight='bold')
    
    # Set column labels (tile names)
    ax2.set_xticks(range(num_tiles))
    ax2.set_xticklabels(tile_names, fontsize=7)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Set row labels (rank positions)
    ax2.set_yticks(range(num_tiles))
    ax2.set_yticklabels(range(num_tiles), fontsize=8)
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Normalized Distance', rotation=270, labelpad=20)
    
    # Add grid
    ax2.set_xticks(np.arange(num_tiles)-.5, minor=True)
    ax2.set_yticks(np.arange(num_tiles)-.5, minor=True)
    ax2.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # Store data for hover interaction on second window
    hover_data2 = {
        'tile_names': tile_names,
        'distance_matrix': distance_matrix,
        'normalized_matrix': normalized_matrix,
        'rank_colored': rank_colored,
        'sorted_rankings': sorted_rankings,
        'tile_images': tile_images,
        'text_box': None,
        'rect': None,
        'image_display': None
    }
    
    # Define hover event handler for sorted rankings window
    def on_hover2(event):
        if event.inaxes != ax2:
            if hover_data2['text_box']:
                hover_data2['text_box'].set_visible(False)
            if hover_data2['rect']:
                hover_data2['rect'].set_visible(False)
            if hover_data2['image_display']:
                hover_data2['image_display'].set_visible(False)
            fig2.canvas.draw_idle()
            return
        
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        
        if 0 <= x < num_tiles and 0 <= y < num_tiles:
            tile_x = tile_names[x]  # Column tile
            ranked_idx = int(hover_data2['sorted_rankings'][y, x])  # Ranked tile index
            tile_y = tile_names[ranked_idx]
            
            raw_dist = distance_matrix[ranked_idx, x]
            norm_dist = normalized_matrix[ranked_idx, x]
            
            info_text = f"Query: {tile_x}\nRank {y}: {tile_y}\nDist: {raw_dist:.0f}\nNorm: {norm_dist:.3f}"
            
            # Create or update text box
            if hover_data2['text_box'] is None:
                hover_data2['text_box'] = ax2.annotate('', xy=(event.xdata, event.ydata),
                                                       xytext=(10, 10), textcoords='offset points',
                                                       fontsize=9,
                                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                                                       ha='left', va='bottom')
            else:
                hover_data2['text_box'].xy = (event.xdata, event.ydata)
            
            hover_data2['text_box'].set_text(info_text)
            hover_data2['text_box'].set_visible(True)
            
            # Load and display tile images
            try:
                if tile_x in hover_data2['tile_images'] and tile_y in hover_data2['tile_images']:
                    img_x = hover_data2['tile_images'][tile_x]
                    img_y = hover_data2['tile_images'][tile_y]
                    
                    if img_x is not None and img_y is not None:
                        # Resize to thumbnails (80x80)
                        thumb_size = 80
                        img_x_thumb = cv2.resize(img_x, (thumb_size, thumb_size))
                        img_y_thumb = cv2.resize(img_y, (thumb_size, thumb_size))
                        
                        # Convert to RGB
                        img_x_rgb = cv2.cvtColor(img_x_thumb, cv2.COLOR_BGR2RGB)
                        img_y_rgb = cv2.cvtColor(img_y_thumb, cv2.COLOR_BGR2RGB)
                        
                        # Combine side by side
                        combined = np.hstack([img_x_rgb, img_y_rgb])
                        
                        # Create or update image display
                        if hover_data2['image_display'] is None:
                            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
                            imagebox = OffsetImage(combined, zoom=0.8)
                            hover_data2['image_display'] = AnnotationBbox(imagebox, 
                                                                           xy=(event.xdata, event.ydata),
                                                                           xybox=(10, -50),
                                                                           boxcoords='offset points',
                                                                           pad=0.5,
                                                                           frameon=True)
                            ax2.add_artist(hover_data2['image_display'])
                        else:
                            hover_data2['image_display'].offsetbox.set_data(combined)
                            hover_data2['image_display'].xy = (event.xdata, event.ydata)
                        
                        hover_data2['image_display'].set_visible(True)
            except Exception as e:
                print(f"Error loading images for display: {e}")
            
            # Draw rectangle around cell
            if hover_data2['rect'] is None:
                from matplotlib.patches import Rectangle
                hover_data2['rect'] = Rectangle((x-0.5, y-0.5), 1, 1, 
                                               fill=False, edgecolor='black', linewidth=2)
                ax2.add_patch(hover_data2['rect'])
            else:
                hover_data2['rect'].set_xy((x-0.5, y-0.5))
            
            hover_data2['rect'].set_visible(True)
            fig2.canvas.draw_idle()
        else:
            if hover_data2['text_box']:
                hover_data2['text_box'].set_visible(False)
            if hover_data2['rect']:
                hover_data2['rect'].set_visible(False)
            if hover_data2['image_display']:
                hover_data2['image_display'].set_visible(False)
            fig2.canvas.draw_idle()
    
    # Connect hover event to second window
    fig2.canvas.mpl_connect('motion_notify_event', on_hover2)
    
    manager2 = plt.get_current_fig_manager()
    manager2.window.showMaximized()
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automatically detect and analyze tiles from video')
    parser.add_argument('--video', type=str, default='video.mp4', help='Video file path (default: video.mp4)')
    parser.add_argument('--start-frame', type=int, default=0, help='Start frame number (default: 0)')
    parser.add_argument('--num-frames', type=int, default=20, help='Number of consecutive frames to process (default: 20)')
    parser.add_argument('--blocks', type=int, default=16, help='Blocks per side for dhash (default: 16)')
    parser.add_argument('--clahe', action='store_true', default=True, help='Apply CLAHE preprocessing (default: True)')
    parser.add_argument('--no-clahe', dest='clahe', action='store_false', help='Disable CLAHE preprocessing')
    
    args = parser.parse_args()
    
    main(args.video, args.start_frame, args.num_frames, args.blocks, args.clahe)

