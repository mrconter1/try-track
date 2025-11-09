import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import binascii
import argparse
from datetime import datetime
from scipy.fftpack import dct

def get_block_descriptor(block, mode='mean', blob_threshold=None, blob_min_size=1):
    """Compute block descriptor based on mode"""
    flat = block.flatten()
    
    if mode == 'mean':
        return int(np.mean(flat))
    elif mode == 'median':
        return int(np.median(flat))
    elif mode == 'min':
        return int(np.min(flat))
    elif mode == 'max':
        return int(np.max(flat))
    elif mode == 'std_dev':
        return int(np.std(flat))
    elif mode == 'dhash':
        return compute_dhash_byte(block)
    elif mode == 'phash':
        return compute_phash_byte(block)
    elif mode == 'blob_count':
        return compute_blob_count(block, blob_threshold, blob_min_size)
    elif mode == 'lbp':
        return compute_lbp_byte(block)
    else:
        raise ValueError(f"Unknown descriptor mode: {mode}")

def compute_dhash_byte(block):
    """
    Compute perceptual dHash for a block.
    Returns a single byte (0-255) representing difference patterns.
    """
    h, w = block.shape
    
    # Split block into quadrants
    h_mid = h // 2
    w_mid = w // 2
    
    q1 = block[:h_mid, :w_mid].mean()  # top-left
    q2 = block[:h_mid, w_mid:].mean()  # top-right
    q3 = block[h_mid:, :w_mid].mean()  # bottom-left
    q4 = block[h_mid:, w_mid:].mean()  # bottom-right
    
    # Compute 8 bits from different comparisons
    bits = []
    bits.append(1 if q1 > q2 else 0)       # Bit 0: left > right (top half)
    bits.append(1 if q3 > q4 else 0)       # Bit 1: left > right (bottom half)
    bits.append(1 if q1 > q3 else 0)       # Bit 2: top > bottom (left half)
    bits.append(1 if q2 > q4 else 0)       # Bit 3: top > bottom (right half)
    bits.append(1 if q1 > q4 else 0)       # Bit 4: top-left > bottom-right
    bits.append(1 if q2 > q3 else 0)       # Bit 5: top-right > bottom-left
    bits.append(1 if (q1 + q4) > (q2 + q3) else 0)  # Bit 6: diagonal1 > diagonal2
    bits.append(1 if block.mean() > 128 else 0)      # Bit 7: bright or dark
    
    # Convert 8 bits to byte value (0-255)
    byte_val = 0
    for i, bit in enumerate(bits):
        byte_val += bit * (2 ** i)
    
    return byte_val

def compute_phash_byte(block):
    """
    Compute perceptual hash (pHash) using DCT for a block.
    Returns a single byte (0-255) representing frequency content.
    """
    # Normalize block to float
    block_float = block.astype(np.float32)
    
    # Compute 2D DCT
    dct_2d = dct(dct(block_float.T, norm='ortho').T, norm='ortho')
    
    # Extract low-frequency region (top-left 4x4)
    low_freq = dct_2d[:4, :4]
    
    # Compute bits from low-frequency coefficients
    # Flatten and compare each to average
    low_freq_flat = low_freq.flatten()
    avg_coeff = np.mean(low_freq_flat)
    
    bits = []
    for coeff in low_freq_flat[:8]:  # Use first 8 coefficients
        bits.append(1 if coeff > avg_coeff else 0)
    
    # Convert 8 bits to byte value (0-255)
    byte_val = 0
    for i, bit in enumerate(bits):
        byte_val += bit * (2 ** i)
    
    return byte_val

def compute_blob_count(block, blob_threshold=None, blob_min_size=1):
    """
    Count number of blobs/connected components in a block.
    If blob_threshold is None, uses Otsu's adaptive threshold.
    If blob_threshold is a number (0-255), uses that fixed threshold.
    Filters out blobs smaller than blob_min_size pixels.
    Returns blob count (0-255).
    """
    if blob_threshold is None:
        # Use Otsu's automatic threshold
        _, binary = cv2.threshold(block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Use fixed threshold
        _, binary = cv2.threshold(block, blob_threshold, 255, cv2.THRESH_BINARY)
    
    # Find connected components with labels
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    
    # Count blobs, filtering by minimum size
    blob_count = 0
    for label_id in range(1, num_labels):  # Skip label 0 (background)
        # Count pixels with this label
        blob_size = np.sum(labels == label_id)
        if blob_size >= blob_min_size:
            blob_count += 1
    
    # Cap at 255 for byte value
    return min(blob_count, 255)

def compute_lbp_byte(block):
    """
    Compute Local Binary Pattern (LBP) descriptor for a block.
    LBP captures local texture by comparing each pixel to its 8 neighbors.
    Returns a single byte (0-255) representing the dominant LBP pattern.
    """
    h, w = block.shape
    lbp_values = []
    
    # Compute LBP for interior pixels (excluding border)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            center = block[y, x]
            
            # Get 8 neighbors in order (clockwise from top)
            neighbors = [
                block[y-1, x-1],  # top-left
                block[y-1, x],    # top
                block[y-1, x+1],  # top-right
                block[y, x+1],    # right
                block[y+1, x+1],  # bottom-right
                block[y+1, x],    # bottom
                block[y+1, x-1],  # bottom-left
                block[y, x-1],    # left
            ]
            
            # Create 8-bit pattern: 1 if neighbor >= center, else 0
            lbp_bits = [1 if neighbor >= center else 0 for neighbor in neighbors]
            
            # Convert to byte value (0-255)
            lbp_byte = 0
            for i, bit in enumerate(lbp_bits):
                lbp_byte += bit * (2 ** i)
            
            lbp_values.append(lbp_byte)
    
    # Return the median LBP value as representative
    if lbp_values:
        return int(np.median(lbp_values))
    else:
        return 0

def generate_tile_signature(image_path, blocks_per_side=8, descriptor_mode='mean', blob_threshold=None, blob_min_size=1, use_blur=False, use_equalize=False, use_contrast_stretch=False, use_clahe=False):
    """Generate block signature for a tile image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read {image_path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (256, 256))
    
    # Apply preprocessing options
    processed = resized.copy()
    
    if use_blur:
        # Apply Gaussian blur to reduce noise/grain
        processed = cv2.GaussianBlur(processed, (5, 5), 1.0)
    
    if use_equalize:
        # Apply histogram equalization for better contrast
        processed = cv2.equalizeHist(processed)
    
    if use_contrast_stretch:
        # Apply contrast stretching (normalize to full 0-255 range)
        p2, p98 = np.percentile(processed, (2, 98))
        processed = np.clip((processed - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    
    if use_clahe:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Better than simple histogram equalization - preserves local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)
    
    block_size = 256 // blocks_per_side
    signature = []
    
    for row in range(blocks_per_side):
        for col in range(blocks_per_side):
            y_start = row * block_size
            y_end = (row + 1) * block_size
            x_start = col * block_size
            x_end = (col + 1) * block_size
            
            block = processed[y_start:y_end, x_start:x_end]
            descriptor = get_block_descriptor(block, descriptor_mode, blob_threshold, blob_min_size)
            signature.append(descriptor)
    
    signature_bytes = bytes(signature)
    hash_str = binascii.hexlify(signature_bytes).decode('ascii')
    
    return hash_str

def hex_to_bytes(hex_str):
    """Convert hex string to array of byte values"""
    return [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]

def manhattan_distance(hash1, hash2):
    """Compute Manhattan distance between two hashes"""
    bytes1 = hex_to_bytes(hash1)
    bytes2 = hex_to_bytes(hash2)
    return sum(abs(b1 - b2) for b1, b2 in zip(bytes1, bytes2))

def euclidean_distance(hash1, hash2):
    """Compute Euclidean distance between two hashes"""
    bytes1 = hex_to_bytes(hash1)
    bytes2 = hex_to_bytes(hash2)
    return np.sqrt(sum((b1 - b2)**2 for b1, b2 in zip(bytes1, bytes2)))

def chebyshev_distance(hash1, hash2):
    """Compute Chebyshev distance between two hashes (max absolute difference)"""
    bytes1 = hex_to_bytes(hash1)
    bytes2 = hex_to_bytes(hash2)
    return max(abs(b1 - b2) for b1, b2 in zip(bytes1, bytes2))

def compute_distance(hash1, hash2, mode='manhattan'):
    """Compute distance using specified mode"""
    if mode == 'manhattan':
        return manhattan_distance(hash1, hash2)
    elif mode == 'euclidean':
        return euclidean_distance(hash1, hash2)
    elif mode == 'chebyshev':
        return chebyshev_distance(hash1, hash2)
    else:
        raise ValueError(f"Unknown distance mode: {mode}")

def main(blocks_per_side=8, distance_mode='manhattan', descriptor_mode='mean', blob_threshold=None, blob_min_size=1, use_blur=False, use_equalize=False, use_contrast_stretch=False, use_clahe=False):
    export_folder = os.path.join(os.getcwd(), "export")
    
    # Read original metadata
    metadata_path = os.path.join(export_folder, "metadata.json")
    with open(metadata_path, 'r') as f:
        old_metadata = json.load(f)
    
    threshold_str = f", threshold={blob_threshold}" if blob_threshold is not None else ""
    min_size_str = f", min_size={blob_min_size}" if blob_min_size > 1 else ""
    preproc_str = []
    if use_blur:
        preproc_str.append("blur")
    if use_equalize:
        preproc_str.append("equalize")
    if use_contrast_stretch:
        preproc_str.append("contrast-stretch")
    if use_clahe:
        preproc_str.append("clahe")
    preproc_info = f", preproc=[{', '.join(preproc_str)}]" if preproc_str else ""
    print(f"Generating tile signatures ({blocks_per_side}x{blocks_per_side} blocks, {descriptor_mode} descriptor{threshold_str}{min_size_str}{preproc_info}, {distance_mode} distance)...")
    
    # Generate hashes for all images
    image_hashes = {}
    image_labels = {}
    image_files = []
    
    for label, image_files_list in old_metadata.items():
        for image_file in image_files_list:
            image_path = os.path.join(export_folder, image_file)
            
            if not os.path.exists(image_path):
                print(f"Image not found: {image_file}")
                continue
            
            hash_str = generate_tile_signature(image_path, blocks_per_side, descriptor_mode, blob_threshold, blob_min_size, use_blur, use_equalize, use_contrast_stretch, use_clahe)
            if hash_str is None:
                continue
            
            image_hashes[image_file] = hash_str
            image_labels[image_file] = label
            image_files.append(image_file)
            print(f"  {image_file}: {hash_str[:16]}...")
    
    # Sort images by number
    image_files.sort(key=lambda x: int(x.split('.')[0]))
    
    num_images = len(image_files)
    print(f"\nProcessed {num_images} images")
    
    # Create distance matrix
    print("Computing pairwise distances...")
    distance_matrix = np.zeros((num_images, num_images))
    
    for i in range(num_images):
        for j in range(num_images):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                distance_matrix[i][j] = compute_distance(image_hashes[image_files[i]], image_hashes[image_files[j]], distance_mode)
    
    # Normalize to 0-1
    min_dist = np.min(distance_matrix[distance_matrix > 0])
    max_dist = np.max(distance_matrix)
    normalized_matrix = (distance_matrix - min_dist) / (max_dist - min_dist)
    
    print(f"Distance range: {min_dist:.0f} - {max_dist:.0f}")
    print(f"Normalized range: {normalized_matrix.min():.3f} - {normalized_matrix.max():.3f}")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    im = ax.imshow(normalized_matrix, cmap='RdYlGn_r', aspect='auto')
    
    # Store data for hover interaction
    hover_data = {
        'image_files': image_files,
        'image_labels': image_labels,
        'distance_matrix': distance_matrix,
        'normalized_matrix': normalized_matrix,
        'text_box': None,
        'rect': None,
        'image_display': None,
        'export_folder': export_folder
    }
    
    # Set ticks and labels
    ax.set_xticks(range(num_images))
    ax.set_yticks(range(num_images))
    
    tick_labels = [f"{name.replace('.png', '')} ({image_labels[name]})" for name in image_files]
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Distance (0=similar, 1=different)', rotation=270, labelpad=20)
    
    # Labels
    ax.set_xlabel('Image', fontsize=10)
    ax.set_ylabel('Image', fontsize=10)
    ax.set_title('Tile Image Similarity Matrix (Manhattan Distance)', fontsize=12, fontweight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(num_images)-.5, minor=True)
    ax.set_yticks(np.arange(num_images)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # Define hover event handler
    def on_hover(event):
        if event.inaxes != ax:
            # Mouse not over axes
            if hover_data['text_box']:
                hover_data['text_box'].set_visible(False)
            if hover_data['rect']:
                hover_data['rect'].set_visible(False)
            if hover_data['image_display']:
                hover_data['image_display'].set_visible(False)
            fig.canvas.draw_idle()
            return
        
        # Get current cursor position
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        
        # Check if within bounds
        if 0 <= x < num_images and 0 <= y < num_images:
            # Get image information
            img_y = image_files[y]
            img_x = image_files[x]
            label_y = image_labels[img_y]
            label_x = image_labels[img_x]
            
            # Get distance values
            raw_dist = distance_matrix[y, x]
            norm_dist = normalized_matrix[y, x]
            
            # Create info text
            info_text = f"Y: {img_y} ({label_y})\nX: {img_x} ({label_x})\nDist: {raw_dist:.0f}\nNorm: {norm_dist:.3f}"
            
            # Create or update text box - position near mouse using annotate
            if hover_data['text_box'] is None:
                hover_data['text_box'] = ax.annotate('', xy=(event.xdata, event.ydata),
                                                      xytext=(10, 10), textcoords='offset points',
                                                      fontsize=9,
                                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                                                      ha='left', va='bottom')
            else:
                # Update text box position to follow mouse
                hover_data['text_box'].xy = (event.xdata, event.ydata)
            
            hover_data['text_box'].set_text(info_text)
            hover_data['text_box'].set_visible(True)
            
            # Load and display tile images
            try:
                img_y_path = os.path.join(hover_data['export_folder'], img_y)
                img_x_path = os.path.join(hover_data['export_folder'], img_x)
                
                if os.path.exists(img_y_path) and os.path.exists(img_x_path):
                    # Load images
                    img_y_cv = cv2.imread(img_y_path)
                    img_x_cv = cv2.imread(img_x_path)
                    
                    if img_y_cv is not None and img_x_cv is not None:
                        # Resize to thumbnails (80x80)
                        thumb_size = 80
                        img_y_thumb = cv2.resize(img_y_cv, (thumb_size, thumb_size))
                        img_x_thumb = cv2.resize(img_x_cv, (thumb_size, thumb_size))
                        
                        # Convert to RGB
                        img_y_rgb = cv2.cvtColor(img_y_thumb, cv2.COLOR_BGR2RGB)
                        img_x_rgb = cv2.cvtColor(img_x_thumb, cv2.COLOR_BGR2RGB)
                        
                        # Combine side by side with labels
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
                            # Update image display
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
            # Outside grid
            if hover_data['text_box']:
                hover_data['text_box'].set_visible(False)
            if hover_data['rect']:
                hover_data['rect'].set_visible(False)
            if hover_data['image_display']:
                hover_data['image_display'].set_visible(False)
            fig.canvas.draw_idle()
    
    # Connect hover event
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    # Maximize window
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze tile image similarity')
    parser.add_argument('--blocks', type=int, default=8, help='Blocks per side (default: 8, so 8x8=64 blocks)')
    parser.add_argument('--distance', type=str, default='manhattan', choices=['manhattan', 'euclidean', 'chebyshev'], help='Distance metric (default: manhattan)')
    parser.add_argument('--descriptor', type=str, default='mean', choices=['mean', 'median', 'min', 'max', 'std_dev', 'dhash', 'phash', 'blob_count', 'lbp'], help='Block descriptor mode (default: mean)')
    parser.add_argument('--blob-threshold', type=int, default=None, help='Threshold for blob detection (0-255, default: Otsu adaptive)')
    parser.add_argument('--blob-min-size', type=int, default=1, help='Minimum blob size in pixels (default: 1, filters noise)')
    parser.add_argument('--blur', action='store_true', help='Apply Gaussian blur for noise reduction')
    parser.add_argument('--equalize', action='store_true', help='Apply histogram equalization')
    parser.add_argument('--contrast-stretch', action='store_true', help='Apply contrast stretching')
    parser.add_argument('--clahe', action='store_true', help='Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) - better for lighting variations')
    args = parser.parse_args()
    
    main(blocks_per_side=args.blocks, distance_mode=args.distance, descriptor_mode=args.descriptor, blob_threshold=args.blob_threshold, blob_min_size=args.blob_min_size, use_blur=args.blur, use_equalize=args.equalize, use_contrast_stretch=args.contrast_stretch, use_clahe=args.clahe)

