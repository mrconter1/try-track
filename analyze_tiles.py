import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import binascii
import argparse
from datetime import datetime
from scipy.fftpack import dct

def get_block_descriptor(block, mode='mean'):
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
    elif mode == 'dhash':
        return compute_dhash_byte(block)
    elif mode == 'phash':
        return compute_phash_byte(block)
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

def generate_tile_signature(image_path, blocks_per_side=8, descriptor_mode='mean'):
    """Generate block signature for a tile image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read {image_path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (256, 256))
    # Skip histogram equalization for now - might be causing false positives
    
    block_size = 256 // blocks_per_side
    signature = []
    
    for row in range(blocks_per_side):
        for col in range(blocks_per_side):
            y_start = row * block_size
            y_end = (row + 1) * block_size
            x_start = col * block_size
            x_end = (col + 1) * block_size
            
            block = resized[y_start:y_end, x_start:x_end]
            descriptor = get_block_descriptor(block, descriptor_mode)
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

def compute_distance(hash1, hash2, mode='manhattan'):
    """Compute distance using specified mode"""
    if mode == 'manhattan':
        return manhattan_distance(hash1, hash2)
    elif mode == 'euclidean':
        return euclidean_distance(hash1, hash2)
    else:
        raise ValueError(f"Unknown distance mode: {mode}")

def main(blocks_per_side=8, distance_mode='manhattan', descriptor_mode='mean'):
    export_folder = os.path.join(os.getcwd(), "export")
    
    # Read original metadata
    metadata_path = os.path.join(export_folder, "metadata.json")
    with open(metadata_path, 'r') as f:
        old_metadata = json.load(f)
    
    print(f"Generating tile signatures ({blocks_per_side}x{blocks_per_side} blocks, {descriptor_mode} descriptor, {distance_mode} distance)...")
    
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
            
            hash_str = generate_tile_signature(image_path, blocks_per_side, descriptor_mode)
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
    
    # Maximize window
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze tile image similarity')
    parser.add_argument('--blocks', type=int, default=8, help='Blocks per side (default: 8, so 8x8=64 blocks)')
    parser.add_argument('--distance', type=str, default='manhattan', choices=['manhattan', 'euclidean'], help='Distance metric (default: manhattan)')
    parser.add_argument('--descriptor', type=str, default='mean', choices=['mean', 'median', 'min', 'max', 'dhash', 'phash'], help='Block descriptor mode (default: mean)')
    args = parser.parse_args()
    
    main(blocks_per_side=args.blocks, distance_mode=args.distance, descriptor_mode=args.descriptor)

