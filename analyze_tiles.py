import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import binascii
from datetime import datetime

def generate_tile_signature(image_path):
    """Generate 8x8 block signature for a tile image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read {image_path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (256, 256))
    # Skip histogram equalization for now - might be causing false positives
    
    block_size = 32
    signature = []
    
    for row in range(8):
        for col in range(8):
            y_start = row * block_size
            y_end = (row + 1) * block_size
            x_start = col * block_size
            x_end = (col + 1) * block_size
            
            block = resized[y_start:y_end, x_start:x_end]
            avg_intensity = int(block.mean())
            signature.append(avg_intensity)
    
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

def main():
    export_folder = os.path.join(os.getcwd(), "export")
    
    # Read original metadata
    metadata_path = os.path.join(export_folder, "metadata.json")
    with open(metadata_path, 'r') as f:
        old_metadata = json.load(f)
    
    print("Generating tile signatures...")
    
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
            
            hash_str = generate_tile_signature(image_path)
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
                distance_matrix[i][j] = manhattan_distance(image_hashes[image_files[i]], image_hashes[image_files[j]])
    
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
    
    tick_labels = [name.replace('.png', '') for name in image_files]
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
    
    # Add label text to each cell
    for i in range(num_images):
        for j in range(num_images):
            label = image_labels[image_files[j]]
            text_color = 'white' if normalized_matrix[i, j] > 0.5 else 'black'
            ax.text(j, i, label, ha="center", va="center", color=text_color, fontsize=6, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

