import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def hex_to_bytes(hex_str):
    """Convert hex string to array of byte values"""
    return [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]

def manhattan_distance(hash1, hash2):
    """Compute Manhattan distance between two hashes"""
    bytes1 = hex_to_bytes(hash1)
    bytes2 = hex_to_bytes(hash2)
    return sum(abs(b1 - b2) for b1, b2 in zip(bytes1, bytes2))

def main():
    # Load metadata
    export_folder = os.path.join(os.getcwd(), "export")
    metadata_path = os.path.join(export_folder, "metadata_signatures.json")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract image names and hashes
    images = metadata["images"]
    image_names = sorted(images.keys(), key=lambda x: int(x.split('.')[0]))  # Sort by number
    hashes = [images[name]["hash"] for name in image_names]
    labels = [images[name]["label"] for name in image_names]
    
    num_images = len(image_names)
    print(f"Processing {num_images} images...")
    
    # Create distance matrix
    distance_matrix = np.zeros((num_images, num_images))
    
    print("Computing pairwise distances...")
    for i in range(num_images):
        for j in range(num_images):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                distance_matrix[i][j] = manhattan_distance(hashes[i], hashes[j])
    
    # Normalize to 0-1
    min_dist = np.min(distance_matrix[distance_matrix > 0])  # Exclude diagonal zeros
    max_dist = np.max(distance_matrix)
    normalized_matrix = (distance_matrix - min_dist) / (max_dist - min_dist)
    
    print(f"Distance range: {min_dist:.0f} - {max_dist:.0f}")
    print(f"Normalized range: {normalized_matrix.min():.3f} - {normalized_matrix.max():.3f}")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Use 'RdYlGn_r' colormap (red=dissimilar, green=similar)
    im = ax.imshow(normalized_matrix, cmap='RdYlGn_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(num_images))
    ax.set_yticks(range(num_images))
    
    # Use image names as labels (remove .png extension)
    tick_labels = [name.replace('.png', '') for name in image_names]
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    
    # Rotate x labels for readability
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
            label = labels[j]
            # Use white text on dark backgrounds (distance > 0.5), black on light
            text_color = 'white' if normalized_matrix[i, j] > 0.5 else 'black'
            ax.text(j, i, label, ha="center", va="center", color=text_color, fontsize=6, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

