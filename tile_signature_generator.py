import cv2
import os
import json
from datetime import datetime
import binascii

def generate_tile_signature(image_path):
    """
    Generate 8x8 block signature for a tile image.
    Returns: hash string of the signature
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to 256x256
    resized = cv2.resize(gray, (256, 256))
    
    # Normalize brightness/contrast
    equalized = cv2.equalizeHist(resized)
    
    # Create 8x8 blocks (256/8 = 32 pixels per block)
    block_size = 32
    signature = []
    
    for row in range(8):
        for col in range(8):
            y_start = row * block_size
            y_end = (row + 1) * block_size
            x_start = col * block_size
            x_end = (col + 1) * block_size
            
            block = equalized[y_start:y_end, x_start:x_end]
            avg_intensity = int(block.mean())
            signature.append(avg_intensity)
    
    # Convert signature to hex hash
    signature_bytes = bytes(signature)
    hash_str = binascii.hexlify(signature_bytes).decode('ascii')
    
    return hash_str

def main():
    export_folder = os.path.join(os.getcwd(), "export")
    
    # Read current metadata to get label mapping
    old_metadata_path = os.path.join(export_folder, "metadata.json")
    with open(old_metadata_path, 'r') as f:
        old_metadata = json.load(f)
    
    # Create new metadata structure
    new_metadata = {
        "metadata": {
            "creation_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "block_size": 8,
            "image_resolution": 256,
            "total_images": 0
        },
        "images": {}
    }
    
    # Process each image
    image_count = 0
    for label, image_files in old_metadata.items():
        for image_file in image_files:
            image_path = os.path.join(export_folder, image_file)
            
            if not os.path.exists(image_path):
                print(f"Image not found: {image_file}")
                continue
            
            # Generate signature/hash
            hash_str = generate_tile_signature(image_path)
            if hash_str is None:
                continue
            
            new_metadata["images"][image_file] = {
                "label": label,
                "hash": hash_str
            }
            
            image_count += 1
            print(f"Processed {image_file}: {hash_str[:16]}...")
    
    new_metadata["metadata"]["total_images"] = image_count
    
    # Save new metadata with different name
    new_metadata_path = os.path.join(export_folder, "metadata_signatures.json")
    with open(new_metadata_path, 'w') as f:
        json.dump(new_metadata, f, indent=2)
    
    print(f"\nSignature generation complete!")
    print(f"Processed {image_count} images")
    print(f"Metadata saved to {new_metadata_path}")

if __name__ == "__main__":
    main()

