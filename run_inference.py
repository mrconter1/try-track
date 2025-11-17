import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# Re-define the model architecture class here so the script is self-contained
class CrossDetectorModel(nn.Module):
    """Mobile-friendly cross detection model."""
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        self.head = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

def run_inference(video_path, model_path, frame_number, stride, threshold):
    """Load a model and run inference on a specific video frame."""
    
    # --- 1. Load Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    model = CrossDetectorModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"[Error] Model file not found at '{model_path}'. Please train a model first.")
        return
    
    model.eval()
    print(f"[Info] Model '{model_path}' loaded successfully.")

    # --- 2. Load Frame ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video file: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number >= total_frames:
        print(f"[Error] Invalid frame number. Video has {total_frames} frames (0-indexed).")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"[Error] Failed to read frame {frame_number}.")
        return
    cap.release()
    print(f"[Info] Loaded frame {frame_number} from '{video_path}'.")

    # --- 3. Create Overlapping Tiles ---
    img_h, img_w = frame.shape[:2]
    tile_size = 128
    tiles = []
    tile_coords = []

    for y in range(0, img_h - tile_size + 1, stride):
        for x in range(0, img_w - tile_size + 1, stride):
            tile = frame[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            tile_coords.append((x, y))
    
    if not tiles:
        print("[Error] Frame is smaller than tile size (128x128). Cannot run inference.")
        return

    print(f"[Info] Generated {len(tiles)} overlapping tiles.")

    # --- 4. Pre-process Tiles and Run Inference ---
    batch = []
    for tile in tiles:
        img = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        batch.append(torch.from_numpy(img))
    
    batch_tensor = torch.stack(batch).to(device)

    with torch.no_grad():
        outputs = model(batch_tensor)
    
    # --- 5. Post-process Predictions ---
    detections = []
    scores = torch.sigmoid(outputs[:, 0])

    for i in range(len(scores)):
        if scores[i] > threshold:
            tile_x, tile_y = tile_coords[i]
            
            # Get coordinates within the tile (normalized)
            pred_x_norm, pred_y_norm = outputs[i, 1:].cpu().numpy()
            
            # Convert to pixel coordinates within the tile
            pred_x_tile = pred_x_norm * tile_size
            pred_y_tile = pred_y_norm * tile_size
            
            # Convert to global coordinates on the full frame
            global_x = tile_x + pred_x_tile
            global_y = tile_y + pred_y_tile
            
            detections.append((global_x, global_y, scores[i]))

    print(f"[Info] Found {len(detections)} potential crosses above threshold {threshold}.")

    # --- Optional: Non-Maximum Suppression (if you have many overlapping detections) ---
    # (Skipping for now for simplicity, but could be added if needed)

    # --- 6. Visualize Detections ---
    output_image = frame.copy()
    for x, y, score in detections:
        # Draw a crosshair
        px, py = int(x), int(y)
        color = (0, 255, 0) # Green
        size = 15
        thickness = 2
        cv2.line(output_image, (px - size, py), (px + size, py), color, thickness)
        cv2.line(output_image, (px, py - size), (px, py + size), color, thickness)
        
        # You can also add the score text if you want
        # cv2.putText(output_image, f"{score:.2f}", (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    window_name = f"Detections on Frame {frame_number}"
    cv2.imshow(window_name, output_image)
    print("[Info] Displaying results. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Run cross detection inference on a video frame.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--model", type=str, default="cross_detector_best.pth", help="Path to the trained model .pth file.")
    parser.add_argument("--frame", type=int, required=True, help="The frame number to process.")
    parser.add_argument("--stride", type=int, default=64, help="Stride for overlapping tiles. Smaller stride = more detections.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Confidence threshold for a detection to be considered valid (0.0 to 1.0).")
    
    args = parser.parse_args()

    run_inference(args.video, args.model, args.frame, args.stride, args.threshold)

if __name__ == "__main__":
    main()
