import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import argparse
import time

class GridDataset(Dataset):
    def __init__(self, state_file, img_size=256):
        self.img_size = img_size
        self.data = []
        
        if not os.path.exists(state_file):
            print(f"State file not found: {state_file}")
            return

        with open(state_file, 'r') as f:
            state = json.load(f)
            
        # Parse frame_grid_config
        if 'frame_grid_config' in state:
            for key_str, value in state['frame_grid_config'].items():
                # key_str is "path/to/video.mp4,frame_idx"
                try:
                    parts = key_str.rsplit(',', 1)
                    if len(parts) == 2:
                        video_path, frame_idx_str = parts
                        frame_idx = int(frame_idx_str)
                        
                        # value is [points, subdiv_x, subdiv_y, has_grid]
                        # points is list of [x, y] or None
                        points_raw = value[0]
                        subdiv_x = value[1]
                        subdiv_y = value[2]
                        has_grid = value[3] if len(value) > 3 else False
                        
                        if has_grid:
                            # Ensure we have 4 valid points
                            if len(points_raw) == 4 and all(p is not None for p in points_raw):
                                points = [tuple(p) for p in points_raw]
                                self.data.append({
                                    'video_path': video_path,
                                    'frame_idx': frame_idx,
                                    'points': points,
                                    'subdiv_x': subdiv_x,
                                    'subdiv_y': subdiv_y,
                                    'has_grid': 1.0
                                })
                        else:
                            # Negative sample
                            self.data.append({
                                'video_path': video_path,
                                'frame_idx': frame_idx,
                                'has_grid': 0.0
                            })
                except Exception as e:
                    print(f"Error parsing entry {key_str}: {e}")

        print(f"Loaded {len(self.data)} labeled frames.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_path = item['video_path']
        frame_idx = item['frame_idx']
        
        # Read frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Return a black frame if reading fails
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            target = np.zeros(7, dtype=np.float32) # Conf=0
            return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(target)

        h_orig, w_orig = frame.shape[:2]
        
        # Resize image
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL and Apply Augmentation
        from PIL import Image
        img_pil = Image.fromarray(img)
        
        # Augmentation: Random brightness, contrast, saturation
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(), # divides by 255
            # transforms.Normalize(...) # skipped to keep compatible with simple inference in tool
        ])
        
        img = transform(img_pil)
        
        # Prepare target: [Confidence, OriginX, OriginY, VecUX, VecUY, VecVX, VecVY]
        target = np.zeros(7, dtype=np.float32)
        
        if item['has_grid'] > 0.5:
            target[0] = 1.0 # Confidence
            
            # Calculate Anchor
            try:
                p1, p2, p3, p4 = item['points']
                sub_x = item['subdiv_x']
                sub_y = item['subdiv_y']
                
                src_pts = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
                dst_pts = np.float32([p1, p2, p3, p4])
                
                H = cv2.getPerspectiveTransform(src_pts, dst_pts)
                
                # Center in pixel coords
                center_img = np.array([[[w_orig / 2.0, h_orig / 2.0]]], dtype=np.float32)
                
                H_inv = np.linalg.inv(H)
                center_logical = cv2.perspectiveTransform(center_img, H_inv)[0][0]
                
                grid_gx = center_logical[0] * sub_x
                grid_gy = center_logical[1] * sub_y
                nearest_gx = round(grid_gx)
                nearest_gy = round(grid_gy)
                
                origin_logical = np.array([[[nearest_gx / sub_x, nearest_gy / sub_y]]], dtype=np.float32)
                basis_u_logical = np.array([[[ (nearest_gx + 1) / sub_x, nearest_gy / sub_y ]]], dtype=np.float32)
                basis_v_logical = np.array([[[ nearest_gx / sub_x, (nearest_gy + 1) / sub_y ]]], dtype=np.float32)
                
                origin_px = cv2.perspectiveTransform(origin_logical, H)[0][0]
                basis_u_px = cv2.perspectiveTransform(basis_u_logical, H)[0][0]
                basis_v_px = cv2.perspectiveTransform(basis_v_logical, H)[0][0]
                
                vec_u = basis_u_px - origin_px
                vec_v = basis_v_px - origin_px
                
                # Normalize to 0-1 range
                target[1] = origin_px[0] / w_orig
                target[2] = origin_px[1] / h_orig
                target[3] = vec_u[0] / w_orig
                target[4] = vec_u[1] / h_orig
                target[5] = vec_v[0] / w_orig
                target[6] = vec_v[1] / h_orig
                
            except Exception as e:
                print(f"Error calculating anchor for {video_path} frame {frame_idx}: {e}")
                target[0] = 0.0 # Mark as invalid/no grid if calculation fails

        return img, torch.from_numpy(target)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    state_file = os.path.join(os.path.expanduser("~"), ".grid_tool_state.json")
    dataset = GridDataset(state_file)
    
    if len(dataset) == 0:
        print("No data found. Label some frames first!")
        return

    # Simple split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    # val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0) # Optional
    
    # Model: MobileNetV3 Small
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # Modify classifier
    # The last layer is model.classifier[3]
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 7) # 1 conf + 6 coords
    
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Loss function
    # Target: [Conf, Ox, Oy, Ux, Uy, Vx, Vy]
    # Pred:   [Conf, Ox, Oy, Ux, Uy, Vx, Vy]
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss(reduction='none')
    
    num_epochs = 50
    print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            
            # Split outputs
            pred_conf = outputs[:, 0]
            pred_coords = outputs[:, 1:]
            
            true_conf = targets[:, 0]
            true_coords = targets[:, 1:]
            
            # 1. Confidence Loss (Classification)
            loss_conf = bce_loss(pred_conf, true_conf)
            
            # 2. Coordinate Loss (Regression) - only for frames with grid
            loss_coords_raw = mse_loss(pred_coords, true_coords)
            
            # Mask: Only penalize coords if true_conf is 1
            mask = true_conf.unsqueeze(1).expand_as(loss_coords_raw)
            loss_coords = (loss_coords_raw * mask).sum() / (mask.sum() + 1e-6)
            
            loss = loss_conf + 10.0 * loss_coords # Weight coords more
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "grid_model.pth")
    print("Training finished. Model saved to grid_model.pth")

if __name__ == "__main__":
    train()

