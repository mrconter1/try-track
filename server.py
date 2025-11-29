from fastapi import FastAPI, UploadFile, File, Response, Form
import torch
import cv2
import numpy as np
import uvicorn
from io import BytesIO
import torch.nn as nn
import torchvision.models as models
import time
import json
import itertools

# Define the model class (must match your trained model)
class MobileUNet(nn.Module):
    """Mobile-optimized U-Net with MobileNetV2 backbone."""
    
    def __init__(self, pretrained=False):
        super().__init__()
        
        mobilenet = models.mobilenet_v2(weights=None)
        self.encoder = mobilenet.features
        
        self.up1 = nn.ConvTranspose2d(1280, 96, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(96 + 96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(96, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(32, 24, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(24 + 24, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.ConvTranspose2d(24, 16, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.final_up = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.out = nn.Sequential(
            nn.Conv2d(16, 1, 1),
        )
    
    def forward(self, x):
        skip_connections = []
        skip_indices = [1, 3, 6, 13]
        
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            if idx in skip_indices:
                skip_connections.append(x)
        
        x = self.up1(x)
        x = torch.cat([x, skip_connections[3]], dim=1)
        x = self.dec1(x)
        
        x = self.up2(x)
        x = torch.cat([x, skip_connections[2]], dim=1)
        x = self.dec2(x)
        
        x = self.up3(x)
        x = torch.cat([x, skip_connections[1]], dim=1)
        x = self.dec3(x)
        
        x = self.up4(x)
        x = torch.cat([x, skip_connections[0]], dim=1)
        x = self.dec4(x)
        
        x = self.final_up(x)
        x = self.out(x)
        
        return x

# --- Detection Logic (Moved from Client) ---

def find_peaks(heatmap, threshold):
    """Find local maxima above threshold."""
    thresh_val = int(threshold * 255)
    heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    _, binary = cv2.threshold(heatmap_uint8, thresh_val, 255, cv2.THRESH_BINARY)
    
    points = []
    if cv2.countNonZero(binary) > 0:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append((cx, cy))
    return points

def is_convex(pts):
    """Check if 4 points form a convex polygon."""
    center = np.mean(pts, axis=0)
    sorted_pts = sorted(pts, key=lambda p: np.arctan2(p[1]-center[1], p[0]-center[0]))
    
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    cp_signs = []
    for i in range(4):
        p1 = sorted_pts[i]
        p2 = sorted_pts[(i + 1) % 4]
        p3 = sorted_pts[(i + 2) % 4]
        cp = cross_product(p1, p2, p3)
        cp_signs.append(np.sign(cp))
        
    return all(s > 0 for s in cp_signs) or all(s < 0 for s in cp_signs), sorted_pts

def check_quad_constraints(quad_pts, all_points, margin=5):
    is_conv, ordered_pts = is_convex(quad_pts)
    if not is_conv:
        return False, []
        
    # Check angles
    for i in range(4):
        p1 = np.array(ordered_pts[i-1])
        p2 = np.array(ordered_pts[i])
        p3 = np.array(ordered_pts[(i+1)%4])
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        if l1 == 0 or l2 == 0: return False, []
        
        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)))
        if angle < 60 or angle > 120: # Relaxed Squarish check
            return False, []
            
    # Check aspect ratio
    side_lengths = []
    for i in range(4):
        p1 = np.array(ordered_pts[i])
        p2 = np.array(ordered_pts[(i+1)%4])
        side_lengths.append(np.linalg.norm(p1 - p2))
        
    max_side = max(side_lengths)
    min_side = min(side_lengths)
    ratio = max_side / (min_side + 1e-6)
    
    if ratio > 1.75: # Aspect ratio constraint
        return False, []

    # Check for points inside (simplified)
    poly_contour = np.array(ordered_pts, dtype=np.int32)
    for p in all_points:
        if any(np.array_equal(p, c) for c in quad_pts): continue
        dist = cv2.pointPolygonTest(poly_contour, (float(p[0]), float(p[1])), True)
        if dist > -margin:
            return False, []
            
    return True, ordered_pts

def find_quads(points):
    if len(points) < 4: return []
    search_points = points[:25] # Cap search space
    valid_quads = []
    for quad_combo in itertools.combinations(search_points, 4):
        is_valid, ordered_pts = check_quad_constraints(quad_combo, points)
        if is_valid:
            valid_quads.append(ordered_pts)
    return valid_quads

# --- Server Setup ---

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model once
model_path = "cross_net_224_best.pth"
try:
    model = MobileUNet(pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    dummy = torch.zeros(1, 3, 480, 640).to(device)
    model(dummy)
    print(f"Model {model_path} loaded and warmed up!")
except Exception as e:
    print(f"Error loading model: {e}")

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
request_count = 0

@app.get("/")
def read_root():
    return {"status": "running"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    mode: str = Form("heatmap"), 
    threshold: float = Form(0.5)
):
    global request_count
    t0 = time.time()
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None: return {"error": "Failed to decode"}
    
    # Preprocess
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Pad
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    
    if pad_h > 0 or pad_w > 0:
        img_padded = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        img_padded = img_rgb
        
    tensor = torch.from_numpy(img_padded.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor = (tensor - mean) / std
    
    # Inference
    with torch.no_grad():
        output = model(tensor)
        heatmap = output.squeeze().cpu().numpy()
    
    if pad_h > 0 or pad_w > 0:
        heatmap = heatmap[:h, :w]
    
    t_infer = time.time()

    # Output logic
    if mode == "heatmap":
        heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
        _, encoded = cv2.imencode('.jpg', heatmap_uint8, [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = encoded.tobytes()
        media_type = "image/jpeg"
    
    elif mode == "points":
        points = find_peaks(heatmap, threshold)
        data = json.dumps({"points": points}).encode('utf-8')
        media_type = "application/json"
        
    elif mode == "quads":
        points = find_peaks(heatmap, threshold)
        quads = find_quads(points)
        # Serialize as list of list of lists
        data = json.dumps({"points": points, "quads": quads}).encode('utf-8')
        media_type = "application/json"
        
    else:
        return {"error": "Unknown mode"}

    t_end = time.time()
    
    request_count += 1
    if request_count % 30 == 0:
        total_ms = (t_end - t0) * 1000
        print(f"Server ({mode}): Total {total_ms:.1f}ms")
        
    return Response(content=data, media_type=media_type)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1111)
