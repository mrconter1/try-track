from fastapi import FastAPI, UploadFile, File, Response
import torch
import cv2
import numpy as np
import uvicorn
from io import BytesIO
import torch.nn as nn
import torchvision.models as models
import time

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
            # No sigmoid here - model trained with MSE loss outputs 0-1 directly
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

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model once
try:
    model = MobileUNet(pretrained=False).to(device)
    # Load with map_location to handle GPU loading
    checkpoint = torch.load("cross_net_v7m_best.pth", map_location=device)
    
    # Handle state dict structure
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    # Warmup
    dummy = torch.zeros(1, 3, 480, 640).to(device)
    model(dummy)
    print("Model loaded and warmed up!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure 'cross_net_v7m_best.pth' is in the same directory.")

# Normalization constants
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

request_count = 0

@app.get("/")
def read_root():
    return {"status": "running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global request_count
    t0 = time.time()
    
    # Read bytes
    contents = await file.read()
    t1 = time.time()
    
    # Decode
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Failed to decode image"}
    
    t2 = time.time()
    
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
        
    # To Tensor
    tensor = torch.from_numpy(img_padded.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor = (tensor - mean) / std
    
    t3 = time.time()
    
    # Inference
    with torch.no_grad():
        output = model(tensor)
        heatmap = output.squeeze().cpu().numpy()
    
    t4 = time.time()
    
    # Post-process
    if pad_h > 0 or pad_w > 0:
        heatmap = heatmap[:h, :w]
    
    # Compress response: Float 0-1 -> Uint8 0-255 -> JPG
    heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    # Use quality 80 for good balance of speed/size/quality for heatmap
    _, heatmap_encoded = cv2.imencode('.jpg', heatmap_uint8, [cv2.IMWRITE_JPEG_QUALITY, 80])
    
    t5 = time.time()
    
    # Stats
    request_count += 1
    if request_count % 30 == 0:
        read_ms = (t1 - t0) * 1000
        decode_ms = (t2 - t1) * 1000
        prep_ms = (t3 - t2) * 1000
        infer_ms = (t4 - t3) * 1000
        pack_ms = (t5 - t4) * 1000
        total_ms = (t5 - t0) * 1000
        print(f"Server: Read {read_ms:.1f} | Dec {decode_ms:.1f} | Prep {prep_ms:.1f} | Infer {infer_ms:.1f} | Pack {pack_ms:.1f} | Total {total_ms:.1f}ms")
    
    return Response(content=heatmap_encoded.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1111)
