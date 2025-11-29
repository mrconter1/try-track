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
import argparse
import os
import math
import glob

try:
    from turbojpeg import TurboJPEG
    # Try to find library automatically
    try:
        jpeg = TurboJPEG()
        has_turbojpeg = True
    except RuntimeError:
        # Try common paths on Ubuntu/Debian
        possible_paths = [
            '/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0',
            '/usr/lib/libturbojpeg.so.0',
            '/usr/lib64/libturbojpeg.so.0'
        ]
        has_turbojpeg = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    jpeg = TurboJPEG(path)
                    has_turbojpeg = True
                    break
                except: pass
                
    if has_turbojpeg:
        print("Using TurboJPEG for fast decoding")
    else:
        print("TurboJPEG library not found, using OpenCV")
except ImportError:
    has_turbojpeg = False
    print("TurboJPEG python package not found, using OpenCV")

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

# --- Detection Logic ---

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
    # Pure python implementation for speed
    # pts is list of (x,y)
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    
    # Sort by angle
    # math.atan2 is faster than np.arctan2 for single scalars
    sorted_pts = sorted(pts, key=lambda p: math.atan2(p[1]-cy, p[0]-cx))
    
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Check cross product signs
    p0, p1, p2, p3 = sorted_pts
    cp1 = cross_product(p0, p1, p2)
    cp2 = cross_product(p1, p2, p3)
    cp3 = cross_product(p2, p3, p0)
    cp4 = cross_product(p3, p0, p1)
    
    if (cp1 > 0 and cp2 > 0 and cp3 > 0 and cp4 > 0) or \
       (cp1 < 0 and cp2 < 0 and cp3 < 0 and cp4 < 0):
        return True, sorted_pts
    return False, []

def check_quad_constraints(quad_pts, all_points, margin=5):
    # Pure Python implementation
    is_conv, ordered_pts = is_convex(quad_pts)
    if not is_conv:
        return False, []
        
    p0, p1, p2, p3 = ordered_pts
    
    # Helper for dist and dot
    def dist_sq(a, b): return (a[0]-b[0])**2 + (a[1]-b[1])**2
    def dot(a, b, c): # Vector BA dot BC
        vx1, vy1 = a[0]-b[0], a[1]-b[1]
        vx2, vy2 = c[0]-b[0], c[1]-b[1]
        return vx1*vx2 + vy1*vy2, (vx1**2 + vy1**2), (vx2**2 + vy2**2)

    # Check aspect ratio first (fastest)
    d01 = dist_sq(p0, p1)
    d12 = dist_sq(p1, p2)
    d23 = dist_sq(p2, p3)
    d30 = dist_sq(p3, p0)
    
    sides = [d01, d12, d23, d30]
    max_s = max(sides)
    min_s = min(sides)
    
    # ratio check (squared)
    # 1.75^2 = 3.0625
    if max_s > 3.1 * min_s:
        return False, []
        
    # Check angles
    # We need cos(angle). cos(theta) = dot / (mag1 * mag2)
    # range 60-120 degrees -> cos(60)=0.5, cos(120)=-0.5
    # So we need |cos(theta)| <= 0.5
    
    # Corner 0 (p3-p0-p1)
    dp, l1_sq, l2_sq = dot(p3, p0, p1)
    if l1_sq == 0 or l2_sq == 0: return False, []
    cos_sq = (dp * dp) / (l1_sq * l2_sq)
    # if cos_theta > 0.5 or cos_theta < -0.5 -> cos_sq > 0.25
    # Wait, 60-120 deg means the angle is "not too sharp, not too flat"
    # cos(60) = 0.5, cos(120) = -0.5.
    # So we strictly want values between -0.5 and 0.5
    # So cos^2 < 0.25
    if cos_sq > 0.25: return False, []
    
    # Corner 1 (p0-p1-p2)
    dp, l1_sq, l2_sq = dot(p0, p1, p2)
    if l1_sq == 0 or l2_sq == 0: return False, []
    cos_sq = (dp * dp) / (l1_sq * l2_sq)
    if cos_sq > 0.25: return False, []
    
    # Corner 2 (p1-p2-p3)
    dp, l1_sq, l2_sq = dot(p1, p2, p3)
    if l1_sq == 0 or l2_sq == 0: return False, []
    cos_sq = (dp * dp) / (l1_sq * l2_sq)
    if cos_sq > 0.25: return False, []
    
    # Corner 3 (p2-p3-p0)
    dp, l1_sq, l2_sq = dot(p2, p3, p0)
    if l1_sq == 0 or l2_sq == 0: return False, []
    cos_sq = (dp * dp) / (l1_sq * l2_sq)
    if cos_sq > 0.25: return False, []

    # Check for points inside
    # Use OpenCV for this part as it's optimized C++
    poly_contour = np.array(ordered_pts, dtype=np.int32)
    for p in all_points:
        if p in quad_pts: continue
        # Simple bounding box check first?
        # Maybe not worth overhead
        dist = cv2.pointPolygonTest(poly_contour, (float(p[0]), float(p[1])), True)
        if dist > -margin:
            return False, []
            
    return True, ordered_pts

def find_quads(points):
    t_fq_start = time.time()
    if len(points) < 4: return []
    
    # Use top 25 points
    search_points = points[:25] 
    
    # Ensure points are tuples for faster access/hashing if needed
    # (they usually come as tuples from find_peaks)
    
    valid_quads = []
    
    combos = list(itertools.combinations(search_points, 4))
    t_combo_gen = time.time()
    
    count = 0
    
    for quad_combo in combos:
        is_valid, ordered_pts = check_quad_constraints(quad_combo, points)
        count += 1
        
        if is_valid:
            valid_quads.append(ordered_pts)
            
    t_fq_end = time.time()
    # Debug print
    # if count > 1000:
    #     print(f"FindQuads: {count} checks. Total: {(t_fq_end - t_combo_gen)*1000:.1f}ms")
    
    # --- Outlier Rejection based on Area ---
    if len(valid_quads) > 2:
        areas = []
        for q in valid_quads:
            arr = np.array(q, dtype=np.float32)
            areas.append(cv2.contourArea(arr))
            
        median_area = np.median(areas)
        
        # Filter: Keep quads within 0.5x to 2.0x of median area
        area_filtered_quads = []
        area_filtered_indices = []
        for i, area in enumerate(areas):
            if 0.5 * median_area <= area <= 2.0 * median_area:
                area_filtered_quads.append(valid_quads[i])
                area_filtered_indices.append(i)
        
        # --- Outlier Rejection based on Vanishing Points ---
        if len(area_filtered_quads) > 2:
            def get_line_intersection(p1, p2, p3, p4):
                x1, y1 = p1
                x2, y2 = p2
                x3, y3 = p3
                x4, y4 = p4
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-6: return None
                px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
                py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
                return (px, py)
            
            vp1_list = []
            vp2_list = []
            for q in area_filtered_quads:
                pts = np.array(q, dtype=np.float32)
                vp1 = get_line_intersection(pts[0], pts[1], pts[3], pts[2])
                vp2 = get_line_intersection(pts[0], pts[3], pts[1], pts[2])
                vp1_list.append(vp1)
                vp2_list.append(vp2)
            
            # Filter out None VPs
            valid_vp1 = [v for v in vp1_list if v is not None]
            valid_vp2 = [v for v in vp2_list if v is not None]
            
            if len(valid_vp1) > 2 and len(valid_vp2) > 2:
                # Compute median VP
                median_vp1 = (np.median([v[0] for v in valid_vp1]), np.median([v[1] for v in valid_vp1]))
                median_vp2 = (np.median([v[0] for v in valid_vp2]), np.median([v[1] for v in valid_vp2]))
                
                # Compute distances from median
                def vp_dist(vp, median):
                    if vp is None: return float('inf')
                    return math.sqrt((vp[0] - median[0])**2 + (vp[1] - median[1])**2)
                
                # Find median distance to use as threshold
                dists1 = [vp_dist(v, median_vp1) for v in vp1_list]
                dists2 = [vp_dist(v, median_vp2) for v in vp2_list]
                
                median_dist1 = np.median([d for d in dists1 if d < float('inf')])
                median_dist2 = np.median([d for d in dists2 if d < float('inf')])
                
                # Threshold: 3x median distance (generous)
                thresh1 = max(median_dist1 * 3, 1000)  # Min 1000px to avoid overly strict
                thresh2 = max(median_dist2 * 3, 500)
                
                vp_filtered_quads = []
                for i, q in enumerate(area_filtered_quads):
                    d1 = dists1[i]
                    d2 = dists2[i]
                    if d1 <= thresh1 and d2 <= thresh2:
                        vp_filtered_quads.append(q)
                
                return vp_filtered_quads
        
        return area_filtered_quads
        
    return valid_quads

# --- Homography Logic ---

def compute_robust_homography(quads):
    """Compute H using aggregated vanishing points from all quads."""
    if not quads: return None
    
    lines_h = [] 
    lines_v = [] 
    
    for q in quads:
        pts = np.array(q, dtype=np.float32)
        lines_h.append((pts[0], pts[1]))
        lines_h.append((pts[3], pts[2]))
        lines_v.append((pts[0], pts[3]))
        lines_v.append((pts[1], pts[2]))
        
    def intersect(l1, l2):
        p1, p2 = l1
        p3, p4 = l2
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6: return None 
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return np.array([px, py], dtype=np.float32)

    def get_vp_center(lines):
        intersections = []
        if len(lines) < 2: return None
        for _ in range(20): 
            idx1, idx2 = np.random.choice(len(lines), 2, replace=False)
            vp = intersect(lines[idx1], lines[idx2])
            if vp is not None:
                if np.linalg.norm(vp) > 1e6: continue 
                intersections.append(vp)
        
        if not intersections: return None
        pts = np.array(intersections)
        median = np.median(pts, axis=0)
        dists = np.linalg.norm(pts - median, axis=1)
        valid_pts = pts[dists < np.median(dists) * 2 + 100] 
        if len(valid_pts) == 0: return median
        return np.mean(valid_pts, axis=0)

    vp_h = get_vp_center(lines_h)
    vp_v = get_vp_center(lines_v)
    
    if vp_h is None or vp_v is None: return None

    vh = np.append(vp_h, 1)
    vv = np.append(vp_v, 1)
    l_inf = np.cross(vh, vv)
    l_inf = l_inf / l_inf[2] 
    
    H_rect = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [l_inf[0], l_inf[1], l_inf[2]]
    ], dtype=np.float32)
    
    # Find best quad
    best_quad = None
    max_area = 0
    for q in quads:
        pts = np.array(q, np.float32)
        area = cv2.contourArea(pts)
        if area > max_area:
            max_area = area
            best_quad = pts
            
    if best_quad is None: return None
    
    rect_quad = cv2.perspectiveTransform(best_quad.reshape(1, -1, 2), H_rect).reshape(-1, 2)
    dst_sq = np.array([[0,0], [100,0], [100,100], [0,100]], dtype=np.float32)
    H_affine, _ = cv2.findHomography(rect_quad, dst_sq)
    
    H_full = np.dot(H_affine, H_rect)
    return H_full

# --- Server Setup ---

app = FastAPI()
request_count = 0
model = None
mean = None
std = None
device = None

@app.get("/")
def read_root():
    return {"status": "running"}

@app.post("/predict")
def predict(
    file: UploadFile = File(...), 
    mode: str = Form("heatmap"), 
    threshold: float = Form(0.5)
):
    global request_count
    t_start = time.time()
    
    contents = file.file.read()
    t_read = time.time()
    
    if has_turbojpeg:
        try:
            img = jpeg.decode(contents)
        except Exception:
            # Fallback if decode fails (e.g. not a jpeg)
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None: return {"error": "Failed to decode"}
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img_padded = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        img_padded = img_rgb
        
    tensor = torch.from_numpy(img_padded.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor = (tensor - mean) / std
    
    t_preprocess = time.time()
    
    with torch.no_grad():
        output = model(tensor)
        heatmap = output.squeeze().cpu().numpy()
    
    t_inference = time.time()
    
    if pad_h > 0 or pad_w > 0:
        heatmap = heatmap[:h, :w]
    
    if mode == "heatmap":
        heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
        if has_turbojpeg:
            data = jpeg.encode(heatmap_uint8, quality=80)
        else:
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
        data = json.dumps({"points": points, "quads": quads}).encode('utf-8')
        media_type = "application/json"
        
    elif mode == "homography":
        points = find_peaks(heatmap, threshold)
        quads = find_quads(points)
        
        # Try Robust
        H = compute_robust_homography(quads)
        
        # Fallback to Best Quad
        if H is None and quads:
            best_quad = max(quads, key=lambda q: cv2.contourArea(np.array(q, np.float32)))
            src_pts = np.array(best_quad, np.float32)
            dst_pts = np.array([[0,0], [100,0], [100,100], [0,100]], dtype=np.float32)
            H, _ = cv2.findHomography(src_pts, dst_pts)
            
        h_list = H.tolist() if H is not None else None
        data = json.dumps({"points": points, "quads": quads, "h": h_list}).encode('utf-8')
        media_type = "application/json"
        
    else:
        return {"error": "Unknown mode"}

    t_end = time.time()
    
    # Print timing for every request to debug
    # print(f"[{mode}] Total: {(t_end - t_start)*1000:.1f}ms | "
    #       f"Read: {(t_read - t_start)*1000:.1f}ms | "
    #       f"Pre: {(t_preprocess - t_read)*1000:.1f}ms | "
    #       f"Infer: {(t_inference - t_preprocess)*1000:.1f}ms | "
    #       f"Post: {(t_end - t_inference)*1000:.1f}ms")
          
    return Response(content=data, media_type=media_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crossing Detector Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda/cpu)")
    parser.add_argument("--model", type=str, default="cross_net_224_best.pth", help="Path to model checkpoint")
    args = parser.parse_args()
    
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
        
    print(f"Using device: {device}")
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        exit(1)
        
    try:
        model = MobileUNet(pretrained=False).to(device)
        checkpoint = torch.load(args.model, map_location=device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        dummy = torch.zeros(1, 3, 480, 640).to(device)
        model(dummy)
        print(f"Model {args.model} loaded and warmed up!")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
        
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
