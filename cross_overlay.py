"""
Screen region capture with crossing detection heatmap display.
Drag the capture frame, see results in a separate window.
"""

import tkinter as tk
import numpy as np
import torch
import cv2
from PIL import Image, ImageTk
import mss
import argparse

from cross_annotator_gui import MobileUNet, MobileUNetV3Small, MobileUNetV3Large


class CrossingOverlay:
    def __init__(self, model_path, width=480, height=640, opacity=1.0, scale=1.0):
        self.width = width
        self.height = height
        self.opacity = opacity
        self.scale = scale  # Downscale factor for faster inference
        self.running = True
        self.paused = False
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Capture: {width}x{height}, Inference: {int(width*scale)}x{int(height*scale)}")
        self.model = self._load_model(model_path)
        
        # Screen capture
        self.sct = mss.mss()
        
        # Normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        # Setup GUI
        self._setup_gui()
        
    def _load_model(self, model_path):
        """Load the trained model."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Debug: show checkpoint structure
        print(f"Checkpoint type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        
        # Debug: print first 10 keys
        print(f"State dict keys (first 10):")
        for i, k in enumerate(list(state_dict.keys())[:10]):
            print(f"  {k}")
        
        # Count parameters
        num_params = sum(p.numel() for p in state_dict.values())
        print(f"Total parameters: {num_params:,}")
        
        # Create model
        print("Using: MobileNetV2")
        model = MobileUNet(pretrained=False)
        
        # Check for key mismatches
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        missing = model_keys - loaded_keys
        extra = loaded_keys - model_keys
        if missing:
            print(f"Missing keys ({len(missing)}): {list(missing)[:3]}...")
        if extra:
            print(f"Extra keys ({len(extra)}): {list(extra)[:3]}...")
        if not missing and not extra:
            print("All keys match!")
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return model
    
    def _setup_gui(self):
        """Create capture frame and result window."""
        # Main root (hidden)
        self.root = tk.Tk()
        self.root.withdraw()
        
        # Capture frame window - transparent with colored border
        self.capture_frame = tk.Toplevel(self.root)
        self.capture_frame.title("Capture Region (drag me)")
        # Center on screen
        screen_w = self.capture_frame.winfo_screenwidth()
        screen_h = self.capture_frame.winfo_screenheight()
        center_x = (screen_w - self.width) // 2
        center_y = (screen_h - self.height) // 2
        self.capture_frame.geometry(f"{self.width}x{self.height}+{center_x}+{center_y}")
        self.capture_frame.attributes('-topmost', True)
        self.capture_frame.resizable(False, False)
        
        # Make window transparent
        self.capture_frame.attributes('-transparentcolor', 'magenta')
        self.capture_frame.configure(bg='magenta')
        
        # Create border using 4 thin frames on edges
        border = 4
        # Top border
        tk.Frame(self.capture_frame, bg='lime', height=border).place(x=0, y=0, relwidth=1)
        # Bottom border  
        tk.Frame(self.capture_frame, bg='lime', height=border).place(x=0, rely=1, y=-border, relwidth=1)
        # Left border
        tk.Frame(self.capture_frame, bg='lime', width=border).place(x=0, y=0, relheight=1)
        # Right border
        tk.Frame(self.capture_frame, bg='lime', width=border).place(relx=1, x=-border, y=0, relheight=1)
        
        # Result window - vertically centered, at left edge
        self.result_window = tk.Toplevel(self.root)
        self.result_window.title("Detection Result")
        result_y = (screen_h - self.height) // 2
        self.result_window.geometry(f"{self.width}x{self.height}+0+{result_y}")
        self.result_window.attributes('-topmost', True)
        self.result_window.resizable(False, False)
        
        self.result_canvas = tk.Canvas(self.result_window, width=self.width, height=self.height, bg='black')
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        self.result_image = None
        
        # Status
        self.status_var = tk.StringVar(value="FPS: 0.0 | Opacity: 0.5")
        self.status_label = tk.Label(self.result_window, textvariable=self.status_var,
                                      bg='black', fg='lime', font=('Consolas', 9))
        self.status_label.place(x=5, y=5)
        
        # Bindings
        self.capture_frame.bind('<Escape>', lambda e: self._quit())
        self.result_window.bind('<Escape>', lambda e: self._quit())
        self.capture_frame.bind('<Up>', lambda e: self._adjust_opacity(0.1))
        self.capture_frame.bind('<Down>', lambda e: self._adjust_opacity(-0.1))
        self.result_window.bind('<Up>', lambda e: self._adjust_opacity(0.1))
        self.result_window.bind('<Down>', lambda e: self._adjust_opacity(-0.1))
        
        # Continuous mode (default on)
        self.continuous = True
        self.last_time = None
        self.fps = 0
        self.root.after(100, self._update_loop)
        
        # Handle window close
        self.capture_frame.protocol("WM_DELETE_WINDOW", self._quit)
        self.result_window.protocol("WM_DELETE_WINDOW", self._quit)
    
    def _adjust_opacity(self, delta):
        self.opacity = max(0.1, min(1.0, self.opacity + delta))
        self._update_status()
    
    def _update_status(self):
        self.status_var.set(f"FPS: {self.fps:.1f} | Opacity: {self.opacity:.1f}")
    
    def _quit(self):
        self.running = False
        self.root.destroy()
    
    def _capture_screen(self):
        """Capture the region under the capture frame (transparent, no hide needed)."""
        # Get frame position (inside the borders)
        border = 4
        x = self.capture_frame.winfo_x() + border
        y = self.capture_frame.winfo_y() + 30  # Title bar offset
        w = self.width - 2 * border
        h = self.height - 2 * border
        
        # Capture directly - frame is transparent
        monitor = {"left": x, "top": y, "width": w, "height": h}
        screenshot = self.sct.grab(monitor)
        img = np.array(screenshot)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def _run_inference(self, img):
        """Run model inference."""
        orig_h, orig_w = img.shape[:2]
        
        # Downscale for faster inference
        if self.scale < 1.0:
            new_h, new_w = int(orig_h * self.scale), int(orig_w * self.scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        h, w = img.shape[:2]
        
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        
        if pad_h > 0 or pad_w > 0:
            img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            img_padded = img
        
        tensor = torch.from_numpy(img_padded.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        tensor = (tensor - self.mean) / self.std
        
        with torch.no_grad():
            output = self.model(tensor)
            # Model uses MSE loss, outputs 0-1 directly, no sigmoid needed
            heatmap = output.squeeze().cpu().numpy()
        
        if pad_h > 0 or pad_w > 0:
            heatmap = heatmap[:h, :w]
        
        # Upscale heatmap back to original size
        if self.scale < 1.0:
            heatmap = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
        return heatmap
    
    def _create_overlay_image(self, img, heatmap):
        """Blend original image with green heatmap overlay."""
        h, w = img.shape[:2]
        
        # Resize heatmap if needed
        if heatmap.shape != (h, w):
            heatmap = cv2.resize(heatmap, (w, h))
        
        intensity = np.clip(heatmap, 0, 1)
        
        # Green overlay
        overlay = np.zeros_like(img)
        overlay[:, :, 1] = 255  # Green channel
        
        # Blend: original * (1 - intensity*opacity) + green * intensity*opacity
        blend_factor = (intensity * self.opacity)[:, :, np.newaxis]
        result = img * (1 - blend_factor) + overlay * blend_factor
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _capture_and_process(self):
        """Capture screen and run detection."""
        import time
        try:
            start_time = time.time()
            
            img = self._capture_screen()
            heatmap = self._run_inference(img)
            result = self._create_overlay_image(img, heatmap)
            
            # Display in result window
            pil_img = Image.fromarray(result)
            pil_img = pil_img.resize((self.width, self.height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(pil_img)
            
            if self.result_image is None:
                self.result_image = self.result_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            else:
                self.result_canvas.itemconfig(self.result_image, image=self.photo)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                self.fps = 1.0 / elapsed
            
            # Debug: print heatmap statistics
            print(f"Heatmap - min: {heatmap.min():.3f}, max: {heatmap.max():.3f}, mean: {heatmap.mean():.3f}")
            self._update_status()
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_loop(self):
        """Continuous capture loop."""
        if not self.running:
            return
        
        self._capture_and_process()
        
        self.root.after(10, self._update_loop)
    
    def run(self):
        """Start the application."""
        print("\nControls:")
        print("  Up/Down   - Adjust opacity")
        print("  Escape    - Quit")
        print("\nDrag the green frame over content to analyze.")
        print()
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Crossing detector overlay")
    parser.add_argument('--model', type=str, required=True, help='Path to model .pth file')
    parser.add_argument('--width', type=int, default=480, help='Capture width (default: 480)')
    parser.add_argument('--height', type=int, default=640, help='Capture height (default: 640)')
    parser.add_argument('--opacity', type=float, default=1.0, help='Overlay opacity (default: 1.0)')
    parser.add_argument('--scale', type=float, default=1.0, help='Inference scale factor for speed (default: 1.0 = full res)')
    args = parser.parse_args()
    
    overlay = CrossingOverlay(
        model_path=args.model,
        width=args.width,
        height=args.height,
        opacity=args.opacity,
        scale=args.scale
    )
    overlay.run()


if __name__ == "__main__":
    main()
