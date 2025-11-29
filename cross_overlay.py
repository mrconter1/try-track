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
    def __init__(self, model_path, window_size=480, opacity=0.5):
        self.window_size = window_size
        self.opacity = opacity
        self.running = True
        self.paused = False
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
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
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        if 'backbone.0.0' in list(state_dict.keys())[0] or any('backbone.0.0' in k for k in state_dict.keys()):
            model = MobileUNet()
        elif any('backbone.0.block.0' in k for k in state_dict.keys()):
            num_params = sum(p.numel() for p in state_dict.values())
            if num_params > 5_000_000:
                model = MobileUNetV3Large()
            else:
                model = MobileUNetV3Small()
        else:
            model = MobileUNet()
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        print(f"Model loaded from {model_path}")
        return model
    
    def _setup_gui(self):
        """Create capture frame and result window."""
        # Main root (hidden)
        self.root = tk.Tk()
        self.root.withdraw()
        
        # Capture frame window - transparent with colored border
        self.capture_frame = tk.Toplevel(self.root)
        self.capture_frame.title("Capture Region (drag me)")
        self.capture_frame.geometry(f"{self.window_size}x{self.window_size}+100+100")
        self.capture_frame.attributes('-topmost', True)
        
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
        
        # Small label in corner
        self.capture_label = tk.Label(self.capture_frame, text="SPACE=capture C=continuous", 
                                       bg='lime', fg='black', font=('Consolas', 9))
        self.capture_label.place(x=border, y=border)
        
        # Result window
        self.result_window = tk.Toplevel(self.root)
        self.result_window.title("Detection Result")
        self.result_window.geometry(f"{self.window_size}x{self.window_size}+{100 + self.window_size + 20}+100")
        
        self.result_canvas = tk.Canvas(self.result_window, width=self.window_size, height=self.window_size, bg='black')
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        self.result_image = None
        
        # Status
        self.status_var = tk.StringVar(value="Ready | Opacity: 0.5")
        self.status_label = tk.Label(self.result_window, textvariable=self.status_var,
                                      bg='black', fg='lime', font=('Consolas', 9))
        self.status_label.place(x=5, y=5)
        
        # Bindings
        self.capture_frame.bind('<space>', lambda e: self._capture_and_process())
        self.result_window.bind('<space>', lambda e: self._capture_and_process())
        self.capture_frame.bind('<Escape>', lambda e: self._quit())
        self.result_window.bind('<Escape>', lambda e: self._quit())
        self.capture_frame.bind('<Up>', lambda e: self._adjust_opacity(0.1))
        self.capture_frame.bind('<Down>', lambda e: self._adjust_opacity(-0.1))
        self.result_window.bind('<Up>', lambda e: self._adjust_opacity(0.1))
        self.result_window.bind('<Down>', lambda e: self._adjust_opacity(-0.1))
        self.capture_frame.bind('<c>', lambda e: self._toggle_continuous())
        self.result_window.bind('<c>', lambda e: self._toggle_continuous())
        
        # Continuous mode
        self.continuous = False
        self.root.after(100, self._update_loop)
        
        # Handle window close
        self.capture_frame.protocol("WM_DELETE_WINDOW", self._quit)
        self.result_window.protocol("WM_DELETE_WINDOW", self._quit)
    
    def _adjust_opacity(self, delta):
        self.opacity = max(0.1, min(1.0, self.opacity + delta))
        self._update_status()
        
    def _toggle_continuous(self):
        self.continuous = not self.continuous
        self._update_status()
        if self.continuous:
            self.capture_label.config(text="[CONTINUOUS] C=stop")
        else:
            self.capture_label.config(text="SPACE=capture C=continuous")
    
    def _update_status(self):
        mode = "Continuous" if self.continuous else "Ready"
        self.status_var.set(f"{mode} | Opacity: {self.opacity:.1f}")
    
    def _quit(self):
        self.running = False
        self.root.destroy()
    
    def _capture_screen(self):
        """Capture the region under the capture frame (transparent, no hide needed)."""
        # Get frame position (inside the borders)
        border = 4
        x = self.capture_frame.winfo_x() + border
        y = self.capture_frame.winfo_y() + 30  # Title bar offset
        w = self.window_size - 2 * border
        h = self.window_size - 2 * border
        
        # Capture directly - frame is transparent
        monitor = {"left": x, "top": y, "width": w, "height": h}
        screenshot = self.sct.grab(monitor)
        img = np.array(screenshot)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def _run_inference(self, img):
        """Run model inference."""
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
            heatmap = torch.sigmoid(output).squeeze().cpu().numpy()
        
        if pad_h > 0 or pad_w > 0:
            heatmap = heatmap[:h, :w]
            
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
        try:
            img = self._capture_screen()
            heatmap = self._run_inference(img)
            result = self._create_overlay_image(img, heatmap)
            
            # Display in result window
            pil_img = Image.fromarray(result)
            pil_img = pil_img.resize((self.window_size, self.window_size), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(pil_img)
            
            if self.result_image is None:
                self.result_image = self.result_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            else:
                self.result_canvas.itemconfig(self.result_image, image=self.photo)
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_loop(self):
        """Continuous capture loop."""
        if not self.running:
            return
        
        if self.continuous:
            self._capture_and_process()
        
        self.root.after(100, self._update_loop)
    
    def run(self):
        """Start the application."""
        print("\nControls:")
        print("  Space     - Capture & detect")
        print("  C         - Toggle continuous mode")
        print("  Up/Down   - Adjust opacity")
        print("  Escape    - Quit")
        print("\nDrag the green frame over content, press SPACE to capture.")
        print()
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Crossing detector overlay")
    parser.add_argument('--model', type=str, required=True, help='Path to model .pth file')
    parser.add_argument('--size', type=int, default=480, help='Window size (default: 480)')
    parser.add_argument('--opacity', type=float, default=0.5, help='Overlay opacity (default: 0.5)')
    args = parser.parse_args()
    
    overlay = CrossingOverlay(
        model_path=args.model,
        window_size=args.size,
        opacity=args.opacity
    )
    overlay.run()


if __name__ == "__main__":
    main()
