"""
Screen region capture with crossing detection heatmap display.
Drag the capture frame, see results in a separate window.
Supports both PyTorch (.pth) and ONNX (.onnx) models.
"""

import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk
import mss
import argparse
import threading
import queue
import time

# Try to import onnxruntime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Try to import torch (only needed for .pth files)
try:
    import torch
    from cross_annotator_gui import MobileUNet
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CrossingOverlay:
    def __init__(self, model_path, width=480, height=640, opacity=1.0, scale=1.0):
        self.width = width
        self.height = height
        self.opacity = opacity
        self.scale = scale  # Downscale factor for faster inference
        self.running = True
        self.paused = False
        self.use_onnx = model_path.endswith('.onnx')
        
        print(f"Capture: {width}x{height}, Inference: {int(width*scale)}x{int(height*scale)}")
        
        # Load model
        if self.use_onnx:
            if not ONNX_AVAILABLE:
                raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime")
            self._load_onnx_model(model_path)
        else:
            if not TORCH_AVAILABLE:
                raise RuntimeError("torch not installed")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            self._load_pytorch_model(model_path)
        
        # Screen capture (created per-thread)
        self.sct = None
        
        # Normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Async processing
        self.result_queue = queue.Queue(maxsize=1)
        self.capture_lock = threading.Lock()
        self.last_capture_pos = (0, 0)
        self.processing = False
        
        # Setup GUI
        self._setup_gui()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
    def _load_onnx_model(self, model_path):
        """Load ONNX model with onnxruntime."""
        # Use all available providers
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.ort_session = ort.InferenceSession(model_path, providers=providers)
        self.ort_input_name = self.ort_session.get_inputs()[0].name
        print(f"ONNX model loaded from {model_path}")
        print(f"ONNX providers: {self.ort_session.get_providers()}")
    
    def _load_pytorch_model(self, model_path):
        """Load PyTorch model."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        
        self.model = MobileUNet(pretrained=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Normalization tensors for PyTorch
        self.torch_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.torch_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        print(f"PyTorch model loaded from {model_path}")
    
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
        border = 1
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
    
    
    def _run_inference(self, img):
        """Run model inference (ONNX or PyTorch)."""
        orig_h, orig_w = img.shape[:2]
        
        t0 = time.time()
        
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
        
        t1 = time.time()
        
        # Prepare input tensor (NCHW format, normalized)
        input_data = img_padded.astype(np.float32) / 255.0
        input_data = (input_data - self.mean) / self.std
        input_data = input_data.transpose(2, 0, 1)  # HWC -> CHW
        input_data = input_data[np.newaxis, ...]  # Add batch dim
        
        t2 = time.time()
        
        if self.use_onnx:
            # ONNX Runtime inference
            output = self.ort_session.run(None, {self.ort_input_name: input_data})[0]
            heatmap = output.squeeze()
        else:
            # PyTorch inference
            tensor = torch.from_numpy(input_data).to(self.device)
            with torch.no_grad():
                output = self.model(tensor)
            heatmap = output.squeeze().cpu().numpy()
        
        t3 = time.time()
        
        if pad_h > 0 or pad_w > 0:
            heatmap = heatmap[:h, :w]
        
        # Upscale heatmap back to original size
        if self.scale < 1.0:
            heatmap = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        t4 = time.time()
        
        # Print inference breakdown
        print(f"  [Infer] Prep: {(t1-t0)*1000:.1f}ms | ToTensor: {(t2-t1)*1000:.1f}ms | Forward: {(t3-t2)*1000:.1f}ms | Post: {(t4-t3)*1000:.1f}ms")
            
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
    
    def _process_loop(self):
        """Background thread for capture and inference."""
        # Create mss instance in this thread (not thread-safe across threads)
        sct = mss.mss()
        
        while self.running:
            try:
                t0 = time.time()
                
                # Get current capture position
                with self.capture_lock:
                    x, y = self.last_capture_pos
                
                # Capture screen
                border = 1
                w = self.width - 2 * border
                h = self.height - 2 * border
                monitor = {"left": x + border, "top": y + 30, "width": w, "height": h}
                screenshot = sct.grab(monitor)
                t1 = time.time()
                
                img = np.array(screenshot)[:, :, :3]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                t2 = time.time()
                
                # Run inference
                heatmap = self._run_inference(img)
                t3 = time.time()
                
                result = self._create_overlay_image(img, heatmap)
                t4 = time.time()
                
                # Calculate times
                capture_ms = (t1 - t0) * 1000
                convert_ms = (t2 - t1) * 1000
                infer_ms = (t3 - t2) * 1000
                overlay_ms = (t4 - t3) * 1000
                total_ms = (t4 - t0) * 1000
                fps = 1000.0 / total_ms if total_ms > 0 else 0
                
                # Print profiling every frame
                print(f"Capture: {capture_ms:.1f}ms | Convert: {convert_ms:.1f}ms | Infer: {infer_ms:.1f}ms | Overlay: {overlay_ms:.1f}ms | Total: {total_ms:.1f}ms ({fps:.1f} FPS)")
                
                # Put result in queue (non-blocking, drop old frames)
                try:
                    self.result_queue.put_nowait((result, fps, heatmap))
                except queue.Full:
                    pass
                    
            except Exception as e:
                print(f"Process error: {e}")
                time.sleep(0.1)
    
    def _update_loop(self):
        """Main thread UI update loop."""
        if not self.running:
            return
        
        # Update capture position
        with self.capture_lock:
            self.last_capture_pos = (self.capture_frame.winfo_x(), self.capture_frame.winfo_y())
        
        # Check for new results
        try:
            result, fps, heatmap = self.result_queue.get_nowait()
            
            # Display in result window
            pil_img = Image.fromarray(result)
            pil_img = pil_img.resize((self.width, self.height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(pil_img)
            
            if self.result_image is None:
                self.result_image = self.result_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            else:
                self.result_canvas.itemconfig(self.result_image, image=self.photo)
            
            self.fps = fps
            self._update_status()
            
        except queue.Empty:
            pass
        
        self.root.after(16, self._update_loop)  # ~60fps UI update
    
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
