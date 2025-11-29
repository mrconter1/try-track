"""
Export PyTorch crossing detector model to ONNX and TFLite formats.

Usage:
    python export_model.py --model cross_net_v6_best.pth --output cross_net
    
This creates:
    - cross_net.onnx (for ONNX Runtime, conversion to other formats)
    - cross_net.tflite (for Android/iOS) - requires additional dependencies
"""

import argparse
import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class MobileUNet(nn.Module):
    """Mobile-optimized U-Net with MobileNetV2 backbone for crossing detection."""
    
    def __init__(self, pretrained=False):
        super().__init__()
        
        if pretrained:
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
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
            nn.Sigmoid()
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


class MobileUNetV3Small(nn.Module):
    """Lighter U-Net with MobileNetV3-Small backbone (~2.5x faster than V2)."""
    
    def __init__(self, pretrained=False):
        super().__init__()
        
        if pretrained:
            mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            mobilenet = models.mobilenet_v3_small(weights=None)
        self.encoder = mobilenet.features
        
        # MobileNetV3-Small: 576 final channels, skip channels at [16, 16, 24, 48]
        self.up1 = nn.ConvTranspose2d(576, 48, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(48 + 48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(24 + 24, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(24, 16, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.final_up = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.out = nn.Sequential(
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        skip_connections = []
        # MobileNetV3-Small skip indices: [0, 1, 3, 8] for resolutions H/2, H/4, H/8, H/16
        skip_indices = [0, 1, 3, 8]
        
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


class MobileUNetV3Large(nn.Module):
    """U-Net with MobileNetV3-Large backbone (more capacity than V3-Small)."""
    
    def __init__(self, pretrained=False):
        super().__init__()
        
        if pretrained:
            mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        else:
            mobilenet = models.mobilenet_v3_large(weights=None)
        self.encoder = mobilenet.features
        
        # MobileNetV3-Large: 960 final channels, skip channels at [16, 24, 40, 112]
        self.up1 = nn.ConvTranspose2d(960, 112, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(112 + 112, 112, 3, padding=1),
            nn.BatchNorm2d(112),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(112, 40, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(40 + 40, 40, 3, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(40, 24, 2, stride=2)
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
            nn.Sigmoid()
        )
    
    def forward(self, x):
        skip_connections = []
        # MobileNetV3-Large skip indices: [0, 2, 4, 11] for resolutions H/2, H/4, H/8, H/16
        skip_indices = [0, 2, 4, 11]
        
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


def export_to_onnx(model, output_path, input_size=(1, 3, 640, 480)):
    """Export model to ONNX format."""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_size)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    print(f"Exported ONNX model to: {output_path}")
    print(f"  Input shape: {input_size}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def export_to_tflite(onnx_path, output_path):
    """Convert ONNX to TFLite (requires onnx-tf and tensorflow)."""
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError:
        print("\nTo export TFLite, install dependencies:")
        print("  pip install onnx onnx-tf tensorflow")
        return False
    
    # Load ONNX model
    print(f"\nConverting ONNX to TFLite...")
    onnx_model = onnx.load(onnx_path)
    
    # Convert to TensorFlow
    tf_rep = prepare(onnx_model)
    
    # Save as SavedModel
    saved_model_path = output_path.replace('.tflite', '_saved_model')
    tf_rep.export_graph(saved_model_path)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Exported TFLite model to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # Cleanup
    import shutil
    if os.path.exists(saved_model_path):
        shutil.rmtree(saved_model_path)
    
    return True


def convert_to_fp16(onnx_path, output_path):
    """Convert ONNX model to FP16 (half precision)."""
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        print("\nTo export FP16, install dependencies:")
        print("  pip install onnx onnxconverter-common")
        return False
    
    print(f"\nConverting to FP16...")
    model = onnx.load(onnx_path)
    fp16_model = float16.convert_float_to_float16(model)
    onnx.save(fp16_model, output_path)
    
    print(f"Exported FP16 ONNX model to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export crossing detector model")
    parser.add_argument("--model", required=True, help="Path to PyTorch model (.pth)")
    parser.add_argument("--output", default="cross_net", help="Output name (without extension)")
    parser.add_argument("--height", type=int, default=480, help="Input height (default: 480)")
    parser.add_argument("--width", type=int, default=640, help="Input width (default: 640)")
    parser.add_argument("--tflite", action="store_true", help="Also export to TFLite")
    parser.add_argument("--fp16", action="store_true", help="Also export to FP16 ONNX")
    parser.add_argument("--backbone", choices=["mobilenetv2", "mobilenetv3-small", "mobilenetv3-large"], default="mobilenetv2", help="Encoder backbone")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.model}")
    if args.backbone == "mobilenetv3-small":
        print("Using MobileNetV3-Small backbone")
        model = MobileUNetV3Small(pretrained=False)
    elif args.backbone == "mobilenetv3-large":
        print("Using MobileNetV3-Large backbone")
        model = MobileUNetV3Large(pretrained=False)
    else:
        print("Using MobileNetV2 backbone")
        model = MobileUNet(pretrained=False)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()
    print("Model loaded successfully")
    
    # Export to ONNX
    onnx_path = f"{args.output}.onnx"
    export_to_onnx(model, onnx_path, input_size=(1, 3, args.height, args.width))
    
    # Convert to FP16 if requested
    if args.fp16:
        fp16_path = f"{args.output}_fp16.onnx"
        convert_to_fp16(onnx_path, fp16_path)
    
    # Export to TFLite if requested
    if args.tflite:
        tflite_path = f"{args.output}.tflite"
        export_to_tflite(onnx_path, tflite_path)
    
    print("\nDone!")
    print("\nNext steps for Android:")
    print("  1. Copy .onnx or .tflite to your Android app's assets folder")
    print("  2. Use ONNX Runtime or TFLite interpreter in Kotlin")
    print("  3. Preprocess: resize, normalize with ImageNet mean/std")
    print("  4. Run inference and find peaks in output heatmap")


if __name__ == "__main__":
    main()

