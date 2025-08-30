#!/usr/bin/env python3
"""
CUDA Status Checker and Performance Diagnostic Tool

This script checks your CUDA setup and provides recommendations for improving frame rate.
"""

import torch
import cv2
import time
import numpy as np
import os

def check_cuda_status():
    """Check CUDA availability and device information."""
    print("🔍 CUDA Status Check")
    print("=" * 50)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {'✅ YES' if cuda_available else '❌ NO'}")
    
    if cuda_available:
        # Get CUDA device count
        device_count = torch.cuda.device_count()
        print(f"CUDA Devices: {device_count}")
        
        # Get current device
        current_device = torch.cuda.current_device()
        print(f"Current Device: {current_device}")
        
        # Get device properties
        device_props = torch.cuda.get_device_properties(current_device)
        print(f"Device Name: {device_props.name}")
        print(f"Compute Capability: {device_props.major}.{device_props.minor}")
        print(f"Total Memory: {device_props.total_memory / 1024**3:.2f} GB")
        print(f"Multi-Processor Count: {device_props.multi_processor_count}")
        
        # Check if CUDA is being used
        device = torch.device("cuda:0")
        test_tensor = torch.randn(1000, 1000).to(device)
        print(f"CUDA Device Test: {'✅ Working' if test_tensor.device.type == 'cuda' else '❌ Failed'}")
        
    else:
        print("❌ CUDA is not available. This will significantly impact performance.")
        print("   Consider installing CUDA-enabled PyTorch if you have an NVIDIA GPU.")
    
    print()

def check_pytorch_version():
    """Check PyTorch version and build information."""
    print("📦 PyTorch Information")
    print("=" * 50)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda if torch.version.cuda else 'N/A'}")
    print(f"cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print(f"cuDNN Available: {'✅ YES' if torch.backends.cudnn.is_available() else '❌ NO'}")
    print()

def performance_test():
    """Run a simple performance test."""
    print("⚡ Performance Test")
    print("=" * 50)
    
    # Test CPU vs GPU inference speed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Create test image
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Test OpenCV operations
    start_time = time.time()
    for _ in range(100):
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    cv_time = (time.time() - start_time) / 100
    print(f"OpenCV Operations: {cv_time*1000:.2f} ms per frame")
    
    # Test YOLO model loading (if available)
    try:
        print("\n🔄 Loading YOLOv5 model for testing...")
        model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
        model.to(device)
        
        if device.type == "cuda":
            model.half()  # Use half precision for speed
        
        # Warm up
        for _ in range(3):
            _ = model(test_image)
        
        # Test inference speed
        start_time = time.time()
        for _ in range(10):
            _ = model(test_image)
        inference_time = (time.time() - start_time) / 10
        
        print(f"YOLOv5n Inference: {inference_time*1000:.2f} ms per frame")
        print(f"Estimated FPS: {1.0/inference_time:.1f}")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
        
    except Exception as e:
        print(f"❌ Could not test YOLO model: {e}")
    
    print()

def optimization_recommendations():
    """Provide optimization recommendations."""
    print("🚀 Optimization Recommendations")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("1. 🚨 INSTALL CUDA: Install CUDA-enabled PyTorch for GPU acceleration")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print()
    
    print("2. 📹 Use Substream: Switch to camera substream for lower resolution")
    print("   - Modify RTSP URL to use /Streaming/Channels/102 (substream)")
    print("   - Lower resolution = faster processing")
    print()
    
    print("3. 🎯 Use Smaller Model: Switch to YOLOv5n for speed")
    print("   - Set YOLO_MODEL=yolov5n in camera_config.env")
    print("   - YOLOv5n is ~3x faster than YOLOv5s")
    print()
    
    print("4. ⚙️ Adjust Confidence Threshold: Increase for speed")
    print("   - Set CONFIDENCE_THRESHOLD=0.5 in camera_config.env")
    print("   - Higher threshold = fewer detections = faster processing")
    print()
    
    print("5. 🔄 Reduce Processing: Skip frames if needed")
    print("   - Process every 2nd or 3rd frame for speed")
    print("   - Motion detection can still work on skipped frames")
    print()
    
    print("6. 🖥️ Monitor GPU Usage: Check if GPU is being utilized")
    print("   - Use nvidia-smi to monitor GPU usage")
    print("   - Ensure CUDA is actually being used")
    print()

def check_camera_config():
    """Check current camera configuration."""
    print("📷 Camera Configuration Check")
    print("=" * 50)
    
    # Check environment file
    env_file = "camera_config.env"
    if os.path.exists(env_file):
        print(f"✅ {env_file} exists")
        try:
            with open(env_file, 'r') as f:
                content = f.read()
                if 'YOLO_MODEL' in content:
                    model_line = [line for line in content.split('\n') if 'YOLO_MODEL' in line][0]
                    print(f"   {model_line.strip()}")
                if 'CONFIDENCE_THRESHOLD' in content:
                    conf_line = [line for line in content.split('\n') if 'CONFIDENCE_THRESHOLD' in line][0]
                    print(f"   {conf_line.strip()}")
        except:
            print("   Could not read configuration details")
    else:
        print(f"❌ {env_file} not found")
    
    print()

def main():
    """Run all diagnostic checks."""
    print("🚀 ASECAM Camera Performance Diagnostic Tool")
    print("=" * 60)
    print()
    
    check_cuda_status()
    check_pytorch_version()
    check_camera_config()
    performance_test()
    optimization_recommendations()
    
    print("💡 Quick Performance Boost:")
    print("   - Run: python check_cuda_status.py")
    print("   - Check if CUDA is working")
    print("   - Consider switching to YOLOv5n model")
    print("   - Use camera substream for lower resolution")
    print()

if __name__ == "__main__":
    main()
