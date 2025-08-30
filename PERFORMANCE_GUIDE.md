# 🚀 Performance Optimization Guide

This guide helps you optimize your ASECAM camera system for better frame rate and performance.

## 🔍 **First: Check Your CUDA Status**

Run the diagnostic tool to see what's affecting your performance:

```bash
python check_cuda_status.py
```

This will show you:
- ✅ Whether CUDA is working
- 📊 Your GPU specifications
- ⚡ Current performance metrics
- 🎯 Optimization recommendations

## 🚨 **Common Performance Issues & Solutions**

### **1. CUDA Not Working (Most Common)**
**Symptoms:** Low FPS, CPU usage high, GPU usage low
**Solution:** Install CUDA-enabled PyTorch

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision

# Install CUDA-enabled version (for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### **2. Using Wrong YOLO Model**
**Current:** `yolov5s` (balanced)
**Faster:** `yolov5n` (3x faster, slightly less accurate)

Edit `camera_config.env`:
```env
YOLO_MODEL=yolov5n
```

### **3. Processing Every Frame**
**Current:** Processing every frame
**Faster:** Process every 2nd or 3rd frame

Edit `performance_config.env`:
```env
PROCESS_EVERY_N_FRAMES=2
```

### **4. Using Main Stream Instead of Substream**
**Current:** High-resolution main stream
**Faster:** Lower-resolution substream

Edit `performance_config.env`:
```env
USE_SUBSTREAM=true
```

## ⚡ **Quick Performance Boost (5 minutes)**

### **Step 1: Create Performance Config**
```bash
cp performance_config.env.example performance_config.env
```

### **Step 2: Edit for Speed**
```env
# Fastest settings
PROCESS_EVERY_N_FRAMES=2
USE_SUBSTREAM=true
YOLO_MODEL=yolov5n
CONFIDENCE_THRESHOLD=0.5
```

### **Step 3: Test Performance**
```bash
python camera_yolo_detection.py
```

## 📊 **Performance Settings Comparison**

| Setting | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| `PROCESS_EVERY_N_FRAMES=1` | ⭐⭐ | ⭐⭐⭐⭐⭐ | Best tracking |
| `PROCESS_EVERY_N_FRAMES=2` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Balanced |
| `PROCESS_EVERY_N_FRAMES=3` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Fastest |

| Model | Speed | Accuracy | Memory |
|-------|-------|----------|---------|
| `yolov5n` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| `yolov5s` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| `yolov5m` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## 🎯 **Recommended Configurations**

### **High Performance (Gaming/Real-time)**
```env
PROCESS_EVERY_N_FRAMES=2
USE_SUBSTREAM=true
YOLO_MODEL=yolov5n
CONFIDENCE_THRESHOLD=0.6
```

### **Balanced (Default)**
```env
PROCESS_EVERY_N_FRAMES=1
USE_SUBSTREAM=true
YOLO_MODEL=yolov5s
CONFIDENCE_THRESHOLD=0.5
```

### **High Accuracy (Security)**
```env
PROCESS_EVERY_N_FRAMES=1
USE_SUBSTREAM=false
YOLO_MODEL=yolov5s
CONFIDENCE_THRESHOLD=0.25
```

## 🔧 **Advanced Optimizations**

### **1. Frame Skipping with Motion Detection**
```env
PROCESS_EVERY_N_FRAMES=3
```
- Object detection: Every 3rd frame
- Motion detection: Every frame
- Door tracking: Every 3rd frame
- **Result:** 3x faster processing, motion still responsive

### **2. Substream Resolution**
- **Main stream:** 1080p/4K (slower)
- **Substream:** 720p/480p (faster)
- **Switch in camera settings** if available

### **3. GPU Memory Management**
```python
# Add to your script if using CUDA
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU memory
```

## 📈 **Expected Performance Improvements**

| Optimization | FPS Improvement | Setup Time |
|--------------|-----------------|------------|
| CUDA installation | 3-5x | 10 minutes |
| YOLOv5n model | 2-3x | 1 minute |
| Frame skipping | 2-3x | 1 minute |
| Substream | 1.5-2x | 1 minute |
| **Combined** | **8-15x** | **15 minutes** |

## 🚨 **Troubleshooting**

### **Still Slow After Optimizations?**
1. **Check GPU usage:** `nvidia-smi`
2. **Check CPU usage:** Task Manager
3. **Check network:** Camera stream quality
4. **Check resolution:** Camera output size

### **Motion Detection Not Working with Frame Skip?**
- Motion detection works on ALL frames
- Only object detection is skipped
- Door tracking uses object detection results

### **Door Tracking Accuracy with Frame Skip?**
- Tracking is maintained between frames
- Higher frame skip = less smooth tracking
- Recommended: 2-3 frames max for door tracking

## 🎉 **Quick Start Commands**

```bash
# 1. Check current performance
python check_cuda_status.py

# 2. Create performance config
cp performance_config.env.example performance_config.env

# 3. Edit for speed (see examples above)

# 4. Test optimized system
python camera_yolo_detection.py

# 5. Monitor FPS improvement
```

## 💡 **Pro Tips**

- **Start with `PROCESS_EVERY_N_FRAMES=2`** - best balance
- **Use `yolov5n` for testing**, upgrade to `yolov5s` if accuracy needed
- **Motion detection always works** regardless of frame skipping
- **Door tracking accuracy** maintained with smart interpolation
- **Monitor GPU memory** if using CUDA

Your system should now run much faster! 🚀⚡
