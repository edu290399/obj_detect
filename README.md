# ASECAM Camera Object Detection with YOLOv5

This project provides real-time object detection using a pre-trained YOLOv5 model connected to an ASECAM 8MP IP camera via ONVIF/RTSP.

## Features

- **Real-time object detection** from IP camera stream
- **ONVIF camera discovery** with fallback RTSP paths
- **Secure credential management** using environment variables
- **Automatic reconnection** if stream drops
- **Performance metrics** (FPS, detection count)
- **GPU acceleration** support (CUDA)

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Camera Settings

Copy `camera_config.env.example` to `camera_config.env` and update with your camera details:

```bash
# Copy example config
cp camera_config.env.example camera_config.env

# Edit the file with your camera details
notepad camera_config.env
```

**Example configuration:**
```env
# ASECAM Camera Configuration
CAMERA_HOST=192.168.1.10
CAMERA_PORT=80
CAMERA_USERNAME=admin
CAMERA_PASSWORD=your_password_here

# YOLOv5 Model Configuration
YOLO_MODEL=yolov5s
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.45
```

### 3. Run Object Detection

```bash
python camera_yolo_detection.py
```

## Usage

- **Press 'q'** to quit the detection window
- **FPS display** shows real-time performance
- **Detection count** shows number of objects detected
- **Bounding boxes** with labels and confidence scores

## Security Notes

⚠️ **IMPORTANT**: Never commit `camera_config.env` to version control!

- The `.gitignore` file prevents accidental commits of sensitive data
- Use environment variables for all credentials and IP addresses
- Keep your camera credentials secure and private

## Troubleshooting

### Camera Connection Issues

1. **Verify network connectivity** to camera IP
2. **Check ONVIF service** is enabled on camera
3. **Verify credentials** in `camera_config.env`
4. **Try different RTSP paths** (script will attempt fallbacks)

### Performance Issues

1. **Use smaller YOLO model** (yolov5n instead of yolov5s)
2. **Lower confidence threshold** for faster detection
3. **Enable GPU acceleration** if available
4. **Use substream** for lower resolution/faster processing

### Model Loading Issues

1. **Check internet connection** (first run downloads model)
2. **Verify PyTorch installation**
3. **Use local .pt file** if network issues persist

## File Structure

```
obj_detect/
├── camera_yolo_detection.py    # Main detection script
├── camera_config.env           # Camera credentials (not in git)
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Supported Models

- `yolov5n` - Fastest, smallest (recommended for real-time)
- `yolov5s` - Balanced speed/accuracy (default)
- `yolov5m` - Better accuracy, slower
- `yolov5l` - High accuracy, slower
- `yolov5x` - Best accuracy, slowest

## License

This project is for educational and personal use. Ensure compliance with your camera's terms of service and local privacy laws.
