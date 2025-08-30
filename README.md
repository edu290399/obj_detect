# ASECAM Camera Object Detection with YOLOv5

This project provides real-time object detection using a pre-trained YOLOv5 model connected to an ASECAM 8MP IP camera via ONVIF/RTSP.

## Features

- **Real-time object detection** from IP camera stream
- **ONVIF camera discovery** with fallback RTSP paths
- **Secure credential management** using environment variables
- **Automatic reconnection** if stream drops
- **Performance metrics** (FPS, detection count)
- **GPU acceleration** support (CUDA)
- **🚪 Door tracking and person counting** with automatic entry/exit detection
- **📸 Motion detection** with automatic image capture and 10-second cooldown
- **🎯 Full image visibility** ensuring complete camera feed display

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

### 3. Configure Door Boundary (Optional)

If you want to track people entering/leaving your house:

```bash
# Interactive door boundary configuration
python test_door_config.py

# Copy example door config
cp door_config.env.example door_config.env
```

### 4. Run Object Detection

```bash
# Full system with motion detection and door tracking
python camera_yolo_detection.py

# Simple camera view with motion detection
python connect_onvif_camera.py
```

## Usage

### Basic Object Detection
- **Press 'q'** to quit the detection window
- **FPS display** shows real-time performance
- **Detection count** shows number of objects detected
- **Bounding boxes** with labels and confidence scores

### Door Tracking & Person Counting
- **Automatic counting** of people entering/leaving
- **Visual door boundary** with cyan line
- **Person tracking** with green movement paths
- **Real-time count overlay** showing entries/exits
- **Event logging** to `door_logs/` directory

### Motion Detection
- **Automatic capture** when motion is detected
- **10-second cooldown** between captures
- **Motion contours** displayed in green
- **Images saved** to timestamped directories

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

## Advanced Features

### Door Tracking System
- **`door_tracker.py`**: Core door tracking and person counting logic
- **`test_door_config.py`**: Interactive door boundary configuration tool
- **`door_config.env.example`**: Configuration template for door settings
- **`DOOR_TRACKING_README.md`**: Comprehensive door tracking documentation

### Motion Detection System
- **`motion_detector.py`**: Motion detection with automatic capture
- **`motion_config.env.example`**: Motion detection configuration
- **`test_motion_detection.py`**: Motion detection testing script

### Camera Utilities
- **`camera_utils.py`**: Display management and overlay functions
- **`test_camera_display.py`**: Camera display testing script

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
