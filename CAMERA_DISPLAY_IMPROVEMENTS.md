# Camera Display Improvements - Full Image Visibility + Motion Detection

## Overview

This document describes the improvements made to ensure that the ASECAM house camera displays the **full image available** without cropping or scaling issues, and now includes **automatic motion detection with picture capture** and a 10-second cooldown.

## Problem Solved

Previously, the camera display windows could:
- Be too large for the screen, causing parts of the image to be hidden
- Not maintain proper aspect ratios
- Not show the complete camera feed
- Have inconsistent window sizing
- **NEW**: No automatic motion detection or picture capture

## Solutions Implemented

### 1. Resizable Windows (`cv2.WINDOW_NORMAL`)

All camera display windows now use `cv2.WINDOW_NORMAL` which allows users to:
- Resize the window using the mouse
- Move the window around the screen
- See the full image by adjusting the window size

### 2. Smart Window Sizing

- **Initial Size**: Windows start at a reasonable size (1280x720) that fits most screens
- **Screen Centering**: Windows are automatically positioned in the center of the screen
- **Resolution Awareness**: Window size adapts to the camera's actual resolution

### 3. Full Frame Validation

The system now:
- Detects the camera's actual resolution
- Warns if frame sizes don't match expected dimensions
- Ensures no image data is lost during processing

### 4. Enhanced Information Display

Each camera view now shows:
- **FPS**: Real-time frame rate
- **Resolution**: Actual camera resolution (e.g., "1920x1080")
- **Detections**: Number of objects detected (for YOLO applications)
- **Camera Info**: ASECAM 8MP camera identification
- **User Instructions**: Clear guidance on how to interact with the window
- **NEW**: Motion detection status and cooldown timer

### 5. 🆕 Motion Detection & Automatic Picture Capture

The system now automatically:
- **Detects Movement**: Monitors camera frames for motion using advanced computer vision
- **Saves Pictures**: Automatically captures and saves images when motion is detected
- **10-Second Cooldown**: Prevents excessive captures with configurable cooldown period
- **Smart Filtering**: Ignores minor movements and noise
- **Organized Storage**: Saves images with timestamps in dedicated directories

## Files Modified

### 1. `camera_utils.py` (UPDATED)
- **Purpose**: Centralized utility functions for camera display
- **Key Functions**:
  - `get_screen_resolution()`: Detect screen dimensions
  - `create_resizable_window()`: Create properly sized windows
  - `setup_camera_display()`: Complete camera display setup
  - `ensure_full_frame_visible()`: Validate frame dimensions
  - `add_camera_info_overlay()`: Add comprehensive information overlay
  - **NEW**: `draw_motion_contours()`: Visualize motion detection areas

### 2. `motion_detector.py` (NEW)
- **Purpose**: Advanced motion detection with automatic picture capture
- **Key Features**:
  - **Motion Detection**: Frame difference analysis with noise reduction
  - **Automatic Capture**: Saves images when motion is detected
  - **Cooldown System**: Configurable time between captures (default: 10 seconds)
  - **Smart Filtering**: Configurable sensitivity and minimum area thresholds
  - **Organized Storage**: Timestamped files in dedicated directories
  - **Debug Support**: Saves motion masks for troubleshooting

### 3. `camera_yolo_detection.py` (UPDATED)
- **Purpose**: Main ASECAM camera object detection application
- **Improvements**:
  - Uses utility functions for cleaner code
  - Ensures full frame visibility
  - Better error handling and user feedback
  - **NEW**: Integrated motion detection with automatic picture capture
  - **NEW**: Visual motion indicators and cooldown status

### 4. `connect_onvif_camera.py` (UPDATED)
- **Purpose**: Simple ASECAM camera stream viewer
- **Improvements**:
  - Resizable display window
  - Full image visibility
  - Enhanced information overlay
  - **NEW**: Motion detection with automatic picture capture

### 5. `test_motion_detection.py` (NEW)
- **Purpose**: Test script to verify motion detection functionality
- **Features**:
  - Creates synthetic motion for testing
  - Tests different detector configurations
  - Visual feedback and statistics
  - Interactive controls (reset cooldown, quit)

## How to Use

### 1. Basic Camera Viewing with Motion Detection
```bash
python connect_onvif_camera.py
```

### 2. Object Detection with Motion Capture
```bash
python camera_yolo_detection.py
```

### 3. Test Motion Detection System
```bash
python test_motion_detection.py
```

### 4. Test Display Utilities
```bash
python test_camera_display.py
```

## Motion Detection Features

### Automatic Picture Capture
- **Trigger**: Movement detected in camera view
- **Cooldown**: 10 seconds between captures (configurable)
- **Storage**: Organized in timestamped directories
- **Format**: High-quality JPEG with motion annotations

### Motion Detection Settings
- **Sensitivity**: Configurable threshold (default: 25.0)
- **Minimum Area**: Ignores tiny movements (default: 500 pixels)
- **Noise Reduction**: Advanced filtering for reliable detection
- **Cooldown**: Prevents excessive captures

### Visual Feedback
- **Motion Contours**: Green outlines around moving areas
- **Status Display**: Real-time motion detection status
- **Cooldown Timer**: Shows time remaining until next capture
- **Capture Notifications**: Console messages for each saved image

## User Interface Features

### Window Controls
- **Resize**: Drag window corners/edges to resize
- **Move**: Click and drag window title bar
- **Maximize**: Double-click title bar or use maximize button
- **Close**: Press 'q' key or close button

### Information Display
- **Top Left**: FPS and detection count
- **Top Center**: Motion detection status
- **Top Right**: Cooldown timer (when active)
- **Bottom Left**: Camera model and IP address
- **Bottom Center**: Current resolution
- **Bottom Right**: User instructions

### Visual Indicators
- **Green Text**: Performance metrics (FPS, detections)
- **Green Contours**: Motion detection areas
- **White Text**: Camera information
- **Cyan Text**: User instructions and additional info
- **Red Text**: Motion detection alerts

## Technical Details

### Motion Detection Algorithm
1. **Frame Conversion**: Convert to grayscale for processing
2. **Noise Reduction**: Apply Gaussian blur to reduce noise
3. **Frame Difference**: Calculate absolute difference between frames
4. **Thresholding**: Apply binary threshold to identify motion
5. **Morphological Operations**: Clean up noise with opening/closing
6. **Contour Detection**: Find significant motion areas
7. **Area Filtering**: Ignore movements below minimum threshold

### Picture Capture System
- **Automatic Triggering**: Motion detection triggers capture
- **Cooldown Management**: Prevents excessive captures
- **File Organization**: Timestamped directories and filenames
- **Quality Preservation**: Full resolution images saved
- **Metadata Overlay**: Motion information and timestamps

### Screen Resolution Detection
- Attempts to detect actual screen resolution
- Falls back to common resolutions (1920x1080) if detection fails
- Ensures windows fit within screen bounds

### Window Management
- Uses OpenCV's `WINDOW_NORMAL` flag for resizable windows
- Automatically centers windows on screen
- Prevents windows from being larger than the screen

### Frame Processing
- Validates frame dimensions against camera properties
- Warns about resolution mismatches
- Maintains aspect ratio during any necessary scaling

## Configuration Options

### Motion Detection Settings
```python
# In your camera scripts, you can customize:
motion_detector = create_motion_detector(
    cooldown_seconds=10.0,      # Time between captures
    motion_threshold=25.0,      # Sensitivity (lower = more sensitive)
    min_area=500,               # Minimum motion area in pixels
    save_directory="custom_path" # Custom save location
)
```

### Environment Variables
```bash
# camera_config.env
CAMERA_HOST=192.168.1.10
CAMERA_PORT=80
CAMERA_USERNAME=admin
CAMERA_PASSWORD=your_password
YOLO_MODEL=yolov5s
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.45
```

## Troubleshooting

### Common Issues

1. **Window Too Large**
   - Solution: Resize the window using mouse or press 'q' to quit and restart

2. **Image Not Visible**
   - Check camera connection and network settings
   - Verify camera credentials in `camera_config.env`

3. **Poor Performance**
   - Lower camera resolution if possible
   - Check network bandwidth for RTSP stream
   - Adjust motion detection sensitivity

4. **Too Many Motion Captures**
   - Increase cooldown period
   - Increase motion threshold
   - Increase minimum area threshold

5. **No Motion Detection**
   - Check motion detection settings
   - Verify camera is providing stable video
   - Test with motion detection test script

6. **Display Errors**
   - Ensure OpenCV is properly installed
   - Check if display drivers support the required features

### Debug Information
The system provides detailed logging:
- Camera connection status
- Stream resolution and FPS
- Frame processing warnings
- Motion detection events
- Picture capture confirmations
- Error messages with context

### Motion Detection Testing
Use the test script to verify functionality:
```bash
python test_motion_detection.py
```
This creates synthetic motion and tests all detection features.

## Future Enhancements

### Planned Improvements
1. **Multi-Monitor Support**: Better handling of multiple displays
2. **Custom Window Layouts**: Save/restore window positions
3. **Resolution Presets**: Quick switching between common resolutions
4. **Performance Monitoring**: Real-time performance metrics
5. **Recording Integration**: Save video with full resolution
6. **Advanced Motion Detection**: AI-powered motion analysis
7. **Cloud Storage**: Automatic upload of motion captures
8. **Alert System**: Email/SMS notifications for motion events

### Configuration Options
- Custom window sizes
- Display preferences
- Overlay customization
- Performance tuning parameters
- Motion detection sensitivity profiles
- Capture quality settings

## Conclusion

These improvements ensure that users can see the **complete ASECAM camera feed** without any cropping or scaling issues, while also providing **intelligent motion detection with automatic picture capture**. The resizable windows and intelligent sizing make it easy to view the full image while maintaining good performance and user experience.

The **motion detection system** automatically monitors for movement and saves high-quality images with a 10-second cooldown, making it perfect for security monitoring and activity tracking. The modular utility functions also make it easier to maintain and extend the camera display functionality across different applications in the project.

### Key Benefits
- ✅ **Full Image Visibility**: See complete camera feed without cropping
- ✅ **Automatic Motion Detection**: Intelligent movement detection
- ✅ **Automatic Picture Capture**: Saves images when motion detected
- ✅ **10-Second Cooldown**: Prevents excessive captures
- ✅ **Professional Interface**: Clean, informative display
- ✅ **Easy Configuration**: Customizable detection settings
- ✅ **Organized Storage**: Timestamped files in dedicated directories
