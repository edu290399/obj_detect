# Tello Drone Real-time Object Detection with YOLOv5

A Python-based system for controlling a Ryze Tello drone (TLW004) while performing real-time object detection using YOLOv5. This project combines drone control capabilities with state-of-the-art computer vision for autonomous aerial surveillance and object recognition.

## 🚁 Features

- **Drone Control**: Automated takeoff, hover, and safe landing
- **Real-time Object Detection**: YOLOv5 integration for live video analysis
- **Multiple Video Stream Methods**: Robust fallback approaches for Tello video
- **Safety Features**: Automatic landing on exit, error handling, and battery monitoring
- **Performance Monitoring**: FPS tracking and frame processing statistics
- **Flexible Configuration**: Adjustable confidence thresholds and model weights

## 📋 Prerequisites

- **Hardware**: Ryze Tello drone (TLW004 model)
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: 3.8 or higher
- **Network**: Wi-Fi connection to Tello drone network
- **Memory**: At least 4GB RAM (8GB recommended)
- **Storage**: 2GB free space for models and dependencies

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd obj_detect
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv obj_detect
obj_detect\Scripts\activate

# macOS/Linux
python3 -m venv obj_detect
source obj_detect/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import cv2, torch, djitellopy; print('All dependencies installed successfully!')"
```

## 🔧 Dependencies

The project requires the following key packages:

- **PyTorch** (≥1.9): Deep learning framework for YOLOv5
- **OpenCV** (≥4.5): Computer vision and video processing
- **djitellopy2** (≥2.5.0): Tello drone control library
- **NumPy** (≥1.19): Numerical computing
- **av** (≥10.0.0): Audio/video processing for drone streams

## 🚀 Usage

### Basic Operation

1. **Connect to Tello Wi-Fi**
   - Power on your Tello drone
   - Connect your computer to the Tello Wi-Fi network (SSID: `TELLO-XXXXXX`)
   - Wait for connection to establish

2. **Run the Detection System**
   ```bash
   python tello_yolov5_realtime.py --weights yolov5s --conf 0.25 --iou 0.45
   ```

3. **Flight Sequence**
   - The drone will automatically take off
   - Rise 20cm and hover
   - Begin real-time object detection
   - Press 'q' in the video window to land and exit

### Command Line Options

```bash
python tello_yolov5_realtime.py [OPTIONS]

Options:
  --weights TEXT    YOLOv5 model weights (yolov5n, yolov5s, yolov5m, yolov5l, yolov5x) or path to .pt file
  --conf FLOAT     Confidence threshold (0.0-1.0) [default: 0.25]
  --iou FLOAT      NMS IoU threshold (0.0-1.0) [default: 0.45]
  --help           Show help message
```

### Model Selection

- **yolov5n**: Fastest, lowest accuracy (3.2M parameters)
- **yolov5s**: Balanced speed/accuracy (7.2M parameters) - **Recommended**
- **yolov5m**: Higher accuracy, slower (21.2M parameters)
- **yolov5l**: High accuracy, slower (46.5M parameters)
- **yolov5x**: Highest accuracy, slowest (87.7M parameters)

## 🎮 Controls

- **Automatic**: Drone follows pre-programmed flight path
- **Exit**: Press 'q' in video window to land and exit
- **Emergency**: Use Ctrl+C in terminal for immediate landing

## 🔍 Video Stream Troubleshooting

The system implements multiple fallback methods for video stream initialization:

1. **Standard djitellopy frame reader**
2. **Direct UDP video capture** (port 11111)
3. **Extended timeout initialization**
4. **Complete stream restart**
5. **Fallback to no-video mode**

### Common Video Issues

- **Black screen**: Wait for stream initialization (up to 15 seconds)
- **Stream timeout**: System automatically tries alternative methods
- **No video feed**: Detection continues with console output
- **Connection drops**: Automatic reconnection attempts

## 🛡️ Safety Features

- **Automatic landing** on script exit or error
- **Battery monitoring** with low-battery warnings
- **Error handling** for connection failures
- **Safe shutdown** procedures
- **Emergency stop** via keyboard interrupt

## 📊 Performance

### System Requirements

- **CPU**: Intel i5/AMD Ryzen 5 or better
- **GPU**: NVIDIA GPU with CUDA support (optional, for acceleration)
- **RAM**: 4GB minimum, 8GB recommended
- **Network**: Stable Wi-Fi connection to drone

### Performance Tips

- Use `yolov5n` for maximum FPS
- Enable CUDA if available for GPU acceleration
- Close unnecessary applications during operation
- Ensure stable Wi-Fi connection
- Monitor battery level (minimum 20% recommended)

## 🐛 Troubleshooting

### Connection Issues

```bash
# Check Wi-Fi connection
netsh wlan show interfaces  # Windows
iwconfig                    # Linux
ifconfig                    # macOS

# Verify drone IP reachability
ping 192.168.10.1
```

### Video Stream Problems

1. **Restart drone and reconnect**
2. **Check firewall settings** (allow UDP port 11111)
3. **Verify network adapter settings**
4. **Try different video initialization methods**

### Import Errors

```bash
# Reinstall dependencies
pip uninstall -r requirements.txt
pip install -r requirements.txt

# Check Python version
python --version
```

### Performance Issues

- Lower confidence threshold (`--conf 0.15`)
- Use smaller model (`--weights yolov5n`)
- Close background applications
- Check system resource usage

## 🔧 Development

### Project Structure

```
obj_detect/
├── tello_yolov5_realtime.py  # Main application
├── realtime_yolov5.py        # Webcam-based YOLOv5 demo
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

### Key Functions

- `tello_connect()`: Establish drone connection
- `load_yolov5_model()`: Load and configure YOLOv5
- `detect_objects()`: Run object detection inference
- `run_tello_detection()`: Main detection loop
- `ensure_safe_land()`: Safety context manager

### Extending the Project

- Add manual flight controls (WASD keys)
- Implement autonomous navigation
- Add recording capabilities
- Integrate with other AI models
- Add web interface for remote monitoring

## 📚 Technical Details

### YOLOv5 Architecture

- **Model**: Single-stage object detector
- **Input**: RGB images (640x640 by default)
- **Output**: Bounding boxes, class labels, confidence scores
- **Classes**: 80 COCO dataset classes (person, car, dog, etc.)

### Drone Communication

- **Protocol**: UDP for video, TCP for control
- **Video**: H.264 encoded stream at 720p
- **Control**: Command-based interface
- **Ports**: 8889 (control), 11111 (video)

### Video Processing Pipeline

1. **Frame Capture**: From Tello video stream
2. **Preprocessing**: Resize and normalize for YOLOv5
3. **Inference**: Run YOLOv5 detection
4. **Post-processing**: NMS and confidence filtering
5. **Visualization**: Draw bounding boxes and labels
6. **Display**: Show processed video feed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

- **Safety First**: Always operate drones in safe, open areas
- **Legal Compliance**: Follow local drone regulations
- **Risk Awareness**: Drone operation involves inherent risks
- **Testing**: Test in controlled environments before production use
- **Supervision**: Never leave drones unattended during operation

## 📞 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the error logs and console output
3. Verify your setup matches the requirements
4. Check the project's issue tracker
5. Contact the development team

## 🔄 Version History

- **v1.0.0**: Initial release with basic drone control and YOLOv5 integration
- **v1.1.0**: Added robust video stream handling and fallback methods
- **v1.2.0**: Enhanced error handling and safety features

---

**Happy Flying! 🚁✨**
