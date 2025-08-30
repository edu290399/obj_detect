# 🚪 Door Tracking & Person Counting System

This system automatically tracks people entering and leaving your house by detecting when they cross a defined door boundary in your ASECAM camera feed.

## ✨ Features

- **Automatic Person Detection**: Uses YOLOv5 to detect people in real-time
- **Door Boundary Tracking**: Configurable line that counts entries/exits
- **Smart Movement Analysis**: Tracks person movement across frames
- **Entry/Exit Counting**: Maintains separate counts for people entering vs. leaving
- **Persistent Logging**: Saves all events and counts to files
- **Visual Feedback**: Shows door boundary, person tracks, and count overlay
- **Easy Configuration**: Interactive tool to set door boundary

## 🚀 Quick Start

### 1. Run the Full System
```bash
python camera_yolo_detection.py
```

This will:
- Connect to your ASECAM camera
- Run YOLOv5 object detection
- Detect motion and save captures
- **Track people crossing the door boundary**
- **Count entries and exits automatically**

### 2. Configure Door Boundary (First Time)
```bash
python test_door_config.py
```

This interactive tool helps you:
- Draw the door boundary line
- Adjust the position visually
- Save the configuration

## ⚙️ Configuration

### Door Boundary Settings
Create `door_config.env` file (or use the configuration tool):

```env
# Door boundary coordinates (in pixels)
DOOR_X1=200
DOOR_Y1=300
DOOR_X2=400
DOOR_Y2=300

# Door name/label
DOOR_NAME=Main Door

# Tracking settings
TRACKING_FRAMES=30
MIN_CROSSING_DISTANCE=20

# Save directory for logs and counts
SAVE_DIRECTORY=door_logs
```

### How to Set Door Boundary

1. **Run the configuration tool**:
   ```bash
   python test_door_config.py
   ```

2. **Draw the door line**:
   - Left click to set the door boundary
   - Drag endpoints to adjust
   - The line should cross where people enter/exit

3. **Save configuration**:
   - Press 's' to save
   - Press 'r' to reset to defaults
   - Press 'q' to quit

## 📊 How It Works

### 1. Person Detection
- YOLOv5 detects people in each frame
- Each person gets a unique ID
- Position is tracked across frames

### 2. Movement Tracking
- System follows each person's movement
- Creates a path of their movement
- Maintains tracking for 30 frames (configurable)

### 3. Door Crossing Detection
- Calculates when someone crosses the door line
- Uses perpendicular distance calculation
- Determines entry vs. exit direction

### 4. Counting Logic
- **Entry**: Person moves from outside to inside
- **Exit**: Person moves from inside to outside
- Counts are updated in real-time
- Events are logged with timestamps

## 🎯 Visual Indicators

### Door Boundary
- **Cyan line**: The door boundary to cross
- **Cyan circles**: Endpoints of the boundary
- **Magenta circle**: Center of the boundary

### Person Tracking
- **Green lines**: Movement paths of people
- **Green circles**: Current positions
- **Person IDs**: P1, P2, P3, etc.

### Count Overlay
- **Green text**: Number of entries
- **Red text**: Number of exits
- **Real-time updates**: Counts change as people cross

## 📁 Output Files

### Door Counts (`door_logs/door_counts.json`)
```json
{
  "entries": 5,
  "exits": 3,
  "last_updated": "2024-01-15T14:30:25.123456"
}
```

### Event Log (`door_logs/door_events.log`)
```
2024-01-15T14:25:10.123456 - Person 1 in the house
2024-01-15T14:26:15.234567 - Person 2 in the house
2024-01-15T14:28:30.345678 - Person 1 out the house
```

## 🔧 Customization

### Tracking Sensitivity
- **`TRACKING_FRAMES`**: How long to track a person (default: 30)
- **`MIN_CROSSING_DISTANCE`**: Minimum distance to count as crossing (default: 20px)

### Door Boundary
- **Horizontal line**: Good for front doors
- **Vertical line**: Good for side entrances
- **Diagonal line**: Good for angled entrances

### Camera Positioning
- **Outside looking in**: Standard setup
- **Inside looking out**: May need to swap entry/exit logic
- **Side angle**: Adjust boundary line accordingly

## 🚨 Troubleshooting

### Door Boundary Not Working
1. **Check coordinates**: Ensure boundary is visible in camera view
2. **Adjust sensitivity**: Increase `MIN_CROSSING_DISTANCE` if too sensitive
3. **Verify detection**: Make sure people are being detected by YOLOv5

### False Counts
1. **Reposition boundary**: Move away from high-traffic areas
2. **Increase tracking frames**: More frames = more stable tracking
3. **Adjust crossing distance**: Larger distance = more deliberate crossing required

### Performance Issues
1. **Reduce tracking frames**: Lower `TRACKING_FRAMES` value
2. **Check camera resolution**: Lower resolution = faster processing
3. **Verify YOLO model**: Use smaller models (yolov5n, yolov5s) for speed

## 📱 Usage Examples

### Basic Monitoring
```bash
# Start the system
python camera_yolo_detection.py

# Watch the counts update in real-time
# Press 'q' to quit
```

### Configuration Testing
```bash
# Configure door boundary
python test_door_config.py

# Test with your camera
python camera_yolo_detection.py
```

### Log Analysis
```bash
# View current counts
cat door_logs/door_counts.json

# View recent events
tail -f door_logs/door_events.log
```

## 🔒 Privacy & Security

- **Local Processing**: All detection happens on your device
- **No Cloud Uploads**: Data stays private
- **Configurable Logging**: Choose what to save
- **Secure Storage**: Logs stored in local directory

## 🆘 Support

### Common Issues
1. **Camera not connecting**: Check network and credentials
2. **No person detection**: Verify YOLOv5 model is loaded
3. **Counts not updating**: Check door boundary position

### Debug Mode
- Run with verbose output to see detailed tracking
- Check console for person detection status
- Verify door boundary coordinates

## 🎉 What You Get

With this system, you'll have:
- ✅ **Real-time person counting** at your door
- ✅ **Automatic entry/exit detection**
- ✅ **Persistent logging** of all events
- ✅ **Visual tracking** of people's movements
- ✅ **Easy configuration** and customization
- ✅ **Professional monitoring** interface

Your ASECAM camera is now a smart door monitor that automatically tracks everyone coming and going! 🏠👥
