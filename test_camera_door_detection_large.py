#!/usr/bin/env python3
"""
Real-time Camera Door Detection with Large Display

This script tests the smart door detection system using a real camera
with a large display optimized for 1920x1080 screens.
"""

import cv2
import numpy as np
import time
from datetime import datetime
import os
from smart_door_detector import create_smart_door_detector
from camera_utils import setup_camera_display, add_camera_info_overlay

def load_camera_config():
    """Load camera configuration from environment file."""
    from dotenv import load_dotenv
    load_dotenv('camera_config.env')
    
    return {
        'host': os.getenv('CAMERA_HOST', '192.168.1.10'),
        'port': int(os.getenv('CAMERA_PORT', '80')),
        'username': os.getenv('CAMERA_USERNAME', 'admin'),
        'password': os.getenv('CAMERA_PASSWORD', 'your_password_here'),
        'rtsp_url': os.getenv('RTSP_URL', None)
    }

def create_camera_stream(camera_config):
    """Create camera stream from configuration."""
    if camera_config['rtsp_url']:
        # Use direct RTSP URL if provided
        cap = cv2.VideoCapture(camera_config['rtsp_url'])
    else:
        # Try to create RTSP URL from ONVIF config
        rtsp_url = f"rtsp://{camera_config['username']}:{camera_config['password']}@{camera_config['host']}:554/stream1"
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            # Try alternative stream paths
            alt_paths = [
                f"rtsp://{camera_config['username']}:{camera_config['password']}@{camera_config['host']}:554/stream2",
                f"rtsp://{camera_config['username']}:{camera_config['password']}@{camera_config['host']}:554/h264Preview_01_main",
                f"rtsp://{camera_config['username']}:{camera_config['password']}@{camera_config['host']}:554/h264Preview_01_sub"
            ]
            
            for alt_path in alt_paths:
                cap = cv2.VideoCapture(alt_path)
                if cap.isOpened():
                    print(f"✅ Connected using: {alt_path}")
                    break
    
    if not cap.isOpened():
        print("❌ Could not connect to camera. Using webcam as fallback...")
        cap = cv2.VideoCapture(0)  # Use default webcam
    
    return cap

def setup_large_display():
    """Setup large display window optimized for 1920x1080."""
    window_name = "Real-time Door Detection - Large Display"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Set window size to nearly full screen (leaving some margin)
    cv2.resizeWindow(window_name, 1800, 1000)
    
    # Position window in center of screen
    cv2.moveWindow(window_name, 60, 40)
    
    return window_name

def add_large_overlay(frame, detector, fps, frame_count):
    """Add large, visible overlay information."""
    output = frame.copy()
    
    # Create semi-transparent overlay for stats
    overlay = output.copy()
    cv2.rectangle(overlay, (50, 50), (400, 250), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    
    # Get door statistics
    stats = detector.get_stats()
    
    # Large, clear text for visibility
    cv2.putText(output, "DOOR DETECTION SYSTEM", (70, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    cv2.putText(output, f"ENTRIES: {stats['entries']}", (70, 140),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    cv2.putText(output, f"EXITS: {stats['exits']}", (70, 180),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    cv2.putText(output, f"FPS: {fps:.1f}", (70, 220),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add current time
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(output, current_time, (70, 250),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add instructions
    cv2.putText(output, "Press 'q' to quit", (50, 1000),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return output

def main():
    """Main function for real-time camera door detection."""
    print("🚪 Real-time Camera Door Detection with Large Display")
    print("=" * 60)
    print("This will test the door detection system using your camera")
    print("with a large display optimized for 1920x1080 screens.")
    print()
    
    # Load camera configuration
    print("📷 Loading camera configuration...")
    camera_config = load_camera_config()
    
    # Create camera stream
    print("🔌 Connecting to camera...")
    cap = create_camera_stream(camera_config)
    
    if not cap.isOpened():
        print("❌ Failed to open camera. Exiting...")
        return
    
    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_camera = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Camera connected: {frame_width}x{frame_height} @ {fps_camera:.1f} FPS")
    
    # Create smart door detector
    print("🔍 Initializing smart door detector...")
    detector = create_smart_door_detector()
    
    # Setup large display
    window_name = setup_large_display()
    
    print("\n🖼️ Large display window created!")
    print("📋 Instructions:")
    print("   - Camera feed will be displayed in large window")
    print("   - Door detection will run automatically")
    print("   - Person tracking and counting will be shown")
    print("   - Press 'q' to quit")
    print()
    
    # Performance tracking
    frame_count = 0
    prev_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from camera")
                break
            
            # Resize frame to 1080p for better processing and display
            frame = cv2.resize(frame, (1920, 1080))
            
            # Detect doors (run every 30 frames to avoid performance issues)
            if frame_count % 30 == 0:
                doors = detector.detect_doors(frame)
                if doors:
                    print(f"🔍 Detected {len(doors)} door passages")
            
            # Simulate person detection (in real system, this would come from YOLO)
            # For testing, we'll create some simulated person boxes
            if frame_count % 60 == 0:  # Every 2 seconds at 30 FPS
                # Create simulated person detection boxes
                person_boxes = []
                
                # Simulate person moving through door area
                if frame_count % 120 == 0:  # Every 4 seconds
                    # Simulate person entering
                    person_boxes.append([800, 600, 900, 800])
                    print("🚶 Simulated person detected (entering)")
                elif frame_count % 120 == 60:  # Every 4 seconds, offset
                    # Simulate person exiting
                    person_boxes.append([900, 500, 1000, 700])
                    print("🚶 Simulated person detected (exiting)")
                
                if person_boxes:
                    detector.update_tracks(person_boxes, frame)
            
            # Calculate FPS
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_time))
            prev_time = now
            frame_count += 1
            
            # Draw door tracking visualizations
            vis = detector.draw_door_boundaries(frame)
            vis = detector.draw_person_tracks(vis)
            
            # Add large overlay
            vis = add_large_overlay(vis, detector, fps, frame_count)
            
            # Display frame
            cv2.imshow(window_name, vis)
            
            # Exit on 'q'
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    finally:
        cap.release()
        detector.save_counts()
        cv2.destroyAllWindows()
        print("✅ Camera door detection test completed!")

if __name__ == "__main__":
    main()
