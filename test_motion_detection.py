#!/usr/bin/env python3
"""
Test script to verify motion detection functionality.
This script creates synthetic motion and tests the motion detector.
"""

import cv2
import numpy as np
import time
import os
from motion_detector import create_motion_detector


def create_test_scene(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a simple test scene."""
    # Create a simple background
    scene = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    # Add some static elements
    cv2.rectangle(scene, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(scene, (400, 300), 50, (0, 255, 0), -1)  # Green circle
    
    # Add some text
    cv2.putText(scene, "Test Scene", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return scene


def add_moving_object(scene: np.ndarray, frame_num: int, width: int = 640, height: int = 480) -> np.ndarray:
    """Add a moving object to the scene."""
    frame = scene.copy()
    
    # Create a moving red square
    x = int(50 + (frame_num * 2) % (width - 100))
    y = int(150 + int(np.sin(frame_num * 0.1) * 50))
    
    cv2.rectangle(frame, (x, y), (x + 30, y + 30), (0, 0, 255), -1)
    
    # Add motion indicator
    cv2.putText(frame, f"Moving Object - Frame {frame_num}", (10, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame


def test_motion_detection():
    """Test motion detection with synthetic video."""
    print("Testing Motion Detection System")
    print("=" * 40)
    
    # Create motion detector with 5-second cooldown for testing
    motion_detector = create_motion_detector(cooldown_seconds=5.0)
    print(f"Motion detector created with {motion_detector.cooldown_seconds}s cooldown")
    print(f"Saves to: {motion_detector.save_directory}")
    
    # Create test scene
    width, height = 640, 480
    base_scene = create_test_scene(width, height)
    
    # Create test window
    window_name = "Motion Detection Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    frame_count = 0
    motion_count = 0
    capture_count = 0
    
    try:
        print("\nStarting motion detection test...")
        print("Press 'q' to quit, 'r' to reset cooldown")
        print("Moving red square should trigger motion detection")
        
        while True:
            # Create frame with moving object
            frame = add_moving_object(base_scene, frame_count, width, height)
            
            # Process motion detection
            motion_detected, saved_path = motion_detector.process_frame(frame)
            
            if motion_detected:
                motion_count += 1
                if saved_path:
                    capture_count += 1
                    print(f"📸 Frame {frame_count}: Motion captured! ({capture_count} total)")
                else:
                    cooldown_status = motion_detector.get_cooldown_status()
                    print(f"⏰ Frame {frame_count}: Motion detected but in cooldown ({cooldown_status})")
            
            # Add motion detection overlay
            motion_status = "MOTION DETECTED!" if motion_detected else "No Motion"
            motion_cooldown_status = motion_detector.get_cooldown_status()
            
            # Draw motion contours if motion was detected
            if motion_detected:
                _, motion_mask, contours = motion_detector.detect_motion(frame)
                if contours:
                    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            
            # Add information overlay
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Motion: {motion_status}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if motion_detected else (0, 255, 255), 2)
            
            if motion_cooldown_status:
                cv2.putText(frame, f"Cooldown: {motion_cooldown_status}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            cv2.putText(frame, f"Motion Events: {motion_count} | Captures: {capture_count}", (10, height - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit, 'r' to reset cooldown", (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Handle key presses
            key = cv2.waitKey(100) & 0xFF  # 100ms delay for 10 FPS
            if key == ord('q'):
                break
            elif key == ord('r'):
                motion_detector.reset_cooldown()
                print("🔄 Cooldown reset!")
            
            frame_count += 1
            
            # Limit test to 300 frames (30 seconds at 10 FPS)
            if frame_count >= 300:
                print("\nTest completed after 300 frames")
                break
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
    finally:
        # Get final statistics
        stats = motion_detector.get_stats()
        print(f"\nFinal Statistics:")
        print(f"  Total frames processed: {stats['frame_count']}")
        print(f"  Motion events detected: {motion_count}")
        print(f"  Images captured: {capture_count}")
        print(f"  Save directory: {stats['save_directory']}")
        
        # Cleanup
        motion_detector.cleanup()
        cv2.destroyAllWindows()
        print("\nMotion detection test completed!")


def test_motion_detector_settings():
    """Test different motion detector settings."""
    print("\nTesting Motion Detector Settings")
    print("=" * 40)
    
    # Test different configurations
    configs = [
        {"cooldown": 3.0, "threshold": 15.0, "min_area": 200},
        {"cooldown": 5.0, "threshold": 25.0, "min_area": 500},
        {"cooldown": 10.0, "threshold": 35.0, "min_area": 1000},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nTest {i+1}: Cooldown={config['cooldown']}s, Threshold={config['threshold']}, MinArea={config['min_area']}")
        
        detector = create_motion_detector(
            cooldown_seconds=config['cooldown'],
            motion_threshold=config['threshold'],
            min_area=config['min_area']
        )
        
        print(f"  Created detector with settings:")
        print(f"    Cooldown: {detector.cooldown_seconds}s")
        print(f"    Threshold: {detector.motion_threshold}")
        print(f"    Min Area: {detector.min_area}")
        print(f"    Save Directory: {detector.save_directory}")
        
        detector.cleanup()


if __name__ == "__main__":
    try:
        test_motion_detector_settings()
        test_motion_detection()
    except Exception as e:
        print(f"Test failed: {e}")
        cv2.destroyAllWindows()
