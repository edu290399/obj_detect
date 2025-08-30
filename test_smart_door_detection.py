#!/usr/bin/env python3
"""
Test Script for Smart Door Detection System

This script tests the automatic door detection and person tracking capabilities.
"""

import cv2
import numpy as np
import time
from smart_door_detector import create_smart_door_detector

def create_test_image():
    """Create a test image with a door and people."""
    # Create a 1080p test image
    img = np.ones((1080, 1920, 3), dtype=np.uint8) * 120
    
    # Draw textured wall
    for i in range(0, 1920, 30):
        for j in range(0, 1080, 30):
            color_variation = np.random.randint(-30, 30)
            wall_color = (120 + color_variation, 120 + color_variation, 100 + color_variation)
            cv2.rectangle(img, (i, j), (i + 30, j + 30), wall_color, -1)
    
    # Draw door frame (dark brown)
    door_x = 1400
    door_y = 300
    door_w = 300
    door_h = 600
    
    cv2.rectangle(img, (door_x, door_y), (door_x + door_w, door_y + door_h), (60, 30, 20), -1)
    
    # Draw door opening (darker interior)
    cv2.rectangle(img, (door_x + 20, door_y + 20), 
                 (door_x + door_w - 20, door_y + door_h - 20), (40, 20, 10), -1)
    
    # Draw tiled floor inside
    for i in range(door_x + 20, door_x + door_w - 20, 40):
        for j in range(door_y + door_h - 20, 1080, 40):
            tile_color = (180 + np.random.randint(-20, 20), 
                         180 + np.random.randint(-20, 20), 
                         180 + np.random.randint(-20, 20))
            cv2.rectangle(img, (i, j), (i + 40, j + 40), tile_color, -1)
            cv2.rectangle(img, (i, j), (i + 40, j + 40), (150, 150, 150), 1)
    
    # Draw threshold/mat
    cv2.rectangle(img, (door_x - 50, door_y + door_h), 
                 (door_x + door_w + 50, door_y + door_h + 60), (30, 30, 30), -1)
    
    # Draw some people (simulating detection boxes)
    # Person 1: Entering (facing away from camera)
    person1_x = 1200
    person1_y = 800
    person1_w = 80
    person1_h = 200
    cv2.rectangle(img, (person1_x, person1_y), 
                 (person1_x + person1_w, person1_y + person1_h), (0, 255, 0), 2)
    cv2.putText(img, "Person 1 (Entering)", (person1_x, person1_y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Person 2: Exiting (facing camera)
    person2_x = 1600
    person2_y = 900
    person2_w = 80
    person2_h = 200
    cv2.rectangle(img, (person2_x, person2_y), 
                 (person2_x + person2_w, person2_y + person2_h), (0, 0, 255), 2)
    cv2.putText(img, "Person 2 (Exiting)", (person2_x, person2_y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Add title
    cv2.putText(img, "Smart Door Detection Test", (50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(img, "Green: Entering, Red: Exiting", (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return img

def simulate_person_movement(detector, frame):
    """Simulate person movement through the door area."""
    # Simulate Person 1 entering (moving towards door)
    person1_boxes = [
        [1200, 800, 1280, 1000],  # Starting position
        [1300, 750, 1380, 950],   # Moving towards door
        [1400, 700, 1480, 900],   # At door threshold
        [1500, 650, 1580, 850],   # Through door
    ]
    
    # Simulate Person 2 exiting (moving away from door)
    person2_boxes = [
        [1600, 900, 1680, 1100],  # Starting position
        [1500, 850, 1580, 1050],  # Moving towards door
        [1400, 800, 1480, 1000],  # At door threshold
        [1300, 750, 1380, 950],   # Through door
    ]
    
    print("🚶 Simulating Person 1 entering...")
    for i, box in enumerate(person1_boxes):
        detector.update_tracks([box], frame)
        time.sleep(0.5)
        print(f"   Frame {i+1}: Position {box[:2]}")
    
    print("🚶 Simulating Person 2 exiting...")
    for i, box in enumerate(person2_boxes):
        detector.update_tracks([box], frame)
        time.sleep(0.5)
        print(f"   Frame {i+1}: Position {box[:2]}")
    
    # Get final stats
    stats = detector.get_stats()
    print(f"\n📊 Final Statistics:")
    print(f"   Entries: {stats['entries']}")
    print(f"   Exits: {stats['exits']}")
    print(f"   Active Tracks: {stats['active_tracks']}")
    print(f"   Detected Doors: {stats['detected_doors']}")

def main():
    """Main test function."""
    print("🚪 Smart Door Detection Test")
    print("=" * 40)
    
    # Create test image
    test_image = create_test_image()
    
    # Create smart door detector
    detector = create_smart_door_detector()
    
    print("🔍 Detecting doors in test image...")
    
    # Detect doors
    doors = detector.detect_doors(test_image)
    print(f"✅ Detected {len(doors)} door passages:")
    
    for i, door in enumerate(doors):
        print(f"   Door {i+1}: {door} (confidence: {door.confidence:.2f})")
    
    # Draw door boundaries
    result_image = detector.draw_door_boundaries(test_image)
    
    # Simulate person movement
    print("\n🚶 Simulating person movement...")
    simulate_person_movement(detector, test_image)
    
    # Draw final visualization
    result_image = detector.draw_person_tracks(result_image)
    result_image = detector.draw_count_overlay(result_image)
    
    # Show results
    cv2.namedWindow("Smart Door Detection Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Smart Door Detection Test", 1200, 800)
    cv2.imshow("Smart Door Detection Test", result_image)
    
    print("\n🖼️ Test image displayed. Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("✅ Test completed!")

if __name__ == "__main__":
    main()
