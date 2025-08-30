#!/usr/bin/env python3
"""
Large Display Door Detection Test

This script tests the smart door detection system with a large display
optimized for 1920x1080 screens. It shows the door detection, person
tracking, and counting in real-time with maximum visibility.
"""

import cv2
import numpy as np
import time
from datetime import datetime
from smart_door_detector import create_smart_door_detector

def create_large_test_image():
    """Create a large test image optimized for 1920x1080 display."""
    # Create a 1080p test image
    img = np.ones((1080, 1920, 3), dtype=np.uint8) * 120
    
    # Draw textured wall with larger patterns for better visibility
    for i in range(0, 1920, 50):  # Larger wall tiles
        for j in range(0, 1080, 50):
            color_variation = np.random.randint(-40, 40)
            wall_color = (120 + color_variation, 120 + color_variation, 100 + color_variation)
            cv2.rectangle(img, (i, j), (i + 50, j + 50), wall_color, -1)
            cv2.rectangle(img, (i, j), (i + 50, j + 50), (80, 80, 80), 2)
    
    # Draw large door frame (dark brown) - centered and prominent
    # Make it more rectangular and door-like for detection
    door_x = 800
    door_y = 200
    door_w = 300  # Reduced width to improve aspect ratio
    door_h = 700  # Keep height for good aspect ratio
    
    # Door frame with strong edges for detection
    cv2.rectangle(img, (door_x, door_y), (door_x + door_w, door_y + door_h), (60, 30, 20), -1)
    cv2.rectangle(img, (door_x, door_y), (door_x + door_w, door_y + door_h), (30, 15, 10), 8)  # Thicker border
    
    # Door opening (darker interior) - create strong contrast
    cv2.rectangle(img, (door_x + 20, door_y + 20), 
                 (door_x + door_w - 20, door_y + door_h - 20), (20, 10, 5), -1)
    
    # Add door threshold line (thick line for detection)
    threshold_y = door_y + door_h + 50
    cv2.line(img, (door_x - 100, threshold_y), (door_x + door_w + 100, threshold_y), (255, 255, 0), 8)
    
    # Add additional door-like features for better detection
    # Door handle (small rectangle)
    handle_x = door_x + door_w - 50
    handle_y = door_y + door_h // 2
    cv2.rectangle(img, (handle_x, handle_y), (handle_x + 20, handle_y + 40), (139, 69, 19), -1)
    
    # Door frame corners (reinforce rectangular shape)
    corner_size = 30
    cv2.rectangle(img, (door_x, door_y), (door_x + corner_size, door_y + corner_size), (40, 20, 10), -1)
    cv2.rectangle(img, (door_x + door_w - corner_size, door_y), (door_x + door_w, door_y + corner_size), (40, 20, 10), -1)
    cv2.rectangle(img, (door_x, door_y + door_h - corner_size), (door_x + corner_size, door_y + door_h), (40, 20, 10), -1)
    cv2.rectangle(img, (door_x + door_w - corner_size, door_y + door_h - corner_size), (door_x + door_w, door_y + door_h), (40, 20, 10), -1)
    
    # Draw tiled floor inside (larger tiles for visibility)
    for i in range(door_x + 20, door_x + door_w - 20, 60):
        for j in range(door_y + door_h - 20, 1080, 60):
            tile_color = (180 + np.random.randint(-30, 30), 
                         180 + np.random.randint(-30, 30), 
                         180 + np.random.randint(-30, 30))
            cv2.rectangle(img, (i, j), (i + 60, j + 60), tile_color, -1)
            cv2.rectangle(img, (i, j), (i + 60, j + 60), (150, 150, 150), 2)
    
    # Add large, clear title
    cv2.putText(img, "SMART DOOR DETECTION SYSTEM", (100, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
    cv2.putText(img, "Testing Person Detection & Counting", (100, 140),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
    
    # Add instructions
    cv2.putText(img, "Press SPACE to simulate person movement", (100, 200),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(img, "Press 'q' to quit", (100, 230),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return img

def simulate_person_movement_large(detector, frame, display_frame):
    """Simulate person movement with large, visible tracking."""
    print("\n🚶 Simulating Person 1 ENTERING (moving towards door)...")
    
    # Person 1: Entering (moving towards door from left)
    person1_positions = [
        [600, 800, 680, 1000],   # Starting position (left side)
        [700, 750, 780, 950],    # Moving towards door
        [800, 700, 880, 900],    # At door threshold
        [900, 650, 980, 850],    # Through door
        [1000, 600, 1080, 800],  # Inside
    ]
    
    for i, box in enumerate(person1_positions):
        # Update detector
        detector.update_tracks([box], frame)
        
        # Draw current position with large, visible markers
        display = display_frame.copy()
        
        # Draw all previous positions as trail
        for j, prev_box in enumerate(person1_positions[:i+1]):
            alpha = 0.3 + (j / len(person1_positions)) * 0.7
            color = (0, int(255 * alpha), 0)
            cv2.rectangle(display, (prev_box[0], prev_box[1]), 
                         (prev_box[2], prev_box[3]), color, 3)
            cv2.circle(display, ((prev_box[0] + prev_box[2])//2, 
                                (prev_box[1] + prev_box[3])//2), 15, color, -1)
        
        # Draw current position prominently
        cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 5)
        cv2.circle(display, ((box[0] + box[2])//2, (box[1] + box[3])//2), 20, (0, 255, 0), -1)
        cv2.putText(display, f"ENTERING {i+1}/5", (box[0], box[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Add door tracking visualization
        display = detector.draw_door_boundaries(display)
        display = detector.draw_person_tracks(display)
        display = detector.draw_count_overlay(display)
        
        # Show frame
        cv2.imshow("Large Door Detection Test", display)
        cv2.waitKey(500)  # Wait 500ms between frames
        
        print(f"   Position {i+1}: {box[:2]} -> Door threshold")
    
    print("🚶 Simulating Person 2 EXITING (moving away from door)...")
    
    # Person 2: Exiting (moving away from door to right)
    person2_positions = [
        [1000, 600, 1080, 800],  # Starting position (inside)
        [900, 650, 980, 850],    # Moving towards door
        [800, 700, 880, 900],    # At door threshold
        [700, 750, 780, 950],    # Through door
        [600, 800, 680, 1000],   # Outside
    ]
    
    for i, box in enumerate(person2_positions):
        # Update detector
        detector.update_tracks([box], frame)
        
        # Draw current position with large, visible markers
        display = display_frame.copy()
        
        # Draw all previous positions as trail
        for j, prev_box in enumerate(person2_positions[:i+1]):
            alpha = 0.3 + (j / len(person2_positions)) * 0.7
            color = (0, 0, int(255 * alpha))
            cv2.rectangle(display, (prev_box[0], prev_box[1]), 
                         (prev_box[2], prev_box[3]), color, 3)
            cv2.circle(display, ((prev_box[0] + prev_box[2])//2, 
                                (prev_box[1] + prev_box[3])//2), 15, color, -1)
        
        # Draw current position prominently
        cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 5)
        cv2.circle(display, ((box[0] + box[2])//2, (box[1] + box[3])//2), 20, (0, 0, 255), -1)
        cv2.putText(display, f"EXITING {i+1}/5", (box[0], box[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Add door tracking visualization
        display = detector.draw_door_boundaries(display)
        display = detector.draw_person_tracks(display)
        display = detector.draw_count_overlay(display)
        
        # Show frame
        cv2.imshow("Large Door Detection Test", display)
        cv2.waitKey(500)  # Wait 500ms between frames
        
        print(f"   Position {i+1}: {box[:2]} -> Outside")
    
    # Get final stats
    stats = detector.get_stats()
    print(f"\n📊 Final Statistics:")
    print(f"   Entries: {stats['entries']}")
    print(f"   Exits: {stats['exits']}")
    print(f"   Active Tracks: {stats['active_tracks']}")
    print(f"   Detected Doors: {stats['detected_doors']}")

def main():
    """Main test function with large display."""
    print("🚪 Large Display Door Detection Test")
    print("=" * 50)
    print("This test will show the door detection system in a large window")
    print("optimized for 1920x1080 screens with maximum visibility.")
    print()
    
    # Create large test image
    print("🖼️ Creating large test image...")
    test_image = create_large_test_image()
    
    # Create smart door detector
    print("🔍 Initializing smart door detector...")
    detector = create_smart_door_detector()
    
    print("🔍 Detecting doors in test image...")
    
    # Detect doors
    doors = detector.detect_doors(test_image)
    print(f"✅ Detected {len(doors)} door passages:")
    
    for i, door in enumerate(doors):
        print(f"   Door {i+1}: {door} (confidence: {door.confidence:.2f})")
    
    # Create display window
    window_name = "Large Door Detection Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Set window size to nearly full screen (leaving some margin)
    cv2.resizeWindow(window_name, 1800, 1000)
    
    # Position window in center of screen
    cv2.moveWindow(window_name, 60, 40)
    
    # Draw initial door boundaries
    result_image = detector.draw_door_boundaries(test_image)
    result_image = detector.draw_count_overlay(result_image)
    
    # Show initial image
    cv2.imshow(window_name, result_image)
    
    print("\n🖼️ Large test image displayed!")
    print("📋 Instructions:")
    print("   - Press SPACE to simulate person movement")
    print("   - Press 'q' to quit")
    print("   - Watch the door detection and person tracking in action")
    print()
    
    # Main loop
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar
            print("\n🎬 Starting person movement simulation...")
            simulate_person_movement_large(detector, test_image, test_image.copy())
            
            # Show final result
            final_image = detector.draw_door_boundaries(test_image)
            final_image = detector.draw_person_tracks(final_image)
            final_image = detector.draw_count_overlay(final_image)
            cv2.imshow(window_name, final_image)
            
            print("\n✅ Simulation completed! Press SPACE again or 'q' to quit.")
    
    cv2.destroyAllWindows()
    print("✅ Large display test completed!")

if __name__ == "__main__":
    main()
