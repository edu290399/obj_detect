#!/usr/bin/env python3
"""
Door Configuration Test Script

This script helps you configure the door boundary for person counting.
It allows you to draw and adjust the door line interactively.

Usage:
    python test_door_config.py
"""

import cv2
import numpy as np
import os
from dotenv import load_dotenv

# Load door configuration
load_dotenv('door_config.env')

class DoorConfigurator:
    def __init__(self):
        # Based on the main door image - line across the bottom of the doorway opening
        self.door_x1 = int(os.getenv('DOOR_X1', '350'))
        self.door_y1 = int(os.getenv('DOOR_Y1', '900'))
        self.door_x2 = int(os.getenv('DOOR_X2', '650'))
        self.door_y2 = int(os.getenv('DOOR_Y2', '900'))
        self.dragging = False
        self.drag_point = None
        
        # Create a test image (you can replace this with your actual camera feed)
        self.test_image = self.create_test_image()
        
    def create_test_image(self):
        """Create a test image to visualize the door boundary."""
        # Create a more realistic door view based on ASECAM camera perspective
        # Using 1080p resolution to match the new door coordinates
        img = np.ones((1080, 1280, 3), dtype=np.uint8) * 120  # 1080p resolution
        
        # Draw textured wall (light cream with stucco effect)
        for i in range(0, 1280, 20):
            for j in range(0, 1080, 20):
                color_variation = np.random.randint(-20, 20)
                wall_color = (120 + color_variation, 120 + color_variation, 100 + color_variation)
                cv2.rectangle(img, (i, j), (i + 20, j + 20), wall_color, -1)
        
        # Draw door frame (dark brown)
        door_frame_left = 500
        door_frame_right = 780
        door_frame_top = 200
        door_frame_bottom = 600
        
        cv2.rectangle(img, (door_frame_left, door_frame_top), 
                     (door_frame_right, door_frame_bottom), (60, 30, 20), -1)
        
        # Draw door opening (darker interior)
        door_left = 520
        door_right = 760
        door_top = 220
        door_bottom = 580
        
        cv2.rectangle(img, (door_left, door_top), 
                     (door_right, door_bottom), (40, 20, 10), -1)
        
        # Draw door handle (silver)
        handle_x = 750
        handle_y = 400
        cv2.circle(img, (handle_x, handle_y), 8, (192, 192, 192), -1)
        cv2.circle(img, (handle_x, handle_y), 8, (100, 100, 100), 2)
        
        # Draw tiled floor inside (light tiles)
        for i in range(door_left, door_right, 30):
            for j in range(door_bottom, 1080, 30):
                tile_color = (180 + np.random.randint(-10, 10), 
                             180 + np.random.randint(-10, 10), 
                             180 + np.random.randint(-10, 10))
                cv2.rectangle(img, (i, j), (i + 30, j + 30), tile_color, -1)
                cv2.rectangle(img, (i, j), (i + 30, j + 30), (150, 150, 150), 1)
        
        # Draw threshold/mat in front of door
        threshold_y = 580
        cv2.rectangle(img, (door_frame_left - 50, threshold_y), 
                     (door_frame_right + 50, threshold_y + 40), (30, 30, 30), -1)
        
        # Add some objects on the right (like in your image)
        # Pole with red cap
        cv2.rectangle(img, (1000, 300), (1020, 600), (80, 80, 80), -1)  # Dark pole
        cv2.circle(img, (1010, 300), 15, (0, 0, 255), -1)  # Red cap
        
        # Blue pole with bucket
        cv2.rectangle(img, (1100, 400), (1120, 650), (100, 150, 255), -1)  # Blue pole
        cv2.rectangle(img, (1080, 650), (1140, 700), (60, 40, 20), -1)  # Bucket
        
        # Add camera info overlay (like in your image)
        cv2.rectangle(img, (1100, 50), (1250, 100), (0, 255, 0), -1)
        cv2.putText(img, "41%", (1110, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add main title and instructions
        cv2.putText(img, "ASECAM Camera - Door View", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(img, "Draw door boundary line across the doorway", (50, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, "Click and drag to adjust the boundary", (50, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, "The line should cross where people walk", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return img
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for door boundary adjustment."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking near door line endpoints
            if self.distance_to_point(x, y, self.door_x1, self.door_y1) < 20:
                self.dragging = True
                self.drag_point = 1
            elif self.distance_to_point(x, y, self.door_x2, self.door_y2) < 20:
                self.dragging = True
                self.drag_point = 2
            else:
                # Create new door line
                self.door_x1, self.door_y1 = x, y
                self.door_x2, self.door_y2 = x + 200, y
                self.dragging = True
                self.drag_point = 2
                
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.drag_point == 1:
                self.door_x1, self.door_y1 = x, y
            elif self.drag_point == 2:
                self.door_x2, self.door_y2 = x, y
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.drag_point = None
    
    def distance_to_point(self, x1, y1, x2, y2):
        """Calculate distance between two points."""
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def draw_door_boundary(self, img):
        """Draw the current door boundary on the image."""
        output = img.copy()
        
        # Draw door boundary line
        cv2.line(output, 
                 (self.door_x1, self.door_y1),
                 (self.door_x2, self.door_y2),
                 (0, 255, 255), 3)  # Cyan line
        
        # Draw endpoints
        cv2.circle(output, (self.door_x1, self.door_y1), 8, (0, 255, 255), -1)
        cv2.circle(output, (self.door_x2, self.door_y2), 8, (0, 255, 255), -1)
        
        # Draw door center
        center_x = (self.door_x1 + self.door_x2) // 2
        center_y = (self.door_y1 + self.door_y2) // 2
        cv2.circle(output, (center_x, center_y), 5, (255, 0, 255), -1)  # Magenta
        
        # Draw coordinates
        cv2.putText(output, f"({self.door_x1},{self.door_y1})", 
                   (self.door_x1 + 10, self.door_y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(output, f"({self.door_x2},{self.door_y2})", 
                   (self.door_x2 + 10, self.door_y2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw door label
        cv2.putText(output, "DOOR BOUNDARY", 
                   (center_x - 60, center_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return output
    
    def draw_instructions(self, img):
        """Draw instructions on the image."""
        output = img.copy()
        
        # Instructions overlay
        instructions = [
            "DOOR CONFIGURATION",
            "==================",
            "Left click: Set door line",
            "Drag endpoints: Adjust line",
            "Press 's': Save configuration",
            "Press 'r': Reset to defaults",
            "Press 'q': Quit"
        ]
        
        y_offset = 150
        for i, instruction in enumerate(instructions):
            color = (255, 255, 255) if i < 2 else (200, 200, 200)
            cv2.putText(output, instruction, (50, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        return output
    
    def save_configuration(self):
        """Save the current door configuration to door_config.env."""
        config_content = f"""# Door Tracking Configuration
# Generated by test_door_config.py

# Door boundary coordinates (in pixels)
DOOR_X1={self.door_x1}
DOOR_Y1={self.door_y1}
DOOR_X2={self.door_x2}
DOOR_Y2={self.door_y2}

# Door name/label
DOOR_NAME=Main Door

# Tracking settings
TRACKING_FRAMES=30
MIN_CROSSING_DISTANCE=20

# Save directory for logs and counts
SAVE_DIRECTORY=door_logs
"""
        
        try:
            with open('door_config.env', 'w') as f:
                f.write(config_content)
            print(f"✅ Configuration saved to door_config.env")
            print(f"   Door boundary: ({self.door_x1},{self.door_y1}) to ({self.door_x2},{self.door_y2})")
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
    
    def reset_configuration(self):
        """Reset door boundary to default values."""
        # Based on the main door image - line across the bottom of the doorway opening
        self.door_x1 = 350
        self.door_y1 = 900
        self.door_x2 = 650
        self.door_y2 = 900
        print("🔄 Configuration reset to defaults")
    
    def run(self):
        """Run the door configuration interface."""
        cv2.namedWindow("Door Configuration", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Door Configuration", self.mouse_callback)
        
        print("🚪 Door Configuration Tool")
        print("==========================")
        print("Use this tool to configure the door boundary for person counting.")
        print("The door boundary should be a line that people cross when entering/exiting.")
        print()
        
        while True:
            # Create display image
            display_img = self.test_image.copy()
            display_img = self.draw_door_boundary(display_img)
            display_img = self.draw_instructions(display_img)
            
            # Show the image
            cv2.imshow("Door Configuration", display_img)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_configuration()
            elif key == ord('r'):
                self.reset_configuration()
        
        cv2.destroyAllWindows()
        print("👋 Door configuration tool closed.")


def main():
    """Main function to run the door configuration tool."""
    configurator = DoorConfigurator()
    configurator.run()


if __name__ == "__main__":
    main()
