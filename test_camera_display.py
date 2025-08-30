#!/usr/bin/env python3
"""
Test script to verify camera display utilities work correctly.
This script creates a test image and displays it using the camera utilities
to ensure full image visibility.
"""

import cv2
import numpy as np
from camera_utils import (
    setup_camera_display,
    ensure_full_frame_visible,
    add_camera_info_overlay,
    create_resizable_window
)


def create_test_image(width: int = 1920, height: int = 1080) -> np.ndarray:
    """Create a test image with grid pattern and text to verify full visibility."""
    # Create a test image with a grid pattern
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add grid lines
    grid_spacing = 100
    for x in range(0, width, grid_spacing):
        cv2.line(image, (x, 0), (x, height), (50, 50, 50), 1)
    for y in range(0, height, grid_spacing):
        cv2.line(image, (0, y), (width, y), (50, 50, 50), 1)
    
    # Add corner markers
    marker_size = 50
    cv2.rectangle(image, (0, 0), (marker_size, marker_size), (0, 255, 0), 3)
    cv2.rectangle(image, (width - marker_size, 0), (width, marker_size), (0, 255, 0), 3)
    cv2.rectangle(image, (0, height - marker_size), (marker_size, height), (0, 255, 0), 3)
    cv2.rectangle(image, (width - marker_size, height - marker_size), (width, height), (0, 255, 0), 3)
    
    # Add center marker
    center_x, center_y = width // 2, height // 2
    cv2.circle(image, (center_x, center_y), 30, (255, 0, 0), 5)
    
    # Add resolution text
    cv2.putText(image, f"Test Image: {width}x{height}", (center_x - 200, center_y + 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Add corner coordinates
    cv2.putText(image, "(0,0)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, f"({width},{height})", (width - 150, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return image


def test_camera_display():
    """Test the camera display utilities with a synthetic image."""
    print("Testing camera display utilities...")
    
    # Create test images at different resolutions
    test_resolutions = [
        (1280, 720),   # HD
        (1920, 1080),  # Full HD
        (2560, 1440),  # 2K
        (3840, 2160),  # 4K
    ]
    
    for width, height in test_resolutions:
        print(f"\nTesting resolution: {width}x{height}")
        
        # Create test image
        test_image = create_test_image(width, height)
        
        # Create a mock VideoCapture object for testing
        class MockVideoCapture:
            def __init__(self, width, height):
                self.width = width
                self.height = height
                self.fps = 30.0
            
            def get(self, prop):
                if prop == cv2.CAP_PROP_FRAME_WIDTH:
                    return self.width
                elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                    return self.height
                elif prop == cv2.CAP_PROP_FPS:
                    return self.fps
                return 0
        
        mock_cap = MockVideoCapture(width, height)
        
        # Test the display setup
        window_name = f"Test Display - {width}x{height}"
        try:
            # Setup display
            frame_width, frame_height, fps = setup_camera_display(mock_cap, window_name)
            print(f"  Display setup: {frame_width}x{frame_height} @ {fps:.1f} FPS")
            
            # Ensure full frame visibility
            validated_frame = ensure_full_frame_visible(test_image, (frame_width, frame_height))
            print(f"  Frame validation: {validated_frame.shape[1]}x{validated_frame.shape[0]}")
            
            # Add information overlay
            overlay_frame = add_camera_info_overlay(
                validated_frame, 
                "TEST_CAMERA", 
                (frame_width, frame_height), 
                fps, 
                0,
                f"Test Resolution: {width}x{height}"
            )
            
            # Display the image
            cv2.imshow(window_name, overlay_frame)
            print(f"  Displaying test image... Press any key to continue or 'q' to quit")
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
                
        except Exception as e:
            print(f"  Error: {e}")
        finally:
            cv2.destroyWindow(window_name)
    
    cv2.destroyAllWindows()
    print("\nTest completed!")


def test_window_management():
    """Test window management functions."""
    print("\nTesting window management...")
    
    # Test screen resolution detection
    try:
        from camera_utils import get_screen_resolution
        screen_w, screen_h = get_screen_resolution()
        print(f"Detected screen resolution: {screen_w}x{screen_h}")
    except Exception as e:
        print(f"Screen resolution detection failed: {e}")
    
    # Test window creation
    try:
        test_window = "Test Window Management"
        create_resizable_window(test_window, 800, 600)
        print("Test window created successfully")
        
        # Create a simple test image
        test_img = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(test_img, "Test Window", (300, 300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        cv2.imshow(test_window, test_img)
        print("Test image displayed. Press any key to continue...")
        cv2.waitKey(0)
        
        cv2.destroyWindow(test_window)
        print("Test window destroyed")
        
    except Exception as e:
        print(f"Window management test failed: {e}")


if __name__ == "__main__":
    print("Camera Display Utilities Test")
    print("=" * 40)
    
    try:
        test_window_management()
        test_camera_display()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
    finally:
        cv2.destroyAllWindows()
        print("\nAll tests completed!")
