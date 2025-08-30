"""
Camera utility functions for ensuring proper display and full image visibility.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def get_screen_resolution() -> Tuple[int, int]:
    """
    Get the screen resolution to ensure the camera view fits properly.
    
    Returns:
        Tuple of (width, height) in pixels
    """
    try:
        # Try to get screen resolution using OpenCV
        screen = cv2.getWindowImageRect("temp")
        if screen is not None:
            return screen[2], screen[3]  # width, height
    except:
        pass
    
    # Fallback to common resolutions
    return 1920, 1080  # Default to 1080p


def create_resizable_window(window_name: str, initial_width: int = 1280, initial_height: int = 720):
    """
    Create a resizable window that can be adjusted by the user.
    
    Args:
        window_name: Name of the window
        initial_width: Initial window width in pixels
        initial_height: Initial window height in pixels
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Set initial window size
    cv2.resizeWindow(window_name, initial_width, initial_height)
    
    # Try to position the window in the center of the screen
    try:
        screen_w, screen_h = get_screen_resolution()
        x = max(0, (screen_w - initial_width) // 2)
        y = max(0, (screen_h - initial_height) // 2)
        cv2.moveWindow(window_name, x, y)
    except:
        pass


def resize_frame_to_fit_screen(frame: np.ndarray, max_width: int = None, max_height: int = None) -> np.ndarray:
    """
    Resize frame to fit within screen bounds while maintaining aspect ratio.
    
    Args:
        frame: Input frame (numpy array)
        max_width: Maximum allowed width (default: screen width)
        max_height: Maximum allowed height (default: screen height)
    
    Returns:
        Resized frame that fits within the specified bounds
    """
    if max_width is None or max_height is None:
        screen_w, screen_h = get_screen_resolution()
        max_width = max_width or screen_w
        max_height = max_height or screen_h
    
    # Get current frame dimensions
    h, w = frame.shape[:2]
    
    # Calculate scaling factors
    scale_w = max_width / w
    scale_h = max_height / h
    
    # Use the smaller scale to ensure the frame fits in both dimensions
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale, only downscale if needed
    
    if scale < 1.0:
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize the frame
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_frame
    
    return frame


def add_camera_info_overlay(frame: np.ndarray, host: str, resolution: Tuple[int, int], fps: float, 
                           detections: int = 0, additional_info: str = "", 
                           motion_status: str = None, motion_cooldown: str = None) -> np.ndarray:
    """
    Add comprehensive camera information overlay to the frame.
    
    Args:
        frame: Input frame
        host: Camera host/IP address
        resolution: Camera resolution as (width, height)
        fps: Current FPS
        detections: Number of object detections
        additional_info: Additional information to display
        motion_status: Motion detection status (e.g., "Motion Detected", "No Motion")
        motion_cooldown: Motion detection cooldown status
    
    Returns:
        Frame with information overlay
    """
    output = frame.copy()
    
    # FPS and detection info
    cv2.putText(
        output,
        f"FPS: {fps:.1f} | Detections: {detections}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    
    # Motion detection status (if provided)
    if motion_status:
        # Use different colors based on motion status
        if "Motion" in motion_status and "No" not in motion_status:
            color = (0, 255, 0)  # Green for motion detected
        else:
            color = (0, 255, 255)  # Cyan for no motion
        
        cv2.putText(
            output,
            f"Motion: {motion_status}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    
    # Motion cooldown status (if provided)
    if motion_cooldown:
        cv2.putText(
            output,
            f"Cooldown: {motion_cooldown}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    
    # Camera info and resolution
    cv2.putText(
        output,
        f"ASECAM 8MP - {host} - {resolution[0]}x{resolution[1]}",
        (10, output.shape[0] - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    
    # Additional info if provided
    if additional_info:
        cv2.putText(
            output,
            additional_info,
            (10, output.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    
    # User instructions
    cv2.putText(
        output,
        "Press 'q' to quit | Use mouse to resize window | Full image visible",
        (10, output.shape[0] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    
    return output


def ensure_full_frame_visible(frame: np.ndarray, expected_resolution: Tuple[int, int]) -> np.ndarray:
    """
    Ensure the frame is displayed at full resolution and warn if there are size mismatches.
    
    Args:
        frame: Input frame
        expected_resolution: Expected resolution as (width, height)
    
    Returns:
        The frame (unchanged, but validated)
    """
    actual_resolution = (frame.shape[1], frame.shape[0])  # OpenCV uses (height, width)
    
    if actual_resolution != expected_resolution:
        print(f"Warning: Frame size mismatch. Expected {expected_resolution[0]}x{expected_resolution[1]}, got {actual_resolution[0]}x{actual_resolution[1]}")
        print("This may indicate the camera is not providing the full resolution or there's a connection issue.")
    
    return frame


def setup_camera_display(cap: cv2.VideoCapture, window_name: str) -> Tuple[int, int, float]:
    """
    Setup camera display and get camera properties.
    
    Args:
        cap: OpenCV VideoCapture object
        window_name: Name of the display window
    
    Returns:
        Tuple of (frame_width, frame_height, fps)
    """
    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_camera = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera stream: {frame_width}x{frame_height} @ {fps_camera:.1f} FPS")
    print(f"Full image dimensions: {frame_width}x{frame_height}")
    
    # Create resizable window
    create_resizable_window(window_name, min(1280, frame_width), min(720, frame_height))
    
    return frame_width, frame_height, fps_camera


def draw_motion_contours(frame: np.ndarray, contours: list, motion_detected: bool) -> np.ndarray:
    """
    Draw motion detection contours on the frame.
    
    Args:
        frame: Input frame
        contours: List of contours to draw
        motion_detected: Whether motion was detected
    
    Returns:
        Frame with motion contours drawn
    """
    output = frame.copy()
    
    if contours and len(contours) > 0:
        # Draw contours
        color = (0, 255, 0) if motion_detected else (0, 255, 255)
        cv2.drawContours(output, contours, -1, color, 2)
        
        # Add motion indicator text
        status_text = "MOTION DETECTED" if motion_detected else "No Motion"
        cv2.putText(
            output,
            status_text,
            (10, output.shape[0] - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    
    return output
