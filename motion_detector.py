"""
Motion detection utility for ASECAM camera system.
Detects movement in camera frames and saves pictures with configurable cooldown.
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
from typing import Tuple, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MotionDetector:
    """
    Motion detection class that monitors camera frames for movement
    and saves pictures when motion is detected.
    """
    
    def __init__(self, 
                 save_directory: str = "motion_captures",
                 cooldown_seconds: float = 10.0,
                 motion_threshold: float = 25.0,
                 min_area: int = 500,
                 blur_kernel_size: int = 21):
        """
        Initialize the motion detector.
        
        Args:
            save_directory: Directory to save motion capture images
            cooldown_seconds: Time between captures (seconds)
            motion_threshold: Sensitivity threshold for motion detection
            min_area: Minimum area of motion to trigger capture
            blur_kernel_size: Kernel size for Gaussian blur (must be odd)
        """
        self.save_directory = save_directory
        self.cooldown_seconds = cooldown_seconds
        self.motion_threshold = motion_threshold
        self.min_area = min_area
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        
        # State variables
        self.last_capture_time = 0.0
        self.frame_count = 0
        self.previous_frame = None
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_directory, exist_ok=True)
        
        # Initialize background subtractor for more advanced motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, 
            varThreshold=16, 
            detectShadows=False
        )
        
        logger.info(f"Motion detector initialized: cooldown={cooldown_seconds}s, threshold={motion_threshold}, min_area={min_area}")
    
    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[List]]:
        """
        Detect motion in the current frame.
        
        Args:
            frame: Current camera frame (BGR format)
            
        Returns:
            Tuple of (motion_detected, motion_mask, contours)
        """
        if frame is None:
            return False, None, None
        
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        # Initialize previous frame if this is the first frame
        if self.previous_frame is None:
            self.previous_frame = blurred.copy()
            return False, None, None
        
        # Calculate absolute difference between current and previous frame
        frame_diff = cv2.absdiff(self.previous_frame, blurred)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_area]
        
        # Check if significant motion was detected
        motion_detected = len(significant_contours) > 0
        
        # Update previous frame
        self.previous_frame = blurred.copy()
        self.frame_count += 1
        
        return motion_detected, thresh, significant_contours
    
    def should_capture(self) -> bool:
        """
        Check if enough time has passed since the last capture.
        
        Returns:
            True if capture should proceed, False if in cooldown
        """
        current_time = time.time()
        time_since_last_capture = current_time - self.last_capture_time
        
        if time_since_last_capture >= self.cooldown_seconds:
            return True
        
        return False
    
    def save_motion_capture(self, frame: np.ndarray, motion_mask: Optional[np.ndarray] = None, 
                           contours: Optional[List] = None) -> Optional[str]:
        """
        Save the current frame as a motion capture image.
        
        Args:
            frame: Frame to save
            motion_mask: Optional motion mask for debugging
            contours: Optional contours for annotation
            
        Returns:
            Path to saved image, or None if save failed
        """
        if not self.should_capture():
            remaining_time = self.cooldown_seconds - (time.time() - self.last_capture_time)
            logger.info(f"Motion detected but in cooldown. {remaining_time:.1f}s remaining.")
            return None
        
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"motion_{timestamp}.jpg"
            filepath = os.path.join(self.save_directory, filename)
            
            # Create a copy of the frame for saving
            save_frame = frame.copy()
            
            # Add motion detection overlay if contours are provided
            if contours is not None:
                # Draw contours on the frame
                cv2.drawContours(save_frame, contours, -1, (0, 255, 0), 2)
                
                # Add motion detection info
                cv2.putText(save_frame, f"Motion Detected - {timestamp}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(save_frame, f"Contours: {len(contours)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Save the frame
            success = cv2.imwrite(filepath, save_frame)
            
            if success:
                self.last_capture_time = time.time()
                logger.info(f"Motion capture saved: {filepath}")
                
                # Also save motion mask for debugging if requested
                if motion_mask is not None:
                    mask_filename = f"motion_mask_{timestamp}.jpg"
                    mask_filepath = os.path.join(self.save_directory, mask_filename)
                    cv2.imwrite(mask_filepath, motion_mask)
                    logger.info(f"Motion mask saved: {mask_filepath}")
                
                return filepath
            else:
                logger.error(f"Failed to save motion capture: {filepath}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving motion capture: {e}")
            return None
    
    def get_cooldown_status(self) -> Optional[str]:
        """
        Get the current cooldown status.
        
        Returns:
            String describing cooldown status, or None if ready
        """
        if self.should_capture():
            return None
        
        remaining_time = self.cooldown_seconds - (time.time() - self.last_capture_time)
        return f"{remaining_time:.1f}s"
    
    def process_frame(self, frame: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Process a frame for motion detection and capture if needed.
        
        Args:
            frame: Current camera frame
            
        Returns:
            Tuple of (motion_detected, saved_image_path)
        """
        # Detect motion
        motion_detected, motion_mask, contours = self.detect_motion(frame)
        
        if motion_detected:
            logger.info(f"Motion detected! Frame: {self.frame_count}, Contours: {len(contours)}")
            
            # Try to save capture
            saved_path = self.save_motion_capture(frame, motion_mask, contours)
            
            return True, saved_path
        else:
            return False, None
    
    def get_stats(self) -> dict:
        """
        Get current motion detector statistics.
        
        Returns:
            Dictionary with current stats
        """
        current_time = time.time()
        time_since_last_capture = current_time - self.last_capture_time
        
        return {
            'frame_count': self.frame_count,
            'last_capture_time': self.last_capture_time,
            'time_since_last_capture': time_since_last_capture,
            'cooldown_remaining': max(0, self.cooldown_seconds - time_since_last_capture),
            'ready_for_capture': self.should_capture(),
            'save_directory': self.save_directory
        }
    
    def reset_cooldown(self):
        """Reset the cooldown timer."""
        self.last_capture_time = 0.0
        logger.info("Motion detector cooldown reset")
    
    def cleanup(self):
        """Clean up resources."""
        self.previous_frame = None
        logger.info("Motion detector cleaned up")


def create_motion_detector(save_directory: str = None, 
                          cooldown_seconds: float = 10.0,
                          motion_threshold: float = 25.0,
                          min_area: int = 500,
                          blur_kernel_size: int = 21) -> MotionDetector:
    """
    Factory function to create a motion detector with configurable settings.
    
    Args:
        save_directory: Directory to save captures (default: auto-generated timestamped)
        cooldown_seconds: Time between captures (default: 10.0)
        motion_threshold: Sensitivity threshold for motion detection (default: 25.0)
        min_area: Minimum area of motion to trigger capture (default: 500)
        blur_kernel_size: Kernel size for Gaussian blur (default: 21)
        
    Returns:
        Configured MotionDetector instance
    """
    if save_directory is None:
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_directory = f"motion_captures_{timestamp}"
    
    return MotionDetector(
        save_directory=save_directory,
        cooldown_seconds=cooldown_seconds,
        motion_threshold=motion_threshold,
        min_area=min_area,
        blur_kernel_size=blur_kernel_size
    )
