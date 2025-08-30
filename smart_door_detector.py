#!/usr/bin/env python3
"""
Smart Door Detection and Person Counting System

This module automatically detects doors using computer vision and tracks
person motion through the passage area to count entrances and exits.
"""

import cv2
import numpy as np
import time
from datetime import datetime
import json
import os
from typing import List, Tuple, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DoorPassage:
    """Represents a detected door passage area."""
    
    def __init__(self, x1: int, y1: int, x2: int, y2: int, confidence: float):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.confidence = confidence
        self.center_x = (x1 + x2) // 2
        self.center_y = (y1 + y2) // 2
        self.width = abs(x2 - x1)
        self.height = abs(y2 - y1)
        
    def __repr__(self):
        return f"DoorPassage({self.x1},{self.y1},{self.x2},{self.y2}, conf:{self.confidence:.2f})"

class PersonMotion:
    """Tracks a person's motion through the door area."""
    
    def __init__(self, person_id: int):
        self.person_id = person_id
        self.trajectory = []  # List of (x, y, frame_count, timestamp)
        self.orientation_history = []  # List of orientation estimates
        self.entered_door = False
        self.exited_door = False
        self.direction = None  # "in" or "out"
        self.entry_time = None
        self.last_seen = time.time()
        
    def add_position(self, x: int, y: int, frame_count: int, orientation: float):
        """Add a new position to the trajectory."""
        timestamp = time.time()
        self.trajectory.append((x, y, frame_count, timestamp))
        self.orientation_history.append(orientation)
        self.last_seen = timestamp
        
        # Keep only last 30 positions to avoid memory issues
        if len(self.trajectory) > 30:
            self.trajectory.pop(0)
            self.orientation_history.pop(0)
    
    def has_crossed_door(self, door_passage: DoorPassage) -> bool:
        """Check if person has crossed the door passage area."""
        if len(self.trajectory) < 3:
            return False
            
        # Check if trajectory crosses the door passage line
        for i in range(len(self.trajectory) - 1):
            x1, y1, _, _ = self.trajectory[i]
            x2, y2, _, _ = self.trajectory[i + 1]
            
            if self._lines_intersect(x1, y1, x2, y2, 
                                   door_passage.x1, door_passage.y1, 
                                   door_passage.x2, door_passage.y2):
                return True
        return False
    
    def _lines_intersect(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """Check if two line segments intersect."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        A, B = (x1, y1), (x2, y2)
        C, D = (x3, y3), (x4, y4)
        
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    def get_orientation(self) -> float:
        """Get the average orientation from recent frames."""
        if not self.orientation_history:
            return 0.0
        
        # Use last 5 orientation estimates
        recent = self.orientation_history[-5:]
        return np.mean(recent)
    
    def is_facing_camera(self) -> bool:
        """Determine if person is facing the camera (front view)."""
        orientation = self.get_orientation()
        # Assuming 0 degrees is facing camera, 180 is facing away
        # Adjust these thresholds based on your camera setup
        return abs(orientation) < 90  # Front view if within 90 degrees

class SmartDoorDetector:
    """Automatically detects doors and tracks person motion."""
    
    def __init__(self, save_directory: str = "door_logs"):
        self.save_directory = save_directory
        self.door_passages: List[DoorPassage] = []
        self.person_motions: Dict[int, PersonMotion] = {}
        self.entries = 0
        self.exits = 0
        self.frame_count = 0
        self.next_person_id = 1
        
        # Door detection parameters
        self.door_detection_interval = 300  # Detect doors every 300 frames
        self.min_door_width = 100
        self.min_door_height = 150
        self.door_confidence_threshold = 0.6
        
        # Motion tracking parameters
        self.motion_threshold = 25
        self.min_motion_area = 500
        self.tracking_frames = 30
        
        # Create save directory
        os.makedirs(save_directory, exist_ok=True)
        
        # Load previous counts
        self.load_counts()
        
    def detect_doors(self, frame: np.ndarray) -> List[DoorPassage]:
        """Detect doors in the frame using computer vision."""
        if self.frame_count % self.door_detection_interval != 0:
            return self.door_passages
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Morphological operations to connect edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_passages = []
            
            for contour in contours:
                # Filter by area
                area = cv2.contourArea(contour)
                if area < 1000:  # Minimum area threshold
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by dimensions (door-like proportions)
                if w < self.min_door_width or h < self.min_door_height:
                    continue
                
                # Check aspect ratio (doors are typically taller than wide)
                aspect_ratio = h / w
                if aspect_ratio < 1.5:  # Door should be taller than wide
                    continue
                
                # Calculate confidence based on contour properties
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    # Doors should have low circularity (more rectangular)
                    if circularity < 0.3:
                        # Additional check: look for straight edges
                        epsilon = 0.02 * perimeter
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        # Doors typically have 4 corners
                        if len(approx) >= 4:
                            confidence = 0.8
                            # Boost confidence if it's near the bottom of the frame (entrance level)
                            if y + h > frame.shape[0] * 0.6:  # Bottom 40% of frame
                                confidence += 0.2
                            
                            if confidence >= self.door_confidence_threshold:
                                passage = DoorPassage(x, y, x + w, y + h, confidence)
                                detected_passages.append(passage)
            
            # Sort by confidence and keep top 3
            detected_passages.sort(key=lambda p: p.confidence, reverse=True)
            self.door_passages = detected_passages[:3]
            
            logger.info(f"Detected {len(self.door_passages)} door passages")
            
        except Exception as e:
            logger.error(f"Error detecting doors: {e}")
            
        return self.door_passages
    
    def estimate_person_orientation(self, person_box: List[float]) -> float:
        """Estimate person orientation based on bounding box and motion."""
        if len(person_box) != 4:
            return 0.0
        
        x1, y1, x2, y2 = person_box
        width = x2 - x1
        height = y2 - y1
        
        # Simple heuristic: if width > height, person is likely sideways
        # This is a basic approach - could be improved with pose estimation
        if width > height * 1.2:
            return 90.0  # Sideways
        elif height > width * 1.2:
            return 0.0   # Front/back view
        else:
            return 45.0  # Diagonal
    
    def update_tracks(self, person_boxes: List[List[float]], frame: np.ndarray):
        """Update person tracking and detect door crossings."""
        self.frame_count += 1
        
        # Detect doors periodically
        self.detect_doors(frame)
        
        # Update existing tracks
        current_person_ids = set()
        
        for person_box in person_boxes:
            # Find closest existing track
            best_match_id = None
            best_distance = float('inf')
            
            for person_id, motion in self.person_motions.items():
                if motion.trajectory:
                    last_x, last_y, _, _ = motion.trajectory[-1]
                    center_x = (person_box[0] + person_box[2]) / 2
                    center_y = (person_box[1] + person_box[3]) / 2
                    
                    distance = np.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
                    if distance < 100 and distance < best_distance:  # 100px threshold
                        best_distance = distance
                        best_match_id = person_id
            
            # Update existing track or create new one
            if best_match_id is not None:
                person_id = best_match_id
                motion = self.person_motions[person_id]
                logger.debug(f"Using existing track for person {person_id}")
            else:
                person_id = self.next_person_id
                motion = PersonMotion(person_id)
                self.person_motions[person_id] = motion
                logger.debug(f"Created new track for person {person_id}: {type(motion)}")
                self.next_person_id += 1
            
            current_person_ids.add(person_id)
            
            # Add position to track
            center_x = int((person_box[0] + person_box[2]) / 2)
            center_y = int((person_box[1] + person_box[3]) / 2)
            orientation = self.estimate_person_orientation(person_box)
            
            motion.add_position(center_x, center_y, self.frame_count, orientation)
            
            # Check for door crossing
            self._check_door_crossing(motion, person_id)
        
        # Remove old tracks
        current_time = time.time()
        expired_ids = []
        for person_id, motion in self.person_motions.items():
            # Debug: Check what type of object we have
            if not hasattr(motion, 'last_seen'):
                logger.error(f"Invalid motion object for person {person_id}: {type(motion)} - {motion}")
                expired_ids.append(person_id)
                continue
                
            if current_time - motion.last_seen > 5.0:  # 5 second timeout
                expired_ids.append(person_id)
        
        for person_id in expired_ids:
            del self.person_motions[person_id]
    
    def _check_door_crossing(self, motion: PersonMotion, person_id: int):
        """Check if a person has crossed any door passage."""
        if motion.entered_door or motion.exited_door:
            return
        
        for door_passage in self.door_passages:
            if motion.has_crossed_door(door_passage):
                # Determine direction based on person orientation
                is_facing_camera = motion.is_facing_camera()
                
                if is_facing_camera:
                    # Person facing camera = exiting
                    direction = "out"
                    self.exits += 1
                    motion.exited_door = True
                else:
                    # Person facing away = entering
                    direction = "in"
                    self.entries += 1
                    motion.entered_door = True
                
                motion.direction = direction
                motion.entry_time = datetime.now()
                
                logger.info(f"🚪 Person {person_id} {direction} the house! "
                           f"(Total: {self.entries} in, {self.exits} out)")
                
                # Log the event
                self.log_entry_exit(person_id, direction, motion.entry_time)
                
                # Save updated counts
                self.save_counts()
                break
    
    def log_entry_exit(self, person_id: int, direction: str, timestamp: datetime):
        """Log door entry/exit events."""
        log_file = os.path.join(self.save_directory, "door_events.log")
        
        try:
            with open(log_file, 'a') as f:
                f.write(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} - "
                        f"Person {person_id} {direction} the house\n")
        except Exception as e:
            logger.error(f"Error logging door event: {e}")
    
    def save_counts(self):
        """Save current entry/exit counts."""
        counts_file = os.path.join(self.save_directory, "door_counts.json")
        
        try:
            counts_data = {
                "entries": self.entries,
                "exits": self.exits,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(counts_file, 'w') as f:
                json.dump(counts_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving door counts: {e}")
    
    def load_counts(self):
        """Load previous entry/exit counts."""
        counts_file = os.path.join(self.save_directory, "door_counts.json")
        
        try:
            if os.path.exists(counts_file):
                with open(counts_file, 'r') as f:
                    counts_data = json.load(f)
                    self.entries = counts_data.get("entries", 0)
                    self.exits = counts_data.get("exits", 0)
                    logger.info(f"Loaded counts: {self.entries} entries, {self.exits} exits")
        except Exception as e:
            logger.error(f"Error loading door counts: {e}")
    
    def draw_door_boundaries(self, img: np.ndarray) -> np.ndarray:
        """Draw detected door boundaries on the image."""
        output = img.copy()
        
        for i, passage in enumerate(self.door_passages):
            # Draw door boundary rectangle
            color = (0, 255, 255)  # Cyan
            thickness = 3
            
            cv2.rectangle(output, (passage.x1, passage.y1), 
                         (passage.x2, passage.y2), color, thickness)
            
            # Draw passage center
            cv2.circle(output, (passage.center_x, passage.center_y), 8, color, -1)
            
            # Draw confidence and label
            label = f"Door {i+1} ({passage.confidence:.2f})"
            cv2.putText(output, label, (passage.x1, passage.y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return output
    
    def draw_person_tracks(self, img: np.ndarray) -> np.ndarray:
        """Draw person tracking trails on the image."""
        output = img.copy()
        
        for person_id, motion in self.person_motions.items():
            if len(motion.trajectory) < 2:
                continue
            
            # Draw trajectory line
            points = [(int(x), int(y)) for x, y, _, _ in motion.trajectory]
            cv2.polylines(output, [np.array(points)], False, (0, 255, 0), 2)
            
            # Draw current position
            if motion.trajectory:
                last_x, last_y, _, _ = motion.trajectory[-1]
                cv2.circle(output, (int(last_x), int(last_y)), 6, (0, 255, 0), -1)
                
                # Draw person ID
                cv2.putText(output, f"P{person_id}", (int(last_x) + 10, int(last_y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return output
    
    def draw_count_overlay(self, img: np.ndarray) -> np.ndarray:
        """Draw entry/exit count overlay on the image."""
        output = img.copy()
        
        # Create semi-transparent overlay
        overlay = output.copy()
        cv2.rectangle(overlay, (50, 50), (300, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        # Draw counts
        cv2.putText(output, "DOOR COUNTER", (70, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(output, f"ENTRIES: {self.entries}", (70, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(output, f"EXITS: {self.exits}", (70, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw current time
        current_time = datetime.now().strftime("%H:%M")
        cv2.putText(output, current_time, (70, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return output
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            "entries": self.entries,
            "exits": self.exits,
            "active_tracks": len(self.person_motions),
            "detected_doors": len(self.door_passages),
            "frame_count": self.frame_count
        }

def create_smart_door_detector(save_directory: str = "door_logs") -> SmartDoorDetector:
    """Factory function to create a smart door detector."""
    return SmartDoorDetector(save_directory)
