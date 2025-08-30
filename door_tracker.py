import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os


@dataclass
class DoorBoundary:
    """Represents a door boundary line for tracking entries/exits."""
    x1: int
    y1: int
    x2: int
    y2: int
    name: str = "Main Door"
    
    def get_center(self) -> Tuple[int, int]:
        """Get the center point of the door boundary."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def get_direction_vector(self) -> Tuple[int, int]:
        """Get the direction vector of the door boundary."""
        return (self.x2 - self.x1, self.y2 - self.y1)


@dataclass
class PersonTrack:
    """Tracks a person's position and movement across frames."""
    id: int
    boxes: List[List[int]]  # List of [x1, y1, x2, y2] boxes
    center_points: List[Tuple[int, int]]  # List of center points
    last_seen: int  # Frame number when last seen
    crossed_door: bool = False
    entry_direction: Optional[str] = None  # "in" or "out"
    entry_time: Optional[datetime] = None


class DoorTracker:
    """
    Tracks people crossing a door boundary and counts entries/exits.
    """
    
    def __init__(self, 
                 door_boundary: DoorBoundary,
                 tracking_frames: int = 30,
                 min_crossing_distance: int = 20,
                 save_directory: str = "door_logs"):
        """
        Initialize the door tracker.
        
        Args:
            door_boundary: The door boundary line to track
            tracking_frames: Number of frames to keep tracking a person
            min_crossing_distance: Minimum distance to consider a crossing
            save_directory: Directory to save entry/exit logs
        """
        self.door_boundary = door_boundary
        self.tracking_frames = tracking_frames
        self.min_crossing_distance = min_crossing_distance
        self.save_directory = save_directory
        
        # Person tracking
        self.next_person_id = 1
        self.active_tracks: Dict[int, PersonTrack] = {}
        
        # Counters
        self.entries = 0
        self.exits = 0
        
        # Create save directory
        os.makedirs(save_directory, exist_ok=True)
        
        # Load previous counts if they exist
        self.load_counts()
    
    def load_counts(self):
        """Load previous entry/exit counts from file."""
        count_file = os.path.join(self.save_directory, "door_counts.json")
        if os.path.exists(count_file):
            try:
                with open(count_file, 'r') as f:
                    data = json.load(f)
                    self.entries = data.get('entries', 0)
                    self.exits = data.get('exits', 0)
                    print(f"Loaded previous counts: {self.entries} entries, {self.exits} exits")
            except Exception as e:
                print(f"Could not load previous counts: {e}")
    
    def save_counts(self):
        """Save current entry/exit counts to file."""
        count_file = os.path.join(self.save_directory, "door_counts.json")
        try:
            with open(count_file, 'w') as f:
                json.dump({
                    'entries': self.entries,
                    'exits': self.exits,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"Could not save counts: {e}")
    
    def log_entry_exit(self, person_id: int, direction: str, timestamp: datetime):
        """Log an entry or exit event."""
        log_file = os.path.join(self.save_directory, "door_events.log")
        try:
            with open(log_file, 'a') as f:
                f.write(f"{timestamp.isoformat()} - Person {person_id} {direction} the house\n")
        except Exception as e:
            print(f"Could not log event: {e}")
    
    def update_tracks(self, person_boxes: List[List[int]], frame_number: int):
        """
        Update person tracks with new detections.
        
        Args:
            person_boxes: List of [x1, y1, x2, y2] boxes for detected people
            frame_number: Current frame number
        """
        # Create center points for new detections
        new_centers = []
        for box in person_boxes:
            x1, y1, x2, y2 = box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            new_centers.append(center)
        
        # Update existing tracks
        updated_tracks = set()
        
        for person_id, track in list(self.active_tracks.items()):
            if frame_number - track.last_seen > self.tracking_frames:
                # Remove old tracks
                del self.active_tracks[person_id]
                continue
            
            # Find closest new detection to this track
            if new_centers:
                distances = [np.linalg.norm(np.array(center) - np.array(track.center_points[-1])) 
                           for center in new_centers]
                min_dist_idx = np.argmin(distances)
                min_distance = distances[min_dist_idx]
                
                if min_distance < 100:  # Reasonable tracking distance
                    # Update track
                    track.boxes.append(person_boxes[min_dist_idx])
                    track.center_points.append(new_centers[min_dist_idx])
                    track.last_seen = frame_number
                    updated_tracks.add(min_dist_idx)
                    
                    # Check for door crossing
                    self._check_door_crossing(track, person_id)
        
        # Create new tracks for unmatched detections
        for i, (box, center) in enumerate(zip(person_boxes, new_centers)):
            if i not in updated_tracks:
                new_track = PersonTrack(
                    id=self.next_person_id,
                    boxes=[box],
                    center_points=[center],
                    last_seen=frame_number
                )
                self.active_tracks[self.next_person_id] = new_track
                self.next_person_id += 1
    
    def _check_door_crossing(self, track: PersonTrack, person_id: int):
        """Check if a person has crossed the door boundary."""
        if track.crossed_door or len(track.center_points) < 3:  # Need at least 3 points for reliable crossing
            return
        
        # Get door center and direction
        door_center = self.door_boundary.get_center()
        door_direction = self.door_boundary.get_direction_vector()
        
        # Calculate perpendicular vector to door line
        perp_vector = (-door_direction[1], door_direction[0])
        perp_vector = np.array(perp_vector) / np.linalg.norm(perp_vector)
        
        # Get the last few points to check for actual crossing
        recent_points = track.center_points[-3:]  # Last 3 points
        
        # Calculate distances from each point to door line
        distances = []
        for point in recent_points:
            point_array = np.array(point)
            distance = np.dot(point_array - door_center, perp_vector)
            distances.append(distance)
        
        # Check if person actually crossed the door line (went from one side to the other)
        if len(distances) >= 2:
            # Check if we have points on both sides of the door
            first_side = distances[0] > 0  # True if on "inside" side
            last_side = distances[-1] > 0   # True if on "inside" side
            
            # Only count if person actually crossed from one side to the other
            if first_side != last_side:  # Different sides = actual crossing
                # Determine direction based on movement
                if not first_side and last_side:  # From outside to inside
                    direction = "in"
                    self.entries += 1
                else:  # From inside to outside
                    direction = "out"
                    self.exits += 1
                
                track.crossed_door = True
                track.entry_direction = direction
                track.entry_time = datetime.now()
                
                print(f"🚪 Person {person_id} {direction} the house! "
                      f"(Total: {self.entries} in, {self.exits} out)")
                
                # Log the event
                self.log_entry_exit(person_id, direction, track.entry_time)
                
                # Save updated counts
                self.save_counts()
    
    def get_counts(self) -> Tuple[int, int]:
        """Get current entry and exit counts."""
        return self.entries, self.exits
    
    def draw_door_boundary(self, frame: np.ndarray) -> np.ndarray:
        """Draw the door boundary on the frame."""
        output = frame.copy()
        
        # Draw thick door boundary line for better visibility
        cv2.line(output, 
                 (self.door_boundary.x1, self.door_boundary.y1),
                 (self.door_boundary.x2, self.door_boundary.y2),
                 (0, 255, 255), 5)  # Thicker cyan line
        
        # Draw door center point
        center = self.door_boundary.get_center()
        cv2.circle(output, center, 12, (0, 255, 255), -1)
        cv2.circle(output, center, 12, (0, 0, 0), 2)  # Black border
        
        # Draw door label with larger font and background
        label_text = self.door_boundary.name
        label_x = center[0] - 80
        label_y = center[1] - 30
        
        # Background rectangle for label
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        
        cv2.rectangle(output, 
                     (label_x - 10, label_y - text_height - 10),
                     (label_x + text_width + 10, label_y + 10),
                     (0, 0, 0), -1)
        
        cv2.rectangle(output, 
                     (label_x - 10, label_y - text_height - 10),
                     (label_x + text_width + 10, label_y + 10),
                     (0, 255, 255), 2)
        
        # Draw label text
        cv2.putText(output, label_text, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        return output
    
    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """Draw person tracks on the frame."""
        output = frame.copy()
        
        for person_id, track in self.active_tracks.items():
            if len(track.center_points) > 1:
                # Draw track line with thicker line
                points = np.array(track.center_points, dtype=np.int32)
                cv2.polylines(output, [points], False, (0, 255, 0), 3)
                
                # Draw current position with larger circle
                current_pos = track.center_points[-1]
                cv2.circle(output, current_pos, 8, (0, 255, 0), -1)
                cv2.circle(output, current_pos, 8, (0, 0, 0), 2)  # Black border
                
                # Draw person ID with larger font and background
                id_text = f"P{person_id}"
                id_x = current_pos[0] + 15
                id_y = current_pos[1] - 15
                
                # Background rectangle for ID
                (text_width, text_height), baseline = cv2.getTextSize(
                    id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                
                cv2.rectangle(output, 
                             (id_x - 5, id_y - text_height - 5),
                             (id_x + text_width + 5, id_y + 5),
                             (0, 0, 0), -1)
                
                cv2.rectangle(output, 
                             (id_x - 5, id_y - text_height - 5),
                             (id_x + text_width + 5, id_y + 5),
                             (0, 255, 0), 2)
                
                # Draw ID text
                cv2.putText(output, id_text, (id_x, id_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return output
    
    def draw_count_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw entry/exit count overlay on the frame."""
        output = frame.copy()
        
        # Get frame dimensions for responsive sizing
        frame_height, frame_width = frame.shape[:2]
        
        # Create larger background rectangle for 1920x1080 monitors
        overlay_width = min(500, frame_width // 3)  # Responsive width
        overlay_height = 150
        overlay_x = 20
        overlay_y = 20
        
        # Semi-transparent background
        overlay_bg = output.copy()
        cv2.rectangle(overlay_bg, (overlay_x, overlay_y), 
                     (overlay_x + overlay_width, overlay_y + overlay_height), 
                     (0, 0, 0), -1)
        
        # Blend with original frame for transparency effect
        alpha = 0.8
        output = cv2.addWeighted(overlay_bg, alpha, output, 1 - alpha, 0)
        
        # Draw border
        cv2.rectangle(output, (overlay_x, overlay_y), 
                     (overlay_x + overlay_width, overlay_y + overlay_height), 
                     (255, 255, 255), 3)
        
        # Draw title with larger font
        title_y = overlay_y + 40
        cv2.putText(output, "DOOR COUNTER", (overlay_x + 20, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Draw entries with large green text
        entries_y = overlay_y + 80
        cv2.putText(output, f"ENTRIES: {self.entries}", (overlay_x + 20, entries_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Draw exits with large red text
        exits_y = overlay_y + 120
        cv2.putText(output, f"EXITS: {self.exits}", (overlay_x + 20, exits_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        
        return output


def create_door_tracker(door_x1: int, door_y1: int, door_x2: int, door_y2: int,
                       door_name: str = "Main Door",
                       tracking_frames: int = 30,
                       min_crossing_distance: int = 20,
                       save_directory: str = "door_logs") -> DoorTracker:
    """
    Factory function to create a door tracker with the specified boundary.
    
    Args:
        door_x1, door_y1: Start point of door boundary
        door_x2, door_y2: End point of door boundary
        door_name: Name/label for the door
        tracking_frames: Number of frames to keep tracking a person
        min_crossing_distance: Minimum distance to consider a crossing
        save_directory: Directory to save entry/exit logs
    
    Returns:
        Configured DoorTracker instance
    """
    door_boundary = DoorBoundary(
        x1=door_x1, y1=door_y1,
        x2=door_x2, y2=door_y2,
        name=door_name
    )
    
    return DoorTracker(
        door_boundary=door_boundary,
        tracking_frames=tracking_frames,
        min_crossing_distance=min_crossing_distance,
        save_directory=save_directory
    )
