from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import os
import cv2
from kivy.factory import Factory

@dataclass
class DroneDetection:
    """Represents a single drone detection."""
    track_id: int
    bbox: np.ndarray
    confidence: float
    class_id: int
    timestamp: float

class ResultRepository:
    """Stores and manages detection and tracking results."""

    def __init__(self):
        """Initialize the repository."""
        self.detections: Dict[int, List[DroneDetection]] = {}
        self.current_frame_index = 0
        self.start_time = None
        self.drones_dir = 'drones'
        os.makedirs(self.drones_dir, exist_ok=True)

    def add_detection(self, track_id: int, bbox: np.ndarray, confidence: float, class_id: int, timestamp: float):
        """Add a new detection to the repository.

        Parameters
        ----------
        track_id : int
            Unique identifier for the drone track
        bbox : np.ndarray
            Bounding box coordinates [x1, y1, x2, y2]
        confidence : float
            Detection confidence score
        class_id : int
            Class identifier
        timestamp : float
            Detection timestamp
        """
        if track_id not in self.detections:
            self.detections[track_id] = []
        
        detection = DroneDetection(
            track_id=track_id,
            bbox=bbox,
            confidence=confidence,
            class_id=class_id,
            timestamp=timestamp
        )
        self.detections[track_id].append(detection)

    def save_drone_image(self, frame: np.ndarray, x1: float, y1: float, x2: float, y2: float, track_id: int):
        """Save a drone image when first detected.

        Parameters
        ----------
        frame : np.ndarray
            Video frame containing the drone
        x1, y1, x2, y2 : float
            Bounding box coordinates
        track_id : int
            Unique identifier for the drone track
        """
        try:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1 = max(0, min(x1, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0]))
            x2 = max(0, min(x2, frame.shape[1]))
            y2 = max(0, min(y2, frame.shape[0]))
            
            drone_img = frame[y1:y2, x1:x2]
            
            if drone_img.size > 0 and drone_img.shape[0] > 0 and drone_img.shape[1] > 0:
                h, w = drone_img.shape[:2]
                size = max(h, w)
                pad_h = (size - h) // 2
                pad_w = (size - w) // 2
                
                padded_img = np.full((size, size, 3), 255, dtype=np.uint8)
                padded_img[pad_h:pad_h+h, pad_w:pad_w+w] = drone_img
                
                cv2.imwrite(f'{self.drones_dir}/drone_{track_id}.png', cv2.cvtColor(padded_img, cv2.COLOR_RGB2BGR))
                print(f"Saved drone image for track {track_id}")
        except Exception as e:
            print(f"Error saving drone image for track {track_id}: {e}")

    def get_drone_track(self, track_id: int) -> List[DroneDetection]:
        """Get all detections for a specific drone track.

        Parameters
        ----------
        track_id : int
            Track ID to get detections for

        Returns
        -------
        List[DroneDetection]
            List of detections for the track
        """
        return self.detections.get(track_id, [])

    def get_all_tracks(self) -> Dict[int, List[DroneDetection]]:
        """Get all stored tracks.

        Returns
        -------
        Dict[int, List[DroneDetection]]
            Dictionary mapping track IDs to their detections
        """
        return self.detections

    def get_drone_count(self) -> int:
        """Get the total number of unique drones detected.

        Returns
        -------
        int
            Number of unique drone tracks
        """
        return len(self.detections)

    def update_frame_index(self, frame_index: int):
        """Update the current frame index.

        Parameters
        ----------
        frame_index : int
            New frame index
        """
        self.current_frame_index = frame_index

    def cleanup(self):
        """Clean up all stored data and saved images."""
        self.detections.clear()
        self.current_frame_index = 0
        self.start_time = None
        
        try:
            if os.path.exists(self.drones_dir):
                for file in os.listdir(self.drones_dir):
                    if file.startswith('drone_') and file.endswith('.png'):
                        file_path = os.path.join(self.drones_dir, file)
                        Factory.Image(source=file_path).reload()
                        os.remove(file_path)
        except Exception as e:
            print(f"Error cleaning up drone images: {e}")

    def query_tracks(self, drone_id: Optional[int] = None) -> Dict[int, List[DroneDetection]]:
        """Query tracks with optional filtering by drone ID.

        Parameters
        ----------
        drone_id : Optional[int]
            Optional drone ID to filter by

        Returns
        -------
        Dict[int, List[DroneDetection]]
            Filtered dictionary of tracks
        """
        if drone_id is None:
            return self.detections
        return {drone_id: self.detections.get(drone_id, [])}
