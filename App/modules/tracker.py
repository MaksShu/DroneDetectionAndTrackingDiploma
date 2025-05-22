import numpy as np
from bytetracker import BYTETracker
from typing import List, Tuple, Union, Any


class DroneTracker:
    """Wrapper around BYTETracker for drone tracking.
    
    This class provides methods to track drones across video frames
    using the BYTETracker algorithm.
    """

    def __init__(self):
        """Initialize the drone tracker."""
        self.tracker = BYTETracker()

    def reset(self):
        """Reset the tracker state.
        
        Clears all tracking data and initializes a new tracker.
        """
        self.tracker.tracked_stracks.clear()
        self.tracker.lost_stracks.clear()
        self.tracker.removed_stracks.clear()
        self.tracker.frame_id = 0
        self.tracker._next_id = 1
        from bytetracker.byte_tracker import BaseTrack
        BaseTrack._count = 0
        del self.tracker
        self.tracker = BYTETracker()

    def update(self, detections: List[List[float]], frame: np.ndarray) -> List[List[float]]:
        """Update tracks with new detections.

        Parameters
        ----------
        detections : List[List[float]]
            List of detections in format [x1, y1, x2, y2, conf, class_id]
        frame : np.ndarray
            Current frame for visualization

        Returns
        -------
        List[List[float]]
            List of tracked objects in format [x1, y1, x2, y2, track_id, class_id, conf]
        """
        if len(detections) > 0:
            tracked_objects = self.tracker.update(np.array(detections), frame)
        else:
            tracked_objects = []
        return tracked_objects

    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get a unique color for a track ID.

        Parameters
        ----------
        track_id : int
            Track ID to get color for

        Returns
        -------
        Tuple[int, int, int]
            RGB color tuple with values in range 0-255
        """
        hue = (track_id * 0.618033988749895) % 1.0
        
        h = hue
        s = 0.8
        v = 0.9
        
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c

        if h < 1/6:
            r, g, b = c, x, 0
        elif h < 2/6:
            r, g, b = x, c, 0
        elif h < 3/6:
            r, g, b = 0, c, x
        elif h < 4/6:
            r, g, b = 0, x, c
        elif h < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
