from kivy.graphics import Color, Line, Rectangle, RoundedRectangle
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, BooleanProperty
import numpy as np
from typing import Dict, List
from modules.result_repository import DroneDetection

class DroneMap(Widget):
    """Widget for visualizing drone positions and movements."""

    scale = NumericProperty(1.0) 
    visible = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tracks = {}
        self.track_colors = {}
        self.frame_size = (1920, 1080)
        self.bind(size=self._update_canvas, pos=self._update_canvas, visible=self._update_canvas)

    def _update_canvas(self, *args):
        """Update the canvas when the widget size or position changes."""
        self.canvas.clear()
        
        if not self.visible:
            return

        with self.canvas:
            frame_aspect = self.frame_size[0] / self.frame_size[1]
            widget_aspect = self.width / self.height

            if widget_aspect > frame_aspect:
                scaled_width = self.height * frame_aspect
                scaled_height = self.height
                x_offset = (self.width - scaled_width) / 2
                y_offset = 0
            else:
                scaled_width = self.width
                scaled_height = self.width / frame_aspect
                x_offset = 0
                y_offset = (self.height - scaled_height) / 2

            self.scale = scaled_width / self.frame_size[0]

            with self.canvas.before:
                self.canvas.before.clear()
                Color(1, 1, 1, 1)
                RoundedRectangle(
                    pos=(self.x + x_offset, self.y + y_offset),
                    size=(scaled_width, scaled_height),
                    radius=[20,]
                )

            for track_id, positions in self.tracks.items():
                if not positions:
                    continue

                color = self.track_colors.get(track_id, (1, 1, 1, 1))
                Color(*color)

                points = []
                for pos in positions:
                    x = self.x + x_offset + pos[0] * self.scale
                    y = self.y + y_offset + (self.frame_size[1] - pos[1]) * self.scale
                    points.extend([x, y])
                
                if len(points) >= 4:
                    Line(points=points, width=1.5)

                if positions:
                    last_pos = positions[-1]
                    x = self.x + x_offset + last_pos[0] * self.scale
                    y = self.y + y_offset + (self.frame_size[1] - last_pos[1]) * self.scale
                    Line(circle=(x, y, 5), width=2)

            with self.canvas.after:
                self.canvas.after.clear()

    def set_frame_size(self, width: int, height: int):
        """Set the frame size for coordinate scaling.

        Parameters
        ----------
        width : int
            Frame width in pixels
        height : int
            Frame height in pixels
        """
        self.frame_size = (width, height)
        self._update_canvas()

    def update_tracks(self, tracks: Dict[int, List[DroneDetection]]):
        """Update the tracks with new detections.

        Parameters
        ----------
        tracks : Dict[int, List[DroneDetection]]
            Dictionary mapping track IDs to their detections
        """
        self.tracks.clear()
        self.track_colors.clear()

        for track_id, detections in tracks.items():
            if not detections:
                continue

            positions = []
            for det in detections:
                bbox = det.bbox
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                positions.append((center_x, center_y))

            self.tracks[track_id] = positions
            self.track_colors[track_id] = self._get_track_color(track_id)

        self._update_canvas()

    def _get_track_color(self, track_id: int) -> tuple:
        """Generate a unique color for a track ID.

        Parameters
        ----------
        track_id : int
            Track ID to generate color for

        Returns
        -------
        tuple
            RGBA color tuple
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

        return (r + m, g + m, b + m, 1.0)

    def clear(self):
        """Clear all tracks."""
        self.tracks.clear()
        self.track_colors.clear()
        self._update_canvas()

    def set_visible(self, visible: bool):
        """Set the visibility of the map.

        Parameters
        ----------
        visible : bool
            Whether the map should be visible
        """
        self.visible = visible
        self._update_canvas()
