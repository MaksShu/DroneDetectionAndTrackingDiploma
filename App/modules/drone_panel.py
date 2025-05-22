from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.factory import Factory
from typing import Dict, List
from modules.result_repository import DroneDetection
import os

class DronePanel(BoxLayout):
    """Panel for displaying detected drones and their information."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = 10
        self.padding = 10
        self.drone_list = None
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components."""
        # Title
        title = Label(
            text='Detected Drones',
            font_size='24sp',
            bold=True,
            size_hint_y=None,
            height='50dp',
            color=(0.2, 0.2, 0.2, 1)
        )
        self.add_widget(title)

        # Scrollable list
        scroll = ScrollView()
        self.drone_list = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            spacing=10,
            padding=10
        )
        self.drone_list.bind(minimum_height=self.drone_list.setter('height'))
        scroll.add_widget(self.drone_list)
        self.add_widget(scroll)

    def update_drones(self, tracks: Dict[int, List[DroneDetection]]):
        """Update the drone list with new detections.

        Parameters
        ----------
        tracks : Dict[int, List[DroneDetection]]
            Dictionary mapping track IDs to their detections
        """
        self.drone_list.clear_widgets()

        for track_id, detections in tracks.items():
            if not detections:
                continue

            avg_confidence = sum(d.confidence for d in detections) / len(detections)

            first_detection = detections[0]
            last_detection = detections[-1]

            card = Factory.DroneCard()
            image_path = f'drones/drone_{track_id}.png'
            if os.path.exists(image_path):
                card.ids.image.source = ''
                card.ids.image.source = image_path
                card.ids.image.reload()

            card.ids.label.text = f'DRN-{track_id:03d}'
            card.ids.first.text = f'First detected at {self.seconds_to_time(first_detection.timestamp)}'
            card.ids.last.text = f'Last detected at {self.seconds_to_time(last_detection.timestamp)}'
            card.ids.confidence.text = f'Avg. confidence: {avg_confidence:.2f}'

            self.drone_list.add_widget(card)

    def seconds_to_time(self, seconds):
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def clear(self):
        """Clear all drone cards."""
        self.drone_list.clear_widgets()
