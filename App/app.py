import os
from kivy.app import App
from kivy.factory import Factory
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
from kivy.lang import Builder
from plyer import filechooser
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
import cv2
import numpy as np

from modules.detector import YOLOv12Detector
from modules.tracker import DroneTracker
from modules.video_player import VideoPlayer
from modules.result_repository import ResultRepository
from modules.frame_source import VideoFileSource, URLVideoSource, StreamSource
from modules.drone_panel import DronePanel
from modules.drone_map import DroneMap
from utils import seconds_to_time

Window.clearcolor = (0.95, 0.95, 0.95, 1)

Builder.load_file('layout.kv')


class DroneTrackingApp(App):
    """Main application class for drone detection and tracking."""
    
    current_time = StringProperty('')
    current_fps = StringProperty('- FPS')
    icon = 'icons/icon.png'

    def __init__(self, **kwargs):
        """Initialize the application."""
        super().__init__(**kwargs)
        self.current_time = '00:00:00'
        self.video_image = None
        self.slider = None
        self.user_adjusting_slider = False

        self.detector = YOLOv12Detector(model_path="best.pt")
        self.tracker = DroneTracker()
        self.video_player = VideoPlayer()
        self.result_repository = ResultRepository()
        self.frame_source = None
        self.drone_panel = None
        self.drone_map = None

    def build(self):
        """Build the application UI."""
        self.root = BoxLayout(orientation='horizontal', padding=20, spacing=20)
        self.root.bind(size=self.left_background)

        with self.root.canvas.before:
            Color(0.1, 0.1, 0.1, 1)
            Rectangle(
                pos=(self.root.x, self.root.y),
                size=(self.root.width * 0.6, self.root.height),
            )

        self.setup_left_panel()
        self.setup_right_panel()

        return self.root

    def left_background(self, *args):
        """Update left panel background when window is resized."""
        self.root.canvas.before.clear()
        with self.root.canvas.before:
            Color(0.1, 0.1, 0.1, 1)
            Rectangle(
                pos=(self.root.x, self.root.y),
                size=(self.root.width * 0.6, self.root.height),
            )

    def setup_left_panel(self):
        """Set up the left panel containing video display and controls."""
        left_panel = BoxLayout(orientation='vertical', size_hint_x=0.6, spacing=10)

        # File info label
        self.file_info_label = Label(
            text='',
            size_hint=(1, None),
            height='40dp',
            color=(1, 1, 1, 1),
            halign='center',
            valign='middle',
            font_size='16sp',
            opacity=0
        )
        left_panel.add_widget(self.file_info_label)

        # Video container
        video_container = Factory.VideoContainer(size_hint=(1, 0.8))

        # Button container
        button_container = BoxLayout(
            orientation='vertical',
            size_hint=(None, None),
            size=('280dp', '280dp'),
            pos_hint={'center_x': 0.5, 'center_y': 0.7},
            spacing=20,
            padding=[108, 0, 108, 0]
        )

        # First row - single button
        first_row = BoxLayout(
            orientation='horizontal',
            size_hint=(1, None),
            height='64dp',
            spacing=0,
            padding=[0, 0, 0, 0]
        )

        # Second row - two buttons
        second_row = BoxLayout(
            orientation='horizontal',
            size_hint=(1, None),
            height='64dp',
            spacing=20,
            padding=[0, 0, 0, 0]
        )

        # Create and add buttons
        file_btn = Factory.ImageButton(
            text='',
            size=('64dp', '64dp'),
            background_normal='icons/file.png',
            background_down='icons/file.png'
        )
        file_btn.bind(on_press=self.open_file_chooser)

        url_btn = Factory.ImageButton(
            text='',
            size=('64dp', '64dp'),
            background_normal='icons/url.png',
            background_down='icons/url.png'
        )
        url_btn.bind(on_press=self.open_url_input)

        stream_btn = Factory.ImageButton(
            text='',
            size=('64dp', '64dp'),
            background_normal='icons/stream.png',
            background_down='icons/stream.png'
        )
        stream_btn.bind(on_press=self.open_stream_input)

        first_row.add_widget(file_btn)
        second_row.add_widget(url_btn)
        second_row.add_widget(stream_btn)
        button_container.add_widget(first_row)
        button_container.add_widget(second_row)
        video_container.add_widget(button_container)

        # Video image widget
        self.video_image = Factory.VideoImage(
            allow_stretch=True,
            keep_ratio=True,
            size_hint=(1, 1),
            pos_hint={'x': 0, 'y': 0},
            opacity=0
        )
        video_container.add_widget(self.video_image)

        # Set up video player
        self.video_player.setup_display(self.video_image)
        self.video_player.on_frame_update = self.process_frame
        self.video_player.on_frame_index_update = self.update_slider_position
        self.video_player.on_play = self.on_video_play
        self.video_player.on_pause = self.on_video_pause
        self.video_player.on_stop = self.on_video_stop

        # Controls
        controls = Factory.ControlContainer(size_hint_y=0.25)
        controls.add_widget(Factory.TimeHeader())

        # Slider
        self.slider = Slider(
            min=0,
            max=1,
            value=0,
            step=1,
            cursor_size=(20, 20),
            background_width=4,
            value_track_color=(0, 0.5, 1, 1),
            value_track_width=4
        )
        self.slider.bind(value=self.on_slider_value_change)
        self.slider.bind(on_touch_down=self.on_slider_touch_down)
        self.slider.bind(on_touch_up=self.on_slider_touch_up)
        controls.add_widget(self.slider)

        Window.bind(on_mouse_up=self.force_slider_release)

        # Control buttons
        btn_row = BoxLayout(size_hint_y=None, height=40, spacing=10)
        prev_btn = Factory.RoundedButton(
            text='<-',
            width='20dp',
            background_color=(0.3, 0.3, 0.3, 1)
        )
        self.play_button = Factory.RoundedButton(
            text='|>',
            width='20dp',
            background_color=(0, 0.5, 1, 1)
        )
        next_btn = Factory.RoundedButton(
            text='->',
            width='20dp',
            background_color=(0.3, 0.3, 0.3, 1)
        )
        
        prev_btn.bind(on_press=self.restart_video)
        self.play_button.bind(on_press=self.toggle_playback)
        next_btn.bind(on_press=self.stop_and_show_buttons)
        
        btn_row.add_widget(prev_btn)
        btn_row.add_widget(self.play_button)
        btn_row.add_widget(next_btn)
        controls.add_widget(btn_row)

        left_panel.add_widget(video_container)
        left_panel.add_widget(controls)
        self.root.add_widget(left_panel)

    def setup_right_panel(self):
        """Set up the right panel containing drone information and map."""
        right_panel = BoxLayout(orientation='vertical', size_hint_x=0.4, spacing=10)

        # Drone panel
        self.drone_panel = DronePanel()
        right_panel.add_widget(self.drone_panel)

        # Map container
        map_container = BoxLayout(
            orientation='vertical',
            size_hint=(1, 0.4),
            padding=[10, 10, 10, 10]
        )

        # Drone map
        self.drone_map = DroneMap(
            size_hint=(1, 1),
            pos_hint={'center_x': 0.5, 'center_y': 0.5}
        )
        map_container.add_widget(self.drone_map)
        right_panel.add_widget(map_container)

        self.root.add_widget(right_panel)

    def process_frame(self, skip_frame=False):
        """Process a single frame from the video source."""
        if self.frame_source is None:
            return None

        frame = self.frame_source.get_frame()
        if frame is None:
            return None

        current_timestamp = self.video_player.current_frame_index / self.video_player.fps

        if skip_frame:
            return frame

        detections = self.detector.detect(frame)
        if len(detections) > 0:
            tracked_objects = self.tracker.update(np.array(detections), frame)

            for det in tracked_objects:
                if det is None:
                    continue
                x1, y1, x2, y2, track_id, class_id, conf = det
                track_id = int(track_id)

                if track_id not in self.result_repository.detections:
                    self.result_repository.save_drone_image(frame, x1, y1, x2, y2, track_id)

                self.result_repository.add_detection(
                    track_id=track_id,
                    bbox=np.array([x1, y1, x2, y2]),
                    confidence=float(conf),
                    class_id=int(class_id),
                    timestamp=current_timestamp
                )

                color = self.tracker.get_track_color(track_id)
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                cv2.rectangle(frame, pt1, pt2, color, 2)
                label = f'ID {track_id}'
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            all_tracks = self.result_repository.get_all_tracks()
            self.drone_panel.update_drones(all_tracks)
            self.drone_map.update_tracks(all_tracks)

        return frame

    def open_file_chooser(self, instance):
        """Open file chooser dialog."""
        filechooser.open_file(on_selection=self.load_video)

    def load_video(self, selection):
        """Load a video file."""
        if not selection:
            return

        self.cleanup()

        if isinstance(selection, list):
            path = selection[0]
            self.frame_source = VideoFileSource(path)
            self.filename = os.path.basename(path)
        elif isinstance(selection, str) and selection.lower().startswith(("http://", "https://")):
            self.frame_source = URLVideoSource(selection)
            self.filename = selection.split('/')[-1]
        elif isinstance(selection, StreamSource):
            self.frame_source = selection
            self.filename = selection.url

        self.video_player.total_frames = self.frame_source.get_total_frames()
        self.slider.max = self.video_player.total_frames
        self.slider.value = 0
        self.slider.disabled = False

        self.video_image.opacity = 1
        self.file_info_label.opacity = 1

        self.video_fps = self.frame_source.get_fps()
        self.file_info_label.text = f'{self.filename} | {self.video_fps:.0f} FPS | - FPS'

        self.video_player.on_fps_update = self.update_fps

        self.video_player.start_playback(self.frame_source.get_fps())

        self.drone_map.set_frame_size(
            int(self.frame_source.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.frame_source.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

        if self.drone_map:
            self.drone_map.opacity = 1
            self.drone_map.set_visible(True)

    def on_slider_value_change(self, instance, value):
        """Handle slider value changes."""
        if self.frame_source is None:
            return

        self.current_time = seconds_to_time(value / self.video_player.fps)

        if self.user_adjusting_slider:
            self.video_player.seek(int(value))

    def on_slider_touch_down(self, instance, touch):
        """Handle slider touch down event."""
        if instance.collide_point(*touch.pos) and self.frame_source is not None:
            self.user_adjusting_slider = True
            self.video_player.pause()
            self.play_button.text = '|>'

    def on_slider_touch_up(self, instance, touch):
        """Handle slider touch up event."""
        if instance.collide_point(*touch.pos) and self.frame_source is not None:
            self.user_adjusting_slider = False
            self.video_player.seek(int(instance.value))
            self.video_player.resume()
            self.play_button.text = '| |'

    def force_slider_release(self, *args):
        """Force release of slider when mouse button is released."""
        if self.user_adjusting_slider:
            self.user_adjusting_slider = False
            self.video_player.seek(int(self.slider.value))
            self.video_player.resume()
            self.play_button.text = '| |'

    def cleanup(self):
        """Clean up resources."""
        if self.frame_source:
            self.frame_source.release()
            self.frame_source = None
        
        self.video_player.cleanup()
        self.tracker.reset()
        self.result_repository.cleanup()
        self.drone_panel.clear()

        self.video_image.opacity = 0
        self.file_info_label.opacity = 0
        self.file_info_label.text = ''

        if self.drone_map:
            self.drone_map.opacity = 0
            self.drone_map.clear()

    def open_url_input(self, instance):
        """Handle opening a video from URL."""
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)

        url_input = TextInput(
            hint_text='Paste video URL here',
            multiline=False,
            size_hint_y=None,
            height='40dp'
        )
        content.add_widget(url_input)

        buttons = BoxLayout(size_hint_y=None, height='40dp', spacing=10)
        load_btn = Button(text='Load')
        cancel_btn = Button(text='Cancel')
        buttons.add_widget(load_btn)
        buttons.add_widget(cancel_btn)
        content.add_widget(buttons)

        popup = Popup(
            title='Open Video from URL',
            content=content,
            size_hint=(None, None),
            size=('400dp', '200dp'),
            auto_dismiss=False
        )

        load_btn.bind(on_release=lambda btn: (
            self.load_video(url_input.text.strip()) if url_input.text.strip() else None,
            popup.dismiss()
        ))
        cancel_btn.bind(on_release=lambda btn: popup.dismiss())

        popup.open()

    def open_stream_input(self, instance):
        """Handle opening a video stream via popup dialog."""
        form = GridLayout(cols=2, spacing=10, padding=10, size_hint_y=None)
        form.bind(minimum_height=form.setter('height'))

        form.add_widget(Label(text='RTSP URL:', size_hint_x=None, width=100))
        self.url_input = TextInput(multiline=False, size_hint_y=None, height=30)
        form.add_widget(self.url_input)

        form.add_widget(Label(text='Username:', size_hint_x=None, width=100))
        self.user_input = TextInput(multiline=False, size_hint_y=None, height=30)
        form.add_widget(self.user_input)

        form.add_widget(Label(text='Password:', size_hint_x=None, width=100))

        self.pass_input = TextInput(password=True, multiline=False, size_hint_y=None, height=30)
        form.add_widget(self.pass_input)

        btn_layout = BoxLayout(size_hint_y=None, height=40, spacing=10, padding=(10,0))
        btn_open = Button(text='Open', size_hint_x=0.5)
        btn_cancel = Button(text='Cancel', size_hint_x=0.5)
        btn_layout.add_widget(btn_open)
        btn_layout.add_widget(btn_cancel)

        content = BoxLayout(orientation='vertical')
        content.add_widget(form)
        content.add_widget(btn_layout)

        self._stream_popup = Popup(
            title='Open RTSP Stream',
            content=content,
            size_hint=(0.8, None),
            height=280,
        )

        btn_open.bind(on_press=self._on_open_stream_confirm)
        btn_cancel.bind(on_press=lambda btn: self._stream_popup.dismiss())

        self._stream_popup.open()

    def _on_open_stream_confirm(self, btn):
        url = self.url_input.text.strip()
        username = self.user_input.text.strip() or None
        password = self.pass_input.text or None

        self._stream_popup.dismiss()

        if not url:
            Popup(title='Error',
                  content=Label(text='URL cannot be empty'),
                  size_hint=(0.6, None), height=150).open()
            return

        try:
            self.stream_source = StreamSource(
                stream=url,
                username=username,
                password=password,
            )

            self.load_video(self.stream_source)

        except ValueError as e:
            Popup(title='Connection Error',
                  content=Label(text=str(e)),
                  size_hint=(0.6, None), height=150).open()

    def on_stop(self):
        """Called when the application is closing."""
        self.cleanup()

    def seek_relative(self, offset):
        """Seek relative to current position."""
        if self.frame_source is None:
            return
        new_value = self.slider.value + offset
        new_value = max(0, min(new_value, self.slider.max))
        self.slider.value = new_value
        self.video_player.seek(int(new_value))

    def toggle_playback(self, instance):
        """Toggle video playback."""
        if self.frame_source is None:
            return
        if self.video_player.is_playing:
            self.video_player.pause()
            instance.text = '|>'
        else:
            self.video_player.resume()
            instance.text = '| |'

    def update_slider_position(self, frame_index):
        """Update slider position based on current frame index."""
        if self.frame_source is None or self.user_adjusting_slider:
            return
        self.slider.value = frame_index
        self.current_time = seconds_to_time(frame_index / self.video_player.fps)

    def restart_video(self, instance):
        """Restart video playback from the beginning."""
        if self.frame_source is None:
            return

        self.video_player.current_frame_index = 0
        self.slider.value = 0
        self.current_time = '00:00:00'

        self.video_player.resume()

    def stop_and_show_buttons(self, instance):
        """Stop video playback and show input buttons."""
        if self.frame_source is None:
            return

        self.video_player.pause()

        self.cleanup()

        self.slider.value = 0
        self.current_time = '00:00:00'
        self.play_button.text = '|>'

    def on_video_play(self):
        """Handle video play event."""
        self.play_button.text = '| |'
        if self.drone_map:
            self.drone_map.set_visible(True)

    def on_video_pause(self):
        """Handle video pause event."""
        self.play_button.text = '|>'

    def on_video_stop(self):
        """Handle video stop event."""
        self.play_button.text = '|>'
        self.slider.value = 0
        self.current_time = '00:00:00'

    def update_fps(self, fps):
        """Update the FPS display."""
        if fps is None:
            self.file_info_label.text = f'{self.filename} | {self.video_fps:.0f} FPS | - FPS'
        else:
            self.file_info_label.text = f'{self.filename} | {self.video_fps:.0f} FPS | {fps:.1f} FPS'


if __name__ == '__main__':
    DroneTrackingApp().run()
