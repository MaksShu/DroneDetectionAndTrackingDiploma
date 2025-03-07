import os
from kivy.app import App
from kivy.factory import Factory
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.graphics.texture import Texture
from plyer import filechooser
from kivy.clock import Clock
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

Window.clearcolor = (0.95, 0.95, 0.95, 1)

Builder.load_file('layout.kv')


def seconds_to_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


class DroneTrackingApp(App):
    current_time = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__()
        self.timer = None
        self.current_time = '00:00:00'
        self.video_caption = None
        self.video_image = None
        self.drone_list = None
        self.load_btn = None
        self.video_container = None
        self.left_panel = None
        self.fps = None
        self.total_frames = None
        self.tracker = None
        self.cap = None
        self.model = None
        self.video_path = None
        self.user_adjusting_slider = None
        self.last_detected_frame = None
        self.saved_frames = None
        self.current_frame_index = None
        self.slider = None
        self.image = None
        self.select_button = None
        self.layout = None
        self.track_colors = None

    def build(self):
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
        self.root.canvas.before.clear()
        with self.root.canvas.before:
            Color(0.1, 0.1, 0.1, 1)
            Rectangle(
                pos=(self.root.x, self.root.y),
                size=(self.root.width * 0.6, self.root.height),
            )

    def setup_left_panel(self):
        self.left_panel = BoxLayout(orientation='vertical', size_hint_x=0.6, spacing=10)

        self.video_container = Factory.VideoContainer(size_hint=(1, 0.8))

        self.load_btn = Factory.RoundedButton(
            text='Load Video',
            size_hint=(None, None),
            size=('150dp', '50dp'),
            pos_hint={'center_x': 0.5, 'center_y': 0.5}
        )
        self.load_btn.bind(on_press=self.open_file_chooser)

        self.video_container.add_widget(self.load_btn)

        self.left_panel.add_widget(self.video_container)

        controls = Factory.ControlContainer(size_hint_y=0.2)
        controls.add_widget(Factory.TimeHeader())

        self.slider = Slider()
        self.slider.bind(value=self.on_slider_value_change)
        self.slider.bind(on_touch_down=self.on_slider_touch_down)
        self.slider.bind(on_touch_up=self.on_slider_touch_up)
        controls.add_widget(self.slider)


        self.user_adjusting_slider = False

        Window.bind(on_mouse_up=self.force_slider_release)

        btn_row = BoxLayout(size_hint_y=None, height=40, spacing=10)

        btn_row.add_widget(Factory.RoundedButton(text='<-', width='20dp'))
        btn_row.add_widget(Factory.RoundedButton(text='| |', width='20dp', background_color=(0, 0.5, 1, 1)))
        btn_row.add_widget(Factory.RoundedButton(text='->', width='20dp', on_press=self.open_file_chooser))
        controls.add_widget(btn_row)

        self.left_panel.add_widget(controls)
        self.root.add_widget(self.left_panel)

    def setup_right_panel(self):
        right_panel = BoxLayout(orientation='vertical', size_hint_x=0.4, spacing=10)

        title = Label(
            text='Detected Drones',
            font_size='24sp',
            bold=True,
            size_hint_y=None,
            height='50dp',
            color=(0.2, 0.2, 0.2, 1)
        )
        right_panel.add_widget(title)

        scroll = ScrollView()
        self.drone_list = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            spacing=10,
            padding=10
        )
        self.drone_list.bind(minimum_height=self.drone_list.setter('height'))

        drones = [
            {'id': 'DRN-001', 'time': '12:30:15'},
            {'id': 'DRN-002', 'time': '12:35:22'},
            {'id': 'DRN-003', 'time': '12:42:08'},
            {'id': 'DRN-004', 'time': '12:44:33'}
        ]

        for drone in drones:
            card = Factory.DroneCard()
            card.ids.image.source = f'D:\Мої файли\Projects\DroneDetectionAndTrackingDiploma\App\ex1.png'
            card.ids.label.text = drone['id']
            card.ids.first.text = f'Detected at {drone["time"]}'
            card.ids.last.text = f'Last detected at {drone["time"]}'
            self.drone_list.add_widget(card)

        scroll.add_widget(self.drone_list)
        right_panel.add_widget(scroll)
        self.root.add_widget(right_panel)

    def open_file_chooser(self, instance):
        filechooser.open_file(on_selection=self.load_video)

    def on_slider_value_change(self, instance, value):
        if not self.user_adjusting_slider:
            return

        new_index = int(value)
        if new_index <= self.last_detected_frame:
            self.current_frame_index = new_index
            if self.current_frame_index < len(self.saved_frames):
                self.display_frame(self.saved_frames[self.current_frame_index])
            self.slider.value = self.current_frame_index
        else:
            self.slider.value = self.last_detected_frame

    def on_slider_touch_down(self, instance, touch):
        if instance.collide_point(*touch.pos):
            self.user_adjusting_slider = True

    def on_slider_touch_up(self, instance, touch):
        if instance.collide_point(*touch.pos):
            self.user_adjusting_slider = False

    def force_slider_release(self, *args):
        self.user_adjusting_slider = False

    def load_video(self, selection):
        if selection:
            self.video_container.clear_widgets()

            self.video_image = Image(
                allow_stretch=True,
                keep_ratio=True,
                size_hint=(1, 1),
                pos_hint={'x': 0, 'y': 0}
            )

            meta_box = BoxLayout(
                size_hint=(1, None),
                height='30dp',
                pos_hint={'top': 1},
                padding=10
            )
            self.video_caption = Label(
                text='video.mp4 | 60 FPS',
                color=(1, 1, 1, 1),
                halign='left'
            )
            meta_box.add_widget(self.video_caption)

            self.video_container.add_widget(self.video_image)
            self.video_container.add_widget(meta_box)

            self.video_path = selection[0]
            self.current_time = '00:00:00'
            self.saved_frames = []
            self.current_frame_index = 0
            self.last_detected_frame = 0
            self.start_detection()

    def start_detection(self):
        self.model = YOLO("D:\\Мої файли\\Projects\\DroneDetectionAndTrackingDiploma\\App\\yolov8s.pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        self.cap = cv2.VideoCapture(self.video_path)

        self.tracker = DeepSort(max_age=30)
        self.track_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
            (128, 0, 255), (0, 255, 128), (0, 128, 255), (255, 255, 128), (255, 128, 255),
            (128, 255, 255), (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0),
            (192, 0, 192), (0, 192, 192), (192, 128, 0), (192, 0, 128), (128, 192, 0)
        ]

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.max = self.total_frames

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.video_caption.text = f'{os.path.basename(self.video_path)} | {int(self.fps)} FPS'

        if self.timer:
            Clock.unschedule(self.timer)
        self.timer = Clock.schedule_interval(self.update, 1.0 / self.fps)

    def update(self, dt):
        if not self.user_adjusting_slider:
            self.current_time = seconds_to_time(self.current_frame_index / self.fps)
            if self.current_frame_index < len(self.saved_frames):
                frame = self.saved_frames[self.current_frame_index]
            else:
                ret, frame = self.cap.read()

                if not ret or self.current_frame_index >= self.total_frames:
                    # self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # self.current_frame_index = 0
                    # self.slider.value = 0
                    return

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.model(frame)

                detections = []

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        if conf > 0.5:
                            detections.append(([x1, y1, x2, y2], conf, cls))

                tracked_objects = self.tracker.update_tracks(detections, frame=frame)
                for track in tracked_objects:
                    if not track.is_confirmed():
                        continue
                    x1, y1, x2, y2 = map(int, track.to_ltrb())
                    track_id = track.track_id

                    color = self.track_colors[int(track_id) % len(self.track_colors)]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f'ID {track_id}'

                    font_scale = 1.0
                    thickness = 2
                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                    text_x = max(0, min(x1, frame.shape[1] - text_w))
                    text_y = max(y1, text_h + 10)

                    cv2.rectangle(frame, (text_x, text_y - text_h - 5), (text_x + text_w, text_y + baseline), color, -1)

                    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                                thickness, cv2.LINE_AA)

                frame = cv2.flip(frame, 0)
                self.saved_frames.append(frame)
                self.last_detected_frame = self.current_frame_index

            self.display_frame(frame)
            self.current_frame_index += 1
            self.slider.value = self.current_frame_index

    def display_frame(self, frame):
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.video_image.texture = texture


if __name__ == '__main__':
    DroneTrackingApp().run()
