from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
import numpy as np
import cv2

class VideoPlayer:
    """Handles video playback and frame display."""

    def __init__(self, max_buffer_size: int = 300):
        self.max_buffer_size = max_buffer_size
        self.saved_frames = []
        self.current_frame_index = 0
        self.last_detected_frame = 0
        self._is_playing = True
        self.timer = None
        self.fps = 30
        self.total_frames = 0
        self.video_image = None
        self.on_frame_update = None
        self.on_frame_index_update = None
        self.on_play = None
        self.on_pause = None
        self.on_stop = None

    @property
    def is_playing(self):
        """Get the current playback state."""
        return self._is_playing

    def setup_display(self, video_image: Image):
        """Set up the display widget.

        Parameters
        ----------
        video_image : Image
            Kivy Image widget for displaying frames
        """
        self.video_image = video_image

    def start_playback(self, fps: float):
        """Start video playback.

        Parameters
        ----------
        fps : float
            Frames per second for playback
        """
        self.fps = fps
        if self.timer:
            Clock.unschedule(self.timer)
        self._is_playing = True
        self.timer = Clock.schedule_interval(self.update, 1.0 / self.fps)
        if self.on_play:
            self.on_play()

    def stop_playback(self):
        """Stop video playback."""
        if self.timer:
            Clock.unschedule(self.timer)
            self.timer = None
        self._is_playing = False
        if self.on_stop:
            self.on_stop()

    def pause(self):
        """Pause video playback."""
        if self.timer:
            Clock.unschedule(self.timer)
            self.timer = None
        self._is_playing = False
        if self.on_pause:
            self.on_pause()

    def resume(self):
        """Resume video playback."""
        if not self.timer:
            self._is_playing = True
            self.timer = Clock.schedule_interval(self.update, 1.0 / self.fps)
            if self.on_play:
                self.on_play()

    def seek(self, frame_index: int):
        """Seek to a specific frame.

        Parameters
        ----------
        frame_index : int
            Target frame index
        """
        min_available_frame = max(0, self.last_detected_frame - self.max_buffer_size + 1)
        frame_index = max(min_available_frame, min(frame_index, self.last_detected_frame))
        self.current_frame_index = frame_index
        if self.current_frame_index < len(self.saved_frames):
            self.display_frame(self.saved_frames[self.current_frame_index])
            if self.on_frame_index_update:
                self.on_frame_index_update(self.current_frame_index)

    def update(self, dt):
        """Update frame display (called by Kivy clock).

        Parameters
        ----------
        dt : float
            Time delta since last update
        """
        if not self._is_playing:
            return

        if self.current_frame_index < len(self.saved_frames):
            frame = self.saved_frames[self.current_frame_index]
        else:
            if self.on_frame_update:
                frame = self.on_frame_update()
                if frame is None:
                    return
                if len(self.saved_frames) >= self.max_buffer_size:
                    self.saved_frames.pop(0)
                self.saved_frames.append(frame)
                self.last_detected_frame = self.current_frame_index
            else:
                return

        self.display_frame(frame)
        self.current_frame_index += 1

        if self.on_frame_index_update:
            self.on_frame_index_update(self.current_frame_index)

    def display_frame(self, frame: np.ndarray):
        """Display a frame on the video widget.

        Parameters
        ----------
        frame : np.ndarray
            RGB frame to display
        """
        if self.video_image is None:
            return

        frame = cv2.flip(frame, 0)
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.video_image.texture = texture

    def cleanup(self):
        """Clean up resources."""
        self.stop_playback()
        self.saved_frames = []
        self.current_frame_index = 0
        self.last_detected_frame = 0
        self._is_playing = False
