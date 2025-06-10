from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
import numpy as np
import time
from collections import deque


class VideoPlayer:
    """Handles video playback and frame display."""

    def __init__(self, max_buffer_size: int = 300):
        self.max_buffer_size = max_buffer_size
        self.saved_frames = deque(maxlen=max_buffer_size)
        self.processed_frames = deque(maxlen=100)
        self.current_frame_index = 0
        self.last_detected_frame = 0
        self._is_playing = True
        self.display_timer = None
        self.process_timer = None
        self.fps = 30
        self.total_frames = 0
        self.video_image = None
        self.texture = None
        self.on_frame_update = None
        self.on_frame_index_update = None
        self.on_play = None
        self.on_pause = None
        self.on_stop = None
        self.on_fps_update = None
        self.current_fps = 0.0
        self.is_processing = False
        self.last_frame_time = None
        self.frames_processed = 0
        self.fps_update_interval = 10
        self.frame_skip = 1

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
        self.last_frame_time = time.time()
        if self.display_timer:
            Clock.unschedule(self.display_timer)
        if self.process_timer:
            Clock.unschedule(self.process_timer)
            
        self._is_playing = True

        self.display_timer = Clock.schedule_interval(self.update_display, 1.0 / self.fps)
        self.process_timer = Clock.schedule_interval(self.process_frame, 1.0 / (self.fps * 2))
        
        if self.on_play:
            self.on_play()

    def stop_playback(self):
        """Stop video playback."""
        if self.display_timer:
            Clock.unschedule(self.display_timer)
            self.display_timer = None
        if self.process_timer:
            Clock.unschedule(self.process_timer)
            self.process_timer = None
        self._is_playing = False
        if self.on_stop:
            self.on_stop()

    def pause(self):
        """Pause video playback."""
        if self.display_timer:
            Clock.unschedule(self.display_timer)
            self.display_timer = None
        if self.process_timer:
            Clock.unschedule(self.process_timer)
            self.process_timer = None
        self._is_playing = False
        if self.on_pause:
            self.on_pause()

    def resume(self):
        """Resume video playback."""
        if not self.display_timer and not self.process_timer:
            self._is_playing = True

            self.display_timer = Clock.schedule_interval(self.update_display, 1.0 / self.fps)
            self.process_timer = Clock.schedule_interval(self.process_frame, 1.0 / (self.fps * 2))
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
        seek_index = frame_index - min_available_frame
        if seek_index < len(self.saved_frames):
            self.display_frame(self.saved_frames[seek_index])
            if self.on_frame_index_update:
                self.on_frame_index_update(self.current_frame_index)

    def process_frame(self, dt):
        """Process frames as fast as possible."""
        if not self._is_playing or self.is_processing:
            return

        self.is_processing = True
        
        if self.on_frame_update:
            if self.current_frame_index % self.frame_skip != 0:
                self.on_frame_update(skip_frame=True)
                self.current_frame_index += 1
                if self.on_frame_index_update:
                    self.on_frame_index_update(self.current_frame_index)
                self.is_processing = False
                return

            frame = self.on_frame_update()
            if frame is not None:
                self.frames_processed += 1
                if self.frames_processed >= self.fps_update_interval:
                    current_time = time.time()
                    time_diff = current_time - self.last_frame_time
                    if time_diff > 0:
                        self.current_fps = self.fps_update_interval / time_diff
                        self.frame_skip = min(max(1, round(self.fps / self.current_fps)), 1)
                        if self.on_fps_update:
                            self.on_fps_update(self.current_fps)
                    self.last_frame_time = current_time
                    self.frames_processed = 0

                if len(self.saved_frames) >= self.max_buffer_size:
                    self.saved_frames.popleft()
                self.saved_frames.append(frame)
                self.processed_frames.append(frame)
                self.last_detected_frame = self.current_frame_index
            else:
                if self.on_fps_update:
                    self.on_fps_update(None)
                self.stop_playback()

        self.is_processing = False

    def update_display(self, dt):
        """Update display at video frame rate."""
        if not self._is_playing or not self.processed_frames:
            return

        frame = self.processed_frames.popleft()
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

        buf = frame.tobytes()
        
        if self.texture is None:
            self.texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            self.texture.flip_vertical()
            
        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.video_image.texture = self.texture

    def cleanup(self):
        """Clean up resources."""
        self.stop_playback()
        self.saved_frames.clear()
        self.processed_frames.clear()
        self.current_frame_index = 0
        self.last_detected_frame = 0
        self._is_playing = False
        self.frames_processed = 0
        self.texture = None
