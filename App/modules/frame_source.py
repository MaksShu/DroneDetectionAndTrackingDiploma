import cv2
import numpy as np
from typing import Optional
from yt_dlp import YoutubeDL
import time


class FrameSource:
    """Abstract frame source that yields video frames to the pipeline."""

    def get_frame(self) -> Optional[np.ndarray]:
        """Return next RGB frame as a NumPy array, or None when finished.
        
        Returns
        -------
        Optional[np.ndarray]
            RGB frame as numpy array, or None if no more frames
        """
        raise NotImplementedError

    def get_fps(self) -> float:
        """Get the frame rate of the video source.
        
        Returns
        -------
        float
            Frames per second
        """
        raise NotImplementedError

    def get_total_frames(self) -> int:
        """Get the total number of frames in the video source.
        
        Returns
        -------
        int
            Total number of frames
        """
        raise NotImplementedError

    def release(self):
        """Release any resources used by the frame source."""
        raise NotImplementedError


class VideoFileSource(FrameSource):
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")

    def get_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_total_frames(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def release(self):
        self.cap.release()


class URLVideoSource(FrameSource):
    def __init__(self, url: str):
        """
        Stream a YouTube URL via yt_dlp, without downloading.
        Exposes accurate FPS and an estimated total‐frame count.
        """
        ydl_opts = {
            "format": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            "quiet": True,
            "skip_download": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        stream_url = info["requested_formats"][0].get("url")
        if not stream_url:
            raise ValueError(f"yt_dlp failed to extract a playable URL from {url}")

        fps = info.get("fps") or 0.0
        duration = info.get("duration") or 0

        self.cap = cv2.VideoCapture(stream_url)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video stream: {stream_url}")

        cap_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.
        self._fps = float(fps or cap_fps)
        self._total_frames = int(self._fps * duration) if (self._fps and duration) else -1

    def get_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_fps(self) -> float:
        return self._fps

    def get_total_frames(self) -> int:
        return self._total_frames

    def release(self):
        self.cap.release()


class StreamSource(FrameSource):
    def __init__(
        self,
        stream: str,
        username: str = None,
        password: str = None,
        reconnect_delay: float = 5.0,
        max_retries: int = 5,
    ):
        """
        stream: RTSP URI (e.g. "rtsp://<ip>:554/stream")
        username/password: credentials if required
        reconnect_delay: seconds between reconnect attempts
        max_retries: how many times to retry before giving up
        """
        if username and password:
            prefix, rest = stream.split("://", 1)
            stream = f"{prefix}://{username}:{password}@{rest}"

        self.url = stream
        self.reconnect_delay = reconnect_delay
        self.max_retries = max_retries
        self.cap = None
        self._open_capture()

    def _open_capture(self):
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open RTSP stream (UDP): {self.url}")
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def _reconnect(self) -> bool:
        """Attempt to reconnect up to max_retries times."""
        if self.cap:
            self.cap.release()
        for attempt in range(1, self.max_retries + 1):
            time.sleep(self.reconnect_delay)
            try:
                self._open_capture()
            except ValueError:
                continue
            if self.cap.isOpened():
                return True
        return False

    def get_frame(self) -> Optional[np.ndarray]:
        """Read one frame; if it fails, try reconnecting once."""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            if self._reconnect():
                ret, frame = self.cap.read()
            else:
                return None

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_fps(self) -> float:
        """Return the camera’s reported FPS."""
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_total_frames(self) -> int:
        """Live streams don’t have a total frame count."""
        return -1

    def release(self):
        """Clean up the VideoCapture."""
        if self.cap:
            self.cap.release()
