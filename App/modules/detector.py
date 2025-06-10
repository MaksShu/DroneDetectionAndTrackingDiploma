import torch
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Optional, Union


class YOLOv12Detector:
    """Wrapper around a final YOLOv12 model for drone detection."""

    def __init__(self, model_path: str = "best.pt", device: Optional[str] = None):
        """Initialize the detector.
        
        Parameters
        ----------
        model_path : str, optional
            Path to the YOLO weights file, by default "best.pt"
        device : Optional[str], optional
            Device to run inference on, by default None (auto-detect)
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the YOLO model and prepare it for inference."""
        self.model = YOLO(self.model_path)
        self.model.model.fuse()
        self.model.eval()
        self.model = self.model.to(self.device)

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.1, 
               iou_threshold: float = 0.3) -> List[List[float]]:
        """Run detection on a single RGB frame.

        Parameters
        ----------
        frame : np.ndarray
            RGB frame to process
        conf_threshold : float, optional
            Confidence threshold for detections, by default 0.1
        iou_threshold : float, optional
            IoU threshold for NMS, by default 0.3

        Returns
        -------
        List[List[float]]
            List of detections in format [x1, y1, x2, y2, conf, class_id]
        """
        results = self.model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        if isinstance(results, list):
            results = results[0]

        return results.boxes.data.cpu().numpy()
