# celltracker/detectors/__init__.py

from .yolo_detector import YoloDetector
from .cellpose_detector import CellposeDetector

__all__ = ["YoloDetector", "CellposeDetector"]
