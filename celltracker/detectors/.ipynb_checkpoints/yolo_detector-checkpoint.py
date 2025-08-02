import cv2
import numpy as np
from ultralytics import YOLO
from .base import BaseDetector

class YoloDetector(BaseDetector):
    def __init__(self, model_path: str, device: str):
        self.model = YOLO(model_path)
        dev = device if not device.isdigit() else f"cuda:{device}"
        print(f"[YoloDetector] loading on device = {dev}")
        self.model.to(dev)

    def detect(self, tile_img: np.ndarray) -> list[np.ndarray]:
        results = self.model.predict(source=tile_img, verbose=False, save=False)
        coords = []
        for r in results:
            if hasattr(r, "masks") and r.masks is not None:
                for xy in r.masks.xy:
                    coords.append(xy.astype("int32"))
        return coords
