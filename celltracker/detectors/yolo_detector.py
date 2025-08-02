import cv2
from ultralytics import YOLO
from .base import BaseDetector

class YoloDetector(BaseDetector):
    def __init__(self, model_path: str, device: str):
        self.model = YOLO(model_path)
        dev = device if not device.isdigit() else f"cuda:{device}"
        self.model.to(dev)

    def detect(self, tile_img):
        results = self.model.predict(source=tile_img, verbose=False, save=False)
        masks = []
        for r in results:
            if hasattr(r, "masks") and r.masks is not None:
                for coords in r.masks.xy:
                    masks.append(coords)
        return masks
