import cv2
import numpy as np
from cellpose import models
from .base import BaseDetector

class CellposeDetector(BaseDetector):
    def __init__(self, _, device: str, flow_threshold=0.2, cellprob_threshold=0.0):
        use_gpu = device.startswith("cuda")
        self.model = models.CellposeModel(model_type="cyto3", gpu=use_gpu)
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold

    def detect(self, tile_img):
        img = tile_img.astype("float32")
        if img.ndim == 3:
            img = img.mean(axis=2)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        masks, *_ = self.model.eval(
            img,
            diameter=None,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            normalize={"tile_norm_blocksize": 64},
        )

        coords_list = []
        for inst_id in range(1, int(masks.max()) + 1):
            mask = (masks == inst_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                coords_list.append(cnt.reshape(-1, 2))
        return coords_list
