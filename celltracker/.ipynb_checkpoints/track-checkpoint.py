import numpy as np
from .geometry import centroid
from .color_utils import get_track_color
from .config import BASE_IOU_MATCH, ADAPTIVE_ALPHA

class Track:
    __slots__ = ("id", "poly", "feat", "age", "missed", "generation", "history",
                 "adaptive_iou", "vel", "area_ema", "feat_ema", "display_color")

    def __init__(self, tid, poly, feat):
        self.id = tid
        self.poly = poly
        self.feat = self._norm(feat)
        self.age = 1
        self.missed = 0
        self.generation = 1
        self.history = [centroid(poly)]
        self.adaptive_iou = BASE_IOU_MATCH
        self.vel = (0.0, 0.0)
        self.area_ema = poly.area
        self.feat_ema = self.feat.copy()
        self.display_color = get_track_color(self.id)

    def _norm(self, f):
        n = np.linalg.norm(f) + 1e-9
        return f / n

    def predict_centroid(self):
        cx, cy = self.history[-1]
        return (cx + self.vel[0], cy + self.vel[1])

    def update(self, poly, feat, matched_iou):
        c_prev = self.history[-1]
        c_new = centroid(poly)
        vx = c_new[0] - c_prev[0]; vy = c_new[1] - c_prev[1]
        self.vel = (0.8*self.vel[0] + 0.2*vx, 0.8*self.vel[1] + 0.2*vy)
        feat = self._norm(feat)
        self.poly = poly
        self.feat = feat
        self.feat_ema = 0.8*self.feat_ema + 0.2*feat
        self.area_ema = 0.9*self.area_ema + 0.1*poly.area
        self.age += 1
        self.missed = 0
        self.history.append(c_new)
        self.adaptive_iou = (1 - ADAPTIVE_ALPHA) * self.adaptive_iou + ADAPTIVE_ALPHA * matched_iou

    def mark_missed(self):
        self.missed += 1
