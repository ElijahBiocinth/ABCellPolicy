import numpy as np
from shapely.geometry import Polygon
from shapely.errors import TopologicalError
from .config import MIN_AREA, SIMPLIFY_TOL

def safe_polygon(coords):
    try:
        arr = np.asarray(coords, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 3:
            return None
        p = Polygon(arr)
        if p.is_empty:
            return None
        if not p.is_valid:
            p = p.buffer(0)
        if not isinstance(p, Polygon) or p.is_empty or p.area < MIN_AREA:
            return None
        return p
    except Exception:
        return None

def simplify_poly(p: Polygon):
    try:
        sp = p.simplify(SIMPLIFY_TOL, preserve_topology=True)
        if sp.is_empty or not isinstance(sp, Polygon):
            return p
        return sp
    except Exception:
        return p

def compute_iou(p1: Polygon, p2: Polygon):
    try:
        inter = p1.intersection(p2).area
        union = p1.area + p2.area - inter
        if union <= 0:
            return 0.0
        return inter / union
    except TopologicalError:
        return 0.0
    except Exception:
        return 0.0

def centroid(p: Polygon):
    c = p.centroid
    return c.x, c.y
