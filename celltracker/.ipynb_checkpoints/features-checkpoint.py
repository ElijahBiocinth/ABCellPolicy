import cv2
import numpy as np
import math
from shapely.geometry import Polygon
from numba import njit
from .config import USE_INTENSITY, USE_HU, USE_MOMENTS

def extract_mask(img, poly: Polygon):
    minx, miny, maxx, maxy = map(int, poly.bounds)
    h_img, w_img = img.shape[:2]
    minx = max(minx, 0)
    miny = max(miny, 0)
    maxx = min(maxx, w_img - 1)
    maxy = min(maxy, h_img - 1)
    w = maxx - minx + 1
    h = maxy - miny + 1
    if w <= 0 or h <= 0:
        return None, None, None
    roi = img[miny:maxy + 1, minx:maxx + 1]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[(x - minx, y - miny) for x, y in np.array(poly.exterior.coords)]], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    return roi, mask, (minx, miny)

def poly_features(img, poly: Polygon):
    feat = []
    roi, mask, _ = extract_mask(img, poly)
    if roi is None:
        return np.zeros(24, dtype=np.float32)

    moments = cv2.moments(mask, binaryImage=True)

    if USE_INTENSITY:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inside = gray[mask == 255]
        if inside.size:
            mean_val = inside.mean(); std_val = inside.std()
            p10 = np.percentile(inside, 10); p50 = np.percentile(inside, 50); p90 = np.percentile(inside, 90)
            contrast = (p90 - p10) / (p90 + p10 + 1e-6)
        else:
            mean_val = std_val = p10 = p50 = p90 = contrast = 0.0
        feat.extend([mean_val/255.0, std_val/255.0, p10/255.0, p50/255.0, p90/255.0, contrast])

    if USE_HU:
        h = cv2.HuMoments(moments).flatten()
        h = np.array([-np.sign(v) * math.log10(abs(v) + 1e-12) for v in h], dtype=np.float32)
        feat.extend(h.tolist())

    if USE_MOMENTS:
        if moments['mu20'] + moments['mu02'] != 0 and moments['m00'] != 0:
            cov_xx = moments['mu20'] / moments['m00']; cov_yy = moments['mu02'] / moments['m00']
            cov_xy = moments['mu11'] / moments['m00']
            cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
            evals, _ = np.linalg.eig(cov)
            evals = np.sort(np.abs(evals))
            ecc = 1 - (evals[0] / (evals[1] + 1e-9)) if evals[1] > 0 else 0.0
        else:
            ecc = 0.0
        cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnt:
            area = cv2.contourArea(cnt[0])
            hull = cv2.convexHull(cnt[0])
            hull_area = cv2.contourArea(hull) if hull is not None else area
            solidity = area / (hull_area + 1e-6)
        else:
            solidity = 0.0
        feat.extend([float(ecc), float(solidity)])

    minx, miny, maxx, maxy = poly.bounds
    bw = maxx - minx + 1e-6; bh = maxy - miny + 1e-6
    aspect = bw / bh; fill = poly.area / (bw * bh)
    feat.extend([float(aspect), float(fill)])

    feat = np.array(feat, dtype=np.float32)
    if feat.shape[0] < 24:
        feat = np.pad(feat, (0, 24 - feat.shape[0]))
    else:
        feat = feat[:24]
    n = np.linalg.norm(feat) + 1e-9
    return feat / n

@njit(cache=True, fastmath=True)
def _feature_distance_numba(f1, f2):
    cos = (f1 * f2).sum() / ((np.linalg.norm(f1) * np.linalg.norm(f2)) + 1e-9)
    cos_d = (1 - cos) * 0.5
    l1 = np.mean(np.abs(f1 - f2))
    return 0.5 * cos_d + 0.5 * l1

def feature_distance(f1, f2):
    if f1 is None or f2 is None:
        return 1.0
    from . import config
    if config._NUMBA:
        return float(_feature_distance_numba(f1.astype(np.float32), f2.astype(np.float32)))
    cos = float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-9))
    cos_d = (1 - cos) * 0.5
    l1 = np.mean(np.abs(f1 - f2))
    return 0.5 * cos_d + 0.5 * l1

def shape_distance(p1: Polygon, p2: Polygon):
    a1, a2 = p1.area, p2.area
    if a1 <= 0 or a2 <= 0:
        return 1.0
    ar = abs(a1 - a2) / max(a1, a2)
    per1 = p1.length; per2 = p2.length
    pr = abs(per1 - per2) / max(per1, per2, 1e-6)
    return min(1.0, 0.5 * ar + 0.5 * pr)
