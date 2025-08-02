import json
import math
import numpy as np
import pandas as pd
import cv2
from shapely.geometry import Polygon, MultiPolygon

from .config import PX_TO_UM, METRICS_LIST, TURN_ANGLE_THRESH_DEG, ARREST_SPEED_THRESH

def _fast_ellipse(coords: np.ndarray):
    if coords is None or len(coords) < 5:
        return None
    pts = coords.astype(np.float32)
    try:
        (_, _), (w, h), angle = cv2.fitEllipse(pts)
        return ((None, None), (w, h), angle, np.sqrt(1 - (min(w, h) / max(w, h))**2))
    except Exception:
        return None

def _wrap_angles(rad_arr: np.ndarray) -> np.ndarray:
    return (rad_arr + np.pi) % (2 * np.pi) - np.pi

def compute_metrics_for_polygon(pts_list):
    px_to_um = PX_TO_UM
    empty = dict.fromkeys(METRICS_LIST, np.nan)
    if pts_list is None or len(pts_list) == 0:
        return empty
    try:
        poly = Polygon(pts_list)
    except Exception:
        return empty
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return empty
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda p: p.area)

    area = poly.area * (px_to_um ** 2)
    peri = poly.length * px_to_um

    coords = np.array(poly.exterior.coords)
    ell = _fast_ellipse(coords)
    ecc = ell[3] if ell else np.nan
    ori = ell[2] if ell else np.nan
    circ = (4 * math.pi * area) / (peri ** 2) if peri > 0 else np.nan

    return {
        'area': area,
        'eccentricity': ecc,
        'orientation': ori,
        'circularity': circ
    }

def lag1_acf(x: np.ndarray) -> float:
    if x.size < 3 or np.allclose(x, x[0]):
        return 0.0
    x1, x2 = x[:-1], x[1:]
    return np.corrcoef(x1, x2)[0,1] if np.std(x1) and np.std(x2) else 0.0

def compute_track_dynamic_metrics(track: pd.DataFrame) -> dict:
    x_um = track['cx'].values * PX_TO_UM
    y_um = track['cy'].values * PX_TO_UM

    v_x = np.diff(x_um)
    v_y = np.diff(y_um)
    speeds = np.hypot(v_x, v_y)
    acc = np.diff(speeds)

    path_len = speeds.sum()
    net_disp = np.hypot(x_um[-1] - x_um[0], y_um[-1] - y_um[0])

    msd = ((x_um - x_um[0])**2 + (y_um - y_um[0])**2).mean()
    mi = net_disp / path_len if path_len > 0 else np.nan
    dr = np.degrees(np.arctan2(y_um[-1] - y_um[0], x_um[-1] - x_um[0]))

    angles = np.arctan2(v_y, v_x)
    turn_angles = _wrap_angles(np.diff(angles)) if len(angles) > 1 else np.array([])
    mta = np.degrees(np.mean(np.abs(turn_angles))) if turn_angles.size else np.nan
    dp  = np.mean(np.cos(turn_angles)) if turn_angles.size else np.nan

    x_cm, y_cm = x_um.mean(), y_um.mean()
    rg = np.sqrt(((x_um - x_cm)**2 + (y_um - y_cm)**2).mean())

    arrest = np.mean(speeds < ARREST_SPEED_THRESH) if speeds.size else np.nan

    if 'area' in track.columns and track['area'].notna().sum() > 1 and speeds.size == (track.shape[0] - 1):
        try:
            smc = np.corrcoef(speeds, track['area'].values[1:])[0,1]
        except Exception:
            smc = np.nan
    else:
        smc = np.nan

    rmc = (np.mean(np.abs(np.diff(speeds))) / speeds.mean()) if speeds.size > 1 and speeds.mean() > 0 else np.nan

    vcc = np.corrcoef(v_x, v_y)[0,1] if v_x.size > 1 else np.nan

    ms = speeds.mean() if speeds.size else np.nan
    ma = np.mean(np.abs(acc)) if acc.size else np.nan

    return {
        'MSD': msd,
        'Directional Persistence': dp,
        'Meandering Index': mi,
        'Mean Turning Angle (deg)': mta,
        'Radius of Gyration': rg,
        'Arrest Coefficient': arrest,
        'Shape-Motion Coupling': smc,
        'Relative Motion Change': rmc,
        'Velocity Cross-Correlation': vcc,
        'Directionality Ratio (deg)': dr,
        'Mean Speed (µm/frame)': ms,
        'Mean Acceleration (µm/frame²)': ma
    }
