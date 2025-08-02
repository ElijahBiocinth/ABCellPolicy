import math
import numpy as np
from .config import (
    REID_MAX,
    REID_MAX_AGE_GAP,
    REID_POS_DIST,
    REID_SCORE_THRESH,
    REID_MIN_AREA_RATIO,
    REID_MAX_AREA_RATIO,
    PRINT_REID_MATCHES,
)
from .geometry import centroid
from .features import feature_distance

REID_BUFFER = []

def archive_track(tr):
    REID_BUFFER.append({
        "id": tr.id,
        "feat": tr.feat_ema.copy(),
        "last_area": tr.area_ema,
        "last_pos": tr.history[-1],
        "age_gap": 0
    })
    if len(REID_BUFFER) > REID_MAX:
        REID_BUFFER.pop(0)

def increment_reid_age():
    drop = []
    for rec in REID_BUFFER:
        rec["age_gap"] += 1
        if rec["age_gap"] > REID_MAX_AGE_GAP:
            drop.append(rec)
    for rec in drop:
        REID_BUFFER.remove(rec)

def reid_try(p, feat):
    c = centroid(p)
    feat = feat / (np.linalg.norm(feat) + 1e-9)
    best = None
    for rec in REID_BUFFER:
        dpos = math.hypot(c[0] - rec["last_pos"][0], c[1] - rec["last_pos"][1])
        if dpos > REID_POS_DIST:
            continue
        if not (REID_MIN_AREA_RATIO * rec["last_area"] <= p.area <= REID_MAX_AREA_RATIO * rec["last_area"]):
            continue
        fd = feature_distance(rec["feat"], feat)
        area_ratio = abs(p.area - rec["last_area"]) / max(p.area, rec["last_area"], 1e-9)
        score = fd + 0.5 * area_ratio + 0.002 * dpos
        if best is None or score < best[0]:
            best = (score, rec["id"], dpos, fd, area_ratio)
    if best and best[0] < REID_SCORE_THRESH:
        if PRINT_REID_MATCHES:
            print(f"[REID] id={best[1]} score={best[0]:.3f} dpos={best[2]:.1f} fd={best[3]:.3f} ar={best[4]:.3f}")
        return best[1]
    return None
