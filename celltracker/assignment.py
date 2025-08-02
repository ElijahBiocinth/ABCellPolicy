import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from shapely.ops import unary_union
from .config import (
    BASE_IOU_MATCH, SOFT_IOU_MIN, SOFT_IOU_MIN_RELAX,
    DIST_GATE_FACTOR, RELAX_DIST_MULT, MAX_COST_CAP,
    ADAPTIVE_COST_MULT, ADAPTIVE_COST_OFFSET, MIN_MATCH_RATIO,
    K_INIT, EARLY_ZERO_MATCH_LIMIT, CENTROID_FALLBACK_MAX_DIST_MULT,
    CONTINUITY_TARGET, MAX_RECOVERY_PASSES, RECOVERY_GATE_MULT,
    RECOVERY_FEATURE_THRESH, RECOVERY_SHAPE_THRESH,
    GREEDY_MAX_DIST_MULT, GREEDY_FEATURE_THRESH, GREEDY_SHAPE_THRESH,
    SPLIT_IOU_THRESH, MERGE_IOU_THRESH
)
from .geometry import compute_iou, centroid
from .features import feature_distance, shape_distance

def current_weights(frame_idx):
    if frame_idx < K_INIT:
        return dict(IOU=0.30, DIST=0.35, APP=0.20, SHAPE=0.15)
    return dict(IOU=IOU_WEIGHT, DIST=DIST_WEIGHT, APP=APPEARANCE_WEIGHT, SHAPE=SHAPE_WEIGHT)

def build_cost_matrix(tracks, new_polys, new_feats, frame_idx, relax=False):
    M = len(tracks)
    N = len(new_polys)
    C = np.full((M, N), np.inf, dtype=np.float32)
    raw_costs = []
    if not tracks or not new_polys:
        return C, raw_costs, None
    weights = current_weights(frame_idx)
    iou_w = weights['IOU']; dist_w = weights['DIST']; app_w = weights['APP']; shape_w = weights['SHAPE']
    soft_min_iou = SOFT_IOU_MIN_RELAX if relax else SOFT_IOU_MIN
    for i, tr in enumerate(tracks):
        p1 = tr.poly
        c_pred = tr.predict_centroid()
        soft_thr = min(tr.adaptive_iou, BASE_IOU_MATCH)
        for j, (p2, f2) in enumerate(zip(new_polys, new_feats)):
            iou = compute_iou(p1, p2)
            if iou < soft_min_iou:
                continue
            c2 = centroid(p2)
            base_gate = max(15.0, DIST_GATE_FACTOR * (math.sqrt(p1.area) + 0.3*math.hypot(*tr.vel)))
            gate = base_gate * (RELAX_DIST_MULT if relax else 1.0)
            char_len = math.sqrt(p1.area)
            gate = min(gate, 2.0 * char_len)
            dist = math.hypot(c_pred[0] - c2[0], c_pred[1] - c2[1])
            if dist > gate:
                continue
            dist_norm = min(1.0, dist / gate)
            feat_d = feature_distance(tr.feat_ema, f2)
            shape_d = shape_distance(p1, p2)
            iou_deficit = 0.0 if relax else max(0.0, soft_thr - iou)
            cost = (iou_w * (1 - iou)
                    + dist_w * dist_norm
                    + app_w * feat_d
                    + shape_w * shape_d
                    + (0.3 * iou_deficit))
            raw_costs.append(cost)
            C[i, j] = cost
    if raw_costs and not relax and frame_idx >= K_INIT:
        med = float(np.median(raw_costs))
        max_cost = min(MAX_COST_CAP, med * ADAPTIVE_COST_MULT + ADAPTIVE_COST_OFFSET)
        if max_cost < min(raw_costs):
            max_cost = min(MAX_COST_CAP, min(raw_costs) + 0.05)
        C[C > max_cost] = np.inf
    elif raw_costs and relax:
        C[C > MAX_COST_CAP] = np.inf
    return C, raw_costs, None

def solve_assignment(C):
    if not np.isfinite(C).any():
        return []
    row_mask = np.isfinite(C).any(axis=1)
    col_mask = np.isfinite(C).any(axis=0)
    reduced_rows = np.where(row_mask)[0]
    reduced_cols = np.where(col_mask)[0]
    C_reduced = C[row_mask][:, col_mask]
    valid_row_mask = np.isfinite(C_reduced).any(axis=1)
    valid_col_mask = np.isfinite(C_reduced).any(axis=0)
    if not valid_row_mask.any() or not valid_col_mask.any():
        return []
    if (not np.all(valid_row_mask)) or (not np.all(valid_col_mask)):
        C_reduced = C_reduced[valid_row_mask][:, valid_col_mask]
        reduced_rows = reduced_rows[valid_row_mask]
        reduced_cols = reduced_cols[valid_col_mask]
    try:
        r_idx, c_idx = linear_sum_assignment(C_reduced)
    except ValueError:
        return []
    out = []
    for rr, cc in zip(r_idx, c_idx):
        cost = C_reduced[rr, cc]
        if np.isfinite(cost):
            out.append((reduced_rows[rr], reduced_cols[cc], cost))
    return out

def centroid_fallback(tracks, new_polys):
    matches = []
    used_new = set()
    for i, tr in enumerate(tracks):
        c_pred = tr.predict_centroid()
        best = None; best_j = None
        gate = max(30.0, CENTROID_FALLBACK_MAX_DIST_MULT * math.sqrt(tr.poly.area))
        for j, p in enumerate(new_polys):
            if j in used_new:
                continue
            c2 = centroid(p)
            d = math.hypot(c_pred[0] - c2[0], c_pred[1] - c2[1])
            if d <= gate and (best is None or d < best):
                best = d; best_j = j
        if best_j is not None:
            matches.append((i, best_j, best / gate))
            used_new.add(best_j)
    return matches

def recovery_pass_expand(tracks, new_polys, new_feats, frame_idx,
                         unmatched_track_indices, unmatched_new_indices, pass_id):
    matches = []; used_new = set()
    if pass_id == 0:
        for ti in unmatched_track_indices:
            tr = tracks[ti]
            p1 = tr.poly; c_pred = tr.predict_centroid()
            base_gate = max(15.0, DIST_GATE_FACTOR * (math.sqrt(p1.area) + 0.3*math.hypot(*tr.vel)))
            gate = base_gate * RECOVERY_GATE_MULT
            best = None; best_j = None
            for j in unmatched_new_indices:
                p2 = new_polys[j]
                c2 = centroid(p2)
                dist = math.hypot(c_pred[0] - c2[0], c_pred[1] - c2[1])
                if dist > gate:
                    continue
                dist_norm = min(1.0, dist / gate)
                fd = feature_distance(tr.feat_ema, new_feats[j])
                if fd > RECOVERY_FEATURE_THRESH:
                    continue
                sd = shape_distance(p1, p2)
                if sd > RECOVERY_SHAPE_THRESH:
                    continue
                cost = 0.6*dist_norm + 0.25*fd + 0.15*sd
                if best is None or cost < best:
                    best = cost; best_j = j
            if best_j is not None:
                matches.append((ti, best_j, best)); used_new.add(best_j)
    else:
        ordering = sorted(unmatched_track_indices, key=lambda i: (tracks[i].missed, -tracks[i].age))
        for ti in ordering:
            tr = tracks[ti]
            p1 = tr.poly; c_pred = tr.predict_centroid()
            gate = max(30.0, GREEDY_MAX_DIST_MULT * math.sqrt(p1.area))
            best = None; best_j = None
            for j in unmatched_new_indices:
                if j in used_new:
                    continue
                p2 = new_polys[j]; c2 = centroid(p2)
                dist = math.hypot(c_pred[0] - c2[0], c_pred[1] - c2[1])
                if dist > gate:
                    continue
                dist_norm = min(1.0, dist / gate)
                fd = feature_distance(tr.feat_ema, new_feats[j])
                if fd > GREEDY_FEATURE_THRESH:
                    continue
                sd = shape_distance(p1, p2)
                if sd > GREEDY_SHAPE_THRESH:
                    continue
                cost = 0.55*dist_norm + 0.30*fd + 0.15*sd
                if best is None or cost < best:
                    best = cost; best_j = j
            if best_j is not None:
                matches.append((ti, best_j, best)); used_new.add(best_j)
    return matches

def merge_matches(primary, secondary):
    track_best = {}; det_used = set()
    for t, d, c in primary:
        track_best[t] = (d, c); det_used.add(d)
    for t, d, c in secondary:
        if d in det_used and t not in track_best:
            continue
        if t not in track_best:
            if d not in det_used:
                track_best[t] = (d, c); det_used.add(d)
        else:
            d0, c0 = track_best[t]
            if c < c0 and (d == d0 or d not in det_used):
                if d != d0:
                    det_used.discard(d0); det_used.add(d)
                track_best[t] = (d, c)
    return [(t, d, c) for t, (d, c) in track_best.items()], 0

def assign_tracks_two_stage(tracks, new_polys, new_feats, frame_idx):
    C_strict, _, _ = build_cost_matrix(tracks, new_polys, new_feats, frame_idx, relax=False)
    matches_strict = solve_assignment(C_strict)
    used_relax = False; matches = matches_strict
    if frame_idx < K_INIT and not matches:
        C_relax_early, _, _ = build_cost_matrix(tracks, new_polys, new_feats, frame_idx, relax=True)
        early_relax = solve_assignment(C_relax_early)
        if early_relax:
            matches = early_relax; used_relax = True
    if tracks and len(matches) < MIN_MATCH_RATIO * len(tracks):
        C_relax, _, _ = build_cost_matrix(tracks, new_polys, new_feats, frame_idx, relax=True)
        relax_matches = solve_assignment(C_relax)
        if len(relax_matches) > len(matches):
            matches = relax_matches; used_relax = True
    if tracks and not matches and frame_idx < (K_INIT + EARLY_ZERO_MATCH_LIMIT):
        cf = centroid_fallback(tracks, new_polys)
        if cf:
            matches = cf; used_relax = True
    prev = len(tracks); matched = {t for t, _, _ in matches}
    continuity = len(matched)/prev if tracks else 1.0
    recovery_passes = 0
    if continuity < CONTINUITY_TARGET and tracks:
        unmatched_t = [i for i in range(len(tracks)) if i not in matched]
        matched_new = {d for _, d, _ in matches}
        unmatched_n = [j for j in range(len(new_polys)) if j not in matched_new]
        for pid in range(MAX_RECOVERY_PASSES):
            if continuity >= CONTINUITY_TARGET:
                break
            rec = recovery_pass_expand(tracks, new_polys, new_feats, frame_idx, unmatched_t, unmatched_n, pid)
            if rec:
                merged, _ = merge_matches(matches, rec)
                matches = merged
                matched = {t for t, _, _ in matches}
                matched_new = {d for _, d, _ in matches}
                unmatched_t = [i for i in range(len(tracks)) if i not in matched]
                unmatched_n = [j for j in range(len(new_polys)) if j not in matched_new]
                continuity = len(matched)/prev if tracks else 1.0
                recovery_passes = pid + 1
    unmatched_new = [j for j in range(len(new_polys)) if j not in {d for _, d, _ in matches}]
    costs = [c for *_, c in matches]
    median_cost = float(np.median(costs)) if costs else None
    return matches, unmatched_new, costs, median_cost, used_relax, continuity, recovery_passes

def detect_splits_merges(tracks, new_polys, matches):
    old2new = defaultdict(list); new2old = defaultdict(list)
    for r, c, _ in matches:
        old2new[r].append(c); new2old[c].append(r)
    splits = []; merges = []
    for r, new_ids in old2new.items():
        if len(new_ids) > 1:
            try:
                union_new = unary_union([new_polys[i] for i in new_ids])
            except Exception:
                continue
            if compute_iou(tracks[r].poly, union_new) >= SPLIT_IOU_THRESH:
                splits.append((r, new_ids))
    for c, old_ids in new2old.items():
        if len(old_ids) > 1:
            try:
                union_old = unary_union([tracks[i].poly for i in old_ids])
            except Exception:
                continue
            if compute_iou(union_old, new_polys[c]) >= MERGE_IOU_THRESH:
                merges.append((c, old_ids))
    return splits, merges
