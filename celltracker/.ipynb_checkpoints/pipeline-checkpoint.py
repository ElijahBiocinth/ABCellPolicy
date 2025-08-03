import os
import json
import sqlite3
import cv2
import numpy as np
from pathlib import Path
from . import config
from .config import (
    DEFAULT_DB_PATH, DEFAULT_SRC_FOLDER, DEFAULT_MODEL_PATH,
    TILE_SIZE, OVERLAP, DEBUG_FIRST_N_FRAMES
)
from .db import init_db, insert_tracks, insert_metrics, insert_lineage, close_db
from .progress import start_timing, one_line_progress
from .tiling import tile_image, stitch_polygons, merge_overlapping
from .geometry import safe_polygon
from .features import poly_features
from .assignment import assign_tracks_two_stage, detect_splits_merges
from .reid import archive_track, increment_reid_age
from .track import Track
from .visualization import draw_tracks
from .detectors import YoloDetector, CellposeDetector

def run_pipeline(args):
    if args.jit:
        config._NUMBA = True

    db_path    = args.db.resolve() if args.db else Path(DEFAULT_DB_PATH).expanduser()
    src_folder = args.src.resolve()
    model_path = args.model.resolve()
    out_dir    = (args.out or (src_folder / "annotated")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = init_db(db_path)

    if args.backend.lower() == "cellpose":
        detector = CellposeDetector(str(model_path), args.device)
    else:
        detector = YoloDetector(str(model_path), args.device)

    tracks = []
    next_id = 0
    last_assignments = {}
    total_switches = 0

    img_paths = sorted(
        p for p in src_folder.iterdir()
        if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    )
    total = len(img_paths)

    start_timing(total)
    one_line_progress(0, total, extra="init")

    for frame_idx, img_path in enumerate(img_paths, start=1):
        img_name = img_path.name
        img = cv2.imread(str(img_path))
        
        if img is None:
            continue
            
        h, w = img.shape[:2]
        tiles = tile_image(img, TILE_SIZE, OVERLAP)
        boxes = [(x, y) for x, y, _ in tiles]
        polys_per_tile = []
        
        for (x, y, tile) in tiles:
            coords_list = detector.detect(tile)
            pts = []
            for coords in coords_list:
                p = safe_polygon(coords)
                if p is not None:
                    pts.append(p)
            polys_per_tile.append(pts)

        new_polys = stitch_polygons(polys_per_tile, boxes)
        new_polys = merge_overlapping(new_polys)
        new_feats = [poly_features(img, p) for p in new_polys]
        matches, unmatched, costs, median_cost, used_relax, continuity, rec_passes = (
            assign_tracks_two_stage(tracks, new_polys, new_feats, frame_idx - 1)
        )
        splits, merges = detect_splits_merges(tracks, new_polys, matches)
        lineage_rows = []
        for parent, children in splits:
            for child in children:
                lineage_rows.append((
                    frame_idx - 1,
                    tracks[parent].id,
                    next_id + child,
                    tracks[parent].generation,
                    tracks[parent].generation + 1,
                    'split'
                ))
                
        for child, parents in merges:
            for parent in parents:
                lineage_rows.append((
                    frame_idx - 1,
                    tracks[parent].id,
                    next_id + child,
                    tracks[parent].generation,
                    tracks[parent].generation + 1,
                    'merge'
                ))
                
        if lineage_rows:
            insert_lineage(conn, lineage_rows)
            
        current_assignments = {}
        
        for ti, pi, cost in matches:
            tr = tracks[ti]
            tr.update(new_polys[pi], new_feats[pi], matched_iou=1-cost)
            current_assignments[ti] = pi
            
        for pi in unmatched:
            tr = Track(next_id, new_polys[pi], new_feats[pi])
            tracks.append(tr)
            current_assignments[len(tracks)-1] = pi
            next_id += 1

        dead = [tr for tr in tracks if tr.missed > args.first_n or tr.missed > 5]
        
        for tr in dead:
            archive_track(tr)
        tracks = [tr for tr in tracks if tr.missed <= args.first_n and tr.missed <= 5]
        increment_reid_age()

        switches = sum(
            1 for tid, det in current_assignments.items()
            if last_assignments.get(tid) not in (None, det)
        )
        
        total_switches += switches
        last_assignments = current_assignments

        insert_metrics(conn, (
            frame_idx - 1,
            len(tracks),
            len(matches),
            float(np.mean([1-c for c in costs])) if costs else None,
            len(splits),
            len(merges),
            switches,
            median_cost,
            int(used_relax),
            continuity,
            rec_passes
        ))

        track_rows = []
        for tr in tracks:
            coords = [[float(x), float(y)] for x, y in tr.poly.exterior.coords]
            track_rows.append((
                img_name,
                frame_idx - 1,
                w,
                h,
                tr.id,
                tr.generation,
                json.dumps(coords)
            ))
        insert_tracks(conn, track_rows)

        if not args.no_vis:
            vis = draw_tracks(img.copy(), tracks)
            cv2.imwrite(str(out_dir / f"ann_{img_name}"), vis)

        one_line_progress(
            frame_idx, total,
            extra=f"{img_name} trk={len(tracks)} m={len(matches)} sw={switches}"
        )

    close_db(conn)
    one_line_progress(total, total, extra=f"done DB={db_path}")
    print(f"Total ID switches: {total_switches}")
