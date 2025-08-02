import sqlite3
import json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from .database import read_tracks
from .config import DB_PATHS, PX_TO_UM, FRAME_DT, N_THREADS, METRICS_LIST, DYN_TRACK_COLS
from .features import compute_metrics_for_polygon, compute_track_dynamic_metrics

def aggregate():
    total_polys = 0
    for path in DB_PATHS.values():
        with sqlite3.connect(path) as conn:
            total_polys += pd.read_sql_query('SELECT COUNT(*) AS n FROM tracks', conn)['n'][0]
    pbar = tqdm(total=total_polys, desc='Polygons', ncols=120)

    frame_dfs = []
    track_dfs = []

    for scene_label, db_path in DB_PATHS.items():
        df = read_tracks(db_path)

        df['poly_pts'] = df['polygon_points'].apply(
            lambda js: np.array(json.loads(js), int) if js else None
        )
        df['centroid'] = df['poly_pts'].apply(
            lambda arr: arr.mean(axis=0) if arr is not None and len(arr)>0 else np.array([np.nan, np.nan])
        )
        df[['cx','cy']] = np.vstack(df['centroid'].values)

        df[['dx_px','dy_px']] = df.groupby('track_id')[['cx','cy']].diff().fillna(0)
        df['disp_px'] = np.hypot(df['dx_px'], df['dy_px'])
        df['disp_um'] = df['disp_px'] * PX_TO_UM
        df['speed_um_per_frame'] = df['disp_um'] / FRAME_DT
        df['acc_um_per_frame2'] = df.groupby('track_id')['speed_um_per_frame'].diff().fillna(0)

        results = []
        with ThreadPoolExecutor(max_workers=N_THREADS) as exe:
            for res in exe.map(compute_metrics_for_polygon, df['poly_pts']):
                results.append(res)
                pbar.update(1)
        metrics_df = pd.DataFrame(results, columns=METRICS_LIST)

        df_raw = pd.concat([
            df[['time_frame','track_id','cx','cy','disp_um','speed_um_per_frame','acc_um_per_frame2']].reset_index(drop=True),
            metrics_df
        ], axis=1)
        df_raw['scene'] = scene_label

        frame_cols = METRICS_LIST + ['disp_um','speed_um_per_frame','acc_um_per_frame2']
        df_frame = (
            df_raw
            .groupby(['scene','time_frame'], sort=False)[frame_cols]
            .mean()
            .reset_index()
        )
        frame_dfs.append(df_frame)

        dyn_rows = []
        for tid, sub in df_raw.groupby('track_id', sort=False):
            d = compute_track_dynamic_metrics(sub.sort_values('time_frame'))
            d['track_id'] = tid
            dyn_rows.append(d)
        df_dyn = pd.DataFrame(dyn_rows)

        df_meta = (
            df_raw
            .groupby('track_id', sort=False)['time_frame']
            .agg(start_frame='min', end_frame='max')
            .reset_index()
        )

        df_track = df_meta.merge(df_dyn, on='track_id', how='left')
        df_track['scene'] = scene_label
        track_dfs.append(df_track)

    pbar.close()

    all_frames_df = pd.concat(frame_dfs, ignore_index=True)
    all_tracks_df = pd.concat(track_dfs, ignore_index=True)
    scene_track_df = (
        all_tracks_df
        .groupby('scene', sort=False)[DYN_TRACK_COLS]
        .mean()
        .reset_index()
    )
    return all_frames_df, all_tracks_df, scene_track_df
