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
    total = 0
    for path in DB_PATHS.values():
        with sqlite3.connect(path) as c:
            total += pd.read_sql_query('SELECT COUNT(*) AS n FROM tracks', c)['n'][0]
    pbar = tqdm(total=total, desc='Polygons', ncols=120)

    frame_dfs, track_dfs = [], []

    for scene, db in DB_PATHS.items():
        df = read_tracks(db)
        df['poly_pts'] = df['polygon_points'].apply(lambda js: np.array(json.loads(js), int) if js else None)
        df['centroid'] = df['poly_pts'].apply(lambda a: a.mean(axis=0) if a is not None and len(a)>0 else np.array([np.nan, np.nan]))
        df[['cx','cy']] = np.vstack(df['centroid'].values)
        df[['dx','dy']] = df.groupby('track_id')[['cx','cy']].diff().fillna(0)
        df['disp_um'] = np.hypot(df['dx'], df['dy']) * PX_TO_UM
        df['speed_um_per_frame'] = df['disp_um'] / FRAME_DT
        df['acc_um_per_frame2'] = df.groupby('track_id')['speed_um_per_frame'].diff().fillna(0)

        with ThreadPoolExecutor(max_workers=N_THREADS) as exe:
            metrics = list(exe.map(compute_metrics_for_polygon, df['poly_pts']))
            for _ in metrics: pbar.update(1)
                
        met_df = pd.DataFrame(metrics, columns=METRICS_LIST)
        df_raw = pd.concat([df[['time_frame','track_id','cx','cy','disp_um','speed_um_per_frame','acc_um_per_frame2']], met_df], axis=1)
        df_raw['scene'] = scene
        frame_df = df_raw.groupby(['scene','time_frame'], sort=False)[METRICS_LIST + ['disp_um','speed_um_per_frame','acc_um_per_frame2']].mean().reset_index()
        frame_dfs.append(frame_df)
        dyn = []
        
        for tid, sub in df_raw.groupby('track_id', sort=False):
            d = compute_track_dynamic_metrics(sub.sort_values('time_frame'))
            d['track_id'] = tid
            dyn.append(d)
            
        track_df = pd.DataFrame(dyn)
        start_end = df_raw.groupby('track_id')['time_frame'].agg(start_frame='min', end_frame='max').reset_index()
        track_df = start_end.merge(track_df, on='track_id')
        track_df['scene'] = scene
        track_dfs.append(track_df)

    pbar.close()

    all_frames = pd.concat(frame_dfs, ignore_index=True)
    all_tracks = pd.concat(track_dfs, ignore_index=True)
    scene_tracks = all_tracks.groupby('scene', sort=False)[DYN_TRACK_COLS].mean().reset_index()

    return all_frames, all_tracks, scene_tracks
