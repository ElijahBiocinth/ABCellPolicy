import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import rankdata
import cellanalyser.config as config
from .aggregator import aggregate
from .stats import friedman_neff, parallel_permutation, conover_posthoc
from .visualization import (
    plot_frame_shape_metrics,
    plot_frame_motion_metrics,
    plot_track_dynamic_matrix,
    plot_friedman_all_metrics_heatmaps,
    plot_overall_friedman_heatmaps
)

def run_pipeline():
    frames_df, tracks_df, scene_tracks_df = aggregate()
    out = config.OUT_DIR
    os.makedirs(out, exist_ok=True)
    frames_df.to_csv(os.path.join(out, 'frame_metrics.csv'), index=False)
    tracks_df.to_csv(os.path.join(out, 'track_metrics.csv'), index=False)
    scene_tracks_df.to_csv(os.path.join(out, 'track_metrics_per_scene.csv'), index=False)
    plot_frame_shape_metrics(frames_df, out)
    plot_frame_motion_metrics(frames_df, out)
    plot_track_dynamic_matrix(tracks_df, out)
    
    if config.PERFORM_STATS and len(config.DB_PATHS) >= 3:
        scenes = list(config.DB_PATHS.keys())
        half = config.SMOOTH_WINDOW // 2
        α = config.ALPHA_TEST
        METRICS = [
            'area','eccentricity','orientation','circularity',
            'disp_um','speed_um_per_frame','acc_um_per_frame2'
        ]
        frames_sets = [set(frames_df.loc[frames_df['scene']==s,'time_frame']) for s in scenes]
        common = sorted(set.intersection(*frames_sets))
        pair_rows = []
        
        for metric in METRICS:
            mean_df = (
                frames_df
                .pivot_table(index='time_frame', columns='scene', values=metric, aggfunc='mean')
                .loc[common]
            )
            ma_df  = mean_df.rolling(config.SMOOTH_WINDOW, center=True, min_periods=config.SMOOTH_WINDOW).mean()
            raw_df = mean_df.copy()

            for label, df_in in (('MA', ma_df), ('RAW', raw_df)):
                k = len(scenes)
                n = len(common)
                pairs = [(i,j) for i in range(k) for j in range(i+1,k)]
                pair_sig = {p: np.zeros(n, bool) for p in pairs}
                blocks = [
                    df_in.iloc[max(0,t-half):t+half+1].to_numpy()
                    for t in range(n)
                ]
                pbar = tqdm(total=n, desc=f'Friedman {metric} {label}', ncols=120)
                
                for t, blk in enumerate(blocks):
                    if blk.shape[0] < 3 or np.any(np.nanstd(blk,axis=0)==0):
                        pbar.update(1)
                        continue

                    Q, p_val, neff, rho = friedman_neff(blk)
                    
                    if p_val < α:
                        _, _ = parallel_permutation(blk, 1000, config.N_THREADS)
                        ranks = np.vstack([rankdata(r) for r in blk])
                        pm = conover_posthoc(ranks)
                        for (i,j) in pairs:
                            pair_sig[(i,j)][t] = (pm[i,j] < α)
                            
                    pbar.update(1)
                pbar.close()

                for (i,j), arr in pair_sig.items():
                    for t, sig in enumerate(arr):
                        pair_rows.append({
                            'metric': metric,
                            'window_type': label,
                            'time_frame': common[t],
                            'scene_i': scenes[i],
                            'scene_j': scenes[j],
                            'significant': bool(sig)
                        })

        plot_friedman_all_metrics_heatmaps(pair_rows, scenes, common, out)
        plot_overall_friedman_heatmaps(pair_rows, scenes, common, out)

    print(f'Done! Results in {config.OUT_DIR}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_paths',      type=json.loads)
    parser.add_argument('--out_dir')
    parser.add_argument('--px_to_um',      type=float)
    parser.add_argument('--frame_dt',      type=float)
    parser.add_argument('--smooth_window', type=int)
    parser.add_argument('--alpha_test',    type=float)
    parser.add_argument('--perform_stats', type=json.loads)
    parser.add_argument('--n_threads',     type=int)
    args = parser.parse_args()

    config.DB_PATHS.clear()
    config.DB_PATHS.update(args.db_paths)
    config.OUT_DIR        = args.out_dir
    config.PX_TO_UM       = args.px_to_um
    config.FRAME_DT       = args.frame_dt
    config.SMOOTH_WINDOW  = args.smooth_window
    config.ALPHA_TEST     = args.alpha_test
    config.PERFORM_STATS  = args.perform_stats
    config.N_THREADS      = args.n_threads

    run_pipeline()
