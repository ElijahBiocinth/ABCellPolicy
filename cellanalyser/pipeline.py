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
    plot_friedman_scene_heatmap
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
        k = len(scenes)
        half = config.SMOOTH_WINDOW // 2
        alpha = config.ALPHA_TEST

        METRICS = [
            'area','eccentricity','orientation','circularity',
            'disp_um','speed_um_per_frame','acc_um_per_frame2'
        ]

        for metric in METRICS:
            mean_df = (frames_df
                       .pivot_table(index='time_frame', columns='scene',
                                    values=metric, aggfunc='mean')
                       .loc[:, scenes])
            ma_df  = mean_df.rolling(config.SMOOTH_WINDOW,
                                     center=True,
                                     min_periods=config.SMOOTH_WINDOW).mean()
            raw_df = mean_df.copy()

            for window_label, df_in in (('MA', ma_df), ('RAW', raw_df)):
                common_frames = df_in.index.to_list()
                pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
                pair_sig = {p: np.zeros(len(common_frames), bool) for p in pairs}

                blocks = [df_in.iloc[max(0, t-half):t+half+1].to_numpy()
                          for t in range(len(common_frames))]
                pbar = tqdm(total=len(blocks),
                            desc=f'Friedman {metric} {window_label}', ncols=120)
                for t, block in enumerate(blocks):
                    if block.shape[0] < 3 or np.any(np.nanstd(block, axis=0)==0):
                        pbar.update(1)
                        continue

                    Q, p_cl, neff, rho = friedman_neff(block)
                    if p_cl < alpha:
                        _, p_perm = parallel_permutation(block, 1000, config.N_THREADS)
                        # пост-hoc на рангах
                        ranks = np.vstack([rankdata(row) for row in block])
                        pm = conover_posthoc(ranks, alpha)
                        for (i, j) in pairs:
                            pair_sig[(i, j)][t] = (pm[i, j] < alpha)
                    pbar.update(1)
                pbar.close()

                plot_friedman_scene_heatmap(pair_sig, scenes, common_frames,
                                            metric, window_label, out)

    print(f'Готово! Результаты в {out}')


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
