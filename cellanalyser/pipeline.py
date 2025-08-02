# ABCellPolicy/cellanalyser/pipeline.py

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import cellanalyser.config as config
from .aggregator import aggregate
from .stats import friedman_neff, parallel_permutation, posthoc_conover_friedman
from .visualization import (
    plot_frame_shape_metrics,
    plot_frame_motion_metrics,
    plot_track_dynamic_matrix,
    plot_friedman_scene_heatmaps,
    plot_overall_friedman_heatmaps
)
from scipy.stats import rankdata

def run_pipeline():
    # 1) Aggregate all metrics
    frames_df, tracks_df, scene_tracks_df = aggregate()
    out_dir = config.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # 2) Save raw tables
    frames_df.to_csv(os.path.join(out_dir, 'frame_metrics.csv'), index=False)
    tracks_df.to_csv(os.path.join(out_dir, 'track_metrics.csv'), index=False)
    scene_tracks_df.to_csv(os.path.join(out_dir, 'track_metrics_per_scene.csv'), index=False)

    # 3) Static & dynamic visualizations
    plot_frame_shape_metrics(frames_df, out_dir)
    plot_frame_motion_metrics(frames_df, out_dir)
    plot_track_dynamic_matrix(tracks_df, out_dir)

    # 4) Friedman + post-hoc Conover for MA & RAW
    if config.PERFORM_STATS and len(config.DB_PATHS) >= 3:
        scenes = list(config.DB_PATHS.keys())
        k = len(scenes)
        half = config.SMOOTH_WINDOW // 2
        alpha = config.ALPHA_TEST

        METRICS_LIST = [
            'area','eccentricity','orientation','circularity',
            'disp_um','speed_um_per_frame','acc_um_per_frame2'
        ]
        pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
        # only frames present in all scenes
        frames_by_scene = {
            sc: set(frames_df.loc[frames_df['scene']==sc, 'time_frame'])
            for sc in scenes
        }
        common_frames = sorted(set.intersection(*frames_by_scene.values()))

        perm_rows = []
        all_pair_rows = []
        total_iters = len(METRICS_LIST) * 2 * len(common_frames)
        pbar = tqdm(total=total_iters, desc='Friedman all', ncols=120)

        for metric in METRICS_LIST:
            # pivot to shape (frames × scenes)
            mean_df = (
                frames_df
                .pivot_table(index='time_frame', columns='scene',
                             values=metric, aggfunc='mean')
                .loc[common_frames]
            )
            # MA vs RAW
            ma_df  = mean_df.rolling(config.SMOOTH_WINDOW,
                                     center=True,
                                     min_periods=config.SMOOTH_WINDOW).mean()
            raw_df = mean_df.copy()

            for label, df_in in (('MA', ma_df), ('RAW', raw_df)):
                # initialize pairwise flags
                pair_sig = {p: np.zeros(len(common_frames), bool) for p in pairs}
                # sliding window blocks
                blocks = [
                    df_in.iloc[max(0, i-half):min(len(common_frames), i+half+1)].to_numpy()
                    for i in range(len(common_frames))
                ]

                for idx, block in enumerate(blocks):
                    # skip if too few rows or zero-variance
                    if block.shape[0] < 3 or np.any(np.nanstd(block, axis=0) == 0):
                        pbar.update(1)
                        continue

                    # classic Friedman + N_eff
                    Q, p_classic, neff, rho = friedman_neff(block)
                    # permutation only if classic significant
                    if p_classic < alpha:
                        _, p_perm = parallel_permutation(block, 1000, config.N_THREADS)
                    else:
                        p_perm = np.nan

                    perm_rows.append({
                        'metric': metric,
                        'window_type': label,
                        'time_frame': common_frames[idx],
                        'p_classic': p_classic,
                        'p_perm': p_perm,
                        'N_eff': neff,
                        'rho': rho
                    })

                    # post-hoc pairwise Conover
                    if p_classic < alpha:
                        ranks = np.vstack([rankdata(row) for row in block])
                        try:
                            pmat = posthoc_conover_friedman(ranks, p_adjust='holm')
                            for (i1, i2) in pairs:
                                pair_sig[(i1, i2)][idx] = pmat[i1, i2] < alpha
                        except ImportError:
                            # fallback to Tukey HSD
                            from statsmodels.stats.multicomp import pairwise_tukeyhsd
                            data = block.flatten(order='F')
                            labs = np.repeat(np.arange(k), block.shape[0])
                            tk = pairwise_tukeyhsd(data, labs, alpha=alpha)
                            for row in tk._results_table.data[1:]:
                                i1_, i2_, *_ , pval = row
                                i1_, i2_ = int(i1_), int(i2_)
                                pair_sig[(i1_, i2_)][idx] = pval < alpha

                    pbar.update(1)

                # record pairwise per frame
                for t, frame in enumerate(common_frames):
                    for (i1, i2) in pairs:
                        all_pair_rows.append({
                            'metric': metric,
                            'window_type': label,
                            'time_frame': frame,
                            'scene_i': scenes[i1],
                            'scene_j': scenes[i2],
                            'significant': bool(pair_sig[(i1, i2)][t])
                        })

        pbar.close()

        # save stats tables
        pd.DataFrame(perm_rows).to_csv(
            os.path.join(out_dir, 'friedman_results.csv'), index=False
        )
        pd.DataFrame(all_pair_rows).to_csv(
            os.path.join(out_dir, 'friedman_pairwise.csv'), index=False
        )

        # MA & RAW heatmaps
        plot_friedman_scene_heatmaps(all_pair_rows, scenes, common_frames, out_dir)
        plot_overall_friedman_heatmaps(all_pair_rows, scenes, common_frames, out_dir)

    print(f'Done. Results in {out_dir}')


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
