# cellanalyser/pipeline.py

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
    plot_friedman_two_heatmaps,
    plot_overall_friedman_heatmaps
)

def run_pipeline():
    # 1) Собираем данные из всех баз
    frames_df, tracks_df, scene_tracks_df = aggregate()

    # 2) Базовые графики
    os.makedirs(config.OUT_DIR, exist_ok=True)
    plot_frame_shape_metrics(frames_df, config.OUT_DIR)
    plot_frame_motion_metrics(frames_df, config.OUT_DIR)
    plot_track_dynamic_matrix(tracks_df, config.OUT_DIR)

    # 3) Статистика Friedman + визуализация
    if config.PERFORM_STATS and len(config.DB_PATHS) >= 3:
        METRICS = config.METRICS_LIST
        scenes = sorted(frames_df['scene'].unique())
        frames = None
        half = config.SMOOTH_WINDOW // 2
        all_pair_rows = []

        for metric in METRICS:
            # матрица scene × time_frame
            mean_df = (
                frames_df
                .pivot_table(index='time_frame',
                             columns='scene',
                             values=metric,
                             aggfunc='mean')
                .loc[:, scenes]
            )
            if frames is None:
                frames = mean_df.index.to_list()

            ma_df  = mean_df.rolling(config.SMOOTH_WINDOW,
                                     center=True,
                                     min_periods=config.SMOOTH_WINDOW).mean()
            raw_df = mean_df.copy()
            pairs  = [(i, j) for i in range(len(scenes)) for j in range(i+1, len(scenes))]

            def compute_pair_sig(df_window):
                # возвращает dict (i,j)→bool array по всем фреймам
                pair_sig = {p: np.zeros(len(frames), bool) for p in pairs}
                blocks   = [
                    df_window.iloc[max(0, t-half):t+half+1].to_numpy()
                    for t in range(len(frames))
                ]
                pbar = tqdm(total=len(blocks),
                            desc=f'Friedman {metric}',
                            ncols=120)
                for t, block in enumerate(blocks):
                    pbar.update(1)
                    if block.shape[0] < 3 or np.any(np.nanstd(block, axis=0) == 0):
                        continue

                    # классический Friedman с N_eff
                    Q, p_cl, neff, rho = friedman_neff(block)
                    if p_cl < config.ALPHA_TEST:
                        # пермутационный тест
                        _, p_perm = parallel_permutation(
                            block,
                            n_perm=1000,
                            n_jobs=min(4, config.N_THREADS)
                        )
                        # Conover post-hoc: теперь с alpha
                        ranks = np.vstack([rankdata(row) for row in block])
                        pmat  = conover_posthoc(ranks, config.ALPHA_TEST)
                        for (i, j) in pairs:
                            pair_sig[(i, j)][t] = (pmat[i, j] < config.ALPHA_TEST)
                pbar.close()
                return pair_sig

            pair_sig_ma  = compute_pair_sig(ma_df)
            pair_sig_raw = compute_pair_sig(raw_df)

            # собираем строки для overall heatmap
            for label, sigdict in [('MA', pair_sig_ma), ('RAW', pair_sig_raw)]:
                for (i, j), arr in sigdict.items():
                    for t, sig in enumerate(arr):
                        all_pair_rows.append({
                            'window_type': label,
                            'time_frame':  frames[t],
                            'scene_i':     scenes[i],
                            'scene_j':     scenes[j],
                            'significant': bool(sig)
                        })

            # рисуем «двухэтажные» тепловые карты для этой метрики
            plot_friedman_two_heatmaps(
                pair_sig_ma, pair_sig_raw,
                scenes, frames,
                metric, config.OUT_DIR
            )

        # и финальный общий тепломап по всем метрикам
        plot_overall_friedman_heatmaps(
            all_pair_rows,
            scenes, frames,
            config.OUT_DIR
        )

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--db_paths',      type=json.loads, required=True)
    p.add_argument('--out_dir',       required=True)
    p.add_argument('--px_to_um',      type=float, default=config.PX_TO_UM)
    p.add_argument('--frame_dt',      type=float, default=config.FRAME_DT)
    p.add_argument('--smooth_window', type=int,   default=config.SMOOTH_WINDOW)
    p.add_argument('--alpha_test',    type=float, default=config.ALPHA_TEST)
    p.add_argument('--perform_stats', type=json.loads, default=json.dumps(config.PERFORM_STATS))
    p.add_argument('--n_threads',     type=int,   default=config.N_THREADS)
    args = p.parse_args()

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
