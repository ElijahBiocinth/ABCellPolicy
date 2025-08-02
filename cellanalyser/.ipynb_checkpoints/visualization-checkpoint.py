# ABCellPolicy/cellanalyser/visualization.py

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from itertools import product
from .stats import bootstrap_ci
import seaborn as sns

def _slugify(name: str) -> str:
    return re.sub(r'[^0-9A-Za-z]+', '_', name).strip('_')

PALETTE = plt.get_cmap('Accent')

def plot_frame_shape_metrics(frames_df: pd.DataFrame, out_dir: str):
    metrics = ['area', 'eccentricity', 'orientation', 'circularity']
    scenes = frames_df['scene'].unique()
    for metric in metrics:
        slug = _slugify(metric)

        # time series + moving average
        plt.figure(figsize=(10,6))
        for idx, sc in enumerate(scenes):
            color = PALETTE(idx / max(1, len(scenes)-1))
            sub = frames_df[frames_df['scene']==sc] \
                      .dropna(subset=[metric]).sort_values('time_frame')
            x, y = sub['time_frame'].values, sub[metric].values
            plt.scatter(x, y, s=10, alpha=0.2, color=color, edgecolors='none', label=sc)
            window = max(1, int(len(y)*0.03))
            ma = pd.Series(y).rolling(window, center=True, min_periods=1).mean().values
            plt.plot(x, ma, linewidth=1, color=color)
        plt.xlabel('Frame', fontsize=24)
        plt.ylabel(metric.title(), fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(title='Scene', fontsize=18, title_fontsize=18, markerscale=5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'frame_{slug}_timeseries.png'), dpi=300)
        plt.close()

        # violin + means
        plt.figure(figsize=(10,6))
        data = [frames_df.loc[frames_df['scene']==sc, metric].dropna().values for sc in scenes]
        parts = plt.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for idx, pc in enumerate(parts['bodies']):
            c = PALETTE(idx / max(1, len(scenes)-1))
            pc.set_facecolor(c)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        means = [np.mean(d) if len(d)>0 else np.nan for d in data]
        plt.scatter(np.arange(1,len(scenes)+1), means, marker='_', s=400,
                    linewidths=1, color='gray', zorder=3)
        plt.xticks(np.arange(1,len(scenes)+1), scenes, fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('Scene', fontsize=24)
        plt.ylabel(metric.title(), fontsize=24)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'frame_{slug}_violin.png'), dpi=300)
        plt.close()

def plot_frame_motion_metrics(frames_df: pd.DataFrame, out_dir: str):
    motion_metrics = [
        ('disp_um', 'Displacement (µm)'),
        ('speed_um_per_frame', 'Speed (µm/frame)'),
        ('acc_um_per_frame2', 'Acceleration (µm/frame²)')
    ]
    scenes = frames_df['scene'].unique()
    for col, label in motion_metrics:
        slug = _slugify(col)

        plt.figure(figsize=(10,6))
        for idx, sc in enumerate(scenes):
            color = PALETTE(idx / max(1, len(scenes)-1))
            sub = frames_df[frames_df['scene']==sc] \
                      .dropna(subset=[col]).sort_values('time_frame')
            x, y = sub['time_frame'].values, sub[col].values
            plt.scatter(x, y, s=10, alpha=0.2, color=color, edgecolors='none', label=sc)
            window = max(1, int(len(y)*0.03))
            ma = pd.Series(y).rolling(window, center=True, min_periods=1).mean().values
            plt.plot(x, ma, linewidth=1, color=color)
        plt.xlabel('Frame', fontsize=24)
        plt.ylabel(label, fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(title='Scene', fontsize=18, title_fontsize=18, markerscale=5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'frame_{slug}_timeseries.png'), dpi=300)
        plt.close()

        plt.figure(figsize=(10,6))
        data = [frames_df.loc[frames_df['scene']==sc, col].dropna().values for sc in scenes]
        parts = plt.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for idx, pc in enumerate(parts['bodies']):
            c = PALETTE(idx / max(1, len(scenes)-1))
            pc.set_facecolor(c)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        means = [np.mean(d) if len(d)>0 else np.nan for d in data]
        plt.scatter(np.arange(1,len(scenes)+1), means, marker='_', s=400,
                    linewidths=1, color='gray', zorder=3)
        plt.xticks(np.arange(1,len(scenes)+1), scenes, fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('Scene', fontsize=24)
        plt.ylabel(label, fontsize=24)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'frame_{slug}_violin.png'), dpi=300)
        plt.close()

def plot_track_dynamic_matrix(tracks_df: pd.DataFrame, out_dir: str):
    from .config import DYN_TRACK_COLS
    n_metrics = len(DYN_TRACK_COLS)
    n_cols = 4
    n_rows = math.ceil(n_metrics / n_cols)
    scenes = tracks_df['scene'].unique()
    cmap = plt.get_cmap('Accent')

    plt.figure(figsize=(4*n_cols, 4*n_rows))
    for i, metric in enumerate(DYN_TRACK_COLS, start=1):
        ax = plt.subplot(n_rows, n_cols, i)
        data = [tracks_df.loc[tracks_df['scene']==sc, metric].dropna().values
                for sc in scenes]

        parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for idx, pc in enumerate(parts['bodies']):
            color = cmap(idx / max(1, len(scenes)-1))
            pc.set_facecolor(color)
            pc.set_edgecolor('gray')
            pc.set_alpha(0.3)

        means = [np.mean(vals) if len(vals)>0 else np.nan for vals in data]
        for xi, m in enumerate(means, start=1):
            ax.hlines(m, xi-0.3, xi+0.3, colors='gray', linewidth=1)
            ax.text(xi, m, f"{m:.2f}", ha='center', va='center',
                    fontsize=10,
                    bbox=dict(facecolor='gray', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))

        flat = np.concatenate(data) if data else np.array([])
        if len(flat) > 0:
            lower, upper, mean_bs = bootstrap_ci(flat, num_iterations=10000, alpha=0.05)
            ax.set_title(
                f"{metric}\nμ={mean_bs:.2f}  CI=[{lower:.2f},{upper:.2f}]",
                fontsize=12, pad=10
            )
        else:
            ax.set_title(metric, fontsize=10, pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'track_dynamic_matrix.png'), dpi=300)
    plt.close()

def plot_friedman_scene_heatmaps(pair_rows: list, scenes: list, frames: list, out_dir: str):
    """
    For each metric, two vertical heatmaps (MA then RAW), scenes on Y, frames on X,
    color = count of significant Conover comparisons per cell.
    """
    df = pd.DataFrame(pair_rows)
    metrics = df['metric'].unique()
    fig, axes = plt.subplots(
        nrows=2*len(metrics), ncols=1,
        figsize=(max(10,len(frames)*0.04), 2*len(metrics)*1.5),
        sharex=True
    )
    for mi, metric in enumerate(metrics):
        for wi, window in enumerate(['MA','RAW']):
            ax = axes[2*mi + wi]
            sub = df[(df['metric']==metric)&(df['window_type']==window)]
            counts = pd.DataFrame(0, index=scenes, columns=frames)
            for _, row in sub.iterrows():
                t,row_i,row_j = row['time_frame'], row['scene_i'], row['scene_j']
                counts.at[row_i, t] += 1
                counts.at[row_j, t] += 1
            sns.heatmap(counts, ax=ax, cmap='magma',
                        cbar=(wi==0), vmin=0, vmax=counts.values.max(),
                        xticklabels=max(1,int(len(frames)/10)), yticklabels=scenes)
            ax.set_ylabel('Scene', fontsize=10)
            ax.set_title(f'{metric} ({window})', fontsize=12)
            if wi==1:
                ax.set_xlabel('Frame', fontsize=10)
            else:
                ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'friedman_scene_heatmaps.png'), dpi=500)
    plt.close()

def plot_overall_friedman_heatmaps(pair_rows: list, scenes: list, frames: list, out_dir: str):
    """
    Two heatmaps (MA, RAW), scenes on Y, frames on X,
    color = number of metrics with Conover-p<α in that cell.
    """
    df = pd.DataFrame(pair_rows)
    fig, axes = plt.subplots(
        nrows=2, ncols=1,
        figsize=(max(10,len(frames)*0.04), 2*len(scenes)*0.3),
        sharex=True
    )
    for wi, window in enumerate(['MA','RAW']):
        ax = axes[wi]
        sub = df[df['window_type']==window]
        counts = pd.DataFrame(0, index=scenes, columns=frames)
        for _, row in sub.iterrows():
            t,row_i,row_j = row['time_frame'], row['scene_i'], row['scene_j']
            counts.at[row_i, t] += 1
            counts.at[row_j, t] += 1
        sns.heatmap(counts, ax=ax, cmap='magma', cbar=True,
                    vmin=0, vmax=counts.values.max(),
                    xticklabels=max(1,int(len(frames)/10)), yticklabels=scenes)
        ax.set_ylabel('Scene', fontsize=10)
        ax.set_title(f'All metrics ({window})', fontsize=12)
        ax.set_xlabel('Frame' if wi==1 else '')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'overall_friedman_heatmaps.png'), dpi=300)
    plt.close()
