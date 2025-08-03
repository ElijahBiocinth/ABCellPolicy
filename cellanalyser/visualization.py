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

        plt.figure(figsize=(10,6))
        for idx, sc in enumerate(scenes):
            color = PALETTE(idx / max(1, len(scenes)-1))
            sub = frames_df[frames_df['scene']==sc]\
                    .dropna(subset=[metric])\
                    .sort_values('time_frame')
            x_vals = sub['time_frame'].values
            y_vals = sub[metric].values
            plt.scatter(x_vals, y_vals,
                        s=10, alpha=0.2,
                        color=color, edgecolors='none',
                        label=sc)
            window = max(1, int(len(y_vals)*0.03))
            ma = pd.Series(y_vals).rolling(window, min_periods=1, center=True).mean().values
            plt.plot(x_vals, ma, linewidth=1, color=color, alpha=1.0)
            
        plt.xlabel('Frame', fontsize=24)
        plt.ylabel(metric.title(), fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(title='Scene', fontsize=18, title_fontsize=18, markerscale=5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'frame_{slug}_timeseries.png'), dpi=300)
        plt.close()

        plt.figure(figsize=(10,6))
        data = [frames_df.loc[frames_df['scene']==sc, metric].dropna().values for sc in scenes]
        parts = plt.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        
        for idx, pc in enumerate(parts['bodies']):
            c = PALETTE(idx / max(1, len(scenes)-1))
            pc.set_facecolor(c)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
            
        means = [np.mean(d) if len(d)>0 else np.nan for d in data]
        plt.scatter(np.arange(1, len(scenes)+1), means,
                    marker='_', s=400, linewidths=1,
                    color='gray', zorder=3)
        plt.xticks(np.arange(1, len(scenes)+1), scenes, fontsize=18, rotation=0)
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
            sub = frames_df[frames_df['scene']==sc]\
                    .dropna(subset=[col])\
                    .sort_values('time_frame')
            x_vals = sub['time_frame'].values
            y_vals = sub[col].values
            plt.scatter(x_vals, y_vals,
                        s=10, alpha=0.2,
                        color=color, edgecolors='none',
                        label=sc)
            window = max(1, int(len(y_vals)*0.03))
            ma = pd.Series(y_vals).rolling(window, min_periods=1, center=True).mean().values
            plt.plot(x_vals, ma, linewidth=1, color=color, alpha=1.0)
            
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
        plt.scatter(np.arange(1, len(scenes)+1), means,
                    marker='_', s=400, linewidths=1,
                    color='gray', zorder=3)
        plt.xticks(np.arange(1, len(scenes)+1), scenes, fontsize=18, rotation=0)
        plt.yticks(fontsize=18)
        plt.xlabel('Scene', fontsize=24)
        plt.ylabel(label, fontsize=24)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'frame_{slug}_violin.png'), dpi=300)
        plt.close()

def plot_track_dynamic_matrix(tracks_df: pd.DataFrame, out_dir: str):
    from .config import DYN_TRACK_COLS
    from .stats import bootstrap_ci
    n_metrics = len(DYN_TRACK_COLS)
    n_cols = 4
    n_rows = math.ceil(n_metrics / n_cols)
    scenes = tracks_df['scene'].unique()
    cmap = plt.get_cmap('Accent')
    plt.figure(figsize=(4*n_cols, 4*n_rows))
    for i, metric in enumerate(DYN_TRACK_COLS, start=1):
        ax = plt.subplot(n_rows, n_cols, i)
        data = [
            tracks_df.loc[tracks_df['scene']==sc, metric]
                     .dropna().values
            for sc in scenes
        ]
        parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for idx, pc in enumerate(parts['bodies']):
            color = cmap(idx / max(1, len(scenes)-1))
            pc.set_facecolor(color)
            pc.set_edgecolor('gray')
            pc.set_alpha(0.3)
            
        means = [np.mean(vals) if len(vals)>0 else np.nan for vals in data]
        for xi, m in enumerate(means, start=1):
            ax.hlines(m, xi-0.3, xi+0.3, colors='gray', linewidth=1)
            ax.text(
                xi, m,
                f"{m:.2f}",
                ha='center', va='center',
                fontsize=10,
                bbox=dict(facecolor='gray', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3')
            )

        flat = np.concatenate(data) if data else np.array([])
        if len(flat) > 0:
            lower, upper, mean_bs = bootstrap_ci(flat, num_iterations=10000, alpha=0.05)
            ax.set_title(
                f"{metric}\nμ={mean_bs:.2f}  CI=[{lower:.2f},{upper:.2f}]",
                fontsize=12, pad=10
            )
        else:
            ax.set_title(metric, fontsize=10, pad=10)

        ax.set_xticks(np.arange(1, len(scenes)+1))
        ax.set_xticklabels(scenes, rotation=45, fontsize=8)
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'track_dynamic_matrix.png'), dpi=300)
    plt.close()

def plot_friedman_all_metrics_heatmaps(pair_rows: list,
                                       scenes: list,
                                       frames: list,
                                       out_dir: str):
    df = pd.DataFrame(pair_rows)
    metrics = df['metric'].unique()
    windows = ['MA', 'RAW']
    n_metrics = len(metrics)
    n_frames = len(frames)

    fig, axes = plt.subplots(n_metrics * 2, 1,
                             figsize=(max(10, n_frames * 0.04),
                                      n_metrics * 2),
                             sharex=True)

    for mi, metric in enumerate(metrics):
        for wi, window in enumerate(windows):
            ax = axes[mi*2 + wi] if n_metrics > 1 else axes[wi]
            sub = df[(df['metric'] == metric) & (df['window_type'] == window)]
            counts = np.zeros((len(scenes), n_frames), int)
            for _, row in sub.iterrows():
                if not row['significant']:
                    continue
                i = scenes.index(row['scene_i'])
                j = scenes.index(row['scene_j'])
                t = frames.index(row['time_frame'])
                counts[i, t] += 1
                counts[j, t] += 1
            df_cnt = pd.DataFrame(counts, index=scenes, columns=frames)
            sns.heatmap(
                df_cnt,
                ax=ax,
                cmap='magma',
                cbar=False,
                linewidths=0.1,
                linecolor='white'
            )

            desired = 10
            step = max(1, n_frames // desired)
            ticks = list(range(0, n_frames, step))
            labels = [frames[i] for i in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=45)
            ax.set_yticklabels(scenes)
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            ax.set_ylabel('Scene', fontsize=12)
            ax.set_title(f'{metric} ({window})', fontsize=14)

    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    sns.heatmap(
        pd.DataFrame(np.zeros((1,1))),
        cbar=True, cbar_ax=cax, cmap='magma',
        norm=plt.Normalize(vmin=0, vmax=df_cnt.values.max())
    )
    cax.set_ylabel('# significant pairs', fontsize=12)

    plt.tight_layout(rect=[0,0,0.85,1])
    plt.savefig(os.path.join(out_dir, 'friedman_all_metrics.png'), dpi=500)
    plt.close()


def plot_overall_friedman_heatmaps(pair_rows: list,
                                   scenes: list,
                                   frames: list,
                                   out_dir: str):
    df = pd.DataFrame(pair_rows)
    windows = ['MA', 'RAW']
    n_frames = len(frames)

    fig, axes = plt.subplots(
        nrows=2, ncols=1,
        figsize=(max(10, n_frames * 0.04), len(scenes) * 0.5 * 2),
        sharex=True
    )
    for wi, window in enumerate(windows):
        ax = axes[wi]
        sub = df[df['window_type'] == window]
        counts = pd.DataFrame(0, index=scenes, columns=frames)
        for _, row in sub.iterrows():
            if not row['significant']:
                continue
            i = row['scene_i']
            j = row['scene_j']
            t = row['time_frame']
            counts.at[i, t] += 1
            counts.at[j, t] += 1

        sns.heatmap(
            counts,
            ax=ax,
            cmap='magma',
            cbar=True,
            linewidths=0.1,
            linecolor='white'
        )
        desired = 40
        step = max(1, n_frames // desired)
        ticks = list(range(0, n_frames, step))
        labels = [frames[i] for i in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(scenes)

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylabel('Scene', fontsize=12)
        ax.set_title(f'All metrics ({window})', fontsize=14)
        if wi == 1:
            ax.set_xlabel('Frame', fontsize=12)
        else:
            ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'overall_friedman_heatmaps.png'), dpi=300)
    plt.close()
