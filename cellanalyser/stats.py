# ABCellPolicy/cellanalyser/stats.py

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, chi2
from concurrent.futures import ProcessPoolExecutor

try:
    import scikit_posthocs as sp
    USE_CONOVER = True
except ImportError:
    USE_CONOVER = False

def lag1_acf(x: np.ndarray) -> float:
    if x.size < 3 or np.allclose(x, x[0]):
        return 0.0
    x1, x2 = x[:-1], x[1:]
    return np.corrcoef(x1, x2)[0,1] if np.std(x1) and np.std(x2) else 0.0

def friedman_neff(block: np.ndarray):
    """
    Классический тест Фридмана + оценка эффективного размера выборки.
    Возвращает (Q, p_classic, N_eff, rho_bar).
    """
    n_win, k_loc = block.shape
    ranks = np.vstack([rankdata(row) for row in block])
    Rj = ranks.sum(axis=0)
    rhos = [lag1_acf(block[:, j]) for j in range(k_loc)]
    rho_bar = np.nanmean(rhos)
    n_eff = max(2.0, n_win * (1 - rho_bar) / (1 + rho_bar))
    Q = (12.0 / (k_loc * n_eff * (k_loc + 1))) * np.sum(Rj**2) - 3 * n_eff * (k_loc + 1)
    p = 1.0 - chi2.cdf(Q, k_loc - 1)
    return Q, p, n_eff, rho_bar

def _perm_worker(args):
    block, n_perm = args
    stat_obs, _ = friedmanchisquare(*[block[:, j] for j in range(block.shape[1])])
    count = 0
    for _ in range(n_perm):
        perm = np.apply_along_axis(np.random.permutation, 1, block)
        stat, _ = friedmanchisquare(*[perm[:, j] for j in range(perm.shape[1])])
        if stat >= stat_obs:
            count += 1
    p_perm = (count + 1) / (n_perm + 1)
    return stat_obs, p_perm

def parallel_permutation(block: np.ndarray, n_perm: int, n_jobs: int):
    """
    Пермутационный тест Фридмана: делим n_perm на n_jobs и усредняем p.
    """
    per_job = max(1, n_perm // n_jobs)
    jobs = [(block, per_job) for _ in range(n_jobs)]
    with ProcessPoolExecutor(max_workers=n_jobs) as exe:
        stats, ps = zip(*exe.map(_perm_worker, jobs))
    return stats[0], float(np.mean(ps))

def posthoc_conover_friedman(ranks: np.ndarray, p_adjust: str = 'holm') -> np.ndarray:
    """
    Пост-хок Conover для теста Фридмана через scikit-posthocs.
    Возвращает матрицу p-значений.
    """
    if not USE_CONOVER:
        raise ImportError("Для Conover нужен пакет scikit-posthocs")
    return sp.posthoc_conover_friedman(ranks, p_adjust=p_adjust).values

def bootstrap_ci(data: np.ndarray, num_iterations: int = 10000, alpha: float = 0.05):
    """
    Bootstrap-доверительный интервал для среднего.
    Возвращает (lower, upper, mean_bootstrap).
    """
    n = len(data)
    boot_means = [
        np.mean(np.random.choice(data, size=n, replace=True))
        for _ in range(num_iterations)
    ]
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper, float(np.mean(boot_means))
