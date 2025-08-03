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
    n, k = block.shape
    ranks = np.vstack([rankdata(r) for r in block])
    Rj = ranks.sum(axis=0)
    rhos = [lag1_acf(block[:, j]) for j in range(k)]
    rho_bar = np.nanmean(rhos)
    n_eff = max(2.0, n * (1 - rho_bar) / (1 + rho_bar))
    Q = (12.0 / (k * n_eff * (k + 1))) * np.sum(Rj**2) - 3 * n_eff * (k + 1)
    p = 1 - chi2.cdf(Q, k - 1)
    
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
    per_job = max(1, n_perm // n_jobs)
    jobs = [(block, per_job) for _ in range(n_jobs)]
    
    with ProcessPoolExecutor(max_workers=n_jobs) as exe:
        results = list(exe.map(_perm_worker, jobs))
        
    stats, ps = zip(*results)
    
    return stats[0], float(np.mean(ps))


def posthoc_conover_friedman(ranks: np.ndarray, p_adjust: str = 'holm') -> np.ndarray:
    if not USE_CONOVER:
        raise ImportError("Conover scikit-posthocs")
    return sp.posthoc_conover_friedman(ranks, p_adjust=p_adjust).values
conover_posthoc = posthoc_conover_friedman

def bootstrap_ci(data: np.ndarray, num_iterations: int = 10000, alpha: float = 0.05):
    n = len(data)
    boot_means = [
        np.mean(np.random.choice(data, size=n, replace=True))
        for _ in range(num_iterations)
    ]
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper, float(np.mean(boot_means))
