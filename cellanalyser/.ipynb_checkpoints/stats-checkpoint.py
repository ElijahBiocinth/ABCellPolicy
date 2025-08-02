import numpy as np
from scipy.stats import friedmanchisquare, rankdata, chi2
from concurrent.futures import ProcessPoolExecutor

try:
    import scikit_posthocs as sp
    USE_CONOVER = True
except ImportError:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    USE_CONOVER = False

def lag1_acf(x: np.ndarray) -> float:
    if x.size < 3 or np.allclose(x, x[0]):
        return 0.0
    x1, x2 = x[:-1], x[1:]
    if np.std(x1)==0 or np.std(x2)==0:
        return 0.0
    return np.corrcoef(x1, x2)[0,1]

def friedman_neff(block: np.ndarray):
    n_win, k_loc = block.shape
    ranks = np.vstack([rankdata(row) for row in block])
    Rj = ranks.sum(axis=0)
    rhos = [lag1_acf(block[:, j]) for j in range(k_loc)]
    rho_bar = float(np.nanmean(rhos))
    n_eff = max(2.0, n_win * (1 - rho_bar) / (1 + rho_bar))
    Q = (12.0 / (k_loc * n_eff * (k_loc + 1))) * np.sum(Rj**2) - 3 * n_eff * (k_loc + 1)
    p = 1.0 - chi2.cdf(Q, k_loc - 1)
    return Q, p, n_eff, rho_bar

def _perm_worker(args):
    block, n_perm = args
    stat_obs, _ = friedmanchisquare(*[block[:, j] for j in range(block.shape[1])])
    count = 0
    for _ in range(n_perm):
        perm = np.apply_along_axis(np.random.permutation, 0, block)
        stat_p, _ = friedmanchisquare(*[perm[:, j] for j in range(perm.shape[1])])
        if stat_p >= stat_obs:
            count += 1
    return stat_obs, (count + 1) / (n_perm + 1)

def parallel_permutation(block: np.ndarray, n_perm: int, n_jobs: int):
    per_job = max(1, n_perm // n_jobs)
    jobs = [(block, per_job) for _ in range(n_jobs)]
    with ProcessPoolExecutor(max_workers=n_jobs) as exe:
        results = exe.map(_perm_worker, jobs)
    stats, ps = zip(*results)
    return stats[0], float(np.mean(ps))

def conover_posthoc(block: np.ndarray, alpha: float):
    ranked = np.apply_along_axis(rankdata, 1, block)
    k = ranked.shape[1]
    if USE_CONOVER:
        pmat = sp.posthoc_conover(ranked, p_adjust='holm').values
    else:
        import pandas as pd
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        df = pd.DataFrame(ranked, columns=list(range(k)))
        melted = df.melt(var_name='group', value_name='rank')
        tuk = pairwise_tukeyhsd(melted['rank'], melted['group'], alpha=alpha)
        groups = list(tuk.groupsunique)
        pmat = np.ones((k, k))
        for row in tuk._results_table.data[1:]:
            g1, g2, _, pval, _ = row
            i, j = groups.index(g1), groups.index(g2)
            pmat[i, j] = pval
            pmat[j, i] = pval
    return pmat

def bootstrap_ci(data: np.ndarray, num_iterations: int = 10000, alpha: float = 0.05):
    n = len(data)
    boot_means = [np.mean(np.random.choice(data, size=n, replace=True))
                  for _ in range(num_iterations)]
    lower = np.percentile(boot_means, 100 * (alpha/2))
    upper = np.percentile(boot_means, 100 * (1 - alpha/2))
    return lower, upper, float(np.mean(boot_means))
