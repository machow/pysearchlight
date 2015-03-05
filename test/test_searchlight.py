import numpy as np
from pysearchlight.search import run_searchlight, gen_searchlight_ind

# -----------------------------------------------------------------------------
# Trivial mean func searchlight example
# -----------------------------------------------------------------------------
mask = np.ones([2,2,2])
data = np.arange(8).reshape([2,2,2])

# get center for each searchlight
kwargs = dict(cutoff=0, metric='euclidean', shape=mask.shape)
centers = gen_searchlight_ind(thr=1, mask=mask, **kwargs)
# calculate mean for each searchlight
result1 = run_searchlight(centers, np.mean, data, center_kwargs=kwargs)


# -----------------------------------------------------------------------------
# Min, max searchlight example
# -----------------------------------------------------------------------------
mask = np.ones([1,1,4])
data = np.arange(mask.size).reshape(mask.shape)

# get center for each searchlight
kwargs = dict(cutoff=1, metric='chebyshev', shape=mask.shape)
centers = gen_searchlight_ind(thr=1, mask=mask, **kwargs)
# calculate mean for each searchlight
def min_max(d): return np.min(d), np.max(d)
result2 = run_searchlight(centers, min_max, data, center_kwargs=kwargs)

# -----------------------------------------------------------------------------
# ISC example
# -----------------------------------------------------------------------------
# 3 subs x (1 x 1 x 15) x 10 time points
# this data is structured so the pairwise correlations are
# -.4, -.3, ..., 0, ..., 1  (there are 15 total)
data = [sub for sub in np.load('test/example_corr_data.npy')]
mask = np.ones([1,1,15])
#
## get center for each searchlight
kwargs = dict(cutoff=0,shape=mask.shape)
centers = gen_searchlight_ind(thr=1, mask=mask, **kwargs)
## calculate mean for each searchlight
def avg_pairwise_corr(d):
    """Calculate average pairwise correlation"""
    corrs = np.corrcoef(np.vstack(d))
    return corrs[np.tril_indices_from(corrs, -1)].mean()

result3 = run_searchlight(centers, avg_pairwise_corr, data, center_kwargs=kwargs)
assert np.allclose(np.array([result3[(0,0,ii)] for ii in range(15)]),
                            np.arange(-.4, 1.1, .1))
