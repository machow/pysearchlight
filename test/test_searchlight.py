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
