PySearchlight
=============
Package for dividing and conquering searchlight (and other) analyses.

How it works
------------
### Simple Searchlight
One horribly arbitrary but simple demonstration of how this library works is by using a searchlight over a single point.
First we'll get searchlight centers from `gen_searchlight_ind`, 
and then we'll pass centers and a function to operate over searchlight data to `run_searchlight`.

```python
import numpy as np
from pysearchlight import run_searchlight, gen_searchlight_ind

# Make mask for selecting searchlight centers
mask = np.ones([2,2,2])                 # don't mask any center points
# Make simple data, counting up to 8
data = np.arange(mask.size).reshape(mask.shape)
# set searchlight options
kwargs = dict(cutoff=0, metric='euclidean', shape=mask.shape)
# get center for each searchlight (using mask)
centers = gen_searchlight_ind(thr=1, mask=mask, **kwargs)
```

with `centers` in hand, we can run the searchlight.
Any function that can run over a Nx(non-spatial dims), where N is the voxels selected for the searchlight, will work.
Below, we just use numpy's mean function. This function will be run over every searchlight.

```python
# calculate mean for each searchlight
result1 = run_searchlight(centers, np.mean, data, center_kwargs=kwargs)

result1
#{(0, 0, 0): 0.0,
# (0, 0, 1): 1.0,
# (0, 1, 0): 2.0,
# (0, 1, 1): 3.0,
# (1, 0, 0): 4.0,
# (1, 0, 1): 5.0,
# (1, 1, 0): 6.0,
# (1, 1, 1): 7.0}
```

### A Little Fancier
Here, we mix things up by getting searchlights with 3 points in them.

```python
# -----------------------------------------------------------------------------
# Min, max searchlight example
# -----------------------------------------------------------------------------
mask = np.ones([1,1,4])
data = np.arange(mask.size).reshape(mask.shape)

# get center for each searchlight
kwargs = dict(cutoff=1, metric='chebyshev', shape=mask.shape)
centers = gen_searchlight_ind(thr=1, mask=mask, **kwargs)
```

Notice also that `metric='chebyshev'` is specifying a cube (though it doesn't matter with the dimensions of this data).
pysearchlight calculates distance using `scipy.spatial.cdist`, so metrics can be any distance type it accepts.
Finally, let's return the maximum and minumum for each searchlight.

```
def min_max(d): return np.min(d), np.max(d)
result2 = run_searchlight(centers, min_max, data, center_kwargs=kwargs)

result2
#{(0, 0, 0): (0, 1),
# (0, 0, 1): (0, 2), 
# (0, 0, 2): (1, 3), 
# (0, 0, 3): (2, 3)}
```

### Shape of Searchlight Data
Conveniently, we can use numpy's `shape` function to see what data the searchlight is using.
Below, the data has 2 non-spatial dimensions..

```python
mask = np.ones([4,4,4])
data = np.arange(mask.size*2).reshape(list(mask.shape)+[2])

# get center for each searchlight
kwargs = dict(cutoff=1, metric='chebyshev', shape=mask.shape)
centers = gen_searchlight_ind(thr=1, mask=mask, **kwargs)
result3 = run_searchlight(centers, np.shape, data, center_kwargs=kwargs)

result3
#{(0, 0, 0): (8, 2),
# (0, 0, 1): (12, 2),
# (0, 0, 2): (12, 2),
# ...
# (1, 1, 1): (27, 2),
# ...
# (3, 3, 3): (8, 2)}
```

One thing that should stick out here, is that when the searchlight extends beyond the data, it ends up using fewer points.
You can get around this by making your mask larger than your center points...

```python
new_mask = np.zeros([10,10,10])
new_mask[4:8, 4:8, 4:8] = 1
new_data = np.zeros([10,10,10, 2])
new_data[4:8, 4:8, 4:8] = data
kwargs['shape'] = new_mask.shape
centers = gen_searchlight_ind(thr=0, mask=new_mask, **kwargs)
result4 = run_searchlight(centers, np.shape, new_data, center_kwargs=kwargs)

result4

#{(4, 4, 4): (27, 2),
# (4, 4, 5): (27, 2),
# (4, 4, 6): (27, 2),
# ...
#}
```

After making this example, I decided to should probably add an option to require searchlight is completely within array.


### Cut the Crap, Let's do a Bonifide Searchlight


### Divide and Conquer

Format for Parallel Jobs
------------------------

Example
-------
