from pysearchlight.search import gen_searchlight_ind, calc_slices, searchlight_ind
import numpy as np
import tempfile

#gen_searchlight_ind
def test_gen_ind_from_mask():
    # no centers + thr=0 returns all mask coords as centers
    ind = gen_searchlight_ind(None, mask=np.ones([2,2,2]), thr=0, cutoff=0)
    assert len(ind) == 2**3

def test_gen_ind_outside_mask():
    # centers outside mask aren't returned
    ind = gen_searchlight_ind([[3,3,3]], mask=np.ones([2,2,2]), thr=0, cutoff=0)
    assert len(ind) == 0

def test_gen_ind_nans():
    # nans produce the same result as 0 in a mask
    m1 = np.ones([2,2,2])
    m2 = m1.copy()
    m1[0] = 0
    m2[0] = np.nan
    c1 = gen_searchlight_ind(None, mask=m1, thr=0, cutoff=0)
    c2 = gen_searchlight_ind(None, mask=m2, thr=0, cutoff=0)
    assert len(c1) == len(c2)

def test_gen_ind_save_and_load():
    with tempfile.TemporaryFile() as tf:
        gen_searchlight_ind(None, mask=np.ones([3,3,3]), thr=0, output=tf, cutoff=0)
        tf.seek(0)
        cnfg = np.load(tf)[()]
        for k in ['centers', 'kwargs']: assert k in cnfg


# -----------------------------------------------------------------------------
# calc_slices
# -----------------------------------------------------------------------------
def test_total_batches():
    # b_ttl is total batches necessary (not +1)
    ttl = 5
    assert len(calc_slices(22, slice(0,None), ttl)) == ttl

def test_no_lost_batches():
    # slices will reconstruct all centers
    slices = calc_slices(22, slice(0,None), 5)
    arr = range(22)
    assert sum([arr[s] for s in slices], []) == arr

def test_same_size_batches():
    # batch 0 and batch 1 are the same size
    s1 = calc_slices(22, 0, 5)
    s2 = calc_slices(22, 1, 5)
    assert (s1.stop - s1.start) == (s2.stop - s2.start)


# -----------------------------------------------------------------------------
# searchlight_ind(c, r, shape, metric)
# -----------------------------------------------------------------------------
def test_single_index():
    # array with one entry
    assert len(searchlight_ind([0,0,0], cutoff=0, shape=[1,1,1])[0]) == 1
    # array with two entries
    assert len(searchlight_ind([0,0,0], cutoff=0, shape=[1,1,2])[0]) == 1

# center + r outside shape
def test_cutoff_edges():
    assert len(searchlight_ind([0,0,0], cutoff=1, shape=[1,1,2])[0]) == 2
    # cutoff two, but only 1 entry (edges)
    assert len(searchlight_ind([0,0,0], cutoff=1, shape=[1,1,1])[0]) == 1

def test_center_outside_shape():
    # center outside shape
    assert len(searchlight_ind([2,2,2], cutoff=0, shape=[1,1,1])[0]) == 0  # return empty, should raise

def test_index_shape():
    ind = searchlight_ind([7,7,7], cutoff=2, shape=[15,15,15])
    assert len(ind[0]) == len(ind[1]) == len(ind[2])
    # make into 3-tuple coords
    assert len(zip(*ind)) == len(ind[0])
    assert len(zip(*ind)[0]) == 3

# indices correspond to metric
def test_sphere_size():
    # sphere
    kwargs = dict(center=[7,7,7], cutoff=2, shape=[15,15,15])
    sphere = searchlight_ind(metric='euclidean', **kwargs)
    assert len(sphere[0]) == 33
    cube = searchlight_ind(metric='chebyshev', **kwargs)
    assert len(cube[0]) == 5**3
