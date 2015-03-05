import numpy as np
import pandas as pd
import nibabel as nib
import argh
from argh import arg
from pycorr.pietools import load_nii_or_npy
import os

# pysearchlight specific
from methods import pattern_similarity

# Command-line functions ------------------------------------------------------

# TODO rename cutoff of radius?
@arg('--mask', type=load_nii_or_npy)
def gen_searchlight_ind(centers=None, mask=None, thr=.7, output='', **kwargs):
    """Find all indices from centers with usable voxels over threshold.

    Parameters:
        centers: list of center points to make searchlight around (defaults to all non-masked)
        cutoff:  radius around each center
        mask:    3d spatial mask (of usable voxels set to 1)
        thr :    proportion of usable voxels necessary
        output : optional file to output centers and center kwargs to
        kwargs: arguments to pass to searchlight_ind

    Returns:
        2-Tuple of usable centers, and list with indices from searchlight_ind function.
    """
    # make centers a list of 3-tuple coords  of nonzero mask values if not given
    mask = mask.copy()
    mask[np.isnan(mask)] = 0
    centers = zip(*np.nonzero(mask)) if centers is None else centers

    # add mask to kwargs if not there
    if not kwargs.get('shape'): kwargs['shape'] = mask.shape

    good_center = []
    cycle = 0
    for center in centers:
        cycle += 1
        # TODO need to get number of max possible indices
        ind = searchlight_ind(center, **kwargs)
        if mask[ind].mean() >= thr:     # TODO chokes on nans?
            #all_inds.append(ind)
            good_center.append(center)

    # optionally pickle a dictionary of center: indices
    if output: 
        # For now, just store good center points
        np.save(output, {'centers' : good_center,
                         'kwargs'  : kwargs}
                         )
        return "finished"

    return good_center

@arg('total_ind', type=lambda x: np.load(x)[()]['centers'], help='total centers in searchlight')
@arg('batch_num', type=int, help='batch number')
@arg('batch_ttl', type=int, help='total batches')
def calc_slices(total_ind, batch_num, batch_ttl, print_uniq=False):
    """Convenience function for breaking centers into batches for parallel jobs.
    
    Note that voxels are returned as [(x, y, z), ...]
    """
    if hasattr(total_ind, '__len__'): total_ind = len(total_ind)
    step = total_ind / (batch_ttl - bool(total_ind % batch_ttl))
    chunks = [slice(ii, ii+step) for ii in range(0, total_ind, step)]
    return chunks[batch_num]

def unique_vox(centers, cutoff, shape=(70,70,70)):
    s = set()
    for c in centers:
        s.update(zip(*searchlight_ind(c, cutoff, shape)))
    return list(s)

@arg('fdir', type=str, help="directory where batches are saved")
@arg('max_batch_num', type=int, help="largest batch number (for finding unsaved jobs)")
@arg('--centers', type=lambda x: np.load(x)[()]['centers'], help="file holding centers (for checking voxels)")
def check_output_dir(fdir, max_batch_num, centers=None):
    """Verify that all jobs saved correctly to output directory"""
    voxels = set()
    files = set()
    for fname in glob(os.path.join(fdir, '*')):
        voxels.update(np.load(fname)[()])
        files.add(int(fname.split('/')[-1].split('.')[0]))
    print "Total number of filled voxels:\t", len(voxels)

    if max_batch_num:
        print "=== Missing Batches ==="
        for ii in sorted(set(range(max_batch_num)) - files): print ii

    if centers is not None:
        print "Missing voxels: \t", len(set(centers) - voxels)

def sub_from_batches(subnum, batches, ref_nii, empty=-2):
    """Return nifti with all subject data from batches.

    Assumes first dim of each output is subject.

    Parameters:
        subnum:  index of subject within batch entries
        batches: list of all batches (should be coord : entry dicts)
        ref_nii: reference nifti for output
        empty:   default value for empty voxels

    """
    example_entry = batches[0].itervalues().next()

    dv_dims = example_entry.shape[1:]
    dim_xyz = ref_nii.shape[:-1]

    print "dv dims: ", dv_dims
    print "xyz: ", dim_xyz
    
    dat = np.ones(dim_xyz + dv_dims, dtype='float32')*empty

    for batch in batches:
        # put each center into dest arrays
        for coord, entry in batch.iteritems():
            # cast coord into tuple just in case it is a list 
            # (np would index differently)
            dat[tuple(coord)] = entry[subnum]
            
    nii = nib.Nifti1Image(dat, ref_nii.get_affine(), ref_nii.get_header())
    return nii
    

@arg('ref_nii', type=nib.load, help="example nifti for output hdr and shape")
def stitch(in_dir, out_dir, ref_nii, empty=-2, subs=None):
    """Stitch together individual nifti for each subject in batches.

    Parameters:
        in_dir:  input directory (holding batches)
        out_dir: output directory (will hold subject niftis)
        ref_nii: reference nifti
        empty  : default value for empty voxels
        subs   : subjects to stitch (defaults to all)

    """
    # make all directories necessary
    try: os.makedirs(out_dir)
    except OSError: pass
    # batches to stitch
    batch_files = glob(os.path.join(in_dir, '*'))
    batches = [np.load(fname)[()] for fname in batch_files]  # TODO could change to iterator (if memory concerns)
    example_entry = batches[0][batches[0].keys()[0]]

    subs = subs if subs else range(example_entry.shape[0])
    print "N Subs: ", len(subs)

    for subnum in subs:
        print 'stitching: ', subnum
        nii = sub_from_batches(subnum, batches, ref_nii, empty=empty)
        nib.save(nii, os.path.join(out_dir, 'sub_' + str(subnum) + '.nii.gz'))



@arg('centers', type=lambda x: np.load(x)[()]['centers'], help="npy file with center points")
@arg('d', type=str, help="folder with data (as npy or nii)")
@arg('center_kwargs', type=lambda x: np.load(x)[()]['kwargs'], help="npy file with kwargs for getting center points")
@arg('output', type=str, help="folder to store results in")
@arg('--offset_TR', type=int)
@arg('--nan_thresh', type=int, help="max proportion of nans voxels necessary")
@arg('--batch_num', type=int, help="starting index for running in parallel")
@arg('--batches_ttl', type=int, help="total number of batches (for parallel)")
def run_pattern_searchlight(centers, d, center_kwargs, output,
                            TRs="", offset_TR=0, nan_thresh = .7,
                            batch_num=None, batches_ttl=None):
    """Convenience wrapper for searchlight running pattern similarity"""

    all_indices = zip(*unique_vox(centers, center_kwargs['cutoff']))

    print "number of centers is:\t", len(centers)
    print "number of voxels is:\t", len(all_indices[0])

    if type(d) is str: d = [SubMMap(fname, all_indices) for fname in glob(d)]
    if type(TRs) is str: TRs = pd.read_csv(TRs)
    # prepare output directory
    try: os.makedirs(output)
    except OSError: pass

    output = os.path.join(output, str(batch_num))

    indx = calc_slices(len(centers), batch_num, batches_ttl)
    print "slice is:\t",
    print indx
    chunk_centers = centers[indx]

    run_searchlight(chunk_centers, pattern_similarity, d, 
                    center_kwargs = center_kwargs, output = output,
                    TRs = TRs, offset_TR=offset_TR, nan_thresh=nan_thresh)

# Searchlight Functions -------------------------------------------------------

from glob import glob
class SubMMap:
    def __init__(self, fname, all_indices):
        self.fname = fname
        self.data = self.hash_indices(load_nii_or_npy(self.fname), all_indices)

    def __getitem__(self, x):
        return np.array([self.data[ind] for ind in zip(*x)])

    @staticmethod
    def hash_indices(arr, indices):
        """Creates hash table from indices (that are structured like output of np.nonzero)"""
        return {ind : np.asarray(arr[ind[0], ind[1], ind[2]]).copy() for ind in zip(*indices)}


def run_searchlight(centers, func, d, center_kwargs=None, output="", **kwargs):
    """Convenience function for running a pattern comparison method.

    Currently only runs pattern_disc1.
    """

    if centers is None:
        centers = gen_searchlight_ind(cutoff=center_kwargs['cutoff'], mask=np.ones(d.shape), thr=0)

    out = {}
    cycle = 0
    for c in centers:
        cycle += 1
        if cycle % 500 == 0: print cycle
        # Get indices from center
        ind = searchlight_ind(c, **center_kwargs)
        # Mask to get only indices used in searchlight
        if isinstance(d, list): d_search = [sub[ind] for sub in d]   # sub x space[ind] x time
        else: d_search = d[ind]
        # Pass to pattern comparison method
        out[c] = func(d_search, **kwargs)

    print "outputting to:\t", output
    if output: np.save(output, out)
    else: return out


import scipy.spatial as spat
def searchlight_ind(center, cutoff, shape, metric='euclidean'):
    """Return indices for searchlight where distance <= cutoff

    Parameters:
        center: point around which to make searchlight
        cutoff: radius of searchlight
        shape:  shape of data
        metric: distance metric to evaluate against cutoff    

    Returns:
        numpy array of shape (3, N_indices) for subsetting data
    """
    center = np.array(center)
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    z = np.arange(shape[2])

    #First mask the obvious points- may actually slow down your calculation depending.
    x=x[abs(x-center[0])<=cutoff]
    y=y[abs(y-center[1])<=cutoff]
    z=z[abs(z-center[2])<=cutoff]


    #Generate grid of points
    X,Y,Z=np.meshgrid(x,y,z)
    data=np.vstack((X.ravel(),Y.ravel(),Z.ravel())).T
    
    distance=spat.distance.cdist(data,center.reshape(1,-1), metric).ravel()
    return data[distance<=cutoff].T.tolist()   # return list like np.nonzero

def main():
    parser = argh.ArghParser()
    parser.add_commands([gen_searchlight_ind, run_pattern_searchlight, calc_slices, check_output_dir, stitch])
    parser.dispatch()

if __name__ == '__main__':
    main()
