import numpy as np
import os
# Specific functions fed to searchlight ---------------------------------------
# TODO should move these out (into pycorr?) eventually

def pattern_similarity(d_list, TRs, offset_TR=5, pattern_indx=None, nan_thresh=None):
    """
    Parameters:
        pattern_indx:   boolean mask over segment-segment correlation matrix
        d_interl:       list of nifti images with dim (xyzt)
        d_intact:       same
        TRs_interl:     dataframe describing each TR from story blueprint
        TRs_intact:     same
        offset_TR:      number of TRs to shift data (to offset for HRF)

    """
    segmat = load_mvpa(d_list, TRs, None, offset_TR)
    
    # TODO: What if voxel nans for sub?
    segcors = crosscor_full(segmat, nan_thresh=nan_thresh)
    # Do comparison
    #return segcors.mean(axis=0)[pattern_indx].mean()
    return segcors # sub x seg x seg


from pycorr.funcs_correlate import crosscor, standardize, sum_tc
def crosscor_full(A, B=None, nan_thresh=None):
    """From data (dims sub x seg x vox) calculate sub against others pattern cor

    Parameters:
        A: sub x seg x vox matrix
        B: (optional) seg x vox matrix

    Returns:
        seg x seg correlation matrix
    """
    # standardize all along last dim, so don't need to in correlation
    A = standardize(A)

    all_cors = []
    # Within group correlations
    if B is None:
        others = sum_tc(A)
        for sub in A:
            # check for nan
            to_remove = np.any(np.isnan(sub), axis=0)
            if np.any(to_remove):
                tmp_sub = sub[...,to_remove]
                tmp_others = others[..., to_remove]
            else:
                tmp_sub, tmp_others = sub, others
            # cross correlate (1 x seg x seg)
            if nan_thresh is not None and to_remove.mean() > nan_thresh:
                cormat = np.empty(sub.shape[0:1]*2) * np.nan
            else:
                cormat = crosscor(tmp_sub, standardize(tmp_others - tmp_sub), standardized=True)
            all_cors.append(cormat)
        return np.array(all_cors)
    # Between group correlations
    else:
        B = standardize(B)
        for sub in A:
            cormat = crosscor(sub, B, standardized=True)
            all_cors.append(cormat)
        return np.array(all_cors)


# Functions taken from event_analysis -----------------------------------------

def subset_from_TRs(mat, TRs, offset=0):
    """Subset last dim of nparray using TR index."""
    indx = TRs.tshift(freq=offset).index       # align TRs to mat
    #mask = np.array([ii in indx for ii in range(mat.shape[-1])])
    # change to directly select using index.
    # will throw an error if index is longer than releavant mat dim!
    return mat[..., indx.tolist()]

def load_mvpa(all_fnames, TRs, bad_vox, offset_TR, collapse=True):
    """Return matrix of shape (sub x seg x vox)

    Parameters:
        all_fnames:     names of nifti files to load for sub dimension
        TRs:            dataframe with cond column and order column
        bad_vox:        mask with true for voxels to be discarded
        offset_TR:      TRs to shift timecourses before subsetting (to take into account lag, etc..)
        collapse:       whether to take mean along last axis

    Notes:
        If collapse is False, then sub and seg dims are lists.

    """
    subs_list = []
    for fname in all_fnames:
        # Load Array
        if type(fname) is str:
            subname, ext = os.path.splitext(os.path.split(fname)[-1])
            #subkey = "_".join(subname.split('_')[:2])
            arr = np.load(fname)
        else: arr = fname

        # Standardize, make sure no NaNs
        arr = standardize(arr)[~bad_vox if bad_vox is not None else Ellipsis] #TODO write more clearly
        #arr = arr[ np.isnan(arr).sum(axis=-1) == 0]     #remove any nans (from no var?)

        # Get individual segments
        # since it sorts in order of columns, will be sorted by cond first, then order
        segs_list = []
        cond_list = []
        # TODO remove hard coded conditions
        for ii, g in TRs.query("cond in ['Slumlord', 'Overview']").groupby(['cond', 'order']):
            #print ii
            cond_list.append(ii)
            segarr = subset_from_TRs(arr, g, offset=offset_TR)
            segs_list.append(segarr)
        # optionally collapse time dimension
        if collapse:
            mat = np.vstack([seg.mean(axis=1) for seg in segs_list])
            subs_list.append(mat)
        else:
            subs_list.append(segs_list)
        #print cond_list

    M = np.array(subs_list) if collapse else subs_list # Make field array, with sub names?
    return M
