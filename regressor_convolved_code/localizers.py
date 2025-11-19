import os
import numpy as np
import pandas as pd
from scipy.stats import zscore, t, norm
from scipy.special import legendre
from nilearn.glm.first_level import make_first_level_design_matrix

PROJ_DIR = '/r/d7/IRB/Guo/members/ian/DP_project/'
RAW_DATA_DIR = '/r/d7/IRB/Guo/DP_project/data/'


def legendre_polynomials(n_tp, poly_order=2):
    # Make drift model regressors
    x = np.linspace(-1, 1, n_tp)
    poly = np.zeros((n_tp, poly_order + 1))
    for order in range(poly_order + 1):
        poly[:, order] = legendre(order)(x)
    return poly


def get_dar_nuisance(subj, run):
    # Get localizer confounds for dataset-dar
    motion_parameters = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    first_temporal_derivatives = [m+'_derivative1' for m in motion_parameters]
    columns = motion_parameters + first_temporal_derivatives

    darts_preproc_dir = os.path.join(PROJ_DIR, 'preproc/dataset-dar/derivatives/')
    darts_fmriprep_dir = os.path.join(darts_preproc_dir, f'fmriprep/{subj}/ses-01/func/')
    confounds_fn = f'{subj}_ses-01_task-loc_{run}_desc-confounds_timeseries.tsv'
    confounds_file = os.path.join(darts_fmriprep_dir, confounds_fn)
    raw_confounds_df = pd.read_csv(confounds_file, sep='\t')

    # Replace first nan in 'framewise_displacement with 0'
    raw_confounds = np.nan_to_num(raw_confounds_df[columns].values)
    # Zscore confounds, fmriprep recommendation
    confounds = np.nan_to_num(zscore(raw_confounds, axis=0))
    
    # Add drift regressors
    poly = legendre_polynomials(n_tp=confounds.shape[0])
    return np.concatenate((confounds, poly), axis=1) # shape of (126, 15)


def get_dar_contrasts(return_mains=True, return_subs=False, return_only=False):
    # face, body, object, scene, sobject
    main_contrasts = {
        'face-all-contrast': [4, -1, -1, -1, -1],
        'body-all-contrast': [-1, 4, -1, -1, -1],
        'object-all-contrast': [-1, -1, 4, -1, -1],
        'scene-all-contrast': [-1, -1, -1, 4, -1],
        }
    sub_contrasts = {
        'face-contrast': [1, 0, -1, 0, 0],
        'body-contrast': [0, 1, -1, 0, 0],
        'object-contrast': [0 ,0, 1, 0, -1],
        'scene-contrast': [0 ,0, -1, 1, 0],
        }
    only_contrasts = {
        'face-only': [1, 0, 0, 0, 0],
        'body-only': [0, 1, 0, 0, 0],
        'object-only': [0 ,0, 1, 0, 0],
        'scene-only': [0 ,0, 0, 1, 0],
        }
    res_dict = {}
    if return_mains == True:
        res_dict.update(main_contrasts)
    if return_subs == True:
        res_dict.update(sub_contrasts)
    if return_only == True:
        res_dict.update(only_contrasts)
    return res_dict


def get_got_nuisance(subj):
    # Get localizer confounds for dataset-got
    motion_parameters = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    first_temporal_derivatives = [m+'_derivative1' for m in motion_parameters]
    columns = motion_parameters + first_temporal_derivatives

    #change for our own regressor output
    darts_preproc_dir = os.path.join(PROJ_DIR, 'preproc/dataset-got/derivatives/')
    darts_fmriprep_dir = os.path.join(darts_preproc_dir, f'fmriprep/{subj}/func/')
    confounds_fn = f'{subj}_task-localiser_desc-confounds_timeseries.tsv'
    confounds_file = os.path.join(darts_fmriprep_dir, confounds_fn)
    raw_confounds_df = pd.read_csv(confounds_file, sep='\t')
    #_____________

    # Replace first nan in 'framewise_displacement with 0'
    raw_confounds = np.nan_to_num(raw_confounds_df[columns].values)
    # Zscore confounds, fmriprep recommendation
    confounds = np.nan_to_num(zscore(raw_confounds, axis=0))
    
    # Add drift regressors
    poly = legendre_polynomials(n_tp=confounds.shape[0])
    return np.concatenate((confounds, poly), axis=1) # shape of (126, 15)


def get_got_contrasts(return_mains=True, return_subs=False, return_only=False):
    # face, scene, scram
    main_contrasts = {
        'face-all-contrast': [2, -1, -1,],
        'scene-all-contrast': [-1, 2, -1,],
        }
    sub_contrasts = {
        'face-contrast': [1, 0, -1,],
        'scene-contrast': [1, -1, 0,],
        }
    only_contrasts = {
        'face-only': [1, 0, 0],
        'scene-only': [0, 1, 0],
    }
    
    res_dict = {}
    if return_mains == True:
        res_dict.update(main_contrasts)
    if return_subs == True:
        res_dict.update(sub_contrasts)
    if return_only == True:
        res_dict.update(only_contrasts)
    return res_dict


def t_to_z(t_values, n_tp, n_regs):
    # Set degrees of freedom
    df = n_tp - n_regs
    
    z_scores = []
    for t_value in t_values:
        # Convert t to two-tailed p-value
        p_value = 2 * t.sf(abs(t_value), df)

        # Convert to z-score and restore the sign
        z_score = norm.isf(p_value / 2) * np.sign(t_value)
        z_scores.append(z_score)

    return np.array(z_scores)


def _make_stimuli_regressors(subj, run, tr=2, n_tp=126, hrf_model='spm', oversampling=128):
    # Deprecated, see regresssors_dataset-***.py -- NOT SURE WHERE THIS IS
    # Create regressors from BIDS-formatted events file
    bids_dir = os.path.join(RAW_DATA_DIR, f'BIDs/{subj}/ses-01/func/')
    timing_fn = f'{subj}_ses-01_task-loc_{run}_events.tsv'
    timing_file = os.path.join(bids_dir, timing_fn)
    events_df = pd.read_csv(timing_file, sep='\t')
    # print(events_df)

    stimuli = ['face', 'body', 'object', 'scene', 'sobject']
    frames = np.arange(n_tp) * tr
    stimuli_regressors = make_first_level_design_matrix(
        frame_times=frames,
        events=events_df,
        hrf_model=hrf_model,
        oversampling=oversampling,
    )
    return stimuli_regressors[stimuli].values


def _test():
    subj = 'sub-01'
    run = 'run-1'
    regs = _make_stimuli_regressors(subj, run)
    print(regs.shape)

    conf = get_dar_nuisance(subj, run)
    print(conf.shape)


if __name__ == "__main__":
    _test()
    