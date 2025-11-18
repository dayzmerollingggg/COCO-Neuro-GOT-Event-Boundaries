# First level analysis

import os
import sys
import numpy as np
import pandas as pd
import neuroboros as nb
from scipy.stats import zscore, t, norm
from scipy.special import legendre
from joblib import Parallel, delayed, parallel_backend

SCRIPTS_DIR = '/mnt/labdata/got_project/daisy/got_project'#os.path.expanduser('~/Documents/got_project')
sys.path.append(SCRIPTS_DIR)
from utils import get_got_subjects, load_mask

PROJ_DIR = '/mnt/labdata/got_project' #correct
FMRI_DATA_DIR = os.path.join(PROJ_DIR, 'data') #correct
DATA_DIR = os.path.join(PROJ_DIR, 'daisy/data') #correct
REGRESSORS_DIR = os.path.join(DATA_DIR, 'regressors') #correct

#OUTPUT_DIR = os.path.join(DATA_DIR, 'glm/simple_contrasts')
OUTPUT_DIR = os.path.join(DATA_DIR, 'glm/compare_contrasts')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def legendre_polynomials(n_tp, poly_order=2):
    # Make drift model regressors
    x = np.linspace(-1, 1, n_tp)
    poly = np.zeros((n_tp, poly_order + 1))
    for order in range(poly_order + 1):
        poly[:, order] = legendre(order)(x)
    return poly


def t_to_z(t_values, n_tp, n_regs):
    # Get z-score from t-value
    df = n_tp - n_regs # Set degrees of freedom
    
    z_scores = []
    for t_value in t_values:
        # Convert t to two-tailed p-value
        p_value = 2 * t.sf(abs(t_value), df)

        # Convert to z-score and restore the sign
        z_score = norm.isf(p_value / 2) * np.sign(t_value)
        z_scores.append(z_score)

    return np.array(z_scores)


def get_got_nuisance(subj):
    # Get movie scan confounds
    motion_parameters = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    first_temporal_derivatives = [m+'_derivative1' for m in motion_parameters]
    columns = motion_parameters + first_temporal_derivatives

    preproc_dir = os.path.join(FMRI_DATA_DIR, 'derivatives')
    fmriprep_dir = os.path.join(preproc_dir, f'fmriprep/{subj}/func')
    confounds_fn = f'{subj}_task-GoT_desc-confounds_timeseries.tsv'
    confounds_file = os.path.join(fmriprep_dir, confounds_fn)
    raw_confounds_df = pd.read_csv(confounds_file, sep='\t')

    # Replace first nan in 'framewise_displacement with 0'
    raw_confounds = np.nan_to_num(raw_confounds_df[columns].values)
    # Zscore confounds, fmriprep recommendation
    confounds = np.nan_to_num(zscore(raw_confounds, axis=0))
    
    # Add drift regressors
    poly = legendre_polynomials(n_tp=confounds.shape[0])
    return np.concatenate((confounds, poly), axis=1) # shape of (126, 15)


def calculate_glm(subj, hemi, regressors, contrasts):
    # Get nuisance 
    nuisance = get_got_nuisance(subj)

    # Get veridical data
    mask = load_mask(hemi)
    denoised_dir = os.path.join(FMRI_DATA_DIR, f'denoised/{subj}')
    data_fn = f'{subj}_task-GoT_space-fsaverage5_{hemi}_denoised.npy'
    data = np.load(os.path.join(denoised_dir, data_fn))[:, mask]
    data_matrix = np.nan_to_num(zscore(data, axis=0))
    
    # Calculate glm using neuroboros
    betas, ts = nb.glm(
        dm=data_matrix,
        nuisance=nuisance,
        design=regressors,
        contrasts=contrasts
        )
    
    # Get z from t-stat
    n_tp = data_matrix.shape[0]
    n_regs = regressors.shape[1] + nuisance.shape[1]
    zs = t_to_z(t_values=ts, n_tp=n_tp, n_regs=n_regs)

    return betas, ts, zs

def create_simple_contrasts_zero_one(design_matrix):
    print(np.eye(len(design_matrix.columns)))
    # This creates simple contrasts for a quick and dirty analysis
    # For a design matrix with 3 features (e.g. person_knowledge_features), the contrasts would look like
    # 1     0       0
    # 0     1       0
    # 0     0       1
    return np.eye(len(design_matrix.columns))


def pipe_wrapper(subj, hemi, regressors, contrasts_array, category_name):
    # For parallel processsing
    out_fn = os.path.join(OUTPUT_DIR, f'{subj}_{category_name}_{hemi}.npz')
    # if os.path.exists(out_fn):
    #     print(f'{out_fn} exists, skipping')
    #     return
    contrasts = [contrasts_array[c, :] for c in range(contrasts_array.shape[0])] # convert array to list of rows for neuroboros
    #contrasts =np.array([[-1.0, 1.0]])
    betas, ts, zs = calculate_glm(subj, hemi, regressors, contrasts)
    np.savez(out_fn, betas=betas, ts=ts, zs=zs)
    

if __name__ == "__main__":
    hemis = ['hemi-L', 'hemi-R']
    subjects = get_got_subjects()
    regressors_files = [f for f in os.listdir(REGRESSORS_DIR) if '.csv' in f]
    
    jobs = []
    for test_reg in regressors_files:
        category_name = test_reg.replace('_regressors.csv', '')
        regressors_fn = os.path.join(REGRESSORS_DIR, test_reg)
        design_matrix = pd.read_csv(regressors_fn, index_col=0) 
        contrasts = create_simple_contrasts_zero_one(design_matrix)
        for subj in subjects:
            for hemi in hemis:
                jobs.append(delayed(pipe_wrapper)(subj, hemi, design_matrix, contrasts, category_name))
                
    with parallel_backend("loky", inner_max_num_threads=1):
        Parallel(n_jobs=4, verbose=2)(jobs)
