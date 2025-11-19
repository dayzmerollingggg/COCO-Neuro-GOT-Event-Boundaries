# Localizer for dataset-dar, veridical

import os
import sys
import numpy as np
import pandas as pd
import neuroboros as nb
from scipy.stats import zscore
from joblib import Parallel, delayed, parallel_backend

SCRIPTS_DIR = os.path.expanduser('~/DP_project')
sys.path.append(SCRIPTS_DIR)
from utils import load_mask, get_dar_subjects #I don't have this script
from localizers import get_dar_nuisance, get_dar_contrasts, t_to_z 

#t_to_z is degrees of freedom

PROJ_DIR = '/r/d7/IRB/Guo/members/ian/DP_project/'
PREPROC_DIR = os.path.join(PROJ_DIR, 'preproc/dataset-dar/')
REGRESSORS_DIR = os.path.join(PROJ_DIR, 'localizers/dataset-dar/regressors/')
OUTPUT_DIR = os.path.join(PROJ_DIR, 'localizers/dataset-dar/veridical')
os.makedirs(OUTPUT_DIR, exist_ok=True)

#most important script, not sure if need all the subjects, runs, hemi, and contrasts
#since we only have two regressors
#regressor/timestamps/example_regressor_camera_cuts.csv
#regressor/timestamps/example_regressor_scene_cuts.csv
def calculate_glm_veridical(subj, run, hemi, contrasts):
    # Get design matrix
    regressors_fn = os.path.join(REGRESSORS_DIR, f'regressors_{subj}_{run}.csv')
    regressors = pd.read_csv(regressors_fn, index_col=0).values

    # Get nuisance 
    nuisance = get_dar_nuisance(subj, run)

    # Get veridical data from ian's denoised database
    #   would go through different subjects and runs
    # sub-control01/sub-control01_task-GoT_space-fsaverage5_hemi-L_denoised.npy
    mask = load_mask(hemi)
    denoised_dir = os.path.join(PREPROC_DIR, f'denoised/{subj}')
    data_fn = f'{subj}_ses-01_task-loc_{run}_space-fsaverage5_{hemi}_denoised.npy' 
    data = np.load(os.path.join(denoised_dir, data_fn))[:, mask]
    dm = np.nan_to_num(zscore(data, axis=0))
    
    # Calculate glm using neuroboros
    betas, ts = nb.glm(
        dm=dm,
        nuisance=nuisance,
        design=regressors,
        contrasts=contrasts
        )
    
    # Get z from t-stat
    n_tp = dm.shape[0]
    n_regs = regressors.shape[1] + nuisance.shape[1]
    zs = t_to_z(t_values=ts, n_tp=n_tp, n_regs=n_regs)

    return betas, ts, zs


def pipe_wrapper(subj, run, hemi, contrasts_dict, output_dir):
    # For parallel processsing -- I don't think I need this?
    out_fn = os.path.join(output_dir, f'{subj}_{run}_{hemi}.npz')
    # if os.path.exists(out_fn):
    #     print(f'{out_fn} exists, skipping')
    #     return
    contrasts = [contrasts_dict[c] for c in contrasts_dict]
    betas, ts, zs = calculate_glm_veridical(subj, run, hemi, contrasts)
    np.savez(out_fn, betas=betas, ts=ts, zs=zs)
    

def _testing(): # one test case
    subj = 'sub-01'
    run = 'run-1'
    hemi = 'hemi-L'
    dar_contrasts = get_dar_contrasts(return_subs=True, return_only=True)
    contrasts = [dar_contrasts[c] for c in dar_contrasts]
    betas, ts, zs = calculate_glm_veridical(subj, run, hemi, contrasts)
    print(betas.shape, ts.shape, zs.shape)


if __name__ == "__main__": #going through all subjects and runs, making parallel processes
    _testing()

    subjects = get_dar_subjects(two_tasks=False)
    darts_contrasts = get_dar_contrasts(return_subs=True, return_only=True)
    jobs = []
    for subj in subjects:
        for run in [f'run-{r+1}' for r in range(5)]:
            for hemi in ['hemi-L', 'hemi-R']:
               jobs.append(delayed(pipe_wrapper)(subj, run, hemi, darts_contrasts, OUTPUT_DIR))
    
    with parallel_backend("loky", inner_max_num_threads=1):
        Parallel(n_jobs=4, verbose=2)(jobs)
