# First level analysis

import os
import sys
import numpy as np
import pandas as pd
import neuroboros as nb
from scipy.stats import zscore, t, norm
from scipy.special import legendre
from joblib import Parallel, delayed, parallel_backend
from scipy.stats import ttest_1samp # You'll need this import
import matplotlib.pyplot as plt
from brainplotlib import brain_plot

SCRIPTS_DIR = '/mnt/labdata/got_project/daisy/got_project'#os.path.expanduser('~/Documents/got_project')
sys.path.append(SCRIPTS_DIR)
from utils import get_got_subjects, load_mask

PROJ_DIR = '/mnt/labdata/got_project' #correct
FMRI_DATA_DIR = os.path.join(PROJ_DIR, 'data') #correct
DATA_DIR = os.path.join(PROJ_DIR, 'daisy/brain_plot_data_output') #correct
REGRESSORS_DIR = os.path.join(DATA_DIR, 'regressors') #correct

#OUTPUT_DIR
GLM_DIR = os.path.join(DATA_DIR, 'glm/compare_contrasts')
AVERAGED_DATA_DIR = os.path.join(DATA_DIR, 'averaged_data/compare_contrasts')
FIG_DIR = os.path.join(DATA_DIR, 'figures/compare_contrasts')
os.makedirs(AVERAGED_DATA_DIR, exist_ok=True)
os.makedirs(GLM_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def legendre_polynomials(n_tp, poly_order=2):
    # Make drift model regressors
    x = np.linspace(-1, 1, n_tp)
    poly = np.zeros((n_tp, poly_order + 1))
    for order in range(poly_order + 1):
        poly[:, order] = legendre(order)(x)
    return poly


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




def run_glm_manual(subj, hemi, regressor_cam, regressor_scene):
    nuisance = get_got_nuisance(subj)
    #camera regressor only
    regressor_cam_full = np.hstack([regressor_cam, nuisance])
    #scene regressor only
    regressor_scene_full = np.hstack([regressor_scene, nuisance])
    #full regressor
    regressor = np.hstack([regressor_cam, regressor_scene_full])
    #data matrix
    mask = load_mask(hemi)
    denoised_dir = os.path.join(FMRI_DATA_DIR, f'denoised/{subj}')
    data_fn = f'{subj}_task-GoT_space-fsaverage5_{hemi}_denoised.npy'
    data = np.load(os.path.join(denoised_dir, data_fn))[:, mask]
    ds = np.nan_to_num(zscore(data, axis=0))

    #baseline beta for scene
    beta_scene, ss_r_scene = np.linalg.lstsq(regressor_scene_full, ds, rcond=-1)[:2]
    diff_scene = ds - np.dot(regressor_scene_full, beta_scene)
    sigma_scene = np.sqrt(np.sum(diff_scene**2, axis=0) / (ds.shape[0] - regressor_scene_full.shape[1]))
    cov_scene = np.dot(regressor_scene_full.T, regressor_scene_full)
    inv_scene = np.linalg.inv(cov_scene)
    scene_baseline_beta = beta_scene[0, :] 
    se_scene = sigma_scene * np.sqrt(inv_scene[0, 0])
    baseline_t_scene = scene_baseline_beta / se_scene 

    #baseline beta for camera
    beta_cam, ss_r_cam = np.linalg.lstsq(regressor_cam_full, ds, rcond=-1)[:2]
    diff_cam = ds - np.dot(regressor_cam_full, beta_cam)
    sigma_cam = np.sqrt(np.sum(diff_cam**2, axis=0) / (ds.shape[0] - regressor_cam_full.shape[1]))
    cov_cam = np.dot(regressor_cam_full.T, regressor_cam_full)
    inv_cam = np.linalg.inv(cov_cam)
    cam_basline_beta = beta_cam[0, :] 
    se_cam = sigma_cam * np.sqrt(inv_cam[0, 0])
    baseline_t_cam = cam_basline_beta / se_cam 

    #regressor beta
    beta, ss_r = np.linalg.lstsq(regressor, ds, rcond=-1)[:2]
    diff = ds - np.dot(regressor, beta)
    sigma = np.sqrt(np.sum(diff**2, axis=0) / (ds.shape[0] - regressor_cam_full.shape[1]))
    cov = np.dot(regressor.T, regressor)
    inv = np.linalg.inv(cov)
    #    contrast_names = ['camera_cuts','scene_cuts','camera_vs_scene','scene_vs_camera','camera_cuts_only','scene_cuts_only']
    # contrast list
    contrasts = [
        # baseline_cam, baseline_scene, camera, scene
        [1, -1], #camera vs scene
        [-1, 1], #scene vs camera
        [1, 0], #camera only
        [0, 1] #scene only
    ]
    
    ts = [baseline_t_cam,baseline_t_scene] # Start the list with the baseline t-statistic
    betas = [cam_basline_beta,scene_baseline_beta] # Start the list with the baseline betas

    # Start the loop from the second contrast (index 1) which is face-all
    for contrast in contrasts: 
        R = np.concatenate([np.array(contrast), np.zeros((regressor.shape[1] - len(contrast), ))]).reshape((1, -1))

        mid = R @ inv @ R.T # (1xR) * (RxR) * (Rx1) -> scalar
        mid_val = float(mid.item()) # Get the scalar value
        se_contrast = sigma * np.sqrt(mid_val) 
        R_beta = np.dot(R, beta).ravel()
        t = R_beta / se_contrast
        
        ts.append(t)
        betas.append(R_beta) 

    return np.array(ts), np.array(betas)

def pipe_wrapper(subj, hemi, regressor_cam, regressor_scene):
    # For parallel processsing
    #would i have to 
    out_fn = os.path.join(GLM_DIR, f'{subj}_{hemi}.npz')

    betas, ts = run_glm_manual(subj, hemi, regressor_cam,regressor_scene)
    np.savez(out_fn, betas=betas, ts=ts)
    
def average_glm(contrast_names):
    groups = ['control', 'DP']
    control_subjects = get_got_subjects('control')
    dp_subjects = get_got_subjects('DP')
    hemis = ['hemi-L', 'hemi-R']

    for contrast_index, contrast in enumerate(contrast_names):
        for group in groups:
            subjects = get_got_subjects(group)
            group_data = []
            for subj in subjects:
                brain_data = []
                for hemi in hemis:
                    fn = f'{subj}_{hemi}.npz'
                    # data = np.load(os.path.join(GLM_DIR, fn))['ts']#[contrast_index, :]
                    # print(data.shape)
                    data = np.load(os.path.join(GLM_DIR, fn))['ts'][contrast_index, :] #would this be ok with how many contrasts there are?
                    brain_data.append(data)
                brain_array = np.concatenate(brain_data, axis=0) #originally axis=1
                group_data.append(brain_array)
            
            group_array = np.dstack(group_data)
            #print(group_array.shape)
            group_average = np.mean(group_array, axis=2)
            #print(group_average.shape)

            out_fn = f'{group}_{contrast}.npy'
            np.save(os.path.join(AVERAGED_DATA_DIR, out_fn), group_average)

def plot_brains(plot_data, titles, vmax=3, cbar_label='t', plot_cbar=True, plot_titles=False):
    vmin = -1 * vmax
    fig, axs = plt.subplots(nrows=1, ncols=len(plot_data), figsize=(12, 8))
    for i, title in enumerate(titles):
        ax = axs[i]
        img = brain_plot(plot_data[i], vmin=vmin, vmax=vmax, cmap='seismic')
        ax.imshow(img)
        ax.axis('off')
        if plot_titles == True:
            ax.set_title(title, fontsize=28)
    if plot_cbar == True:
        norm = plt.Normalize(vmin, vmax)
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap='seismic'),
            ax=axs,
            orientation='horizontal',
            shrink=0.7,
            label=f'{cbar_label}-value',
            )
        cbar.ax.tick_params(labelsize=16) # cbar tick fontsize
        cbar.ax.xaxis.label.set_fontsize(22) # cbar title fontsize

    return fig, axs
 


if __name__ == "__main__":
    hemis = ['hemi-L', 'hemi-R']
    subjects = get_got_subjects()
    CAM_REGRESSOR_FILE = 'camera_cuts_regressors.csv'
    SCENE_REGRESSOR_FILE = 'scene_cuts_regressors.csv'

    # Construct the full paths
    cam_regressors_fn = os.path.join(REGRESSORS_DIR, CAM_REGRESSOR_FILE)
    scene_regressors_fn = os.path.join(REGRESSORS_DIR, SCENE_REGRESSOR_FILE)

    # Load the data into the specific variables
    regressor_cam = pd.read_csv(cam_regressors_fn, index_col=0)
    regressor_scene = pd.read_csv(scene_regressors_fn, index_col=0)
        
    contrast_names = ['camera_cuts_baseline','scene_cuts_baseline','camera_vs_scene','scene_vs_camera','camera_cuts_vs_zero','scene_cuts_vs_zero']

    jobs = []

    for subj in subjects:
        for hemi in hemis:
            jobs.append(delayed(pipe_wrapper)(subj, hemi,regressor_cam, regressor_scene))
                
    with parallel_backend("loky", inner_max_num_threads=1):
        Parallel(n_jobs=4, verbose=2)(jobs)
    average_glm(contrast_names)
    # Set vars for reference
    groups = ['control', 'DP']
    categories = [
        'camera_cuts',
        'scene_cuts',
    ]
    # Plot perceptual features group average t-maps

    for i, contrast in enumerate(contrast_names):
        print("inside")
        plot_data = []
        for group in groups:
            fn = f'{group}_{contrast}.npy'
            group_data = np.load(os.path.join(AVERAGED_DATA_DIR, fn))
            contrast_data = group_data[0, :]
            plot_data.append(contrast_data)
        
        fig, axs = plot_brains(plot_data, groups, plot_titles=True)
        fig.suptitle(contrast, fontsize=32)
        plt.show()
        save_fn = os.path.join(FIG_DIR, f'{contrast}.png')
        fig.savefig(save_fn, bbox_inches='tight') # Save the figure to your designated directory
        plt.close(fig) # Close the figure to free memory
       