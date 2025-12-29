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

SCRIPTS_DIR = '/mnt/labdata/got_project/test/brain_plot_glm_code'#os.path.expanduser('~/Documents/got_project')
sys.path.append(SCRIPTS_DIR)
from utils import get_got_subjects, load_mask

LOW_LVL_DIR = '/mnt/labdata/got_project/test/low_level_data/new2sec_lowlvl' #correct
DNN_DIR = '/mnt/labdata/got_project/test/alexnet' #correct
PROJ_DIR = '/mnt/labdata/got_project' #correct
FMRI_DATA_DIR = os.path.join(PROJ_DIR, 'data') #correct
DATA_DIR = os.path.join(PROJ_DIR, 'test/brain_plot_data_output') #correct
REGRESSORS_DIR = os.path.join(DATA_DIR, 'regressor_output') #correct
REGRESSORS_DIR_INBETWEEN = os.path.join(DATA_DIR, 'regressor_output_inbetween') #correct



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
    # Load audio and hsv and motion energy regressors
    low_lvl_dir = os.path.join(LOW_LVL_DIR)
    dnn_dir = os.path.join(DNN_DIR)
    dnn_fn = f'visual_features_pca1_new.npy'
    dnn_data = np.load(os.path.join(dnn_dir, dnn_fn), allow_pickle=True)[:-1]
    #audio, hsv, motion energy
    audio_fn = f'audio_pitch_output.csv'
    hsv_fn = f'hsv_output.csv'
    motion_fn = f'motion_energy_output.csv'
    audio_data = pd.read_csv(os.path.join(low_lvl_dir, audio_fn))[:-1]
    hsv_data = pd.read_csv(os.path.join(low_lvl_dir, hsv_fn))[:-1]
    motion_data = pd.read_csv(os.path.join(low_lvl_dir, motion_fn))[:-1]
    audio_regressors = audio_data[[ 'Average_Pitch_Hz','Average_Amplitude']].values
    hsv_regressors = hsv_data[[ 'Average_H','Average_S','Average_V']].values
    motion_regressors = motion_data[['Average_Motion_Energy']].values
    # Replace first nan in 'framewise_displacement with 0' for alignment
    raw_confounds = np.nan_to_num(raw_confounds_df[columns].values)
    raw_confounds_with_low_level = np.concatenate((dnn_data,audio_regressors, hsv_regressors, motion_regressors,raw_confounds), axis=1) 
    # Zscore confounds, fmriprep recommendation
    #confounds = np.nan_to_num(zscore(raw_confounds_with_low_level, axis=0)) #uncomment for low level
    confounds = np.nan_to_num(zscore(raw_confounds, axis=0)) #no low level
    #print("confounds model: " + str(confounds.shape))
    
    # Add drift regressors
    poly = legendre_polynomials(n_tp=confounds.shape[0])
    #print("drift model: " + str(poly.shape))
    return np.concatenate((confounds, poly), axis=1) 




def run_glm_manual(subj, hemi, regressor_norm, regressor_inbetween):
    nuisance = get_got_nuisance(subj)
    #print("nuisance shape: " + str(nuisance.shape))
    #camera regressor only
    regressor_norm_full = np.hstack([regressor_norm, nuisance])
    #print("regressor cam full shape: " + str(regressor_cam_full.shape))
    #scene regressor only
    regressor_inbetween_full = np.hstack([regressor_inbetween, nuisance])
    #print("regressor scene full shape: " + str(regressor_scene_full.shape))

    regressor = np.hstack([regressor_norm, regressor_inbetween_full])
    #print("full regressor shape: " + str(regressor.shape))  
    #data matrix
    mask = load_mask(hemi)
    denoised_dir = os.path.join(FMRI_DATA_DIR, f'denoised/{subj}')
    data_fn = f'{subj}_task-GoT_space-fsaverage5_{hemi}_denoised.npy'
    data = np.load(os.path.join(denoised_dir, data_fn))[:, mask]
    ds = np.nan_to_num(zscore(data, axis=0))

    #baseline beta for scene
    beta_inbetween, ss_r_inbetween = np.linalg.lstsq(regressor_inbetween_full, ds, rcond=-1)[:2]
    diff_inbetween = ds - np.dot(regressor_inbetween_full, beta_inbetween)
    sigma_inbetween = np.sqrt(np.sum(diff_inbetween**2, axis=0) / (ds.shape[0] - regressor_inbetween_full.shape[1]))
    cov_inbetween = np.dot(regressor_inbetween_full.T, regressor_inbetween_full)
    inv_inbetween = np.linalg.inv(cov_inbetween)
    inbetween_baseline_beta = beta_inbetween[0, :] 
    se_inbetween = sigma_inbetween * np.sqrt(inv_inbetween[0, 0])
    baseline_t_scene = inbetween_baseline_beta / se_inbetween 
    #baseline beta for camera
    beta_norm, ss_r_norm = np.linalg.lstsq(regressor_norm_full, ds, rcond=-1)[:2]
    diff_norm = ds - np.dot(regressor_norm_full, beta_norm)
    sigma_norm = np.sqrt(np.sum(diff_norm**2, axis=0) / (ds.shape[0] - regressor_norm_full.shape[1]))
    cov_norm = np.dot(regressor_norm_full.T, regressor_norm_full)
    inv_norm = np.linalg.inv(cov_norm)
    norm_baseline_beta = beta_norm[0, :] 
    se_norm = sigma_norm * np.sqrt(inv_norm[0, 0])
    baseline_t_norm = norm_baseline_beta / se_norm 


    #regressor beta
    beta, ss_r = np.linalg.lstsq(regressor, ds, rcond=-1)[:2]
    diff = ds - np.dot(regressor, beta)
    sigma = np.sqrt(np.sum(diff**2, axis=0) / (ds.shape[0] - regressor_norm_full.shape[1]))
    cov = np.dot(regressor.T, regressor)
    inv = np.linalg.inv(cov)
    #    contrast_names = ['camera_cuts','scene_cuts','camera_vs_scene','scene_vs_camera','camera_cuts_only','scene_cuts_only']
    # contrast list
    contrasts = [

        [1, -1],  # norm_vs_inbetween
        [-1, 1]
    ]

    ts = [baseline_t_norm,baseline_t_scene] # Start the list with the baseline t-statistic
    betas = [norm_baseline_beta,inbetween_baseline_beta] # Start the list with the baseline betas
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


def pipe_wrapper(glm_dir, subj, hemi, regressor_norm, regressor_inbetween):
    # For parallel processsing
    #would i have to 
    out_fn = os.path.join(glm_dir, f'{subj}_{hemi}.npz')

    betas, ts = run_glm_manual(subj, hemi, regressor_norm,regressor_inbetween)
    np.savez(out_fn, betas=betas, ts=ts)
    
def average_glm(data_dir, glm_dir, contrast_names):
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
                    data = np.load(os.path.join(glm_dir, fn))['ts'][contrast_index, :] #would this be ok with how many contrasts there are?
                    brain_data.append(data)
                brain_array = np.concatenate(brain_data, axis=0) #originally axis=1
                group_data.append(brain_array)
            
            group_array = np.dstack(group_data)
            #print(group_array.shape)
            group_average = np.mean(group_array, axis=2)
            #print(group_average.shape)

            out_fn = f'{group}_{contrast}.npy'
            np.save(os.path.join(data_dir, out_fn), group_average)
#'''
def plot_brains(plot_data, titles, vmax=3, cbar_label='t', plot_cbar=True, plot_titles=False, ax=None):
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
#'''
'''
def plot_brains(plot_data, titles, vmax=3, cbar_label='t', plot_cbar=True, plot_titles=False, axs=None):
    vmin = -1 * vmax
    
    # If no axes are provided, create a standalone figure as before
    if axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=len(plot_data), figsize=(12, 8))
    else:
        fig = axs[0].get_figure()

    for i, title in enumerate(titles):
        ax = axs[i]
        img = brain_plot(plot_data[i], vmin=vmin, vmax=vmax, cmap='seismic')
        ax.imshow(img)
        ax.axis('off')
        if plot_titles:
            ax.set_title(title, fontsize=20) # Slightly smaller for grid layout

    # if plot_cbar:
    #     norm = plt.Normalize(vmin, vmax)
    #     # We attach the colorbar to the specific row's axes
    #     cbar = fig.colorbar(
    #         plt.cm.ScalarMappable(norm=norm, cmap='seismic'),
    #         ax=axs,
    #         orientation='horizontal',
    #         shrink=0.7,
    #         label=f'{cbar_label}-value',
    #         pad=0.02
    #     )
    #     cbar.ax.tick_params(labelsize=12)
    #     cbar.ax.xaxis.label.set_fontsize(14)

    return fig, axs
'''
def run_regressors(base_type='scene', shift_type='base', amplitude_type='bool', regressor_type='regressor_raw', additional_data=''):
    file_change = f'{base_type}_{shift_type}_{amplitude_type}_{regressor_type}{additional_data}'
    GLM_DIR = os.path.join(DATA_DIR, f'glm/{file_change}')
    AVERAGED_DATA_DIR = os.path.join(DATA_DIR, f'averaged_data/{file_change}')
    FIG_DIR = os.path.join(DATA_DIR, f'figures/{file_change}')
    os.makedirs(AVERAGED_DATA_DIR, exist_ok=True)
    os.makedirs(GLM_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    jobs = []
    if regressor_type == 'regressor_raw':
        NORM_REGRESSOR_CONVOLVED_FILE = f'{base_type}_{shift_type}_{amplitude_type}_{regressor_type}.csv'
        INBETWEEN_REGRESSOR_CONVOLVED_FILE = f'{base_type}_inbetween_{shift_type}_{amplitude_type}_{regressor_type}.csv'

        # Construct the full paths
        norm_regressors_fn = os.path.join(REGRESSORS_DIR, f'{base_type}_{shift_type}',NORM_REGRESSOR_CONVOLVED_FILE)
        inbetween_regressors_fn = os.path.join(REGRESSORS_DIR_INBETWEEN, f'{base_type}_inbetween_{shift_type}',INBETWEEN_REGRESSOR_CONVOLVED_FILE)
        # Load the data into the specific variables for csv raw
        regressor_norm = pd.read_csv(norm_regressors_fn, index_col=0)
        regressor_inbetween = pd.read_csv(inbetween_regressors_fn, index_col=0)
    elif regressor_type == 'convolved':
    #load the data into the specific variables for npy convolved
        NORM_REGRESSOR_CONVOLVED_FILE = f'{base_type}_{shift_type}_{amplitude_type}_{regressor_type}.npy'
        INBETWEEN_REGRESSOR_CONVOLVED_FILE = f'{base_type}_inbetween_{shift_type}_{amplitude_type}_{regressor_type}.npy'

        # Construct the full paths
        norm_regressors_fn = os.path.join(REGRESSORS_DIR, f'{base_type}_{shift_type}',NORM_REGRESSOR_CONVOLVED_FILE)
        inbetween_regressors_fn = os.path.join(REGRESSORS_DIR_INBETWEEN, f'{base_type}_inbetween_{shift_type}',INBETWEEN_REGRESSOR_CONVOLVED_FILE)
        regressor_norm = np.load(norm_regressors_fn, allow_pickle=True).reshape(-1, 1)
        regressor_inbetween = np.load(inbetween_regressors_fn, allow_pickle=True).reshape(-1, 1)

    for subj in subjects:
        for hemi in hemis:
            jobs.append(delayed(pipe_wrapper)(GLM_DIR, subj, hemi,regressor_norm, regressor_inbetween))
    
    # --- 4. Run Parallel Processing ---
    with parallel_backend("loky", inner_max_num_threads=1):
        Parallel(n_jobs=4, verbose=2)(jobs)
    plot_all_contrasts(AVERAGED_DATA_DIR, FIG_DIR, GLM_DIR, file_change)

def plot_all_contrasts(data_dir, fig_dir, glm_dir, file_change):
    average_glm(data_dir, glm_dir, CONTRASTS)

    # Set vars for reference
    groups = ['control', 'DP']
    categories = [
        'camera_cuts',
        'scene_cuts',
        'medium_length'
    ]
    '''
    num_contrasts = len(CONTRASTS)
    num_groups = len(groups)

    # Adjust figsize based on number of contrasts (width, height)
    fig, big_axs = plt.subplots(nrows=num_contrasts, ncols=num_groups, 
                                figsize=(14, 5 * num_contrasts))

    # Global title for the whole file
    fig.suptitle(file_change.replace('_', ' '), fontsize=32, y=0.98)

    for i, contrast in enumerate(CONTRASTS):
        # Determine the axes for the current row
        # If only 1 contrast, big_axs is 1D; if multiple, it's 2D
        current_row_axs = big_axs[i] if num_contrasts > 1 else big_axs
        
        plot_data = []
        for group in groups:
            fn = f'{group}_{contrast}.npy'
            group_data = np.load(os.path.join(data_dir, fn))
            plot_data.append(group_data[0, :])
        
        # Call the modified function passing the specific row axes
        plot_brains(plot_data, groups, plot_titles=(i==0), axs=current_row_axs, plot_cbar=True)
        
        # Add the Contrast Name to the side of the row
        current_row_axs[0].annotate(contrast, xy=(-0.1, 0.5), xycoords='axes fraction',
                                    rotation=90, ha='center', va='center', 
                                    fontsize=24, fontweight='bold')
    cax = fig.add_axes([0.3, 0.94, 0.4, 0.02]) 
    vmax = 3
    norm = plt.Normalize(-vmax, vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='seismic')
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('t-value', fontsize=20, labelpad=10)
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for suptitle
    save_fn = os.path.join(fig_dir, f'{file_change}_all_contrasts.png')
    fig.savefig(save_fn, bbox_inches='tight', dpi=150)
    plt.close()
    '''
    for i, contrast in enumerate(CONTRASTS):
        #print("inside")
        max_val = 0
        plot_data = []
        for group in groups:
            fn = f'{group}_{contrast}.npy'
            group_data = np.load(os.path.join(data_dir, fn))
            contrast_data = group_data[0, :]
            plot_data.append(contrast_data)
            max_val = max(max_val, max(contrast_data))

        fig, axs = plot_brains(plot_data, groups, vmax=max_val, plot_titles=True)
        fig.suptitle(contrast, fontsize=32)
        plt.show()
        save_fn = os.path.join(fig_dir, f'{contrast}_{file_change}.png')
        fig.savefig(save_fn, bbox_inches='tight') # Save the figure to your designated directory
        plt.close(fig) # Close the figure to free memory
CONTRASTS = [
    'norm_baseline',
    'inbetween_baseline',
    'norm_vs_inbetween',
    'inbetween_vs_norm'
]
#CHANGE THIS
# shift_type = 'base'
# amplitude_type = 'bool'
# regressor_type = 'convolved'
# additional_data = 'combined'
# file_change = f'{shift_type}_{amplitude_type}_{regressor_type}_{additional_data}'
# GLM_DIR = os.path.join(DATA_DIR, f'glm/{file_change}')
# AVERAGED_DATA_DIR = os.path.join(DATA_DIR, f'averaged_data/{file_change}')
# FIG_DIR = os.path.join(DATA_DIR, f'figures/{file_change}')
# os.makedirs(AVERAGED_DATA_DIR, exist_ok=True)
# os.makedirs(GLM_DIR, exist_ok=True)
# os.makedirs(FIG_DIR, exist_ok=True)
#use this as key to create different runs with different regressors
hemis = ['hemi-L', 'hemi-R']
subjects = get_got_subjects()
BASE_NAMES = ['scene_cuts', 'medium_cuts', 'scene_cuts_inbetween', 'medium_cuts_inbetween']
SHIFT_TYPES = ['base', 'shift_4s'] # 'shift_2s',
REGRESSOR_TYPES = ['regressor_raw', 'convolved']
AMPLITUDE_TYPES = ['bool', 'amp']
#INBETWEEN = ['normal', 'inbetween']
#CONTRAST_INDEX_MAP = {name: index for index, name in enumerate(CONTRASTS)}
if __name__ == "__main__":
    #run bool only
    #run convolved base 
    #run shifted raw
    #plot_all_contrasts()
    #additional data = '_'
    run_regressors(base_type='scene_cuts', shift_type='shift_4s', amplitude_type='bool', regressor_type='regressor_raw',additional_data='_inbetween_nolowlevel')
    run_regressors(base_type='scene_cuts', shift_type='base', amplitude_type='bool', regressor_type='convolved',additional_data='_inbetween_nolowlevel')
    run_regressors(base_type='medium_length', shift_type='shift_4s', amplitude_type='bool', regressor_type='regressor_raw',additional_data='_inbetween_nolowlevel')
    run_regressors(base_type='medium_length', shift_type='base', amplitude_type='bool', regressor_type='convolved',additional_data='_inbetween_nolowlevel')  
    #plot_all_contrasts()
    #run_regressors()
    #UPDATE contrast names based on the new regressors    


       