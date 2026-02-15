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
from scipy import stats
import matplotlib.pyplot as plt
from brainplotlib import brain_plot

#change this to your scripts directory
SCRIPTS_DIR = '/mnt/labdata/got_project/test/brain_plot_glm_code'#os.path.expanduser('~/Documents/got_project')
sys.path.append(SCRIPTS_DIR)
#this is for utils file, which should be in scripts directory
from utils import get_got_subjects, load_mask 

#i used a lot of different directories and brought them in via hard coding
LOW_LVL_DIR = '/mnt/labdata/got_project/test/low_level_data/new2sec_lowlvl' 
DNN_DIR = '/mnt/labdata/got_project/test/alexnet' 
PROJ_DIR = '/mnt/labdata/got_project' 
FMRI_DATA_DIR = os.path.join(PROJ_DIR, 'data') 
DATA_DIR = os.path.join(PROJ_DIR, 'test/brain_plot_data_output') 
REGRESSORS_DIR = os.path.join(DATA_DIR, 'regressor_output') 
REGRESSORS_DIR_INBETWEEN = os.path.join(DATA_DIR, 'regressor_output_inbetween') 



def legendre_polynomials(n_tp, poly_order=2):
    # Make drift model regressors to account for low frequency noise
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
    confounds = np.nan_to_num(zscore(raw_confounds_with_low_level, axis=0)) #uncomment for low level
    #confounds = np.nan_to_num(zscore(raw_confounds, axis=0)) #uncomment for no low level
    #print("confounds model: " + str(confounds.shape))
    
    # Add drift regressors behind counfounds
    # returned shape: (n_timepoints, n_confounds + n_drift_terms)
    poly = legendre_polynomials(n_tp=confounds.shape[0])
    #print("drift model: " + str(poly.shape))
    return np.concatenate((confounds, poly), axis=1) 


def multiple_comparisons_fdr(t_stats, df, q=0.05):
    """
    Manually computes FDR (Benjamini-Hochberg) for a map of t-statistics.
    """
    # 1. Convert T-stats to P-values (two-tailed)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))
    
    # Flatten p-values to handle the vertex array
    p_flat = p_values.flatten()
    m = len(p_flat)
    
    # 2. Sort p-values and keep track of original indices
    sort_indices = np.argsort(p_flat)
    p_sorted = p_flat[sort_indices]
    
    # 3. Create the BH thresholds: (i/m) * q
    # We use i+1 because ranks start at 1
    ranks = np.arange(1, m + 1)
    thresholds = (ranks / m) * q
    
    # 4. Find the largest k where p_i <= threshold_i
    below_threshold = p_sorted <= thresholds
    
    if not np.any(below_threshold):
        return np.zeros_like(p_values, dtype=bool) # No significant voxels
    
    # Find the largest index that satisfies the condition
    max_k_idx = np.where(below_threshold)[0][-1]
    p_threshold = p_sorted[max_k_idx]
    
    # 5. Everything less than or equal to the p-value at max_k is significant
    significant_flat = p_flat <= p_threshold
    
    return significant_flat.reshape(p_values.shape)
def run_glm_manual(subj, hemi, regressor_norm, regressor_inbetween):
    '''
    what i do is very weird, i first get the baseline t stat for each regressor by running a glm with just that regressor and nuisance
    then i run the glm with both regressors and nuisance, and compute contrasts between the two
    this way i can get the baseline activation for each regressor as well as contrasts between them
    technically you can just make with no baseline regressors but i keep them in
    1. get nuisance from helper function 
    2. create full regressor matrices for each regressor + nuisance
    3. run glm with each regressor + nuisance to get baseline betas and t stats
    4. create full regressor matrix with both regressors + nuisance
    5. run glm with both regressors + nuisance
    6. compute contrasts between the two regressors
    7. return t stats and betas for baseline and contrasts and save to file
    8. plot group averages and save plots
    '''

    
    # print(f"--- DEBUG: {subj} {hemi} ---")
    # print(f"Norm Sum: {norm_sum} | Inbetween Sum: {inbet_sum}")
    # print(f"Inputs are identical: {is_identical}")

    # if is_identical:
    #     print("CRITICAL: Inputs are identical before GLM starts!")
    nuisance = get_got_nuisance(subj)
    #print("nuisance shape: " + str(nuisance.shape))
    regressor_norm_full = np.hstack([regressor_norm, nuisance])
    regressor_inbetween_full = np.hstack([regressor_inbetween, nuisance])
    regressor = np.hstack([regressor_norm, regressor_inbetween, nuisance])
    correlation = np.corrcoef(regressor[:, 0], regressor[:, 1])[0, 1]
    # print(f"Correlation between Norm and Inbetween columns: {correlation:.4f}")
    # print(f"Full Regressor Shape: {regressor.shape}")
    # 2. Combined model (Corrected)
    # #camera regressor only
    # regressor_norm_full = np.hstack([regressor_norm, nuisance])
    # #print("regressor cam full shape: " + str(regressor_cam_full.shape))
    # #scene regressor only
    # regressor_inbetween_full = np.hstack([regressor_inbetween, nuisance])
    # #print("regressor scene full shape: " + str(regressor_scene_full.shape))

    # #full regressor with both, norm first then inbetween
    # regressor = np.hstack([regressor_norm, regressor_inbetween_full])
    #print("full regressor shape: " + str(regressor.shape))  
    #data matrix
    mask = load_mask(hemi)
    #load denoised data that was processed from data folder
    denoised_dir = os.path.join(FMRI_DATA_DIR, f'denoised/{subj}')
    data_fn = f'{subj}_task-GoT_space-fsaverage5_{hemi}_denoised.npy'
    data = np.load(os.path.join(denoised_dir, data_fn))[:, mask]
    ds = np.nan_to_num(zscore(data, axis=0))

    #baseline beta for inbetween
    beta_inbetween, ss_r_inbetween = np.linalg.lstsq(regressor_inbetween_full, ds, rcond=-1)[:2]
    diff_inbetween = ds - np.dot(regressor_inbetween_full, beta_inbetween)
    sigma_inbetween = np.sqrt(np.sum(diff_inbetween**2, axis=0) / (ds.shape[0] - regressor_inbetween_full.shape[1]))
    cov_inbetween = np.dot(regressor_inbetween_full.T, regressor_inbetween_full)
    inv_inbetween = np.linalg.inv(cov_inbetween)
    inbetween_baseline_beta = beta_inbetween[0, :] 
    se_inbetween = sigma_inbetween * np.sqrt(inv_inbetween[0, 0])
    baseline_t_inbetween = inbetween_baseline_beta / se_inbetween 
    #baseline beta for norm
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

        [1, -1],  # norm_vs_inbetween these will look the same since they are flipped
        [-1, 1]
    ]

    ts = [baseline_t_norm,baseline_t_inbetween] # Start the list with the baseline t-statistic
    betas = [norm_baseline_beta,inbetween_baseline_beta] # Start the list with the baseline betas
    # Start the loop from the second contrast (index 1) 
    for contrast in contrasts: 
        R = np.concatenate([np.array(contrast), np.zeros((regressor.shape[1] - len(contrast), ))]).reshape((1, -1))

        mid = R @ inv @ R.T # (1xR) * (RxR) * (Rx1) -> scalar
        mid_val = float(mid.item()) # Get the scalar value
        se_contrast = sigma * np.sqrt(mid_val) 
        R_beta = np.dot(R, beta).ravel()
        t = R_beta / se_contrast
        
        ts.append(t)
        betas.append(R_beta) 
    t_corr = np.corrcoef(baseline_t_norm, baseline_t_inbetween)[0, 1]
    
    #multiple comparisons correction
    df_baseline = ds.shape[0] - regressor_norm_full.shape[1]
    df_combined = ds.shape[0] - regressor.shape[1]
    sig_masks = []
    for i, t_map in enumerate(ts):
        # Use the correct df: first two are baseline, rest are contrasts
        current_df = df_baseline if i < 2 else df_combined
        
        # Call your manual_fdr function
        mask = multiple_comparisons_fdr(t_map, current_df, q=0.05)
        sig_masks.append(mask)

    return np.array(betas), np.array(ts), np.array(sig_masks)
    #return np.array(ts), np.array(betas)


def pipe_wrapper(glm_dir, subj, hemi, regressor_norm, regressor_inbetween):
 
    out_fn = os.path.join(glm_dir, f'{subj}_{hemi}.npz')

    betas, ts, sig_masks = run_glm_manual(subj, hemi, regressor_norm,regressor_inbetween)
    np.savez(out_fn, betas=betas, ts=ts, sig_masks=sig_masks)

def calculate_group_fdr(group_betas, q=0.05):
    '''
    group_betas: array of shape (n_vertices, n_subjects)
    '''
    n_subjects = group_betas.shape[1]
    df_group = n_subjects - 1  # Degrees of freedom for group level
    
    # Calculate Group T-stat: Mean / SEM
    group_mean = np.mean(group_betas, axis=1)
    group_std = np.std(group_betas, axis=1, ddof=1)
    group_sem = group_std / np.sqrt(n_subjects)
    
    group_t = group_mean / group_sem
    
    # Apply your manual FDR function
    group_mask = multiple_comparisons_fdr(group_t, df_group, q=q)
    
    return group_t, group_mask

def average_glm_fdr(data_dir, glm_dir, contrast_names):
    groups = ['control', 'DP']
    control_subjects = get_got_subjects('control')
    dp_subjects = get_got_subjects('DP')
    hemis = ['hemi-L', 'hemi-R']
    for contrast_index, contrast in enumerate(contrast_names):
        for group in groups:
            subjects = get_got_subjects(group)
            all_betas = []
            
            for subj in subjects:
                subj_data = []
                for hemi in hemis:
                    # Load the betas you saved in the npz
                    data = np.load(os.path.join(glm_dir, f'{subj}_{hemi}.npz'))['betas'][contrast_index, :]
                    subj_data.append(data)
                
                # Combine L and R hemispheres for the subject
                all_betas.append(np.concatenate(subj_data)) 

            # Create matrix: (vertices, subjects)
            group_beta_matrix = np.stack(all_betas, axis=1)
            
            # Run the Group Stats
            group_t, group_sig_mask = calculate_group_fdr(group_beta_matrix)
            
            # Apply the mask to the T-map for visualization (zeros out non-sig)
            thresholded_t_map = group_t * group_sig_mask
            plot_ready_map = thresholded_t_map.copy()
            plot_ready_map[plot_ready_map == 0] = np.nan
            # Save results
            np.save(os.path.join(data_dir, f'{group}_{contrast}_tstat.npy'), group_t)
            np.save(os.path.join(data_dir, f'{group}_{contrast}_tstat_threshold.npy'), plot_ready_map)
            np.save(os.path.join(data_dir, f'{group}_{contrast}_fdr_mask.npy'), group_sig_mask)
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
                    #data = np.load(os.path.join(glm_dir, fn))['ts'][contrast_index, :] 
                    data = np.load(os.path.join(glm_dir, fn))['sig_masks'][contrast_index, :] 
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
def plot_brains_grid(plot_rows, col_titles, row_titles, vmax=3, cbar_label='t'):
    vmin = -1 * vmax
    num_rows = len(plot_rows)
    num_cols = len(col_titles)
    
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(16, 10))
    
    for r in range(num_rows):
        for c in range(num_cols):
            ax = axs[r, c]
            # Use seismic for T-stats (row 0), maybe something else for mask (row 1)?
            # Here we stick to seismic as requested.
            img = brain_plot(plot_rows[r][c], vmin=vmin, vmax=vmax, cmap='seismic')
            ax.imshow(img)
            ax.axis('off')
            
            # Titles only on the top row
            if r == 0:
                ax.set_title(col_titles[c], fontsize=24)
            
            # Labels for the rows on the far left
            if c == 0:
                ax.text(-0.1, 0.5, row_titles[r], transform=ax.transAxes, 
                        rotation=90, va='center', ha='right', fontsize=20, fontweight='bold')

    if True: # Plotting one shared colorbar
        norm = plt.Normalize(vmin, vmax)
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap='seismic'),
            ax=axs, orientation='horizontal', shrink=0.5, pad=0.05,
            label=f'{cbar_label}-value'
        )
        cbar.ax.tick_params(labelsize=14)
    
    return fig, axs
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
def plot_brains_all contrasts(plot_data, titles, vmax=3, cbar_label='t', plot_cbar=True, plot_titles=False, axs=None):
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
def run_regressors(base_type='scene', shift_type='base', amplitude_type='bool', regressor_type='raw_regressor', additional_data=''):
    '''
    I have created different files for each regressor for convovlved and raw, unconvolved and shifted
    This function will load the appropriate regressor files based on the input parameters
    '''
    #this file_change key is useful for creating different directories for different regressor types
    file_change = f'{base_type}_{shift_type}_{amplitude_type}_{regressor_type}{additional_data}'
    #store all created GLMs
    GLM_DIR = os.path.join(DATA_DIR, f'glm_2026/{file_change}')
    AVERAGED_DATA_DIR = os.path.join(DATA_DIR, f'averaged_data_2026/{file_change}')
    #store the created figures
    FIG_DIR = os.path.join(DATA_DIR, f'figures_2026/{file_change}')
    os.makedirs(AVERAGED_DATA_DIR, exist_ok=True)
    os.makedirs(GLM_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    jobs = []
    NORM_REGRESSOR_CONVOLVED_FILE = f'{base_type}_{shift_type}_{amplitude_type}_{regressor_type}.csv'
    INBETWEEN_REGRESSOR_CONVOLVED_FILE = f'{base_type}_inbetween_{shift_type}_{amplitude_type}_{regressor_type}.csv'

    # Construct the full paths
    norm_regressors_fn = os.path.join(REGRESSORS_DIR, f'{base_type}_{shift_type}',NORM_REGRESSOR_CONVOLVED_FILE)
    inbetween_regressors_fn = os.path.join(REGRESSORS_DIR_INBETWEEN, f'{base_type}_inbetween_{shift_type}',INBETWEEN_REGRESSOR_CONVOLVED_FILE)
    # print("norm regressor file: " + norm_regressors_fn)
    # print("inbetween regressor file: " + inbetween_regressors_fn)
    # Load the data into the specific variables for csv raw
    regressor_norm = pd.read_csv(norm_regressors_fn) #index_col=0
    regressor_inbetween = pd.read_csv(inbetween_regressors_fn)


    #making jobs for parallel processing
    for subj in subjects:
        for hemi in hemis:
            jobs.append(delayed(pipe_wrapper)(GLM_DIR, subj, hemi,regressor_norm, regressor_inbetween))
    
    # --- 4. Run Parallel Processing ---
    with parallel_backend("loky", inner_max_num_threads=1):
        Parallel(n_jobs=4, verbose=2)(jobs)
    plot_all_contrasts_fdr(AVERAGED_DATA_DIR, FIG_DIR, GLM_DIR, file_change)
def plot_all_contrasts(data_dir, fig_dir, glm_dir, file_change):
    average_glm(data_dir, glm_dir, CONTRASTS)

    # Set vars for reference
    groups = ['control', 'DP']
    categories = [
        'scene_cuts',
        'medium_length'
    ]

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
            #print("max val: " + str(max_val))
        fig, axs = plot_brains(plot_data, groups, vmax=max_val, plot_titles=True)
        fig.suptitle(contrast, fontsize=32)
        plt.show()
        save_fn = os.path.join(fig_dir, f'{contrast}_{file_change}_fdr.png')
        fig.savefig(save_fn, bbox_inches='tight') # Save the figure to your designated directory
        plt.close(fig) # Close the figure to free memory

def plot_all_contrasts_fdr(data_dir, fig_dir, glm_dir, file_change):
    # Ensure data is generated
    average_glm_fdr(data_dir, glm_dir, CONTRASTS)

    groups = ['control', 'DP']
    row_labels = ['Raw T-Stat', 'T-Thresholded','FDR Mask']

    for contrast in CONTRASTS:
        t_row = []
        mask_row = []
        t_threshold_row=[]
        max_t = 0
        
        for group in groups:
            # Load T-stat
            t_fn = f'{group}_{contrast}_tstat.npy'
            t_data = np.load(os.path.join(data_dir, t_fn))
            t_row.append(t_data)
            t_threshold_fn = f'{group}_{contrast}_tstat_threshold.npy'
            t_threshold_data = np.load(os.path.join(data_dir, t_threshold_fn))
            t_threshold_row.append(t_threshold_data)
            
            # Load Mask
            m_fn = f'{group}_{contrast}_fdr_mask.npy'
            m_data = np.load(os.path.join(data_dir, m_fn))
            mask_row.append(m_data)
            
            # Update vmax based on actual T-stats, not the binary mask
            max_t = max(max_t, np.percentile(np.abs(t_data), 98)) 

        # Combine into rows
        plot_data = [t_row, t_threshold_row, mask_row]
        
        fig, axs = plot_brains_grid(
            plot_rows=plot_data, 
            col_titles=groups, 
            row_titles=row_labels, 
            vmax=max_t, 
            cbar_label='t'
        )
        
        fig.suptitle(f'Contrast: {contrast}', fontsize=32, y=0.95)
        
        save_fn = os.path.join(fig_dir, f'{contrast}_{file_change}_grid.png')
        fig.savefig(save_fn, bbox_inches='tight')
        plt.show()
        plt.close(fig)

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
BASE_NAMES = ['scene_cuts', 'medium_length', 'scene_cuts_inbetween', 'medium_length_inbetween']
SHIFT_TYPES = ['base', 'shift_4s'] # 'shift_2s',
REGRESSOR_TYPES = ['raw_regressor', 'convolved']
AMPLITUDE_TYPES = ['bool', 'amp']
#INBETWEEN = ['normal', 'inbetween']
#CONTRAST_INDEX_MAP = {name: index for index, name in enumerate(CONTRASTS)}
if __name__ == "__main__":
    #uncomment to run different regressors

    #2/15/2026 only ran fdr corrected ones (make sure to change code to not fdr functions) and also only ran convolved data, no low level
    #run_regressors(base_type='scene_cuts', shift_type='shift_4s', amplitude_type='bool', regressor_type='raw_regressor',additional_data='_inbetween_fdr')
    run_regressors(base_type='scene_cuts', shift_type='base', amplitude_type='bool', regressor_type='convolved',additional_data='_inbetween_fdr')
    #run_regressors(base_type='medium_length', shift_type='shift_4s', amplitude_type='bool', regressor_type='raw_regressor',additional_data='_inbetween_fdr')
    run_regressors(base_type='medium_length', shift_type='base', amplitude_type='bool', regressor_type='convolved',additional_data='_inbetween_fdr')

    #no low level you have to uncomment something in confounds
    #run_regressors(base_type='scene_cuts', shift_type='shift_4s', amplitude_type='bool', regressor_type='raw_regressor',additional_data='_inbetween_nolowlevel')
    # run_regressors(base_type='scene_cuts', shift_type='base', amplitude_type='bool', regressor_type='convolved',additional_data='_inbetween_nolowlevel')
    # run_regressors(base_type='medium_length', shift_type='base', amplitude_type='bool', regressor_type='convolved',additional_data='_inbetween_nolowlevel')
    # run_regressors(base_type='medium_length', shift_type='shift_4s', amplitude_type='bool', regressor_type='raw_regressor',additional_data='_inbetween_nolowlevel')

    #UPDATE contrast names based on the new regressors    


        #i don't know how else to do it better lol but if it is a raw regressor load csv 
    # if regressor_type == 'raw_regressor':
    #     NORM_REGRESSOR_CONVOLVED_FILE = f'{base_type}_{shift_type}_{amplitude_type}_{regressor_type}.csv'
    #     INBETWEEN_REGRESSOR_CONVOLVED_FILE = f'{base_type}_inbetween_{shift_type}_{amplitude_type}_{regressor_type}.csv'

    #     # Construct the full paths
    #     norm_regressors_fn = os.path.join(REGRESSORS_DIR, f'{base_type}_{shift_type}',NORM_REGRESSOR_CONVOLVED_FILE)
    #     inbetween_regressors_fn = os.path.join(REGRESSORS_DIR_INBETWEEN, f'{base_type}_inbetween_{shift_type}',INBETWEEN_REGRESSOR_CONVOLVED_FILE)
    #     # print("norm regressor file: " + norm_regressors_fn)
    #     # print("inbetween regressor file: " + inbetween_regressors_fn)
    #     # Load the data into the specific variables for csv raw
    #     regressor_norm = pd.read_csv(norm_regressors_fn) #index_col=0
    #     regressor_inbetween = pd.read_csv(inbetween_regressors_fn)

    #     # print(f"Norm CSV Columns: {regressor_norm.columns.tolist()}")
    #     # print(f"Inbetween CSV Columns: {regressor_inbetween.columns.tolist()}")
    # #or load npy for convolved
    # elif regressor_type == 'convolved':
    # #load the data into the specific variables for npy convolved
    #     NORM_REGRESSOR_CONVOLVED_FILE = f'{base_type}_{shift_type}_{amplitude_type}_{regressor_type}.npy'
    #     INBETWEEN_REGRESSOR_CONVOLVED_FILE = f'{base_type}_inbetween_{shift_type}_{amplitude_type}_{regressor_type}.npy'

    #     # Construct the full paths
    #     norm_regressors_fn = os.path.join(REGRESSORS_DIR, f'{base_type}_{shift_type}',NORM_REGRESSOR_CONVOLVED_FILE)
    #     inbetween_regressors_fn = os.path.join(REGRESSORS_DIR_INBETWEEN, f'{base_type}_inbetween_{shift_type}',INBETWEEN_REGRESSOR_CONVOLVED_FILE)
        
    #     regressor_norm = np.load(norm_regressors_fn, allow_pickle=True).reshape(-1, 1)
    #     regressor_inbetween = np.load(inbetween_regressors_fn, allow_pickle=True).reshape(-1, 1)