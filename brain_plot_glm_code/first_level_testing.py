import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import zscore, t, norm
from scipy.special import legendre
from joblib import Parallel, delayed, parallel_backend
from numpy.linalg import lstsq, inv # inv is crucial for T-stat calculation

# --- CONSTANTS ---
SCRIPTS_DIR = '/mnt/labdata/got_project/daisy/got_project'
sys.path.append(SCRIPTS_DIR)
# Assuming these utility functions are available in utils.py
from utils import get_got_subjects, load_mask 

PROJ_DIR = '/mnt/labdata/got_project'
FMRI_DATA_DIR = os.path.join(PROJ_DIR, 'data')
DATA_DIR = os.path.join(PROJ_DIR, 'daisy/data')
REGRESSORS_DIR = os.path.join(DATA_DIR, 'regressors')

# Using the compare_contrasts directory structure for output
OUTPUT_DIR = os.path.join(DATA_DIR, 'glm/compare_contrasts_custom_reg') 
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


def calculate_full_voxel_stats(regressors_df, data_matrix, nuisance):
    """
    Calculates the regression coefficients (betas), T-statistics, and Z-scores
    for *each voxel* using the standard GLM: Y_voxel = X_regressor * B_regressor + X_nuisance * B_nuisance + error.
    
    The external regressor ('regressors_df') is the predictor (X), 
    and the BOLD data ('data_matrix') is the target (Y), as is standard 
    in fMRI analysis.
    
    T-stats are calculated for the BOLD predictor (now the first regressor) 
    using the full contrast array method.
    
    Args:
        regressors_df (np.ndarray): The primary external regressor (T x 1).
        data_matrix (np.ndarray): The fMRI BOLD data (T x V), 
                                  where T is time points and V is voxels. (The new Y)
        nuisance (np.ndarray): The nuisance regressors (T x N).
    
    Returns:
        np.ndarray: The primary regressor's beta coefficients (V,).
        np.ndarray: The T-statistics for the primary beta (V,).
        np.ndarray: The Z-scores for the primary beta (V,).
    """
    n_tp = data_matrix.shape[0]
    n_voxels = data_matrix.shape[1]
    
    # Ensure the primary regressor is (T x 1)
    X_regressor = regressors_df.reshape(-1, 1) if regressors_df.ndim == 1 else regressors_df

    # 1. DEFINE THE FULL, CONSTANT DESIGN MATRIX (X)
    # X_full = [X_regressor | X_nuisance] (T x R)
    X_full = np.concatenate((X_regressor, nuisance), axis=1) 
    
    n_regs = X_full.shape[1] # Total number of regressors (R)
    df = n_tp - n_regs # Degrees of freedom for the model

    # 2. DEFINE THE CONSTANT CONTRAST VECTOR (c)
    # Contrast the first regressor (the BOLD/External predictor): c = [1, 0, 0, ..., 0]
    '''
    contrast_vector = np.zeros(n_regs)
    contrast_vector[0] = 1.0 
    c = contrast_vector.reshape(1, -1) # Reshape contrast to (1 x R)
    '''
    # Define the contrast vector to test (B_nuisance_1 - B_voxel)
    # The full vector will be [ -1, 1, 0, 0, ..., 0 ]
    contrast_vector = np.zeros(n_regs)
    # Contrast B_voxel (index 0) with a weight of -1
    contrast_vector[0] = -1.0 
    # Contrast B_nuisance_1 (index 1) with a weight of +1
    if n_regs > 1:
        contrast_vector[1] = 1.0 
    
    # Reshape contrast to (1 x R) for matrix multiplication
    c = contrast_vector.reshape(1, -1)
    
    # 3. CALCULATE CONSTANT INVERSE GRAM MATRIX (X^T X)^{-1}
    # This matrix is the same for all voxels, so calculate it once.
    try:
        XTX_inv = inv(X_full.T @ X_full)
    except np.linalg.LinAlgError:
        print("Error: Design matrix X^T X is singular. Cannot proceed.")
        return np.zeros(n_voxels), np.zeros(n_voxels), np.zeros(n_voxels)
        
    # Initialize arrays to store the results
    primary_betas = np.zeros(n_voxels)
    t_stats = np.zeros(n_voxels)
    
    # 4. ITERATE OVER VOXELS (Y_voxel)
    for v in range(n_voxels):
        # Y is the BOLD signal for the current voxel (T x 1)
        Y_voxel = data_matrix[:, v:v+1]
        
        # OLS Regression: B = (X^T X)^-1 X^T Y
        betas_all, rss_sum_sq, rank, s = lstsq(X_full, Y_voxel, rcond=None)
        
        # 5. Extract Beta for Primary regressor (index 0)
        primary_betas[v] = betas_all[0].item()
        
        # 6. Calculate Residual Sum of Squares (RSS)
        # Note: lstsq returns the sum of squares of residuals if rank < T. 
        # For a well-posed problem, the residual sum is in rss_sum_sq[0] or 
        # needs to be calculated manually if the tuple is empty.
        if len(rss_sum_sq) == 0:
            Y_hat = X_full @ betas_all
            rss = np.sum((Y_voxel - Y_hat)**2)
        else:
            rss = rss_sum_sq[0] 
            
        # 7. Calculate Mean Squared Error (MSE) / Error Variance (sigma^2)
        mse = rss / df
        
        # 8. Calculate Variance of the Contrast: Var(c*Beta) = MSE * c * (X^T X)^{-1} * c^T
        var_contrast = mse * (c @ XTX_inv @ c.T).item()
        
        # 9. Calculate Standard Error of the Contrast: SE(c*Beta)
        se_contrast = np.sqrt(var_contrast)
        
        # 10. Calculate T-statistic: T = (c*Beta) / SE(c*Beta)
        # Handle cases where SE is near zero to avoid division by zero
        if se_contrast < 1e-12:
            t_stat = 0.0
        else:
            t_stat = (c @ betas_all).item() / se_contrast
        
        t_stats[v] = t_stat
        
    # 11. Convert T-stats to Z-scores using the correct DF
    # Calculate the p-value corresponding to the T-statistic
    p_values = t.cdf(t_stats, df=df) 
    # Convert the p-value to a Z-score (using the inverse CDF of the standard normal distribution)
    z_scores = norm.ppf(p_values)
    
    return primary_betas, t_stats, z_scores



def pipe_wrapper(subj, hemi, y_regressor, category_name):
    # For parallel processsing
    out_fn = os.path.join(OUTPUT_DIR, f'{subj}_{category_name}_{hemi}.npz')

    nuisance = get_got_nuisance(subj) 


    mask = load_mask(hemi)
    denoised_dir = os.path.join(FMRI_DATA_DIR, f'denoised/{subj}')
    data_fn = f'{subj}_task-GoT_space-fsaverage5_{hemi}_denoised.npy'
    data = np.load(os.path.join(denoised_dir, data_fn))[:, mask]
    data_matrix = np.nan_to_num(zscore(data, axis=0))
    
    bold_betas, t_stats, X_full_placeholder = calculate_full_voxel_stats(y_regressor, data_matrix, nuisance)

    np.savez(out_fn, betas=bold_betas, ts=t_stats)
    

if __name__ == "__main__":
    hemis = ['hemi-L', 'hemi-R']
    subjects = get_got_subjects()
    regressors_files = [f for f in os.listdir(REGRESSORS_DIR) if '.csv' in f]
    
    jobs = []
    
    for test_reg in regressors_files:
        category_name = test_reg.replace('_regressors.csv', '')
        regressors_fn = os.path.join(REGRESSORS_DIR, test_reg)
        design_matrix_df = pd.read_csv(regressors_fn, index_col=0) 
        
        y_regressor = design_matrix_df.iloc[:, 0].values.reshape(-1, 1) 
        
        for subj in subjects:
            for hemi in hemis:
                jobs.append(delayed(pipe_wrapper)(subj, hemi, y_regressor, category_name))
                                
    with parallel_backend("loky", inner_max_num_threads=1):
        Parallel(n_jobs=4, verbose=2)(jobs)