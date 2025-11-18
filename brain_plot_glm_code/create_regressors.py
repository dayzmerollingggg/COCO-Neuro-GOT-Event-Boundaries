# Create regressors from annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix


PROJ_DIR = '/mnt/labdata/got_project'
DATA_DIR = os.path.join(PROJ_DIR, 'ian/data')
FEAT_DIR = os.path.join(DATA_DIR, 'features_renamed')
ROOT_OUTPUT_DIR = os.path.join(DATA_DIR, 'regressors')
os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)
TIMING_DIR = os.path.join(ROOT_OUTPUT_DIR, 'timing_files')
os.makedirs(TIMING_DIR, exist_ok=True)
FIG_DIR = os.path.join(ROOT_OUTPUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def load_features_cleaned(feat_file=None):
    feat_df = pd.read_csv(os.path.join(FEAT_DIR, feat_file), index_col=0, sep='\t')
    return feat_df


def create_events_df(feat_df):
    timing_dfs = []
    for feat in feat_df.columns:
        amplitude = feat_df[feat].values # set the annotation value to be the amplitude
        duration = np.full(len(amplitude), 4)
        duration[-1] = 2 # set the last annotation to be 2 seconds
        onset = np.arange(len(amplitude)) * 4
        trial_type = [feat] * len(amplitude)
        data = {
            'onset': onset,
            'duration': duration,
            'modulation': amplitude,
            'trial_type': trial_type,
        }
        timing_df = pd.DataFrame(data)
        timing_dfs.append(timing_df)
    events_df = pd.concat(timing_dfs, ignore_index=True) # merge all features into a single timing file
    events_df = events_df.sort_values(by='onset').reset_index(drop=True) # sort by onset for human readability
    return events_df


def create_timing_files():
    category_files = os.listdir(FEAT_DIR)
    category_names = [s.replace('.tsv', '') for s in category_files]
    for i, feat_file in enumerate(category_files):
        feat_df = load_features_cleaned(feat_file)
        out_fn = os.path.join(TIMING_DIR, f'{category_names[i]}.csv')
        events_df = create_events_df(feat_df)
        events_df.to_csv(out_fn)


def create_regressors(events_df):
    tr = 2
    n_scans = 389 # 389 timepoints in fmri data (389*2=778s)
    frame_times = np.arange(n_scans) * tr
    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events_df,
        hrf_model='spm',
        drift_model=None
        )
    return design_matrix


if __name__ == "__main__":
    create_timing_files()
    timing_files = os.listdir(TIMING_DIR)
    for timing_file in timing_files:
        print(timing_file)
        category_name = timing_file.replace('.csv', '')

        # get feature names to reorder design matrix columns
        feat_file = os.path.join(FEAT_DIR, f'{category_name}.tsv')
        feat_df = pd.read_csv(feat_file, sep='\t', index_col=0)
        ordered_features = list(feat_df.columns)

        # create regressors
        events_df = pd.read_csv(os.path.join(TIMING_DIR, timing_file), index_col=0)
        design_matrix = create_regressors(events_df)
        
        # reorder regressors to make life easier
        regressors = design_matrix[ordered_features]

        # save outputs
        regressors.to_csv(os.path.join(ROOT_OUTPUT_DIR, f'{category_name}_regressors.csv'))

        # plot outputs for inspection
        fig ,ax = plt.subplots(figsize=(8, 12))
        plot_design_matrix(regressors, ax=ax)
        fig.savefig(os.path.join(FIG_DIR, f'{category_name}.png'), bbox_inches='tight')
        plt.close(fig)
        # break