import os
import sys
import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix



def make_regressors(events_df, stimuli, tr, n_tp, hrf_model='spm', oversampling=128):
    frames = np.arange(n_tp) * tr
    stimuli_regressors = make_first_level_design_matrix(
        frame_times=frames,
        events=events_df,
        hrf_model=hrf_model,
        oversampling=oversampling,
    )
    return stimuli_regressors[stimuli]


def load_timing_file():
    # Load dataset-darts timing files
    events_df = pd.read_csv("examplefile.tsv", sep='\t')
    return events_df


if __name__ == "__main__":
    #subjects = get_dar_subjects(two_tasks=False)
    #runs = [i+1 for i in range(5)]

    stimuli = ['face', 'body', 'object', 'scene', 'sobject']
    tr = 2
    n_tp = 126

    events_df = load_timing_file()
    regressors = make_regressors(
        events_df=events_df,
        stimuli=stimuli,
        tr=tr,
        n_tp=n_tp,
    )
    regressors.to_csv("exampleoutput.csv")

            