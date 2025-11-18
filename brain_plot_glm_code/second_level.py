# Second level analysis - make group averages

import os
import sys
import numpy as np
import pandas as pd

SCRIPTS_DIR = '/mnt/labdata/got_project/daisy/got_project'
sys.path.append(SCRIPTS_DIR)
from utils import get_got_subjects, load_mask

PROJ_DIR = '/mnt/labdata/got_project'
DATA_DIR = os.path.join(PROJ_DIR, 'daisy/data')
FEAT_DIR = os.path.join(DATA_DIR, 'feature_names') #create feature_names txt
#GLM_DIR = os.path.join(DATA_DIR, 'glm/simple_contrasts')
#OUTPUT_DIR = os.path.join(DATA_DIR, 'results/simple_contrasts')
GLM_DIR = os.path.join(DATA_DIR, 'glm/compare_contrasts_custom_reg')
OUTPUT_DIR = os.path.join(DATA_DIR, 'results/compare_contrasts_custom_reg')
os.makedirs(OUTPUT_DIR, exist_ok=True)


if __name__ == "__main__":
    groups = ['control', 'DP']
    control_subjects = get_got_subjects('control')
    dp_subjects = get_got_subjects('DP')
    hemis = ['hemi-L', 'hemi-R']
    feature_files = os.listdir(FEAT_DIR)
    categories = [f.replace('.txt', '') for f in feature_files]
    
    for category in categories:
        with open(os.path.join(FEAT_DIR, f'{category}.txt'), 'r') as f:
            feature_names = f.readlines()
        feat_names = [feat.replace('\n', '') for feat in feature_names]
        for group in groups:
            subjects = get_got_subjects(group)
            group_data = []
            for subj in subjects:
                brain_data = []
                for hemi in hemis:
                    fn = f'{subj}_{category}_{hemi}.npz'
                    data = np.load(os.path.join(GLM_DIR, fn))['ts']
                    brain_data.append(data)
                brain_array = np.concatenate(brain_data, axis=0) #originally axis=1
                group_data.append(brain_array)
            
            group_array = np.dstack(group_data)
            print(group_array.shape)
            group_average = np.mean(group_array, axis=2)
            print(group_average.shape)

            out_fn = f'simple_contrasts_{group}_{category}.npy'
            np.save(os.path.join(OUTPUT_DIR, out_fn), group_average)
            