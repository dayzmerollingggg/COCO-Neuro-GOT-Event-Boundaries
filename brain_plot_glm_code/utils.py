# Utility functions for the project
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brainplotlib import brain_plot

PROJ_DIR = '/mnt/labdata/got_project'
RAW_DATA_DIR_GOT = os.path.join(PROJ_DIR, 'data/GoT_dataset')


def load_mask(hemi, icoorder=5):
    """
    Load fsaverage7 cortical masks (medial wall removed), slice by icoorder

    Args:
        hemi: str; 'lh' or 'rh'

        icoorder: int; default = 5, corresponding to fsaverage5
        n_vertices = 4**icoorder * 10 + 2
    """
    if hemi == 'hemi-L':
        hemi = 'lh'
    elif hemi == 'hemi-R':
        hemi = 'rh'
    
    masks_dir = os.path.join(PROJ_DIR, 'utils/cortical_surface_masks')
    file = f'fsaverage_{hemi}_mask.npy'

    n_vertices = 4**icoorder * 10 + 2
    
    mask = np.load(os.path.join(masks_dir, file))[:n_vertices]
    return mask


def get_got_subjects(group='all', familiarity='both', return_df=False, drop_DP15=True):
    participants_file = os.path.join(RAW_DATA_DIR_GOT, 'participants.tsv')
    participants_df = pd.read_csv(participants_file, delimiter='\t')
    
    if familiarity not in ['both', 'Familiar', 'Unfamiliar']:
        raise ValueError('Invalid familiarity: options are both, Familiar, or Unfamiliar')
    elif familiarity == 'both':
        res_df = participants_df.copy()
    else:
        res_df = participants_df.loc[participants_df['familiarity'] == familiarity].copy()

    
    if group not in ['all', 'control', 'DP']:
        raise ValueError('Invalid group: options are all, control, or DP')
    elif group == 'all':
        pass
    elif group == 'control':
        res_df = res_df.loc[res_df['group'] == 'Control'] # handle the capital C here

    else:
        res_df = res_df.loc[res_df['group'] == group]
    
    if drop_DP15 is True: ### sub-DP15 missing localiser, fieldmaps
        res_df.drop(res_df[res_df['participant_id'] == 'sub-DP15'].index, inplace=True)
    
    if return_df is True:
        return res_df
    else:
        return list(res_df['participant_id'].values)
    


def plot_brain(vector, title=None, vmax=4, vmin=None, cmap='seismic', cbar_label='t-value', plot_cbar=True):
    # Copied here for quick reference, copy/pasting
    if vmin == None:
        vmin = -1 * vmax
    fig, ax = plt.subplots()
    img = brain_plot(vector, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title)
    if plot_cbar == True:
        norm = plt.Normalize(vmin, vmax)
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            orientation='horizontal',
            shrink=0.5,
            label=cbar_label,
            )
    return fig, ax
