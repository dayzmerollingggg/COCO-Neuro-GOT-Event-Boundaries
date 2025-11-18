# Plot simple contrasts from second level analysis testing
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brainplotlib import brain_plot

SCRIPTS_DIR = '/mnt/labdata/got_project/daisy/got_project'
sys.path.append(SCRIPTS_DIR)
from utils import plot_brain

PROJ_DIR = '/mnt/labdata/got_project'
DATA_DIR = os.path.join(PROJ_DIR, 'daisy/data')
FEAT_DIR = os.path.join(DATA_DIR, 'features_renamed')
#RES_DIR = os.path.join(DATA_DIR, 'results/simple_contrasts')
#FIG_DIR = os.path.join(DATA_DIR, 'figures/simple_contrasts')
RES_DIR = os.path.join(DATA_DIR, 'results/compare_contrasts_custom_reg')
FIG_DIR = os.path.join(DATA_DIR, 'figures/compare_contrasts_custom_reg')
os.makedirs(FIG_DIR, exist_ok=True)

# funcs

def load_feature_names(category):
    fn = os.path.join(FEAT_DIR, f'{category}.tsv')
    feat_df = pd.read_csv(fn, sep='\t', index_col=0)
    return list(feat_df.columns)


def plot_brains(plot_data, titles, vmax=4, cbar_label='t', plot_cbar=True, plot_titles=False):
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
# Set vars for reference
groups = ['control', 'DP']
categories = [
    'camera_cuts',
    'scene_cuts',
]
# Plot perceptual features group average t-maps

ordered_feats = ['scene_cuts','camera_cuts']
for i, feat in enumerate(ordered_feats):
    print("inside")
    plot_data = []
    for group in groups:
        fn = f'simple_contrasts_{group}_{feat}.npy'
        group_data = np.load(os.path.join(RES_DIR, fn))
        contrast_data = group_data[0, :]
        plot_data.append(contrast_data)
    
    fig, axs = plot_brains(plot_data, groups, plot_titles=True)
    fig.suptitle(feat, fontsize=32)
    plt.show()
    save_fn = os.path.join(FIG_DIR, f'simple_contrast_{feat}.png')
    fig.savefig(save_fn, bbox_inches='tight') # Save the figure to your designated directory
    plt.close(fig) # Close the figure to free memory
        

