# Compare alexnet outputs from different video clip sources
import os
import numpy as np
import pandas as pd


PROJ_DIR = '/mnt/labdata/got_project/ian'


if __name__ == "__main__":
    alexnet_Rebecca = os.path.join(PROJ_DIR, 'data/alexnet/Rebecca')
    alexnet_Daisy = os.path.join(PROJ_DIR, 'data/alexnet/Daisy')

    # Should be sorted properly after *renaming.py
    outputs_Rebecca = sorted(os.listdir(alexnet_Rebecca))
    outputs_Daisy = sorted(os.listdir(alexnet_Daisy))

    if len(outputs_Daisy) != len(outputs_Rebecca):
        raise ValueError ('Number of outputs is unequal')
    else:
        n_clips = len(outputs_Daisy)
    
    corrs = []
    for i in range(n_clips): 
        features_fn_Rebecca = outputs_Rebecca[i]
        features_fn_Daisy = outputs_Daisy[i]

        features_Rebecca = np.load(
            os.path.join(alexnet_Rebecca, features_fn_Rebecca)
            )
        features_Daisy = np.load(
            os.path.join(alexnet_Daisy, features_fn_Daisy)
            )

        r = np.corrcoef(features_Daisy, features_Rebecca)[0 ,1]
        corrs.append(r)
    
    print(np.array(corrs))
    out_df = pd.DataFrame()
    out_df['filenames_Daisy'] = [f.replace('.npy', '') for f in outputs_Daisy]
    out_df['filenames_Rebecca'] = [f.replace('.npy', '') for f in outputs_Rebecca]
    out_df['pearsonr'] = np.array(corrs)
    out_df.to_csv(os.path.expanduser('~/Documents/got_projectalexnet_scripts/alexnet_correlations.csv'))

