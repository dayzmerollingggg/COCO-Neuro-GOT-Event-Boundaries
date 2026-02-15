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