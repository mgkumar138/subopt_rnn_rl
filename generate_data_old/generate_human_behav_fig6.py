
'''
Script to generate figures 6 in paper
'''

#%%
import matplotlib.pyplot as plt
from utils_funcs import saveload, get_mean_ci
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler


#%%
from scipy.stats import ttest_ind
import scipy.io as sio
sz_pat = 102 #first 102 / 134 in the model data are patients
#115 is trial num? use all for mean / std
#2 - nassar uses 2nd index always but idk what it is
idx = 1

# mod_data = sio.loadmat('data/nassar2021/slidingWindowFits_model_23-Nov-2021.mat')
sub_data = sio.loadmat('data/nassar2021/slidingWindowFits_subjects_23-Nov-2021.mat')

# mod_data = np.asarray(mod_data['binRegData'])
sub_data = np.asarray(sub_data['binRegData'])
cutoff = 95
x_axis = np.arange(115)
x_ind = np.array(x_axis>cutoff)
#so raw python array structure is-
#data[0][0][CP/OB][pat / control ][:][1]
#TO-DO clean up this structure.
cp = 1
ob = 0

cp_c = np.clip(sub_data[0][0][cp][sz_pat:][:][:][:,:,idx],0,1)
ob_c = np.clip(sub_data[0][0][ob][sz_pat:][:][:][:,:,idx],0,1)
cp_p = np.clip(sub_data[0][0][cp][:sz_pat][:][:][:,:,idx],0,1)
ob_p = np.clip(sub_data[0][0][ob][:sz_pat][:][:][:,:,idx],0,1)

area_cp_c = np.trapz(cp_c[:,x_ind], x_axis[x_ind], axis=1)
area_ob_c = np.trapz(ob_c[:,x_ind], x_axis[x_ind], axis=1)
area_cp_p = np.trapz(cp_p[:,x_ind], x_axis[x_ind], axis=1)
area_ob_p = np.trapz(ob_p[:,x_ind], x_axis[x_ind], axis=1)


# area_cp_c = np.trapz(np.mean(cp_c[:,x_ind],axis=0), x_axis[x_ind])
# area_ob_c = np.trapz(np.mean(ob_c[:,x_ind],axis=0), x_axis[x_ind])
# area_cp_p = np.trapz(np.mean(cp_p[:,x_ind],axis=0), x_axis[x_ind])
# area_ob_p = np.trapz(np.mean(ob_p[:,x_ind],axis=0), x_axis[x_ind])


# cp_c = np.abs(sub_data[0][0][cp][sz_pat:][:][:][:,:,1])
# ob_c = np.abs(sub_data[0][0][ob][sz_pat:][:][:][:,:,1])
# cp_p = np.abs(sub_data[0][0][cp][:sz_pat][:][:][:,:,1])
# ob_p = np.abs(sub_data[0][0][ob][:sz_pat][:][:][:,:,1])
# area_cp_c = np.trapz(cp_c[:,x_ind], axis=1)
# area_ob_c = np.trapz(ob_c[:,x_ind], axis=1)
# area_cp_p = np.trapz(cp_p[:,x_ind], axis=1)
# area_ob_p = np.trapz(ob_p[:,x_ind], axis=1)


darea_c = area_cp_c - area_ob_c
darea_p = area_cp_p - area_ob_p

areas = [(area_cp_c, 'orange', 'CP Control'),
         (area_cp_p, 'deepskyblue', 'CP Patients'),
         (area_ob_c, 'brown', 'OB Control'),
         (area_ob_p, 'cadetblue', 'OB Patients'),
         (darea_c, 'k', '$\Delta$ Area Control'),
         (darea_p, 'blue', '$\Delta$ Area Patients')]

# Plot configuration
fig, ax = plt.subplots(figsize=(3,2.5))
bar_width = 1
positions = [0, 1, 3, 4, 6, 7]  # Positions for bars with gaps

# Plot bars
for i, (data, color, label) in enumerate(areas):
    mean_val = np.mean(data)
    std_val = np.std(data)/np.sqrt(len(data))
    ax.bar(positions[i], mean_val, yerr=std_val, width=bar_width, color=color, label=label)
    print(mean_val, std_val)

# Annotate significance
significance_levels = []

# Compare and annotate
maxy = 0
comparisons = [(0, 1), (2, 3), (4, 5)]
for idx1, idx2 in comparisons:
    t, p = ttest_ind(areas[idx1][0], areas[idx2][0], equal_var=True)
    pos = (positions[idx1] + positions[idx2]) / 2
    if p < 0.0001:
        significance = '****'
    elif p < 0.001:
        significance = '***'
    elif p < 0.01:
        significance = '**'
    elif p < 0.05:
        significance = '*'
    else:
        significance = ' N.S.'  # not significant
    significance_levels.append((pos, p, significance))
    ax.text(pos, max(np.mean(areas[idx1][0]), np.mean(areas[idx2][0])) *1.1, f'{significance}', # {t:.3f},{p:.3f}
            ha='center', va='bottom', color='black', fontsize=10)

    if maxy< max(np.mean(areas[idx1][0]), np.mean(areas[idx2][0]))*1.1:
        maxy = max(np.mean(areas[idx1][0]), np.mean(areas[idx2][0]))*1.1
# Customizing the plot
ax.set_xticks([(positions[i] + positions[i+1]) / 2 for i in range(0, len(positions), 2)])
ax.set_xticklabels(['CP', 'OB', '$\Delta$ Area'])
ax.set_ylabel('Area')
# ax.legend(fontsize=8, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
ax.set_ylim([0.0,maxy*1.1])
ax.set_title(f'Prediction Errors > {cutoff}')
fig.tight_layout()
fig.savefig(f'./human_analysis/area_bar_{cutoff}.svg')
fig.savefig(f'./human_analysis/area_bar_{cutoff}.png',dpi=300)


# %%

cutoff = 0
x_axis = np.arange(115)
x_ind = np.array(x_axis>cutoff)
data = [[cp_c, ob_c],[cp_p, ob_p]]
labels = [['CP Control', 'OB Control'],['CP Patient', 'OB Patient']]
colors = [['orange','brown'],['deepskyblue', 'cadetblue']]

f,axs = plt.subplots(2,2,figsize=(3*2,2.5*2))
f2,ax2 = plt.subplots(figsize=(3,2.5))
for c in range(2):
    for s in range(2):
        axs[s,c].plot(x_axis[x_ind], np.clip(data[s][c],0,1)[:,x_ind].T, alpha=0.2)
        axs[s,c].plot(x_axis[x_ind], np.mean(np.clip(data[s][c],0,1)[:,x_ind],axis=0), color='k', linewidth=5)
        axs[s,c].set_title(labels[s][c])

        ax2.plot(x_axis[x_ind], np.mean(np.clip(data[s][c],0,1)[:,x_ind],axis=0), color=colors[s][c], linewidth=5,label=labels[s][c])

ax2.set_ylabel('Learning rate')
ax2.set_xlabel('Prediction Error')
ax2.set_title('Subject behavior')
ax2.axvline(20, color='k',linestyle='--')
# ax2.legend(fontsize=8, loc=2)
# ax2.axvline(60, color='k',linestyle='--')
ax2.axvline(95, color='k',linestyle='--')
f.tight_layout()
f2.tight_layout()
f2.savefig(f'./human_analysis/lr_pe.svg')
f2.savefig(f'./human_analysis/lr_pe.png',dpi=300)
# %%



#%%
mincutoff = 25
maxcutoff = 95
x_axis = np.arange(115)
x_ind = (x_axis > mincutoff) & (x_axis < maxcutoff)
x_ind = np.array(x_ind)
#so raw python array structure is-
#data[0][0][CP/OB][pat / control ][:][1]
#TO-DO clean up this structure.
cp = 1
ob = 0

cp_c = np.clip(sub_data[0][0][cp][sz_pat:][:][:][:,:,idx],0,1)
ob_c = np.clip(sub_data[0][0][ob][sz_pat:][:][:][:,:,idx],0,1)
cp_p = np.clip(sub_data[0][0][cp][:sz_pat][:][:][:,:,idx],0,1)
ob_p = np.clip(sub_data[0][0][ob][:sz_pat][:][:][:,:,idx],0,1)

area_cp_c = np.trapz(cp_c[:,x_ind], x_axis[x_ind], axis=1)
area_ob_c = np.trapz(ob_c[:,x_ind], x_axis[x_ind], axis=1)
area_cp_p = np.trapz(cp_p[:,x_ind], x_axis[x_ind], axis=1)
area_ob_p = np.trapz(ob_p[:,x_ind], x_axis[x_ind], axis=1)


# area_cp_c = np.trapz(np.mean(cp_c[:,x_ind],axis=0), x_axis[x_ind])
# area_ob_c = np.trapz(np.mean(ob_c[:,x_ind],axis=0), x_axis[x_ind])
# area_cp_p = np.trapz(np.mean(cp_p[:,x_ind],axis=0), x_axis[x_ind])
# area_ob_p = np.trapz(np.mean(ob_p[:,x_ind],axis=0), x_axis[x_ind])


# cp_c = np.abs(sub_data[0][0][cp][sz_pat:][:][:][:,:,1])
# ob_c = np.abs(sub_data[0][0][ob][sz_pat:][:][:][:,:,1])
# cp_p = np.abs(sub_data[0][0][cp][:sz_pat][:][:][:,:,1])
# ob_p = np.abs(sub_data[0][0][ob][:sz_pat][:][:][:,:,1])
# area_cp_c = np.trapz(cp_c[:,x_ind], axis=1)
# area_ob_c = np.trapz(ob_c[:,x_ind], axis=1)
# area_cp_p = np.trapz(cp_p[:,x_ind], axis=1)
# area_ob_p = np.trapz(ob_p[:,x_ind], axis=1)


darea_c = area_cp_c - area_ob_c
darea_p = area_cp_p - area_ob_p

areas = [(area_cp_c, 'orange', 'CP Control'),
         (area_cp_p, 'deepskyblue', 'CP Patients'),
         (area_ob_c, 'brown', 'OB Control'),
         (area_ob_p, 'cadetblue', 'OB Patients'),
         (darea_c, 'k', '$\Delta$ Area Control'),
         (darea_p, 'blue', '$\Delta$ Area Patients')]

# Plot configuration
fig, ax = plt.subplots(figsize=(3,2.5))
bar_width = 1
positions = [0, 1, 3, 4, 6, 7]  # Positions for bars with gaps

# Plot bars
for i, (data, color, label) in enumerate(areas):
    mean_val = np.mean(data)
    std_val = np.std(data)/np.sqrt(len(data))
    ax.bar(positions[i], mean_val, yerr=std_val, width=bar_width, color=color, label=label)
    print(mean_val, std_val)

# Annotate significance
significance_levels = []

# Compare and annotate
maxy = 0
comparisons = [(0, 1), (2, 3), (4, 5)]
for idx1, idx2 in comparisons:
    t, p = ttest_ind(areas[idx1][0], areas[idx2][0])
    pos = (positions[idx1] + positions[idx2]) / 2
    if p < 0.0001:
        significance = '****'
    elif p < 0.001:
        significance = '***'
    elif p < 0.01:
        significance = '**'
    elif p < 0.05:
        significance = '*'
    else:
        significance = ' N.S.'  # not significant
    significance_levels.append((pos, p, significance))
    ax.text(pos, max(np.mean(areas[idx1][0]), np.mean(areas[idx2][0])) *1.1, f'{significance}', #{np.round(t,2)}{p:.3f}
            ha='center', va='bottom', color='black', fontsize=10)

    if maxy< max(np.mean(areas[idx1][0]), np.mean(areas[idx2][0]))*1.1:
        maxy = max(np.mean(areas[idx1][0]), np.mean(areas[idx2][0]))*1.1
# Customizing the plot
ax.set_xticks([(positions[i] + positions[i+1]) / 2 for i in range(0, len(positions), 2)])
ax.set_xticklabels(['CP', 'OB', '$\Delta$ Area'])
ax.set_ylabel('Area')
# ax.legend(fontsize=8, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=1)
ax.set_ylim([0.0,maxy*1.1])
ax.set_title(f'{mincutoff} < Prediction Errors < {maxcutoff}')
fig.tight_layout()
fig.savefig(f'./human_analysis/area_bar_{mincutoff}_{maxcutoff}.svg')
fig.savefig(f'./human_analysis/area_bar_{mincutoff}_{maxcutoff}.png',dpi=300)


# %%
