'''
Script to generate figures 2,4,5 in paper
'''

#%%
import matplotlib.pyplot as plt
from utils_funcs import saveload, get_mean_ci
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler

N_inits = 2048

params= ['$\gamma$', '$\\beta_{\delta}$','$p_{reset}$',r'$\tau$'] # , '$t_{rollout}$'

f,axs = plt.subplots(3, 4, figsize=(2.5*4, 2*3))
flabels = ['Stable','Unstable','All']
fcolors = ['dodgerblue','salmon','violet']


f2,axs2 = plt.subplots(4, 4, figsize=(2.5*4, 2*4))
alabels = ['CP', 'OB']
acolors= ['orange', 'brown']

f3, axs3 = plt.subplots(2,2, figsize=(2.5*2, 2.*2))
f4, axs4 = plt.subplots(2,2, figsize=(2.5*2, 2.*2))
axs3 = axs3.flatten()
axs4 = axs4.flatten()

f5, axs5 = plt.subplots(2,2, figsize=(2.5*2, 2.*2))
axs5 = axs5.flatten()

for p, param_name in enumerate(params):
    [param, areas, valida] = saveload(f'./analysis/{param_name}_area',1,'load')

    if p ==3:
        [param, fps] = saveload(f'./analysis/{param_name}_fps',1,'load')
        validf = valida
    else:
        [param, fps, validf] = saveload(f'./analysis/{param_name}_fps',1,'load')

    [param, fps, validf] = saveload(f'./analysis/{param_name}_fps_2048',1,'load')
    # if N_inits == 2000 and param_name == '$\\beta_{\\delta}$':
    #     idx = np.array([0,1,1,1,0,1,0,1,1],dtype=bool)
    #     param = list(np.array(param)[idx])
    #     fps = fps[idx]
    #     validf = validf[idx]



    # print(validf, valida)

    dfarea = areas[:,:,0] - areas[:,:,1]
    for i in range(3):

        dffps = fps[:,:,0,i] - fps[:,:,1,i] 

        x = dfarea.reshape(-1)
        y = dffps.reshape(-1)

        # xind = x>10
        # x = x[xind]
        # y = y[xind]

        # percentile_01 = np.percentile(y, 10)  # 1st percentile
        # percentile_99 = np.percentile(y, 90)  # 99th percentile

        # x = x[(y >= percentile_01) & (y <= percentile_99)]
        # y = y[(y >= percentile_01) & (y <= percentile_99)]
        print(x.shape, y.shape)

        r_value, p_value = spearmanr(x,y)
        # p_value = np.round(p_value,5)
        # r_value, p_value = pearsonr(x,y)

        slope, intercept, r_pearson, p_pearson, std_err = linregress(x, y)
        yfit = slope * x + intercept

        if p_value< 1e-10:
            pval = '$<10^{-10}$'
        elif p_value< 1e-5:
            pval = '$<10^{-5}$'
        elif p_value< 1e-4:
            pval = '$<10^{-4}$'
        elif p_value< 1e-3:
            pval = '$<10^{-3}$'
        elif p_value< 1e-2:
            pval = '$<10^{-2}$'
        
        else:
            pval = f' = {np.round(p_value,3)}'

        axs[i,p].plot(x, yfit, color='k', label=f'$R={r_value :.3f}$\n$p ${pval}')
        # axs[i,p].plot(x, yfit, color='k', label=f'$R^2={r_value**2:.3f}$\n$p ${pval}')
        axs[i,p].scatter(x,y, s=2, edgecolors=fcolors[i],facecolors='none')
        axs[i,p].legend(fontsize=8)

        if i == 1:
            axs4[p].plot(x, yfit, color='k', label=f'$R={r_value :.3f}$\n$p ${pval}')
            # axs4[p].plot(x, yfit, color='k', label=f'$R^2={r_value**2 :.3f}$\n$p ${pval}')
            axs4[p].scatter(x,y, s=2, edgecolors=fcolors[i],facecolors='none')
            axs4[p].legend(fontsize=8)



        if i == 0:
            axs[i,p].set_title(param_name)
            axs2[i,p].set_title(param_name)
        if p == 0: 
            axs[i,p].set_ylabel(f'$\Delta$ {flabels[i]} FPs')
            axs2[i,p].set_ylabel(f'{flabels[i]} FPs')
        if i == 2:
            axs[i,p].set_xlabel('$\Delta$ Area')
    

        # plot fp vs param
        for j in range(2):
            m,s = get_mean_ci(fps[:,:,j,i],validf)

            axs2[i,p].plot(param, m, label=alabels[j], color=acolors[j])
            axs2[i,p].fill_between(x=param, y1=m-s, y2=m+s, alpha=0.2, color=acolors[j])
        
        m,s = get_mean_ci(dffps,validf)
        axs2[i,p].plot(param, m, label='$\Delta$ FP', color='k', linewidth=2)
        axs2[i,p].fill_between(x=param, y1=m-s, y2=m+s, alpha=0.2, color='k')
        axs2[0,0].legend(fontsize=8)

        if p ==3:
            axs2[i,p].set_xscale('log')

        
        if i == 1:
            for j in range(2):
                m,s = get_mean_ci(fps[:,:,j,i],validf)

                axs5[p].plot(param, m, label=alabels[j], color=acolors[j])
                axs5[p].fill_between(x=param, y1=m-s, y2=m+s, alpha=0.2, color=acolors[j])
            
            m,s = get_mean_ci(dffps,validf)
            axs5[p].plot(param, m, label='$\Delta$ FP', color='k', linewidth=2)
            axs5[p].fill_between(x=param, y1=m-s, y2=m+s, alpha=0.2, color='k')
            axs5[3].legend(fontsize=8)

            if p ==3:
                axs5[p].set_xscale('log')
            
            if p == 0 or p==2:
                axs5[p].set_ylabel(f'{flabels[i]} FPs')
            
            axs5[p].set_xlabel(param_name)


        


    # plot area vs param
    for j in range(2):
        m,s = get_mean_ci(areas[:,:,j],valida)

        axs2[3,p].plot(param, m, label=alabels[j], color=acolors[j])
        axs2[3,p].fill_between(x=param, y1=m-s, y2=m+s, alpha=0.2, color=acolors[j])
    
    m,s = get_mean_ci(dfarea,valida)
    axs2[3,p].plot(param, m, label='$\Delta$ Area', color='k', linewidth=2)
    axs2[3,p].fill_between(x=param, y1=m-s, y2=m+s, alpha=0.2, color='k')

    # axs2[3,p].axhline(15.334331775422573, color='k', linestyle='--', linewidth=0.5, label='$\Delta$ Ct')
    # axs2[3,p].axhline(10.264073341704066, color='deepskyblue', linestyle='--', linewidth=0.5, label='$\Delta$ Sz')

    axs2[3,0].set_ylabel(f'Area')
    axs2[3,p].set_xlabel(param_name)
    axs2[3,0].legend(fontsize=8)

    if p ==3:
        axs2[3,p].set_xscale('log')
    

    # plot area in square
    for j in range(2):
        m,s = get_mean_ci(areas[:,:,j],valida)

        axs3[p].plot(param, m, label=alabels[j], color=acolors[j])
        axs3[p].fill_between(x=param, y1=m-s, y2=m+s, alpha=0.2, color=acolors[j])
    
    m,s = get_mean_ci(dfarea,valida)
    axs3[p].plot(param, m, label='$\Delta$ Area', color='k', linewidth=2)
    axs3[p].fill_between(x=param, y1=m-s, y2=m+s, alpha=0.2, color='k')

    # axs3[p].axhline(15.334331775422573, color='k', linestyle='--', linewidth=0.5, label='$\Delta$ Ct')
    # axs3[p].axhline(10.264073341704066, color='deepskyblue', linestyle='--', linewidth=0.5, label='$\Delta$ Sz')

    if p == 0 or p == 2:
        axs3[p].set_ylabel(f'Area')
        axs4[p].set_ylabel(f'$\Delta$ Unstable FPs')
    axs3[p].set_xlabel(param_name)
    if p ==0:
        axs3[p].legend(fontsize=8, )
    axs4[p].set_title(param_name)
    if p >1:
        axs4[p].set_xlabel(f'$\Delta$ Area')
    if p ==3:
        axs3[p].set_xscale('log')



handles, labels = axs3[0].get_legend_handles_labels()

# Add a figure-level legend at the bottom of the figure
# f3.legend(handles, labels, loc='lower center', ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.05))



f.tight_layout()
f2.tight_layout()
f3.tight_layout()
f4.tight_layout()
f5.tight_layout()

f.savefig(f'./analysis/R_{N_inits}.svg')
f.savefig(f'./analysis/R_{N_inits}.png', dpi=300)

f2.savefig(f'./analysis/A_{N_inits}.svg')
f2.savefig(f'./analysis/A_{N_inits}.png', dpi=300)

f3.savefig(f'./analysis/A_square_{N_inits}.svg')
f3.savefig(f'./analysis/A_square_{N_inits}.png', dpi=300)

f4.savefig(f'./analysis/R_square_{N_inits}.svg')
f4.savefig(f'./analysis/R_square_{N_inits}.png', dpi=300)

f5.savefig(f'./analysis/FP_square_{N_inits}.svg')
f5.savefig(f'./analysis/FP_square_{N_inits}.png', dpi=300)


