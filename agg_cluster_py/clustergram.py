import matplotlib.cm as cm
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import numpy as np

def clean_axis(axx):
    '''
    remove axis tick labels
    remove spine, black markers between boxes of image
    '''
    axx.get_xaxis().set_ticks([])
    axx.get_yaxis().set_ticks([])
    for sp in axx.spines.values():
        sp.set_visible(False)

def plot_clustergram(data,clusters,labels=None,savename = None,scale='log'):
    fig = plt.figure(figsize = (12,12))
    heatmap_gs = gs.GridSpec(2,2,wspace=0.0,
                             hspace = 0.0,
                             width_ratios=[1,0.25],
                             height_ratios=[0.25,1])
    #plot dendrogram
    den_ax = fig.add_subplot(heatmap_gs[0,0])
    den = sch.dendrogram(clusters, color_threshold=np.inf)#,ax=den_ax,color_threshold=np.inf)
    
    #heatmap
    ax = fig.add_subplot(heatmap_gs[1,0])
    axi = ax.imshow(data,
                    interpolation='nearest',
                    cmap = cm.coolwarm,
                    origin='lower',aspect='auto')
    
    #colorbar
    scale_ax = fig.add_subplot(heatmap_gs[0,1])
    cb = fig.colorbar(axi,scale_ax)
    if scale == 'log':
        cb.set_label('Log Distance')
    else:
        cb.set_label('Distance')
    cb.ax.yaxis.set_ticks_position('right')
    cb.ax.yaxis.set_label_position('right')
    cb.outline.set_linewidth(0)
    
    #reset axis labels
    clean_axis(ax)
    clean_axis(den_ax)
    #xlabels
    if labels != None:
        x_labels = [labels[w_count] for w_count in den['leaves']]
        ax.set_xticks(np.arange(len(labels)))
        xlabs = ax.set_xticklabels(x_labels)
        for w in xlabs:
            w.set_rotation(90)
        #ylabels
        ax.set_yticks(np.arange(len(labels)))
        ax.yaxis.set_ticks_position('right')
        ax.set_yticklabels(x_labels)
    
    #remove xticks
    for l in ax.get_xticklines() + ax.get_yticklines():
        l.set_markersize(0)
    if savename != None:
        plt.savefig(savename,format='pdf')
    plt.show(block=False)
