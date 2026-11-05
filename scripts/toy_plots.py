"""Missing-relation toy figures, regenerated from saved results only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm


def plot_run(run, *, world=0, mask=0, coverages=(0., .1, .5, 1.),
             heatmap_cmap='viridis_r', error_cmap='RdBu_r', save_files=True, show=True):
    """Export fixed-truth matrices and world-level coverage summaries; no simulation."""
    OUT = Path(run)
    if not (OUT/'COMPLETE.json').is_file():
        raise ValueError('Load a complete toy run before plotting.')
    USED = json.loads((OUT/'protocol.json').read_text())['config']
    WORLD, MASK = world, mask
    SHOW_COVERAGES = list(coverages)
    if not 0 <= world < USED['worlds'] or not 0 <= mask < USED['masks']:
        raise ValueError('world or mask is outside the saved run limits')
    if not SHOW_COVERAGES or any(q not in USED['coverages'] for q in SHOW_COVERAGES):
        raise ValueError('Choose coverages from the saved values: '+str(USED['coverages']))
    HEATMAP_CMAP, ERROR_CMAP = heatmap_cmap, error_cmap
    SAVE = save_files
    FIG = OUT/'figures'
    FIG.mkdir(exist_ok=True)
    scores = pd.read_csv(OUT/'scores.csv')
    primary = scores[scores.prior.eq(USED['prior'])].copy()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
     'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
     'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'lines.linewidth':1.9,
     'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'})
    def save(fig,name):
        if SAVE:
            for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=300,bbox_inches='tight')
        if show: plt.show()
    def tidy(ax):ax.grid(alpha=.15,lw=.6);ax.set_axisbelow(True)

    # One unchanged target, identical ordering and scales for every panel.
    with np.load(OUT/'worlds'/f'world_{WORLD:03d}.npz') as bank:
        truth=bank['truth'];order=np.argsort(bank['xy'][:,0],kind='stable')
        matrices=[truth];labels=['Generating law'];errors=[]
        for q in SHOW_COVERAGES:
            qi=USED['coverages'].index(q);pred=bank[f'prediction_m{MASK}_q{qi}']
            matrices.append(pred);errors.append(pred-truth)
            actual=bank[f'known_m{MASK}_q{qi}'].sum()/(len(truth)*(len(truth)-1))
            labels.append(f'{actual:.0%} relations observed')
    positive=np.concatenate([p[p>0] for p in matrices])
    norm=LogNorm(vmin=positive.min(),vmax=positive.max())
    limit=max(float(np.max(np.abs(e))) for e in errors);limit=max(limit,1e-12)
    errornorm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    fig,axes=plt.subplots(2,len(matrices),figsize=(13.5,5.6),layout='constrained')
    cmap=plt.get_cmap(HEATMAP_CMAP).copy();cmap.set_bad('#eeeeee')
    for k,(matrix,label) in enumerate(zip(matrices,labels)):
        view=matrix[np.ix_(order,order)]
        im=axes[0,k].imshow(np.ma.masked_less_equal(view,0),cmap=cmap,norm=norm,interpolation='nearest')
        axes[0,k].set_title(label,pad=7)
        axes[0,k].set_xticks([]);axes[0,k].set_yticks([])
        if k==0:
            axes[1,k].axis('off');axes[1,k].text(.05,.5,'Same generating law\nfor all coverage levels.\n\nGrey: zero MC probability\nor diagonal.',transform=axes[1,k].transAxes,va='center',fontsize=9)
        else:
            err=errors[k-1][np.ix_(order,order)]
            em=axes[1,k].imshow(err,cmap=ERROR_CMAP,norm=errornorm,interpolation='nearest')
            axes[1,k].set_title('Prediction − truth',pad=7)
            axes[1,k].set_xticks([]);axes[1,k].set_yticks([])
            axes[1,k].set_xlabel('Destination')
    axes[0,0].set_ylabel('Origin');axes[1,1].set_ylabel('Origin')
    fig.colorbar(im,ax=axes[0,:],label='Conditional destination probability (log scale)',shrink=.75)
    fig.colorbar(em,ax=axes[1,1:],label='Signed probability error',shrink=.75)
    fig.suptitle('Fixed truth | Reconstruction from missing relations'+(' | PILOT' if USED['profile']=='quick' else ''),fontsize=11)
    save(fig,'01_fixed_truth_reconstruction')

    # Average masks within each world first: masks are NOT independent worlds.
    metrics=['CPC','KL','known_MAE','hidden_MAE','audit_MAE']
    world_values=primary.groupby(['world','coverage'])[metrics].mean().reset_index()
    def summarise(metric):
        return world_values.groupby('coverage')[metric].agg(mean='mean',low=lambda x:x.quantile(.1),high=lambda x:x.quantile(.9))
    fig,axes=plt.subplots(1,3,figsize=(10.8,4.3),layout='constrained')
    for ax,metric,color,label in [(axes[0],'CPC','#ee1245','All corridors'),
     (axes[1],'known_MAE','#21918c','Known relations'),(axes[1],'hidden_MAE','#64708a','Hidden relations'),
     (axes[2],'audit_MAE','#ee1245','Same hidden audit corridors')]:
        s=summarise(metric).dropna()
        ax.plot(100*s.index,s['mean'],marker='o',ms=4,color=color,label=label)
        ax.fill_between(100*s.index,s.low,s.high,color=color,alpha=.2,lw=0)
    axes[0].set(ylabel='Flow overlap (CPC)',ylim=(0,1.04),title='A  Overall reconstruction')
    axes[0].axhline(1,color='#606060',ls=':',lw=1)
    axes[1].set(ylabel='Mean absolute probability error',title='B  Known and hidden relations')
    axes[2].set(ylabel='Mean absolute probability error',title='C  Fixed hidden corridors')
    axes[1].set_ylim(bottom=0);axes[2].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel('Observed relations (%)');tidy(ax);ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Partial input, unchanged truth'+(' | PILOT' if USED['profile']=='quick' else ''),fontsize=11)
    save(fig,'02_coverage_and_reconstruction')
    if SAVE:world_values.to_csv(FIG/'world_level_metrics.csv',index=False)
    print('Panel B supports change with coverage; panel c keeps the same hidden routes through 75%.')
    print('Full coverage excluded from panel c: those audit relations are then observed.')

    return FIG
