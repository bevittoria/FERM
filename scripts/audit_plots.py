"""Component and metric comparisons for a selected complete local audit."""
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FixedLocator, FuncFormatter, MultipleLocator
from ferm_model import metrics

def _load_comparison(run_name, experiment, fit_alpha, eval_alpha):
    from pathlib import Path
    from datetime import datetime, timezone
    from uuid import uuid4
    import os, sys, json, hashlib
    os.environ.setdefault('MPLCONFIGDIR','/tmp/ferm_asia_audit_mpl')
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, FixedLocator, FuncFormatter, MultipleLocator
    ROOT = Path(__file__).resolve().parents[1]
    from asia_workflow import load_archive
    from ferm_model import rm_law, estimate, metrics
    CDF_RUN=Path(run_name)
    CDF_EXPERIMENT=experiment
    CDF_FIT_ALPHA=fit_alpha;CDF_EVAL_ALPHA=eval_alpha
    assert (CDF_RUN/'COMPLETE.json').exists(), 'Incomplete run'
    protocol=json.loads((CDF_RUN/'protocol.json').read_text())
    SOURCE = ROOT / 'data/asia_2022'
    if not np.isfinite(eval_alpha) or eval_alpha <= 0:
        raise ValueError('eval_alpha must be positive and finite.')
    hashes={f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in protocol['source_sha256']}
    assert hashes==protocol['source_sha256'], 'Inputs have changed since the saved run'
    nodes,distance,raw,train,gdp=load_archive(SOURCE)
    codes=nodes.iso3.tolist();n=len(codes);off=~np.eye(n,dtype=bool)
    assert pd.read_csv(CDF_RUN/'nodes.csv').iso3.tolist()==codes
    rm=rm_law(nodes,distance)
    gravity_frame = pd.read_csv(CDF_RUN/'gravity_probabilities.csv', index_col=0)
    assert gravity_frame.index.tolist() == codes and gravity_frame.columns.tolist() == codes
    gravity_prob = gravity_frame.to_numpy(float)
    assert np.allclose(gravity_prob.sum(1), 1) and np.all(np.diag(gravity_prob) == 0)
    gravity_fit = json.loads((CDF_RUN/'gravity_fit.json').read_text())
    frame=pd.read_csv(SOURCE/'flows_test_h2.csv',index_col=0)
    assert frame.index.tolist()==codes and frame.columns.tolist()==codes
    obs=frame.to_numpy(float);totals=obs.sum(1)
    assert np.isfinite(obs).all() and (obs>=0).all() and obs.sum()>0
    choices=pd.read_csv(CDF_RUN/'all_frozen_choices.csv')
    CURVES=[('GDP','GDP','#64708a',':'),('SCI','SCI','#56b4e9',':'),
            ('Barrier only','Shift','#9467bd',':'),('GDP + SCI','GDP + SCI','#009d9a',':'),
            ('GDP + barrier','GDP + shift','#e69f00',':'),
            ('SCI + barrier','SCI + shift','#9a6642',':'),
            ('GDP + SCI + barrier','GDP + SCI + shift','#d64265','-')]
    flow_models=[('Observed H2',obs,'#171717','-'),('RM',totals[:,None]*rm,'#929292','-'),
                 ('Gravity',totals[:,None]*gravity_prob,'#30743d','-')]
    assert np.allclose((totals[:,None]*gravity_prob).sum(1),totals)
    cdf_selected=[dict(label='Gravity',**metrics(gravity_prob,obs,rm))]
    for label,family,color,style in CURVES:
        chosen=choices[choices.experiment.eq(experiment)&choices.fit_alpha.eq(fit_alpha)
            &choices.group.eq('historical')&choices.gdp_mode.eq('log_z')
            &choices.sci_mode.eq('log_z_origin')&choices['mask'].eq('global')&choices.family.eq(family)]
        assert len(chosen)==1, f'{family}: expected one frozen choice, got {len(chosen)}'
        row=chosen.iloc[0];key=row['key']
        if key=='RM':prob=rm
        else:
            with np.load(CDF_RUN/experiment/f'final_{key}.npz') as bank:counts=bank['counts'].sum(0)
            prob=estimate(counts,rm,eval_alpha)
        predicted=totals[:,None]*prob
        assert np.allclose(predicted.sum(1),totals)
        flow_models.append((label,predicted,color,style))
        cdf_selected.append(dict(label=label,key=key,g=row.g,lam=row.lam,b=row.b,sigma=row.sigma,**metrics(prob,obs,rm)))
    return ROOT, CDF_RUN, hashes, codes, n, off, rm, obs, totals, flow_models, cdf_selected, gravity_prob, gravity_fit, CURVES, distance

def export_component_distributions(run_name, experiment='baseline', fit_alpha=50., eval_alpha=50., *, output_root=None, show=True):
    """Export figures/tables for the selected audit; reuse its figure directory."""
    ROOT, CDF_RUN, hashes, codes, n, off, rm, obs, totals, flow_models, cdf_selected, gravity_prob, gravity_fit, CURVES, distance = _load_comparison(run_name, experiment, fit_alpha, eval_alpha)
    output_root = Path(output_root) if output_root is not None else CDF_RUN.parent / 'figures'
    cdf_unique,inverse=np.unique(distance[off],return_inverse=True)
    def distance_cdf(flows):
        w=np.bincount(inverse,weights=flows[off],minlength=len(cdf_unique))
        return np.cumsum(w)/w.sum()
    cdf_table=pd.DataFrame({'distance_km':cdf_unique,**{label:distance_cdf(flows) for label,flows,_,_ in flow_models}})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':9,
        'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(10,3.9),layout='constrained')
    combined_tables=[]
    for label,flows,color,style in flow_models:
        lw=1.8 if label=='Observed H2' else 1.3
        distance_y=distance_cdf(flows)
        axes[0].step(np.r_[0,cdf_unique],np.r_[0,distance_y],where='post',color=color,ls=style,lw=lw,label=label)
        xx,counts=np.unique(flows[off],return_counts=True);yy=np.cumsum(counts)/len(flows[off])
        axes[1].step(np.r_[0,xx],np.r_[0,yy],where='post',color=color,ls=style,lw=lw,label=label)
        assert np.isclose(distance_y[-1],1) and np.isclose(yy[-1],1)
        combined_tables.append(pd.DataFrame(dict(model=label,flow=xx,cumulative_share=yy)))
    axes[0].set(xlim=(0,cdf_unique.max()*1.015),xlabel='Origin–destination distance (km)',ylabel='Cumulative share of migration flow')
    axes[0].xaxis.set_major_locator(MultipleLocator(3000))
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:,.0f}'))
    axes[1].set_xscale('symlog',linthresh=1)
    upper=max(float(flows[off].max()) for _,flows,_,_ in flow_models)*1.08
    axes[1].set(xlim=(0,upper),xlabel='H2 flow per corridor (migrant count)',ylabel='Cumulative share of corridors')
    axes[1].xaxis.set_major_locator(FixedLocator([0,1,10,100,1000,10000,100000]))
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:,.0f}'))
    for ax,title in zip(axes,['A  Migration distance','B  Corridor flow size']):
        ax.set_title(title,loc='left',fontsize=10,pad=9)
        ax.set_ylim(0,1.025);ax.set_yticks([0,.25,.5,.75,1])
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.grid(axis='y',color='#dddddd',lw=.5,alpha=.7);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False);ax.tick_params(length=3,width=.65,labelsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=5,
               frameon=False,fontsize=7.5,handlelength=2.6,columnspacing=1.6)
    combined_out=output_root/'component_distributions'
    combined_out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:fig.savefig(combined_out/f'distance_and_flow.{ext}',dpi=400,bbox_inches='tight')
    cdf_table.to_csv(combined_out/'distance_cdf.csv',index=False)
    pd.concat(combined_tables,ignore_index=True).to_csv(combined_out/'flow_size_cdf.csv',index=False)
    pd.DataFrame(cdf_selected).to_csv(combined_out/'selected_models.csv',index=False)
    pd.DataFrame(gravity_prob,index=codes,columns=codes).to_csv(combined_out/'gravity_probabilities.csv')
    (combined_out/'gravity_fit.json').write_text(json.dumps(gravity_fit,indent=2))
    (combined_out/'provenance.json').write_text(json.dumps(dict(run=str(CDF_RUN),experiment=experiment,
        fit_alpha=fit_alpha,eval_alpha=eval_alpha,curves=CURVES,source_sha256=hashes,gravity=gravity_fit),indent=2))
    (combined_out / 'caption.tex').write_text(
        '\\caption{Asia audit: seven separately H1-calibrated FERM component specifications, analytic RM and origin-constrained Gravity. Gravity uses $p_{ij}\\propto n_j^{\\beta}d_{ij}^{-\\gamma}$ with nonnegative exponents fitted on H1 only. Predictions condition on observed H2 origin totals. Distance CDFs are flow-weighted; flow-size CDFs give equal weight to all off-diagonal corridors, including observed zeros. Comparisons are descriptive, with no hypothesis tests. '
        + f'{n} locations and {int(off.sum())} corridors; fitting alpha={fit_alpha:g}, evaluation alpha={eval_alpha:g}. '
        + 'Nonneutral FERM probabilities pool the archived final Monte Carlo replicas before regularization.'
        + '}\n')
    print('Export:',combined_out)
    if show: plt.show()
    return combined_out

def plot_metric_comparison(run_name, experiment='baseline', fit_alpha=50., eval_alpha=50., *, output_root=None, show=True):
    """Export figures/tables for the selected audit; reuse its figure directory."""
    ROOT, CDF_RUN, hashes, codes, n, off, rm, obs, totals, flow_models, cdf_selected, gravity_prob, gravity_fit, CURVES, distance = _load_comparison(run_name, experiment, fit_alpha, eval_alpha)
    output_root = Path(output_root) if output_root is not None else CDF_RUN.parent / 'figures'
    cdf_unique,inverse=np.unique(distance[off],return_inverse=True)
    def distance_cdf(flows):
        w=np.bincount(inverse,weights=flows[off],minlength=len(cdf_unique))
        return np.cumsum(w)/w.sum()
    cdf_table=pd.DataFrame({'distance_km':cdf_unique,**{label:distance_cdf(flows) for label,flows,_,_ in flow_models}})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':9,
        'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    from matplotlib.ticker import PercentFormatter
    rows=[]
    obs_distance=distance_cdf(obs)
    obs_sizes=obs[off]
    err_rm=np.abs(np.log10((totals[:,None]*rm[...]+1)/(obs+1)))[off]
    for label,flows,color,style in flow_models:
        if label=='Observed H2':continue
        # Supremum differences of empirical CDFs, descriptive only (no KS p-values).
        distance_gap=float(np.max(np.abs(distance_cdf(flows)-obs_distance)))
        grid=np.union1d(obs_sizes,flows[off])
        actual=np.searchsorted(np.sort(obs_sizes),grid,side='right')/len(obs_sizes)
        predicted=np.searchsorted(np.sort(flows[off]),grid,side='right')/len(obs_sizes)
        size_gap=float(np.max(np.abs(predicted-actual)))
        error=np.abs(np.log10((flows[off]+1)/(obs_sizes+1)))
        fraction=float(((error<err_rm)&~np.isclose(error,err_rm,atol=1e-10,rtol=0)).mean())
        # Read pooled scores rather than reconstruct probabilities from zero-outflow rows.
        if label=='RM':score=metrics(rm,obs,rm)
        else:score=next(r for r in cdf_selected if r['label']==label)
        rows.append(dict(model=label,color=color,distance_gap=distance_gap,size_gap=size_gap,
                         improved_fraction=fraction,gain_bits=score['gain_bits'],cpc=score['cpc']))
    table=pd.DataFrame(rows)
    fig,axes=plt.subplots(1,5,figsize=(12,4.3),sharey=True,layout='constrained')
    columns=['distance_gap','size_gap','improved_fraction','gain_bits','cpc']
    titles=['A  Distance distribution','B  Flow-size distribution','C  Corridors improved','D  Allocation gain','E  Flow overlap']
    labels=['Max. CDF gap\nLower is better','Max. CDF gap\nLower is better','Share improved vs RM\nHigher is better','Bits / migrant vs RM\nHigher is better','CPC\nHigher is better']
    y=np.arange(len(table))
    for ax,col,title,xlabel in zip(axes,columns,titles,labels):
        vals=table[col].to_numpy();best=np.argmin(vals) if col.endswith('gap') else np.argmax(vals)
        for i,(v,c) in enumerate(zip(vals,table.color)):
            ax.barh(i,v,color=c,height=.58,alpha=.85)
            ax.annotate(f'{v:.1%}' if col in ['distance_gap','size_gap','improved_fraction'] else f'{v:.3f}',xy=(v,i),xytext=(7,0),textcoords='offset points',va='center',fontsize=7)
        ax.plot(vals[best],best,'*',color='black',ms=7,zorder=4)
        ax.set_title(title,loc='left',fontsize=8.5,pad=9);ax.set_xlabel(xlabel,fontsize=8)
        lo, hi = min(0., float(vals.min())), max(0., float(vals.max()))
        span = max(hi - lo, 1e-6)
        ax.set_xlim(lo - (.12 * span if lo < 0 else 0), hi + .37 * span);ax.grid(axis='x',alpha=.15,lw=.5);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        if col in ['distance_gap','size_gap','improved_fraction']:ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    axes[0].set_yticks(y,table.model);axes[0].invert_yaxis()
    fig.suptitle('Different diagnostics assess different aspects of allocation',fontsize=10)
    fig.supxlabel('★ Best value among the displayed specifications; descriptive comparisons on H2',fontsize=8)
    out=output_root/'metric_comparison'
    out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:fig.savefig(out/f'metric_comparison.{ext}',dpi=350,bbox_inches='tight')
    table.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(gravity_prob,index=codes,columns=codes).to_csv(out/'gravity_probabilities.csv')
    (out/'gravity_fit.json').write_text(json.dumps(gravity_fit,indent=2))
    (out/'provenance.json').write_text(json.dumps(dict(run=str(CDF_RUN),experiment=experiment,fit_alpha=fit_alpha,
        eval_alpha=eval_alpha,source_sha256=hashes,gravity=gravity_fit,aggregate_metric='supremum absolute CDF difference; descriptive, no test'),indent=2))
    (out / 'caption.tex').write_text(
        '\\caption{Asia audit: seven separately H1-calibrated FERM component specifications, analytic RM and origin-constrained Gravity. Gravity uses $p_{ij}\\propto n_j^{\\beta}d_{ij}^{-\\gamma}$ with nonnegative exponents fitted on H1 only. Predictions condition on observed H2 origin totals. Distance CDFs are flow-weighted; flow-size CDFs give equal weight to all off-diagonal corridors, including observed zeros. Comparisons are descriptive, with no hypothesis tests. Panels show maximum distance-CDF gap, maximum flow-size-CDF gap, fraction of corridors with lower absolute log-ratio error than RM, bits per migrant gained over RM, and CPC. Lower gaps and higher other metrics are better; stars mark best displayed values. '
        + f'{n} locations and {int(off.sum())} corridors; fitting alpha={fit_alpha:g}, evaluation alpha={eval_alpha:g}. '
        + 'Nonneutral FERM probabilities pool the archived final Monte Carlo replicas before regularization.'
        + '}\n')
    print(table.drop(columns='color').to_string(index=False));print('Export:',out)
    if show: plt.show()
    return out

