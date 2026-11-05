"""Asia data preparation, H1 calibration and independent H2 simulations."""
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import platform
import re
import time
import uuid
import numpy as np
import pandas as pd
from IPython.display import display
from ferm_model import rm_law, job, estimate, bits, metrics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/asia_2022'

@dataclass
class Config:
    sigmas:tuple=(.5,1.,1.5,2.,3.,4.,5.,6.,8.,12.,24.)
    weights:tuple=(.25,.5,.75)
    fit_particles:int=20000
    fit_repeats:int=2
    final_particles:int=200000
    final_repeats:int=3
    pseudo_count:float=50.
    seed:int=20261013
    workers:int=2


def normalize_sci(raw,mode):
    a=np.asarray(raw,float).copy();np.fill_diagonal(a,np.nan)
    observed=np.isfinite(a)
    if (a[observed]<0).any():raise ValueError('SCI must be nonnegative')
    z=np.zeros_like(a);stats=[]
    if mode=='raw_minmax':
        v=a[observed];lo,hi=v.min(),v.max()
        if hi<=lo:raise ValueError('Constant global SCI')
        z[observed]=2*(v-lo)/(hi-lo)-1
        stats=[dict(origin='ALL',count=len(v),center=(lo+hi)/2,scale=(hi-lo)/2,fallback=False)]
    elif mode in ('log_global','log_origin'):
        x=np.log1p(a)
        if mode=='log_global':
            v=x[observed];mean,sd=v.mean(),v.std(ddof=0)
            if sd<=0:raise ValueError('Constant global SCI')
            z[observed]=(v-mean)/sd
            stats=[dict(origin='ALL',count=len(v),center=mean,scale=sd,fallback=False)]
        else:
            for i in range(len(a)):
                mask=observed[i];v=x[i,mask];mean=float(v.mean()) if len(v) else 0.
                sd=float(v.std(ddof=0)) if len(v) else 0.
                fallback=len(v)<2 or sd<1e-12
                if not fallback:z[i,mask]=(v-mean)/sd
                stats.append(dict(origin=i,count=len(v),center=mean,scale=sd,fallback=fallback))
    else:raise ValueError(mode)
    # Missing values are neutral shifts by convention, not raw SCI zero.
    np.fill_diagonal(z,0)
    return z,observed,pd.DataFrame(stats)


def load_archive(source):
    source=Path(source);nodes=pd.read_csv(source/'nodes.csv');codes=nodes.iso3.tolist()
    def matrix(name):
        frame=pd.read_csv(source/name,index_col=0)
        if frame.index.tolist()!=codes or frame.columns.tolist()!=codes:raise ValueError(f'Misaligned {name}')
        return frame.to_numpy(float)
    d=matrix('distances_km.csv');raw=matrix('sci_2021_raw.csv');train=matrix('flows_train_h1.csv')
    logg=np.log1p(nodes.gdp_per_capita_2021.to_numpy());g=(logg-logg.mean())/logg.std(ddof=0)
    return nodes,d,raw,train,np.tile(g,(len(g),1))


@dataclass
class BarrierConfig(Config):
    g_values:tuple=(0.,.125,.2,.25,.5)
    b_values:tuple=(0.,.25,.5,1.,2.)
    lambda_values:tuple=(0.,.125,.25,.5)
    seed:int=20261014


def build_candidates(gdp,z,observed,cfg):
    n=len(z);off=~np.eye(n,dtype=bool)
    candidates=[];families={}
    for mask_name,mask in [('global',off),('observed',observed)]:
        for g in cfg.g_values:
            for b in cfg.b_values:
                for lam in cfg.lambda_values:
                    # b=0 has no dependence on the barrier mask; reuse identical particles.
                    key=(g,b,lam,mask_name if b else 'none')
                    existing=next((c for c in candidates if c['key']==key),None)
                    if existing is None:
                        cid=len(candidates)
                        existing=dict(id=cid,label='RM' if cid==0 else 'additive',matrix=g*gdp-b*mask+lam*z,
                                      sigma=1.,weight=None,variant=mask_name,g=g,b=b,lam=lam,mask=key[-1],key=key)
                        candidates.append(existing)
                    cid=existing['id']
                    if b==0 and lam==0:families.setdefault('GDP',[]).append(cid)
                    if g==0 and b==0:families.setdefault('relation',[]).append(cid)
                    if b==0:families.setdefault('GDP_relation',[]).append(cid)
                    if g==0 and lam==0:families.setdefault('barrier_'+mask_name,[]).append(cid)
                    if g==0:families.setdefault('barrier_relation_'+mask_name,[]).append(cid)
                    if lam==0:families.setdefault('GDP_barrier_'+mask_name,[]).append(cid)
                    families.setdefault('GDP_barrier_relation_'+mask_name,[]).append(cid)
    return candidates,{k:sorted(set(v)) for k,v in families.items()}


def normalize_origin_minmax(raw):
    a=np.asarray(raw,float).copy();np.fill_diagonal(a,np.nan)
    observed=np.isfinite(a);z=np.zeros_like(a);stats=[]
    if (a[observed]<0).any():raise ValueError('SCI must be nonnegative')
    for i in range(len(a)):
        v=a[i,observed[i]]
        lo=float(v.min()) if len(v) else 0.;hi=float(v.max()) if len(v) else 0.
        fallback=len(v)<2 or hi<=lo
        if not fallback:z[i,observed[i]]=2*(v-lo)/(hi-lo)-1
        stats.append(dict(origin=i,count=len(v),minimum=lo,maximum=hi,fallback=fallback))
    return z,observed,pd.DataFrame(stats)

def run_calibration(*, GDP_GRID=(0., .125, .2, .25, .5),
                    BARRIER_GRID=(0., .25, .5, 1., 2.), SCI_GRID=(0., .125, .25, .5),
                    FIT_PARTICLES_PER_ORIGIN=200_000, FIT_REPEATS=10_000,
                    SEED=20261014, N_WORKERS=2, PRIOR_TOTAL=50.,
                    INCLUDE_OBSERVED_MASK_CONTROL=True, RUN_LABEL='asia_calibration',
                    output_root=None, run_dir=None):
    """Fit on H1 only, freeze parameters and return the new run directory."""
    output_root = Path(output_root) if output_root is not None else ROOT / 'outputs'
    def valid_grid(values,name):
        values=tuple(sorted(set(float(v) for v in values)))
        if not values or values[0]!=0 or not all(np.isfinite(v) and v>=0 for v in values):
            raise ValueError(name+': include zero and use only finite nonnegative numbers')
        return values
    for name,v in [('particles',FIT_PARTICLES_PER_ORIGIN),('repeats',FIT_REPEATS),('workers',N_WORKERS)]:
        if isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<1:raise ValueError(name+' must be a positive integer')
    if not isinstance(SEED,(int,np.integer)) or SEED<0:raise ValueError('Invalid seed')
    if not np.isfinite(PRIOR_TOTAL) or PRIOR_TOTAL<=0:raise ValueError('Invalid prior')
    cfg=BarrierConfig(g_values=valid_grid(GDP_GRID,'GDP'),b_values=valid_grid(BARRIER_GRID,'barrier'),
                      lambda_values=valid_grid(SCI_GRID,'SCI'),fit_particles=int(FIT_PARTICLES_PER_ORIGIN),
                      fit_repeats=int(FIT_REPEATS),seed=int(SEED),workers=int(N_WORKERS),pseudo_count=float(PRIOR_TOTAL))
    nodes,distance,raw,train,gdp=load_archive(SOURCE);codes=nodes.iso3.tolist();n=len(codes)
    z,observed,stats=normalize_sci(raw,'log_origin');stats['origin']=codes
    rm=rm_law(nodes,distance)
    candidates,families=build_candidates(gdp,z,observed,cfg)
    if not INCLUDE_OBSERVED_MASK_CONTROL:
        families={f:ids for f,ids in families.items() if not f.endswith('_observed')}
    active_ids=sorted(set(i for ids in families.values() for i in ids))
    print('Origins:',n,'; MC candidates:',len(active_ids)-1,'; fitting runs:',(len(active_ids)-1)*cfg.fit_repeats)
    print('Attempts per origin/candidate:',cfg.fit_particles*cfg.fit_repeats)
    print('H2 has not been loaded. SCI normalization: log-z per origin.')

    # 3. RUN CALIBRATION (a new directory for each run)
    label=re.sub(r'[^a-zA-Z0-9_-]+','_',RUN_LABEL).strip('_') or 'calibration'
    OUT=Path(run_dir) if run_dir is not None else output_root/'asia_custom_calibration'/(label+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'replicas').mkdir()
    settings=dict(g_grid=cfg.g_values,b_grid=cfg.b_values,sci_grid=cfg.lambda_values,sigma=1,
                  particles_per_origin=cfg.fit_particles,repeats=cfg.fit_repeats,seed=cfg.seed,workers=cfg.workers,
                  prior_total=cfg.pseudo_count,observed_mask_control=bool(INCLUDE_OBSERVED_MASK_CONTROL))
    input_paths=[SOURCE/f for f in ['nodes.csv','distances_km.csv','sci_2021_raw.csv','flows_train_h1.csv']]
    input_paths += [Path(__file__), ROOT/'scripts/ferm_model.py']
    protocol=dict(status='running',created_utc=datetime.now(timezone.utc).isoformat(),settings=settings,
                  selection='H1 cross-entropy only, pooled MC counts; exact RM candidate; no H2 read',
                  seed_rule='[seed, 1, candidate_id, replicate_index]',
                  candidate_ids=active_ids,families=families,
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    nodes.to_csv(OUT/'nodes.csv',index=False);stats.to_csv(OUT/'normalization.csv',index=False)
    np.savez_compressed(OUT/'inputs.npz',SCI=z,observed_mask=observed,GDP=gdp,H1=train,distances=distance,RM=rm)
    (OUT/'candidates.json').write_text(json.dumps([{k:v for k,v in candidates[i].items() if k!='matrix'} for i in active_ids],indent=2))
    # RM has no simulated row here; raw arrays contain only nonneutral active candidates.
    ids=[i for i in active_ids if i!=0];positions={cid:k for k,cid in enumerate(ids)}
    counts=np.zeros((len(ids),cfg.fit_repeats,n,n),dtype=np.int64)
    tasks=[(cid,r,nodes,distance,candidates[cid]['matrix'],1.,cfg.fit_particles,[cfg.seed,1,cid,r]) for cid in ids for r in range(cfg.fit_repeats)]
    def record(result):
        cid,r,c=result
        assert (c>=0).all() and (c.sum(axis=1)<=cfg.fit_particles).all() and (np.diag(c)==0).all()
        counts[positions[cid],r]=c
        np.savez_compressed(OUT/'replicas'/f'candidate_{cid:04d}_rep_{r:03d}.npz',counts=c,unaccepted=cfg.fit_particles-c.sum(axis=1))
    print('Output:',OUT,flush=True);start=time.monotonic()
    if cfg.workers==1:
        for done,t in enumerate(tasks,1):
            record(job(t))
            if done%10==0 or done==len(tasks):print(f'Fit MC {done}/{len(tasks)}',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures=[pool.submit(job,t) for t in tasks]
            for done,future in enumerate(as_completed(futures),1):
                record(future.result())
                if done%10==0 or done==len(tasks):print(f'Fit MC {done}/{len(tasks)}',flush=True)
    np.savez_compressed(OUT/'calibration_counts.npz',candidate_ids=np.array(ids,dtype=int),counts=counts,unaccepted=cfg.fit_particles-counts.sum(axis=-1))
    rows=[];probabilities={0:rm}
    for cid in active_ids:
        c=candidates[cid];pooled=counts[positions[cid]].sum(axis=0) if cid else None
        p=estimate(pooled,rm,cfg.pseudo_count) if cid else rm;probabilities[cid]=p
        accepted=pooled.sum(axis=1) if cid else None
        rows.append(dict(candidate=cid,label=c['label'],g=c['g'],b=c['b'],lam=c['lam'],mask=c['mask'],train_bits=bits(p,train),
                         min_accepted=int(accepted.min()) if cid else None,
                         zero_acceptance_origins=int((accepted==0).sum()) if cid else 0,
                         max_prior_weight=float((cfg.pseudo_count/(accepted+cfg.pseudo_count)).max()) if cid else 0.))
    trace=pd.DataFrame(rows).set_index('candidate',drop=False)
    choices={f:min(ids,key=lambda cid:trace.loc[cid,'train_bits']) for f,ids in families.items()}
    frozen={f:{k:v for k,v in candidates[cid].items() if k!='matrix'} for f,cid in choices.items()}
    trace.to_csv(OUT/'training_grid.csv',index=False)
    (OUT/'frozen_choices.json').write_text(json.dumps(frozen,indent=2))
    np.savez_compressed(OUT/'selected_H1_probabilities.npz',**{f:probabilities[cid] for f,cid in choices.items()})
    protocol.update(status='complete',elapsed_seconds=time.monotonic()-start,calibration_runs=len(tasks))
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));(OUT/'COMPLETE.json').write_text(json.dumps(dict(candidates=len(ids),calibration_runs=len(tasks)),indent=2))
    print('Calibration complete. Frozen parameters saved to:',OUT/'frozen_choices.json')

    # 4. CALIBRATION RESULTS AND DIAGNOSTICS
    selected=pd.DataFrame([dict(family=f,**v,train_bits=float(trace.loc[v['id'],'train_bits'])) for f,v in frozen.items()])
    display(selected[['family','g','b','lam','train_bits']])
    display(trace.loc[sorted(set(choices.values())),['g','b','lam','min_accepted','max_prior_weight','zero_acceptance_origins']])
    for row in selected.itertuples():
        edges=[name for name,grid in [('g',cfg.g_values),('b',cfg.b_values),('lam',cfg.lambda_values)] if len(grid)>1 and getattr(row,name)==max(grid)]
        if edges:print('Upper grid boundary:',row.family,edges)
    chosen=frozen['GDP_barrier_relation_global']
    print('Parameters for new independent simulations:')
    print(f"GDP_WEIGHT = {chosen['g']}\nSCI_WEIGHT = {chosen['lam']}\nBARRIER = {chosen['b']}\nSIGMA = 1.0")
    print("SCI_NORMALIZATION = 'log_origin'\nBARRIER_SUPPORT = 'global'")
    print('Use a new seed for subsequent simulations; do not reuse fitting counts as an independent check.')
    # Small parameter file for independent simulations.
    (OUT/'simulation_parameters.json').write_text(json.dumps(dict(GDP_WEIGHT=chosen['g'],SCI_WEIGHT=chosen['lam'],BARRIER=chosen['b'],SIGMA=1.,SCI_NORMALIZATION='log_origin',BARRIER_SUPPORT='global'),indent=2))
    return OUT


def run_simulations(calibration_dir, *, N_REPLICATES=10,
                    PARTICLES_PER_ORIGIN=200_000, SEED=20261016, N_WORKERS=2,
                    PRIOR_TOTAL=50., BAND_LEVEL=.99, RUN_LABEL='asia_simulation',
                    output_root=None, run_dir=None):
    """Simulate frozen H1 parameters, score H2, save CDF bands and return the run directory."""
    calibration_dir = Path(calibration_dir)
    fit_protocol = json.loads((calibration_dir / 'protocol.json').read_text())
    if fit_protocol['status'] != 'complete':
        raise ValueError('Calibration is incomplete.')
    if SEED == fit_protocol['settings']['seed']:
        raise ValueError('Use a seed different from the calibration seed.')
    parameters = json.loads((calibration_dir / 'simulation_parameters.json').read_text())
    GDP_WEIGHT = parameters['GDP_WEIGHT']
    SCI_WEIGHT = parameters['SCI_WEIGHT']
    BARRIER = parameters['BARRIER']
    SIGMA = parameters['SIGMA']
    SCI_NORMALIZATION = parameters['SCI_NORMALIZATION']
    BARRIER_SUPPORT = parameters['BARRIER_SUPPORT']
    output_root = Path(output_root) if output_root is not None else ROOT / 'outputs'
    nodes,distance,raw,train,gdp=load_archive(SOURCE)
    codes=nodes.iso3.tolist();n=len(codes);off=~np.eye(n,dtype=bool)
    test_frame=pd.read_csv(SOURCE/'flows_test_h2.csv',index_col=0)
    assert test_frame.index.tolist()==codes and test_frame.columns.tolist()==codes
    test=test_frame.to_numpy(float);totals=test.sum(axis=1);rm=rm_law(nodes,distance)
    print(f'{n} origins, {off.sum()} corridors. H2: {test.sum():,.0f} observed counts.')

    # 3. RUN NEW SIMULATIONS
    for name,value in [('GDP_WEIGHT',GDP_WEIGHT),('SCI_WEIGHT',SCI_WEIGHT),('BARRIER',BARRIER)]:
        if not np.isfinite(value) or value<0:raise ValueError(name+' must be finite and nonnegative')
    if not np.isfinite(SIGMA) or SIGMA<=0:raise ValueError('SIGMA must be positive')
    for name,value in [('N_REPLICATES',N_REPLICATES),('PARTICLES_PER_ORIGIN',PARTICLES_PER_ORIGIN),('N_WORKERS',N_WORKERS)]:
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1:raise ValueError(name+' must be a positive integer')
    if not isinstance(SEED,(int,np.integer)) or SEED<0:raise ValueError('SEED must be a nonnegative integer')
    if not np.isfinite(PRIOR_TOTAL) or PRIOR_TOTAL<=0:raise ValueError('PRIOR_TOTAL must be positive')
    if not 0<BAND_LEVEL<1:raise ValueError('BAND_LEVEL must be between 0 and 1')
    if BARRIER_SUPPORT not in ['global','observed']:raise ValueError('BARRIER_SUPPORT not recognized')
    if SCI_NORMALIZATION=='origin_minmax':z,observed,normalization_stats=normalize_origin_minmax(raw)
    else:z,observed,normalization_stats=normalize_sci(raw,SCI_NORMALIZATION)
    barrier_mask=off if BARRIER_SUPPORT=='global' else observed
    shifts=GDP_WEIGHT*gdp-BARRIER*barrier_mask+SCI_WEIGHT*z
    assert np.allclose(np.diag(shifts),GDP_WEIGHT*np.diag(gdp))
    config=dict(g=GDP_WEIGHT,lam=SCI_WEIGHT,b=BARRIER,sigma=SIGMA,normalization=SCI_NORMALIZATION,
                barrier_support=BARRIER_SUPPORT,repeats=int(N_REPLICATES),particles_per_origin=int(PARTICLES_PER_ORIGIN),
                seed=int(SEED),workers=int(N_WORKERS),prior_total=float(PRIOR_TOTAL),band_level=float(BAND_LEVEL))
    # Subsequent calculations use this saved configuration, not variables modified after the run.
    label=re.sub(r'[^a-zA-Z0-9_-]+','_',RUN_LABEL).strip('_') or 'run'
    OUT=Path(run_dir) if run_dir is not None else output_root/'asia_custom_mc'/(label+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'replicas').mkdir()
    inputs=[SOURCE/f for f in ['nodes.csv','distances_km.csv','sci_2021_raw.csv','flows_train_h1.csv','flows_test_h2.csv']]
    inputs += [Path(__file__), ROOT/'scripts/ferm_model.py']
    protocol=dict(calibration_dir='../calibration',config=config,status='running',created_utc=datetime.now(timezone.utc).isoformat(),
                  method='direct particle Monte Carlo; user-specified parameters, no fit, no bootstrap',
                  conditional_totals='observed H2 origin totals within retained Asia-to-Asia domain',
                  seed_rule='[seed, replicate_index]',central_curve='median of replicate CDFs; each run regularized separately',
                  interval='pointwise empirical percentiles; numerical variability only, not simultaneous/predictive/parameter uncertainty',
                  python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    nodes.to_csv(OUT/'nodes.csv',index=False);normalization_stats.to_csv(OUT/'normalization_stats.csv',index=False)
    np.savez_compressed(OUT/'inputs.npz',shifts=shifts,sci=z,observed_mask=observed,distances=distance,H1=train,H2=test,RM=rm)
    counts=np.zeros((config['repeats'],n,n),dtype=np.int64)
    start=time.monotonic();print('Output:',OUT,flush=True)
    args=[(0,r,nodes,distance,shifts,config['sigma'],config['particles_per_origin'],[config['seed'],r]) for r in range(config['repeats'])]
    def record(result):
        _,r,c=result
        assert (c>=0).all() and (c.sum(axis=1)<=config['particles_per_origin']).all() and (np.diag(c)==0).all()
        counts[r]=c
        np.savez_compressed(OUT/'replicas'/f'replica_{r:04d}.npz',counts=c,unaccepted=config['particles_per_origin']-c.sum(axis=1))
    if config['workers']==1:
        for k,arg in enumerate(args,1):
            record(job(arg));print(f'Replicate {k}/{len(args)} complete',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=config['workers']) as pool:
            futures=[pool.submit(job,arg) for arg in args]
            for k,future in enumerate(as_completed(futures),1):
                record(future.result());print(f'Replicate {k}/{len(args)} complete',flush=True)
    np.savez_compressed(OUT/'counts.npz',counts=counts,unaccepted=config['particles_per_origin']-counts.sum(axis=2))
    protocol.update(status='simulations_complete',elapsed_seconds=time.monotonic()-start)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(f"Completed {config['repeats']} replicates in {protocol['elapsed_seconds']/60:.1f} minutes.")

    # 4. COMPUTE CURVES AND SAVE RESULTS
    cfg=json.loads((OUT/'protocol.json').read_text())['config']
    with np.load(OUT/'counts.npz') as bank:counts=bank['counts']
    order=np.argsort(distance[off],kind='stable');distance_grid=np.unique(distance[off])
    last=np.searchsorted(distance[off][order],distance_grid,side='right')-1
    size_grid=np.unique(np.r_[0,np.geomspace(.001,max(test.max(),totals.max())*1.01,2000)])
    def cdfs(matrix):
        dist=(np.cumsum(matrix[off][order])/matrix.sum())[last]
        sizes=np.searchsorted(np.sort(matrix[off]),size_grid,side='right')/off.sum()
        return dist,sizes
    D=[];F=[];rows=[]
    for r,c in enumerate(counts):
        p=estimate(c,rm,cfg['prior_total']);a,b=cdfs(totals[:,None]*p);D.append(a);F.append(b)
        rows.append(dict(replicate=str(r),**metrics(p,test,rm)))
    D=np.asarray(D);F=np.asarray(F);q=(1-cfg['band_level'])/2
    assert np.all(np.diff(D,axis=1)>=-1e-12) and np.all(np.diff(F,axis=1)>=-1e-12)
    assert np.allclose(D[:,-1],1) and np.allclose(F[:,-1],1)
    bands=dict(distance_grid=distance_grid,size_grid=size_grid,distance_cdfs=D,size_cdfs=F)
    for name,data in [('distance',D),('size',F)]:
        bands[name+'_low'],bands[name+'_median'],bands[name+'_high']=np.quantile(data,[q,.5,1-q],axis=0)
    np.savez_compressed(OUT/'cdf_bands.npz',**bands)
    pooled=estimate(counts.sum(axis=0),rm,cfg['prior_total'])
    np.savez_compressed(OUT/'pooled_probabilities.npz',P=pooled)
    rows += [dict(replicate='pooled',**metrics(pooled,test,rm)),dict(replicate='RM_exact',**metrics(rm,test,rm))]
    score_table=pd.DataFrame(rows);score_table.to_csv(OUT/'scores.csv',index=False);display(score_table)
    accepted=counts.sum(axis=2)
    diag=dict(min_accepted=int(accepted.min()),zero_acceptance_rows=int((accepted==0).sum()),
              max_prior_weight=float((cfg['prior_total']/(accepted+cfg['prior_total'])).max()),
              distance_max_band_width=float(np.max(bands['distance_high']-bands['distance_low'])),
              size_max_band_width=float(np.max(bands['size_high']-bands['size_low'])))
    print('Diagnostics:',diag)
    if cfg['repeats']<40:print(f"Few replicates to estimate the tails of a {cfg['band_level']:.0%} band; interpret with caution.")
    if diag['zero_acceptance_rows']:print('WARNING: some rows rely entirely on the RM prior. See counts.npz.')
    protocol=json.loads((OUT/'protocol.json').read_text());protocol['status']='complete'
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));(OUT/'COMPLETE.json').write_text(json.dumps(diag,indent=2))
    return OUT


