"""Complete normalization, prior, seed and fit-budget audit, using local inputs only.

Numerical protocol extracted from the original Asia normalization audit.
Quick mode reduces budgets/grids but retains every audit category.
"""
from pathlib import Path
import hashlib, json, platform, itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from asia_workflow import load_archive, SOURCE
from ferm_model import rm_law, simulate, estimate, bits, metrics, gravity_terms, fit_loglinear


def run_audit(run_dir, *, profile='quick', seed=20261021):
    """Run all audit experiments and persist raw counts, choices and diagnostics."""
    PROFILE = profile
    if PROFILE not in ('quick', 'production'):
        raise ValueError('profile must be quick or production')
    MASTER_SEED = seed
    FIT_ALPHAS = [5., 50., 500.]
    EVAL_ALPHAS = [5., 50., 500.]
    PRIMARY_ALPHA = 50.
    G_GRID = [0., .125, .2, .25, .5]
    L_GRID = [0., .125, .25, .5]
    B_GRID = [0., .25, .5, 1., 2.]
    MIX_A = [0., .25, .5, .75, 1.]
    MIX_SIGMA = [.5, 1., 1.5, 2., 3., 4., 5., 6., 8., 12., 24.]
    SCI_MODES = ['raw_minmax_global', 'raw_minmax_origin', 'log_z_global',
                 'log_z_origin', 'log_robust_origin', 'rank_gaussian_origin']
    GDP_MODES = ['log_z']
    GROUPS = ['historical', 'matched']
    EXPERIMENTS = [dict(name='baseline', seed=seed, fit_multiplier=1),
                   dict(name='independent_seed', seed=seed+1, fit_multiplier=1),
                   dict(name='larger_fit', seed=seed, fit_multiplier=4)]
    FIT_PARTICLES = 20000
    FIT_REPEATS = 2
    FINAL_PARTICLES = 200000
    FINAL_REPEATS = 3
    if PROFILE == 'quick':
        G_GRID = [0., .125]
        L_GRID = [0., .25]
        B_GRID = [0., 1.]
        MIX_A = [0., .5, 1.]
        MIX_SIGMA = [1., 3.]
        FIT_PARTICLES = 30
        FIT_REPEATS = 1
        FINAL_PARTICLES = 100
        FINAL_REPEATS = 2
    # The quick profile still runs every audit category and all three experiments.
    run = Path(run_dir)
    nodes,distance,raw,train,_=load_archive(SOURCE)
    codes=nodes.iso3.tolist();n=len(codes);off=~np.eye(n,dtype=bool)
    assert n==44 and off.sum()==1892 and len(set(codes))==n
    assert np.isfinite(train).all() and (train>=0).all() and np.diag(train).sum()==0 and train.sum()>0
    assert np.isfinite(distance).all() and (distance>=0).all()
    assert np.isfinite(nodes.population).all() and (nodes.population>=1).all()
    raw=raw.copy();np.fill_diagonal(raw,np.nan)
    observed=np.isfinite(raw)&off
    assert (raw[observed]>=0).all()
    rm=rm_law(nodes,distance)
    input_files=['nodes.csv','distances_km.csv','sci_2021_raw.csv','flows_train_h1.csv','flows_test_h2.csv','protocol.json']
    hashes={f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in input_files}
    coverage=nodes[['iso3','country_name']].copy()
    coverage['observed_SCI']=observed.sum(1);coverage['missing_SCI']=(~observed&off).sum(1)
    print('Missing SCI:',int((~observed&off).sum()),'/',off.sum())


    def transform_vector(v,kind):
        v=np.asarray(v,float)
        if not np.isfinite(v).all(): raise ValueError('Nonfinite observed feature')
        center=0.;scale=0.;fallback=len(v)<2
        if len(v)==0:return v.copy(),dict(center=center,scale=scale,fallback=True)
        if kind.startswith('log_'):v=np.log1p(v)
        if 'minmax' in kind:
            center=(v.max()+v.min())/2;scale=(v.max()-v.min())/2
        elif 'robust' in kind:
            center=np.median(v);scale=(np.quantile(v,.75)-np.quantile(v,.25))/1.349
        elif 'rank_gaussian' in kind:
            fallback=fallback or np.ptp(v)==0
            values=norm.ppf((rankdata(v,method='average')-.5)/len(v))
            return (np.zeros_like(v) if fallback else values),dict(center=None,scale=None,fallback=bool(fallback))
        else:
            center=v.mean();scale=v.std(ddof=0)
        fallback=fallback or scale<1e-12
        return (np.zeros_like(v) if fallback else (v-center)/scale),dict(center=float(center),scale=float(scale),fallback=bool(fallback))

    def sci_transform(mode):
        z=np.zeros((n,n));stats=[]
        if mode.endswith('_origin'):
            for i in range(n):
                mask=observed[i];z[i,mask],s=transform_vector(raw[i,mask],mode)
                stats.append(dict(origin=codes[i],count=int(mask.sum()),**s))
        elif mode.endswith('_global'):
            z[observed],s=transform_vector(raw[observed],mode)
            stats.append(dict(origin='ALL',count=int(observed.sum()),**s))
        else:raise ValueError(mode)
        assert np.isfinite(z).all() and (z[~observed]==0).all()
        return z,pd.DataFrame(stats)

    all_sci=sorted(set(SCI_MODES)|{'raw_minmax_global','raw_minmax_origin','log_z_global','log_z_origin'})
    Z={};norm_tables={}
    for mode in all_sci:Z[mode],norm_tables[mode]=sci_transform(mode)
    X={};gdp_stats=[]
    for mode in sorted(set(GDP_MODES)|{'log_z'}):
        values,s=transform_vector(nodes.gdp_per_capita_2021.to_numpy(float),mode)
        X[mode]=np.tile(values,(n,1));gdp_stats.append(dict(mode=mode,**s))

    candidates={};members=[]
    def register(group,gdp_mode,sci_mode,mask_name,family,g,lam,b,sigma=1.,a=None):
        mask=off if mask_name=='global' else observed
        matrix=g*X[gdp_mode]+lam*Z[sci_mode]-b*mask
        contrast=(matrix-np.diag(matrix)[:,None])/sigma
        key=hashlib.sha256(np.round(contrast,12).tobytes()).hexdigest() if np.any(contrast) else 'RM'
        if key not in candidates:
            candidates[key]=dict(matrix=matrix,sigma=float(sigma))
        members.append(dict(group=group,gdp_mode=gdp_mode,sci_mode=sci_mode,mask=mask_name,
                            family=family,g=float(g),lam=float(lam),b=float(b),sigma=float(sigma),a=a,key=key))
    def additive(group,gdp_mode,sci_mode,lgrid):
        for mask in ['global','observed']:
            for g,lam,b in itertools.product(G_GRID,lgrid,B_GRID):
                fam=['GDP + SCI + shift']
                if lam==0:fam+=['GDP + shift']
                if g==0:fam+=['SCI + shift']
                if g==lam==0:fam+=['Shift']
                if b==0 and mask=='global':
                    fam+=['GDP + SCI']
                    if lam==0:fam+=['GDP']
                    if g==0:fam+=['SCI']
                for f in fam:register(group,gdp_mode,sci_mode,mask,f,g,lam,b)
    if 'historical' in GROUPS:
        for mode in ['log_z_origin','raw_minmax_origin']:
            lgrid=sorted(set(L_GRID+([1.,2.] if mode=='raw_minmax_origin' and PROFILE=='production' else [])))
            additive('historical','log_z',mode,lgrid)
        for mode in ['raw_minmax_global','log_z_global']:
            register('historical','log_z',mode,'global','Mixture',0,0,0)
            for a,sigma in itertools.product(MIX_A,MIX_SIGMA):
                den=np.sqrt(a*a+(1-a)**2)
                register('historical','log_z',mode,'global','Mixture',a/den,(1-a)/den,0,sigma,a)
    if 'matched' in GROUPS:
        for gm,sm in itertools.product(GDP_MODES,SCI_MODES):additive('matched',gm,sm,L_GRID)
    members=pd.DataFrame(members).drop_duplicates()
    family_cols=['group','gdp_mode','sci_mode','mask','family']
    assert len(candidates)>1 and 'RM' in candidates
    assert all('RM' in set(t.key) for _,t in members.groupby(family_cols))
    print('Unique distributions:',len(candidates),'| family memberships:',len(members))
    print('Fit MC calls:',(len(candidates)-1)*FIT_REPEATS*len(EXPERIMENTS))


    run.mkdir(parents=True,exist_ok=False)
    protocol=dict(profile=PROFILE,experiments=EXPERIMENTS,fit_particles=FIT_PARTICLES,fit_repeats=FIT_REPEATS,
        final_particles=FINAL_PARTICLES,final_repeats=FINAL_REPEATS,fit_alphas=FIT_ALPHAS,eval_alphas=EVAL_ALPHAS,
        primary_alpha=PRIMARY_ALPHA,source='data/asia_2022',source_sha256=hashes,groups=GROUPS,
        grids=dict(g=G_GRID,lam=L_GRID,b=B_GRID,a=MIX_A,sigma=MIX_SIGMA),
        sci_modes=SCI_MODES,gdp_modes=GDP_MODES,method='direct MC maxima, fixed threshold, exact RM',
        status='exploratory; smoke is not scientific evidence',
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    protocol['code_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    (run/'protocol.json').write_text(json.dumps(protocol,indent=2))
    members.to_csv(run/'memberships.csv',index=False);nodes.to_csv(run/'nodes.csv',index=False)
    pd.DataFrame(gdp_stats).to_csv(run/'gdp_statistics.csv',index=False)
    for mode in all_sci:
        norm_tables[mode].to_csv(run/f'stats_{mode}.csv',index=False)
        pd.DataFrame(Z[mode],index=codes,columns=codes).to_csv(run/f'feature_{mode}.csv')
    selected_all=[]
    for exp in EXPERIMENTS:
        path=run/exp['name'];path.mkdir()
        attempts=FIT_PARTICLES*exp['fit_multiplier'];fitrows=[]
        for k,(key,cand) in enumerate(candidates.items()):
            if key=='RM':counts=np.zeros((n,n),dtype=np.int64)
            else:
                reps=np.stack([simulate(nodes,distance,cand['matrix'],cand['sigma'],attempts,[exp['seed'],1,r]) for r in range(FIT_REPEATS)])
                counts=reps.sum(0)
                np.savez_compressed(path/f'fit_{key}.npz',counts=reps)
            accepted=counts.sum(1)
            for alpha in FIT_ALPHAS:
                p=rm if key=='RM' else estimate(counts,rm,alpha)
                fitrows.append(dict(key=key,fit_alpha=alpha,train_bits=bits(p,train),
                    min_accepted=None if key=='RM' else int(accepted.min()),
                    zero_acceptance_origins=0 if key=='RM' else int((accepted==0).sum()),
                    max_prior_weight=0 if key=='RM' else float((alpha/(accepted+alpha)).max())))
            pd.DataFrame(fitrows).to_csv(path/'training_scores.csv',index=False)
            if k%20==0:print(exp['name'],'fit',k+1,'/',len(candidates),flush=True)
        fit=pd.DataFrame(fitrows)
        merged=members.merge(fit,on='key',validate='many_to_many')
        chosen=merged.sort_values(['train_bits','g','lam','b','sigma']).groupby(family_cols+['fit_alpha'],sort=False,dropna=False).head(1).copy()
        chosen['experiment']=exp['name']
        # Boundary flags refer to the actual finite family grid, not just a global maximum.
        for par in ['g','lam','b','sigma']:
            limits=members.groupby(family_cols)[par].agg(['min','max']).reset_index()
            chosen=chosen.merge(limits.rename(columns={'min':par+'_min','max':par+'_max'}),on=family_cols)
            chosen[par+'_upper_boundary']=(chosen[par]==chosen[par+'_max'])&(chosen[par+'_max']>chosen[par+'_min'])
        chosen.to_csv(path/'frozen_choices.csv',index=False);selected_all.append(chosen)
        for key in chosen.key.unique():
            if key=='RM':continue
            cand=candidates[key]
            reps=np.stack([simulate(nodes,distance,cand['matrix'],cand['sigma'],FINAL_PARTICLES,[exp['seed'],2,r]) for r in range(FINAL_REPEATS)])
            assert (reps>=0).all() and (reps.sum(2)<=FINAL_PARTICLES).all()
            assert (np.diagonal(reps,axis1=1,axis2=2)==0).all()
            np.savez_compressed(path/f'final_{key}.npz',counts=reps)
        print(exp['name'],'choices frozen; independent simulation complete',flush=True)
    pd.concat(selected_all,ignore_index=True).to_csv(run/'all_frozen_choices.csv',index=False)
    (run/'FROZEN.json').write_text(json.dumps(dict(experiments=[e['name'] for e in EXPERIMENTS])))
    # Fit Gravity using H1 only; persist it so loading/plotting never refits.
    terms = gravity_terms(nodes, pd.DataFrame(distance, index=codes, columns=codes), np.zeros((n,n)))[:,:,:2]
    glp, gravity_fit = fit_loglinear(terms, train, bounds=[(0., None), (0., None)])
    gpd = np.exp(glp)
    pd.DataFrame(gpd, index=codes, columns=codes).to_csv(run/'gravity_probabilities.csv')
    gravity_fit.update(model='origin-constrained gravity', calibration='H1 multinomial likelihood',
        population_exponent=gravity_fit['parameters'][0], distance_decay=gravity_fit['parameters'][1],
        parameter_bounds=[[0., None], [0., None]], uses_gdp_or_sci=False,
        probability_method='analytic conditional softmax; no Monte Carlo or pseudocount')
    (run/'gravity_fit.json').write_text(json.dumps(gravity_fit, indent=2))

    assert (run/'FROZEN.json').exists()
    frame=pd.read_csv(SOURCE/'flows_test_h2.csv',index_col=0)
    assert frame.index.tolist()==codes and frame.columns.tolist()==codes
    test=frame.to_numpy(float)
    assert np.isfinite(test).all() and (test>=0).all() and test.sum()>0 and np.diag(test).sum()==0
    choices=pd.read_csv(run/'all_frozen_choices.csv');scores=[];diag=[];origins=[]
    for (exp,key),group in choices.groupby(['experiment','key']):
        reps=None if key=='RM' else np.load(run/exp/f'final_{key}.npz')['counts']
        pooled=None if reps is None else reps.sum(0)
        for alpha in EVAL_ALPHAS:
            p=rm if key=='RM' else estimate(pooled,rm,alpha)
            assert np.allclose(p.sum(1),1) and np.all(np.diag(p)==0) and (p[off]>0).all()
            base=dict(experiment=exp,key=key,eval_alpha=alpha)
            scores.append(dict(**base,replicate='pooled',**metrics(p,test,rm)))
            if reps is not None:
                for r,c in enumerate(reps):scores.append(dict(**base,replicate=str(r),**metrics(estimate(c,rm,alpha),test,rm)))
                acc=pooled.sum(1)
                for i in range(n):diag.append(dict(**base,origin=codes[i],accepted=int(acc[i]),
                    attempts=FINAL_PARTICLES*FINAL_REPEATS,prior_weight=alpha/(acc[i]+alpha),
                    H2_on_zero_MC=float(test[i,pooled[i]==0].sum())))
            if alpha==PRIMARY_ALPHA:
                prediction=test.sum(1)[:,None]*p;pred_rm=test.sum(1)[:,None]*rm
                err=np.abs(np.log10((prediction+1)/(test+1)));err_rm=np.abs(np.log10((pred_rm+1)/(test+1)))
                terms=np.zeros_like(test);positive=test>0
                terms[positive]=test[positive]*np.log2(p[positive]/rm[positive]);totals=test.sum(1)
                for i in range(n):origins.append(dict(experiment=exp,key=key,origin=codes[i],
                    observed_SCI=int(observed[i].sum()),outflow=totals[i],
                    gain_bits=terms[i].sum()/totals[i] if totals[i]>0 else np.nan,
                    error_reduction=np.sum(test[i]*(err_rm[i]-err[i]))/totals[i] if totals[i]>0 else np.nan))
    score=pd.DataFrame(scores)
    choices.merge(score,on=['experiment','key']).to_csv(run/'all_scores.csv',index=False)
    pd.DataFrame(diag).to_csv(run/'particle_diagnostics.csv',index=False)
    choices.merge(pd.DataFrame(origins),on=['experiment','key']).to_csv(run/'origin_diagnostics.csv',index=False)
    missing=coverage.copy();missing['H2_outflow']=test.sum(1)
    missing.to_csv(run/'coverage.csv',index=False)
    (run/'COMPLETE.json').write_text(json.dumps(dict(profile=PROFILE,missing_corridors=int((~observed&off).sum()),
        missing_H2_flow_share=float(test[~observed&off].sum()/test.sum())),indent=2))
    print('Completed:',run)

    return run
