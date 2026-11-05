"""Fixed-truth missing-relation toy: local simulation, validation and run storage."""
from pathlib import Path
from datetime import datetime, timezone
import json, uuid, hashlib
import numpy as np
import pandas as pd
from IPython.display import display
from ferm_model import simulate, rm_law, estimate, bits

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'Fixed truth, partially observed relation matrix. Version 1, 2026-10-16.\nFor each world, generate geography, populations and one iid Gaussian off-diagonal\nrelation matrix R, zero diagonal. No empirical covariates. The coefficient strength\nis fixed; S*=strength*R and sigma=1. Population supplies opportunity counts.\nTruth is an unsmoothed finite MC conditional law (abort zero acceptance origins).\nGenerate training and test once per world, reuse exactly for every mask/coverage.\nMasking changes only the input, never S*, truth probabilities, or observations.\nPer origin use independent random destination permutations, independent of R/flows.\nReveal the first floor(q*(n-1)) entries: nested masks, same count per row.\nUnknown entries are zero shifts by modelling convention. No rescaling after masking.\nThe visible input equals R at known entries. Calibrate scalar theta on H1-like train\ncross-entropy; theta=0 is analytic RM. This is not matrix completion: unseen\ncoefficients are not inferred. Known benefits can indirectly affect hidden-route\nprobabilities through shared threshold/search and conditional normalization.\nUse independent truth/train/test/fit/check streams and common random numbers\nacross candidates/coverage within a mask for fit/check. Endpoints q=0/1 are shared\nacross masks, preventing artificial mask dependence at endpoints.\nFit/check candidate probabilities use 50 total RM pseudo-counts per origin.\nStore all raw fit counts and selected check counts. Prior sensitivity 5/50/500\nholds the selected theta fixed; it does not test calibration robustness.\nMetrics: origin-equally weighted generating-law CPC and KL, independent test\ncross-entropy, MAE of probabilities on known and hidden entries WITHOUT row\nrenormalization. Known/hidden supports change with coverage, so direct comparison\nis descriptive. A fixed audit subset (last 20% per-row ranks) stays hidden through\n75% coverage: report its MAE to assess changes on identical hidden corridors.\nAt full coverage the audit set becomes known and is excluded from that curve.\nAggregate mask results within world, then mean and 10–90% between worlds.\nMasks are not independent worlds. No bootstrap, no quadrature, no p-values.\nMain view: fixed true matrix alongside predictions at 0/10/50/100% requested\ncoverage, fixed node ordering (x coordinate) and colour scales; signed errors\nbelow. First world and first mask are fixed in advance, no best-case selection.\nSecond view: full allocation CPC, known/hidden MAE, and fixed-hidden audit MAE.\nNo guaranteed monotonic improvement, generalization or RM fallback claims.\nNew unique output directory and protocol saved before simulation. Pilot is a\nnumerical/implementation check, not a paper result. No automatic paper run.\n'

def validate_config(c):
    for name in ['worlds','nodes','masks','truth_particles','fit_particles','fit_repeats',
                 'check_particles','check_repeats','train_per_origin','test_per_origin']:
        if isinstance(c[name], bool) or not isinstance(c[name], int) or c[name] <= 0:
            raise ValueError(name + ' must be a positive integer')
    if isinstance(c['seed'], bool) or not isinstance(c['seed'], int) or c['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if c['nodes'] < 6:
        raise ValueError('nodes must be at least 6')
    for name in ['sigma', 'prior']:
        if not np.isfinite(c[name]) or c[name] <= 0:
            raise ValueError(name + ' must be positive and finite')
    if not np.isfinite(c['strength']) or c['strength'] < 0:
        raise ValueError('strength must be finite and nonnegative')
    for name in ['coverages', 'theta_grid']:
        values = c[name]
        if not values or not all(np.isfinite(x) and x >= 0 for x in values):
            raise ValueError(name + ': finite nonnegative values required')
        if list(values) != sorted(set(values)) or values[0] != 0:
            raise ValueError(name + ': unique increasing values starting at zero')
    if c['coverages'][-1] != 1:
        raise ValueError('coverages must end at 1')
    if not 0 < c['audit_fraction'] <= .25:
        raise ValueError('audit_fraction must be in (0, .25]')
    if int(np.ceil((1-c['audit_fraction'])*(c['nodes']-1))) >= c['nodes']-1:
        raise ValueError('The audit subset is empty: increase nodes or audit_fraction')
    if c['profile'] not in ('quick', 'production'):
        raise ValueError('profile must be quick or production')


def evaluate_prediction(truth,pred,test,known,audit):
    off=~np.eye(len(truth),dtype=bool);positive=truth>0
    err=np.abs(pred-truth)
    def avg(mask):return float(err[mask].mean()) if mask.any() else np.nan
    return dict(CPC=float(np.minimum(truth,pred).sum()/len(truth)),
      KL=float(np.sum(truth[positive]*np.log2(truth[positive]/pred[positive]))/len(truth)),
      test_cross_entropy=float(bits(pred,test)),known_MAE=avg(known),
      hidden_MAE=avg(off&~known),audit_MAE=avg(audit) if not known[audit].any() else np.nan,
      known_routes=int(known.sum()),hidden_routes=int((off&~known).sum()))

def fixed_truth_world(w,c,out):
    n=c['nodes'];off=~np.eye(n,dtype=bool)
    def seed(stage,mask=0,k=0):return [c['seed'],w,stage,mask,k]
    rng=np.random.default_rng(seed(0))
    xy=rng.uniform(size=(n,2));distance=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
    population=np.clip(np.rint(rng.lognormal(8,1,n)),100,100000).astype(int)
    nodes=pd.DataFrame({'iso3':[f'X{i:03d}' for i in range(n)],'population':population})
    relation=rng.normal(size=(n,n));np.fill_diagonal(relation,0.)
    truth_shift=c['strength']*relation
    rm=rm_law(nodes,distance)
    counts=simulate(nodes,distance,truth_shift,c['sigma'],c['truth_particles'],seed(1))
    accepted=counts.sum(1)
    if (accepted==0).any():raise RuntimeError('Truth origin without acceptances: increase truth budget in a new run.')
    truth=counts/accepted[:,None]
    def obs(stage,N):
        rng=np.random.default_rng(seed(stage))
        return np.stack([rng.multinomial(N,truth[i]) for i in range(n)])
    train=obs(2,c['train_per_origin']);test=obs(3,c['test_per_origin'])
    arrays=dict(xy=xy,distance=distance,population=population,relation=relation,
                truth_shift=truth_shift,truth=truth,truth_counts=counts,train=train,test=test,RM=rm)
    rows=[];traces=[];endpoint_cache={}
    for m in range(c['masks']):
        rng=np.random.default_rng(seed(4,m));rank=np.full((n,n),n,dtype=int)
        for i in range(n):
            js=np.flatnonzero(off[i]);js=rng.permutation(js);rank[i,js]=np.arange(n-1)
        audit=off&(rank>=int(np.ceil((1-c['audit_fraction'])*(n-1))))
        assert audit.any()
        arrays[f'rank_mask{m}']=rank;arrays[f'audit_mask{m}']=audit
        previous=np.zeros((n,n),bool)
        for qi,q in enumerate(c['coverages']):
            known=off&(rank<int(np.floor(q*(n-1)+1e-9)))
            assert not (previous&~known).any();previous=known
            if q<=.75:assert not known[audit].any()
            visible=np.where(known,relation,0.)
            arrays[f'known_m{m}_q{qi}']=known
            # The endpoints have identical inputs for every mask.
            cache_key=q if q in (0.,1.) else None
            if cache_key in endpoint_cache:
                best,check=endpoint_cache[cache_key]
            else:
                losses=[];ids=[0] if q==0 else range(len(c['theta_grid']))
                for j in ids:
                    theta=c['theta_grid'][j]
                    if theta==0:
                        pred=rm;fit=None
                    else:
                        stream_mask=0 if q==1 else m
                        fit=np.stack([simulate(nodes,distance,theta*visible,c['sigma'],c['fit_particles'],seed(5,stream_mask,r)) for r in range(c['fit_repeats'])])
                        arrays[f'fit_m{m}_q{qi}_j{j}']=fit
                        pred=estimate(fit.sum(0),rm,c['prior'])
                    loss=float(bits(pred,train));losses.append((loss,j))
                    traces.append(dict(world=w,mask=m,q=q,candidate=j,theta=theta,train_bits=loss,
                     min_accepted=int(fit.sum(0).sum(1).min()) if fit is not None else np.nan,
                     max_prior_weight=float((c['prior']/(fit.sum(0).sum(1)+c['prior'])).max()) if fit is not None else 0.))
                _,best=min(losses)
                if best==0:check=None
                else:
                    stream_mask=0 if q==1 else m
                    check=np.stack([simulate(nodes,distance,c['theta_grid'][best]*visible,c['sigma'],c['check_particles'],seed(6,stream_mask,r)) for r in range(c['check_repeats'])])
                if cache_key is not None:endpoint_cache[cache_key]=(best,check)
            if check is not None:arrays[f'check_m{m}_q{qi}']=check
            for prior in sorted({5.,c['prior'],500.}):
                pred=rm.copy() if best==0 else estimate(check.sum(0),rm,prior)
                assert np.allclose(pred.sum(1),1) and np.allclose(np.diag(pred),0)
                score=evaluate_prediction(truth,pred,test,known,audit)
                assert 0<=score['CPC']<=1+1e-12 and score['KL']>=-1e-10
                rows.append(dict(world=w,mask=m,q=q,coverage=float(known.sum()/off.sum()),prior=prior,
                  theta=c['theta_grid'][best],neutral=best==0,**score,
                  min_check_accepted=int(check.sum(0).sum(1).min()) if check is not None else np.nan,
                  max_check_prior_weight=float((prior/(check.sum(0).sum(1)+prior)).max()) if check is not None else 0.))
                if prior==c['prior']:arrays[f'prediction_m{m}_q{qi}']=pred
        print(f'World {w+1}/{c["worlds"]}, mask {m+1}/{c["masks"]} complete',flush=True)
    np.savez_compressed(out/'worlds'/f'world_{w:03d}.npz',**arrays)
    pd.DataFrame(rows).to_csv(out/'worlds'/f'world_{w:03d}_scores.csv',index=False)
    pd.DataFrame(traces).to_csv(out/'worlds'/f'world_{w:03d}_fit.csv',index=False)
    return rows,traces

def run_fixed_truth(c):
    validate_config(c)
    out=ROOT/'outputs'/('missing_relations_'+c['profile']+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    (out/'worlds').mkdir(parents=True,exist_ok=False)
    sources=[ROOT/'scripts/ferm_model.py', Path(__file__), ROOT/'scripts/toy_plots.py', ROOT/'notebooks/missing_relations_toy.ipynb']
    protocol=dict(config=c,status='running',created_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'frozen_protocol.md').write_text(PROTOCOL)
    try:
        all_rows=[];all_traces=[]
        for w in range(c['worlds']):
            rows,traces= fixed_truth_world(w,c,out);all_rows+=rows;all_traces+=traces
        pd.DataFrame(all_rows).to_csv(out/'scores.csv',index=False)
        pd.DataFrame(all_traces).to_csv(out/'fit_trace.csv',index=False)
        check_run(out)
        protocol['result_sha256'] = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in out.rglob('*') if p.is_file() and p.suffix in ('.npz', '.csv')}
        protocol['status']='complete';(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        (out/'COMPLETE.json').write_text(json.dumps({'worlds':c['worlds'],'masks_per_world':c['masks']}))
    except Exception as error:
        protocol.update(status='failed', error=str(error))
        (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
        raise
    return out


def check_run(run):
    """Verify nested masks, shared endpoints, observation totals and diagnostics."""
    OUT = Path(run)
    USED = json.loads((OUT/'protocol.json').read_text())['config']
    scores = pd.read_csv(OUT/'scores.csv')
    primary = scores[scores.prior.eq(USED['prior'])].copy()
    # Protocol checks: fixed truth/data, nested masks, shared endpoints, no leakage into inputs.
    for w in range(USED['worlds']):
        with np.load(OUT/'worlds'/f'world_{w:03d}.npz') as z:
            n=len(z['truth']);off=~np.eye(n,dtype=bool)
            assert np.all(z['train'].sum(1)==USED['train_per_origin'])
            assert np.all(z['test'].sum(1)==USED['test_per_origin'])
            for m in range(USED['masks']):
                prev=np.zeros((n,n),bool)
                for qi,q in enumerate(USED['coverages']):
                    known=z[f'known_m{m}_q{qi}'];assert not (prev&~known).any();prev=known
                    assert (known.sum(1)==int(np.floor(q*(n-1)+1e-9))).all()
                    if q<=.75:assert not known[z[f'audit_mask{m}']].any()
                for qi in [0,len(USED['coverages'])-1]:
                    assert np.array_equal(z[f'prediction_m{m}_q{qi}'],z[f'prediction_m0_q{qi}'])
    trace=pd.read_csv(OUT/'fit_trace.csv')
    print('All fixed-truth and nesting checks passed.')
    print('Minimum fit acceptances:',trace.min_accepted.min())
    print('Maximum fit prior weight:',trace.max_prior_weight.max())
    print('Minimum final acceptances:',primary.min_check_accepted.min())
    print('Maximum final prior weight:',primary.max_check_prior_weight.max())
    print('Selected upper grid boundary:',(primary.theta==max(USED['theta_grid'])).mean())
    print('Prior sensitivity (selection fixed):')
    display(scores.groupby(['q','prior'])[['CPC','KL']].mean())
    print('Output:',OUT)



def list_runs():
    """List this toy's saved runs; never include Asia runs."""
    rows = []
    for p in sorted((ROOT/'outputs').glob('missing_relations_*/protocol.json')):
        meta = json.loads(p.read_text())
        rows.append(dict(name=p.parent.name, profile=meta['config']['profile'], status=meta['status']))
    return rows


def load_run(name):
    if not name:
        raise ValueError('Set RUN_NAME to the name of a saved toy run.')
    out = (ROOT/'outputs'/name).resolve()
    if out.parent != (ROOT/'outputs').resolve() or not out.name.startswith('missing_relations_'):
        raise ValueError('RUN_NAME must identify a toy run in outputs/.')
    if not (out/'COMPLETE.json').exists():
        raise ValueError(f'Run not found or incomplete: {out}. Available: {list_runs()}')
    meta = json.loads((out/'protocol.json').read_text())
    if meta['status'] != 'complete':
        raise ValueError('Incomplete run')
    for name, digest in meta['result_sha256'].items():
        p = out/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Missing or modified result: '+name)
    print('Run loaded, no simulations:', out)
    return out


def run_or_load(*, mode='new', run_name=None, config=None):
    if mode == 'load':
        return load_run(run_name)
    if mode != 'new' or config is None:
        raise ValueError("Use mode='new' with config, or mode='load' with run_name")
    out = run_fixed_truth(dict(config))
    print('To reuse: MODE="load", RUN_NAME=', repr(out.name))
    return out
