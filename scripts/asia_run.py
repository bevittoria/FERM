"""One entry point for creating, loading and plotting a complete Asia run."""
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def _hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def list_runs():
    """List completed and incomplete runs without running calculations."""
    rows = []
    for path in sorted((ROOT / 'outputs').glob('*/run.json')):
        meta = json.loads(path.read_text())
        rows.append(dict(name=path.parent.name, profile=meta['profile'], status=meta['status']))
    return rows


def load_run(name):
    """Load a complete local run, checking inputs and saved numerical results."""
    if not name:
        raise ValueError('Set RUN_NAME to the name of a run in outputs/.')
    run = (ROOT / 'outputs' / name).resolve()
    if run.parent != (ROOT / 'outputs').resolve():
        raise ValueError('RUN_NAME must be a directory name within outputs/.')
    if not (run / 'COMPLETE.json').is_file():
        raise ValueError(f'Run not found or incomplete: {run}. Available runs: {list_runs()}')
    meta = json.loads((run / 'run.json').read_text())
    if meta['status'] != 'complete':
        raise ValueError('Incomplete run.')
    if meta['input_sha256'] != _hashes(ROOT / 'data'):
        raise ValueError('Local inputs do not match the saved run.')
    for section in ('calibration', 'simulation', 'audit'):
        if not (run / section / 'COMPLETE.json').is_file():
            raise ValueError(f'Incomplete section: {section}')
    for name, digest in meta['result_sha256'].items():
        file = run / name
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Missing or modified result: {name}')
    print('Run loaded (no calculations):', run)
    return run


def run_or_load(*, mode='new', run_name=None, profile='quick', workers=2,
                calibration=None, simulation=None, audit_seed=20261021):
    """Create all numerical results once, or load a completed run with no fitting."""
    if mode == 'load':
        return load_run(run_name)
    if mode != 'new' or profile not in ('quick', 'production'):
        raise ValueError("Use mode='new'/'load' and profile='quick'/'production'.")
    from asia_workflow import run_calibration, run_simulations
    from audit_workflow import run_audit
    cal = dict(N_WORKERS=workers)
    sim = dict(N_WORKERS=workers, BAND_LEVEL=.95)
    if profile == 'quick':
        cal.update(GDP_GRID=(0., .25), BARRIER_GRID=(0., .25), SCI_GRID=(0., .25),
                   FIT_PARTICLES_PER_ORIGIN=500, FIT_REPEATS=2)
        sim.update(N_REPLICATES=3, PARTICLES_PER_ORIGIN=500)
    cal.update(calibration or {})
    sim.update(simulation or {})
    if {'run_dir', 'output_root'} & (cal.keys() | sim.keys()):
        raise ValueError('Output directories are managed by the unified run.')
    label = re.sub(r'[^a-zA-Z0-9_-]+', '_', run_name or profile).strip('_') or profile
    run = ROOT / 'outputs' / f'{label}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}'
    run.mkdir(parents=True, exist_ok=False)
    meta = dict(status='running', profile=profile, created_utc=datetime.now(timezone.utc).isoformat(),
                input_sha256=_hashes(ROOT / 'data'), code_sha256=_hashes(ROOT / 'scripts'),
                overrides=dict(calibration=cal, simulation=sim, audit_seed=audit_seed))
    def save():
        (run / 'run.json').write_text(json.dumps(meta, indent=2))
    save()
    try:
        print('New complete workflow:', run, flush=True)
        run_calibration(**cal, run_dir=run / 'calibration')
        run_simulations(run / 'calibration', **sim, run_dir=run / 'simulation')
        run_audit(run / 'audit', profile=profile, seed=audit_seed)
        meta['result_sha256'] = {
            str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
            for section in ('calibration', 'simulation', 'audit')
            for p in (run / section).rglob('*') if p.is_file()
        }
        meta['status'] = 'complete'
        save()
        (run / 'COMPLETE.json').write_text(json.dumps(dict(status='complete')))
    except Exception as error:
        meta.update(status='failed', error=str(error))
        save()
        raise
    print('Run complete. To reuse it: MODE="load", RUN_NAME=', repr(run.name))
    return run


def plot_run(run, *, experiment='baseline', fit_alpha=50., eval_alpha=50., cmap='RdBu',
             max_runs_to_show=30, show=True):
    """Rebuild figures from this run only; no sampling, optimization or fitting."""
    from calibration_plots import plot_calibration
    from test_plots import plot_test, plot_map
    from audit_plots import export_component_distributions, plot_metric_comparison
    run = Path(run)
    if not (run / 'COMPLETE.json').exists():
        raise ValueError('Complete or load a run before plotting.')
    plot_calibration(run / 'calibration', cmap=cmap, show=show)
    plot_test(run / 'simulation', max_runs_to_show=max_runs_to_show, show=show)
    plot_map(run / 'simulation', cmap=cmap, show=show)
    options = dict(experiment=experiment, fit_alpha=fit_alpha, eval_alpha=eval_alpha,
                   output_root=run / 'figures', show=show)
    export_component_distributions(run / 'audit', **options)
    plot_metric_comparison(run / 'audit', **options)
    print('Figures for run:', run.name)
