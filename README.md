# FERM — Asia 2022 and missing-relations toy

This project is self-contained: all required code and inputs are included.
Each notebook has four code cells (imports, parameters, run/load, figures),
with Markdown explanations.

## Getting started

With Python 3.10 or later, from the project directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab notebooks/asia.ipynb
```

Open `notebooks/missing_relations_toy.ipynb` for the synthetic experiment.
Use the Python kernel associated with this environment.

## Creating or loading a run

- `MODE = 'new'`: compute and save a new run without overwriting existing runs.
- `MODE = 'load'`: set `RUN_NAME` to a directory name returned by `list_runs()`.
  Load a complete run and verify saved results without sampling or refitting.

`PROFILE = 'quick'` reduces grids and particle budgets to check the workflow;
these results are not production evidence. `PROFILE = 'production'` uses the
original budgets. The profile and calculation parameters are ignored when
loading: the saved configuration applies.

The plotting cell can be rerun independently. It updates files in the same
directories instead of creating duplicate output directories. Failed or
incomplete runs cannot be loaded as complete runs.

## Asia experiment

`CALIBRATION` and `SIMULATION` expose the main experiment parameters. The
notebook explicitly shows the reductions applied in quick mode.

The production calibration uses 200,000 attempts per origin and 10,000
replicates per candidate on the original grids. This requires substantial
runtime, memory and disk space. The main evaluation uses 10 independent
replicates of 200,000 attempts per origin.

The audit is a separate protocol: production uses 20,000 attempts and two
replicates for fitting, followed by 200,000 attempts and three replicates for
evaluation. Its settings are defined in `scripts/audit_workflow.py`, controlled
by `PROFILE` and `AUDIT_SEED`; changes to `CALIBRATION` and `SIMULATION` do not
change audit budgets.

### Audits included in every new Asia run

- Six SCI transformations: global/origin min-max, global/origin log-z,
  origin log-robust scaling and origin Gaussianized ranks.
- Historical families, including mixtures, and matched-grid comparisons.
- Global barriers and barriers restricted to corridors with observed SCI.
- Fitting and evaluation pseudocount sensitivity: 5, 50 and 500.
- `baseline`, `independent_seed` and `larger_fit` (fourfold H1 budget).
- Raw counts, frozen choices, acceptance diagnostics and origin diagnostics.
- Gravity with two nonnegative exponents, fitted on H1 only and saved.

Quick mode retains every audit category and all three experiments with
smaller budgets. No external historical archive or saved-result subset in
`data/` is used.

### One directory per Asia run

```text
outputs/<run_name>/
  run.json             configuration, status and hashes
  COMPLETE.json        present only after all stages finish
  calibration/         H1 fit, frozen parameters and heatmap
  simulation/          independent counts, H2 scores, CDFs and map
  audit/               three experiments, sensitivity results and saved Gravity
  figures/             audit component and five-metric comparisons
```

`EXPERIMENT`, `FIT_ALPHA` and `EVAL_ALPHA` select the audit comparison to plot.
Gravity is not refitted while plotting. Changing `FIT_ALPHA` selects a saved
calibration; changing `EVAL_ALPHA` regularizes the saved final counts.

H1 selects parameters; H2 evaluates destination allocation conditional on
observed origin totals. Bands show pointwise numerical Monte Carlo variability,
not predictive uncertainty. Aggregate metrics are descriptive diagnostics.

### Asia code and data

- `scripts/ferm_model.py`: FERM sampler, exact RM, metrics and Gravity benchmark.
- `scripts/asia_workflow.py`: main Asia calibration and simulation.
- `scripts/audit_workflow.py`: complete audit execution.
- `scripts/asia_run.py`: entry point for creating, loading and plotting runs.
- `scripts/calibration_plots.py`, `scripts/test_plots.py`,
  `scripts/audit_plots.py`: plotting functions.
- `data/asia_2022/`: inputs and provenance protocol, without model results.
  `nodes.csv` contains `population` and `gdp_per_capita_2021`, both referring
  to 2021, for the 44 locations in the analysis.
- `data/naturalearth/`: map geometries.

Historical paths in the input protocol are provenance only, not dependencies.

## Missing-relations toy

`notebooks/missing_relations_toy.ipynb` generates all synthetic inputs locally;
it does not use Asia CSV files or external directories. Its `list_runs()`
lists only toy runs.

- `scripts/toy_workflow.py`: world generation, nested masks, theta calibration,
  independent verification, prior sensitivity, protocol checks and loading
  with result-hash verification.
- `scripts/toy_plots.py`: truth/prediction/error matrices and coverage curves.
- `scripts/ferm_model.py`: numerical model shared with Asia.

Results are saved in `outputs/missing_relations_<profile>_<id>/`. The `worlds/`
directory contains counts and matrices for each world; `scores.csv` and
`fit_trace.csv` contain metrics; `figures/` contains PNG/PDF/SVG exports.
Rerunning plots updates the same files. Loading does not generate simulations.

The explicit `CONFIG` dictionary controls the experiment:

| Profile | Worlds | Nodes | Masks | Truth particles | Fit particles | Check particles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| quick | 2 | 8 | 2 | 5,000 | 200 | 500 |
| production | 30 | 40 | 5 | 3,000,000 | 20,000 | 100,000 |

Particle budgets are per origin. Fitting and checking each use two replicates.
With few nodes, actual coverage can differ substantially from requested
coverage; figures report actual coverage.

Prior sensitivity holds theta fixed. Masks are averaged within each world
before computing between-world percentiles. Truth is an unsmoothed finite
Monte Carlo law that remains unchanged across coverage levels. The audit
subset stays hidden through 75% coverage. Missing coefficients are not
reconstructed. No bootstrap, quadrature or hypothesis tests are used.

## Verification

Both quick workflows were executed in isolated copies under `/tmp`. Asia
checks covered every audit category and loading/plotting with fitting and
sampling disabled. Toy results matched the original numerical implementation;
loading, plotting without simulation and detection of modified results were
also checked. Full production runs were not launched for verification.
