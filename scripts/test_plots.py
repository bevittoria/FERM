"""H2 CDF diagnostics and origin map, using only saved run results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter, FuncFormatter, FixedLocator
import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

def plot_test(run_dir, *, max_runs_to_show=30, show=True):
    """Plot and export distance/flow-size CDFs and numerical variability."""
    OUT = Path(run_dir)
    with np.load(OUT / 'inputs.npz') as bank:
        test = bank['H2'].copy()
        distance = bank['distances'].copy()
        rm = bank['RM'].copy()
    totals = test.sum(axis=1)
    # 5. PLOT — red FERM, dashed grey RM, black observations

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'svg.fonttype': 'none'
    })

    # Use the completed run.

    cfg = json.loads((OUT / 'protocol.json').read_text())['config']
    with np.load(OUT / 'cdf_bands.npz') as bank:
        bands = {k: bank[k] for k in bank.files}
    with np.load(OUT / 'pooled_probabilities.npz') as bank:
        pooled = bank['P']

    # Reconstruct CDFs using the grids saved in the run
    distance_grid = bands['distance_grid']
    size_grid = bands['size_grid']
    off = ~np.eye(test.shape[0], dtype=bool)
    order = np.argsort(distance[off], kind='stable')
    last = np.searchsorted(distance[off][order], distance_grid, side='right') - 1

    def cdfs(matrix):
        dist = (np.cumsum(matrix[off][order]) / matrix.sum())[last]
        sizes = np.searchsorted(np.sort(matrix[off]), size_grid, side='right') / off.sum()
        return dist, sizes
    fig, panels = plt.subplots(
        2, 2,
        figsize=(11, 6.8),
        sharex="col",
        gridspec_kw={"height_ratios": [3, 1.3]},
        layout="constrained"
    )
    axes = panels[0]
    residual_axes = panels[1]
    observed_cdfs=cdfs(test);rm_cdfs=cdfs(totals[:,None]*rm)
    for k,(ax,name) in enumerate(zip(axes,['distance','size'])):
        grid=bands[name+'_grid']
        ax.plot(grid,observed_cdfs[k],color='#252525',lw=1.9,label='Observed')
        ax.plot(grid,rm_cdfs[k],color='#606060',ls='--',lw=1.9,label='RM')
        ax.plot(grid,bands[name+'_median'],color='#ee1245',lw=1.9,label='FERM: median across runs')
        ax.fill_between(grid,bands[name+'_low'],bands[name+'_high'],step='post',color='#ee1245',alpha=.5,lw=0)
        ax.set_ylim(0,1.025);ax.set_yticks([0,.25,.5,.75,1]);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:,.0f}'));ax.grid(alpha=.15);ax.legend(frameon=False,loc='upper left',fontsize=8)
    axes[0].set_xscale('log');axes[0].xaxis.set_major_locator(FixedLocator([100,300,1000,3000,10000]));axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:,.0f}'))
    axes[0].set(xlim=(distance_grid.min()*.95,distance_grid.max()*1.04),xlabel='Origin–destination distance (km)',ylabel='Cumulative share of migration flow',title='A  Migration distance')
    axes[1].set_xscale('symlog',linthresh=1);axes[1].xaxis.set_major_locator(FixedLocator([0,1,10,100,1000,10000,100000]));axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:,.0f}'))
    axes[1].set(xlim=(0,max(test.max(),(totals[:,None]*pooled).max())*1.05),xlabel='Flow per corridor (H2 migrant count)',ylabel='Cumulative share of corridors',title='B  Migration flow size')
    # Lower panels: Monte Carlo variability relative to the median.
    # Preserve the statistical width of the bands.
    MAX_RUNS_TO_SHOW = max_runs_to_show

    for k, name in enumerate(["distance", "size"]):
        ax = residual_axes[k]
        grid = bands[name + "_grid"]
        median = bands[name + "_median"]

        # Differences in percentage points.
        lower = 100 * (bands[name + "_low"] - median)
        upper = 100 * (bands[name + "_high"] - median)

        # A subset of replicates, selected without searching for extreme cases.
        # The band still uses every archived replicate.
        replicate_key = name + "_cdfs"
        if replicate_key in bands:
            curves = bands[replicate_key]
            indices = np.linspace(
                0, len(curves) - 1,
                min(MAX_RUNS_TO_SHOW, len(curves)),
                dtype=int
            )
            for j, index in enumerate(indices):
                ax.plot(
                    grid,
                    100 * (curves[index] - median),
                    color="#64708a",
                    lw=0.45,
                    alpha=0.25,
                    label="Individual MC runs" if j == 0 else None,
                    zorder=1
                )

        ax.fill_between(
            grid, lower, upper,
            step="post",
            color="#ee1245",
            alpha=0.4,
            linewidth=0,
            label=f"{cfg['band_level']:.0%} pointwise MC range",
            zorder=2
        )
        ax.axhline(
            0, color="#252525", lw=0.8, ls="--",
            label="Median", zorder=3
        )

        # Band-based scale, symmetric around zero.
        # Replicates outside the band may extend beyond the panel.
        limit = max(np.max(np.abs(lower)), np.max(np.abs(upper)))
        limit = max(1.15 * limit, 1e-4)
        ax.set_ylim(-limit, limit)

        ax.set_ylabel("Deviation from median\n(percentage points)", fontsize=9)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax.grid(alpha=0.15)
        ax.set_title(
            "C  Monte Carlo variation" if k == 0 else "D  Monte Carlo variation",
            loc="left", fontsize=10
        )

    # Move x-axis labels to the lower panels.
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)

    residual_axes[0].set_xlabel("Origin–destination distance (km)")
    residual_axes[1].set_xlabel("Flow per corridor (H2 migrant count)")
    residual_axes[0].legend(
        frameon=False, fontsize=7, loc="upper left", ncol=2
    )
    fig.suptitle(f"Asia-to-Asia H2 2022 | g={cfg['g']:g}, λ={cfg['lam']:g}, b={cfg['b']:g}, σ={cfg['sigma']:g}")
    for ext in ['png','pdf','svg']:fig.savefig(OUT/('distance_and_flow_size.'+ext),dpi=300,bbox_inches='tight')
    if show: plt.show()
    print('Results saved to:',OUT)
    for name in ["distance", "size"]:
        width = bands[name + "_high"] - bands[name + "_low"]
        print(
            name,
            "maximum width:",
            100 * width.max(),
            "percentage points"
        )
    return fig


def plot_map(run_dir, *, cmap='RdBu', width=7.2, save=True, show=True):
    """Plot origin-specific weighted absolute log-error reduction over RM."""
    OUT = Path(run_dir)
    with np.load(OUT / 'inputs.npz') as bank:
        test = bank['H2'].copy()
        distance = bank['distances'].copy()
        rm = bank['RM'].copy()
    totals = test.sum(axis=1)
    codes = pd.read_csv(OUT / 'nodes.csv').iso3.tolist()
    with np.load(OUT / 'pooled_probabilities.npz') as bank:
        pooled = bank['P'].copy()
    ERROR_MAP_CMAP = cmap
    ERROR_MAP_WIDTH = width
    SAVE_MAP = save

    # Data and probabilities from the same run as the CDFs.
    scatter_dir = OUT
    scatter_root = Path(__file__).resolve().parents[1]
    scatter_codes = codes
    scatter_observed = test
    scatter_rm = rm
    scatter_ferm = pooled
    error_map_totals = scatter_observed.sum(axis=1)
    assert (error_map_totals > 0).all()
    error_map_pred_rm = error_map_totals[:, None] * scatter_rm
    error_map_pred_ferm = error_map_totals[:, None] * scatter_ferm
    error_map_error_rm = np.abs(np.log10((error_map_pred_rm + 1) / (scatter_observed + 1)))
    error_map_error_ferm = np.abs(np.log10((error_map_pred_ferm + 1) / (scatter_observed + 1)))
    error_map_terms = scatter_observed * (error_map_error_rm - error_map_error_ferm)
    np.fill_diagonal(error_map_terms, 0.)
    error_map_gain = error_map_terms.sum(axis=1) / error_map_totals
    assert np.isfinite(error_map_gain).all()
    assert np.isclose(np.average(error_map_gain, weights=error_map_totals),
                      error_map_terms.sum() / error_map_totals.sum())
    error_map_nodes = pd.read_csv(scatter_dir / 'nodes.csv')
    assert error_map_nodes.iso3.tolist() == list(scatter_codes)
    error_map_scores = error_map_nodes[['iso3', 'country_name']].copy()
    error_map_scores['gain_abs_log10'] = error_map_gain
    error_map_scores['observed_outflow'] = error_map_totals
    error_map_scores['weighted_error_RM'] = (scatter_observed * error_map_error_rm).sum(axis=1) / error_map_totals
    error_map_scores['weighted_error_FERM'] = (scatter_observed * error_map_error_ferm).sum(axis=1) / error_map_totals
    error_map_geometry = scatter_root / 'data/naturalearth/ne_50m_admin_0_countries.zip'
    error_map_detail_geometry = error_map_geometry.with_name('ne_10m_admin_0_countries.zip')

    def load_origin_map(path):
        world = gpd.read_file(path)
        world = world.to_crs('EPSG:4326')
        world['iso3'] = world.ISO_A3_EH
        assert set(scatter_codes) <= set(world.iso3)
        return world.merge(error_map_scores, on='iso3', how='left', validate='many_to_one')

    error_map_world = load_origin_map(error_map_geometry)
    error_map_detail = load_origin_map(error_map_detail_geometry)
    error_map_lo, error_map_hi = min(0., float(error_map_gain.min())), max(0., float(error_map_gain.max()))
    # TwoSlopeNorm requires strict limits around zero, including for a future all-positive run.
    error_map_span = max(abs(error_map_lo), abs(error_map_hi), 1e-6)
    error_map_norm = TwoSlopeNorm(vcenter=0.,
                            vmin=error_map_lo if error_map_lo < 0 else -error_map_span,
                            vmax=error_map_hi if error_map_hi > 0 else error_map_span)
    ERROR_MAP_CRS = '+proj=aea +lat_1=15 +lat_2=50 +lat_0=30 +lon_0=90 +datum=WGS84 +units=m'

    def origin_error_map_region(ax, world, bounds, crs):
        footprint = box(*bounds)
        crop = world[world.intersects(footprint)].copy()
        crop.geometry = crop.geometry.intersection(footprint)
        crop = crop[~crop.is_empty].to_crs(crs)
        crop.plot(ax=ax, color='#f0f0f0', edgecolor='#bcbcbc', linewidth=.25)
        missing = crop[crop.CONTINENT.eq('Asia') & crop.gain_abs_log10.isna()]
        if len(missing):
            missing.plot(ax=ax, color='#ededed', edgecolor='#bcbcbc', linewidth=.25, hatch='////')
        observed = crop[crop.gain_abs_log10.notna()]
        if len(observed):
            observed.plot(ax=ax, column='gain_abs_log10', cmap=ERROR_MAP_CMAP, norm=error_map_norm,
                          edgecolor='white', linewidth=.35)
        frame = gpd.GeoSeries([footprint.segmentize(1)], crs='EPSG:4326').to_crs(crs)
        xmin, ymin, xmax, ymax = frame.total_bounds
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect('equal')
        ax.axis('off')
        return Transformer.from_crs('EPSG:4326', crs, always_xy=True)

    error_map_fig = plt.figure(figsize=(ERROR_MAP_WIDTH, 5.4))
    error_map_ax = error_map_fig.add_axes([.025, .34, .95, .60])
    error_map_transform = origin_error_map_region(error_map_ax, error_map_world, (24, -12, 147, 59), ERROR_MAP_CRS)
    error_map_ax.set_title('A  Origin-specific error reduction over RM', loc='left', pad=8, fontsize=10)
    for name, lon, lat in [('India',79,22), ('Kazakhstan',68,48), ('Saudi Arabia',45,23),
                           ('Japan',139,37), ('Indonesia',117,-4), ('Mongolia',104,46)]:
        xx, yy = error_map_transform.transform(lon, lat)
        error_map_ax.text(xx, yy, name, ha='center', va='center', fontsize=7, color='#171717',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.75, pad=.7))
    for position, bounds, title in [
        ([.06,.145,.25,.17], (46.5,22,57.5,31), 'B  Gulf'),
        ([.38,.145,.25,.17], (113.4,22.05,114.5,22.6), 'C  Hong Kong / Macao'),
        ([.70,.145,.25,.17], (103.55,1.12,104.12,1.6), 'D  Singapore')]:
        inset = error_map_fig.add_axes(position)
        origin_error_map_region(inset, error_map_detail, bounds,
                          f'+proj=merc +lon_0={(bounds[0]+bounds[2])/2} +datum=WGS84 +units=m')
        inset.set_title(title, loc='left', pad=8, fontsize=10)
    error_map_cax = error_map_fig.add_axes([.25,.06,.5,.018])
    error_map_cb = error_map_fig.colorbar(ScalarMappable(norm=error_map_norm, cmap=ERROR_MAP_CMAP),
                              cax=error_map_cax, orientation='horizontal')
    error_map_ticks = np.unique(np.r_[np.linspace(error_map_norm.vmin, 0., 3),
                                np.linspace(0., error_map_norm.vmax, 3)])
    error_map_cb.set_ticks(error_map_ticks)
    error_map_cb.ax.set_xticklabels([f'{v:.2f}' for v in error_map_ticks], fontsize=7)
    error_map_cb.set_label('Reduction in weighted absolute log10 error', fontsize=9)
    error_map_cb.outline.set_visible(False)
    error_map_cb.ax.axvline(0, color='black', lw=.8)
    error_map_fig.legend(handles=[Patch(facecolor='#ededed', edgecolor='#bcbcbc', hatch='////',
                                  label='Outside analysed sample (Asia)')],
                   loc='upper right', bbox_to_anchor=(.97,.96), frameon=False, fontsize=7)
    if show: plt.show()
    print('Run:', scatter_dir)
    print('Origins with lower weighted absolute log10 error:', int((error_map_gain > 0).sum()),
          'of', len(error_map_gain), '; range:', error_map_gain.min(), error_map_gain.max())
    print('Weighted mean gain:', np.average(error_map_gain, weights=error_map_totals), 'absolute log10 error units')
    if SAVE_MAP:
        error_map_scores.to_csv(scatter_dir / 'origin_gain_abs_log10.csv', index=False)
        for ext in ['png', 'pdf', 'svg']:
            error_map_fig.savefig(scatter_dir / f'05_origin_absolute_log_error_map.{ext}', dpi=300, bbox_inches='tight')
        print('Map and table saved in:', scatter_dir)

    return error_map_fig


