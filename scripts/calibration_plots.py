"""Calibration heatmaps, loaded entirely from a saved H1 run."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_calibration(run_dir, *, cmap='RdBu', width=8.5, show=True):
    """Plot and export calibration heatmaps without running the model."""
    OUT = Path(run_dir)
    MODEL = 'GDP_barrier_relation_global'
    trace = pd.read_csv(OUT / 'training_grid.csv')
    choices = json.loads((OUT / 'frozen_choices.json').read_text())
    # Shared visual grammar. Edit only here to restyle every figure, including the map.
    STYLE={'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
           'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
           'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
           'lines.linewidth':1.7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
           'savefig.facecolor':'white','figure.facecolor':'white','hatch.linewidth':.25}
    plt.rcParams.update(STYLE)
    WIDTH = width
    exports = []
    def panel(ax, text):
        ax.set_title(text, loc='left', pad=8)
    def save(fig, name):
        for ext in ['png', 'pdf', 'svg']:
            file = OUT / (name + '.' + ext)
            fig.savefig(file, dpi=300, bbox_inches='tight')
            exports.append(file)
        if show:
            plt.show()
    CMAP = cmap
    rm_bits = float(trace.loc[trace.label.eq('RM'), 'train_bits'].iloc[0])

    t = trace[trace['mask'].isin(['global', 'none'])].copy()
    a = (rm_bits - t.groupby(['b', 'lam']).train_bits.min()).unstack('lam')
    b = (rm_bits - t[t.b.eq(choices[MODEL]['b'])].pivot(index='g', columns='lam', values='train_bits'))

    vmin = min(a.min().min(), b.min().min())
    vmax = max(a.max().max(), b.max().max())

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.5), layout='constrained')
    for ax, grid, title, ylabel in [
        (axes[0], a, 'A  Gain over RM: profile over GDP weight', 'Common shift, b'),
        (axes[1], b, f"B  Gain over RM: selected shift: b = {choices[MODEL]['b']:g}", 'GDP weight, g')
    ]:
        im = ax.imshow(grid, origin='lower', aspect='auto', cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(grid.columns)), [f'{v:g}' for v in grid.columns])
        ax.set_yticks(range(len(grid.index)), [f'{v:g}' for v in grid.index])
        ax.set(xlabel='Relation weight, λ', ylabel=ylabel)
        row = choices[MODEL]['b' if ylabel.startswith('Common') else 'g']
        ax.plot(list(grid.columns).index(choices[MODEL]['lam']), list(grid.index).index(row), '*', ms=11, mfc='white', mec='black', mew=.6)
        panel(ax, title)

    fig.colorbar(im, ax=axes, label='H1 gain over RM (bits / count)', shrink=.85)
    save(fig, '01_calibration')
    return fig


