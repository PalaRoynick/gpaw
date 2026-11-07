"""Plot the Al benchmark in the convention of Chaput et al., Figure 2."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--reference', type=Path)
    a = p.parse_args()
    data = np.genfromtxt(a.directory / 'couplings.csv',
                         delimiter=',', names=True)
    saved = np.load(a.directory / 'matrix-elements.npz')
    x = data['distance'] / data['distance'].max()
    ticks = saved['ticks'] / saved['ticks'][-1]
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    colors = dict(total='black', V='tab:blue',
                  P='tab:orange', D='tab:green', R='tab:red')
    for key in colors:
        ax.plot(x, data[key], '.-', color=colors[key], lw=1, ms=3,
                label=f'GPAW {key}')
    summary = json.loads((a.directory / 'summary.json').read_text())
    alignment = summary.get('K_highest_mode_longitudinal_overlap')
    branch_mismatch = alignment is not None and alignment < .8
    comparison = dict(status='Unconverged diagnostic; no accuracy claim',
                      branch_order_mismatch=branch_mismatch,
                      K_highest_mode_longitudinal_overlap=alignment,
                      reference='Figure 2 plotted curves, not raw author data',
                      metrics={})
    if a.reference:
        reference = np.genfromtxt(a.reference, delimiter=',', names=True,
                                  dtype=None, encoding='utf-8')
        for key in ('total', 'V', 'P'):
            for repeat in (2, 3, 4):
                select = ((reference['component'] == key)
                          & (reference['conventional_repeat'] == repeat))
                rx = reference['path_fraction'][select]
                ry = reference['coupling_eV'][select]
                ax.plot(rx, ry, color=colors[key], alpha=.2, lw=1)
                if repeat == 4:
                    # Exclude endpoints with branch jumps and Gamma zero modes.
                    mask = np.min(abs(x[:, None] - ticks), axis=1) > .003
                    expected = np.interp(x[mask], rx, ry)
                    delta = data[key][mask] - expected
                    comparison['metrics'][key] = dict(
                        rms_error_eV=float(np.sqrt(np.mean(delta**2))),
                        max_error_eV=float(np.max(abs(delta))))
        ax.plot([], [], color='grey', alpha=.5,
                label='PRB Fig. 2 (2, 3, 4 cubic repeats)')
    for t in ticks:
        ax.axvline(t, color='grey', lw=.5, alpha=.5)
    labels = [r'$\Gamma$', 'X', 'W', 'K', r'$\Gamma$', 'L', 'U', 'X']
    title = ('Al PAW coupling: Eq. (97), initial Gamma t2g triplet\n'
             'Diagnostic settings; convergence not established')
    if branch_mismatch:
        title += '\nK phonon ordering differs from the paper'
    ax.set(xticks=ticks, xticklabels=labels,
           ylabel=r'$\sqrt{\langle |g|^2\rangle}$ (eV)',
           xlim=(0, 1), ylim=(0, None), title=title)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(a.directory / 'couplings.png', dpi=180)
    fig.savefig(a.directory / 'couplings.pdf')
    (a.directory / 'comparison.json').write_text(
        json.dumps(comparison, indent=2) + '\n')
    fig2, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                              layout='constrained')
    bands = saved['energies']
    chosen_band = saved['final_index']
    chosen_mode = saved['mode_index']
    index = np.arange(len(x))
    axes[0].plot(x, bands, color='black', alpha=.5, lw=.7)
    axes[0].plot(x, bands[index, chosen_band], color='limegreen', lw=2,
                 label='Selected final band')
    axes[0].set_ylim(bands[:, 0].min() - .5,
                     bands[0, saved['initial']].max() + 2)
    axes[0].set_ylabel('Electron energy (eV, code gauge)')
    axes[0].legend()
    phonons = saved['phonon_energies'] / .004135667696  # eV to THz
    axes[1].plot(x, phonons, color='black', lw=.8)
    axes[1].plot(x, phonons[index, chosen_mode], color='limegreen', lw=2,
                 label='Selected phonon')
    axes[1].set(xticks=ticks, xticklabels=labels,
                ylabel='Phonon frequency (THz)', xlim=(0, 1))
    axes[1].legend()
    for axis in axes:
        for t in ticks:
            axis.axvline(t, color='grey', alpha=.5, lw=.5)
    fig2.savefig(a.directory / 'selected-bands.png', dpi=180)


if __name__ == '__main__':
    main()
