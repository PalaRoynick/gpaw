# web-page: simultaneous.png

from matplotlib import pyplot, gridspec


def makefig():
    fig = pyplot.figure(figsize=(6.5, 6.5))
    grid = gridspec.GridSpec(2, 1, fig, right=0.95, top=0.95)
    smallgrid = gridspec.GridSpecFromSubplotSpec(2, 1, grid[0], hspace=0.)
    ax0 = fig.add_subplot(smallgrid[0])
    ax1 = fig.add_subplot(smallgrid[1], sharex=ax0)
    smallgrid = gridspec.GridSpecFromSubplotSpec(2, 1, grid[1], hspace=0.)
    ax2 = fig.add_subplot(smallgrid[0], sharex=ax0)
    ax3 = fig.add_subplot(smallgrid[1], sharex=ax0)
    return fig, ax0, ax1, ax2, ax3


def get_data(mode):
    with open(f'Au111-H-{mode}.txt') as f:
        lines = f.read().splitlines()
    potentials = []
    equilibrateds = []
    fmaxs = []
    switch = None
    for line in lines:
        # One line per DFT calculation: the potential the potentiostat read
        # off after it.  "Potential found to be ..." will not do here -- the
        # new code logs that once per *equilibration*, reporting which attempt
        # succeeded, so it skips exactly the intermediate attempts that this
        # figure exists to show.
        if 'excess electrons:' in line and 'potential:' in line:
            potentials += [float(line.split('potential:')[-1].split('V')[0])]
        # always_adjust is on only in the first, loose-tolerance optimization,
        # so the last calculation that logs it is the one after which the
        # script tightened the criteria and went back to sequential mode.
        if 'always_adjust:' in line:
            switch = len(potentials)
        if 'Potential is within tolerance. Equilibrated.' in line:
            diff = [False] * (len(potentials) - len(equilibrateds))
            diff[-1] = True
            equilibrateds.extend(diff)
    with open(f'qn-Au111-H-{mode}.log') as f:
        lines = f.read().splitlines()
    previous = None
    for line in lines:
        if 'BFGS:' not in line:
            continue
        energy, fmax = line.split()[3], line.split()[4]
        # The second optimization restarts from the geometry the first one
        # converged to.  If tightening the potential tolerance changes
        # nothing, its step 0 reuses the cached forces: no DFT calculation
        # runs, no potential is logged, and the line repeats the previous one
        # verbatim.  Skip it, or there is one more force here than there are
        # calculations to plot it against.
        if (energy, fmax) == previous:
            continue
        fmaxs += [float(fmax)]
        previous = (energy, fmax)
    return potentials, equilibrateds, fmaxs, switch


def plot_data(potentials, equilibrateds, fmaxs, axes):
    fmax_indices = []
    axes[0].plot(potentials, 'o-', color='k', markerfacecolor='w')
    for index, (potential, equilibrated) in enumerate(zip(potentials,
                                                          equilibrateds)):
        if equilibrated:
            axes[0].plot(index, potential, 'og', markeredgecolor='k')
            fmax_indices += [index]
    axes[1].plot(fmax_indices, fmaxs, 'ko-')


def shade_plot(ax, xstart):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between(x=[xstart, xlim[-1]], y1=[ylim[0]] * 2, y2=[ylim[1]] * 2,
                    color='0.8')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


fig, ax0, ax1, ax2, ax3 = makefig()
potentials, equilibrateds, fmaxs, _ = get_data('seq')
plot_data(potentials, equilibrateds, fmaxs, [ax0, ax1])
potentials, equilibrateds, fmaxs, switch = get_data('sim')
plot_data(potentials, equilibrateds, fmaxs, [ax2, ax3])

ax3.set_xlabel('DFT calculation')
ax0.text(0.20, 0.75, 'sequential', transform=ax0.transAxes)
ax2.text(0.20, 0.75, 'simultaneous', transform=ax2.transAxes)
ax2.text(0.75, 0.75, '(sequential)', transform=ax2.transAxes)
shade_plot(ax2, switch - 0.5)
shade_plot(ax3, switch - 0.5)
for ax in [ax0, ax2]:
    ax.set_ylabel(r'$\phi$, V')
for ax in [ax1, ax3]:
    ax.set_ylabel(r'max force, eV/$\mathrm{\AA}$')

fig.savefig('simultaneous.png')
