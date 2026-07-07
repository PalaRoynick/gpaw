# creates: layers.svg, kpts.svg, ecut.svg
from matplotlib import pyplot as plt
from pathlib import Path
import json

results = json.loads(Path('results.json').read_text())
heights = {}
eads = {}
for n, k, ecut, h, e in results:
    heights[(n, k, ecut)] = h
    eads[(n, k, ecut)] = e

N = range(1, 10)
h = [heights[(n, 7, 400)] for n in N]
ea = [eads[(n, 7, 400)] for n in N]
fig, axs = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0)
axs[0].plot(N, h)
axs[1].plot(N, ea)
axs[0].set_ylabel('height [Å]')
axs[1].set_ylabel('ads. energy [eV]')
axs[1].set_xlabel('number of layers')
# plt.show()
plt.savefig('layers.svg')

# %%
K = range(4, 18)
h = [heights[(2, k, 400)] for k in K]
ea = [eads[(2, k, 400)] for k in K]
fig, axs = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0)
axs[0].plot(K, h)
axs[1].plot(K, ea)
axs[0].set_ylabel('height [Å]')
axs[1].set_ylabel('ads. energy [eV]')
axs[1].set_xlabel('number of k-points')
plt.savefig('kpts.svg')

# %%
E = range(350, 801, 50)
h = [heights[(2, 7, e)] for e in E]
ea = [eads[(2, 7, e)] for e in E]
fig, axs = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0)
axs[0].plot(E, h)
axs[1].plot(E, ea)
axs[0].set_ylabel('height [Å]')
axs[1].set_ylabel('ads. energy [eV]')
axs[1].set_xlabel('plane-wave cutoff energy [eV]')
plt.savefig('ecut.svg')
