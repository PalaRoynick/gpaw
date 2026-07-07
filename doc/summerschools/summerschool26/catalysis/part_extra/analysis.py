# Script for analysing energy contributions from vibrations
import numpy as np
from ase.io import read
from ase.thermochemistry import HarmonicThermo
from ase.vibrations import Vibrations
from matplotlib import pyplot as plt


DFT_energy_initial = np.loadtxt('energy_initial.txt')
DFT_energy_final = np.loadtxt('energy_final.txt')
print('DFT reaction energy: ', DFT_energy_final - DFT_energy_initial)

vib_initial = Vibrations(read('tight-N2Ru-top.traj'),
                         indices=(8, 9),
                         nfree=4,
                         name='vib_initial')
vib_initial.read()
adsorbate_initial = HarmonicThermo(vib_initial.get_energies())

vib_final = Vibrations(read('tight-2Nads.traj'),
                       indices=(8, 9),
                       nfree=4,
                       name='vib_final')
vib_final.read()
adsorbate_final = HarmonicThermo(vib_final.get_energies())

# The actual analysis
S_initial = []
S_final = []
U_initial = []
U_final = []
temp = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
for T in temp:
    S_initial.append(-T * adsorbate_initial.get_entropy(T, verbose=False))
    U_initial.append(adsorbate_initial.get_internal_energy(T, verbose=False))
    Efree_initial = adsorbate_initial.get_helmholtz_energy(T, verbose=False)
    S_final.append(-T * adsorbate_final.get_entropy(T, verbose=False))
    U_final.append(adsorbate_final.get_internal_energy(T, verbose=False))
    Efree_final = adsorbate_final.get_helmholtz_energy(T, verbose=False)
    print(f'Reaction free energy at {T} K: ',
          DFT_energy_final + Efree_final - (DFT_energy_initial
                                            + Efree_initial))
plt.plot(temp, S_initial, 'r', label='-T*S ini')
plt.plot(temp, U_initial, 'b', label='U ini')
plt.plot(temp, S_final, 'r--', label='-T*S fin')
plt.plot(temp, U_final, 'b--', label='U fin')
plt.plot(temp, np.array(U_initial) + np.array(S_initial), 'm',
         label='E_free initial')
plt.plot(temp, np.array(U_final) + np.array(S_final), 'm--',
         label='E_free final')
plt.legend()
plt.savefig('vib.png')
