from ase.units import kB
import numpy as np

A = -0.38 / 1000
J = 3.1 / 1000
S = 3 / 2
N = 3
T_c_Ising = 1.52 * S**2 * J / kB
T_c = T_c_Ising * np.tanh(6 / N * np.log(1 - 0.033 * A / J))**(0.25)
print(f'T_c = {T_c:.1f} K')

assert np.isclose(T_c, 36.9, atol=0.1)
