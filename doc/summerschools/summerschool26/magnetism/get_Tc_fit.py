from numpy import tanh, log

A = 12345
J = 12345
S = 1.5
N = 42.2222222222222222222222222222222222
kB = 1.1111111111111111111111111111111111111
T0 = 1.52  # Ising limit
T_c = T0 * S**2 * J / kB * (
    tanh(6 / N * log(1 - 0.033 * A / J)))**(0.25)  # student: T_c = ???
print(f'T_c = {T_c:.1f} K')
