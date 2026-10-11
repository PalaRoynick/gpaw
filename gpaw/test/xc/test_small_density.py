import pytest
import numpy as np
from gpaw.xc.libxc import LibXC
from gpaw.xc.kernel import XCKernel
from time import time


@pytest.mark.libxc
def test_small():
    name = 'LDA'
    N = 10000
    for xc in [LibXC(name),
               XCKernel(name)]:
        n0 = 1e-10
        n_sr = np.array([[-2, -0.5, 0.0, 0.45, 0.55, 1.5]]) * n0
        n_sr = np.linspace(-17, -1, N)[np.newaxis]
        # s_xr = np.zeros_like(n_sr)
        e_r = np.empty_like(n_sr[0])
        v_sr = np.zeros_like(n_sr)
        # w_xr = np.zeros_like(n_sr)
        # xc.calculate(e_r, 10**n_sr, v_sr, s_xr, w_xr)
        t1 = time()
        xc.calculate(e_r, 10**n_sr, v_sr)
        # xc.calculate(e_r, n_sr, v_sr)
        t = time() - t1
        print(xc, t / N * 1e9)
        import matplotlib.pyplot as plt
        # plt.plot(n_sr[0], v_sr[0])
        plt.plot(10**n_sr[0], e_r)
        #print(e_r)
        #print(v_sr[0])
    # plt.ylim(-1e-12, 0)
    plt.show()


if __name__ == '__main__':
    test_small()
