import pytest

def test_irfftn_patch():
    from gpaw.gpu import is_hip
    if not is_hip:
        pytest.skip('Only with hip')
    import cupyx
    import cupy as xp
    from gpaw.fftw import irfftn_patch
    for sizes in [(1, 2, 3), (3, 5, 6), (6, 7, 9), (102, 14, 11)]:
        A = xp.random.rand(*sizes)
        rfft_A = cupyx.scipy.fft.rfftn(A)
        Aback = cupyx.scipy.fft.irfftn(rfft_A, A.shape)
        Aback2 = irfftn_patch(rfft_A, A.shape)
        assert xp.allclose(Aback, Aback2)
    
