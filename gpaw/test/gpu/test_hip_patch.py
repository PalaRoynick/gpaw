def test_irfftn_patch():
    import cupyx
    import cupy as xp
    from gpaw.fftw import irfft_patch

    A = xp.random.rand((10,11,12))
    A = xp.random.rand((2,3,4))
    rfft_A = cupyx.scipy.fft.rfftn(A)
    Aback = cupy.scipy.fft.irfftn(rfft_A, A.shape)
    Aback2 = irfft_patch(rfft_A, A.shape)
    print(Aback)
    print(Aback2)
    
