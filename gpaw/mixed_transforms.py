from scipy.fftpack import dst as scipydst
from numpy.fft import fft2, ifft2, rfft2, irfft2, fft, ifft

"""
# In case real transforms are not available

def rfst2(A_g, axes=[0,1]):
    assert axes[0] == 0
    assert axes[1] == 1
    x,y,z = A_g.shape
    temp_g = np.zeros((x*2+2, y*2+2, z))
    temp_g[1:x+1, 1:y+1,:] = A_g
    temp_g[x+2:, 1:y+1,:] = -A_g[::-1, :, :]
    temp_g[1:x+1, y+2:,:] = -A_g[:, ::-1, :]
    temp_g[x+2:, y+2:,:] = A_g[::-1, ::-1, :]
    X = -4*rfft2(temp_g, axes=axes)
    return X[1:x+1, 1:y+1, :].real

def irfst2(A_g, axes=[0,1]):
    assert axes[0] == 0
    assert axes[1] == 1
    x,y,z = A_g.shape
    temp_g = np.zeros((x*2+2, (y*2+2)//2+1, z))
    temp_g[1:x+1, 1:y+1,:] = A_g
    temp_g[x+2:, 1:y+1,:] = -A_g[::-1, :, :]
    return -0.25*irfft2(temp_g, axes=axes)[1:x+1, 1:y+1, :].real
"""


class MixedTransforms:
    """Mixed fast sin and fast fourier transforms helpers

    These functions enable to do efficient parallel Fourier transforms
    with arbitrary boundary conditions (fst vs. fft per each axis).
    """
    def __init__(self, xp):
        # cupyx.scipy.fftpack does not have dst
        import numpy as np
        self.use_scipy_transforms = xp is np
        self.xp = xp

    def rfst2(self, A_g, axes=[0, 1]):
        all = {0, 1, 2}
        third = [all.difference(set(axes)).pop()]

        if self.use_scipy_transforms:
            Y = A_g
            for axis in axes:
                Y = scipydst(Y, axis=axis, type=1)
            Y *= 2**len(axes)
            return Y

        xp = self.xp
        A_g = xp.transpose(A_g, axes + third)
        x, y, z = A_g.shape
        temp_g = xp.zeros((x * 2 + 2, y * 2 + 2, z))
        temp_g[1:x + 1, 1:y + 1, :] = A_g
        temp_g[x + 2:, 1:y + 1, :] = -A_g[::-1, :, :]
        temp_g[1:x + 1, y + 2:, :] = -A_g[:, ::-1, :]
        temp_g[x + 2:, y + 2:, :] = A_g[::-1, ::-1, :]
        X = -4 * rfft2(temp_g, axes=[0, 1])[1:x + 1, 1:y + 1, :].real
        return xp.transpose(X, xp.argsort(axes + third))

    def irfst2(self, A_g, axes=[0, 1]):
        xp = self.xp
        if self.use_scipy_transforms:
            Y = A_g
            for axis in axes:
                Y = scipydst(Y, axis=axis, type=1)
            magic = 1.0 / (16 * xp.prod([A_g.shape[axis] + 1
                           for axis in axes]))
            Y *= magic
            # Y /= 211200
            return Y

        all = {0, 1, 2}
        third = [all.difference(set(axes)).pop()]
        A_g = xp.transpose(A_g, axes + third)
        x, y, z = A_g.shape
        temp_g = xp.zeros((x * 2 + 2, (y * 2 + 2) // 2 + 1, z))
        temp_g[1:x + 1, 1:y + 1, :] = A_g.real
        temp_g[x + 2:, 1:y + 1, :] = -A_g[::-1, :, :].real
        X = -0.25 * irfft2(temp_g, axes=[0, 1])[1:x + 1, 1:y + 1, :]

        T = xp.transpose(X, xp.argsort(axes + third))
        return T

    # This method needs to be taken from fftw / scipy to gain speedup of ~4x
    def fst(self, A_g, axis):
        xp = self.xp
        x, y, z = A_g.shape
        N_c = xp.array([x, y, z])
        N_c[axis] = N_c[axis] * 2 + 2
        temp_g = xp.zeros(N_c, dtype=A_g.dtype)
        if axis == 0:
            temp_g[1:x + 1, :, :] = A_g
            temp_g[x + 2:, :, :] = -A_g[::-1, :, :]
        elif axis == 1:
            temp_g[:, 1:y + 1, :] = A_g
            temp_g[:, y + 2:, :] = -A_g[:, ::-1, :]
        elif axis == 2:
            temp_g[:, :, 1:z + 1] = A_g
            temp_g[:, :, z + 2:] = -A_g[:, ::, ::-1]
        else:
            raise NotImplementedError()
        X = 0.5j * fft(temp_g, axis=axis)
        if axis == 0:
            return X[1:x + 1, :, :]
        elif axis == 1:
            return X[:, 1:y + 1, :]
        elif axis == 2:
            return X[:, :, 1:z + 1]

    def ifst(self, A_g, axis):
        xp = self.xp
        x, y, z = A_g.shape
        N_c = xp.array([x, y, z])
        N_c[axis] = N_c[axis] * 2 + 2
        temp_g = xp.zeros(N_c, dtype=A_g.dtype)

        if axis == 0:
            temp_g[1:x + 1, :, :] = A_g
            temp_g[x + 2:, :, :] = -A_g[::-1, :, :]
        elif axis == 1:
            temp_g[:, 1:y + 1, :] = A_g
            temp_g[:, y + 2:, :] = -A_g[:, ::-1, :]
        elif axis == 2:
            temp_g[:, :, 1:z + 1] = A_g
            temp_g[:, :, z + 2:] = -A_g[:, ::, ::-1]
        else:
            raise NotImplementedError()

        X_g = ifft(temp_g, axis=axis)
        if axis == 0:
            return -2j * X_g[1:x + 1, :, :]
        elif axis == 1:
            return -2j * X_g[:, 1:y + 1, :]
        elif axis == 2:
            return -2j * X_g[:, :, 1:z + 1]

    def transform(self, A_g, axis=None, pbc=True):
        if pbc:
            if A_g.size == 0:
                return A_g.astype(complex)

            return fft(A_g, axis=axis)
        else:
            if A_g.size == 0:
                return A_g

            if not self.use_scipy_transforms:
                return self.fst(A_g, axis)

            y = scipydst(A_g, axis=axis, type=1)
            y *= .5
            return y

    def transform2(self, A_g, axes=None, pbc=[True, True]):
        if all(pbc):
            if A_g.size == 0:
                return A_g.astype(complex)

            return fft2(A_g, axes=axes)
        elif not any(pbc):
            if A_g.size == 0:
                return A_g

            return self.rfst2(A_g, axes=axes)
        else:
            return self.transform(self.transform(A_g,
                                                 axis=axes[0],
                                                 pbc=pbc[0]),
                                  axis=axes[1], pbc=pbc[1])

    def itransform(self, A_g, axis=None, pbc=True):
        if pbc:
            if A_g.size == 0:
                return A_g.astype(complex)

            return ifft(A_g, axis=axis)
        else:
            if A_g.size == 0:
                return A_g

            if not self.use_scipy_transforms:
                x = self.ifst(A_g, axis)
                return x
            y = scipydst(A_g, axis=axis, type=1)
            magic = 1.0 / (A_g.shape[axis] + 1)
            y *= magic
            return y

    def itransform2(self, A_g, axes=None, pbc=[True, True]):
        if all(pbc):
            if A_g.size == 0:
                return A_g.astype(complex)

            return ifft2(A_g, axes=axes)
        elif not any(pbc):
            if A_g.size == 0:
                return A_g

            return self.irfst2(A_g, axes=axes)
        else:
            return self.itransform(self.itransform(A_g,
                                                   axis=axes[0],
                                                   pbc=pbc[0]),
                                   axis=axes[1], pbc=pbc[1])
