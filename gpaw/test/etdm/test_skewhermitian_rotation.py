import numpy as np
import pytest
from gpaw.new.etdm.skewherm_matrix import SkewHermitian
from scipy.linalg import expm


@pytest.mark.parametrize(
    "ndim,dtype,method",
    [
        (2, float, 'egdecomp'),
        (2, complex, 'egdecomp'),
        (3, complex, 'egdecomp'),
        (2, float, 'pade-approx'),
        (2, complex, 'pade-approx'),
        (3, complex, 'pade-approx'),
    ],
)
def test_skewhermitian_rotation(ndim, dtype, method, gpaw_new):
    """
    Test that SkewHermitian rotation_mat correctly computes
    the matrix exponential of a manually reconstructed skew-Hermitian matrix.
    """
    if not gpaw_new:
        pytest.skip('Does not work for old GPAW')

    # Step 1: Create a random parameter vector
    if dtype == float:
        vec_len = (ndim * (ndim - 1)) // 2
    else:
        vec_len = (ndim * (ndim + 1)) // 2

    np.random.seed(42)
    param_vec = np.random.random(vec_len).astype(dtype)
    if dtype == complex:
        param_vec += 1j * np.random.random(vec_len)

    # Step 2: Manually reconstruct full skew-Hermitian matrix
    ind_up = (
        np.triu_indices(ndim, 1)
        if dtype == float
        else np.triu_indices(ndim)
    )
    a_mat_manual = np.zeros((ndim, ndim), dtype=dtype)
    a_mat_manual[ind_up] = param_vec
    a_mat_manual -= a_mat_manual.T.conj()
    # match vec2skewmat
    np.fill_diagonal(a_mat_manual, a_mat_manual.diagonal() * 0.5)

    # Step 3: Compute expected matrix exponential manually
    U_manual = expm(a_mat_manual)

    # Step 4: Use SkewHermitian class
    sk = SkewHermitian(ndim, dtype, data=param_vec, method=method)
    U_class = sk.rotation_mat

    # Step 5: Compare results
    np.testing.assert_allclose(U_class, U_manual, rtol=1e-12, atol=1e-12)

    # Check eigenvalues/eigenvectors consistency
    e_manual, v_manual = np.linalg.eig(1j * a_mat_manual)
    np.testing.assert_allclose(
        np.sort(e_manual),
        np.sort(sk.evals),
        rtol=1e-12,
        atol=1e-12,
    )


def test_pade_gradient_matches_egdecomp(gpaw_new):
    """Test that gradient is identical regardless of forward method.

    The gradient always uses eigendecomposition internally, so both
    methods should give exactly the same result for calc_gradient.
    """
    if not gpaw_new:
        pytest.skip('Does not work for old GPAW')

    ndim = 3
    dtype = complex

    np.random.seed(123)
    vec_len = (ndim * (ndim + 1)) // 2
    param_vec = 0.1 * (np.random.random(vec_len)
                       + 1j * np.random.random(vec_len))

    # Random skew-Hermitian input for gradient (as in objfunc_etdm.py)
    h_raw = np.random.random((ndim, ndim)) + 1j * np.random.random(
        (ndim, ndim))
    h_skew = h_raw - h_raw.T.conj()

    sk_eg = SkewHermitian(ndim, dtype, data=param_vec.copy(),
                          method='egdecomp')
    sk_pa = SkewHermitian(ndim, dtype, data=param_vec.copy(),
                          method='pade-approx')

    # Rotation matrices should match
    np.testing.assert_allclose(
        sk_eg.rotation_mat, sk_pa.rotation_mat,
        rtol=1e-12, atol=1e-12)

    # Gradients should be identical (both use eigendecomp internally)
    grad_eg = sk_eg.calc_gradient(h_skew)
    grad_pa = sk_pa.calc_gradient(h_skew)

    np.testing.assert_allclose(grad_eg, grad_pa, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__])
