import numpy as np
import pytest
from importlib import import_module

from gpaw import GPAW_NO_C_EXTENSION
import gpaw.cgpaw as cgpaw

if GPAW_NO_C_EXTENSION:
    pytest.skip('GPAW_NO_C_EXTENSION', allow_module_level=True)

_gpaw = import_module('_gpaw')

if not getattr(_gpaw, 'gpaw_cpp', False):
    pytest.skip('Requires GPAW C++ build with _gpaw.pw bindings',
                allow_module_level=True)

old_pawexxvv = _gpaw.pawexxvv
old_add_to_density = _gpaw.add_to_density
old_plane_wave_grid = _gpaw.plane_wave_grid
old_pw_insert = _gpaw.pw_insert
old_pw_precond = _gpaw.pw_precond
old_pwlfc_expand = _gpaw.pwlfc_expand
old_pwlfc_expand_old = _gpaw.pwlfc_expand_old
old_mmm = getattr(_gpaw, 'mmm', None)
old_r2k = getattr(_gpaw, 'r2k', None)

_pw = _gpaw.pw
new_pawexxvv = _pw.pawexxvv
new_add_to_density = _pw.add_to_density
new_plane_wave_grid = _pw.plane_wave_grid
new_pw_insert = _pw.pw_insert
new_pw_precond = _pw.pw_precond
new_pwlfc_expand = _pw.pwlfc_expand
new_pwlfc_expand_old = _pw.pwlfc_expand_old
new_mmm = getattr(_pw, 'mmm', None)
new_r2k = getattr(_pw, 'r2k', None)


seed = 42


def assert_close(actual, expected, single_precision):
    if single_precision:
        assert actual == pytest.approx(expected, rel=5e-5, abs=5e-6)
    else:
        assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)


def create_pwlfc_inputs():
    rng = np.random.RandomState(seed)
    nG = 40
    nA = 4
    nS = 5
    nL = nS**2

    f_Gs = rng.randn(nG, nS)
    Gk_Gv = rng.randn(nG, 3)
    pos_av = rng.randn(nA, 3)
    eikR_a = rng.randn(nA) + 1j * rng.randn(nA)
    Y_GL = rng.randn(nG, nL)
    l_s = np.arange(nS, dtype=np.int32)

    a_J = []
    s_J = []
    for a in range(nA):
        for s in range(nS):
            a_J.append(a)
            s_J.append(s)
    a_J = np.asarray(a_J, dtype=np.int32)
    s_J = np.asarray(s_J, dtype=np.int32)

    nI = 0
    for s in s_J:
        nI += 2 * l_s[s] + 1

    return f_Gs, Gk_Gv, pos_av, eikR_a, Y_GL, l_s, a_J, s_J, nI


def op(o, m):
    if o == 'N':
        return m
    if o == 'T':
        return m.T
    return m.T.conj()


@pytest.mark.parametrize('dtype', [np.complex128, np.complex64])
def test_pw_insert_old_vs_new(dtype):
    rng = np.random.RandomState(seed)
    shape = (4, 3, 5)
    size = np.prod(shape)
    nG = 11

    coef_ref = rng.randn(nG) + 1j * rng.randn(nG)
    Q_G = np.sort(rng.choice(size, size=nG, replace=False).astype(np.int32))
    scale = 0.37

    ref = np.empty(shape, dtype=np.complex128)
    old_pw_insert(coef_ref.copy(), Q_G, scale, ref)

    coef = coef_ref.astype(dtype)
    out = np.empty(shape, dtype=dtype)
    new_pw_insert(coef, Q_G, scale, out)

    if dtype == np.complex128:
        out_old = np.empty(shape, dtype=np.complex128)
        old_pw_insert(coef_ref.copy(), Q_G, scale, out_old)
        assert_close(out, out_old, single_precision=False)
    else:
        assert_close(out, ref.astype(np.complex64), single_precision=True)


def test_legacy_cgpaw_single_precision_guard():
    matrix = np.eye(2, dtype=np.float32)

    with pytest.raises(RuntimeError,
                       match="Legacy C-extension function 'pack'"):
        cgpaw.pack(matrix)


def test_pybind_override_allows_single_precision():
    coef_G = np.array([1.0 + 2.0j, -0.5 + 0.25j], dtype=np.complex64)
    Q_G = np.array([1, 3], dtype=np.int32)
    out_Q = np.empty((1, 1, 4), dtype=np.complex64)

    cgpaw.pw_insert(coef_G, Q_G, 0.5, out_Q)

    expected = np.zeros((1, 1, 4), dtype=np.complex64)
    expected.ravel()[Q_G] = 0.5 * coef_G
    assert_close(out_Q, expected, single_precision=True)


@pytest.mark.parametrize(
    'psit_dtype, make_view',
    [(np.float64, False),
     (np.float64, True),
     (np.float32, False),
     (np.float32, True),
     (np.complex128, False),
     (np.complex64, False)])
def test_add_to_density_old_vs_new(psit_dtype, make_view):
    rng = np.random.RandomState(seed)
    f = 0.37

    if np.issubdtype(psit_dtype, np.complexfloating):
        make_view = False

    if make_view:
        full = rng.randn(2, 3, 5).astype(psit_dtype)
        psit = full[:, :, :4]
    elif np.issubdtype(psit_dtype, np.floating):
        psit = rng.randn(2, 3, 4).astype(psit_dtype)
    else:
        psit = (
            rng.randn(2, 3, 4) + 1j * rng.randn(2, 3, 4)
        ).astype(psit_dtype)

    nt_ref = rng.randn(2, 3, 4)
    out = nt_ref.copy()
    new_add_to_density(f, psit, out)

    if psit_dtype in (np.float64, np.complex128):
        out_old = nt_ref.copy()
        old_add_to_density(f, psit, out_old)
        assert_close(out, out_old, single_precision=False)
    else:
        expected = nt_ref.copy()
        expected += (
            f *
            np.abs(psit.astype(
                np.complex64 if psit_dtype == np.complex64 else np.float32
            ))**2
        )
        assert_close(out, expected, single_precision=True)


@pytest.mark.parametrize(
    'dtype, alpha, beta',
    [(np.float64, 0.7, -0.2),
     (np.float32, 0.7, -0.2),
     (np.complex128, 0.7 + 0.1j, -0.2 + 0.3j),
     (np.complex64, 0.7 + 0.1j, -0.2 + 0.3j)])
@pytest.mark.parametrize('opa', ['N', 'T', 'C'])
@pytest.mark.parametrize('opb', ['N', 'T', 'C'])
def test_mmm_old_vs_new(dtype, alpha, beta, opa, opb):
    if old_mmm is None or new_mmm is None:
        pytest.skip('BLAS not available')

    rng = np.random.RandomState(seed)

    if np.issubdtype(dtype, np.floating):
        a_ref = rng.randn(3, 4)
    else:
        a_ref = rng.randn(3, 4) + 1j * rng.randn(3, 4)

    A = op(opa, a_ref)
    if np.issubdtype(dtype, np.floating):
        b_ref = rng.randn(A.shape[1], 5)
        c_ref = rng.randn(A.shape[0], 5)
    else:
        b_ref = rng.randn(A.shape[1], 5) + 1j * rng.randn(A.shape[1], 5)
        c_ref = rng.randn(A.shape[0], 5) + 1j * rng.randn(A.shape[0], 5)
    b_input_ref = op(opb, b_ref).copy()

    if np.issubdtype(dtype, np.complexfloating):
        ref = (
            alpha * op(opa, a_ref).dot(op(opb, b_input_ref)) +
            beta * c_ref
        )
    else:
        ref = (
            float(alpha) * op(opa, a_ref).dot(op(opb, b_input_ref)) +
            float(beta) * c_ref
        )

    a = a_ref.astype(dtype)
    b = b_input_ref.astype(dtype)
    c = c_ref.astype(dtype)
    new_mmm(alpha, a, opa, b, opb, beta, c)

    if dtype in (np.float64, np.complex128):
        c_old = c_ref.astype(dtype)
        old_mmm(alpha,
                a_ref.astype(dtype),
                opa,
                b_input_ref.astype(dtype),
                opb,
                beta,
                c_old)
        assert_close(c, c_old, single_precision=False)
    else:
        assert_close(c, ref.astype(dtype), single_precision=True)


@pytest.mark.parametrize(
    'dtype, alpha, beta',
    [(np.float64, 0.5, 0.2),
     (np.float32, 0.5, 0.2),
     (np.complex128, 0.5 + 0.25j, 0.2),
     (np.complex64, 0.5 + 0.25j, 0.2)])
@pytest.mark.parametrize('trans', ['c', 'n'])
def test_r2k_old_vs_new(dtype, alpha, beta, trans):
    if old_r2k is None or new_r2k is None:
        pytest.skip('BLAS not available')

    rng = np.random.RandomState(seed)

    if trans == 'c':
        ashape = (4, 3)
        cshape = (4, 4)
    else:
        ashape = (3, 4)
        cshape = (4, 4)

    if np.issubdtype(dtype, np.floating):
        a_ref = rng.randn(*ashape)
        b_ref = rng.randn(*ashape)
        c_ref = rng.randn(*cshape)
    else:
        a_ref = rng.randn(*ashape) + 1j * rng.randn(*ashape)
        b_ref = rng.randn(*ashape) + 1j * rng.randn(*ashape)
        c_ref = rng.randn(*cshape) + 1j * rng.randn(*cshape)
        np.fill_diagonal(c_ref, c_ref.diagonal().real)

    a = a_ref.astype(dtype)
    b = b_ref.astype(dtype)
    c = c_ref.astype(dtype)
    new_r2k(alpha, a, b, beta, c, trans)

    if dtype in (np.float64, np.complex128):
        c_old = c_ref.astype(dtype)
        old_r2k(alpha,
                a_ref.astype(dtype),
                b_ref.astype(dtype),
                beta,
                c_old,
                trans)
        assert_close(c, c_old, single_precision=False)
    else:
        ref_dtype = np.float64 if dtype == np.float32 else np.complex128
        a_old = a_ref.astype(ref_dtype)
        b_old = b_ref.astype(ref_dtype)
        c_old = c_ref.astype(ref_dtype)
        alpha_old = float(alpha) if dtype == np.float32 else complex(alpha)
        old_r2k(alpha_old, a_old, b_old, beta, c_old, trans)
        assert_close(np.tril(c),
                     np.tril(c_old.astype(dtype)),
                     single_precision=True)


@pytest.mark.parametrize('dtype', [np.complex128, np.complex64])
def test_pw_precond_old_vs_new(dtype):
    rng = np.random.RandomState(seed)
    real_dtype = np.float64 if dtype == np.complex128 else np.float32
    nG = 23
    ekin = 2.3

    G2_ref = np.abs(rng.randn(nG)) + 0.1
    R_ref = rng.randn(nG) + 1j * rng.randn(nG)
    out_ref = np.empty(nG, dtype=np.complex128)
    old_pw_precond(G2_ref.copy(), R_ref.copy(), ekin, out_ref)

    G2 = G2_ref.astype(real_dtype)
    R = R_ref.astype(dtype)
    out = np.empty(nG, dtype=dtype)
    new_pw_precond(G2, R, ekin, out)

    if dtype == np.complex128:
        out_old = np.empty(nG, dtype=np.complex128)
        old_pw_precond(G2_ref.copy(), R_ref.copy(), ekin, out_old)
        assert_close(out, out_old, single_precision=False)
    else:
        assert_close(out, out_ref.astype(np.complex64), single_precision=True)


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_pawexxvv_old_vs_new(dtype):
    rng = np.random.RandomState(seed)
    nI = 6
    nP = nI * (nI + 1) // 2

    M_ref = rng.randn(nP, nP)
    D_ref = rng.randn(nI, nI)
    ref = old_pawexxvv(M_ref.copy(), D_ref.copy())

    M = M_ref.astype(dtype)
    D = D_ref.astype(dtype)
    out = new_pawexxvv(M, D)

    if dtype == np.float64:
        out_old = old_pawexxvv(M_ref.copy(), D_ref.copy())
        assert_close(out, out_old, single_precision=False)
    else:
        assert_close(out, ref.astype(np.float32), single_precision=True)


@pytest.mark.parametrize('dtype', [np.complex128, np.complex64])
def test_plane_wave_grid_old_vs_new(dtype):
    real_dtype = np.float64 if dtype == np.complex128 else np.float32
    beg_c = np.array([1, 2, 3], dtype=np.int64)
    end_c = np.array([4, 6, 8], dtype=np.int64)
    h_c_ref = np.array([0.1, 0.2, 0.3])
    k_c_ref = np.array([0.7, -0.2, 0.4])
    r0_c_ref = np.array([0.9, -1.1, 0.5])
    shape = tuple((end_c - beg_c).tolist())

    ref = np.empty(shape, dtype=np.complex128)
    old_plane_wave_grid(beg_c, end_c, h_c_ref.copy(), k_c_ref.copy(),
                        r0_c_ref.copy(), ref)

    h_c = h_c_ref.astype(real_dtype)
    k_c = k_c_ref.astype(real_dtype)
    r0_c = r0_c_ref.astype(real_dtype)
    out = np.empty(shape, dtype=dtype)
    new_plane_wave_grid(beg_c, end_c, h_c, k_c, r0_c, out)

    if dtype == np.complex128:
        out_old = np.empty(shape, dtype=np.complex128)
        old_plane_wave_grid(beg_c, end_c, h_c_ref.copy(), k_c_ref.copy(),
                            r0_c_ref.copy(), out_old)
        assert_close(out, out_old, single_precision=False)
    else:
        assert_close(out, ref.astype(np.complex64), single_precision=True)


@pytest.mark.parametrize(
    'fgi_dtype, single_precision',
    [(np.complex128, False),
     (np.float64, False),
     (np.complex64, True),
     (np.float32, True)])
@pytest.mark.parametrize('cc', [False, True])
def test_pwlfc_expand_old_vs_new(fgi_dtype, single_precision, cc):
    (f_Gs_ref, Gk_Gv_ref, pos_av_ref, eikR_a_ref,
     Y_GL_ref, l_s, a_J, s_J, nI) = create_pwlfc_inputs()

    real_dtype = np.float32 if single_precision else np.float64
    complex_dtype = np.complex64 if single_precision else np.complex128
    real_output = np.issubdtype(fgi_dtype, np.floating)
    nG = f_Gs_ref.shape[0]
    gN = 2 * nG if real_output else nG

    ref = np.zeros((gN, nI),
                   dtype=np.float64 if real_output else np.complex128)
    old_pwlfc_expand(f_Gs_ref.copy(), Gk_Gv_ref.copy(), pos_av_ref.copy(),
                     eikR_a_ref.copy(), Y_GL_ref.copy(),
                     l_s.copy(), a_J.copy(), s_J.copy(),
                     cc, ref)

    f_Gs = f_Gs_ref.astype(real_dtype)
    Gk_Gv = Gk_Gv_ref.astype(real_dtype)
    pos_av = pos_av_ref.astype(real_dtype)
    eikR_a = eikR_a_ref.astype(complex_dtype)
    Y_GL = Y_GL_ref.astype(real_dtype)
    out = np.zeros((gN, nI), dtype=fgi_dtype)
    new_pwlfc_expand(f_Gs, Gk_Gv, pos_av, eikR_a, Y_GL,
                     l_s.copy(), a_J.copy(), s_J.copy(),
                     cc, out)

    if single_precision:
        expected = ref.astype(fgi_dtype)
    else:
        out_old = np.zeros_like(ref)
        old_pwlfc_expand(f_Gs_ref.copy(), Gk_Gv_ref.copy(), pos_av_ref.copy(),
                         eikR_a_ref.copy(), Y_GL_ref.copy(),
                         l_s.copy(), a_J.copy(), s_J.copy(),
                         cc, out_old)
        expected = out_old

    assert_close(out, expected, single_precision=single_precision)


@pytest.mark.parametrize(
    'fgi_dtype, single_precision',
    [(np.complex128, False),
     (np.float64, False),
     (np.complex64, True),
     (np.float32, True)])
@pytest.mark.parametrize('cc', [False, True])
def test_pwlfc_expand_old_old_vs_new(fgi_dtype, single_precision, cc):
    (f_Gs_ref, Gk_Gv_ref, pos_av_ref, eikR_a_ref,
     Y_GL_ref, l_s, a_J, s_J, nI) = create_pwlfc_inputs()

    real_dtype = np.float32 if single_precision else np.float64
    complex_dtype = np.complex64 if single_precision else np.complex128
    real_output = np.issubdtype(fgi_dtype, np.floating)
    nG = f_Gs_ref.shape[0]
    nA = pos_av_ref.shape[0]
    gN = 2 * nG if real_output else nG

    GkR_Ga = Gk_Gv_ref @ pos_av_ref.T
    emiGR_ref = np.exp(-1j * GkR_Ga) * eikR_a_ref.reshape((1, nA))

    ref = np.zeros((gN, nI),
                   dtype=np.float64 if real_output else np.complex128)
    old_pwlfc_expand_old(f_Gs_ref.copy(), emiGR_ref.copy(), Y_GL_ref.copy(),
                         l_s.copy(), a_J.copy(), s_J.copy(),
                         cc, ref)

    f_Gs = f_Gs_ref.astype(real_dtype)
    emiGR = emiGR_ref.astype(complex_dtype)
    Y_GL = Y_GL_ref.astype(real_dtype)
    out = np.zeros((gN, nI), dtype=fgi_dtype)
    new_pwlfc_expand_old(f_Gs, emiGR, Y_GL,
                         l_s.copy(), a_J.copy(), s_J.copy(),
                         cc, out)

    if single_precision:
        expected = ref.astype(fgi_dtype)
    else:
        out_old = np.zeros_like(ref)
        old_pwlfc_expand_old(f_Gs_ref.copy(),
                             emiGR_ref.copy(),
                             Y_GL_ref.copy(),
                             l_s.copy(),
                             a_J.copy(),
                             s_J.copy(),
                             cc, out_old)
        expected = out_old

    assert_close(out, expected, single_precision=single_precision)
