#include "pw.h"
#include "python_utils.h"

#include "gpu/cpp/pyarray_utils.hpp"
#include "gpu/cpp/pybind_cupy_type_caster.hpp"

#include "gpaw_complex.h"

#include <algorithm>
#include <cmath>

namespace py = pybind11;

template <typename T> void _pw_insert(int nG,
                int nQ,
                const std::complex<T>* c_G,
                const npy_int32* Q_G,
                T scale,
                std::complex<T>* tmp_Q)
// Does the same as these two lines of Python:
//
//     tmp_Q[:] = 0.0
//     tmp_Q.ravel()[Q_G] = c_G * scale
{
    int Q1 = 0;
    for (int G = 0; G < nG; G++) {
        int Q2 = Q_G[G];
        for (; Q1 < Q2; Q1++)
            tmp_Q[Q1] = 0.0;
        tmp_Q[Q1++] = c_G[G] * scale;
        }
    for (; Q1 < nQ; Q1++)
        tmp_Q[Q1] = 0.0;
}

template <typename T> void _pw_precond(int nG,
                const T* G2_G,
                const std::complex<T>* R_G,
                T ekin,
                std::complex<T>* out_G)
{
    for (int G = 0; G < nG; G++) {
        T x = G2_G[G] / (ekin * T(3.0));
        T a = T(27.0) + x * (T(18.0) + x * (T(12.0) + x * T(8.0)));
        T xx = x * x;
        out_G[G] = (-T(4.0) / (T(3.0) * ekin) *
                    a / (a + T(16.0) * xx * xx) * R_G[G]);
    }
}

template <typename T> void _pwlfc_expand_complex(int nG,
                int nJ,
                int nL,
                int nsplines,
                const T* f_Gs,
                const T* GK_Gv,
                const T* pos_av,
                const std::complex<T>* eikR_a,
                const T* Y_GL,
                const npy_int32* l_s,
                const npy_int32* a_J,
                const npy_int32* s_J,
                bool cc,
                std::complex<T>* f_GI)
{
    const std::complex<T> imag_powers[4] = {
        std::complex<T>(1.0, 0.0),
        std::complex<T>(0.0, -1.0),
        std::complex<T>(-1.0, 0.0),
        std::complex<T>(0.0, 1.0)
    };

    std::complex<T> emiGR(0.0, 0.0);

    for (int G = 0; G < nG; G++) {
        int a_old = -1;
        for (int J = 0; J < nJ; J++) {
            int s = s_J[J];
            int l = l_s[s];

            if (a_J[J] != a_old) {
                a_old = a_J[J];
                T f0 = (GK_Gv[0] * pos_av[0 + 3 * a_old] +
                        GK_Gv[1] * pos_av[1 + 3 * a_old] +
                        GK_Gv[2] * pos_av[2 + 3 * a_old]);
                emiGR = std::complex<T>(std::cos(f0), -std::sin(f0)) * eikR_a[a_old];
            }

            std::complex<T> f1 = emiGR * f_Gs[s] * imag_powers[l % 4];
            for (int m = 0; m < 2 * l + 1; m++) {
                std::complex<T> f = f1 * Y_GL[l * l + m];
                *f_GI++ = cc ? std::conj(f) : f;
            }
        }
        f_Gs += nsplines;
        GK_Gv += 3;
        Y_GL += nL;
    }
}

template <typename T> void _pwlfc_expand_real(int nG,
                int nJ,
                int nL,
                int nsplines,
                const T* f_Gs,
                const T* GK_Gv,
                const T* pos_av,
                const std::complex<T>* eikR_a,
                const T* Y_GL,
                const npy_int32* l_s,
                const npy_int32* a_J,
                const npy_int32* s_J,
                bool cc,
                T* f_GI,
                int nI)
{
    const std::complex<T> imag_powers[4] = {
        std::complex<T>(1.0, 0.0),
        std::complex<T>(0.0, -1.0),
        std::complex<T>(-1.0, 0.0),
        std::complex<T>(0.0, 1.0)
    };

    std::complex<T> emiGR(0.0, 0.0);

    for (int G = 0; G < nG; G++) {
        int a_old = -1;
        for (int J = 0; J < nJ; J++) {
            int s = s_J[J];
            int l = l_s[s];

            if (a_J[J] != a_old) {
                a_old = a_J[J];
                T f0 = (GK_Gv[0] * pos_av[0 + 3 * a_old] +
                        GK_Gv[1] * pos_av[1 + 3 * a_old] +
                        GK_Gv[2] * pos_av[2 + 3 * a_old]);
                emiGR = std::complex<T>(std::cos(f0), -std::sin(f0)) * eikR_a[a_old];
            }

            std::complex<T> f1 = emiGR * f_Gs[s] * imag_powers[l % 4];
            for (int m = 0; m < 2 * l + 1; m++) {
                std::complex<T> f = f1 * Y_GL[l * l + m];
                f_GI[0] = std::real(f);
                f_GI[nI] = cc ? -std::imag(f) : std::imag(f);
                f_GI++;
            }
        }
        f_Gs += nsplines;
        GK_Gv += 3;
        Y_GL += nL;
        f_GI += nI;
    }
}


void cpp_pw_insert(py::array c_G,
                   py::array Q_G,
                   double scale,
                   py::array tmp_Q)
{
    assert(c_G.ndim() == 1);
    assert(Q_G.ndim() == 1);
    assert(tmp_Q.ndim() == 3);
    assert(gpaw::is_c_contiguous(c_G));
    assert(gpaw::is_c_contiguous(Q_G));
    assert(gpaw::is_c_contiguous(tmp_Q));
    int nG = c_G.shape()[0];
    int nQ = tmp_Q.size();

    bool is_double = c_G.dtype().is(py::dtype::of<std::complex<double>>());
    bool is_float = c_G.dtype().is(py::dtype::of<std::complex<float>>());
    
    if (is_double)
    {
        _pw_insert<double>(nG, nQ, 
                   static_cast<const std::complex<double>*>(c_G.data()),
                   static_cast<const npy_int32*>(Q_G.data()), 
                   scale, 
                   static_cast<std::complex<double>*>(tmp_Q.mutable_data()));
    } 
    else if (is_float)
    {
        _pw_insert<float>(nG, nQ, 
                   static_cast<const std::complex<float>*>(c_G.data()),
                   static_cast<const npy_int32*>(Q_G.data()), 
                   static_cast<float>(scale), 
                   static_cast<std::complex<float>*>(tmp_Q.mutable_data()));
    }
    else
    {
        throw std::invalid_argument("Supports only double and float arrays.");
    }
//    nG = 
//    _pw_insert(nG, nQ, c_G, Q_G, scale, tmp_Q)
}

void cpp_pwlfc_expand(py::array f_Gs,
                      py::array GK_Gv,
                      py::array pos_av,
                      py::array eikR_a,
                      py::array Y_GL,
                      py::array l_s,
                      py::array a_J,
                      py::array s_J,
                      bool cc,
                      py::array f_GI)
{
    assert(f_Gs.ndim() == 2);
    assert(GK_Gv.ndim() == 2);
    assert(pos_av.ndim() == 2);
    assert(eikR_a.ndim() == 1);
    assert(Y_GL.ndim() == 2);
    assert(l_s.ndim() == 1);
    assert(a_J.ndim() == 1);
    assert(s_J.ndim() == 1);
    assert(f_GI.ndim() == 2);
    assert(GK_Gv.shape()[1] == 3);
    assert(pos_av.shape()[1] == 3);
    assert(gpaw::is_c_contiguous(f_Gs));
    assert(gpaw::is_c_contiguous(GK_Gv));
    assert(gpaw::is_c_contiguous(pos_av));
    assert(gpaw::is_c_contiguous(eikR_a));
    assert(gpaw::is_c_contiguous(Y_GL));
    assert(gpaw::is_c_contiguous(l_s));
    assert(gpaw::is_c_contiguous(a_J));
    assert(gpaw::is_c_contiguous(s_J));
    assert(gpaw::is_c_contiguous(f_GI));

    int nG = GK_Gv.shape()[0];
    int nJ = a_J.shape()[0];
    int nL = Y_GL.shape()[1];
    int nsplines = f_Gs.shape()[1];

    assert(f_Gs.shape()[0] == nG);
    assert(Y_GL.shape()[0] == nG);
    assert(s_J.shape()[0] == nJ);

    bool is_double_complex = (
        f_Gs.dtype().is(py::dtype::of<double>()) &&
        GK_Gv.dtype().is(py::dtype::of<double>()) &&
        pos_av.dtype().is(py::dtype::of<double>()) &&
        eikR_a.dtype().is(py::dtype::of<std::complex<double>>()) &&
        Y_GL.dtype().is(py::dtype::of<double>()) &&
        l_s.dtype().is(py::dtype::of<npy_int32>()) &&
        a_J.dtype().is(py::dtype::of<npy_int32>()) &&
        s_J.dtype().is(py::dtype::of<npy_int32>()) &&
        f_GI.dtype().is(py::dtype::of<std::complex<double>>()));

    bool is_float_complex = (
        f_Gs.dtype().is(py::dtype::of<float>()) &&
        GK_Gv.dtype().is(py::dtype::of<float>()) &&
        pos_av.dtype().is(py::dtype::of<float>()) &&
        eikR_a.dtype().is(py::dtype::of<std::complex<float>>()) &&
        Y_GL.dtype().is(py::dtype::of<float>()) &&
        l_s.dtype().is(py::dtype::of<npy_int32>()) &&
        a_J.dtype().is(py::dtype::of<npy_int32>()) &&
        s_J.dtype().is(py::dtype::of<npy_int32>()) &&
        f_GI.dtype().is(py::dtype::of<std::complex<float>>()));

    bool is_double_real = (
        f_Gs.dtype().is(py::dtype::of<double>()) &&
        GK_Gv.dtype().is(py::dtype::of<double>()) &&
        pos_av.dtype().is(py::dtype::of<double>()) &&
        eikR_a.dtype().is(py::dtype::of<std::complex<double>>()) &&
        Y_GL.dtype().is(py::dtype::of<double>()) &&
        l_s.dtype().is(py::dtype::of<npy_int32>()) &&
        a_J.dtype().is(py::dtype::of<npy_int32>()) &&
        s_J.dtype().is(py::dtype::of<npy_int32>()) &&
        f_GI.dtype().is(py::dtype::of<double>()));

    bool is_float_real = (
        f_Gs.dtype().is(py::dtype::of<float>()) &&
        GK_Gv.dtype().is(py::dtype::of<float>()) &&
        pos_av.dtype().is(py::dtype::of<float>()) &&
        eikR_a.dtype().is(py::dtype::of<std::complex<float>>()) &&
        Y_GL.dtype().is(py::dtype::of<float>()) &&
        l_s.dtype().is(py::dtype::of<npy_int32>()) &&
        a_J.dtype().is(py::dtype::of<npy_int32>()) &&
        s_J.dtype().is(py::dtype::of<npy_int32>()) &&
        f_GI.dtype().is(py::dtype::of<float>()));

    if (is_double_complex)
    {
        assert(f_GI.shape()[0] == nG);
        _pwlfc_expand_complex<double>(
            nG, nJ, nL, nsplines,
            static_cast<const double*>(f_Gs.data()),
            static_cast<const double*>(GK_Gv.data()),
            static_cast<const double*>(pos_av.data()),
            static_cast<const std::complex<double>*>(eikR_a.data()),
            static_cast<const double*>(Y_GL.data()),
            static_cast<const npy_int32*>(l_s.data()),
            static_cast<const npy_int32*>(a_J.data()),
            static_cast<const npy_int32*>(s_J.data()),
            cc,
            static_cast<std::complex<double>*>(f_GI.mutable_data()));
    }
    else if (is_float_complex)
    {
        assert(f_GI.shape()[0] == nG);
        _pwlfc_expand_complex<float>(
            nG, nJ, nL, nsplines,
            static_cast<const float*>(f_Gs.data()),
            static_cast<const float*>(GK_Gv.data()),
            static_cast<const float*>(pos_av.data()),
            static_cast<const std::complex<float>*>(eikR_a.data()),
            static_cast<const float*>(Y_GL.data()),
            static_cast<const npy_int32*>(l_s.data()),
            static_cast<const npy_int32*>(a_J.data()),
            static_cast<const npy_int32*>(s_J.data()),
            cc,
            static_cast<std::complex<float>*>(f_GI.mutable_data()));
    }
    else if (is_double_real)
    {
        assert(f_GI.shape()[0] == 2 * nG);
        _pwlfc_expand_real<double>(
            nG, nJ, nL, nsplines,
            static_cast<const double*>(f_Gs.data()),
            static_cast<const double*>(GK_Gv.data()),
            static_cast<const double*>(pos_av.data()),
            static_cast<const std::complex<double>*>(eikR_a.data()),
            static_cast<const double*>(Y_GL.data()),
            static_cast<const npy_int32*>(l_s.data()),
            static_cast<const npy_int32*>(a_J.data()),
            static_cast<const npy_int32*>(s_J.data()),
            cc,
            static_cast<double*>(f_GI.mutable_data()),
            f_GI.shape()[1]);
    }
    else if (is_float_real)
    {
        assert(f_GI.shape()[0] == 2 * nG);
        _pwlfc_expand_real<float>(
            nG, nJ, nL, nsplines,
            static_cast<const float*>(f_Gs.data()),
            static_cast<const float*>(GK_Gv.data()),
            static_cast<const float*>(pos_av.data()),
            static_cast<const std::complex<float>*>(eikR_a.data()),
            static_cast<const float*>(Y_GL.data()),
            static_cast<const npy_int32*>(l_s.data()),
            static_cast<const npy_int32*>(a_J.data()),
            static_cast<const npy_int32*>(s_J.data()),
            cc,
            static_cast<float*>(f_GI.mutable_data()),
            f_GI.shape()[1]);
    }
    else
    {
        throw std::invalid_argument(
            "Unsupported dtype combination for pwlfc_expand.");
    }
}

void cpp_pw_precond(py::array G2_G,
                    py::array R_G,
                    double ekin,
                    py::array out_G)
{
    assert(G2_G.ndim() == 1);
    assert(R_G.ndim() == 1);
    assert(out_G.ndim() == 1);
    assert(gpaw::is_c_contiguous(G2_G));
    assert(gpaw::is_c_contiguous(R_G));
    assert(gpaw::is_c_contiguous(out_G));

    int nG = G2_G.shape()[0];
    assert(R_G.shape()[0] == nG);
    assert(out_G.shape()[0] == nG);

    bool is_double = (G2_G.dtype().is(py::dtype::of<double>()) &&
                      R_G.dtype().is(py::dtype::of<std::complex<double>>()) &&
                      out_G.dtype().is(py::dtype::of<std::complex<double>>()));
    bool is_float = (G2_G.dtype().is(py::dtype::of<float>()) &&
                     R_G.dtype().is(py::dtype::of<std::complex<float>>()) &&
                     out_G.dtype().is(py::dtype::of<std::complex<float>>()));

    if (is_double)
    {
        _pw_precond<double>(nG,
            static_cast<const double*>(G2_G.data()),
            static_cast<const std::complex<double>*>(R_G.data()),
            ekin,
            static_cast<std::complex<double>*>(out_G.mutable_data()));
    }
    else if (is_float)
    {
        _pw_precond<float>(nG,
            static_cast<const float*>(G2_G.data()),
            static_cast<const std::complex<float>*>(R_G.data()),
            static_cast<float>(ekin),
            static_cast<std::complex<float>*>(out_G.mutable_data()));
    }
    else
    {
        throw std::invalid_argument(
            "Supports only matching float/complex64 or double/complex128 arrays.");
    }
}

py::array cpp_pawexxvv(py::array M_pp,
                       py::array D_ii)
{
    assert(M_pp.ndim() == 2);
    assert(D_ii.ndim() == 2);
    assert(gpaw::is_c_contiguous(M_pp));
    assert(gpaw::is_c_contiguous(D_ii));

    int NI = D_ii.shape()[0];
    int NP = M_pp.shape()[0];
    assert(D_ii.shape()[1] == NI);
    assert(M_pp.shape()[1] == NP);

    bool is_double = (M_pp.dtype().is(py::dtype::of<double>()) &&
                      D_ii.dtype().is(py::dtype::of<double>()));
    bool is_float = (M_pp.dtype().is(py::dtype::of<float>()) &&
                     D_ii.dtype().is(py::dtype::of<float>()));

    if (is_double)
    {
        py::array_t<double> V_ii({NI, NI});
        auto* M = static_cast<const double*>(M_pp.data());
        auto* D = static_cast<const double*>(D_ii.data());
        auto* V = static_cast<double*>(V_ii.mutable_data());

        for (int i1 = 0; i1 < NI; i1++)
        {
            for (int i2 = 0; i2 < NI; i2++)
            {
                double value = 0.0;
                for (int i3 = 0; i3 < NI; i3++)
                {
                    int p13 = (std::min(i3, i1) *
                               (2 * NI - 1 - std::min(i3, i1)) / 2) +
                              std::max(i3, i1);
                    for (int i4 = 0; i4 < NI; i4++)
                    {
                        int p24 = (std::min(i2, i4) *
                                   (2 * NI - 1 - std::min(i2, i4)) / 2) +
                                  std::max(i2, i4);
                        value += M[p13 * NP + p24] * D[i3 * NI + i4];
                    }
                }
                V[i1 * NI + i2] = value;
            }
        }

        return V_ii;
    }
    else if (is_float)
    {
        py::array_t<float> V_ii({NI, NI});
        auto* M = static_cast<const float*>(M_pp.data());
        auto* D = static_cast<const float*>(D_ii.data());
        auto* V = static_cast<float*>(V_ii.mutable_data());

        for (int i1 = 0; i1 < NI; i1++)
        {
            for (int i2 = 0; i2 < NI; i2++)
            {
                float value = 0.0f;
                for (int i3 = 0; i3 < NI; i3++)
                {
                    int p13 = (std::min(i3, i1) *
                               (2 * NI - 1 - std::min(i3, i1)) / 2) +
                              std::max(i3, i1);
                    for (int i4 = 0; i4 < NI; i4++)
                    {
                        int p24 = (std::min(i2, i4) *
                                   (2 * NI - 1 - std::min(i2, i4)) / 2) +
                                  std::max(i2, i4);
                        value += M[p13 * NP + p24] * D[i3 * NI + i4];
                    }
                }
                V[i1 * NI + i2] = value;
            }
        }

        return V_ii;
    }

    throw std::invalid_argument("Supports only matching float32 or float64 arrays.");
}

template <typename T> void _plane_wave_grid(const long* beg,
                const long* end,
                const T* h,
                const T* vk,
                const T* vr0,
                std::complex<T>* pw)
{
    T kr[3];
    T kr0[3];
    int n[3];
    int ij;
    for (int c = 0; c < 3; c++) {
        n[c] = end[c] - beg[c];
        kr0[c] = vk[c] * vr0[c];
    }
    for (int i = 0; i < n[0]; i++) {
        kr[0] = vk[0] * h[0] * (beg[0] + i) - kr0[0];
        for (int j = 0; j < n[1]; j++) {
            kr[1] = kr[0] + vk[1] * h[1] * (beg[1] + j) - kr0[1];
            ij = (i * n[1] + j) * n[2];
            for (int k = 0; k < n[2]; k++) {
                kr[2] = kr[1] + vk[2] * h[2] * (beg[2] + k) - kr0[2];
                pw[ij + k] = std::complex<T>(std::cos(kr[2]), std::sin(kr[2]));
            }
        }
    }
}

void cpp_plane_wave_grid(py::array beg_c,
                         py::array end_c,
                         py::array h_c,
                         py::array k_c,
                         py::array r0_c,
                         py::array pw_g)
{
    assert(beg_c.ndim() == 1);
    assert(end_c.ndim() == 1);
    assert(h_c.ndim() == 1);
    assert(k_c.ndim() == 1);
    assert(r0_c.ndim() == 1);
    assert(pw_g.ndim() == 3);
    assert(beg_c.shape()[0] == 3);
    assert(end_c.shape()[0] == 3);
    assert(h_c.shape()[0] == 3);
    assert(k_c.shape()[0] == 3);
    assert(r0_c.shape()[0] == 3);
    assert(gpaw::is_c_contiguous(beg_c));
    assert(gpaw::is_c_contiguous(end_c));
    assert(gpaw::is_c_contiguous(h_c));
    assert(gpaw::is_c_contiguous(k_c));
    assert(gpaw::is_c_contiguous(r0_c));
    assert(gpaw::is_c_contiguous(pw_g));

    bool is_double = (beg_c.dtype().is(py::dtype::of<long>()) &&
                      end_c.dtype().is(py::dtype::of<long>()) &&
                      h_c.dtype().is(py::dtype::of<double>()) &&
                      k_c.dtype().is(py::dtype::of<double>()) &&
                      r0_c.dtype().is(py::dtype::of<double>()) &&
                      pw_g.dtype().is(py::dtype::of<std::complex<double>>()));
    bool is_float = (beg_c.dtype().is(py::dtype::of<long>()) &&
                     end_c.dtype().is(py::dtype::of<long>()) &&
                     h_c.dtype().is(py::dtype::of<float>()) &&
                     k_c.dtype().is(py::dtype::of<float>()) &&
                     r0_c.dtype().is(py::dtype::of<float>()) &&
                     pw_g.dtype().is(py::dtype::of<std::complex<float>>()));

    if (is_double)
    {
        _plane_wave_grid<double>(static_cast<const long*>(beg_c.data()),
                                 static_cast<const long*>(end_c.data()),
                                 static_cast<const double*>(h_c.data()),
                                 static_cast<const double*>(k_c.data()),
                                 static_cast<const double*>(r0_c.data()),
                                 static_cast<std::complex<double>*>(pw_g.mutable_data()));
    }
    else if (is_float)
    {
        _plane_wave_grid<float>(static_cast<const long*>(beg_c.data()),
                                static_cast<const long*>(end_c.data()),
                                static_cast<const float*>(h_c.data()),
                                static_cast<const float*>(k_c.data()),
                                static_cast<const float*>(r0_c.data()),
                                static_cast<std::complex<float>*>(pw_g.mutable_data()));
    }
    else
    {
        throw std::invalid_argument(
            "Supports long index arrays and matching float/complex64 or double/complex128 arrays.");
    }
}

bool bind_pw_submodule(PyObject* module)
{
    if (!module || module == Py_None)
    {
        return false;
    }

    py::module_ m = py::reinterpret_borrow<py::module_>(module);
    if (m.is_none())
    {
        return false;
    }

    py::module_ submodule = m.def_submodule("pw", "PW specific C++ bindings for GPAW. (EXPERIMENTAL)");

    // Bind Cupy conversion testers (call from Python-side test)

    submodule.def("pw_insert",
        &cpp_pw_insert,
        py::arg("c_G"), py::arg("Q_G"), py::arg("scale"), py::arg("tmp_Q"),
        R"(Docstring for pw_insert.)"
    );
    submodule.def("pw_precond",
        &cpp_pw_precond,
        py::arg("G2_G"), py::arg("R_G"), py::arg("ekin"), py::arg("out_G"),
        R"(Docstring for pw_precond.)"
    );
    submodule.def("pwlfc_expand",
        &cpp_pwlfc_expand,
        py::arg("f_Gs"), py::arg("GK_Gv"), py::arg("pos_av"),
        py::arg("eikR_a"), py::arg("Y_GL"), py::arg("l_s"),
        py::arg("a_J"), py::arg("s_J"), py::arg("cc"), py::arg("f_GI"),
        R"(Docstring for pwlfc_expand.)"
    );
    submodule.def("pawexxvv",
        &cpp_pawexxvv,
        py::arg("M_pp"), py::arg("D_ii"),
        R"(Docstring for pawexxvv.)"
    );
    submodule.def("plane_wave_grid",
        &cpp_plane_wave_grid,
        py::arg("beg_c"), py::arg("end_c"), py::arg("h_c"),
        py::arg("k_c"), py::arg("r0_c"), py::arg("pw_g"),
        R"(Docstring for plane_wave_grid.)"
    );

    return true;
}
