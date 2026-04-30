#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>


PyObject *calculate_moments(PyObject *self, PyObject *args)
{
  PyArrayObject* beg_v_obj;
  PyArrayObject* h_v_obj;
  PyArrayObject* R_v_obj;     // Center in Bohr : default (0,0,0)
  PyArrayObject* rhot_G_obj;
  PyArrayObject* D_a_obj;
  PyArrayObject* Q_ab_obj;
  PyArrayObject* O_abc_obj;
  PyArrayObject* H_abcd_obj;
  // PyArrayObject* Rc_v_obj;
  // PyArrayObject* pbc_v_obj;

  if (!PyArg_ParseTuple(args, "OOOOOOOO", &beg_v_obj, &h_v_obj, &R_v_obj,
                        &rhot_G_obj, &D_a_obj, &Q_ab_obj, &O_abc_obj, &H_abcd_obj))
  return NULL;

  const long *beg_v = PyArray_DATA(beg_v_obj);
  const double *h_v = PyArray_DATA(h_v_obj);
  const double *R_v = PyArray_DATA(R_v_obj);
  const double *rhot_G = PyArray_DATA(rhot_G_obj);
  double *D_a = PyArray_DATA(D_a_obj);
  double *Q_ab = PyArray_DATA(Q_ab_obj);
  double *O_abc = PyArray_DATA(O_abc_obj);
  double *H_abcd = PyArray_DATA(H_abcd_obj);
  // const double *Rc_v = PyArray_DATA(Rc_v_obj);
  // const double *pbc_v = PyArray_DATA(pbc_v_obj);

  npy_intp* n = PyArray_DIMS(rhot_G_obj);

  const double dV = h_v[0] * h_v[1] * h_v[2];
  double r[3];
  double rr;
  double rq;
  double ro;
  double rh;
  double rrh;
  double rt;
  
  int nn = n[0];

  // Loop over grid
  for (int i = 0; i < n[0]; i++){
    double x = (beg_v[0] + i) * h_v[0];
    for (int j = 0; j < n[1]; j++){
      double y = (beg_v[1] + j) * h_v[1];
      int ij = (i * n[1] + j) * n[2];
      for (int k = 0; k < n[2]; k++){
        double z = (beg_v[2] + k) * h_v[2];
        int G = ij + k;

        r[0] = x - R_v[0]; 
        r[1] = y - R_v[1]; 
        r[2] = z - R_v[2]; 

        rr = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        rq = rr / 3.0;
        ro = rr / 5.0;
        rh = rr / 7.0;
        rrh = ro * rh;
        // rt = rr / 9.0;

        double cd = rhot_G[G] * dV;
        double cq = cd * 3.0 / 2.0;
        double co = cd * 15.0 / 6.0;
        double cH = cd * 105.0 / 24.0;
        double cT = cd * 9.0;

        // Dipoles
        D_a[0] += cd * r[0];
        D_a[1] += cd * r[1];
        D_a[2] += cd * r[2];

        // Get the 6 unique quadrupole elements
        Q_ab[0] += cq * (r[0] * r[0] - rq);
        Q_ab[1] += cq * r[0] * r[1];
        Q_ab[2] += cq * r[0] * r[2];
        Q_ab[4] += cq * (r[1] * r[1] - rq);
        Q_ab[5] += cq * r[1] * r[2];
        Q_ab[8] += cq * (r[2] * r[2] - rq);

        // Get the 10 unique octupole components
        O_abc[0] += co * (r[0] * r[0] * r[0] - 3.0 * r[0] * ro);
        O_abc[1] += co * (r[0] * r[0] * r[1] - r[1] * ro);
        O_abc[2] += co * (r[0] * r[0] * r[2] - r[2] * ro);
        O_abc[4] += co * (r[0] * r[1] * r[1] - r[0] * ro);
        O_abc[5] += co * r[0] * r[1] * r[2];
        O_abc[8] += co * (r[0] * r[2] * r[2] - r[0] * ro);
        O_abc[13] += co * (r[1] * r[1] * r[1] - 3.0 * r[1] * ro);
        O_abc[14] += co * (r[1] * r[1] * r[2] - r[2] * ro);
        O_abc[17] += co * (r[1] * r[2] * r[2] - r[1] * ro);
        O_abc[26] += co * (r[2] * r[2] * r[2] - 3.0 * r[2] * ro);

        // Get the 15 unique hexadecapole components
        H_abcd[0] += cH * (r[0] * r[0] * r[0] * r[0] - 6.0 * rh * r[0] * r[0] + 3.0 * rrh);
        H_abcd[1] += cH * (r[0] * r[0] * r[0] * r[1] - 3.0 * rh * r[0] * r[1]);
        H_abcd[2] += cH * (r[0] * r[0] * r[0] * r[2] - 3.0 * rh * r[0] * r[2]);
        H_abcd[4] += cH * (r[0] * r[0] * r[1] * r[1] - rh * (r[0] * r[0] + r[1] * r[1]) + rrh);
        H_abcd[5] += cH * (r[0] * r[0] * r[1] * r[2] - rh * r[1] * r[2]);
        H_abcd[8] += cH * (r[0] * r[0] * r[2] * r[2] - rh * (r[0] * r[0] + r[2] * r[2]) + rrh);
        H_abcd[13] += cH * (r[0] * r[1] * r[1] * r[1] - 3.0 * rh * r[0] * r[1]);
        H_abcd[14] += cH * (r[0] * r[1] * r[1] * r[2] - rh * r[0] * r[2]);
        H_abcd[17] += cH * (r[0] * r[1] * r[2] * r[2] - rh * r[0] * r[1]);
        H_abcd[26] += cH * (r[0] * r[2] * r[2] * r[2] - 3.0 * rh * r[0] * r[2]);
        H_abcd[40] += cH * (r[1] * r[1] * r[1] * r[1] - 6.0 * rh * r[1] * r[1] + 3.0 * rrh);
        H_abcd[41] += cH * (r[1] * r[1] * r[1] * r[2] - 3.0 * rh * r[1] * r[2]);
        H_abcd[44] += cH * (r[1] * r[1] * r[2] * r[2] - rh * (r[1] * r[1] + r[2] * r[2]) + rrh);
        H_abcd[53] += cH * (r[1] * r[2] * r[2] * r[2] - 3.0 * rh * r[1] * r[2]);
        H_abcd[80] += cH * (r[2] * r[2] * r[2] * r[2] - 6.0 * rh * r[2] * r[2] + 3.0 * rrh);

      }
    }
  }
  Py_RETURN_NONE;
}