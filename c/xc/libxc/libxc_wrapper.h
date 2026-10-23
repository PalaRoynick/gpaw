#pragma once

#include "python_utils.h"
#include <xc.h>

typedef struct
{
  PyObject_HEAD
  /* exchange-correlation energy second derivatives */
  void (*get_fxc)(XC(func_type) *func, double point[7], double der[5][5]);
  XC(func_type) xc_functional;
  XC(func_type) x_functional;
  XC(func_type) c_functional;
  XC(func_type) *functional[2]; /* store either x&c, or just xc */
  int nspin; /* must be common to x and c, so declared redundantly here */
  // True if the functional is allocated on GPU memory
  bool on_device;
} lxcXCFunctionalObject;

// For the raw CPython libxc-wrapper object
PyObject* NewlxcXCFunctionalObject(PyObject *self, PyObject *args);
PyObject* lxcXCFuncNum(PyObject *self, PyObject *args);

// True if the libxc installation supports GPUs
bool libxc_has_gpu_support();

/** Binds newer libxc-specific functions. This is separate from the raw CPython-style lxcXCFunctionalObject bindings in
 * libxc.cpp in order to support GPU array input via our PyDeviceArray wrapper.
 */
void bind_libxc_extras(PyObject* module);
