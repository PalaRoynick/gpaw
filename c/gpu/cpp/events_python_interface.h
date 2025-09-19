#pragma once

// Keep this C99 compliant because this gets included from _gpaw.h

#include "../../gpaw_utils.h"
#include <Python.h>

CLINKAGE PyObject* destroy_expired_events(PyObject* self, PyObject* args);


