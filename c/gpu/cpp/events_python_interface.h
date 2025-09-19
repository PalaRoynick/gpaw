#pragma once

// Keep this C99 compliant because this gets included from _gpaw.h

#include "../../gpaw_utils.h"
#include <Python.h>

CLINKAGE PyObject* destroy_expired_events(PyObject* self, PyObject* args);

CLINKAGE PyObject* is_debug_system_enabled(PyObject* self, PyObject* args);
// Enable of disable the event debug system
CLINKAGE PyObject* toggle_debug_system(PyObject* self, PyObject* args);
// Set delay in microseconds that will be applied after every kernel launch that the debug system is tracking
CLINKAGE PyObject* configure_debug_stream_delay_microseconds(PyObject* self, PyObject* args);
//
CLINKAGE PyObject* ensure_buffer_not_in_use(PyObject* self, PyObject* args);
//
CLINKAGE PyObject* ensure_object_not_in_use(PyObject* self, PyObject* args);
