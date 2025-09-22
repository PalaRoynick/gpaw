#pragma once

// Keep this C99 compliant because this gets included from _gpaw.h

#include "../../gpaw_utils.h"
#include <Python.h>

// Exception type object
extern PyObject* DebugSystemError;

CLINKAGE PyObject* destroy_expired_events(PyObject* self, PyObject* args);

CLINKAGE PyObject* is_debug_system_enabled(PyObject* self, PyObject* args);
// Enable of disable the event debug system
CLINKAGE PyObject* toggle_debug_system(PyObject* self, PyObject* args);
// Set delay in microseconds that will be applied after every kernel launch that the debug system is tracking
CLINKAGE PyObject* configure_debug_stream_delay_microseconds(PyObject* self, PyObject* args);
CLINKAGE PyObject* configure_events_take_pyobject_ownership(PyObject* self, PyObject* args);
//
CLINKAGE PyObject* ensure_buffer_not_in_use(PyObject* self, PyObject* args);
//
CLINKAGE PyObject* ensure_object_not_in_use(PyObject* self, PyObject* args);
// Returns human-readable list of events that currently use the given memory block
CLINKAGE PyObject* get_overlapping_events_as_string(PyObject* self, PyObject* args);

/* Checks through the debug system if memory occupied by the input array is currently in use by any async event.
Return false on failure. Note that a non-array input is valid and will always return true.
If fatal == true, will raise a Python exception containing metadata about the error;
otherwise will just print the exception and clear the error.*/
CLINKAGE bool ensure_no_async_overlap(PyObject* arr, bool fatal);
