#pragma once

/* !!!! Python requires that Python.h is included BEFORE any system headers,
* which is absolutely horrible but we kinda have to comply here...
* Therefore: this header should ALWAYS be the first header to include in files
* that use Python stuff.
* See https://docs.python.org/3/c-api/intro.html#include-files.
*/

/* Includes Python.h and Numpy headers with correct defines.
* Other files that need Python stuff should include this file instead of manually including Python or Numpy.
* Numpy array imports are done in _gpaw_so.c. */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Array import should be done only from the source file that inits the GPAW module
#ifndef __GPAW_SHOULD_IMPORT_NUMPY
    #define NO_IMPORT_ARRAY
#endif

#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#include <numpy/arrayobject.h>
