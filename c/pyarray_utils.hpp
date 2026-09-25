#pragma once

// TODO: combine this with some common python header when moving to full C++.
// Would help with include order requirements from Python.h

#include "python_utils.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "utils.hpp"
#include "gpaw_utils.h"

#include <cstdint>
#include <cassert>
#include <type_traits>


namespace gpaw
{

inline bool is_complex_dtype(pybind11::dtype dtype) { return dtype.kind() == 'c'; }
/* Returns true if the input array has c_contiguous flag set.
Note that contiguous 1D arrays are both C- and F-contiguous. */
inline bool is_c_contiguous(pybind11::array arr) { return arr.flags() & pybind11::array::c_style; }


// Old CPython-style stuff below
// TODO there are also legacy #defines with the same names in array.h. Remove those!!

// Get pointer to the array data
template<typename T>
T* Array_DATA(PyObject* obj)
{
    static_assert(!std::is_pointer<T>::value, "T must be a scalar type");

    // Equivalent to obj.data.ptr
    PyObject* ndarray_data = PyObject_GetAttrString(obj, "data");
    if (ndarray_data == nullptr)
    {
        return nullptr;
    }

    PyObject* ptr_data = PyObject_GetAttrString(ndarray_data, "ptr");
    Py_DECREF(ndarray_data);
    if (ptr_data == nullptr)
    {
        return nullptr;
    }

    T* ptr = reinterpret_cast<T*>(PyLong_AsVoidPtr(ptr_data));
    Py_DECREF(ptr_data);
    return ptr;
}


bool Array_32BIT(PyObject* obj);

// Number of dimensions
int32_t Array_NDIM(PyObject* a);

// Size of d-th dimension
int64_t Array_DIM(PyObject* a, int32_t d);

// Size of array element in bytes
int64_t Array_ITEMSIZE(PyObject* a);

// Total number of elements in the array. See Array_NBYTES for total size in bytes
int64_t Array_SIZE(PyObject* a);

// Total number of bytes consumed by the array
int64_t Array_NBYTES(PyObject* a);

// Get built-in (Numpy) typenumber of elements of the array.
int Array_TYPE(PyObject* a);

// True if the array type is any complex floating point number
bool Array_ISCOMPLEX(PyObject* a);

//~
// Begin overloads for PyArrayObject, ie. Numpy array

template<typename T>
T* Array_DATA(PyArrayObject* a)
{
    return reinterpret_cast<T*>(PyArray_DATA(a));
}

int32_t Array_NDIM(PyArrayObject* a);
int64_t Array_DIM(PyArrayObject* a, int32_t d);
int64_t Array_ITEMSIZE(PyArrayObject* a);
int64_t Array_SIZE(PyArrayObject* a);
int64_t Array_NBYTES(PyArrayObject* a);
int Array_TYPE(PyArrayObject* a);
bool Array_ISCOMPLEX(PyArrayObject* a);
//~ End Numpy

} // namespace gpaw
