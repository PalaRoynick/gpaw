#pragma once

#include "../gpu-runtime.h"
#include "pyarray_utils.hpp"

#include <vector>

/* Implements a "life support" system for Python objects, so that they remain
alive for as long as GPAW's homebrew GPU kernels are using them. Needed because
we often get pointers to existing GPU arrays as inputs from Python side and
use those pointers in kernels. But kernels are asynchronous wrt. the CPU host,
so without proper reference count management Python could destroy the arrays
while kernels are still using them. */

namespace gpaw
{

// TODO remove this and move the borrow_array_data() functionality elsewhere

/* Class PyObjectPinner - used to keep Python objects alive for as long as GPU
kernels are using them. Intended for Cupy arrays specifically.
"Pinning" means that all PyObjects registered with this pinner instance get
their reference count increased. The pinning must be undone at a later time
to avoid memory leaks. */
class PyObjectPinner
{
public:
    PyObjectPinner();
    PyObjectPinner(size_t reserve_count);

    /* Request to 'borrow' array data from a PyObject. The object is assumed to be
    an array, so that gpaw::Array_DATA() returns a valid pointer.
    This function returns that pointer while also registering the PyObject
    for life support. */
    template<typename T>
    T* borrow_array_data(PyObject* obj)
    {
        T* data = Array_DATA<T>(obj);
    #ifdef GPAW_GPU_ARRAY_DEBUG
        assert(data && "Not a Python array object");
    #endif
        if (data)
        {
            objects.push_back(obj);
        }
        return data;
    }

    std::vector<PyObject*> objects;

};

} // namespace gpaw
